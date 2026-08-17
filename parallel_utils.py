import os
import time
import atexit
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

logger = logging.getLogger(__name__)

_cpu_count = os.cpu_count() or 4


def default_thread_workers(n: int = None) -> int:
    return n if n else min(8, max(2, _cpu_count))


_shared_executor = None
_shared_executor_lock = threading.Lock()


def get_shared_executor() -> ThreadPoolExecutor:
    global _shared_executor
    if _shared_executor is None:
        with _shared_executor_lock:
            if _shared_executor is None:
                _shared_executor = ThreadPoolExecutor(max_workers=default_thread_workers())
                atexit.register(_shared_executor.shutdown, wait=False)
    return _shared_executor


def split_into_chunks(items: list, n_chunks: int) -> list:
    if n_chunks <= 0:
        raise ValueError("n_chunks must be a positive integer")

    total = len(items)
    base, remainder = divmod(total, n_chunks)

    chunks = []
    start = 0
    for i in range(n_chunks):
        size = base + (1 if i < remainder else 0)
        chunks.append(items[start:start + size])
        start += size

    return chunks


def get_replica_devices(num_replicas: int, explicit_devices: list = None) -> list:
    """
    Returns a list of length `num_replicas` of torch device strings, one per
    model replica.

    - If `explicit_devices` is given (e.g. from --recognizer-devices), it is
      used directly (cycled if shorter than num_replicas).
    - Else, if 2+ distinct CUDA devices are visible, replicas are spread across
      them round-robin. This is the ONLY situation where running several model
      replicas actually buys real parallel hardware.
    - Else (0 or 1 CUDA device visible), replication cannot add real
      parallelism: every replica would share the same SMs and VRAM bandwidth.
      A warning is logged and every replica is pointed at the same device so
      behavior stays correct — callers should prefer a larger batch_size on a
      single replica instead of >1 replicas in this case.
    """
    if num_replicas <= 0:
        raise ValueError("num_replicas must be a positive integer")

    if explicit_devices:
        if len(explicit_devices) < num_replicas:
            reps = (num_replicas // len(explicit_devices)) + 1
            explicit_devices = (explicit_devices * reps)
        return explicit_devices[:num_replicas]

    if not torch.cuda.is_available():
        if num_replicas > 1:
            logger.warning(
                f"{num_replicas} replicas requested but no CUDA device is visible; "
                f"running on CPU gives no parallelism benefit from replication."
            )
        return ["cpu"] * num_replicas

    n_gpus = torch.cuda.device_count()
    if n_gpus >= 2:
        devices = [f"cuda:{i % n_gpus}" for i in range(num_replicas)]
        logger.info(f"Spreading {num_replicas} replica(s) across {n_gpus} physical GPUs: {devices}")
        return devices

    if num_replicas > 1:
        logger.warning(
            f"--recognizer-replicas={num_replicas} requested but only 1 CUDA device is visible. "
            f"Replicas will NOT run with true hardware parallelism — they'll load duplicate model "
            f"weights and contend for the same GPU cores/VRAM bandwidth, which typically helps little "
            f"or not at all. For a single GPU, prefer keeping --recognizer-replicas 1 and raising "
            f"--recognizer-batch-size instead."
        )
    return ["cuda:0"] * num_replicas


_frame_cache_lock = threading.Lock()
_frame_cache_key = None
_frame_cache_value = None


def get_frame_rgb_uint8(image: torch.Tensor) -> np.ndarray:
    global _frame_cache_key, _frame_cache_value
    key = (id(image), image.data_ptr(), tuple(image.shape))
    with _frame_cache_lock:
        if _frame_cache_key == key:
            return _frame_cache_value
        img_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        _frame_cache_key = key
        _frame_cache_value = img_np
        return img_np


class AdaptiveBatchSizer:
    """
    Learns a good per-forward-pass batch size for a given (model_key, device)
    at runtime, without needing a separate offline calibration pass.

    Strategy: start conservative, probe upward while larger batches keep
    improving items/sec throughput, and back off hard (halve + remember a
    ceiling) the instant an OOM (local CUDA OOM, or a failed/timed-out remote
    request) is reported. State is shared process-wide across every caller
    with the same model_key+device (e.g. multiple recognizer replicas, or
    repeated calls across many images in folder mode), so the learned value
    converges quickly and keeps improving over the run instead of resetting
    per-call.

    This is intentionally simple (no persistence across process restarts) --
    it optimizes for "learns fast within one run", which is what actually
    matters for folder/camera mode. For a single `--mode file` image it will
    typically only get 1-3 probes, which is still enough to avoid an OOM and
    to prefer a larger batch than an overly-conservative fixed default.
    """

    _registry = {}
    _registry_lock = threading.Lock()

    MIN_BATCH = 1
    _GROWTH_FACTOR = 2
    _IMPROVEMENT_THRESHOLD = 1.03  # require >3% throughput gain to keep growing

    def __init__(self, model_key: str, device: str, start_batch: int = 8, max_batch: int = 256):
        self.model_key = model_key
        self.device = str(device)
        self._lock = threading.Lock()
        self.current = max(self.MIN_BATCH, start_batch)
        self.ceiling = max(self.MIN_BATCH, max_batch)
        self.best_throughput = 0.0
        self.best_batch = self.current
        self._probing_up = True

    @classmethod
    def get(cls, model_key: str, device, start_batch: int = 8, max_batch: int = 256) -> "AdaptiveBatchSizer":
        key = (model_key, str(device))
        with cls._registry_lock:
            inst = cls._registry.get(key)
            if inst is None:
                inst = cls(model_key, device, start_batch=start_batch, max_batch=max_batch)
                cls._registry[key] = inst
            return inst

    def suggest(self, n_remaining: int) -> int:
        with self._lock:
            batch = min(self.current, self.ceiling)
        return max(self.MIN_BATCH, min(batch, max(1, n_remaining)))

    def record_success(self, batch_used: int, elapsed: float, n_items: int) -> None:
        if elapsed <= 0 or n_items <= 0:
            return
        throughput = n_items / elapsed
        with self._lock:
            if throughput > self.best_throughput * self._IMPROVEMENT_THRESHOLD:
                self.best_throughput = throughput
                self.best_batch = batch_used
                if self._probing_up and batch_used >= self.current:
                    self.current = min(self.ceiling, max(self.current * self._GROWTH_FACTOR, batch_used + 1))
            elif self._probing_up:
                # Growing further isn't paying off -- settle on the best batch seen so far.
                self._probing_up = False
                self.current = self.best_batch

    def record_oom(self, batch_used: int) -> None:
        with self._lock:
            self.ceiling = max(self.MIN_BATCH, batch_used // 2)
            self.current = self.ceiling
            self._probing_up = False
        logger.warning(
            f"[AdaptiveBatchSizer] backing off for {self.model_key}@{self.device}: "
            f"batch={batch_used} failed (OOM or remote error); capping future batches to {self.ceiling}."
        )


def _is_oom_error(exc: Exception) -> bool:
    if isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", ())):
        return True
    return "out of memory" in str(exc).lower()


def run_batches_adaptive(items: list, forward_fn, model_key: str, device,
                            batch_size: int = None, min_batch: int = 1,
                            start_batch: int = 8, max_batch: int = 256):
    """
    Runs `forward_fn(sub_items) -> list_of_results` over `items` in chunks,
    handling both the "auto" and "fixed" batch-size cases with a single code
    path:

    - `batch_size=None`  -> adaptive mode: chunk size is chosen per-iteration
      by a shared `AdaptiveBatchSizer` for (model_key, device), growing while
      it helps throughput and shrinking on failure.
    - `batch_size=<int>` -> fixed mode: chunk size is capped at this value,
      but OOM backoff is still active as a safety net (halves and retries
      instead of crashing the whole recognition call).

    forward_fn's output list does not need to be the same length as its
    input (e.g. it may drop empty-text results) -- only elapsed time and the
    input length are used for throughput bookkeeping.

    Returns (all_results, total_elapsed_seconds).
    """
    if not items:
        return [], 0.0

    sizer = None if batch_size else AdaptiveBatchSizer.get(model_key, device, start_batch=start_batch, max_batch=max_batch)

    results = []
    total_elapsed = 0.0
    i, n = 0, len(items)

    while i < n:
        remaining = n - i
        chunk = batch_size if batch_size else sizer.suggest(remaining)
        chunk = max(min_batch, min(chunk, remaining))
        sub = items[i:i + chunk]

        try:
            t0 = time.perf_counter()
            out = forward_fn(sub)
            elapsed = time.perf_counter() - t0
        except RuntimeError as e:
            if not _is_oom_error(e):
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if sizer is not None:
                sizer.record_oom(chunk)
            elif chunk > min_batch:
                # Fixed batch_size still gets a one-time emergency halving so a
                # single oversized --recognizer-batch-size doesn't hard-crash a run.
                batch_size = max(min_batch, chunk // 2)
                logger.warning(f"OOM at fixed batch_size={chunk} for {model_key}@{device}; retrying at {batch_size}.")
            else:
                raise
            continue  # retry the same items with the smaller batch

        results.extend(out)
        total_elapsed += elapsed
        if sizer is not None:
            sizer.record_success(chunk, elapsed, len(sub))
        i += len(sub)

    return results, total_elapsed


def run_batches_pipelined(items: list, prepare_fn, compute_fn, model_key: str, device,
                            batch_size: int = None, min_batch: int = 1,
                            start_batch: int = 8, max_batch: int = 256):
    if not items:
        return [], 0.0

    sizer = None if batch_size else AdaptiveBatchSizer.get(model_key, device, start_batch=start_batch, max_batch=max_batch)
    executor = get_shared_executor()
    n = len(items)

    def _submit_prepare(start):
        remaining = n - start
        size = batch_size if batch_size else sizer.suggest(remaining)
        size = max(min_batch, min(size, remaining))
        end = start + size
        chunk_items = items[start:end]
        return executor.submit(prepare_fn, chunk_items), (start, end)

    results = []
    total_elapsed = 0.0

    prefetch_future, prefetch_range = _submit_prepare(0)

    while prefetch_future is not None:
        start, end = prefetch_range
        chunk_items = items[start:end]
        prepared = prefetch_future.result()

        next_start = end
        if next_start < n:
            prefetch_future, prefetch_range = _submit_prepare(next_start)
        else:
            prefetch_future, prefetch_range = None, None

        try:
            t0 = time.perf_counter()
            out = compute_fn(chunk_items, prepared)
            elapsed = time.perf_counter() - t0
        except RuntimeError as e:
            if not _is_oom_error(e):
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            chunk_len = end - start
            if sizer is not None:
                sizer.record_oom(chunk_len)
            elif chunk_len > min_batch:
                batch_size = max(min_batch, chunk_len // 2)
                logger.warning(f"OOM at fixed batch_size={chunk_len} for {model_key}@{device}; retrying at {batch_size}.")
            else:
                raise
            prefetch_future, prefetch_range = _submit_prepare(start)
            continue

        results.extend(out)
        total_elapsed += elapsed
        if sizer is not None:
            sizer.record_success(end - start, elapsed, len(chunk_items))

    return results, total_elapsed


_stream_pool = {}
_stream_pool_lock = threading.Lock()


def get_cuda_streams(n: int):
    if n <= 0:
        return []
    if not torch.cuda.is_available():
        return [None] * n
    with _stream_pool_lock:
        streams = _stream_pool.get(n)
        if streams is None:
            streams = [torch.cuda.Stream() for _ in range(n)]
            _stream_pool[n] = streams
        return streams