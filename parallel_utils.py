import os
import atexit
import threading
from concurrent.futures import ThreadPoolExecutor

import torch

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