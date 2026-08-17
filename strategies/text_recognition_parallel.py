import logging
from concurrent.futures import as_completed

import torch

from .base import TextRecognitionStrategy
from parallel_utils import get_shared_executor, split_into_chunks

logger = logging.getLogger(__name__)


class ParallelTextRecognizer(TextRecognitionStrategy):

    def __init__(self, recognizers: list):
        if not recognizers:
            raise ValueError("ParallelTextRecognizer requires at least one recognizer instance.")

        self.recognizers = recognizers
        self.num_replicas = len(recognizers)

        first = recognizers[0]
        per_replica_batch = getattr(first, "batch_size", None)
        self.model_info = (
            f"Parallel recognition x{self.num_replicas} "
            f"[{first.__class__.__name__}"
            + (f", batch={per_replica_batch}/replica" if per_replica_batch else "")
            + "]"
        )
        self.is_word_level = any(
            getattr(r, "is_word_level", False) or "FastRecognizer" in r.__class__.__name__
            for r in recognizers
        )

        self.last_infer_time = 0.0
        self._executor = get_shared_executor()

        devices = [str(getattr(r, "device", "unknown")) for r in recognizers]
        distinct_devices = set(devices)
        if len(distinct_devices) < self.num_replicas and "cuda" in "".join(distinct_devices):
            logger.warning(
                f"ParallelTextRecognizer has {self.num_replicas} replicas but only "
                f"{len(distinct_devices)} distinct device(s) among them ({sorted(distinct_devices)}). "
                f"Replicas sharing a GPU will NOT run with true parallelism -- consider "
                f"--recognizer-replicas 1 with a larger --recognizer-batch-size instead."
            )

        logger.info(
            f"Parallel text recognizer ready: {self.num_replicas} replica(s) of "
            f"{first.__class__.__name__} on devices {devices} (device-parallel dispatch)."
        )

    @staticmethod
    def _run_on_device(recognizer, crops):
        if not crops:
            return []
        device = getattr(recognizer, "device", None)
        if device is not None and getattr(device, "type", None) == "cuda":
            with torch.cuda.device(device):
                return recognizer.recognize_text(crops)
        return recognizer.recognize_text(crops)

    def recognize_text(self, image_crops: list) -> list:
        if not image_crops:
            return []

        chunks = split_into_chunks(image_crops, self.num_replicas)

        futures = {}
        for recognizer, chunk in zip(self.recognizers, chunks):
            if not chunk:
                continue
            fut = self._executor.submit(self._run_on_device, recognizer, chunk)
            futures[fut] = recognizer

        recognized_output = []
        infer_time = 0.0

        for fut in as_completed(futures):
            recognizer = futures[fut]
            try:
                recognized_output.extend(fut.result())
            except Exception as e:
                logger.warning(f"Recognition replica {recognizer.__class__.__name__} failed: {e}")
            infer_time = max(infer_time, getattr(recognizer, "last_infer_time", 0.0))

        self.last_infer_time = infer_time
        recognized_output.sort(key=lambda r: r["word_id"])
        return recognized_output