import re
import time
import logging

import cv2
import numpy as np
import requests
import torch
from concurrent.futures import as_completed

from .base import TextRecognitionStrategy
from parallel_utils import get_shared_executor, split_into_chunks, AdaptiveBatchSizer

logger = logging.getLogger(__name__)


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


class PPOCRv5_Recognizer(TextRecognitionStrategy):

    _CLEAN_PATTERN = re.compile(r'[^A-Za-z0-9\s\.\,\:\;\!\?\-\>\<\_\(\)\[\]\'\"\/\\@#\$%&\*\+=\~\^\u2022\u2605\u25CF\u25AA\u2023\u2666]')
    _MIN_CROP_DIM = 32

    _MAX_CROPS_PER_REQUEST = 512

    def __init__(self, server_url=None, server_urls=None,
                    batch_size: int = 32, timeout: float = 30.0, max_workers: int = None):

        raw_urls = server_urls if server_urls else server_url
        if raw_urls is None:
            raise ValueError("PPOCRv5_Recognizer requires server_url or server_urls.")
        if isinstance(raw_urls, str):
            raw_urls = [u.strip() for u in raw_urls.split(",") if u.strip()]

        self.batch_size = batch_size
        self.timeout = timeout
        self.last_infer_time = 0.0
        self.last_server_compute_time = 0.0

        self.model_info = "PP-OCRv5 (remote microservice)"
        self.base_urls, self.replica_counts = self._check_servers([u.rstrip("/") for u in raw_urls])
        self.batch_urls = [f"{u}/recognize_batch" for u in self.base_urls]
        self.executor = get_shared_executor()
        self.session = requests.Session()  # reuse TCP connections to the local microservice(s)

        total_replicas = sum(self.replica_counts)
        logger.info(f"PP-OCRv5 recognizer connected via {len(self.base_urls)} microservice replica(s) "
                    f"({total_replicas} total GPU worker(s)): {self.base_urls}")

        if total_replicas == 1 and self.batch_size and self.batch_size < 16:
            logger.warning(
                f"Only 1 GPU worker is visible across all PP-OCRv5 servers, but "
                f"--recognizer-batch-size={self.batch_size} is small. On a single GPU, real "
                f"speedup comes from a LARGER batch inside one model.predict() call, not from "
                f"more concurrent requests -- those just queue up behind the same worker. "
                f"Consider raising --recognizer-batch-size (e.g. 64-128) or using --auto-batch-size."
            )

    def _check_servers(self, urls):
        healthy, infos, replica_counts = [], [], []
        for url in urls:
            try:
                resp = requests.get(f"{url}/health", timeout=5)
                resp.raise_for_status()
                info = resp.json()
                healthy.append(url)
                replica_counts.append(max(1, int(info.get("num_replicas", 1))))
                infos.append(f"{info.get('model_name', 'PP-OCRv5')}@{url} "
                            f"(device={info.get('device', 'unknown')}, replicas={info.get('num_replicas', 1)})")
            except Exception as e:
                logger.warning(f"PP-OCRv5 replica unreachable, skipped: {url} ({e})")

        if not healthy:
            raise RuntimeError(
                f"Could not reach any PP-OCRv5 microservice among {urls}. "
                f"Start each replica in its own environment, e.g. "
                f"'PPOCR_PORT=5005 python ppocr_server.py' (from inside ppocr_env)."
            )

        self.model_info = f"PP-OCRv5 remote cluster ({', '.join(infos)})"
        return healthy, replica_counts

    @staticmethod
    def _encode_crop(crop_tensor: torch.Tensor):
        img_np = (crop_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        h, w = img_np.shape[:2]
        if min(h, w) < PPOCRv5_Recognizer._MIN_CROP_DIM:
            factor = PPOCRv5_Recognizer._MIN_CROP_DIM / max(1, min(h, w))
            img_np = cv2.resize(img_np, (max(1, int(w * factor)), max(1, int(h * factor))), interpolation=cv2.INTER_LANCZOS4)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".png", img_bgr)
        return buf.tobytes() if ok else None

    def _post_once(self, batch_url, crops, server_batch_hint):
        encoded = [self._encode_crop(c["crop_tensor"]) for c in crops]
        files = [
            ("images", (f"crop_{i}.png", data, "image/png"))
            for i, data in enumerate(encoded) if data is not None
        ]
        sent_crops = [c for c, data in zip(crops, encoded) if data is not None]
        if not files:
            return sent_crops, {"results": [], "model_time": 0.0}, 0.0

        t0 = time.perf_counter()
        resp = self.session.post(batch_url, files=files, data={"batch_size": server_batch_hint}, timeout=self.timeout)
        resp.raise_for_status()
        elapsed = time.perf_counter() - t0
        return sent_crops, resp.json(), elapsed

    def _send_with_retry(self, batch_url, crops, server_batch_hint, sizer):
        try:
            return self._post_once(batch_url, crops, server_batch_hint)
        except Exception as e:
            if sizer is not None:
                sizer.record_oom(server_batch_hint)
            retry_hint = max(1, server_batch_hint // 2)
            logger.warning(f"PP-OCRv5 request to {batch_url} failed ({len(crops)} crops, "
                            f"batch_hint={server_batch_hint}): {e}. Retrying once at batch_hint={retry_hint}.")
            try:
                return self._post_once(batch_url, crops, retry_hint)
            except Exception as e2:
                logger.warning(f"PP-OCRv5 retry to {batch_url} also failed ({len(crops)} crops skipped): {e2}")
                return [], {"results": [], "model_time": 0.0}, 0.0

    @torch.inference_mode()
    def recognize_text(self, image_crops: list) -> list:
        valid_crops = [c for c in image_crops
                    if c["crop_tensor"].numel() != 0
                    and c["crop_tensor"].shape[-1] != 0
                    and c["crop_tensor"].shape[-2] != 0]

        if not valid_crops:
            self.last_infer_time = 0.0
            self.last_server_compute_time = 0.0
            return []

        t_wall0 = time.perf_counter()

        valid_crops = sorted(valid_crops, key=lambda c: c["crop_tensor"].shape[-1] / max(1, c["crop_tensor"].shape[-2]))

        n_ports = len(self.batch_urls)
        port_groups = split_into_chunks(valid_crops, n_ports)

        plan = []
        for port_idx, group in enumerate(port_groups):
            if not group:
                continue
            replicas = self.replica_counts[port_idx]
            n_requests = max(1, min(replicas, len(group)))
            n_requests = max(n_requests, _ceil_div(len(group), self._MAX_CROPS_PER_REQUEST))
            sub_groups = split_into_chunks(group, n_requests) if n_requests > 1 else [group]

            base_url = self.base_urls[port_idx]
            sizer = None if self.batch_size else AdaptiveBatchSizer.get(
                f"ppocrv5:{base_url}", "remote", start_batch=64, max_batch=256
            )
            for sub in sub_groups:
                if sub:
                    plan.append((self.batch_urls[port_idx], base_url, sub, sizer))

        logger.info(f"Recognizing {len(valid_crops)} words using PP-OCRv5 "
                    f"({n_ports} port(s), {len(plan)} HTTP request(s), "
                    f"server batch hint={self.batch_size if self.batch_size else 'auto'})...")

        futures = {}
        for batch_url, base_url, crops, sizer in plan:
            server_batch_hint = self.batch_size if self.batch_size else sizer.suggest(len(crops))
            fut = self.executor.submit(self._send_with_retry, batch_url, crops, server_batch_hint, sizer)
            futures[fut] = (base_url, crops, server_batch_hint, sizer)

        recognized_output = []
        server_compute_max = 0.0

        for fut in as_completed(futures):
            base_url, crops, server_batch_hint, sizer = futures[fut]
            sent_crops, payload, elapsed = fut.result()

            if sizer is not None and sent_crops and elapsed > 0:
                sizer.record_success(server_batch_hint, elapsed, len(sent_crops))

            results = payload.get("results", [])
            server_compute_max = max(server_compute_max, payload.get("model_time", 0.0))

            for crop_data, res in zip(sent_crops, results):
                text_str = self._CLEAN_PATTERN.sub('', res.get("text", "")).strip()
                if not text_str:
                    continue
                logger.debug(f" -> Word {crop_data['word_id']} {crop_data['box']}: '{text_str}'")
                recognized_output.append({
                    "word_id": crop_data["word_id"],
                    "box": crop_data["box"],
                    "text": text_str,
                    "conf": float(res.get("conf", 0.0))
                })

        self.last_infer_time = time.perf_counter() - t_wall0
        self.last_server_compute_time = server_compute_max

        recognized_output.sort(key=lambda r: r["word_id"])
        return recognized_output