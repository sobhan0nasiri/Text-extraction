import re
import logging

import cv2
import numpy as np
import requests
import torch
from concurrent.futures import as_completed

from .base import TextRecognitionStrategy
from parallel_utils import get_shared_executor, split_into_chunks

logger = logging.getLogger(__name__)


class PPOCRv5_Recognizer(TextRecognitionStrategy):

    _CLEAN_PATTERN = re.compile(r'[^A-Za-z0-9\s\.\,\:\;\!\?\-\>\<\_\(\)\[\]\'\"\/\\@#\$%&\*\+=\~\^\u2022\u2605\u25CF\u25AA\u2023\u2666]')
    _MIN_CROP_DIM = 32

    def __init__(self, server_url=None, server_urls=None,
                    batch_size: int = 32, timeout: float = 15.0, max_workers: int = None):
        raw_urls = server_urls if server_urls else server_url
        if raw_urls is None:
            raise ValueError("PPOCRv5_Recognizer requires server_url or server_urls.")
        if isinstance(raw_urls, str):
            raw_urls = [u.strip() for u in raw_urls.split(",") if u.strip()]

        self.batch_size = batch_size
        self.timeout = timeout
        self.last_infer_time = 0.0
        self.model_info = "PP-OCRv5 (remote microservice)"
        self.base_urls = self._check_servers([u.rstrip("/") for u in raw_urls])
        self.batch_urls = [f"{u}/recognize_batch" for u in self.base_urls]
        self.executor = get_shared_executor()
        logger.info(f"PP-OCRv5 recognizer connected via {len(self.base_urls)} microservice replica(s): {self.base_urls}")

    def _check_servers(self, urls):
        healthy, infos = [], []
        for url in urls:
            try:
                resp = requests.get(f"{url}/health", timeout=5)
                resp.raise_for_status()
                info = resp.json()
                healthy.append(url)
                infos.append(f"{info.get('model_name', 'PP-OCRv5')}@{url} (device={info.get('device', 'unknown')})")
            except Exception as e:
                logger.warning(f"PP-OCRv5 replica unreachable, skipped: {url} ({e})")

        if not healthy:
            raise RuntimeError(
                f"Could not reach any PP-OCRv5 microservice among {urls}. "
                f"Start each replica in its own environment, e.g. "
                f"'PPOCR_PORT=5005 python ppocr_server.py' (from inside ppocr_env)."
            )

        self.model_info = f"PP-OCRv5 remote cluster ({', '.join(infos)})"
        return healthy

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

    def _send_batch(self, batch_url, files):
        resp = requests.post(batch_url, files=files, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    @torch.inference_mode()
    def recognize_text(self, image_crops: list) -> list:
        valid_crops = [c for c in image_crops
                    if c["crop_tensor"].numel() != 0
                    and c["crop_tensor"].shape[-1] != 0
                    and c["crop_tensor"].shape[-2] != 0]

        if not valid_crops:
            return []

        valid_crops = sorted(valid_crops, key=lambda c: c["crop_tensor"].shape[-1] / max(1, c["crop_tensor"].shape[-2]))

        self.last_infer_time = 0.0
        n_replicas = len(self.batch_urls)
        logger.info(f"Recognizing {len(valid_crops)} words using PP-OCRv5 "
                    f"({n_replicas} port(s), requests of up to {self.batch_size} crops each)...")

        port_groups = split_into_chunks(valid_crops, n_replicas)

        futures = {}
        for port_idx, group in enumerate(port_groups):
            if not group:
                continue
            batch_url = self.batch_urls[port_idx]

            for start in range(0, len(group), self.batch_size):
                sub_batch = group[start:start + self.batch_size]
                encoded = [self._encode_crop(c["crop_tensor"]) for c in sub_batch]
                files = [
                    ("images", (f"crop_{i}.png", data, "image/png"))
                    for i, data in enumerate(encoded) if data is not None
                ]
                if not files:
                    continue
                sent_crops = [c for c, data in zip(sub_batch, encoded) if data is not None]
                fut = self.executor.submit(self._send_batch, batch_url, files)
                futures[fut] = sent_crops

        recognized_output = []
        for fut in as_completed(futures):
            sent_crops = futures[fut]
            try:
                payload = fut.result()
            except Exception as e:
                logger.warning(f"PP-OCRv5 batch request failed ({len(sent_crops)} crops skipped): {e}")
                continue

            results = payload.get("results", [])
            self.last_infer_time = max(self.last_infer_time, payload.get("model_time", 0.0))

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

        recognized_output.sort(key=lambda r: r["word_id"])
        return recognized_output