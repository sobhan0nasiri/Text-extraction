import re
import logging

import cv2
import numpy as np
import requests
import torch

from .base import TextRecognitionStrategy

logger = logging.getLogger(__name__)


class PPOCRv5_Recognizer(TextRecognitionStrategy):

    _CLEAN_PATTERN = re.compile(r'[^A-Za-z0-9\s\.\,\:\;\!\?\-\>\<\_\(\)\[\]\'\"\/\\@#\$%&\*\+=\~\^\u2022\u2605\u25CF\u25AA\u2023\u2666]')
    _MIN_CROP_DIM = 32

    def __init__(self, server_url: str = "http://127.0.0.1:5005",
                    batch_size: int = 32, timeout: float = 15.0):
        self.base_url = server_url.rstrip("/")
        self.batch_url = f"{self.base_url}/recognize_batch"
        self.batch_size = batch_size
        self.timeout = timeout
        self.last_infer_time = 0.0
        self.model_info = "PP-OCRv5 (remote microservice)"
        self._check_server()
        logger.info(f"PP-OCRv5 recognizer connected via microservice at {self.base_url}")

    def _check_server(self):
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            resp.raise_for_status()
            info = resp.json()
            model_name = info.get("model_name", "PP-OCRv5")
            device = info.get("device", "unknown")
            self.model_info = f"PP-OCRv5 remote microservice ({model_name}, device={device})"
        except Exception as e:
            raise RuntimeError(
                f"Could not reach the PP-OCRv5 microservice at {self.base_url}. "
                f"Start it first, in its own environment: "
                f"'python ppocr_service/ppocr_server.py' (from inside ppocr_env). "
                f"Details: {e}"
            )

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
        logger.info(f"Recognizing {len(valid_crops)} words using PP-OCRv5 "
                    f"(remote, batches of {self.batch_size})...")

        recognized_output = []
        for start in range(0, len(valid_crops), self.batch_size):
            batch = valid_crops[start:start + self.batch_size]
            encoded = [self._encode_crop(c["crop_tensor"]) for c in batch]

            files = [
                ("images", (f"crop_{i}.png", data, "image/png"))
                for i, data in enumerate(encoded) if data is not None
            ]
            if not files:
                continue

            try:
                resp = requests.post(self.batch_url, files=files, timeout=self.timeout)
                resp.raise_for_status()
                payload = resp.json()
                results = payload.get("results", [])
                self.last_infer_time += payload.get("model_time", 0.0)
            except Exception as e:
                logger.warning(f"PP-OCRv5 batch request failed ({len(batch)} crops skipped): {e}")
                continue

            sent_crops = [c for c, data in zip(batch, encoded) if data is not None]
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