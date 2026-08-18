import re
import logging
from contextlib import nullcontext
import cv2

import numpy as np
import torch
from doctr.models import recognition_predictor

from .base import TextRecognitionStrategy
from parallel_utils import run_batches_pipelined

logger = logging.getLogger(__name__)


class DocTR_FastRecognizer(TextRecognitionStrategy):

    _CLEAN_PATTERN = re.compile(r'[^A-Za-z0-9\s\.\,\:\;\!\?\-\>\<\_\(\)\[\]\'\"\/\\@#\$%&\*\+=\~\^\u2022\u2605\u25CF\u25AA\u2023\u2666]')
    _MIN_CROP_DIM = 4
    
    def __init__(self, arch: str = "parseq", batch_size: int = 32, use_fp16: bool = True, device: str = None):
        logger.info(f"Loading real weights for docTR fast recognizer ({arch})...")

        self.model = recognition_predictor(arch=arch, pretrained=True, symmetric_pad=False)
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            self.model = self.model.to(self.device)
            if hasattr(self.model, "model"):
                self.model.model = torch.compile(self.model.model, dynamic=True)

        self.batch_size = batch_size
        self.use_fp16 = use_fp16 and self.device.type == "cuda"
        self.model_info = f"docTR fast recognizer ({arch})"
        self.last_infer_time = 0.0
        logger.info(f"docTR fast recognizer ({arch}) loaded successfully on {self.device} "
                    f"[fp16={self.use_fp16}].")

    def _autocast(self):
        if self.use_fp16:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    @torch.inference_mode()
    def recognize_text(self, image_crops: list) -> list:
        if not image_crops:
            return []

        self.last_infer_time = 0.0

        valid_crops = []
        for crop_data in image_crops:
            crop_tensor = crop_data["crop_tensor"]
            if crop_tensor.numel() == 0 or crop_tensor.shape[-1] == 0 or crop_tensor.shape[-2] == 0:
                continue
            valid_crops.append(crop_data)

        if not valid_crops:
            return []

        valid_crops.sort(key=lambda c: c["crop_tensor"].shape[-1] / max(1, c["crop_tensor"].shape[-2]))

        logger.info(f"Recognizing {len(valid_crops)} words using docTR fast recognizer "
                    f"(batch={self.batch_size if self.batch_size else 'auto'})...")

        def _prepare(batch_crops):
            batch_np = []
            for c in batch_crops:
                img_np = c["crop_tensor"].squeeze(0).permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).astype(np.uint8)

                h, w = img_np.shape[:2]
                if min(h, w) < self._MIN_CROP_DIM:
                    factor = self._MIN_CROP_DIM / max(1, min(h, w))
                    img_np = cv2.resize(img_np, (max(1, int(w * factor)), max(1, int(h * factor))), interpolation=cv2.INTER_LANCZOS4)

                batch_np.append(img_np)
            return batch_np

        def _compute(batch_crops, batch_np):
            with self._autocast():
                results = self.model(batch_np)

            batch_output = []
            for crop_data, result in zip(batch_crops, results):
                text_str, confidence = result
                text_str = self._CLEAN_PATTERN.sub('', text_str).strip()
                if not text_str:
                    continue
                logger.debug(f" -> Word {crop_data['word_id']} {crop_data['box']}: '{text_str}'")
                batch_output.append({
                    "word_id": crop_data["word_id"],
                    "box": crop_data["box"],
                    "text": text_str,
                    "conf": float(confidence)
                })
            return batch_output

        recognized_output, self.last_infer_time = run_batches_pipelined(
            valid_crops, _prepare, _compute,
            model_key=f"doctr_fast:{self.model_info}", device=self.device,
            batch_size=self.batch_size, start_batch=8, max_batch=256,
        )

        recognized_output.sort(key=lambda r: r["word_id"])
        return recognized_output