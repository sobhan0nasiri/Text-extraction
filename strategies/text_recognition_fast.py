import re
import time
import logging
from contextlib import nullcontext
import cv2

import numpy as np
import torch
from doctr.models import recognition_predictor

from .base import TextRecognitionStrategy

logger = logging.getLogger(__name__)


class DocTR_FastRecognizer(TextRecognitionStrategy):

    _CLEAN_PATTERN = re.compile(r'[^A-Za-z0-9\s\.\,\:\;\!\?\-\>\<\_\(\)\[\]\'\"\/\\@#\$%&\*\+=\~\^\u2022\u2605\u25CF\u25AA\u2023\u2666]')
    _MIN_CROP_DIM = 4
    
    def __init__(self, arch: str = "parseq", batch_size: int = 32, use_fp16: bool = True):
        logger.info(f"Loading real weights for docTR fast recognizer ({arch})...")

        self.model = recognition_predictor(arch=arch, pretrained=True, symmetric_pad=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            self.model = self.model.cuda()

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

        valid_np_crops, crop_mapping = [], []
        for crop_data in image_crops:
            crop_tensor = crop_data["crop_tensor"]
            if crop_tensor.numel() == 0 or crop_tensor.shape[-1] == 0 or crop_tensor.shape[-2] == 0:
                continue
            img_np = crop_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            
            h, w = img_np.shape[:2]
            if min(h, w) < self._MIN_CROP_DIM:
                factor = self._MIN_CROP_DIM / max(1, min(h, w))
                img_np = cv2.resize(img_np, (max(1, int(w * factor)), max(1, int(h * factor))), interpolation=cv2.INTER_LANCZOS4)
            
            valid_np_crops.append(img_np)
            crop_mapping.append(crop_data)

        if not valid_np_crops:
            return []

        order = sorted(range(len(valid_np_crops)), key=lambda i: valid_np_crops[i].shape[1] / max(1, valid_np_crops[i].shape[0]))
        valid_np_crops = [valid_np_crops[i] for i in order]
        crop_mapping = [crop_mapping[i] for i in order]

        logger.info(f"Recognizing {len(valid_np_crops)} words using docTR fast recognizer...")

        recognized_output = []
        for start in range(0, len(valid_np_crops), self.batch_size):
            end = min(start + self.batch_size, len(valid_np_crops))
            batch_np = valid_np_crops[start:end]
            batch_mapping = crop_mapping[start:end]

            t0 = time.perf_counter()
            with self._autocast():
                results = self.model(batch_np)
            self.last_infer_time += time.perf_counter() - t0

            for crop_data, result in zip(batch_mapping, results):
                text_str, confidence = result
                text_str = self._CLEAN_PATTERN.sub('', text_str).strip()
                if not text_str:
                    continue
                logger.debug(f" -> Word {crop_data['word_id']} {crop_data['box']}: '{text_str}'")
                recognized_output.append({
                    "word_id": crop_data["word_id"],
                    "box": crop_data["box"],
                    "text": text_str,
                    "conf": float(confidence)
                })

        recognized_output.sort(key=lambda r: r["word_id"])
        return recognized_output