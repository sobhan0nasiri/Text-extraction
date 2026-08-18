import re
import logging
from contextlib import nullcontext

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageOps
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, LogitsProcessor, LogitsProcessorList

from .base import TextRecognitionStrategy
from parallel_utils import run_batches_pipelined

logger = logging.getLogger(__name__)


def _remove_unused_pooler(model: torch.nn.Module) -> None:
    encoder = getattr(model, "encoder", None)
    if encoder is not None and getattr(encoder, "pooler", None) is not None:
        logger.info("Removing unused encoder.pooler submodule (not used by TrOCR forward pass).")
        encoder.pooler = None


class AllowedCharsLogitsProcessor(LogitsProcessor):

    ALLOWED_PATTERN = re.compile(r'^[A-Za-z0-9\s\.\,\:\;\!\?\-\<\>\_\(\)\[\]\'\"\/\\@#\$%&\*\+=\~\^]*$')

    def __init__(self, tokenizer, special_token_ids: set):
        super().__init__()
        self.tokenizer = tokenizer
        self.special_token_ids = special_token_ids
        self._mask = None

    def _build_mask(self, vocab_size, device, dtype):
        mask = torch.full((vocab_size,), float("-inf"), dtype=dtype)
        count = 0
        for token_id in range(vocab_size):
            if token_id in self.special_token_ids:
                mask[token_id] = 0.0
                count += 1
                continue
            decoded = self.tokenizer.decode([token_id])
            if self.ALLOWED_PATTERN.match(decoded):
                mask[token_id] = 0.0
                count += 1
        logger.info(f"Vocabulary restricted to English charset: {count}/{vocab_size} tokens allowed.")
        return mask.to(device)

    def __call__(self, input_ids, scores):
        vocab_size = scores.shape[-1]
        if (self._mask is None or self._mask.shape[0] != vocab_size
                or self._mask.device != scores.device or self._mask.dtype != scores.dtype):
            self._mask = self._build_mask(vocab_size, scores.device, scores.dtype)
        return scores + self._mask


class TrOCR_RealWeightsRecognizer(TextRecognitionStrategy):

    _CLEAN_PATTERN = re.compile(r'[^A-Za-z0-9\s\.\,\:\;\!\?\-\>\<\_\(\)\[\]\'\"\/\\@#\$%&\*\+=\~\^]')
    _DARK_BG_MEAN_THRESHOLD = 115.0
    _MIN_CROP_DIM_BEFORE_UPSCALE = 40
    _UPSCALE_TARGET_DIM = 64

    def __init__(self, batch_size: int = 32, num_beams: int = 1,
                    model_name: str = "microsoft/trocr-base-printed",
                    use_fp16: bool = True, normalize_polarity: bool = True,
                    upscale_small_crops: bool = True, device: str = None):
        logger.info(f"Loading real weights for TrOCR ({model_name})...")
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.num_beams = num_beams
        self.normalize_polarity = normalize_polarity
        self.upscale_small_crops = upscale_small_crops

        self.model = VisionEncoderDecoderModel.from_pretrained(model_name, use_safetensors=True)
        _remove_unused_pooler(self.model)
        self.model.to(self.device)
        self.model.eval()
        if self.device.type == "cuda":
            self.model = torch.compile(self.model, dynamic=True)

        self.use_fp16 = use_fp16 and self.device.type == "cuda"
        self.model_info = f"TrOCR ({model_name}, beams={num_beams})"
        self.last_infer_time = 0.0

        gen_cfg = self.model.generation_config
        decoder_start_token_id = gen_cfg.decoder_start_token_id
        eos_token_id = gen_cfg.eos_token_id
        pad_token_id = gen_cfg.pad_token_id

        tok = self.processor.tokenizer
        if decoder_start_token_id is None:
            decoder_start_token_id = tok.bos_token_id or tok.cls_token_id
        if eos_token_id is None:
            eos_token_id = tok.eos_token_id
        if pad_token_id is None:
            pad_token_id = tok.pad_token_id

        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        special_ids = {decoder_start_token_id, eos_token_id, pad_token_id}
        self.allowed_chars_processor = AllowedCharsLogitsProcessor(
            tokenizer=self.processor.tokenizer,
            special_token_ids=special_ids
        )

        logger.info(f"TrOCR (printed, English-restricted) loaded successfully on {self.device} "
                    f"[fp16={self.use_fp16}, num_beams={self.num_beams}, batch_size={self.batch_size}].")

    def _autocast(self):
        if self.use_fp16:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _prepare_pil(self, crop_tensor: torch.Tensor) -> Image.Image:
        pil_image = F.to_pil_image(crop_tensor.squeeze(0))

        if self.upscale_small_crops:
            w, h = pil_image.size
            if min(w, h) < self._MIN_CROP_DIM_BEFORE_UPSCALE and min(w, h) > 0:
                factor = self._UPSCALE_TARGET_DIM / min(w, h)
                new_w, new_h = max(1, int(w * factor)), max(1, int(h * factor))
                pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)

        if self.normalize_polarity:
            gray_mean = float(np.array(pil_image.convert("L"), dtype=np.float32).mean())
            if gray_mean < self._DARK_BG_MEAN_THRESHOLD:
                pil_image = ImageOps.invert(pil_image.convert("RGB"))

        return pil_image

    @staticmethod
    def _dynamic_max_new_tokens(box_widths: list) -> int:
        widest = max(box_widths) if box_widths else 0
        estimate = int(widest / 9) + 6
        return max(8, min(48, estimate))

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

        valid_crops.sort(key=lambda c: (c["box"][2] - c["box"][0]))

        total = len(valid_crops)
        logger.info(f"Recognizing {total} words in batches of "
                    f"{self.batch_size if self.batch_size else 'auto'} "
                    f"(English-restricted, beams={self.num_beams})...")

        def _prepare(batch_crops):
            batch_images = [self._prepare_pil(c["crop_tensor"]) for c in batch_crops]
            batch_widths = [c["box"][2] - c["box"][0] for c in batch_crops]

            pixel_values = self.processor(images=batch_images, return_tensors="pt").pixel_values
            if self.device.type == "cuda":
                pixel_values = pixel_values.pin_memory()

            max_new_tokens = self._dynamic_max_new_tokens(batch_widths)
            return pixel_values, max_new_tokens

        def _compute(batch_crops, prepared):
            pixel_values, max_new_tokens = prepared
            pixel_values = pixel_values.to(self.device, non_blocking=True)

            with self._autocast():
                generated_ids = self.model.generate(
                    pixel_values,
                    logits_processor=LogitsProcessorList([self.allowed_chars_processor]),
                    num_beams=self.num_beams,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=self.eos_token_id,
                    pad_token_id=self.pad_token_id,
                )

            texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

            batch_output = []
            for crop_data, text_str in zip(batch_crops, texts):
                text_str = self._CLEAN_PATTERN.sub('', text_str).strip()
                if not text_str:
                    continue
                logger.debug(f" -> Word {crop_data['word_id']} {crop_data['box']}: '{text_str}'")
                batch_output.append({
                    "word_id": crop_data["word_id"],
                    "box": crop_data["box"],
                    "text": text_str
                })
            return batch_output

        recognized_output, self.last_infer_time = run_batches_pipelined(
            valid_crops, _prepare, _compute,
            model_key=f"trocr:{self.model_info}", device=self.device,
            batch_size=self.batch_size, start_batch=4, max_batch=256,
        )

        recognized_output.sort(key=lambda r: r["word_id"])
        return recognized_output