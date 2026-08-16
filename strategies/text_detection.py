import time
import logging
import torch
from contextlib import nullcontext
from concurrent.futures import as_completed
from doctr.models import detection_predictor
import numpy as np
import cv2
from .base import TextDetectionStrategy
from parallel_utils import get_shared_executor, get_cuda_streams

import torchvision.models.vgg as _tv_vgg

logger = logging.getLogger(__name__)

if not hasattr(_tv_vgg, "model_urls"):
    _tv_vgg.model_urls = {
        "vgg11": "https://download.pytorch.org/models/vgg11-bbd30ac9.pth",
        "vgg13": "https://download.pytorch.org/models/vgg13-c768596a.pth",
        "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
        "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
        "vgg11_bn": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
        "vgg13_bn": "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
        "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
        "vgg19_bn": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
    }


class DocTR_RealWeightsDetector(TextDetectionStrategy):

    def __init__(self, arch: str = "db_resnet50", conf_threshold: float = 0.22, use_fp16: bool = True,
                    bin_thresh: float = 0.25, box_thresh: float = 0.25, unclip_ratio: float = 1.4,
                    detect_input_size: int = 2056):
        
        logger.info(f"Loading real weights for docTR detector ({arch})...")
        self.model = detection_predictor(arch=arch, pretrained=True, preserve_aspect_ratio=False)
        self.conf_threshold = conf_threshold
        self.detect_input_size = detect_input_size

        pp = getattr(self.model, "postprocessor", None)
        if pp is None and hasattr(self.model, "model"):
            pp = getattr(self.model.model, "postprocessor", None)
            
        applied = {}
        for name, value in (("bin_thresh", bin_thresh), ("box_thresh", box_thresh), ("unclip_ratio", unclip_ratio)):
            if pp is not None and hasattr(pp, name):
                setattr(pp, name, value)
                applied[name] = value
            else:
                logger.warning(f"docTR postprocessor has no attribute '{name}'; skipped.")
        logger.info(f"DBNet postprocessor overrides applied: {applied}")

        self.resize_t = getattr(getattr(self.model, "pre_processor", None), "resize", None)
        if self.resize_t is not None and hasattr(self.resize_t, "size"):
            logger.info(f"docTR detector native input size was {self.resize_t.size}. Will be dynamically scaled to max length {self.detect_input_size}.")
        else:
            logger.warning("Could not locate pre_processor.resize.size to inspect/override input resolution.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            self.model = self.model.cuda()

        self.use_fp16 = use_fp16 and self.device.type == "cuda"
        self.model_info = f"docTR DBNet ({arch})"
        self.last_infer_time = 0.0

        logger.info(f"docTR detection weights loaded successfully on {self.device} "
                    f"[arch={arch}, fp16={self.use_fp16}].")

    def _autocast(self):
        if self.use_fp16:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    @torch.inference_mode()
    def detect_text_boxes(self, image: torch.Tensor) -> list:
        h, w = image.shape[-2], image.shape[-1]

        img_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)

        if self.detect_input_size is not None:
            scale = self.detect_input_size / max(h, w)
            new_h = max(32, int(np.round((h * scale) / 32.0) * 32))
            new_w = max(32, int(np.round((w * scale) / 32.0) * 32))

            infer_img = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            if self.resize_t is not None:
                self.resize_t.size = (new_h, new_w)
        else:
            infer_img = img_np

        t0 = time.perf_counter()
        with self._autocast():
            out = self.model([infer_img])
        self.last_infer_time = time.perf_counter() - t0

        detected_boxes, word_id = [], 1
        for xmin_norm, ymin_norm, xmax_norm, ymax_norm, conf in out[0]['words']:
            if conf < self.conf_threshold:
                continue
            x1 = max(0, min(int(round(xmin_norm * w)), w))
            y1 = max(0, min(int(round(ymin_norm * h)), h))
            x2 = max(0, min(int(round(xmax_norm * w)), w))
            y2 = max(0, min(int(round(ymax_norm * h)), h))
            if x1 >= x2 or y1 >= y2:
                continue
            detected_boxes.append({"word_id": word_id, "box": [x1, y1, x2, y2], "score": float(conf)})
            word_id += 1

        return detected_boxes


def _iou(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def merge_multi_detector_boxes(all_boxes, iou_threshold=0.35):
    if not all_boxes:
        return []

    boxes = [dict(b) for b in all_boxes]
    used = [False] * len(boxes)
    clusters = []

    for i in range(len(boxes)):
        if used[i]:
            continue
        cluster = [boxes[i]]
        used[i] = True

        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue
            if _iou(boxes[i]["box"], boxes[j]["box"]) > iou_threshold:
                cluster.append(boxes[j])
                used[j] = True

        clusters.append(cluster)

    result = []
    for idx, cluster in enumerate(clusters, start=1):
        x1_avg = int(np.mean([b["box"][0] for b in cluster]))
        y1_avg = int(np.mean([b["box"][1] for b in cluster]))
        x2_avg = int(np.mean([b["box"][2] for b in cluster]))
        y2_avg = int(np.mean([b["box"][3] for b in cluster]))

        result.append({
            "word_id": idx,
            "box": [x1_avg, y1_avg, x2_avg, y2_avg],
            "votes": len(cluster)
        })

    return result


class DocTR_MultiArchDetector(TextDetectionStrategy):
    def __init__(self, archs=("db_resnet50", "db_mobilenet_v3_large", "linknet_resnet18"),
                conf_threshold: float = 0.22, use_fp16: bool = True):
        self.detectors = [DocTR_RealWeightsDetector(arch=a, conf_threshold=conf_threshold, use_fp16=use_fp16)
                            for a in archs]
        self.model_info = f"docTR ensemble ({', '.join(archs)})"
        self.last_infer_time = 0.0
        self.last_merge_time = 0.0
        self._executor = get_shared_executor()
        self._streams = get_cuda_streams(len(self.detectors))
        logger.info(f"DocTR multi-arch ensemble ready with archs={archs} (stream-parallel dispatch, {len(self.detectors)} models)")

    @staticmethod
    def _run_on_stream(det, image, stream):
        if stream is not None:
            with torch.cuda.stream(stream):
                boxes = det.detect_text_boxes(image)
            stream.synchronize()
            return boxes
        return det.detect_text_boxes(image)

    def detect_text_boxes(self, image: torch.Tensor) -> list:
        futures = {
            self._executor.submit(self._run_on_stream, det, image, self._streams[i]): det
            for i, det in enumerate(self.detectors)
        }
        all_boxes = []
        for fut in as_completed(futures):
            all_boxes.extend(fut.result())
        self.last_infer_time = sum(det.last_infer_time for det in self.detectors)

        t0 = time.perf_counter()
        merged = merge_multi_detector_boxes(all_boxes, iou_threshold=0.35)
        self.last_merge_time = time.perf_counter() - t0

        logger.debug(f"DocTR ensemble merged {len(all_boxes)} raw boxes -> {len(merged)} final boxes.")
        return merged


class CRAFT_TextDetector(TextDetectionStrategy):
    def __init__(self, text_threshold: float = 0.7, link_threshold: float = 0.4, low_text: float = 0.4):
        from craft_text_detector import Craft
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.craft = Craft(output_dir=None, crop_type="box", cuda=(self.device.type == "cuda"),
                            text_threshold=text_threshold, link_threshold=link_threshold, low_text=low_text)
        self.model_info = "CRAFT"
        self.last_infer_time = 0.0
        logger.info(f"CRAFT text detector loaded on {self.device}.")

    def detect_text_boxes(self, image: torch.Tensor) -> list:
        img_np = (image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        t0 = time.perf_counter()
        prediction = self.craft.detect_text(img_np)
        self.last_infer_time = time.perf_counter() - t0

        detected_boxes = []
        for i, poly in enumerate(prediction["boxes"], start=1):
            xs, ys = poly[:, 0], poly[:, 1]
            box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            detected_boxes.append({"word_id": i, "box": box})
        logger.debug(f"CRAFT found {len(detected_boxes)} regions.")
        return detected_boxes


class EAST_TextDetector(TextDetectionStrategy):
    def __init__(self, model_path: str = "frozen_east_text_detection.pb",
                conf_threshold: float = 0.5, nms_threshold: float = 0.4,
                input_size: tuple = (1280, 1280)):
        self.net = cv2.dnn.readNet(model_path)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_w, self.input_h = input_size
        self.model_info = f"EAST ({model_path})"
        self.last_infer_time = 0.0
        logger.info(f"EAST detector loaded from {model_path}.")

    def detect_text_boxes(self, image: torch.Tensor) -> list:
        img_np = (image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        orig_h, orig_w = img_bgr.shape[:2]
        rW, rH = orig_w / float(self.input_w), orig_h / float(self.input_h)
        resized = cv2.resize(img_bgr, (self.input_w, self.input_h))

        blob = cv2.dnn.blobFromImage(resized, 1.0, (self.input_w, self.input_h),
                                    (123.68, 116.78, 103.94), swapRB=True, crop=False)
        self.net.setInput(blob)

        t0 = time.perf_counter()
        scores, geometry = self.net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])
        self.last_infer_time = time.perf_counter() - t0

        boxes, confidences = self._decode(scores, geometry, self.conf_threshold)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)

        detected_boxes = []
        if len(indices) > 0:
            for word_id, i in enumerate(indices.flatten(), start=1):
                x, y, w, h = boxes[i]
                box = [int(x * rW), int(y * rH), int((x + w) * rW), int((y + h) * rH)]
                detected_boxes.append({"word_id": word_id, "box": box})
        logger.debug(f"EAST found {len(detected_boxes)} regions.")
        return detected_boxes

    @staticmethod
    def _decode(scores, geometry, min_conf):
        num_rows, num_cols = scores.shape[2:4]
        boxes, confidences = [], []
        for y in range(num_rows):
            scores_data = scores[0, 0, y]
            x0, x1, x2, x3 = (geometry[0, i, y] for i in range(4))
            angles = geometry[0, 4, y]
            for x in range(num_cols):
                if scores_data[x] < min_conf:
                    continue
                offset_x, offset_y = x * 4.0, y * 4.0
                angle = angles[x]
                cos, sin = np.cos(angle), np.sin(angle)
                h = x0[x] + x2[x]
                w = x1[x] + x3[x]
                end_x = int(offset_x + (cos * x1[x]) + (sin * x2[x]))
                end_y = int(offset_y - (sin * x1[x]) + (cos * x2[x]))
                start_x, start_y = int(end_x - w), int(end_y - h)
                boxes.append((start_x, start_y, int(w), int(h)))
                confidences.append(float(scores_data[x]))
        return boxes, confidences


class MultiModelTextDetector(TextDetectionStrategy):
    def __init__(self, detectors: list):
        self.detectors = detectors
        self.model_info = " + ".join(getattr(d, "model_info", d.__class__.__name__) for d in detectors)
        self.last_infer_time = 0.0
        self.last_merge_time = 0.0
        self._executor = get_shared_executor()
        self._streams = get_cuda_streams(len(self.detectors))
        logger.info(f"Full text-detection ensemble ready with {len(detectors)} models (stream-parallel dispatch).")

    @staticmethod
    def _run_on_stream(det, image, stream):
        if stream is not None:
            with torch.cuda.stream(stream):
                boxes = det.detect_text_boxes(image)
            stream.synchronize()
            return boxes
        return det.detect_text_boxes(image)

    def detect_text_boxes(self, image: torch.Tensor) -> list:
        futures = {
            self._executor.submit(self._run_on_stream, det, image, self._streams[i]): det
            for i, det in enumerate(self.detectors)
        }
        all_boxes = []
        infer_time = 0.0
        for fut in as_completed(futures):
            det = futures[fut]
            try:
                all_boxes.extend(fut.result())
                infer_time += getattr(det, "last_infer_time", 0.0)
            except Exception as e:
                logger.warning(f"{det.__class__.__name__} failed: {e}")
        self.last_infer_time = infer_time

        t0 = time.perf_counter()
        merged = merge_multi_detector_boxes(all_boxes, iou_threshold=0.35)
        self.last_merge_time = time.perf_counter() - t0

        logger.debug(f"Full ensemble merged {len(all_boxes)} raw boxes -> {len(merged)} final boxes.")
        return merged