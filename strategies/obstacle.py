import time
import logging
import torch
import numpy as np
import cv2
from contextlib import nullcontext
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from .base import ObstacleDetectionStrategy, FrameDetections, DetectedObject
from parallel_utils import get_frame_rgb_uint8

logger = logging.getLogger(__name__)


class RTDETR_RealWeightsDetector(ObstacleDetectionStrategy):

    _BACKBONES = {
        "r18vd": "PekingU/rtdetr_v2_r18vd",
        "r34vd": "PekingU/rtdetr_v2_r34vd",
        "r50vd": "PekingU/rtdetr_v2_r50vd",
        "r101vd": "PekingU/rtdetr_v2_r101vd",
    }

    def __init__(self, backbone: str = "r34vd", use_fp16: bool = True,
                screen_threshold: float = 0.3, obstacle_threshold: float = 0.5,
                max_inference_dim: int = 1024, obstruction_area_ratio: float = 0.01):
        checkpoint = self._BACKBONES.get(backbone, backbone)
        logger.info(f"Loading real weights for RT-DETRv2 ({checkpoint})...")
        self.processor = AutoImageProcessor.from_pretrained(checkpoint)
        self.model = AutoModelForObjectDetection.from_pretrained(checkpoint)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.use_fp16 = use_fp16 and self.device.type == "cuda"
        if self.use_fp16:
            logger.info("RT-DETRv2 will run under autocast(fp16) on CUDA.")

        self.target_class_names = {"person"}
        self.screen_class_names = {"laptop", "tvmonitor"}
        self.screen_threshold = screen_threshold
        self.obstacle_threshold = obstacle_threshold
        self.max_inference_dim = max_inference_dim
        self.obstruction_area_ratio = obstruction_area_ratio

        self.model_info = f"RT-DETRv2 ({checkpoint})"
        self.last_infer_time = 0.0

        label2id = {name.lower(): idx for idx, name in self.model.config.id2label.items()}
        self.target_class_ids = {label2id[n] for n in self.target_class_names if n in label2id}
        self.screen_class_ids = {label2id[n] for n in self.screen_class_names if n in label2id}

        missing = (self.target_class_names | self.screen_class_names) - set(label2id.keys())
        for name in missing:
            logger.warning(f"Class '{name}' not found in model's id2label map; skipped.")

        logger.info(f"RT-DETRv2 ({backbone}) loaded successfully on {self.device}. "
                    f"Obstacle classes: {self.target_class_names} -> {self.target_class_ids} | "
                    f"Screen classes: {self.screen_class_names} -> {self.screen_class_ids}")

    def _autocast(self):
        if self.use_fp16:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    @torch.inference_mode()
    def analyze(self, image: torch.Tensor) -> FrameDetections:
        img_np_full = get_frame_rgb_uint8(image)
        orig_h, orig_w = img_np_full.shape[:2]

        scale = min(1.0, self.max_inference_dim / max(orig_h, orig_w))
        if scale < 1.0:
            infer_w, infer_h = int(orig_w * scale), int(orig_h * scale)
            img_np = cv2.resize(img_np_full, (infer_w, infer_h), interpolation=cv2.INTER_AREA)
        else:
            img_np = img_np_full

        logger.info(f"Screen+Obstacle inference resolution: {img_np.shape[1]}x{img_np.shape[0]} (scale={scale:.2f})")

        target_sizes = torch.tensor([img_np.shape[:2]]).to(self.device)
        inputs = self.processor(images=img_np, return_tensors="pt").to(self.device)

        t0 = time.perf_counter()
        with self._autocast():
            outputs = self.model(**inputs)
        self.last_infer_time = time.perf_counter() - t0

        low_threshold = min(self.screen_threshold, self.obstacle_threshold)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=low_threshold
        )[0]

        detections = FrameDetections()
        logger.debug("RT-DETR single-pass output:")

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            class_id = label.item()
            confidence = score.item()
            xmin, ymin, xmax, ymax = [int(v / scale) for v in box.tolist()]
            class_name = self.model.config.id2label.get(class_id, f"id_{class_id}")

            if class_id in self.screen_class_ids and confidence >= self.screen_threshold:
                logger.debug(f" -> [screen] {class_name} | conf={confidence:.2f} | box={[xmin, ymin, xmax, ymax]}")
                if confidence > detections.screen_score:
                    detections.screen_bbox = [xmin, ymin, xmax, ymax]
                    detections.screen_score = confidence

            elif class_id in self.target_class_ids and confidence >= self.obstacle_threshold:
                logger.debug(f" -> [obstacle] {class_name} | conf={confidence:.2f} | box={[xmin, ymin, xmax, ymax]}")
                detections.objects.append(DetectedObject(
                    label=class_name, class_id=class_id, score=confidence,
                    box=[xmin, ymin, xmax, ymax]
                ))

        if detections.screen_bbox is None:
            logger.info("No laptop/tv screen detected in this frame.")
        if not detections.objects:
            logger.info(f"No obstacle-class objects ({'/'.join(sorted(self.target_class_names))}) detected.")

        return detections

    def check_obstruction(self, detections: FrameDetections, paper_bbox: list):
        pxmin, pymin, pxmax, pymax = paper_bbox
        paper_area = max(1, (pxmax - pxmin) * (pymax - pymin))

        for obj in detections.objects:
            xmin, ymin, xmax, ymax = obj.box
            ixmin, iymin = max(xmin, pxmin), max(ymin, pymin)
            ixmax, iymax = min(xmax, pxmax), min(ymax, pymax)

            if ixmin < ixmax and iymin < iymax:
                intersection_area = (ixmax - ixmin) * (iymax - iymin)
                if intersection_area > self.obstruction_area_ratio * paper_area:
                    logger.warning(f"Obstructing object '{obj.label}' detected on the monitor/paper! "
                                    f"Intersection Area: {intersection_area}px "
                                    f"({intersection_area / paper_area:.1%} of paper) | Confidence: {obj.score:.2f}")
                    return obj

        logger.info("No obstructing objects detected on the paper/monitor. Proceeding...")
        return None

    def save_obstacle_debug(self, img_np: np.ndarray, paper_bbox: list, obstacle_bbox: list, class_name: str) -> None:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        pxmin, pymin, pxmax, pymax = paper_bbox
        cv2.rectangle(img_bgr, (pxmin, pymin), (pxmax, pymax), color=(0, 255, 0), thickness=3)
        cv2.putText(img_bgr, "Monitor", (pxmin, max(0, pymin - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        oxmin, oymin, oxmax, oymax = obstacle_bbox
        cv2.rectangle(img_bgr, (oxmin, oymin), (oxmax, oymax), color=(0, 0, 255), thickness=3)
        cv2.putText(img_bgr, f"Obstacle: {class_name}", (oxmin, max(0, oymin - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imwrite("step2_obstacle_detected.jpg", img_bgr)
        logger.debug("Saved obstacle-detection debug image -> step2_obstacle_detected.jpg")