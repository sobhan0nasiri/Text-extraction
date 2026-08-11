import time
import logging
import cv2
import numpy as np
import torch
import torchvision.ops as ops

from strategies.base import (
    CornerDetectionStrategy,
    ObstacleDetectionStrategy,
    RectificationStrategy,
    TextDetectionStrategy,
    TextRecognitionStrategy
)

logger = logging.getLogger(__name__)


def optimized_nms_merge(boxes, scores, iou_threshold=0.3):
    if not boxes:
        return [], []

    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)

    if boxes_tensor.ndim == 1:
        boxes_tensor = boxes_tensor.reshape(-1, 4)

    keep_indices = ops.nms(boxes_tensor, scores_tensor, iou_threshold)

    final_boxes = boxes_tensor[keep_indices].cpu().numpy().tolist()
    final_scores = scores_tensor[keep_indices].cpu().numpy().tolist()

    return final_boxes, final_scores


def _partition_into_columns(boxes, column_gap_threshold=35):
    if not boxes:
        return []

    sorted_boxes = sorted(boxes, key=lambda b: b["box"][0] if isinstance(b, dict) else b[0])
    columns = []

    for b in sorted_boxes:
        b_box = b["box"] if isinstance(b, dict) else b
        b_x1, b_x2 = b_box[0], b_box[2]
        matched_col = None

        for col in columns:
            col_x1 = min(item["box"][0] if isinstance(item, dict) else item[0] for item in col)
            col_x2 = max(item["box"][2] if isinstance(item, dict) else item[2] for item in col)

            if not (b_x1 > col_x2 + column_gap_threshold or b_x2 < col_x1 - column_gap_threshold):
                matched_col = col
                break

        if matched_col is not None:
            matched_col.append(b)
        else:
            columns.append([b])

    columns.sort(key=lambda col: min(item["box"][0] if isinstance(item, dict) else item[0] for item in col))
    return columns

def smart_merge_boxes(
    boxes,
    enable_merging=False,
    min_area=32,
    y_tolerance_ratio=0.40,
    x_gap_multiplier=2.0,
    column_gap_threshold=40,
    max_merge_count=5,
):
    if not boxes:
        return []

    valid_boxes = []
    for b in boxes:
        box_coords = b["box"] if isinstance(b, dict) and "box" in b else b
        w = box_coords[2] - box_coords[0]
        h = box_coords[3] - box_coords[1]
        if w * h >= min_area and w > 1 and h > 1:
            valid_boxes.append(b if isinstance(b, dict) else {"box": box_coords})

    if not valid_boxes:
        return []

    if not enable_merging:
        sorted_boxes = sorted(valid_boxes, key=lambda b: (b["box"][1], b["box"][0]))
        for i, box in enumerate(sorted_boxes, start=1):
            box["word_id"] = i
        return sorted_boxes

    for b in valid_boxes:
        x1, y1, x2, y2 = b["box"]
        b["cy"] = (y1 + y2) / 2.0
        b["h"] = max(1, y2 - y1)

    columns = _partition_into_columns(valid_boxes, column_gap_threshold=column_gap_threshold)

    merged_all = []
    merged_id_counter = 1

    for col in columns:
        sorted_by_y = sorted(col, key=lambda b: b["cy"])
        y_bands = []
        current_band = [sorted_by_y[0]]

        for word in sorted_by_y[1:]:
            last_word = current_band[-1]
            if abs(word["cy"] - last_word["cy"]) < (last_word["h"] * y_tolerance_ratio):
                current_band.append(word)
            else:
                y_bands.append(current_band)
                current_band = [word]

        if current_band:
            y_bands.append(current_band)

        _BULLET_WIDTH_RATIO = 0.05

        def _is_bullet_box(box, ref_h):
            w = box[2] - box[0]
            h = box[3] - box[1]
            return w <= ref_h * _BULLET_WIDTH_RATIO and h <= ref_h * 1.4

        for band in y_bands:
            band.sort(key=lambda w: w["box"][0])
            avg_h_band = sum(w["h"] for w in band) / len(band)
            for w in band:
                w["is_bullet"] = _is_bullet_box(w["box"], avg_h_band)

            clusters = [[w] for w in band]

            while len(clusters) > 1:
                min_gap = float('inf')
                min_idx = -1

                for i in range(len(clusters) - 1):
                    left_cluster = clusters[i]
                    right_cluster = clusters[i+1]

                    if left_cluster[-1].get("is_bullet") or right_cluster[0].get("is_bullet"):
                        continue

                    if len(left_cluster) + len(right_cluster) <= max_merge_count:
                        gap = right_cluster[0]["box"][0] - left_cluster[-1]["box"][2]
                        avg_h = sum(w["h"] for w in left_cluster + right_cluster) / len(left_cluster + right_cluster)
                        if gap <= (avg_h * x_gap_multiplier):
                            if gap < min_gap:
                                min_gap = gap
                                min_idx = i

                if min_idx == -1:
                    break

                clusters[min_idx].extend(clusters.pop(min_idx + 1))

            for cluster in clusters:
                merged_all.append(_merge_box_group(cluster, merged_id_counter))
                merged_id_counter += 1

    return merged_all


def _merge_box_group(group, word_id):
    coords = [g["box"] for g in group]
    x_min = min(c[0] for c in coords)
    y_min = min(c[1] for c in coords)
    x_max = max(c[2] for c in coords)
    y_max = max(c[3] for c in coords)
    return {"box": [x_min, y_min, x_max, y_max], "word_id": word_id}


class OCRPipeline:
    
    _MIN_FONT_SCALE = 0.25
    _MAX_FONT_SCALE = 1.6
    _RECOGNITION_CROP_MAX_HEIGHT = 64
    
    def __init__(self,
                corner_detector: CornerDetectionStrategy,
                obstacle_detector: ObstacleDetectionStrategy,
                rectifier: RectificationStrategy,
                text_detector: TextDetectionStrategy,
                text_recognizer: TextRecognitionStrategy,
                fallback_recognizer: TextRecognitionStrategy = None,
                crop_pad_ratio: float = 0.2,
                realtime_budget_seconds: float = 2.0,
                debug: bool = False,
                enable_adaptive_resolution: bool = False
                ):

        self.corner_detector = corner_detector
        self.obstacle_detector = obstacle_detector
        self.rectifier = rectifier
        self.text_detector = text_detector
        self.text_recognizer = text_recognizer
        self.fallback_recognizer = fallback_recognizer
        self.crop_pad_ratio = crop_pad_ratio
        self.realtime_budget_seconds = realtime_budget_seconds
        self.debug = debug
        self.enable_adaptive_resolution = enable_adaptive_resolution

        self.is_word_level_recognizer = "FastRecognizer" in self.text_recognizer.__class__.__name__

        self._cached_corners = None
        self._cached_coarse_bbox = None
        self._cached_dynamic_bbox = None
        self._cache_iou_threshold = 0.97
        
    _MIN_PX_PER_CHAR = 4.0

    @classmethod
    def _is_geometrically_suspect(cls, text: str, box: list) -> bool:

        if not text:
            return False
        box_w = max(1, box[2] - box[0])
        max_plausible_chars = box_w / cls._MIN_PX_PER_CHAR
        return len(text) > max_plausible_chars

    @staticmethod
    def _model_time(strategy) -> float:
        return getattr(strategy, "last_infer_time", 0.0)

    @staticmethod
    def _model_info(strategy) -> str:
        return getattr(strategy, "model_info", strategy.__class__.__name__)

    @staticmethod
    def _bbox_iou(box_a, box_b):
        if box_a is None or box_b is None:
            return 0.0
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

    def _get_corners(self, image_tensor, coarse_bbox):
        if (self._cached_corners is not None and
                self._bbox_iou(coarse_bbox, self._cached_coarse_bbox) >= self._cache_iou_threshold):
            logger.info("Reusing cached corners (coarse bbox unchanged since last scan).")
            self.corner_detector.last_infer_time = 0.0
            return self._cached_corners, self._cached_dynamic_bbox

        document_corners, dynamic_bbox = self.corner_detector.detect_corners(image_tensor, coarse_bbox=coarse_bbox)
        if document_corners is not None:
            self._cached_corners = document_corners
            self._cached_coarse_bbox = coarse_bbox
            self._cached_dynamic_bbox = dynamic_bbox
        return document_corners, dynamic_bbox

    def _adjust_inference_resolution(self, over_budget: bool):
        factor = 0.85 if over_budget else 1.1
        for det in (self.obstacle_detector, self.corner_detector):
            if hasattr(det, "max_inference_dim"):
                new_dim = int(det.max_inference_dim * factor)
                new_dim = max(320, min(1600, new_dim))
                if new_dim != det.max_inference_dim:
                    det.max_inference_dim = new_dim
                    logger.info(f"Budget-adaptive resize: {det.__class__.__name__}.max_inference_dim -> {new_dim}")

    def save_text_detection_debug(self, image: torch.Tensor, text_bboxes: list) -> None:
        image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
        image_np = image_np.astype(np.uint8)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        for bbox_info in text_bboxes:
            x1, y1, x2, y2 = bbox_info["box"]
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
            cv2.putText(image_bgr, str(bbox_info["word_id"]), (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        cv2.imwrite("step3_text_boxes.jpg", image_bgr)
        logger.debug(f"Saved text-detection debug image -> step3_text_boxes.jpg ({len(text_bboxes)} boxes)")

    def _fit_text_to_box(self, text: str, font, box_w: int, box_h: int):
        scale = min(self._MAX_FONT_SCALE, max(self._MIN_FONT_SCALE, box_h / 32.0))
        thickness = 2 if scale > 0.9 else 1

        for _ in range(30):
            size, _ = cv2.getTextSize(text, font, scale, thickness)
            if size[0] <= box_w * 0.98 and size[1] <= box_h * 0.92:
                return scale, thickness, size
            scale -= 0.05
            thickness = 2 if scale > 0.9 else 1
            if scale <= self._MIN_FONT_SCALE:
                scale = self._MIN_FONT_SCALE
                thickness = 1
                size, _ = cv2.getTextSize(text, font, scale, thickness)
                return scale, thickness, size

        size, _ = cv2.getTextSize(text, font, scale, thickness)
        return scale, thickness, size

    @staticmethod
    def _truncate_to_fit(text: str, font, scale: float, thickness: int, box_w: int):
        ellipsis = "..."
        lo, hi = 0, len(text)
        best = text[:1] if text else ""

        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = (text[:mid].rstrip() + ellipsis) if mid < len(text) else text
            size, _ = cv2.getTextSize(candidate, font, scale, thickness)
            if size[0] <= box_w * 0.98:
                best = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        size, _ = cv2.getTextSize(best, font, scale, thickness)
        return best, size

    def save_visual_reconstruction(self, image: torch.Tensor, recognized_results: list) -> None:
        h, w = image.shape[-2], image.shape[-1]
        canvas = np.full((h, w, 3), 255, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX

        for item in recognized_results:
            x1, y1, x2, y2 = item["box"]
            text = item["text"]
            if not text:
                continue

            box_w = max(1, x2 - x1)
            box_h = max(1, y2 - y1)

            font_scale, thickness, text_size = self._fit_text_to_box(text, font, box_w, box_h)
            display_text = text

            if text_size[0] > box_w * 1.05 and font_scale <= self._MIN_FONT_SCALE + 1e-6:
                display_text, text_size = self._truncate_to_fit(text, font, font_scale, thickness, box_w)

            text_x = x1 + 2
            text_y = y1 + (box_h + text_size[1]) // 2
            text_y = max(y1 + text_size[1] + 2, min(text_y, y2 - 2))

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color=(245, 245, 245), thickness=-1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color=(200, 200, 200), thickness=1)

            cv2.putText(canvas, display_text, (text_x, text_y),
                        font, font_scale, (30, 30, 30), thickness, cv2.LINE_AA)

        cv2.imwrite("step4_reconstructed.jpg", canvas)
        logger.debug(f"Saved visual-reconstruction debug image -> step4_reconstructed.jpg ({len(recognized_results)} words placed)")

    def save_corner_detection_debug(self, image: torch.Tensor, corners) -> None:
        image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
        image_np = image_np.astype(np.uint8)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        if corners is not None:
            pts = np.array(corners, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(image_bgr, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
            for (x, y) in np.array(corners, dtype=np.int32):
                cv2.circle(image_bgr, (int(x), int(y)), radius=8, color=(0, 0, 255), thickness=-1)
        else:
            cv2.putText(image_bgr, "NO CORNERS DETECTED", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imwrite("step1_corners_detected.jpg", image_bgr)
        logger.debug("Saved corner-detection debug image -> step1_corners_detected.jpg")

    @torch.inference_mode()
    def process_image(self, image_tensor: torch.Tensor):
        if image_tensor is None:
            return None

        logger.info("--- Preprocessing ---")

        detections = self.obstacle_detector.analyze(image_tensor)
        coarse_bbox = detections.screen_bbox
        if coarse_bbox:
            logger.info(f"Laptop/Monitor pre-crop bbox found: {coarse_bbox} (score={detections.screen_score:.2f})")

        t_screen_detect = self._model_time(self.obstacle_detector)

        document_corners, dynamic_bbox = self._get_corners(image_tensor, coarse_bbox)

        if document_corners is None:
            logger.warning("No valid screen corners detected. Please reposition and try again.")
            return None

        logger.info(f"Dynamically calculated Monitor/Paper Bounding Box: {dynamic_bbox}")

        if self.debug:
            self.save_corner_detection_debug(image_tensor, document_corners)

        t_corner = self._model_time(self.corner_detector)

        triggering_obstacle = self.obstacle_detector.check_obstruction(detections, paper_bbox=dynamic_bbox)
        if triggering_obstacle is not None:
            if self.debug and hasattr(self.obstacle_detector, "save_obstacle_debug"):
                img_np_full = (image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                self.obstacle_detector.save_obstacle_debug(
                    img_np_full, dynamic_bbox, triggering_obstacle.box, triggering_obstacle.label
                )
            logger.warning("Obstacle detected on the paper. Skipping current frame.")
            return None

        rectified_image = self.rectifier.rectify(image_tensor, corners=document_corners)
        t_rectify = self._model_time(self.rectifier)

        t0 = time.perf_counter()
        optimized_image = self.optimize_image(rectified_image)
        t_optimize = time.perf_counter() - t0

        if self.debug:
            rectified_np = rectified_image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
            rectified_np = np.clip(rectified_np, 0, 255).astype(np.uint8)
            cv2.imwrite("step1_flattened.jpg", cv2.cvtColor(rectified_np, cv2.COLOR_RGB2BGR))
            logger.debug("Saved flattened/rectified debug image -> step1_flattened.jpg")

        logger.info("--- Text Detection ---")

        text_bboxes = self.text_detector.detect_text_boxes(optimized_image)
        t_detection = self._model_time(self.text_detector)
        t_detection_merge = getattr(self.text_detector, "last_merge_time", 0.0)

        if text_bboxes:
            boxes_t = torch.tensor([b["box"] for b in text_bboxes], dtype=torch.float32)
            scores_t = torch.tensor([b.get("score", 1.0) for b in text_bboxes], dtype=torch.float32)
            keep = ops.nms(boxes_t, scores_t, iou_threshold=0.5)
            text_bboxes = [text_bboxes[i] for i in keep.tolist()]
        
        t0 = time.perf_counter()
        text_bboxes = smart_merge_boxes(
            text_bboxes
        )
        t_box_merge = time.perf_counter() - t0

        if self.debug:
            self.save_text_detection_debug(optimized_image, text_bboxes)

        logger.info("--- Text Recognition ---")
        actual_crops = []
        h, w = optimized_image.shape[-2:]

        for bbox_info in text_bboxes:
            x1, y1, x2, y2 = bbox_info["box"]
            box_h = y2 - y1

            pad_x = max(4, min(14, int(box_h * self.crop_pad_ratio)))
            pad_y = max(4, min(12, int(box_h * self.crop_pad_ratio * 0.9)))

            cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            cx2, cy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

            if cx1 >= cx2 or cy1 >= cy2:
                continue

            crop_tensor = optimized_image[..., cy1:cy2, cx1:cx2]
            crop_tensor = self._shrink_crop(crop_tensor, self._RECOGNITION_CROP_MAX_HEIGHT)
            actual_crops.append({
                "word_id": bbox_info["word_id"],
                "box": [x1, y1, x2, y2],
                "crop_tensor": crop_tensor
            })

        final_text_results = self.text_recognizer.recognize_text(actual_crops)
        t_recognition = self._model_time(self.text_recognizer)
        
        t_fallback = 0.0
        fallback_word_count = 0
        if self.fallback_recognizer is not None:
            crop_by_id = {c["word_id"]: c for c in actual_crops}
            weak_ids = [
                r["word_id"] for r in final_text_results
                if r.get("conf", 1.0) < 0.97
                or self._is_geometrically_suspect(r["text"], r["box"])
            ]
            weak_crops = [crop_by_id[wid] for wid in weak_ids if wid in crop_by_id]
            if weak_crops:
                refined = self.fallback_recognizer.recognize_text(weak_crops)
                t_fallback = self._model_time(self.fallback_recognizer)
                fallback_word_count = len(weak_crops)
                refined_map = {r["word_id"]: r for r in refined}
                for r in final_text_results:
                    if r["word_id"] in refined_map and refined_map[r["word_id"]]["text"]:
                        r["text"] = refined_map[r["word_id"]]["text"]

        line_results = self.group_words_into_lines(final_text_results)

        if self.debug:
            self.save_visual_reconstruction(optimized_image, line_results)

        total_time = (t_screen_detect + t_corner + t_rectify + t_optimize +
                        t_detection + t_detection_merge + t_box_merge + t_recognition + t_fallback)

        logger.info("--- Pipeline Time Summary ---")
        logger.info("-" * 60)
        logger.info(f"[TIME] Total model+algorithm time: {total_time:.2f}s (budget: {self.realtime_budget_seconds:.2f}s)")
        logger.info(f"  - Screen/Obstacle Detection [{self._model_info(self.obstacle_detector)}]: {t_screen_detect:.2f}s")
        logger.info(f"  - Corner Detection          [{self._model_info(self.corner_detector)}]: {t_corner:.2f}s")
        logger.info(f"  - Image Optimization        [CLAHE algorithm]: {t_optimize:.2f}s")
        logger.info(f"  - Rectification             [{self._model_info(self.rectifier)}]: {t_rectify:.2f}s")
        logger.info(f"  - Text Detection            [{self._model_info(self.text_detector)}]: {t_detection:.2f}s")
        if t_detection_merge:
            logger.info(f"  - Detection Ensemble Merge  [algorithm]: {t_detection_merge:.2f}s")
        logger.info(f"  - Box Merging               [smart_merge_boxes algorithm]: {t_box_merge:.2f}s")
        logger.info(f"  - Text Recognition          [{self._model_info(self.text_recognizer)}]: {t_recognition:.2f}s ({len(final_text_results)} words)")
        if self.fallback_recognizer is not None:
            logger.info(f"  - Fallback Recognition      [{self._model_info(self.fallback_recognizer)}]: {t_fallback:.2f}s ({fallback_word_count} words)")
        logger.info("-" * 60)

        if self.enable_adaptive_resolution:
            if total_time > self.realtime_budget_seconds:
                self._adjust_inference_resolution(over_budget=True)
            elif total_time < self.realtime_budget_seconds * 0.5:
                self._adjust_inference_resolution(over_budget=False)

        return line_results

    @staticmethod
    def _shrink_crop(crop_tensor, max_height):
        h = crop_tensor.shape[-2]
        if h <= max_height:
            return crop_tensor
        scale = max_height / h
        new_w = max(1, int(crop_tensor.shape[-1] * scale))
        return torch.nn.functional.interpolate(crop_tensor, size=(max_height, new_w), mode="area")

    def optimize_image(self, image: torch.Tensor) -> torch.Tensor:
        img_tensor_cpu = image.squeeze(0).permute(1, 2, 0).cpu()
        img_tensor_cpu = torch.nan_to_num(img_tensor_cpu, nan=0.0, posinf=1.0, neginf=0.0)
        img_np = (img_tensor_cpu.numpy() * 255.0).clip(0, 255).astype(np.uint8)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        small_clahe = clahe.apply(small)
        gray_clahe = cv2.resize(small_clahe, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)

        blurred = cv2.GaussianBlur(gray_clahe, (0, 0), sigmaX=1.0)
        gray_sharp = cv2.addWeighted(gray_clahe, 1.15, blurred, -0.15, 0)
        optimized_np = cv2.cvtColor(gray_sharp, cv2.COLOR_GRAY2RGB)

        optimized_tensor = torch.from_numpy(optimized_np / 255.0).permute(2, 0, 1).unsqueeze(0).float()
        return optimized_tensor.to(image.device)

    @staticmethod
    def _create_line_dict(words: list, line_idx: int) -> dict:
        merged_text = " ".join(w["text"] for w in words)
        min_x = min(w["box"][0] for w in words)
        min_y = min(w["box"][1] for w in words)
        max_x = max(w["box"][2] for w in words)
        max_y = max(w["box"][3] for w in words)
        return {
            "line_id": line_idx,
            "text": merged_text,
            "box": [min_x, min_y, max_x, max_y],
            "words_count": len(words)
        }

    @staticmethod
    def group_words_into_lines(recognized_results: list, y_tolerance_ratio: float = 0.28, x_gap_multiplier: float = 2.0) -> list:
        if not recognized_results:
            return []

        for res in recognized_results:
            x1, y1, x2, y2 = res["box"]
            res["cy"] = (y1 + y2) / 2.0
            res["h"] = max(1, y2 - y1)

        columns = _partition_into_columns(recognized_results, column_gap_threshold=35)
        
        final_lines = []
        line_idx = 1

        for col in columns:
            sorted_by_y = sorted(col, key=lambda r: r["cy"])
            y_bands = []
            current_band = [sorted_by_y[0]]

            for word in sorted_by_y[1:]:
                
                last_word = current_band[-1]
                
                if abs(word["cy"] - last_word["cy"]) < (last_word["h"] * y_tolerance_ratio):
                    current_band.append(word)
                else:
                    y_bands.append(current_band)
                    current_band = [word]

            if current_band:
                y_bands.append(current_band)

            for band in y_bands:
                band.sort(key=lambda w: w["box"][0])
                current_line = [band[0]]

                for word in band[1:]:
                    prev_word = current_line[-1]
                    avg_h = sum(w["h"] for w in current_line) / len(current_line)
                    gap = word["box"][0] - prev_word["box"][2]

                    if gap > (avg_h * x_gap_multiplier):
                        final_lines.append(OCRPipeline._create_line_dict(current_line, line_idx))
                        line_idx += 1
                        current_line = [word]
                    else:
                        current_line.append(word)

                if current_line:
                    final_lines.append(OCRPipeline._create_line_dict(current_line, line_idx))
                    line_idx += 1

        return final_lines
    