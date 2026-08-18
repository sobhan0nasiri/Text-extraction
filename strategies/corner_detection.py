import time
import logging
import cv2
import numpy as np
import torch
from docaligner import DocAligner
from .base import CornerDetectionStrategy
from parallel_utils import get_frame_rgb_uint8

logger = logging.getLogger(__name__)


class DocAligner_RealWeightsDetector(CornerDetectionStrategy):

    def __init__(self, debug: bool = False, max_inference_dim: int = 1280):
        logger.info("Loading real weights for DocAligner (heatmap-regression corner detector)...")
        self.model = DocAligner()
        self.max_inference_dim = max_inference_dim
        self.debug = debug
        self.model_info = "DocAligner (heatmap-regression corner detector)"
        self.last_infer_time = 0.0
        logger.info("DocAligner loaded successfully.")

    @torch.inference_mode()
    def detect_corners(self, image: torch.Tensor, coarse_bbox=None) -> tuple:
        t0 = time.perf_counter()
        try:
            return self._detect_corners_impl(image, coarse_bbox)
        finally:
            self.last_infer_time = time.perf_counter() - t0

    def _detect_corners_impl(self, image: torch.Tensor, coarse_bbox=None) -> tuple:
        img_np_full = get_frame_rgb_uint8(image)
        img_bgr_full = cv2.cvtColor(img_np_full, cv2.COLOR_RGB2BGR)

        h, w = img_bgr_full.shape[:2]

        if coarse_bbox is not None:
            cxmin, cymin, cxmax, cymax = coarse_bbox
            pad_x = int((cxmax - cxmin) * 0.08)
            pad_y = int((cymax - cymin) * 0.08)
            cxmin = max(0, cxmin - pad_x)
            cymin = max(0, cymin - pad_y)
            cxmax = min(w, cxmax + pad_x)
            cymax = min(h, cymax + pad_y)
            crop_x, crop_y = cxmin, cymin
            img_bgr_source = img_bgr_full[cymin:cymax, cxmin:cxmax]
            logger.info(f"Cropping to coarse bbox before corner detection: {[cxmin, cymin, cxmax, cymax]}")
        else:
            crop_x, crop_y = 0, 0
            img_bgr_source = img_bgr_full

        src_h, src_w = img_bgr_source.shape[:2]
        scale = min(1.0, self.max_inference_dim / max(src_h, src_w))
        if scale < 1.0:
            infer_w, infer_h = int(src_w * scale), int(src_h * scale)
            img_bgr = cv2.resize(img_bgr_source, (infer_w, infer_h), interpolation=cv2.INTER_AREA)
        else:
            img_bgr = img_bgr_source

        logger.info(f"Corner-detection inference resolution: {img_bgr.shape[1]}x{img_bgr.shape[0]} (scale={scale:.2f})")

        if self.debug:
            cv2.imwrite("debug_corner_model_input.jpg", img_bgr)
            logger.debug("Saved exact image fed to DocAligner -> debug_corner_model_input.jpg")

        polygon = self.model(img_bgr)

        source = "docaligner"
        if polygon is None or len(polygon) != 4:
            logger.warning("DocAligner could not find a valid 4-corner document/monitor.")
            fallback_corners = self._contour_fallback(img_bgr)
            if fallback_corners is not None:
                logger.info("Recovered corners using contour-based fallback.")
                corners = fallback_corners
                source = "contour"
            elif coarse_bbox is not None:
                logger.info("Contour fallback also failed. Using RT-DETR screen bbox as corner fallback.")
                ih, iw = img_bgr.shape[:2]
                corners = np.array([[0, 0], [iw, 0], [iw, ih], [0, ih]], dtype=np.float32)
                source = "screen_bbox"
            else:
                logger.warning("All corner-detection strategies failed. Falling back to full frame.")
                return None, [0, 0, w, h]
        else:
            corners = np.array(polygon, dtype=np.float32)

        if scale < 1.0:
            corners = corners / scale
        corners[:, 0] += crop_x
        corners[:, 1] += crop_y

        xmin = int(max(0, corners[:, 0].min()))
        ymin = int(max(0, corners[:, 1].min()))
        xmax = int(min(w, corners[:, 0].max()))
        ymax = int(min(h, corners[:, 1].max()))

        logger.info(f"Detected corners (source={source}): {corners.tolist()}")
        logger.info(f"Derived bounding box from corners: {[xmin, ymin, xmax, ymax]}")

        return corners, [xmin, ymin, xmax, ymax]

    @staticmethod
    def _contour_fallback(img_bgr: np.ndarray):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        img_area = img_bgr.shape[0] * img_bgr.shape[1]
        if cv2.contourArea(largest) < 0.12 * img_area:
            return None

        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        return box.astype(np.float32)