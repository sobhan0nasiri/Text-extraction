import time
import logging
import torch
import kornia
import numpy as np
from .base import RectificationStrategy

logger = logging.getLogger(__name__)


class DynamicRectifier(RectificationStrategy):

    def __init__(self, max_output_dim: int = 5760):
        self.max_output_dim = max_output_dim
        self.model_info = "Perspective Warp (kornia get_perspective_transform + warp_perspective)"
        self.last_infer_time = 0.0

    def rectify(self, image: torch.Tensor, corners: list = None) -> torch.Tensor:
        if corners is None:
            raise ValueError("rectify() requires valid `corners`; received None.")

        pts = np.array(corners, dtype=np.float32).reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        widthA = np.linalg.norm(rect[2] - rect[3])
        widthB = np.linalg.norm(rect[1] - rect[0])
        maxWidth = int(max(widthA, widthB))

        heightA = np.linalg.norm(rect[1] - rect[2])
        heightB = np.linalg.norm(rect[0] - rect[3])
        maxHeight = int(max(heightA, heightB))

        scale = self.max_output_dim / max(maxWidth, maxHeight)
        outW, outH = int(maxWidth * scale), int(maxHeight * scale)

        src_points = torch.tensor([rect.tolist()], device=image.device, dtype=image.dtype)
        dst_points = torch.tensor([[[0., 0.], [outW, 0.], [outW, outH], [0., outH]]], device=image.device, dtype=image.dtype)

        t0 = time.perf_counter()
        perspective_transform = kornia.geometry.transform.get_perspective_transform(src_points, dst_points)
        rectified_image = kornia.geometry.transform.warp_perspective(
            image, perspective_transform, dsize=(outH, outW), mode='bilinear', align_corners=True
        )
        self.last_infer_time = time.perf_counter() - t0

        logger.info(f"Image rectified dynamically successfully (Resolution: {outW}x{outH}).")
        return rectified_image