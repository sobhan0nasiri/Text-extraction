import logging
import sys
import threading
import time

import cv2
import torch
from torchvision import transforms
from .base import InputSourceStrategy

logger = logging.getLogger(__name__)


class FileInputSource(InputSourceStrategy):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def get_frame(self) -> torch.Tensor:
        logger.info(f"Loading real image file: {self.file_path}")
        image_bgr = cv2.imread(self.file_path)

        if image_bgr is None:
            raise FileNotFoundError(f"error: File not found at '{self.file_path}'")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        transform = transforms.ToTensor()
        tensor_img = transform(image_rgb).unsqueeze(0)
        return tensor_img


class CameraFrameInputSource(InputSourceStrategy):
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(camera_index, backend)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to connect to camera at index {camera_index}. "
                                f"Check the index or that no other app is using the camera.")

        self.frame_counter = 0
        self._lock = threading.Lock()
        self._latest_frame_bgr = None
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        while self._running:
            ret, frame_bgr = self.cap.read()
            if ret and frame_bgr is not None:
                with self._lock:
                    self._latest_frame_bgr = frame_bgr
                    self.frame_counter += 1
            else:
                time.sleep(0.01)

    def get_frame(self) -> torch.Tensor:
        with self._lock:
            frame_bgr = None if self._latest_frame_bgr is None else self._latest_frame_bgr.copy()

        if frame_bgr is None:
            return None

        cv2.imshow("Live Camera Stream (Press 'q' to stop)", frame_bgr)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        transform = transforms.ToTensor()
        tensor_img = transform(frame_rgb).unsqueeze(0)
        return tensor_img

    def release(self):
        self._running = False
        self._thread.join(timeout=1.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()
            cv2.destroyAllWindows()
            logger.info("Camera connection closed cleanly.")