from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import torch


@dataclass
class DetectedObject:
    label: str
    class_id: int
    score: float
    box: list


@dataclass
class FrameDetections:

    screen_bbox: list = None
    screen_score: float = 0.0
    objects: list = field(default_factory=list)


class ObstacleDetectionStrategy(ABC):
    @abstractmethod
    def analyze(self, image: torch.Tensor) -> FrameDetections:

        pass

    @abstractmethod
    def check_obstruction(self, detections: FrameDetections, paper_bbox: list):
        pass


class RectificationStrategy(ABC):
    @abstractmethod
    def rectify(self, image: torch.Tensor, corners: list = None) -> torch.Tensor:
        pass


class TextDetectionStrategy(ABC):
    @abstractmethod
    def detect_text_boxes(self, image: torch.Tensor) -> list:
        pass


class TextRecognitionStrategy(ABC):
    @abstractmethod
    def recognize_text(self, image_crops: list) -> list:
        pass


class InputSourceStrategy(ABC):
    @abstractmethod
    def get_frame(self) -> torch.Tensor:
        pass


class CornerDetectionStrategy(ABC):
    @abstractmethod
    def detect_corners(self, image: torch.Tensor, coarse_bbox: list = None) -> tuple:
        pass