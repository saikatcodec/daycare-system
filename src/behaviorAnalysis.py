"""Behavioral analysis module using YOLOv12 for cry/fight detection."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from ultralytics import YOLO

from utilities.logger import get_logger

logger = get_logger(__name__)


class BehaviorResult(NamedTuple):
    """Structured result for a single behavioral detection."""
    label: str           # class name (e.g. 'cry', 'fight')
    confidence: float
    bbox: list[float]    # [x1, y1, x2, y2]


class BehaviorAnalysisModel:
    """YOLOv12-based behavior detector for cry and fight events.

    Args:
        model_path: Path to the ``.pt`` checkpoint.
        conf: Minimum detection confidence threshold (0–1).
    """

    def __init__(
        self,
        model_path: str | Path,
        conf: float = 0.9,
    ) -> None:
        self.model_path = Path(model_path)
        self.conf = conf

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Behavior model checkpoint not found: {self.model_path}"
            )

        self.model = YOLO(str(self.model_path))
        logger.info("BehaviorAnalysisModel loaded from '%s'.", self.model_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image) -> list[BehaviorResult]:
        """Run inference and return detected behavioral events.

        Args:
            image: A NumPy BGR array or any source accepted by
                ``YOLO.predict`` (file path, PIL Image, etc.).

        Returns:
            One :class:`BehaviorResult` per detection, ordered by
            descending confidence.  Returns an empty list when nothing
            is detected above the confidence threshold.
        """
        results = self.model.predict(
            image,
            save=False,
            conf=self.conf,
            verbose=False,
        )[0]

        boxes = results.boxes.cpu()
        names = self.model.names

        detections: list[BehaviorResult] = []
        for cl, conf, xyxy in zip(
            boxes.cls.numpy(),
            boxes.conf.numpy(),
            boxes.xyxy.numpy(),
        ):
            detections.append(
                BehaviorResult(
                    label=names[int(cl)],
                    confidence=float(conf),
                    bbox=xyxy.tolist(),
                )
            )

        logger.debug("Detected %d behavior event(s).", len(detections))
        return detections

    def get_class_names(self) -> list[str]:
        """Return all class-name labels from the loaded model.

        Returns:
            List of class names ordered by class index.
        """
        return list(self.model.names.values())
