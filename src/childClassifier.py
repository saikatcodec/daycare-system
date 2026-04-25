"""Child detection and tracking module using YOLOv12 + BoT-SORT."""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

from utilities.logger import get_logger

logger = get_logger(__name__)

# Class index for 'child' in the trained model
_CHILD_CLASS_IDX = 1


class ChildModel:
    """YOLOv12-based child detector and BoT-SORT multi-object tracker.

    Wraps the Ultralytics YOLO API to provide a clean, single-method
    interface for tracking children across video frames.

    Args:
        model_path: Path to the ``.pt`` checkpoint.
        conf: Minimum detection confidence threshold (0–1).
        tracker: Tracker configuration name or path accepted by
            Ultralytics (e.g. ``'botsort.yaml'``).
    """

    def __init__(
        self,
        model_path: str | Path,
        conf: float = 0.6,
        tracker: str = "botsort.yaml",
    ) -> None:
        self.model_path = Path(model_path)
        self.conf = conf
        self.tracker = tracker

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Child model checkpoint not found: {self.model_path}"
            )

        self.model = YOLO(str(self.model_path))
        logger.info("ChildModel loaded from '%s'.", self.model_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def track(self, frame) -> list[tuple[int, int, float, list[float]]]:
        """Track children in a single video frame.

        Args:
            frame: A NumPy BGR array (HWC) as returned by ``cv2.VideoCapture``.

        Returns:
            A list of tuples ``(track_id, class_idx, confidence, bbox)``
            where ``bbox`` is ``[x1, y1, x2, y2]`` in pixel coordinates.
            Returns an empty list when no children are detected.
        """
        results = self.model.track(
            frame,
            tracker=self.tracker,
            save=False,
            conf=self.conf,
            persist=True,
            verbose=False,
            classes=[_CHILD_CLASS_IDX],
        )[0]

        boxes = results.boxes.cpu()
        if not boxes.is_track:
            logger.debug("No tracked children in this frame.")
            return []

        detections: list[tuple[int, int, float, list[float]]] = []
        for tid, cl, conf, xyxy in zip(
            boxes.id.numpy(),
            boxes.cls.numpy(),
            boxes.conf.numpy(),
            boxes.xyxy.numpy(),
        ):
            detections.append(
                (int(tid), int(cl), float(conf), xyxy.tolist())
            )

        logger.debug("Tracked %d child(ren) in frame.", len(detections))
        return detections

    def get_class_names(self) -> list[str]:
        """Return all class-name labels from the loaded model.

        Returns:
            List of class names ordered by class index.
        """
        return list(self.model.names.values())
