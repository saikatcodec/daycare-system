"""End-to-end ChildVision inference pipeline.

Ties together child detection/tracking, fall detection, and behavioral
analysis into a single real-time processing loop that reads from a video
file or a live camera feed.

Inference strategy
------------------
- **Child detection / tracking** : full frame → YOLOv12x + BoT-SORT.
- **Fall detection**             : per-child crop → YOLOv11-pose + SVM.
  A fall is a single-person event; the crop isolates the subject and
  removes background clutter, improving pose accuracy.
- **Behavioral analysis**        : full frame → YOLOv12l (cry / fight).
  Fight detection requires at least two children to be visible
  simultaneously, so it must run on the full frame where all children
  are present.  Running it on a single-child crop would mean the second
  child is never in the image, making fight detection impossible.
  Cry detection also benefits from full-frame context because the model
  was trained on scenes, not tight crops.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path

import cv2
from cv2.typing import MatLike

from src.childClassifier import ChildModel
from src.fallDetection import FallDetect, FallResult
from src.behaviorAnalysis import BehaviorAnalysisModel, BehaviorResult
from utilities.utils import Utils, COLOUR_CHILD, COLOUR_FALL, COLOUR_BEHAVIOR
from utilities.logger import get_logger

logger = get_logger(__name__)

_MAX_WORKERS = 2   # fall detection thread + behavior analysis thread


def _to_frame_coords(
    bbox: list[float],
    origin_x: int,
    origin_y: int,
) -> list[float]:
    """Translate a crop-space bounding box to full-frame coordinates.

    Fall detection runs on a cropped child image, so its output boxes are
    relative to the crop origin.  Add the origin offset before drawing.

    Args:
        bbox: ``[x1, y1, x2, y2]`` in crop-image pixel coordinates.
        origin_x: x pixel of the crop's top-left corner in the full frame.
        origin_y: y pixel of the crop's top-left corner in the full frame.

    Returns:
        ``[x1, y1, x2, y2]`` in full-frame pixel coordinates.
    """
    x1, y1, x2, y2 = bbox
    return [x1 + origin_x, y1 + origin_y, x2 + origin_x, y2 + origin_y]


class SystemPipeline:
    """Orchestrate the full ChildVision monitoring pipeline.

    Args:
        child_model_path: Path to the YOLOv12x child detection ``.pt`` file.
        fall_detect_path: Path to the pickled SVM fall-detection model.
        behavior_model_path: Path to the YOLOv12l behavioral analysis ``.pt`` file.
        video_path: Path to the input video file.  Pass ``None`` to use
            the default system camera (device index 0).
        output_path: Destination path for the annotated output video.
            Pass ``None`` to disable saving.
    """

    def __init__(
        self,
        child_model_path: str | Path,
        fall_detect_path: str | Path,
        behavior_model_path: str | Path,
        video_path: str | Path | None = None,
        output_path: str | Path | None = None,
    ) -> None:
        self.output_path = Path(output_path) if output_path else None
        self._executor = ThreadPoolExecutor(max_workers=_MAX_WORKERS)

        # ── video source ───────────────────────────────────────────────
        source = str(video_path) if video_path else 0
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise ValueError(
                f"Could not open video source: {source!r}. "
                "Check the file path or camera index."
            )
        logger.info("Video source opened: %s.", source)

        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._fps: float = fps if fps and fps > 0 else 20.0
        self._wait_ms: int = max(1, int(1000 / self._fps))

        # ── optional video writer ──────────────────────────────────────
        self._writer: cv2.VideoWriter | None = None
        if self.output_path:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._writer = cv2.VideoWriter(
                str(self.output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self._fps,
                (w, h),
            )
            logger.info("Video writer initialised → %s.", self.output_path)

        # ── sub-models ─────────────────────────────────────────────────
        self._child_model    = ChildModel(child_model_path, conf=0.8)
        self._fall_detector  = FallDetect(fall_detect_path)
        self._behavior_model = BehaviorAnalysisModel(behavior_model_path, conf=0.8)
        logger.info("All sub-models loaded successfully.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Run the processing loop until the video ends or 'q' is pressed."""
        logger.info("Pipeline started. Press 'q' to quit.")
        try:
            while True:
                ret, frame = self._cap.read()
                if not ret:
                    logger.info("Video stream exhausted – stopping.")
                    break

                annotated = self._process_frame(frame)

                if self._writer is not None:
                    self._writer.write(annotated)

                cv2.imshow("ChildVision", annotated)
                if cv2.waitKey(self._wait_ms) & 0xFF == ord("q"):
                    logger.info("Quit key pressed – stopping.")
                    break

        except Exception:
            logger.exception("Unexpected error in pipeline loop.")
            raise
        finally:
            self._release()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_frame(self, frame: MatLike) -> MatLike:
        """Detect, track, analyse, and annotate a single frame.

        Processing order
        ----------------
        1. Detect and track all children in the full frame.
        2. Run behavioral analysis **once on the full frame** in a
           background thread — fight requires ≥2 children visible.
        3. For each tracked child, run fall detection on that child's
           individual crop in parallel with step 2.
        4. Translate all crop-space bounding boxes to full-frame coords.
        5. Draw child, fall, and behavior annotations on the full frame.

        Args:
            frame: Raw BGR frame from the video source.

        Returns:
            Annotated BGR frame ready for display or writing.
        """
        children = self._child_model.track(frame)
        if not children:
            return frame

        # ── STEP 1 : launch behavioral analysis on the FULL FRAME ─────
        # Fight detection needs both children in the same image.
        # Cry detection also benefits from full-scene context.
        # This runs once per frame regardless of how many children exist.
        behavior_future: Future[list[BehaviorResult]] = self._executor.submit(
            self._behavior_model.detect, frame
        )

        # ── STEP 2 : launch fall detection on each child's CROP ───────
        # Fall is a single-person event; cropping removes irrelevant
        # background and improves pose estimation accuracy.
        # Each (track_id, origin, fall_future) tuple is collected so
        # results can be drawn after all futures resolve.
        fall_jobs: list[tuple[int, float, int, int, Future[list[FallResult]]]] = []

        for track_id, _cls, conf, bbox in children:
            origin_x = int(round(bbox[0]))
            origin_y = int(round(bbox[1]))
            crop = Utils.crop_image(frame, bbox)

            fall_future: Future[list[FallResult]] = self._executor.submit(
                self._fall_detector.detect, crop
            )
            fall_jobs.append((track_id, conf, origin_x, origin_y, fall_future))

        # ── STEP 3 : draw child boxes (no waiting needed) ─────────────
        for track_id, conf, _ox, _oy, _ff in fall_jobs:
            # Retrieve bbox for this child from the original children list
            bbox = next(b for tid, _c, _co, b in children if tid == track_id)
            frame = Utils.draw_bounding_box(
                frame,
                label=f"Child {track_id}  {conf:.2f}",
                bbox=bbox,
                color=COLOUR_CHILD,
            )

        # ── STEP 4 : draw fall annotations ────────────────────────────
        # Translate crop-space boxes → full-frame before drawing.
        for track_id, _conf, origin_x, origin_y, fall_future in fall_jobs:
            for fall in fall_future.result():
                if "not-fallen" not in fall.label.lower():
                    frame_bbox = _to_frame_coords(fall.bbox, origin_x, origin_y)
                    frame = Utils.draw_bounding_box(
                        frame,
                        label=f"{fall.label}  {fall.confidence:.2f}",
                        bbox=frame_bbox,
                        color=COLOUR_FALL,
                    )
                    logger.warning(
                        "FALL DETECTED – track_id=%d, bbox=%s.",
                        track_id, frame_bbox,
                    )

        # ── STEP 5 : draw behavior annotations ────────────────────────
        # These bboxes are already in full-frame space — no translation.
        for behavior in behavior_future.result():
            frame = Utils.draw_bounding_box(
                frame,
                label=f"{behavior.label}  {behavior.confidence:.2f}",
                bbox=behavior.bbox,
                color=COLOUR_BEHAVIOR,
            )
            logger.warning(
                "BEHAVIOR DETECTED – label=%s, conf=%.2f, bbox=%s.",
                behavior.label, behavior.confidence, behavior.bbox,
            )

        return frame

    def _release(self) -> None:
        """Release all held resources cleanly."""
        self._executor.shutdown(wait=False)
        if self._writer is not None:
            self._writer.release()
        self._cap.release()
        cv2.destroyAllWindows()
        logger.info("Pipeline resources released.")