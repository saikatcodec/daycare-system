"""Fall detection via YOLOv11-pose keypoints and an SVM classifier."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import NamedTuple

import numpy as np
from ultralytics import YOLO

from utilities.logger import get_logger

logger = get_logger(__name__)

# COCO keypoint indices used in feature extraction
_KP_LEFT_SHOULDER  = 5
_KP_RIGHT_SHOULDER = 6
_KP_LEFT_HIP       = 11
_KP_RIGHT_HIP      = 12

# Fallback pose-model path (can be overridden via constructor)
_DEFAULT_POSE_MODEL = "models/yolo11m-pose.pt"
_DEFAULT_SCALER     = "models/scaler_v1.2.1.pkl"


class FallResult(NamedTuple):
    """Structured result for a single detected person."""
    label: str          # 'fallen' | 'not-fallen'
    bbox: list[int]     # [x1, y1, x2, y2] in pixel coordinates
    confidence: float   # placeholder – SVM hard predictions = 1.0


class FallDetect:
    """Fall detector combining YOLOv11-pose estimation with an SVM classifier.

    Pipeline
    --------
    1. Run pose inference → 17 COCO keypoints + bounding boxes.
    2. Normalize keypoints relative to the nose landmark.
    3. Compute a 4-D feature vector (torso angle, torso ratio,
       bbox ratio, mean keypoint y-offset).
    4. Scale features with the fitted scaler.
    5. Classify with the SVM → ``'fallen'`` or ``'not-fallen'``.

    Args:
        svm_path: Path to a pickled scikit-learn SVM model.
        pose_model_path: Path to the YOLOv11-pose ``.pt`` checkpoint.
        scaler_path: Path to the pickled ``StandardScaler``.
        pose_conf: Keypoint confidence threshold for pose inference.
    """

    def __init__(
        self,
        svm_path: str | Path,
        pose_model_path: str | Path = _DEFAULT_POSE_MODEL,
        scaler_path: str | Path = _DEFAULT_SCALER,
        pose_conf: float = 0.9,
    ) -> None:
        svm_path       = Path(svm_path)
        pose_model_path = Path(pose_model_path)
        scaler_path    = Path(scaler_path)

        for p in (svm_path, pose_model_path, scaler_path):
            if not p.exists():
                raise FileNotFoundError(f"Required file not found: {p}")

        self.pose_conf = pose_conf
        self._pose_model = YOLO(str(pose_model_path))

        with scaler_path.open("rb") as f:
            self._scaler = pickle.load(f)
        with svm_path.open("rb") as f:
            self._svm = pickle.load(f)

        logger.info("FallDetect loaded (pose=%s, svm=%s).", pose_model_path, svm_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image) -> list[FallResult]:
        """Detect fall events for every person visible in *image*.

        Args:
            image: A NumPy BGR array or any source accepted by
                ``YOLO.predict`` (file path, PIL Image, etc.).

        Returns:
            One :class:`FallResult` per detected person.
            Returns an empty list if no persons are found.
        """
        keypoints_batch, bboxes = self._extract_keypoints(image)
        results: list[FallResult] = []

        for keypoints, bbox in zip(keypoints_batch, bboxes):
            features = self._build_features(keypoints, bbox)
            if features is None:
                logger.debug("Skipping person – insufficient keypoint data.")
                continue

            scaled = self._scaler.transform(features)
            prediction = self._svm.predict(scaled)[0]
            label = "fallen" if prediction == 1 else "not-fallen"
            results.append(FallResult(label=label, bbox=bbox.tolist(), confidence=1.0))
            logger.debug("Person bbox=%s → %s.", bbox.tolist(), label)

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_keypoints(
        self, image
    ) -> tuple[np.ndarray, np.ndarray | list]:
        """Run YOLOv11-pose inference and return raw keypoints + boxes."""
        results = self._pose_model.predict(
            image,
            conf=self.pose_conf,
            imgsz=640,
            verbose=False,
        )
        keypoints = results[0].keypoints.xy.cpu().numpy()
        bboxes    = (
            results[0].boxes.xyxy.cpu().numpy().astype(int)
            if len(keypoints) > 0
            else []
        )
        return keypoints, bboxes

    @staticmethod
    def _nose_relative(point: np.ndarray, nose: np.ndarray) -> np.ndarray:
        """Translate *point* to a coordinate system centred on *nose*."""
        return point - nose

    def _build_features(
        self,
        keypoints: np.ndarray,
        bbox: np.ndarray,
    ) -> np.ndarray | None:
        """Construct the 4-D feature vector for one detected person.

        Returns ``None`` if the keypoints array is too small to index.
        """
        if len(keypoints) <= _KP_RIGHT_HIP:
            return None

        nose = keypoints[0]
        kp_rel = np.array(
            [self._nose_relative(kp, nose) for kp in keypoints]
        )  # shape: (17, 2)

        ls = kp_rel[_KP_LEFT_SHOULDER]
        rs = kp_rel[_KP_RIGHT_SHOULDER]
        lh = kp_rel[_KP_LEFT_HIP]
        rh = kp_rel[_KP_RIGHT_HIP]

        angle, torso_ratio = self._torso_orientation(ls, rs, lh, rh)
        bbox_ratio         = self._bbox_ratio(bbox[:2], bbox[2:])
        mean_y             = float(kp_rel[:, 1].mean())

        return np.array([angle, torso_ratio, bbox_ratio, mean_y]).reshape(1, -1)

    @staticmethod
    def _torso_orientation(
        ls: np.ndarray,
        rs: np.ndarray,
        lh: np.ndarray,
        rh: np.ndarray,
    ) -> tuple[float, float]:
        """Compute torso angle (degrees) and shoulder-to-hip aspect ratio."""
        shoulder_center = (ls + rs) / 2
        hip_center      = (lh + rh) / 2
        dx = hip_center[0] - shoulder_center[0]
        dy = hip_center[1] - shoulder_center[1]
        angle  = float(np.degrees(np.arctan2(dy, dx)))
        width  = abs(float(ls[0] - rh[0]))
        height = abs(float(ls[1] - rh[1]))
        ratio  = height / (width + 1e-6)
        return angle, ratio

    @staticmethod
    def _bbox_ratio(p1: np.ndarray, p2: np.ndarray) -> float:
        """Return the height-to-width ratio of a bounding box."""
        width  = abs(float(p1[0] - p2[0]))
        height = abs(float(p1[1] - p2[1]))
        return height / (width + 1e-6)
