"""Shared image-processing helpers for the ChildVision pipeline."""

from __future__ import annotations

import cv2
import numpy as np
from cv2.typing import MatLike

from utilities.logger import get_logger

logger = get_logger(__name__)

# Colour palette (BGR) used across the pipeline
COLOUR_CHILD    = (112,   0,  94)   # purple
COLOUR_FALL     = ( 24,  31, 230)   # red-blue
COLOUR_BEHAVIOR = (244,  31,  20)   # vivid red

_FONT_FACE  = cv2.FONT_HERSHEY_PLAIN
_FONT_SCALE = 1.0
_FONT_THICK = 2


class Utils:
    """Static image helpers with input validation and type annotations."""

    @staticmethod
    def crop_image(image: MatLike, bbox: list[float]) -> MatLike:
        """Crop an OpenCV image to a bounding box, clamped to image bounds.

        Args:
            image: Source BGR image as a NumPy array.
            bbox: Bounding box ``[x1, y1, x2, y2]`` (floats or ints).

        Returns:
            Cropped image region.  Returns the full frame when the
            bounding box is degenerate after clamping.
        """
        if image is None or image.size == 0:
            logger.warning("crop_image received an empty frame; returning as-is.")
            return image

        h, w = image.shape[:2]
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            logger.debug("Degenerate crop bbox %s – returning full frame.", bbox)
            return image

        return image[y1:y2, x1:x2]

    @staticmethod
    def draw_bounding_box(
        image: MatLike,
        label: str,
        bbox: list[float],
        color: tuple[int, int, int],
    ) -> MatLike:
        """Draw a filled-label bounding box on an image in-place.

        Args:
            image: BGR image to annotate.
            label: Text shown above the box.
            bbox: ``[x1, y1, x2, y2]`` coordinates.
            color: BGR colour tuple for the box and label background.

        Returns:
            The annotated image (same object, modified in-place).
        """
        if image is None or image.size == 0:
            logger.warning("draw_bounding_box received an empty frame.")
            return image

        x1, y1, x2, y2 = (int(v) for v in bbox)

        # outer rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=2)

        # label background
        (tw, th), baseline = cv2.getTextSize(
            label, _FONT_FACE, _FONT_SCALE, _FONT_THICK
        )
        lbl_x2 = x1 + tw + baseline
        lbl_y2 = y1 + th + baseline
        cv2.rectangle(image, (x1, y1), (lbl_x2, lbl_y2), color=color, thickness=-1)

        # label text (white)
        cv2.putText(
            image,
            label,
            (x1 + baseline // 2, lbl_y2 - baseline // 2),
            fontFace=_FONT_FACE,
            fontScale=_FONT_SCALE,
            color=(255, 255, 255),
            thickness=_FONT_THICK,
        )
        return image

    @staticmethod
    def resize_frame(frame: MatLike, max_dim: int = 1280) -> MatLike:
        """Downscale a frame so its longest edge is at most *max_dim* pixels.

        No upscaling is applied.  Aspect ratio is always preserved.

        Args:
            frame: Source BGR image.
            max_dim: Maximum allowed length of the longest edge (pixels).

        Returns:
            Resized frame, or the original if already within limits.
        """
        h, w = frame.shape[:2]
        scale = min(max_dim / max(h, w, 1), 1.0)
        if scale < 1.0:
            frame = cv2.resize(
                frame,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        return frame
