"""ChildVision – Intelligent Day-Care Monitoring System.

Entry point.  Edit the paths below or pass them via environment
variables / CLI arguments in future iterations.

Usage
-----
    python main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline import SystemPipeline
from utilities.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    child_model_path    = Path("models/childModel.pt")
    fall_detect_path    = Path("models/fallDetect.pkl")
    behavior_model_path = Path("models/behavioralModel.pt")
    video_path          = Path("data/input_2.mp4")   # set to None for live camera
    output_path         = Path("outputs/output_2.mp4")  # set to None to skip saving

    logger.info("Starting ChildVision pipeline.")
    try:
        pipeline = SystemPipeline(
            child_model_path    = child_model_path,
            fall_detect_path    = fall_detect_path,
            behavior_model_path = behavior_model_path,
            video_path          = video_path,
            output_path         = output_path,
        )
        pipeline.start()
    except FileNotFoundError as exc:
        logger.error("Model or video file missing: %s", exc)
        sys.exit(1)
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        logger.info("ChildVision shut down.")


if __name__ == "__main__":
    main()
