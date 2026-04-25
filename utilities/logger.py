"""Centralised logging configuration for the ChildVision pipeline.

Creates a rotating file handler alongside a console handler so that
logs are both visible during a run and persisted for later review.
Import ``get_logger`` in every module instead of the bare
``logging`` module to ensure consistent formatting.
"""

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_LOG_DIR = os.path.join("logs", datetime.now().strftime("%Y_%m_%d"))
os.makedirs(_LOG_DIR, exist_ok=True)
_LOG_FILE = os.path.join(_LOG_DIR, "childvision.log")


def _build_root_logger() -> logging.Logger:
    root = logging.getLogger("childvision")
    root.setLevel(logging.DEBUG)

    if root.handlers:          # guard against double-initialisation
        return root

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # rotating file handler (10 MB x 5 backups)
    fh = RotatingFileHandler(
        _LOG_FILE,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # console handler (INFO and above only)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    root.addHandler(fh)
    root.addHandler(ch)
    return root


_root_logger = _build_root_logger()


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the 'childvision' hierarchy.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    return _root_logger.getChild(name)
