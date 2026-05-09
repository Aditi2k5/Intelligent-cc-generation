"""
utils/logger.py
===============
Provides a consistently formatted logger for the CC pipeline.

Usage:
    from utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Processing clip: %s", path)
"""

import logging
import sys
from pathlib import Path


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Return a named logger that writes colour-coded output to stdout.

    Parameters
    ----------
    name  : module name, typically __name__
    level : one of DEBUG | INFO | WARNING | ERROR
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    fmt = (
        "%(asctime)s  %(levelname)-8s  "
        "%(name)-28s  %(message)s"
    )
    handler.setFormatter(_ColourFormatter(fmt))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


class _ColourFormatter(logging.Formatter):
    """ANSI colour coding for log levels (gracefully degrades on Windows)."""

    COLOURS = {
        "DEBUG":    "\033[36m",    # cyan
        "INFO":     "\033[32m",    # green
        "WARNING":  "\033[33m",    # yellow
        "ERROR":    "\033[31m",    # red
        "CRITICAL": "\033[35m",    # magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        colour = self.COLOURS.get(record.levelname, "")
        record.levelname = f"{colour}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_file_logger(output_dir: str, level: str = "DEBUG") -> None:
    """
    Attach a file handler to the root logger so every module's messages
    are also written to  <output_dir>/pipeline.log.
    """
    root = logging.getLogger()
    if any(isinstance(h, logging.FileHandler) for h in root.handlers):
        return  # already attached

    log_path = Path(output_dir) / "pipeline.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
    fh.setLevel(getattr(logging, level.upper(), logging.DEBUG))
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
        )
    )
    root.addHandler(fh)