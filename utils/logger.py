import logging
import sys
from pathlib import Path


def get_logger(name: str, level: str = "INFO") -> logging.Logger:

    logger = logging.getLogger(name)
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