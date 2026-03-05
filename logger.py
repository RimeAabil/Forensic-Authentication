"""
utils/logger.py
Centralized logging setup for the entire project.
"""

import logging
import os
from datetime import datetime
import colorlog


def get_logger(name: str, log_dir: str = "logs", level: str = "INFO") -> logging.Logger:
    """
    Returns a logger with both console (colored) and file handlers.

    Args:
        name:    Logger name, typically __name__ of the calling module.
        log_dir: Directory where log files are saved.
        level:   Logging level string (DEBUG, INFO, WARNING, ERROR).

    Returns:
        Configured logging.Logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on re-import
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # ── Console handler (colored) ──────────────────────────────────────────────
    console_fmt = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s [%(levelname)-8s] %(name)s: %(message)s%(reset)s",
        datefmt="%H:%M:%S",
        log_colors={
            "DEBUG":    "cyan",
            "INFO":     "green",
            "WARNING":  "yellow",
            "ERROR":    "red",
            "CRITICAL": "bold_red",
        },
    )
    ch = logging.StreamHandler()
    ch.setFormatter(console_fmt)
    logger.addHandler(ch)

    # ── File handler (plain text) ──────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file  = os.path.join(log_dir, f"{timestamp}_{name.replace('.', '_')}.log")
    file_fmt  = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(file_fmt)
    logger.addHandler(fh)

    return logger
