"""GolfPose logging utilities.

Provides:
  - ``Logger`` — legacy stdout/file tee (kept for backward compat).
  - ``setup_logger()`` — structured Python ``logging`` setup (rank-aware).
  - ``JsonFormatter`` — machine-parseable JSON log lines.
"""

import json
import logging
import os
import sys
from datetime import datetime


# ---------------------------------------------------------------------------
# Legacy Logger (kept for backward-compatibility with golfpose_3d.py)
# ---------------------------------------------------------------------------

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


# ---------------------------------------------------------------------------
# JSON formatter for structured logging
# ---------------------------------------------------------------------------

class JsonFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, ensure_ascii=False)


# ---------------------------------------------------------------------------
# setup_logger — structured, rank-aware logging
# ---------------------------------------------------------------------------

def setup_logger(
    name: str = "golfpose",
    log_file: str = None,
    level: int = logging.INFO,
    rank: int = 0,
    json_format: bool = False,
):
    """Create (or retrieve) a named logger with console + optional file output.

    Parameters
    ----------
    name : str
        Logger name (typically ``'golfpose'``).
    log_file : str or None
        If given, logs are also written to this file.
    level : int
        Logging level for rank-0.  Non-zero ranks are silenced to WARNING.
    rank : int
        Current distributed rank; non-zero ranks get reduced verbosity.
    json_format : bool
        If True the file handler uses ``JsonFormatter``.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    effective_level = level if rank == 0 else logging.WARNING
    logger.setLevel(effective_level)
    logger.propagate = False

    # Console handler (always human-readable)
    console_fmt = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(effective_level)
    ch.setFormatter(console_fmt)
    logger.addHandler(ch)

    # File handler (optional)
    if log_file is not None and rank == 0:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a")
        fh.setLevel(level)
        if json_format:
            fh.setFormatter(JsonFormatter())
        else:
            fh.setFormatter(console_fmt)
        logger.addHandler(fh)

    return logger
