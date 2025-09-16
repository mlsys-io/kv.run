import logging
from logging.handlers import RotatingFileHandler

def get_logger(
    name: str = "orchestrator",
    log_file: str = "worker.log",
    max_bytes: int = 5_242_880,
    backup_count: int = 5,
    level: str = "INFO",
) -> logging.Logger:
    """Return a configured logger with a rotating file handler and console output."""
    logger = logging.getLogger(name)
    if logger.handlers:
        # Already configured
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # File handler (rotating)
    fh = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.addHandler(ch)

    return logger
