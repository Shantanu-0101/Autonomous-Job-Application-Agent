import logging
import sys

from app.config import settings


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("job_agent")
    logger.setLevel(settings.log_level)
    # stderr keeps stdout clean for CLI/JSON output; containers capture both.
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


logger = setup_logging()
