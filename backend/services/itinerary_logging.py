"""行程工作流的本地文件日志。"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Lock

LOG_PATH = Path(__file__).resolve().parents[2] / "logs" / "itinerary.log"
_lock = Lock()


def configure_itinerary_logging(logger: logging.Logger) -> None:
    with _lock:
        if any(
            isinstance(handler, RotatingFileHandler)
            and Path(handler.baseFilename) == LOG_PATH
            for handler in logger.handlers
        ):
            return
        try:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(
                LOG_PATH, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
            )
        except OSError as exc:
            logger.warning("无法创建行程日志文件（%s）", type(exc).__name__)
            return
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
