from __future__ import annotations

import json
import logging
from logging import Logger
from pathlib import Path
from datetime import datetime, timezone

class JsonlFileHandler(logging.Handler):
    def __init__(self, path: Path):
        super().__init__()
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, record: logging.LogRecord) -> None:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "msg": record.getMessage(),
            "name": record.name,
        }
        extra = getattr(record, "extra", None)
        if isinstance(extra, dict):
            payload.update(extra)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

def setup_logging(jsonl_path: Path, console: bool = True) -> Logger:
    logger = logging.getLogger("tsgen")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    logger.addHandler(JsonlFileHandler(jsonl_path))

    if console:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter("[%(levelname)s] %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    return logger
