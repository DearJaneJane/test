"""Runtime logging setup for the enterprise QA skill."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_LOGGER_NAME = "enterprise_qa.audit"


def setup_logging(log_path: str | Path = "logs/enterprise-qa.jsonl") -> None:
    """Configure console-safe structured JSONL logging."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not any(isinstance(handler, logging.FileHandler) and handler.baseFilename == str(path.resolve()) for handler in logger.handlers):
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)


def log_event(event: str, **payload: Any) -> None:
    """Write one audit event without logging secrets."""
    logger = logging.getLogger(_LOGGER_NAME)
    if not logger.handlers:
        return

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **_sanitize(payload),
    }
    logger.info(json.dumps(record, ensure_ascii=False, default=str))


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: ("***" if _looks_secret(key) else _sanitize(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    return value


def _looks_secret(key: str) -> bool:
    lowered = key.lower()
    return any(marker in lowered for marker in ("key", "token", "secret", "password"))
