from __future__ import annotations

import logging
import logging.config
from pathlib import Path

import yaml

from biomed_platform.common.middleware.trace import request_id_ctx


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get() or "none"
        return True


def configure_logging(logging_yaml_path: str | Path) -> None:
    path = Path(logging_yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Logging config not found: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Logging config must be a YAML mapping: {path}")

    logging.config.dictConfig(data)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
