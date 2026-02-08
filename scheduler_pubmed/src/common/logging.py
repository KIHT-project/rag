from __future__ import annotations

import logging
import logging.config
from pathlib import Path
from typing import Any

import yaml


def configure_logging(config_path: Path) -> None:
    if not config_path.exists():
        raise FileNotFoundError(f"Logging configuration file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config: dict[str, Any] = raw if isinstance(raw, dict) else {}
    if not config:
        raise ValueError(f"Logging config is empty or invalid: {config_path}")

    logging.config.dictConfig(config)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
