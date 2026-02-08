from __future__ import annotations

from pathlib import Path

import pytest

from scheduler_pubmed.src.common import logging as logging_mod


def test_configure_logging_raises_for_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError):
        logging_mod.configure_logging(missing)


def test_configure_logging_raises_for_invalid_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "logging.yaml"
    config_path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError):
        logging_mod.configure_logging(config_path)


def test_configure_logging_calls_dictconfig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "logging.yaml"
    config_path.write_text("version: 1\nhandlers: {}\nroot: {level: INFO, handlers: []}\n", encoding="utf-8")

    called: dict[str, object] = {}

    def fake_dictconfig(config: dict[str, object]) -> None:
        called["config"] = config

    monkeypatch.setattr(logging_mod.logging.config, "dictConfig", fake_dictconfig)

    logging_mod.configure_logging(config_path)

    assert isinstance(called.get("config"), dict)


def test_get_logger_returns_named_logger() -> None:
    logger = logging_mod.get_logger("scheduler.tests")
    assert logger.name == "scheduler.tests"
