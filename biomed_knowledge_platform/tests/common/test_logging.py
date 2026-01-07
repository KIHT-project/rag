from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
import yaml

import biomed_platform.common.logging as log_mod
from biomed_platform.common.middleware.trace import request_id_ctx


def _make_record(*, logger_name: str = "unit.test") -> logging.LogRecord:
    return logging.LogRecord(
        name=logger_name,
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )


class TestLoggingModuleUnit:
    def test_given_missing_yaml_file_when_configure_logging_then_raises_file_not_found(self, tmp_path: Path) -> None:
        # Given
        missing = tmp_path / "missing.yaml"

        # When, Then
        with pytest.raises(FileNotFoundError, match=r"Logging config not found"):
            log_mod.configure_logging(missing)

    def test_given_yaml_is_not_mapping_when_configure_logging_then_raises_value_error(self, tmp_path: Path) -> None:
        # Given
        cfg_path = tmp_path / "logging.yaml"
        cfg_path.write_text(yaml.safe_dump(["a", "b"]), encoding="utf-8")

        # When, Then
        with pytest.raises(ValueError, match=r"must be a YAML mapping"):
            log_mod.configure_logging(cfg_path)

    def test_given_valid_yaml_mapping_when_configure_logging_then_calls_dict_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Given
        cfg_path = tmp_path / "logging.yaml"
        cfg: dict[str, Any] = {
            "version": 1,
            "disable_existing_loggers": False,
            "handlers": {},
            "root": {"level": "INFO", "handlers": []},
        }
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        captured: dict[str, Any] = {}

        def fake_dict_config(passed: dict[str, Any]) -> None:
            captured["cfg"] = passed

        monkeypatch.setattr(log_mod.logging.config, "dictConfig", fake_dict_config)

        # When
        log_mod.configure_logging(cfg_path)

        # Then
        assert "cfg" in captured
        assert captured["cfg"]["version"] == 1
        assert captured["cfg"]["root"]["level"] == "INFO"

    def test_given_no_request_id_in_context_when_filter_runs_then_sets_none(self) -> None:
        # Given
        f = log_mod.RequestIdFilter()
        record = _make_record()

        # When
        ok = f.filter(record)

        # Then
        assert ok is True
        assert getattr(record, "request_id") == "none"

    def test_given_request_id_in_context_when_filter_runs_then_sets_that_value(self) -> None:
        # Given
        f = log_mod.RequestIdFilter()
        record = _make_record()

        token = request_id_ctx.set("rid_123")
        try:
            # When
            ok = f.filter(record)
        finally:
            request_id_ctx.reset(token)

        # Then
        assert ok is True
        assert getattr(record, "request_id") == "rid_123"

    def test_given_empty_string_in_context_when_filter_runs_then_sets_none(self) -> None:
        # Given
        f = log_mod.RequestIdFilter()
        record = _make_record()

        token = request_id_ctx.set("")
        try:
            # When
            ok = f.filter(record)
        finally:
            request_id_ctx.reset(token)

        # Then
        assert ok is True
        assert getattr(record, "request_id") == "none"

    def test_given_logger_name_when_get_logger_then_returns_named_logger(self) -> None:
        # Given
        name = "biomed.test.logger"

        # When
        logger = log_mod.get_logger(name)

        # Then
        assert isinstance(logger, logging.Logger)
        assert logger.name == name


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
