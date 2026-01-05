from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

import biomed_platform.core.logging as log_mod


class TestLoggingModuleBDD:
    def test_given_missing_logging_yaml_when_configure_logging_then_raises_file_not_found(
        self, tmp_path: Path
    ) -> None:
        missing = tmp_path / "missing.yaml"

        with pytest.raises(FileNotFoundError, match="Logging config not found"):
            log_mod.configure_logging(missing)

    def test_given_yaml_list_when_configure_logging_then_raises_value_error(
        self, tmp_path: Path
    ) -> None:
        cfg_path = tmp_path / "logging.yaml"
        cfg_path.write_text("- a\n- b\n- c\n", encoding="utf-8")

        with pytest.raises(ValueError, match="must be a YAML mapping"):
            log_mod.configure_logging(cfg_path)

    def test_given_valid_yaml_mapping_when_configure_logging_then_calls_dictconfig(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg_path = tmp_path / "logging.yaml"
        cfg = {
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

        log_mod.configure_logging(cfg_path)

        assert "cfg" in captured
        assert captured["cfg"]["version"] == 1
        assert captured["cfg"]["root"]["level"] == "INFO"

    def test_given_logger_name_when_get_logger_then_returns_named_logger(self) -> None:
        logger = log_mod.get_logger("unit.test")
        assert logger.name == "unit.test"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
