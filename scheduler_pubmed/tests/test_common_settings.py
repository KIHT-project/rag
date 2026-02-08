from __future__ import annotations

from pathlib import Path

import pytest

from scheduler_pubmed.src.common import settings as settings_mod


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _create_valid_config_set(base: Path) -> None:
    _write_yaml(base / "logging.yaml", "version: 1\nhandlers: {}\nroot: {level: INFO, handlers: []}\n")
    _write_yaml(base / "api.yaml", "title: test\n")
    _write_yaml(
        base / "postgres.yaml",
        "postgres:\n  host: localhost\n  postgres_port: 5432\n  postgres_user: user\n  postgres_password: pass\n  postgres_db: db\n  postgres_schema: pubmed_scheduler\n",
    )
    _write_yaml(
        base / "pubmed_scheduler.yaml",
        "enabled: true\nschedule:\n  utc_times: ['02:00']\napi:\n  base_url: http://localhost:9000\n",
    )


def test_config_dir_uses_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(settings_mod.CONFIG_DIR_ENV, str(tmp_path))
    assert settings_mod.config_dir() == tmp_path.resolve()


def test_config_dir_raises_for_missing_env_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(settings_mod.CONFIG_DIR_ENV, "/tmp/does-not-exist-scheduler-pubmed")
    with pytest.raises(FileNotFoundError):
        settings_mod.config_dir()


def test_load_yaml_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        settings_mod._load_yaml(tmp_path / "missing.yaml")


def test_load_yaml_raises_for_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    _write_yaml(path, "- one\n- two\n")

    with pytest.raises(ValueError):
        settings_mod._load_yaml(path)


def test_load_settings_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _create_valid_config_set(tmp_path)
    monkeypatch.setenv(settings_mod.CONFIG_DIR_ENV, str(tmp_path))

    loaded = settings_mod.load_settings()

    assert loaded.require_api()["title"] == "test"
    assert loaded.require_postgres()["postgres_db"] == "db"
    assert loaded.require_scheduler()["enabled"] is True
    assert loaded.logging_path == tmp_path / "logging.yaml"


def test_load_settings_raises_without_logging_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_yaml(tmp_path / "api.yaml", "title: test\n")
    _write_yaml(tmp_path / "postgres.yaml", "postgres: {}\n")
    _write_yaml(tmp_path / "pubmed_scheduler.yaml", "enabled: true\nschedule: {utc_times: []}\napi: {}\n")

    monkeypatch.setenv(settings_mod.CONFIG_DIR_ENV, str(tmp_path))

    with pytest.raises(FileNotFoundError):
        settings_mod.load_settings()


def test_validate_required_rejects_invalid_scheduler_enabled() -> None:
    invalid = settings_mod.AppSettings(
        config_dir=Path("/tmp"),
        by_name={
            "api": {},
            "postgres": {"postgres": {}},
            "pubmed_scheduler": {"enabled": "yes", "schedule": {"utc_times": []}, "api": {}},
        },
    )

    with pytest.raises(ValueError, match="boolean key enabled"):
        settings_mod._validate_required(invalid)


def test_require_postgres_rejects_missing_top_level_mapping() -> None:
    app_settings = settings_mod.AppSettings(
        config_dir=Path("/tmp"),
        by_name={"postgres": {"wrong": {}}},
    )

    with pytest.raises(ValueError, match="top level key postgres"):
        app_settings.require_postgres()
