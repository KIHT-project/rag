from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR_ENV = "PUBMED_SCHEDULER_CONFIG_DIR"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def config_dir() -> Path:
    env_dir = os.getenv(CONFIG_DIR_ENV)
    if env_dir:
        cfg_dir = Path(env_dir).expanduser().resolve()
    else:
        cfg_dir = project_root() / "config"

    if not cfg_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {cfg_dir}")
    return cfg_dir


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


@dataclass(frozen=True)
class AppSettings:
    config_dir: Path
    by_name: dict[str, dict[str, Any]]

    @property
    def logging_path(self) -> Path:
        return self.config_dir / "logging.yaml"

    def require(self, name: str) -> dict[str, Any]:
        if name not in self.by_name:
            raise KeyError(f"Missing config: {name}.yaml")
        return self.by_name[name]

    def require_api(self) -> dict[str, Any]:
        return self.require("api")

    def require_postgres(self) -> dict[str, Any]:
        raw = self.require("postgres")
        value = raw.get("postgres")
        if not isinstance(value, dict):
            raise ValueError("postgres.yaml must contain top level key postgres")
        return value

    def require_scheduler(self) -> dict[str, Any]:
        return self.require("pubmed_scheduler")


def _validate_required(settings: AppSettings) -> None:
    settings.require_api()
    settings.require_postgres()

    scheduler_cfg = settings.require_scheduler()

    enabled = scheduler_cfg.get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError("pubmed_scheduler.yaml must contain boolean key enabled")

    schedule = scheduler_cfg.get("schedule")
    if not isinstance(schedule, dict):
        raise ValueError("pubmed_scheduler.yaml must contain mapping key schedule")

    utc_times = schedule.get("utc_times")
    if not isinstance(utc_times, list):
        raise ValueError("pubmed_scheduler.yaml must contain schedule.utc_times as a list")

    api = scheduler_cfg.get("api")
    if not isinstance(api, dict):
        raise ValueError("pubmed_scheduler.yaml must contain mapping key api")


def load_settings() -> AppSettings:
    cfg_dir = config_dir()

    by_name: dict[str, dict[str, Any]] = {}
    for path in sorted(cfg_dir.glob("*.yaml")):
        if path.name == "logging.yaml":
            continue
        by_name[path.stem] = _load_yaml(path)

    if not (cfg_dir / "logging.yaml").exists():
        raise FileNotFoundError(f"Logging config file not found: {cfg_dir / 'logging.yaml'}")

    settings = AppSettings(config_dir=cfg_dir, by_name=by_name)
    _validate_required(settings)
    return settings
