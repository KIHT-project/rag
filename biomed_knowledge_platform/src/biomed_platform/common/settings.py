from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def configs_dir() -> Path:
    env_dir = os.getenv("BIOMED_CONFIG_DIR")
    if env_dir:
        d = Path(env_dir).expanduser().resolve()
    else:
        d = project_root() / "configs"

    if not d.exists():
        raise FileNotFoundError(f"Configs directory not found: {d}")
    return d


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

    def require_rag(self) -> dict[str, Any]:
        return self.require("rag")

    def require_qdrant(self) -> dict[str, Any]:
        return self.require("qdrant")

    def require_llm(self) -> dict[str, Any]:
        return self.require("llm")


def _validate_required_configs(settings: AppSettings) -> None:
    settings.require_api()
    settings.require_rag()
    settings.require_qdrant()
    settings.require_llm()


def load_settings() -> AppSettings:
    d = configs_dir()

    by_name: dict[str, dict[str, Any]] = {}

    for p in sorted(d.glob("*.yaml")):
        if p.name == "logging.yaml":
            continue
        by_name[p.stem] = _load_yaml(p)

    if not (d / "logging.yaml").exists():
        raise FileNotFoundError(f"Missing required logging config: {d / 'logging.yaml'}")

    settings = AppSettings(config_dir=d, by_name=by_name)
    _validate_required_configs(settings)
    return settings
