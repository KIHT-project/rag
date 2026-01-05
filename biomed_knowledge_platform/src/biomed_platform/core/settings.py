from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def configs_dir() -> Path:
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

    def get(self, name: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
        if default is None:
            default = {}
        return self.by_name.get(name, default)


def load_settings() -> AppSettings:
    d = configs_dir()

    by_name: dict[str, dict[str, Any]] = {}

    for p in sorted(d.glob("*.yaml")):
        if p.name == "logging.yaml":
            continue
        by_name[p.stem] = _load_yaml(p)

    if not (d / "logging.yaml").exists():
        raise FileNotFoundError(f"Missing required logging config: {d / 'logging.yaml'}")

    return AppSettings(config_dir=d, by_name=by_name)
