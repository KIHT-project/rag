import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExportTarget:
    directory: Path
    filename: str

    @property
    def path(self) -> Path:
        return self.directory / self.filename


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise RuntimeError(f"Missing env var {name}")
    return value.strip()


def load_export_target(*, project_root: str) -> ExportTarget:
    """
    Export target is configured only via environment.

    Required
      LABEL_STUDIO_EXPORT_FILENAME

    Optional
      LABEL_STUDIO_EXPORT_DIR, defaults to data/label_studio
    """
    filename = _require_env("LABEL_STUDIO_EXPORT_FILENAME")
    dir_raw = os.getenv("LABEL_STUDIO_EXPORT_DIR", "data/label_studio").strip()

    directory = (Path(project_root) / dir_raw).resolve()

    return ExportTarget(directory=directory, filename=filename)
