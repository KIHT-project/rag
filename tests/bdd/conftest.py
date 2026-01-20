from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

_REPO_ROOT = Path(__file__).resolve().parents[2]
_APP_SRC = _REPO_ROOT / "biomed_knowledge_platform" / "src"
if str(_APP_SRC) not in sys.path:
    sys.path.insert(0, str(_APP_SRC))

from biomed_platform.api.app import create_app  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def bdd_config_env() -> None:
    cfg_dir = Path(__file__).resolve().parent / "config"
    os.environ["BIOMED_CONFIG_DIR"] = str(cfg_dir)


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


@pytest.fixture
def ctx() -> dict:
    return {}

@pytest.fixture(autouse=True)
def cleanup_qdrant_via_script() -> None:
    script = Path(__file__).resolve().parent / "helpers" / "clear_qdrant.py"
    assert script.exists(), f"Missing clear_qdrant script at {script}"
    subprocess.run([sys.executable, str(script)], check=True)
    yield
    subprocess.run([sys.executable, str(script)], check=True)

@pytest.fixture(scope="session", autouse=True)
def bdd_config_env() -> None:
    cfg_dir = Path(__file__).resolve().parent / "config"
    os.environ["BIOMED_CONFIG_DIR"] = str(cfg_dir)

    pg_yaml = cfg_dir / "postgres.yaml"
    data = yaml.safe_load(pg_yaml.read_text(encoding="utf-8"))

    os.environ["BDD_POSTGRES_DSN"] = (
        "postgresql://user:password@localhost:5433/test_biomed_knowledge_platform?sslmode=disable"
    )

@pytest.fixture(autouse=True)
def cleanup_postgres_via_script() -> None:
    yield
    script = Path(__file__).resolve().parent / "helpers" / "clear_postgres.py"
    subprocess.run([sys.executable, str(script)], check=True)

