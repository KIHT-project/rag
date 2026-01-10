from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
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
    yield
    script = Path(__file__).resolve().parents[1] / "helpers" / "clear_qdrant.py"
    subprocess.run([sys.executable, str(script)], check=False)
