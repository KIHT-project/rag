from __future__ import annotations

import os
import socket
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
    os.environ["BDD_POSTGRES_DSN"] = (
        "postgresql://user:password@localhost:5433/test_biomed_knowledge_platform?sslmode=disable"
    )


def _run_script_best_effort(script: Path) -> None:
    if not script.exists():
        return
    subprocess.run([sys.executable, str(script)], check=False)


def _can_connect(host: str, port: int, timeout: float = 0.35) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


@pytest.fixture(autouse=True)
def cleanup_qdrant_via_script() -> None:
    script = Path(__file__).resolve().parent / "helpers" / "clear_qdrant.py"
    _run_script_best_effort(script)
    yield
    _run_script_best_effort(script)


@pytest.fixture(autouse=True)
def cleanup_postgres_via_script() -> None:
    script = Path(__file__).resolve().parent / "helpers" / "clear_postgres.py"
    _run_script_best_effort(script)
    yield
    _run_script_best_effort(script)


@pytest.fixture
def app():
    if not _can_connect("localhost", 5433) or not _can_connect("localhost", 6335):
        pytest.skip("BDD dependencies unavailable: Postgres/Qdrant containers are not reachable")
    return create_app()


@pytest.fixture
def client(app):
    try:
        with TestClient(app) as c:
            yield c
    except RuntimeError as exc:
        msg = str(exc)
        if "Postgres not reachable" in msg:
            pytest.skip(f"BDD dependencies unavailable: {msg}")
        raise


@pytest.fixture
def ctx() -> dict:
    return {}
