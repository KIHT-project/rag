from __future__ import annotations

import importlib
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BDD_ROOT = Path(__file__).resolve().parent
_CORE_APP_SRC = _REPO_ROOT / "biomed_knowledge_platform" / "src"

# Make both app import styles available for BDD execution.
for path in (str(_REPO_ROOT), str(_CORE_APP_SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

_CORE_CONFIG_DIR = _BDD_ROOT / "config"
_SCHEDULER_CONFIG_DIR = _BDD_ROOT / "config_scheduler"

_CORE_POSTGRES_DSN = (
    "postgresql://user:password@localhost:5433/test_biomed_knowledge_platform?sslmode=disable"
)
_SCHEDULER_POSTGRES_DSN = "postgresql://user:password@localhost:5434/test_pubmed_scheduler?sslmode=disable"
_HTTP_READY_ERRORS: dict[str, str] = {}


def _target_for_node(node: pytest.Node) -> str:
    if node.get_closest_marker("scheduler") is not None:
        return "scheduler"
    return "core"


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


def _http_mode_enabled() -> bool:
    raw = (os.getenv("BDD_HTTP_MODE") or "").strip().lower()
    return raw in {"1", "true", "yes"}


def _wait_http_health(base_url: str, timeout_seconds: int | None = None) -> None:
    if timeout_seconds is None:
        raw = (os.getenv("BDD_HTTP_WAIT_SECONDS") or "").strip()
        timeout_seconds = int(raw) if raw.isdigit() else 45

    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None

    while time.time() < deadline:
        try:
            with httpx.Client(base_url=base_url, timeout=2.0) as probe:
                res = probe.get("/health")
                if res.status_code == 200:
                    return
        except Exception as exc:
            last_error = exc
        time.sleep(0.5)

    if last_error is not None:
        raise RuntimeError(f"HTTP service not ready, url={base_url}, error={last_error!r}")
    raise RuntimeError(f"HTTP service not ready, url={base_url}")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "scheduler: run BDD scenario against scheduler_pubmed service",
    )
    config.addinivalue_line(
        "markers",
        "http: run scenario against real HTTP services (dockerized APIs)",
    )


@pytest.fixture
def bdd_target(request: pytest.FixtureRequest) -> str:
    return _target_for_node(request.node)


@pytest.fixture
def bdd_http_mode() -> bool:
    return _http_mode_enabled()


@pytest.fixture(autouse=True)
def bdd_config_env(bdd_target: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if bdd_target == "scheduler":
        monkeypatch.setenv("PUBMED_SCHEDULER_CONFIG_DIR", str(_SCHEDULER_CONFIG_DIR))
        # Scheduler health BDD should not require database availability.
        monkeypatch.setenv("PUBMED_SCHEDULER_RUN_MIGRATIONS_ON_STARTUP", "false")

        monkeypatch.setenv("BDD_POSTGRES_DSN", _SCHEDULER_POSTGRES_DSN)
        monkeypatch.setenv("BDD_POSTGRES_SCHEMA", "pubmed_scheduler")

        monkeypatch.delenv("BIOMED_CONFIG_DIR", raising=False)
        return

    monkeypatch.setenv("BIOMED_CONFIG_DIR", str(_CORE_CONFIG_DIR))
    monkeypatch.setenv("BDD_POSTGRES_DSN", _CORE_POSTGRES_DSN)
    monkeypatch.setenv("BDD_POSTGRES_SCHEMA", "core_db")

    monkeypatch.delenv("PUBMED_SCHEDULER_CONFIG_DIR", raising=False)
    monkeypatch.delenv("PUBMED_SCHEDULER_RUN_MIGRATIONS_ON_STARTUP", raising=False)


@pytest.fixture(autouse=True)
def cleanup_qdrant_via_script(bdd_target: str, bdd_config_env: None) -> None:
    if bdd_target != "core":
        yield
        return

    script = _BDD_ROOT / "helpers" / "clear_qdrant.py"
    _run_script_best_effort(script)
    yield
    _run_script_best_effort(script)


@pytest.fixture(autouse=True)
def cleanup_postgres_via_script(bdd_target: str, bdd_config_env: None) -> None:
    port = 5433 if bdd_target == "core" else 5434
    if not _can_connect("localhost", port):
        yield
        return

    script = _BDD_ROOT / "helpers" / "clear_postgres.py"
    _run_script_best_effort(script)
    yield
    _run_script_best_effort(script)


@pytest.fixture
def app(bdd_target: str, bdd_config_env: None):
    if bdd_target == "core":
        if not _can_connect("localhost", 5433) or not _can_connect("localhost", 6335):
            pytest.skip("BDD dependencies unavailable: Postgres/Qdrant containers are not reachable")

        core_app_module = importlib.import_module("biomed_platform.api.app")
        return core_app_module.create_app()

    scheduler_app_module = importlib.import_module("scheduler_pubmed.src.api.app")
    return scheduler_app_module.create_app()


@pytest.fixture
def client(request: pytest.FixtureRequest, bdd_target: str, bdd_http_mode: bool):
    if bdd_http_mode and request.node.get_closest_marker("http") is not None:
        core_url = (os.getenv("BDD_CORE_API_URL") or "http://127.0.0.1:8000").rstrip("/")
        scheduler_url = (os.getenv("BDD_SCHEDULER_API_URL") or "http://127.0.0.1:9000").rstrip("/")
        base_url = scheduler_url if bdd_target == "scheduler" else core_url

        if bdd_target not in _HTTP_READY_ERRORS:
            try:
                _wait_http_health(base_url)
                _HTTP_READY_ERRORS[bdd_target] = ""
            except RuntimeError as exc:
                _HTTP_READY_ERRORS[bdd_target] = str(exc)

        if _HTTP_READY_ERRORS[bdd_target]:
            pytest.skip(_HTTP_READY_ERRORS[bdd_target])

        with httpx.Client(base_url=base_url, timeout=15.0) as c:
            yield c
        return

    app = request.getfixturevalue("app")
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
