from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass

import httpx


@dataclass(frozen=True, slots=True)
class DockerStack:
    project: str
    compose_file: str

    def up(self) -> None:
        subprocess.run(
            [
                "docker-compose",
                "-p",
                self.project,
                "-f",
                self.compose_file,
                "up",
                "-d",
            ],
            check=True,
        )

    def down(self) -> None:
        subprocess.run(
            [
                "docker-compose",
                "-p",
                self.project,
                "-f",
                self.compose_file,
                "down",
                "-v",
            ],
            check=False,
        )


def _wait_http_ok(url: str, *, timeout_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    last_exc: Exception | None = None

    while time.time() < deadline:
        try:
            with httpx.Client(timeout=2.0) as client:
                res = client.get(url)
                if 200 <= res.status_code < 500:
                    return
        except Exception as exc:
            last_exc = exc
        time.sleep(0.5)

    if last_exc is not None:
        raise RuntimeError(f"Dependency not ready, url={url}, last_error={last_exc!r}")
    raise RuntimeError(f"Dependency not ready, url={url}")


def wait_dependencies(
    *,
    qdrant_url: str,
    ollama_url: str,
    timeout_seconds: int = 3,
) -> None:
    _wait_http_ok(f"{qdrant_url.rstrip('/')}/collections", timeout_seconds=timeout_seconds)
    _wait_http_ok(f"{ollama_url.rstrip('/')}/api/version", timeout_seconds=timeout_seconds)


def build_stack_from_env(*, default_compose_file: str) -> tuple[DockerStack, str, str]:
    compose_file = os.getenv("BDD_COMPOSE_FILE", default_compose_file)
    project = os.getenv("BDD_PROJECT", "biomed-bdd")

    qdrant_url = os.getenv("BDD_QDRANT_URL", "http://127.0.0.1:6335")
    ollama_url = os.getenv("BDD_OLLAMA_URL", "http://127.0.0.1:11435")

    return DockerStack(project=project, compose_file=compose_file), qdrant_url, ollama_url
