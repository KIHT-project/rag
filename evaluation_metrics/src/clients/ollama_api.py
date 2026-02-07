from __future__ import annotations

from typing import Any

import httpx


class OllamaClient:
    def __init__(self, *, base_url: str, timeout_seconds: float = 120.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = httpx.Timeout(timeout_seconds)

    async def chat(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.2,
        num_predict: int = 450,
        seed: int | None = None,
    ) -> str:
        options: dict[str, Any] = {
            "temperature": temperature,
            "num_predict": num_predict,
        }
        if seed is not None:
            options["seed"] = int(seed)

        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": options,
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            r = await client.post(f"{self._base_url}/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
            # Ollama returns {"message": {"role": "...", "content": "..."}}
            return (data.get("message") or {}).get("content") or ""

    async def probe(self) -> None:
        """
        Lightweight connectivity check for Ollama server.
        Raises on connection/auth/server issues.
        """
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            r = await client.get(f"{self._base_url}/api/tags")
            r.raise_for_status()
