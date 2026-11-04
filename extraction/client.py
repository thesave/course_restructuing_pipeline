"""Chat completions from OpenAI-compatible servers (llama.cpp, Ollama, LM Studio, vLLM)."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Any

from extraction.models import Completion, EndpointSpec

THINKING = re.compile(r"<think>.*?</think>", re.S)


class EndpointError(Exception):
    """The server could not be reached or failed; the request may succeed elsewhere."""


class RequestError(Exception):
    """The server rejected the request or returned an unusable reply."""


def api_base(url: str) -> str:
    url = url.rstrip("/")
    return url if url.endswith("/v1") else f"{url}/v1"


def strip_thinking(content: str) -> str:
    """Drop reasoning blocks that some chat templates leave in the content."""
    return THINKING.sub("", content).strip()


class Endpoint:
    def __init__(self, spec: EndpointSpec, model: str, timeout: float) -> None:
        self.spec = spec
        self.url = spec.url or ""
        self.model = spec.model or model
        self.timeout = timeout

    def _send(self, path: str, body: dict | None = None) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.spec.api_key:
            headers["Authorization"] = f"Bearer {self.spec.api_key}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(api_base(self.url) + path, data=data, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:500]
            if error.code >= 500 or error.code == 429:
                raise EndpointError(f"{self.url}: HTTP {error.code}: {detail}") from error
            raise RequestError(f"{self.url}: HTTP {error.code}: {detail}") from error
        except (urllib.error.URLError, OSError) as error:
            raise EndpointError(f"{self.url}: {error}") from error
        except json.JSONDecodeError as error:
            raise EndpointError(f"{self.url}: reply is not JSON: {error}") from error

    def models(self) -> list[str]:
        return [str(entry.get("id")) for entry in self._send("/models").get("data", [])]

    def complete(self, messages: list[dict], request: dict[str, Any], schema: dict[str, Any]) -> Completion:
        body = {
            "model": self.model,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "extraction", "strict": True, "schema": schema},
            },
            **request,
        }
        started = time.monotonic()
        reply = self._send("/chat/completions", body)
        seconds = time.monotonic() - started
        try:
            choice = reply["choices"][0]
            content = choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as error:
            raise RequestError(f"{self.url}: unexpected reply: {str(reply)[:300]}") from error
        if choice.get("finish_reason") == "length":
            raise RequestError(f"{self.url}: reply cut off at max_tokens; raise request.max_tokens")
        usage = {key: int(value) for key, value in (reply.get("usage") or {}).items() if isinstance(value, int)}
        return Completion(strip_thinking(content), usage, seconds)
