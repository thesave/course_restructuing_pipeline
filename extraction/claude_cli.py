"""Completions from headless Claude Code (`claude -p`), billed to its claude.ai login.

Each request starts one `claude -p` process with no tools, our system prompt in
place of Claude Code's, and the reply schema as `--json-schema`. The process
runs in an empty folder with CLAUDE.md files, user settings, MCP servers and
skills switched off, so the model sees only the request (plus Claude Code's
short environment block). One process is one exchange, so a retry sends the
earlier replies and their problems inside a single new message.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import weakref
from typing import Any

from extraction.client import EndpointError, RequestError
from extraction.models import Completion, EndpointSpec

LABEL = "claude-cli"
# Request fields that map to `claude` options; anything else is refused.
OPTIONS = {"effort": "--effort"}
# Result subtypes that concern the request, not the service: retrying the
# same course elsewhere would not help.
REQUEST_FAILURES = ("error_max_structured_output_retries", "error_max_turns")


def flatten(messages: list[dict]) -> tuple[str, str]:
    """The system prompt and one user message carrying the whole conversation."""
    system = "\n\n".join(message["content"] for message in messages if message["role"] == "system")
    turns = [message for message in messages if message["role"] != "system"]
    if not turns or turns[0]["role"] != "user":
        raise RequestError(f"{LABEL}: the conversation must start with a user message")
    parts = [turns[0]["content"]]
    attempt = 0
    for message in turns[1:]:
        if message["role"] == "assistant":
            attempt += 1
            parts.append(f"Your reply number {attempt} was:\n\n{message['content']}")
        else:
            parts.append(message["content"])
    return system, "\n\n---\n\n".join(parts)


def usage_counts(result: dict) -> dict[str, int]:
    """Token counts in the names the OpenAI-compatible servers use."""
    usage = result.get("usage") or {}
    prompt = sum(int(usage.get(name) or 0) for name in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
    counts = {
        "prompt_tokens": prompt,
        "completion_tokens": int(usage.get("output_tokens") or 0),
        "cache_read_tokens": int(usage.get("cache_read_input_tokens") or 0),
        "thinking_tokens": int((usage.get("output_tokens_details") or {}).get("thinking_tokens") or 0),
    }
    return {name: value for name, value in counts.items() if value}


class ClaudeCliEndpoint:
    def __init__(self, spec: EndpointSpec, model: str, timeout: float, request: dict[str, Any]) -> None:
        unknown = sorted(set(request) - set(OPTIONS))
        if unknown:
            raise ValueError(f"{LABEL} supports only the request fields {sorted(OPTIONS)}, not {unknown}")
        self.spec = spec
        self.url = spec.url or LABEL
        self.model = spec.model or model
        self.timeout = timeout
        # An empty working folder, so no project files or settings are picked up.
        self.folder = tempfile.mkdtemp(prefix="extraction-claude-")
        weakref.finalize(self, shutil.rmtree, self.folder, True)
        self.environment = {**os.environ, "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1"}

    def _run(self, arguments: list[str], stdin: str | None = None) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(
                [self.spec.command, *arguments],
                input=stdin,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.folder,
                env=self.environment,
            )
        except subprocess.TimeoutExpired as error:
            raise EndpointError(f"{LABEL}: no reply within {self.timeout:g} s") from error
        except OSError as error:
            raise EndpointError(f"{LABEL}: cannot run {self.spec.command}: {error}") from error

    def command(self, system: str, request: dict[str, Any], schema: dict[str, Any]) -> list[str]:
        arguments = [
            "-p",
            "--model", self.model,
            "--output-format", "json",
            "--json-schema", json.dumps(schema),
            "--system-prompt", system,
            "--tools", "",
            "--no-session-persistence",
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--setting-sources", "project",
        ]
        for name, value in request.items():
            arguments += [OPTIONS[name], str(value)]
        return arguments

    def models(self) -> list[str]:
        version = self._run(["--version"]).stdout.strip() or "unknown version"
        status = self._run(["auth", "status"])
        try:
            auth = json.loads(status.stdout)
        except json.JSONDecodeError as error:
            raise EndpointError(f"{LABEL}: cannot read the login status: {status.stderr.strip()[:300]}") from error
        if not auth.get("loggedIn"):
            raise RequestError(f"{LABEL}: Claude Code is not logged in (run `claude` and /login)")
        return [f"{self.model} through Claude Code {version}, logged in with {auth.get('authMethod', 'unknown method')}"]

    def complete(self, messages: list[dict], request: dict[str, Any], schema: dict[str, Any]) -> Completion:
        system, prompt = flatten(messages)
        started = time.monotonic()
        process = self._run(self.command(system, request, schema), prompt)
        seconds = time.monotonic() - started
        try:
            result = json.loads(process.stdout)
        except json.JSONDecodeError as error:
            detail = (process.stderr or process.stdout).strip()[:500]
            raise EndpointError(f"{LABEL}: exit code {process.returncode}: {detail}") from error
        if result.get("is_error") or result.get("subtype") != "success":
            detail = f"{result.get('subtype')}: {str(result.get('result') or result.get('errors') or '')[:300]}"
            if result.get("subtype") in REQUEST_FAILURES:
                raise RequestError(f"{LABEL}: {detail}")
            raise EndpointError(f"{LABEL}: {detail}")
        served = list(result.get("modelUsage") or {})
        if self.model.startswith("claude-") and served and self.model not in served:
            raise RequestError(f"{LABEL}: asked for {self.model}, served by {', '.join(served)}")
        reply = result.get("structured_output")
        if reply is None:
            raise RequestError(f"{LABEL}: no structured output in the reply: {str(result.get('result'))[:300]}")
        return Completion(json.dumps(reply, ensure_ascii=False), usage_counts(result), seconds)
