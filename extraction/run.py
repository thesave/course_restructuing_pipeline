"""Extract the topics of many courses, spreading requests over several endpoints."""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import pathlib
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol

import yaml
from pydantic import ValidationError

from evidence.prepare import dump_evidence
from extraction.client import EndpointError, RequestError, strip_thinking
from extraction.models import Completion, EndpointsFile, Extraction, RunConfig
from extraction.prompt import COURSE_TEMPLATE, SYSTEM_PROMPT, course_messages, response_schema, retry_message, validate_extraction
from modules_engine.io import atomic_write_text, load_yaml_mapping

FORMAT = "extraction/1"
MANIFEST_FORMAT = "extraction-run/1"
PROMPT_DIGEST = hashlib.sha256((SYSTEM_PROMPT + "\n" + COURSE_TEMPLATE).encode("utf-8")).hexdigest()[:12]
LOGGER = logging.getLogger(__name__)


class Backend(Protocol):
    url: str

    def complete(self, messages: list[dict], request: dict[str, Any], schema: dict[str, Any]) -> Completion: ...


@dataclass
class CourseResult:
    course: str
    status: str  # ok, copied, skipped or failed
    endpoint: str = ""
    attempts: int = 0
    seconds: float = 0.0
    usage: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def load_evidence(evidence_dir: pathlib.Path) -> dict[str, dict]:
    evidences = {
        path.stem.removeprefix("evidence-"): load_yaml_mapping(path, "evidence file")
        for path in sorted(evidence_dir.glob("evidence-*.yml"))
    }
    if not evidences:
        raise ValueError(f"No evidence-*.yml files in {evidence_dir}")
    return evidences


def select_courses(
    evidences: dict[str, dict], courses: tuple[str, ...] = (), service: bool = False, limit: int | None = None
) -> list[str]:
    unknown = sorted(set(courses) - set(evidences))
    if unknown:
        raise ValueError(f"No evidence for courses: {', '.join(unknown)}")
    selected = [course for course in evidences if not courses or course in courses]
    if service:
        selected = [course for course in selected if evidences[course]["facts"]["service"]["value"]]
    return selected[:limit] if limit is not None else selected


def syllabus_groups(evidences: dict[str, dict], selected: list[str]) -> dict[str, list[str]]:
    """Primary course -> selected courses with the same syllabus, extracted once."""
    groups: dict[str, list[str]] = {}
    assigned: set[str] = set()
    for course in selected:
        if course in assigned:
            continue
        siblings = [other for other in evidences[course]["facts"].get("same_syllabus_as", []) if other in selected and other not in assigned]
        groups[course] = siblings
        assigned.update([course, *siblings])
    return groups


def cache_key(model: str, request: dict, schema: dict, messages: list[dict]) -> str:
    payload = json.dumps({"model": model, "request": request, "schema": schema, "messages": messages}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def schema_errors(error: Exception) -> list[str]:
    if isinstance(error, ValidationError):
        return [
            f"the reply does not match the schema at {'.'.join(map(str, entry['loc'])) or 'top level'}: {entry['msg']}"
            for entry in error.errors()[:10]
        ]
    return [f"the reply is not valid JSON: {error}"]


class Extractor:
    def __init__(
        self,
        settings: EndpointsFile,
        output_dir: pathlib.Path,
        cache_dir: pathlib.Path,
        config: RunConfig = RunConfig(),
    ) -> None:
        self.settings = settings
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.config = config
        self.schema = response_schema()

    def output_path(self, course: str) -> pathlib.Path:
        return self.output_dir / f"topics-{course}.yml"

    def failure_path(self, course: str) -> pathlib.Path:
        return self.output_dir / f"failed-{course}.yml"

    def _complete(self, backend: Backend, messages: list[dict]) -> tuple[Completion, str]:
        """A cached completion for exactly these messages, or a new one."""
        key = cache_key(self.settings.model, self.settings.request, self.schema, messages)
        path = self.cache_dir / key[:2] / f"{key}.json"
        if path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            return Completion(cached["content"], cached.get("usage", {}), 0.0), "cache"
        completion = backend.complete(messages, self.settings.request, self.schema)
        record = {
            "model": self.settings.model,
            "endpoint": backend.url,
            "created": _now(),
            "seconds": round(completion.seconds, 1),
            "usage": completion.usage,
            "messages": messages,
            "content": completion.content,
        }
        atomic_write_text(path, json.dumps(record, ensure_ascii=False, indent=1))
        return completion, backend.url

    def write_topics(self, evidence: dict, extraction: Extraction, result: CourseResult, copied_from: str | None = None) -> None:
        payload: dict[str, Any] = {
            "format": FORMAT,
            "course": evidence["course"],
            "title": evidence["title"],
            "run": {
                "model": self.settings.model,
                "endpoint": result.endpoint,
                "prompt": PROMPT_DIGEST,
                "request": self.settings.request,
                "attempts": result.attempts,
                "seconds": round(result.seconds, 1),
                "usage": result.usage,
                "created": _now(),
            },
        }
        if copied_from:
            payload["copied_from"] = copied_from
        payload.update(extraction.model_dump())
        atomic_write_text(self.output_path(evidence["course"]), dump_evidence(payload))
        self.failure_path(evidence["course"]).unlink(missing_ok=True)

    def extract(self, evidence: dict, backend: Backend) -> CourseResult:
        """Request, validate and, with the validation errors, retry; EndpointError propagates."""
        course = evidence["course"]
        result = CourseResult(course, "failed")
        messages = course_messages(evidence)
        content = ""
        for attempt in range(1, self.config.max_attempts + 1):
            result.attempts = attempt
            try:
                completion, served_by = self._complete(backend, messages)
            except RequestError as error:
                result.errors = [str(error)]
                break
            result.endpoint = served_by
            result.seconds += completion.seconds
            for name, value in completion.usage.items():
                result.usage[name] = result.usage.get(name, 0) + value
            content = strip_thinking(completion.content)
            try:
                extraction = Extraction.model_validate(json.loads(content))
            except (json.JSONDecodeError, ValidationError) as error:
                result.errors = schema_errors(error)
            else:
                result.errors = validate_extraction(extraction, evidence)
                if not result.errors:
                    result.status = "ok"
                    self.write_topics(evidence, extraction, result)
                    return result
            messages = messages + [{"role": "assistant", "content": content}, retry_message(result.errors)]
        failure = {"course": course, "model": self.settings.model, "attempts": result.attempts, "errors": result.errors, "last_reply": content}
        atomic_write_text(self.failure_path(course), dump_evidence(failure))
        return result

    def copy(self, source: str, evidence: dict) -> CourseResult | None:
        """Reuse the topics of a course with the same syllabus, if they are valid here too."""
        payload = load_yaml_mapping(self.output_path(source), "topics file")
        extraction = Extraction.model_validate({"topics": payload["topics"], "not_topics": payload["not_topics"]})
        if validate_extraction(extraction, evidence):
            return None
        result = CourseResult(evidence["course"], "copied", endpoint=payload["run"]["endpoint"], attempts=0)
        self.write_topics(evidence, extraction, result, copied_from=source)
        return result

    def run(
        self, evidences: dict[str, dict], selected: list[str], backends: list[tuple[Backend, int]]
    ) -> dict[str, CourseResult]:
        """Extract the selected courses; each backend serves its number of workers at once."""
        results: dict[str, CourseResult] = {}
        groups = syllabus_groups(evidences, selected)
        pending: deque[str] = deque()
        for primary, siblings in groups.items():
            if not self.config.force and all(self.output_path(course).exists() for course in [primary, *siblings]):
                for course in [primary, *siblings]:
                    results[course] = CourseResult(course, "skipped")
            else:
                pending.append(primary)
        condition = threading.Condition()
        record_lock = threading.Lock()
        in_flight = 0
        manifest_path = self.output_dir / "run.yml"
        started = _now()

        def record(result: CourseResult) -> None:
            with record_lock:
                results[result.course] = result
                LOGGER.info(
                    "%s %s via %s in %.0f s (%d attempts)%s",
                    result.course, result.status, result.endpoint or "-", result.seconds, result.attempts,
                    f": {result.errors[0]}" if result.errors else "",
                )
                self.write_manifest(manifest_path, results, backends, started)

        def handle(course: str, backend: Backend) -> None:
            if not self.config.force and self.output_path(course).exists():
                record(CourseResult(course, "skipped"))
            else:
                record(self.extract(evidences[course], backend))
            if results[course].status not in ("ok", "skipped"):
                return
            for sibling in groups.get(course, []):
                if not self.config.force and self.output_path(sibling).exists():
                    record(CourseResult(sibling, "skipped"))
                    continue
                copied = self.copy(course, evidences[sibling])
                record(copied or self.extract(evidences[sibling], backend))

        def worker(backend: Backend) -> None:
            nonlocal in_flight
            failures = 0
            while True:
                with condition:
                    while not pending and in_flight:
                        condition.wait(timeout=1.0)
                    if not pending:
                        return
                    course = pending.popleft()
                    in_flight += 1
                try:
                    handle(course, backend)
                    failures = 0
                except EndpointError as error:
                    failures += 1
                    LOGGER.warning("%s: %s (failure %d); course returned to the queue", course, error, failures)
                    with condition:
                        pending.append(course)
                    if failures >= self.config.max_endpoint_failures:
                        LOGGER.error("%s: giving up on this endpoint", backend.url)
                        return
                    time.sleep(min(60, 5 * 2 ** (failures - 1)))
                finally:
                    with condition:
                        in_flight -= 1
                        condition.notify_all()

        threads = [
            threading.Thread(target=worker, args=(backend,), name=f"{backend.url}#{index}", daemon=True)
            for backend, workers in backends
            for index in range(workers)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        for course in pending:
            record(CourseResult(course, "failed", errors=["no endpoint left to serve the request"]))
        self.write_manifest(manifest_path, results, backends, started)
        return results

    def write_manifest(
        self, path: pathlib.Path, results: dict[str, CourseResult], backends: list[tuple[Backend, int]], started: str
    ) -> None:
        previous = load_yaml_mapping(path, "run manifest").get("courses", {}) if path.exists() else {}
        courses = dict(previous)
        for course, result in sorted(results.items()):
            if result.status == "skipped" and course in previous:
                continue
            entry: dict[str, Any] = {"status": result.status, "endpoint": result.endpoint, "attempts": result.attempts, "seconds": round(result.seconds, 1)}
            if result.usage.get("completion_tokens"):
                entry["completion_tokens"] = result.usage["completion_tokens"]
            if result.errors:
                entry["errors"] = result.errors[:5]
            courses[course] = entry
        servers = {}
        for backend, workers in backends:
            served = [entry for entry in courses.values() if entry.get("endpoint") == backend.url and entry["status"] != "copied"]
            seconds = sum(entry["seconds"] for entry in served)
            tokens = sum(entry.get("completion_tokens", 0) for entry in served)
            servers[backend.url] = {
                "workers": workers,
                "courses": len(served),
                "seconds_in_requests": round(seconds, 1),
                "completion_tokens": tokens,
                "tokens_per_second": round(tokens / seconds, 1) if seconds else None,
            }
        statuses = [entry["status"] for entry in courses.values()]
        manifest = {
            "format": MANIFEST_FORMAT,
            "model": self.settings.model,
            "prompt": PROMPT_DIGEST,
            "request": self.settings.request,
            "started": started,
            "updated": _now(),
            "summary": {status: statuses.count(status) for status in ("ok", "copied", "skipped", "failed") if status in statuses},
            "servers": servers,
            "courses": courses,
        }
        atomic_write_text(path, yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True, width=100))
