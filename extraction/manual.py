"""Requests answered outside this program, such as by Claude Code sub-agents.

`write_requests` writes each course's exact request (the same prompt and schema
as the server runs); `ingest_reply` applies the same checks to a reply and
writes the same topics file, so that the runs can be compared course by course.
"""

from __future__ import annotations

import json
import pathlib
import re

from pydantic import ValidationError

from extraction.client import strip_thinking
from extraction.models import EndpointSpec, EndpointsFile, Extraction
from extraction.prompt import course_messages, response_schema, validate_extraction
from extraction.run import CourseResult, Extractor, _now, schema_errors, syllabus_groups
from modules_engine.io import atomic_write_text, dump_yaml, load_yaml_mapping

AGENT_ENDPOINT = "claude-code-subagent"
FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def request_text(evidence: dict) -> str:
    system, user = course_messages(evidence)
    schema = json.dumps(response_schema(), indent=1)
    return (
        f"# Request for course {evidence['course']}\n\n## System prompt\n\n{system['content']}\n\n"
        f"## Course\n\n{user['content']}\n\n## Reply format\n\n"
        f"Reply with one JSON object that matches this JSON schema, and nothing else:\n\n{schema}\n"
    )


def write_requests(evidences: dict[str, dict], selected: list[str], output_dir: pathlib.Path) -> list[pathlib.Path]:
    """One request file per course to extract; courses with the same syllabus share one."""
    groups = syllabus_groups(evidences, selected)
    folder = output_dir / "requests"
    written = []
    for primary in groups:
        path = folder / f"request-{primary}.md"
        atomic_write_text(path, request_text(evidences[primary]))
        written.append(path)
    atomic_write_text(folder / "groups.yml", dump_yaml({"groups": groups}))
    return written


def ingest_reply(
    evidences: dict[str, dict], output_dir: pathlib.Path, course: str, reply_path: pathlib.Path, model: str
) -> list[str]:
    """Check a reply; when it passes, write the topics file (and copies). Returns the problems."""
    if course not in evidences:
        raise ValueError(f"No evidence for course {course}")
    evidence = evidences[course]
    attempts_path = output_dir / ".attempts" / f"{course}.json"
    history = json.loads(attempts_path.read_text(encoding="utf-8")) if attempts_path.exists() else []
    content = FENCE.sub("", strip_thinking(reply_path.read_text(encoding="utf-8")).strip())
    try:
        extraction = Extraction.model_validate(json.loads(content))
    except (json.JSONDecodeError, ValidationError) as error:
        errors = schema_errors(error)
    else:
        errors = validate_extraction(extraction, evidence)
    if errors:
        atomic_write_text(attempts_path, json.dumps(history + [errors], indent=1))
        return errors

    extractor = Extractor(EndpointsFile(model=model, endpoints=[EndpointSpec(url=AGENT_ENDPOINT)]), output_dir, output_dir / ".cache")
    results = {course: CourseResult(course, "ok", endpoint=AGENT_ENDPOINT, attempts=len(history) + 1)}
    extractor.write_topics(evidence, extraction, results[course])
    groups_path = output_dir / "requests" / "groups.yml"
    groups = load_yaml_mapping(groups_path, "request groups").get("groups", {}) if groups_path.exists() else {}
    for sibling in groups.get(course) or []:
        copied = extractor.copy(course, evidences[sibling])
        results[sibling] = copied or CourseResult(sibling, "failed", errors=[f"the topics of {course} do not fit; extract it separately"])
    manifest_path = output_dir / "run.yml"
    started = load_yaml_mapping(manifest_path, "run manifest").get("started", _now()) if manifest_path.exists() else _now()
    extractor.write_manifest(manifest_path, results, [], started)
    attempts_path.unlink(missing_ok=True)
    return []
