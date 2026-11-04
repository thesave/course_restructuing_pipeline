"""Requests for one course and the checks its reply must pass."""

from __future__ import annotations

import math
import pathlib
import re
from typing import Any, Iterator

from extraction.models import Extraction

PROMPTS = pathlib.Path(__file__).with_name("prompts")
SYSTEM_PROMPT = (PROMPTS / "system.txt").read_text(encoding="utf-8").strip()
COURSE_TEMPLATE = (PROMPTS / "course.txt").read_text(encoding="utf-8").strip()
HOURS_PER_ECTS = 8
BIBLIOGRAPHY_LIMIT = 2500
PROGRAMME_LIMIT = 5
KEY = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")
SHARE_TOLERANCE = 0.1
LISTED_ERRORS = 15


def course_credits(evidence: dict) -> int:
    return max(evidence["facts"].get("credits") or [0])


def expected_topics(credits: int) -> tuple[int, int]:
    """What the prompt asks for: topics of 0.5-1 ECTS."""
    return max(2, credits), max(3, 2 * credits)


def accepted_topics(credits: int) -> tuple[int, int]:
    """What validation accepts: looser than the request."""
    return max(1, math.ceil(credits / 2)), max(4, 3 * credits)


def walk(items: list[dict], ancestors: tuple[str, ...] = ()) -> Iterator[tuple[dict, tuple[str, ...]]]:
    for item in items:
        yield item, ancestors
        yield from walk(item.get("items", []), ancestors + (item["id"],))


def render_items(items: list[dict]) -> str:
    lines = [f"{'  ' * len(ancestors)}[{item['id']}] {item['text']}" for item, ancestors in walk(items)]
    return "\n".join(lines) or "(none)"


def render_programmes(programmes: list[dict]) -> str:
    names = list(dict.fromkeys(
        f"{programme['name']} ({programme['cycle']} cycle)" if programme.get("cycle") else str(programme["name"])
        for programme in programmes
    ))
    if len(names) > PROGRAMME_LIMIT:
        names = names[:PROGRAMME_LIMIT] + [f"and {len(names) - PROGRAMME_LIMIT} more"]
    return "; ".join(names) or "(unknown)"


def render_bibliography(text: str) -> str:
    if not text:
        return "(none)"
    if len(text) <= BIBLIOGRAPHY_LIMIT:
        return text
    return text[:BIBLIOGRAPHY_LIMIT].rstrip() + "\n[... bibliography truncated]"


def course_messages(evidence: dict) -> list[dict[str, str]]:
    credits = course_credits(evidence)
    low, high = expected_topics(credits)
    user = COURSE_TEMPLATE.format(
        title=evidence["title"],
        credits=credits or "unknown",
        hours=credits * HOURS_PER_ECTS if credits else "unknown",
        min_topics=low,
        max_topics=high,
        programmes=render_programmes(evidence["facts"].get("programmes") or []),
        outcomes=render_items(evidence["outcomes"]),
        contents=render_items(evidence["contents"]),
        bibliography=render_bibliography(evidence.get("bibliography") or ""),
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def retry_message(errors: list[str]) -> dict[str, str]:
    listed = errors[:LISTED_ERRORS] + ([f"... and {len(errors) - LISTED_ERRORS} more"] if len(errors) > LISTED_ERRORS else [])
    return {
        "role": "user",
        "content": "Your reply has these problems:\n- " + "\n- ".join(listed) + "\nReply again with the complete corrected answer.",
    }


def _inline_refs(node: Any, definitions: dict[str, Any]) -> Any:
    if isinstance(node, dict):
        if "$ref" in node:
            return _inline_refs(definitions[node["$ref"].rsplit("/", 1)[-1]], definitions)
        return {key: _inline_refs(value, definitions) for key, value in node.items() if key not in ("$defs", "title")}
    if isinstance(node, list):
        return [_inline_refs(value, definitions) for value in node]
    return node


def response_schema() -> dict[str, Any]:
    """The reply's JSON schema without references, which some local servers do not resolve."""
    schema = Extraction.model_json_schema()
    return _inline_refs(schema, schema.get("$defs", {}))


TECHNOLOGY_SPELLINGS = {"c++": "cpp", "c#": "csharp", "f#": "fsharp", ".net": "dotnet"}


def technology_words(technology: str) -> list[str]:
    """The technology as key words: "Unix shell" -> ["unix", "shell"], "C++" -> ["cpp"]."""
    text = technology.lower()
    for spelling, word in TECHNOLOGY_SPELLINGS.items():
        text = text.replace(spelling, f" {word} ")
    return [word for word in re.split(r"[^a-z0-9]+", text) if word]


def names_technology(key: str, technology: str) -> bool:
    """Whether the key contains the technology's words, consecutively."""
    words, tech = key.split("_"), technology_words(technology)
    return bool(tech) and any(words[start : start + len(tech)] == tech for start in range(len(words) - len(tech) + 1))


def validate_extraction(extraction: Extraction, evidence: dict) -> list[str]:
    """Problems of a schema-valid reply, phrased so that they can be sent back to the model."""
    errors: list[str] = []
    entries = list(walk(evidence["outcomes"])) + list(walk(evidence["contents"]))
    known = {item["id"] for item, _ in entries}
    cited = {item_id for topic in extraction.topics for item_id in topic.evidence}
    skipped = {entry.id for entry in extraction.not_topics}

    keys = [topic.key for topic in extraction.topics]
    for key in sorted({key for key in keys if keys.count(key) > 1}):
        errors.append(f"the key {key} is used by several topics")
    for topic in extraction.topics:
        if not KEY.match(topic.key):
            errors.append(f"the key {topic.key!r} is not lowercase snake_case")
        if not topic.evidence:
            errors.append(f"the topic {topic.key} cites no syllabus item")
        if not topic.name.strip() or not topic.description.strip():
            errors.append(f"the topic {topic.key} needs a name and a description")
        if topic.share <= 0:
            errors.append(f"the topic {topic.key} has a share of {topic.share}; shares must be positive")
        if topic.technology is not None and not topic.technology.strip():
            errors.append(f"the topic {topic.key} has an empty technology; use null instead")
        elif topic.technology and names_technology(topic.key, topic.technology):
            errors.append(
                f"the key {topic.key} names its technology {topic.technology}; keys name the concept "
                "and the technology goes only in the technology field"
            )
    unknown = sorted((cited | skipped) - known)
    if unknown:
        errors.append(f"these ids do not exist in the syllabus: {', '.join(unknown)}")
    both = sorted(cited & skipped)
    if both:
        errors.append(f"these items are both cited by a topic and listed in not_topics: {', '.join(both)}")

    covering = cited | skipped
    uncovered = [
        item["id"]
        for item, ancestors in walk(evidence["contents"])
        if not item.get("items") and item["id"] not in covering and not covering.intersection(ancestors)
    ]
    if uncovered:
        errors.append(
            "these content items are neither cited by a topic nor listed in not_topics: " + ", ".join(uncovered)
        )

    total = sum(topic.share for topic in extraction.topics)
    if extraction.topics and abs(total - 1) > SHARE_TOLERANCE:
        errors.append(f"the shares add up to {total:.2f} instead of 1")
    credits = course_credits(evidence)
    if credits:
        low, high = accepted_topics(credits)
        if not low <= len(extraction.topics) <= high:
            wanted_low, wanted_high = expected_topics(credits)
            errors.append(
                f"{len(extraction.topics)} topics for a {credits}-ECTS course; topics of 4 to 8 hours give "
                f"between {wanted_low} and {wanted_high}"
            )
    elif not extraction.topics:
        errors.append("the reply has no topics")
    return errors
