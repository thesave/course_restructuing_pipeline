"""Prepare each course's evidence: citable syllabus items and course facts."""

from __future__ import annotations

import pathlib
import re
from collections import defaultdict
from typing import Any

import yaml
from pydantic import ValidationError

from evidence.models import CourseOverride, OverridesFile
from evidence.text import Item, Parsed, Removed, detect_language, leaves, parse_section, split_sentences
from modules_engine.io import atomic_write_text, load_yaml_mapping

FORMAT = "evidence/1"
SERVICE_SSDS = {"INF/01", "ING-INF/05"}
SECTIONS = {
    "outcomes": {"en": "Learning outcomes", "it": "Conoscenze e abilità da conseguire"},
    "contents": {"en": "Course contents", "it": "Contenuti"},
    "bibliography": {"en": "Readings/Bibliography", "it": "Testi/Bibliografia"},
    "teaching_methods": {"en": "Teaching methods", "it": "Metodi didattici"},
}
ITEM_PREFIXES = {"outcomes": "o", "contents": "c", "prerequisites": "p"}
# Teaching-method sentences suggesting a laboratory. UniBo requires the
# workplace-safety course before laboratory activities.
LABORATORY = re.compile(r"laborator|\blab\b|hands-on|esercitazion|workplace safety|health and safety", re.I)
CYCLES = {2: "second", 3: "first", 5: "single", 6: "single"}
# A leaf this long may hide a failed split.
LONG_ITEM = 400


def section_variants(course: dict, section: str) -> tuple[list[str], str]:
    """Texts of a syllabus section, in English when available, and their language.

    Merged courses may keep several variants of a section as a list.
    """
    syllabus = course.get("syllabus") or {}
    for language in ("en", "it"):
        value = ((syllabus.get(language) or {}).get("contents") or {}).get(SECTIONS[section][language])
        texts = value if isinstance(value, list) else [value]
        texts = list(dict.fromkeys(text.strip() for text in texts if isinstance(text, str) and text.strip()))
        if texts:
            return texts, language
    return [], ""


def _key(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def parse_variants(texts: list[str], language: str) -> Parsed:
    """Parse every variant; later variants add only top-level items not seen before."""
    merged = Parsed()
    seen: set[str] = set()
    for text in texts:
        parsed = parse_section(text, language)
        for target, items in ((merged.items, parsed.items), (merged.prerequisites, parsed.prerequisites)):
            for item in items:
                key = _key(" ".join(leaf.text for leaf in leaves([item])))
                if key not in seen:
                    seen.add(key)
                    target.append(item)
        merged.removed.extend(removed for removed in parsed.removed if removed not in merged.removed)
        merged.equivalent_courses.extend(code for code in parsed.equivalent_courses if code not in merged.equivalent_courses)
    return merged


def number_items(items: list[Item], prefix: str, language: str) -> list[dict]:
    result = []
    for index, item in enumerate(items, start=1):
        identifier = f"{prefix}{index}"
        entry: dict[str, Any] = {"id": identifier, "text": item.text}
        if item.heading:
            entry["heading"] = True
        if item.language and item.language != language:
            entry["language"] = item.language
        if item.items:
            entry["items"] = number_items(item.items, f"{identifier}.", language)
        result.append(entry)
    return result


def laboratory_fact(course: dict) -> dict:
    texts, _ = section_variants(course, "teaching_methods")
    evidence = [
        sentence
        for text in texts
        for line in text.splitlines()
        for sentence in split_sentences(line.strip(" -*•"))
        if LABORATORY.search(sentence)
    ]
    return {"value": bool(evidence), "evidence": list(dict.fromkeys(evidence))}


def course_facts(course: dict) -> dict:
    teachers = course.get("teachers") or []
    modules = [module for teacher in teachers for module in teacher.get("modules") or []]
    disi_teachings = sorted(
        {
            str(module["teaching_id"])
            for teacher in teachers
            if teacher.get("affiliation") == "disi"
            for module in teacher.get("modules") or []
            if module.get("ssd") in SERVICE_SSDS
        }
    )
    programmes = [
        {
            "code": str(programme.get("code")),
            "name": (programme.get("name") or {}).get("en") or (programme.get("name") or {}).get("it"),
            "department": programme.get("department"),
            "campus": programme.get("campus"),
            "cycle": CYCLES.get(programme.get("duration")),
            "languages": programme.get("languages") or [],
        }
        for programme in course.get("programmes") or []
    ]
    schedules = []
    for schedule in course.get("schedules") or []:
        entry = {"from": str(schedule.get("from")), "to": str(schedule.get("to"))}
        if entry not in schedules:
            schedules.append(entry)
    return {
        "credits": course.get("credits") or [],
        "ssds": course.get("ssds") or [],
        "languages": course.get("languages") or [],
        "teaching_modes": course.get("teaching_modes") or [],
        "campi": course.get("campi") or [],
        "schedules": schedules,
        "teaching_ids": sorted({str(module["teaching_id"]) for module in modules if module.get("teaching_id")}),
        "teachers": [
            {"name": teacher.get("name"), "affiliation": teacher.get("affiliation"), "role": teacher.get("role") or []}
            for teacher in teachers
        ],
        "programmes": programmes,
        "service": {
            "value": bool(disi_teachings) and not any(programme["department"] == "disi" for programme in programmes),
            "disi_teaching_ids": disi_teachings,
        },
        "laboratory": laboratory_fact(course),
    }


def course_evidence(course_id: str, course: dict) -> tuple[dict, dict]:
    """The evidence of one course and parser notes for the report."""
    outcome_texts, outcome_language = section_variants(course, "outcomes")
    content_texts, content_language = section_variants(course, "contents")
    bibliography_texts, bibliography_language = section_variants(course, "bibliography")
    outcomes = parse_variants(outcome_texts, outcome_language)
    contents = parse_variants(content_texts, content_language)
    prerequisites = outcomes.prerequisites + contents.prerequisites
    facts = course_facts(course)
    facts["equivalent_courses"] = [code for code in contents.equivalent_courses if code != course_id]
    evidence = {
        "format": FORMAT,
        "course": course_id,
        "title": (course.get("course_title") or {}).get("name", ""),
        "sources": {
            "outcomes": outcome_language,
            "contents": content_language,
            "bibliography": bibliography_language,
        },
        "facts": facts,
        "outcomes": number_items(outcomes.items, "o", outcome_language),
        "contents": number_items(contents.items, "c", content_language),
        "prerequisites": number_items(prerequisites, "p", content_language or outcome_language),
        "bibliography": _normalise_block("\n\n".join(bibliography_texts)),
        "removed": [{"text": removed.text, "reason": removed.reason} for removed in outcomes.removed + contents.removed],
    }
    content_leaves = list(leaves(contents.items))
    notes = {
        "outcome_variants": len(outcome_texts),
        "prose_contents": len(content_leaves) <= 2 or sum(item.prose for item in content_leaves) >= 0.8 * len(content_leaves),
    }
    return evidence, notes


def _normalise_block(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r", "").split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _walk(items: list[dict]):
    for item in items:
        yield item
        yield from _walk(item.get("items", []))


def _leaf_texts(items: list[dict]) -> list[str]:
    return [item["text"] for item in _walk(items) if not item.get("items")]


def _take(evidence: dict, item_id: str, sections: tuple[str, ...]) -> dict:
    for section in sections:
        stack = [evidence[section]]
        while stack:
            items = stack.pop()
            for index, item in enumerate(items):
                if item["id"] == item_id:
                    return items.pop(index)
                stack.append(item.get("items", []))
    raise ValueError(f"course {evidence['course']}: no item {item_id} in {', '.join(sections)}")


def apply_override(evidence: dict, override: CourseOverride) -> None:
    every = ("outcomes", "contents", "prerequisites")
    for item_id in override.drop:
        item = _take(evidence, item_id, every)
        text = " ".join([item["text"]] + _leaf_texts(item.get("items", [])))
        evidence["removed"].append({"text": text, "reason": "review"})
    for item_id in override.prerequisites:
        evidence["prerequisites"].append(_take(evidence, item_id, ("outcomes", "contents")))
    for item_id in override.contents:
        evidence["contents"].append(_take(evidence, item_id, ("prerequisites",)))
    if override.laboratory is not None:
        evidence["facts"]["laboratory"]["value"] = override.laboratory
        evidence["facts"]["laboratory"]["reviewed"] = True
    if override.note:
        evidence["review_note"] = override.note


def load_overrides(path: pathlib.Path | None) -> OverridesFile:
    if path is None or not path.exists():
        return OverridesFile()
    try:
        return OverridesFile.model_validate(load_yaml_mapping(path, "overrides file"))
    except ValidationError as error:
        raise ValueError(f"Invalid overrides file {path}: {error}") from error


def shared_syllabi(evidences: dict[str, dict]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for course_id, evidence in evidences.items():
        texts = _leaf_texts(evidence["outcomes"]) + _leaf_texts(evidence["contents"])
        groups[_key(" ".join(texts))].append(course_id)
    return {
        course_id: [other for other in members if other != course_id]
        for members in groups.values()
        if len(members) > 1
        for course_id in members
    }


class _Dumper(yaml.SafeDumper):
    pass


def _represent_str(dumper: yaml.SafeDumper, value: str):
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style="|" if "\n" in value else None)


_Dumper.add_representer(str, _represent_str)


def dump_evidence(evidence: dict) -> str:
    return yaml.dump(evidence, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=100)


def course_paths(courses_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    """Course id (from the file name, which keeps suffixes such as ``-A``) -> path."""
    return {path.stem.removeprefix("course-"): path for path in sorted(courses_dir.glob("course-*.yml"))}


def prepare_evidence(
    courses_dir: pathlib.Path, output_dir: pathlib.Path, overrides_path: pathlib.Path | None = None
) -> list[pathlib.Path]:
    paths = course_paths(courses_dir)
    if not paths:
        raise ValueError(f"No course-*.yml files in {courses_dir}")
    overrides = load_overrides(overrides_path)
    unknown = sorted(set(overrides.courses) - set(paths))
    if unknown:
        raise ValueError(f"Overrides for unknown courses: {', '.join(unknown)}")
    evidences: dict[str, dict] = {}
    notes: dict[str, dict] = {}
    for course_id, path in paths.items():
        evidences[course_id], notes[course_id] = course_evidence(course_id, load_yaml_mapping(path, "course file"))
        if course_id in overrides.courses:
            apply_override(evidences[course_id], overrides.courses[course_id])
    shared = shared_syllabi(evidences)
    written = []
    for course_id, evidence in evidences.items():
        evidence["facts"]["same_syllabus_as"] = shared.get(course_id, [])
        path = output_dir / f"evidence-{course_id}.yml"
        atomic_write_text(path, dump_evidence(evidence))
        written.append(path)
    report_path = output_dir / "report.md"
    atomic_write_text(report_path, render_report(evidences, notes, overrides))
    written.append(report_path)
    return written


def _short(text: str, limit: int = 110) -> str:
    text = text.replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def render_report(evidences: dict[str, dict], notes: dict[str, dict], overrides: OverridesFile) -> str:
    """Markdown checklist of the cases where parsing may need a reviewer's correction."""
    lines = ["# Evidence preparation report", ""]
    service = [evidence for evidence in evidences.values() if evidence["facts"]["service"]["value"]]
    items = sum(len(_leaf_texts(e["outcomes"])) + len(_leaf_texts(e["contents"])) for e in evidences.values())
    lines += [
        f"{len(evidences)} courses, {len(service)} service courses, {items} outcome and content items, "
        f"{sum(len(e['prerequisites']) for e in evidences.values())} prerequisite statements, "
        f"{len(overrides.courses)} courses with reviewer overrides "
        "(`overrides.yml` next to this report). Each section below lists cases to check; "
        "corrections go into the overrides file, by item id.",
        "",
    ]

    def section(title: str, explanation: str, rows: list[str]) -> None:
        lines.extend([f"## {title} ({len(rows)})", "", explanation, ""])
        lines.extend(rows or ["None."])
        lines.append("")

    def label(evidence: dict) -> str:
        return f"`{evidence['course']}` {evidence['title']}"

    section(
        "Service courses",
        "DISI members deliver INF/01 or ING-INF/05 credits and no associated degree programme belongs "
        "to DISI; the pool for the pilot.",
        [
            f"- {label(e)}, {'/'.join(map(str, e['facts']['credits']))} ECTS, programmes of "
            + ", ".join(sorted({str(p['department']) for p in e["facts"]["programmes"]}))
            for e in service
        ],
    )
    section(
        "Prerequisites",
        "Statements taken out of the outcomes and contents. Check that each one is a prerequisite "
        "and that no course material was swallowed.",
        [
            f"- {label(e)}: " + "; ".join(f"{p['id']} {_short(' '.join([p['text']] + _leaf_texts(p.get('items', []))), 160)}" for p in e["prerequisites"])
            for e in evidences.values()
            if e["prerequisites"]
        ],
    )
    section(
        "Removed text",
        "Administrative remarks and instructor names left out of the items.",
        [
            f"- {label(e)}: " + "; ".join(f"{r['reason']}: {_short(r['text'], 80)}" for r in e["removed"])
            for e in evidences.values()
            if e["removed"]
        ],
    )
    section(
        "Prose contents",
        "Contents written as paragraphs (or in at most two items), split into sentences; check that "
        "the items are sensible units.",
        [
            f"- {label(e)}: {len(_leaf_texts(e['contents']))} items"
            for e in evidences.values()
            if notes[e["course"]]["prose_contents"]
        ],
    )
    long_rows = []
    for e in evidences.values():
        long_items = [
            item
            for item in _walk(e["outcomes"] + e["contents"])
            if not item.get("items")
            and len(item["text"]) > LONG_ITEM
        ]
        if long_items:
            long_rows.append(f"- {label(e)}: " + ", ".join(f"{item['id']} ({len(item['text'])} characters)" for item in long_items))
    section(
        "Long items",
        f"Items over {LONG_ITEM} characters, where splitting may have failed.",
        long_rows,
    )
    section(
        "Untranslated passages",
        "Items of an English section detected as Italian.",
        [
            f"- {label(e)}: " + ", ".join(item["id"] for item in _walk(e["outcomes"] + e["contents"]) if item.get("language"))
            for e in evidences.values()
            if any(item.get("language") for item in _walk(e["outcomes"] + e["contents"]))
        ],
    )
    section(
        "Italian sections",
        "Sections taken from the Italian syllabus because the English one is missing.",
        [
            f"- {label(e)}: " + ", ".join(name for name, language in e["sources"].items() if language == "it")
            for e in evidences.values()
            if "it" in e["sources"].values()
        ],
    )
    section(
        "Outcome variants",
        "Merged courses whose teachings state different learning outcomes; the items join the variants.",
        [f"- {label(e)}: {notes[e['course']]['outcome_variants']} variants" for e in evidences.values() if notes[e["course"]]["outcome_variants"] > 1],
    )
    section(
        "Shared syllabi",
        "Courses with the same outcomes and contents; topic extraction and its review can run once per group.",
        sorted(
            {
                "- " + ", ".join(f"`{c}`" for c in sorted([e["course"], *e["facts"]["same_syllabus_as"]]))
                for e in evidences.values()
                if e["facts"]["same_syllabus_as"]
            }
        ),
    )
    section(
        "Equivalent courses",
        "Courses whose contents declare other codes as the same course.",
        [f"- {label(e)}: " + ", ".join(e["facts"]["equivalent_courses"]) for e in evidences.values() if e["facts"]["equivalent_courses"]],
    )
    section(
        "Laboratory",
        "Courses whose teaching methods suggest a laboratory, with the sentence behind the flag; "
        "set `laboratory: false` in the overrides where it is wrong.",
        [
            f"- {label(e)}{' (reviewed)' if e['facts']['laboratory'].get('reviewed') else ''}: "
            + _short(e["facts"]["laboratory"]["evidence"][0] if e["facts"]["laboratory"]["evidence"] else "set by review")
            for e in evidences.values()
            if e["facts"]["laboratory"]["value"]
        ],
    )
    return "\n".join(lines)
