"""Split syllabus text into citable items.

A syllabus section becomes a tree of items: section headings (``Module 1``,
``PART TWO``) hold groups (``The UML language:``) that hold leaves. Plain
paragraphs are split into sentences. Prerequisite statements and
administrative remarks are taken out of the tree, so that topic extraction
does not mistake them for course material.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Lines that switch the parser into prerequisite mode, and lines that only
# announce the contents (dropped, back to contents mode).
_PREREQUISITE_WORDS = (
    r"(?:pre-?requisites?|requirements|requisiti|prerequisiti|prior knowledge|previous knowledge|"
    r"required background|conoscenze (?:preliminari|pregresse|richieste))"
)
PREREQUISITE_HEADING = re.compile(
    rf"^(?:(?:entry|course|admission|previous course) )?{_PREREQUISITE_WORDS}"
    rf"(?:\s*(?:/|and|&)\s*{_PREREQUISITE_WORDS})?\s*:?$",
    re.I,
)
CONTENTS_HEADING = re.compile(
    r"^(course |detailed )?(contents?|syllabus|programme|program|programma|contenuti|"
    r"topics|course topics)\s*:?$",
    re.I,
)
# "Course contents: BI module" names a part of the contents.
CONTENTS_LEAD = re.compile(r"^(?:course |detailed )?contents?\s*[:–-]\s*(.{1,60})$", re.I)
SECTION_HEADING = re.compile(
    r"^((module|modulo|modules|part|parte|unit|unità|block|blocco|section|chapter)\s*"
    r"([0-9]+|[ivx]+\b|[a-z]\b|one|two|three|four|five)|(first|second|third|fourth) (part|module))",
    re.I,
)
PREREQUISITE_SENTENCE = re.compile(
    r"pre-?requisit|prerequisiti|entry requirements|course requirements|\brequirements?\s*:|"
    r"^(no |a |any )?(prior|previous|preliminary) knowledge|"
    r"(prior|previous|preliminary) knowledge\b.{0,150}\b(is|are) (required|assumed|helpful|recommended|"
    r"necessary|useful|encouraged|expected)|"
    r"\b(require|requires|assume|assumes) (no |any )?(prior|previous) knowledge|"
    r"constraints on previous knowledge|"
    r"(assumed|expected) to (have|know|be familiar|be acquainted)\b|"
    r"(knowledge|familiarity|understanding) of .{0,150}\b(is|are) (assumed|required|expected)|"
    r"(required|recommended|necessary|essential|helpful|assumed|useful) (in order )?(to|for) "
    r"(attend|follow|take|effectively)|"
    r"in order to (effectively )?follow the course|"
    r"students (should|must) (already )?(have|know|be familiar)|"
    r"\bthe course (requires|assumes)\b|"
    r"no (prior|previous|specific) (knowledge|background)",
    re.I,
)
# Outcome phrasing that looks like a prerequisite ("is expected to know").
OUTCOME_SENTENCE = re.compile(r"\b(by|at) the end\b|upon completion|will be able|on successful completion", re.I)
# Administrative remarks: (reason, pattern).
ADMINISTRATIVE = (
    ("accessibility", re.compile(r"specific learning disorders|disabilit", re.I)),
    ("schedule", re.compile(r"dates and times|timetable|office hours|calendario", re.I)),
    ("pointer", re.compile(r"for (more|further) (details|information)|see the course('s)? (main )?(web ?page|website)", re.I)),
    ("link", re.compile(r"^(https?://|www\.)\S+$", re.I)),
    (
        "language of teaching",
        re.compile(
            r"(teaching|lectures|lessons|classes|the course) (will be|is|are) (mostly |entirely |"
            r"held |given |taught |delivered )*in (english|italian)|exams? (will be|is|can be) "
            r"(allowed|held|taken|possible) in",
            re.I,
        ),
    ),
)
# A paragraph opening like this is a prerequisite as a whole.
PREREQUISITE_LEAD = re.compile(r"^(entry |course |admission )?(pre-?requisites?|requirements)\s*:", re.I)
FORMAL_NOTE = re.compile(r"formal note", re.I)
FORMAL_NOTE_END = re.compile(r"cannot be chosen together", re.I)
COURSE_CODE = re.compile(r"^[-*•]?\s*([A-Z]?\d{4,5}(?:-[A-Z])?)\s*[-–]")
INSTRUCTOR = re.compile(
    r"\s*\((?:[^()]*\b)?prof\.?\s[^()]*\)|[;,]\s*(instructors?|lecturers?|teachers?|docenti?)\s*:.*$", re.I
)
INLINE_NUMBER = re.compile(r"(?:^|\s)(?=\d{1,2}[:)]\s)")
LIST_MARKER = re.compile(r"^(?:(\d+|[a-z])[.)]\s*|[-*•–·▪]\s*|o\s+)(?=\S)")
NUMBER_MARKER = re.compile(r"^(\d+|[a-z])[.)]\s*(?=\S)")
BOLD_HEADING = re.compile(r"^\*\*(.+?)\*\*(.*)$")
MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"“(]?[A-Z0-9])")
ABBREVIATIONS = {"e.g.", "i.e.", "etc.", "prof.", "dr.", "vs.", "fig.", "cf.", "al.", "ca.", "approx.", "no.", "ed.", "eds.", "vol.", "pp.", "p."}
ITALIAN_WORDS = {
    "i", "il", "lo", "la", "gli", "le", "della", "delle", "degli", "dei", "del", "dello", "di", "e",
    "per", "con", "che", "una", "un", "sono", "nel", "nella", "alla", "alle", "ai", "dal", "sui",
    "tra", "come", "chiavi", "dati", "linguaggi", "modalità", "gestione", "classi",
}
ENGLISH_WORDS = {
    "the", "of", "and", "to", "in", "for", "with", "is", "are", "on", "by", "an", "a", "from",
    "how", "its", "their", "this", "that", "be", "as", "at", "or",
}
# Words that cannot end a line unless the sentence goes on in the next one.
WRAP_WORDS = {"of", "the", "a", "an", "and", "or", "to", "in", "for", "with", "by", "on", "from", "as", "at", "its", "their", "is", "are"}
WRAPPED_LINE_MIN = 45
GROUP_LINE_LIMIT = 60
HEADING_LINE_LIMIT = 120


@dataclass
class Item:
    text: str
    items: list["Item"] = field(default_factory=list)
    heading: bool = False
    language: str = ""
    # A sentence of a paragraph rather than a line of a list.
    prose: bool = False


@dataclass
class Removed:
    text: str
    reason: str


@dataclass
class Parsed:
    items: list[Item] = field(default_factory=list)
    prerequisites: list[Item] = field(default_factory=list)
    removed: list[Removed] = field(default_factory=list)
    equivalent_courses: list[str] = field(default_factory=list)


@dataclass
class _Line:
    text: str
    kind: str  # heading, number, bullet or plain
    indent: int
    blank_before: bool
    group: bool = False
    blank_after: bool = True


def clean_inline(text: str) -> str:
    text = MARKDOWN_LINK.sub(r"\1", text)
    text = text.replace("**", "").replace("__", "")
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    start = 0
    for match in SENTENCE_END.finditer(text):
        before = text[start:match.start()]
        last_word = before.rsplit(None, 1)[-1].lower() if before.split() else ""
        if last_word in ABBREVIATIONS or re.fullmatch(r"[a-z]\.", last_word):
            continue
        sentences.append(before.strip())
        start = match.end()
    sentences.append(text[start:].strip())
    return [sentence for sentence in sentences if sentence]


def detect_language(text: str, default: str) -> str:
    """``it`` for Italian passages left in an English text, else ``default``."""
    if default != "en":
        return default
    words = re.findall(r"[a-zàèéìòù]+", text.lower())
    italian = sum(word in ITALIAN_WORDS for word in words)
    english = sum(word in ENGLISH_WORDS for word in words)
    return "it" if italian >= 2 and italian > english else default


def _is_prerequisite(text: str) -> bool:
    return bool(PREREQUISITE_SENTENCE.search(text)) and not OUTCOME_SENTENCE.search(text)


def administrative_reason(text: str) -> str | None:
    return next((reason for reason, pattern in ADMINISTRATIVE if pattern.search(text)), None)


def _is_upper_heading(text: str) -> bool:
    letters = [char for char in text if char.isalpha()]
    return (
        len(letters) >= 4
        and sum(char.isupper() for char in letters) >= 0.8 * len(letters)
        and len(text) <= 80
        and not text.endswith((".", ";", ","))
    )


def _raw_lines(text: str) -> list[str]:
    """Normalise markup: rules, glued bold headings and inline bullet lists."""
    lines: list[str] = []
    for line in text.replace("\r", "").split("\n"):
        line = re.sub(r"^\s*[-=_]{3,}\s*", "", line.rstrip())
        stripped = line.strip()
        bold = BOLD_HEADING.match(stripped)
        if bold and (not bold.group(2) or bold.group(2)[0].isalpha()):
            lines.append("# " + bold.group(1).strip())
            line = bold.group(2)
        if len(re.findall(r"(?:^|\s)\d{1,2}[:)]\s", line)) >= 3:
            # "1: Introduction 2: Processes 3: ..." on a single line.
            lines.extend(part.strip().replace(":", ")", 1) for part in INLINE_NUMBER.split(line) if part.strip())
            continue
        if re.search(r"[:;]\s+-\s+\w", line) and not LIST_MARKER.match(line.strip()):
            parts = re.split(r"(?<=[:;])\s+-\s+", line)
            lines.append(parts[0])
            lines.extend("- " + part for part in parts[1:])
            continue
        lines.append(line)
    return lines


def _continues(previous: _Line, previous_raw: str, raw: str) -> bool:
    """Whether ``raw`` continues the hard-wrapped ``previous`` line."""
    stripped = raw.strip()
    if previous.kind == "heading" or stripped.startswith("#") or LIST_MARKER.match(stripped):
        return False
    if raw[:1].isspace() and previous.kind in ("number", "bullet"):
        return True
    if previous.text.endswith((".", ":", ";", "!", "?")):
        return False
    if previous.text.endswith(",") or previous.text.rsplit(None, 1)[-1].lower() in WRAP_WORDS:
        return True
    return stripped[0].islower() and len(previous_raw.strip()) >= WRAPPED_LINE_MIN


def _classify(text: str) -> list[_Line]:
    result: list[_Line] = []
    blank = True
    previous_raw = ""
    for raw in _raw_lines(text):
        stripped = raw.strip()
        if not stripped:
            blank = True
            if result:
                result[-1].blank_after = True
            continue
        if result and not blank and _continues(result[-1], previous_raw, raw):
            result[-1].text = clean_inline(f"{result[-1].text} {stripped}")
            previous_raw = raw
            continue
        if result:
            result[-1].blank_after = blank
        previous_raw = raw
        indent = len(raw) - len(raw.lstrip())
        if stripped.startswith("#"):
            kind, body = "heading", stripped.lstrip("#").strip()
        else:
            marker = LIST_MARKER.match(stripped)
            body = stripped[marker.end():] if marker else stripped
            kind = ("number" if NUMBER_MARKER.match(stripped) else "bullet") if marker else "plain"
            if kind in ("plain", "number") and len(body) <= HEADING_LINE_LIMIT and (
                SECTION_HEADING.match(body) or (kind == "plain" and _is_upper_heading(body))
            ):
                kind = "heading"
        result.append(_Line(clean_inline(body), kind, indent, blank))
        blank = False
    for index, line in enumerate(result):
        following = result[index + 1] if index + 1 < len(result) else None
        if line.kind == "heading":
            continue
        if line.text.endswith(":"):
            line.group = line.kind != "plain" or len(line.text) <= GROUP_LINE_LIMIT
        elif (
            line.kind == "plain"
            and len(line.text) <= GROUP_LINE_LIMIT
            and not line.text.endswith((".", ";", ","))
            and following is not None
            and following.kind in ("number", "bullet")
            and following.indent >= line.indent
        ):
            line.group = True
        elif (
            line.kind in ("number", "bullet")
            and following is not None
            and following.kind in ("number", "bullet")
            and following.indent > line.indent
        ):
            line.group = True
    return result


_RANK = {"heading": 0, "plain": 1, "number": 2, "bullet": 3}


def _list_rank(line: _Line) -> int:
    """Indentation nests first, then numbers hold bullets."""
    return 10 * line.indent + _RANK[line.kind]


def parse_section(text: str, language: str, with_prerequisites: bool = True) -> Parsed:
    """Parse one syllabus section into items, prerequisites and removed text."""
    parsed = Parsed()
    lines = _classify(text)
    mode = "contents"
    # Open nodes: (rank, item, has list children).
    stack: list[list] = []
    in_note = False

    def roots() -> list[Item]:
        return parsed.prerequisites if mode == "prerequisites" else parsed.items

    def attach(item: Item, rank: int, opens: bool) -> None:
        while stack and stack[-1][0] >= rank:
            stack.pop()
        (stack[-1][1].items if stack else roots()).append(item)
        if opens:
            stack.append([rank, item, False])

    def leaf(text: str, line: _Line, prose: bool = False) -> None:
        kept = []
        for sentence in split_sentences(text) if line.kind != "plain" else [text]:
            reason = administrative_reason(sentence)
            if reason:
                parsed.removed.append(Removed(sentence, reason))
            else:
                kept.append(sentence)
        if not kept:
            return
        text = " ".join(kept)
        item = Item(text, language=detect_language(text, language), prose=prose)
        if with_prerequisites and mode == "contents" and _is_prerequisite(text):
            parsed.prerequisites.append(item)
            return
        if line.kind == "plain":
            # A paragraph after a blank line ends the groups that hold a list.
            if line.blank_before:
                while stack and stack[-1][0] > 0 and stack[-1][2]:
                    stack.pop()
            (stack[-1][1].items if stack else roots()).append(item)
            return
        attach(item, _list_rank(line), False)
        if stack:
            stack[-1][2] = True

    for line in lines:
        text = INSTRUCTOR.sub("", line.text)
        if text != line.text:
            parsed.removed.append(Removed(line.text, "instructor"))
        if in_note or (line.kind == "heading" and FORMAL_NOTE.search(text)):
            in_note = not FORMAL_NOTE_END.search(text)
            parsed.removed.append(Removed(text, "formal note"))
            code = COURSE_CODE.match(line.text if line.kind == "plain" else "- " + line.text)
            if code and line.kind == "bullet":
                parsed.equivalent_courses.append(code.group(1))
            continue
        bare = text.rstrip(":").strip()
        if with_prerequisites and PREREQUISITE_HEADING.match(text):
            mode, stack = "prerequisites", []
            continue
        if CONTENTS_HEADING.match(text):
            mode, stack = "contents", []
            continue
        lead = CONTENTS_LEAD.match(text)
        if lead and line.kind in ("plain", "heading"):
            mode, stack = "contents", []
            line.kind, text = "heading", lead.group(1)
            bare = text.rstrip(":").strip()
        if (
            mode == "prerequisites"
            and line.kind == "plain"
            and not line.group
            and line.blank_before
            and line.blank_after
            and len(text) <= 40
            and text[:1].isupper()
            and not text.endswith((".", ";", ",", ":"))
        ):
            # A standalone title such as "Course Objectives" ends a prerequisite block.
            line.kind = "heading"
        if line.kind == "heading":
            if mode == "prerequisites":
                mode = "contents"
            stack.clear()
            attach(Item(bare, heading=True, language=detect_language(bare, language)), 0, True)
            continue
        if line.group and with_prerequisites and mode == "contents" and _is_prerequisite(text):
            # A lead-in such as "Students are expected to have:" opens a list of prerequisites.
            mode, stack = "prerequisites", []
        if line.group:
            reason = administrative_reason(text)
            if reason:
                parsed.removed.append(Removed(text, reason))
                continue
            if line.kind == "plain" and line.blank_before:
                while stack and stack[-1][0] > 0 and stack[-1][2]:
                    stack.pop()
            rank = 1 if line.kind == "plain" else _list_rank(line)
            attach(Item(bare, language=detect_language(bare, language)), rank, True)
            continue
        if line.kind == "plain" and not PREREQUISITE_LEAD.match(text):
            sentences = split_sentences(text)
            for sentence in sentences:
                leaf(sentence, line, prose=len(sentences) > 1)
            continue
        leaf(text, line)
    parsed.items = _prune(parsed.items)
    parsed.prerequisites = _prune(parsed.prerequisites)
    return parsed


def _prune(items: list[Item]) -> list[Item]:
    """Drop headings and groups left without children by removed text."""
    result = []
    for item in items:
        had_children = bool(item.items)
        item.items = _prune(item.items)
        if item.items or not (item.heading or had_children):
            result.append(item)
    return result


def leaves(items: list[Item]):
    for item in items:
        if item.items:
            yield from leaves(item.items)
        else:
            yield item
