"""Compare topic partitions without ground truth.

Course recovery: how far communities just reproduce current courses, as the
NMI between a topic's community and its course over topics taught by a single
course, and the share of courses whose exclusive topics are split. Stability:
for consensus co-assignment, the mean of 4p(1-p) over pairs ever put together
(0 when every detection agrees). Reference agreement: Spearman correlation
between course-course similarities induced by the partition (cosine of the
credits each course spends per community) and by a manual decomposition into
macro-modules (cosine of the CFU per module), as in analyses/cluster*.csv.

    .venv/bin/python -m network_analysis.affinity_evaluation \
        data/restructuring/attempt-.../topics-global.yml \
        --partition NAME=communities.csv [--consensus NAME=coassignment.npz] \
        --reference analyses/cluster1-ext.csv [--output evaluation.md]
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import pathlib
import re
import statistics
from collections import Counter, defaultdict

import numpy as np
import yaml
from scipy.stats import spearmanr

from network_analysis.topic_affinity import load_consensus
from network_analysis.topic_graph import load_course_names, load_course_topics


def load_partition(path: pathlib.Path) -> dict[str, int]:
    """Read ``topic,community`` rows, or topic_graph.py community tables
    (``community,...,topics`` with ``; ``-separated topics)."""
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if rows and "topic" in rows[0]:
        return {row["topic"]: int(row["community"]) for row in rows}
    return {topic: int(row["community"]) for row in rows for topic in row["topics"].split("; ")}


def nmi(first: dict[str, object], second: dict[str, object]) -> float:
    keys = [key for key in first if key in second]
    count = len(keys)
    left = Counter(first[key] for key in keys)
    right = Counter(second[key] for key in keys)
    joint = Counter((first[key], second[key]) for key in keys)
    information = sum(n / count * math.log(n * count / (left[a] * right[b])) for (a, b), n in joint.items())
    entropy = lambda counts: -sum(n / count * math.log(n / count) for n in counts.values())
    total = entropy(left) + entropy(right)
    return 2 * information / total if total else 1.0


def course_recovery(partition: dict[str, int], course_topics: dict[str, dict[str, float]]) -> dict[str, float]:
    owners: dict[str, list[str]] = defaultdict(list)
    for course, topics in course_topics.items():
        for topic in topics:
            owners[topic].append(course)
    exclusive = {topic: courses[0] for topic, courses in owners.items() if len(courses) == 1 and topic in partition}
    by_course: dict[str, set[int]] = defaultdict(set)
    sizes: Counter[str] = Counter(exclusive.values())
    for topic, course in exclusive.items():
        by_course[course].add(partition[topic])
    candidates = [course for course, size in sizes.items() if size > 1]
    return {
        "course_nmi": round(nmi(partition, exclusive), 3),
        "courses_split": round(sum(len(by_course[course]) > 1 for course in candidates) / len(candidates), 3),
    }


def stability(frequency: np.ndarray) -> float:
    upper = frequency[np.triu_indices(frequency.shape[0], 1)]
    together = upper[upper > 0]
    return round(float(np.mean(4 * together * (1 - together))), 3) if together.size else 0.0


def reference_courses(path: pathlib.Path, names: dict[str, str]) -> tuple[list[str], dict[str, np.ndarray]]:
    """Map each reference row to a course id (by the id in parentheses, else by
    a unique title) and return its CFU per macro-module; unmapped rows are skipped."""
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    header = rows[0]
    columns = [index for index, name in enumerate(header) if index and name.strip() and "total" not in name.lower()]
    by_title: dict[str, list[str]] = defaultdict(list)
    for course, name in names.items():
        by_title[name.strip().lower()].append(course)
    vectors: dict[str, np.ndarray] = {}
    for row in rows[1:]:
        label = row[0]
        title = re.sub(r"\s*\([^()]*\)\s*$", "", label).strip()
        tokens = [token.strip() for token in label[len(title):].strip("() ").split(",")]
        candidates = [token for token in tokens if token in names]
        candidates += [token.zfill(len(token) + 1) for token in tokens if token.zfill(len(token) + 1) in names]
        if not candidates and len(by_title.get(title.lower(), [])) == 1:
            candidates = by_title[title.lower()]
        if not candidates or candidates[0] in vectors:
            continue
        vectors[candidates[0]] = np.array([float(row[index] or 0) for index in columns])
    return [header[index].strip() for index in columns], vectors


def cosine_matrix(vectors: list[np.ndarray]) -> np.ndarray:
    matrix = np.array(vectors, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
    return matrix @ matrix.T


def reference_agreement(
    partition: dict[str, int] | None,
    course_topics: dict[str, dict[str, float]],
    reference: dict[str, np.ndarray],
) -> float:
    """Spearman correlation of course similarities; ``partition=None`` uses
    raw topic overlap (each topic its own community) as a baseline."""
    courses = [course for course in reference if course in course_topics]
    labels = sorted({
        (partition.get(topic, f"_{topic}") if partition is not None else topic)
        for course in courses for topic in course_topics[course]
    }, key=str)
    position = {label: index for index, label in enumerate(labels)}
    ours = []
    for course in courses:
        vector = np.zeros(len(labels))
        for topic, share in course_topics[course].items():
            vector[position[partition.get(topic, f"_{topic}") if partition is not None else topic]] += share
        ours.append(vector)
    upper = np.triu_indices(len(courses), 1)
    correlation = spearmanr(cosine_matrix(ours)[upper], cosine_matrix([reference[c] for c in courses])[upper])[0]
    return round(float(correlation), 3)


def evaluate(
    topics_global: pathlib.Path,
    partitions: dict[str, dict[str, int] | None],
    consensuses: dict[str, pathlib.Path],
    references: list[pathlib.Path],
) -> list[dict[str, object]]:
    _, course_topics = load_course_topics(topics_global)
    course_topics = {course: topics for course, topics in course_topics.items() if topics}
    names = load_course_names(topics_global.parent)
    loaded = {path.stem: reference_courses(path, names)[1] for path in references}
    results = []
    for name, partition in partitions.items():
        row: dict[str, object] = {"variant": name}
        if partition is not None:
            sizes = Counter(partition.values())
            row["communities"] = len(sizes)
            row["median_size"] = statistics.median(sizes.values())
            row.update(course_recovery(partition, course_topics))
        if name in consensuses:
            row["instability"] = stability(load_consensus(consensuses[name]).frequencies())
        agreements = {stem: reference_agreement(partition, course_topics, reference) for stem, reference in loaded.items()}
        row.update({f"ref_{stem}": value for stem, value in agreements.items()})
        row["ref_mean"] = round(statistics.fmean(agreements.values()), 3) if agreements else None
        results.append(row)
    return results


def markdown_table(results: list[dict[str, object]]) -> str:
    columns = list(dict.fromkeys(key for row in results for key in row))
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" if c == "variant" else "---:" for c in columns) + "|"]
    lines += ["| " + " | ".join("" if row.get(c) is None else str(row[c]) for c in columns) + " |" for row in results]
    return "\n".join(lines) + "\n"


def _named_path(value: str) -> tuple[str, pathlib.Path]:
    name, separator, path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH: {value}")
    return name, pathlib.Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("topics_global", type=pathlib.Path)
    parser.add_argument("--partition", type=_named_path, action="append", default=[], help="NAME=CSV")
    parser.add_argument("--consensus", type=_named_path, action="append", default=[], help="NAME=coassignment.npz (same NAME as a partition)")
    parser.add_argument("--reference", type=pathlib.Path, action="append", default=[])
    parser.add_argument("--output", type=pathlib.Path, help="Write the table as Markdown and YAML next to it")
    args = parser.parse_args()
    partitions: dict[str, dict[str, int] | None] = {"topic overlap (baseline)": None}
    partitions.update({name: load_partition(path) for name, path in args.partition})
    results = evaluate(args.topics_global, partitions, dict(args.consensus), args.reference)
    table = markdown_table(results)
    print(table)
    if args.output is not None:
        args.output.write_text(table, encoding="utf-8")
        args.output.with_suffix(".yml").write_text(yaml.safe_dump(results, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
