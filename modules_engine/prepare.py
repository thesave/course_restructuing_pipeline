"""Build engine inputs from a restructuring attempt's topic artifacts."""

from __future__ import annotations

import itertools
import pathlib
from collections import defaultdict

from modules_engine.io import atomic_write_text, dump_yaml, load_yaml_mapping


def load_attempt(attempt_dir: pathlib.Path) -> tuple[dict, dict[str, dict]]:
    """Return ``topics-global.yml`` and, per course id, its name and credits."""
    global_topics = load_yaml_mapping(attempt_dir / "topics-global.yml", "global topics file")
    courses: dict[str, dict] = {}
    for path in sorted(attempt_dir.glob("topics-of-course-*.yml")):
        course = load_yaml_mapping(path, "course topics file")["course"]
        course_id = str(course["id"])
        check = global_topics.get("course_credit_check", {}).get(course_id, {})
        courses[course_id] = {"title": course.get("name", ""), "credits": check.get("credits")}
    return global_topics, courses


def topic_courses(global_topics: dict) -> dict[str, set[str]]:
    return {
        topic: {str(course) for course in weights.get("per_course", {})}
        for topic, weights in global_topics["topic_weights"].items()
    }


def co_teaching_affinity(memberships: dict[str, set[str]], topics: list[str], minimum: float) -> list[list]:
    """Jaccard index of the sets of current courses teaching each topic pair."""
    by_course: dict[str, set[str]] = defaultdict(set)
    for topic in topics:
        for course in memberships[topic]:
            by_course[course].add(topic)
    pairs = {tuple(sorted(pair)) for members in by_course.values() for pair in itertools.combinations(members, 2)}
    result = []
    for first, second in sorted(pairs):
        shared = len(memberships[first] & memberships[second])
        score = round(shared / len(memberships[first] | memberships[second]), 3)
        if score >= minimum:
            result.append([first, second, score])
    return result


def prepare_inputs(
    attempt_dir: pathlib.Path,
    output_dir: pathlib.Path,
    course_ids: tuple[str, ...] = (),
    min_affinity: float = 0.0,
) -> list[pathlib.Path]:
    """Write ``topics.yml`` and ``affinity.yml`` for the topics of the given
    courses (all courses when none is given). Affinity uses co-teaching over
    the whole corpus, not only the selected courses."""
    global_topics, courses = load_attempt(attempt_dir)
    memberships = topic_courses(global_topics)
    unknown = sorted(set(course_ids) - set(courses))
    if unknown:
        raise ValueError(f"Unknown course ids in {attempt_dir}: {unknown}")
    selected = set(course_ids) or set(courses)
    topics = sorted(topic for topic, members in memberships.items() if members & selected)
    descriptions = global_topics["topics"]
    weights = global_topics["topic_weights"]
    topics_path = output_dir / "topics.yml"
    affinity_path = output_dir / "affinity.yml"
    atomic_write_text(topics_path, dump_yaml({
        "source": str(attempt_dir),
        "courses": {course: courses[course] for course in sorted(selected)},
        "topics": {
            topic: {
                "description": descriptions[topic],
                "ects": weights[topic]["ects"],
                "courses": sorted(memberships[topic] & selected),
            }
            for topic in topics
        },
    }))
    atomic_write_text(affinity_path, dump_yaml({
        "source": f"co-teaching Jaccard index over all courses of {attempt_dir}",
        "pairs": co_teaching_affinity(memberships, topics, min_affinity),
    }, flow_leaves=True))
    return [topics_path, affinity_path]
