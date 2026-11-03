from __future__ import annotations

import pathlib
from typing import Any, Iterable, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from modules_engine.models import (
    AffinityFile,
    ConstraintsFile,
    Dependency,
    DependenciesFile,
    Objective,
    PrefixedModule,
    PrefixedModulesFile,
    Problem,
    TopicsFile,
)


ModelT = TypeVar("ModelT", bound=BaseModel)
# Format tag of the review files saved by the review page.
REVIEW_FORMAT = "modules-review/1"


def load_yaml_mapping(path: pathlib.Path, description: str) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as error:
        raise ValueError(f"{description} does not exist: {path}") from error
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Could not read {description} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a YAML mapping: {path}")
    return payload


def _load_model(path: pathlib.Path, description: str, model: type[ModelT]) -> ModelT:
    payload = load_yaml_mapping(path, description)
    try:
        return model.model_validate(payload)
    except ValidationError as error:
        raise ValueError(f"Invalid {description} {path}: {error}") from error


def dependency_cycle(topics: Iterable[str], dependencies: Iterable[Dependency]) -> list[str] | None:
    """Return one cycle of the topic dependency graph, or None if it is acyclic."""
    successors: dict[str, list[str]] = {topic: [] for topic in topics}
    for edge in dependencies:
        successors[edge.requires].append(edge.for_)
    state: dict[str, int] = {}
    for start in sorted(successors):
        if start in state:
            continue
        path = [start]
        stack = [iter(sorted(successors[start]))]
        state[start] = 1
        while stack:
            child = next(stack[-1], None)
            if child is None:
                state[path.pop()] = 2
                stack.pop()
            elif state.get(child) == 1:
                return path[path.index(child):] + [child]
            elif child not in state:
                state[child] = 1
                path.append(child)
                stack.append(iter(sorted(successors[child])))
    return None


def validate_problem(problem: Problem) -> None:
    """Check cross-file references; every problem is reported at once."""
    topics = problem.topics.topics
    errors: list[str] = []

    def unknown(keys: Iterable[str], where: str) -> None:
        missing = sorted(set(keys) - set(topics))
        if missing:
            errors.append(f"{where} references unknown topics: {missing}")

    seen: set[tuple[str, str]] = set()
    for edge in problem.dependencies:
        unknown((edge.requires, edge.for_), f"dependency {edge.requires} -> {edge.for_}")
        if (edge.requires, edge.for_) in seen:
            errors.append(f"duplicate dependency {edge.requires} -> {edge.for_}")
        seen.add((edge.requires, edge.for_))
    for constraint in problem.constraints:
        unknown(constraint.together or constraint.apart or (), f"constraint {constraint.id}")
    for module in problem.prefixed:
        unknown(module.topics, f"pre-fixed module {module.key}")
    needs_affinity = [
        *(f"objective term {term.term}" for term in problem.objective.terms if term.term == "cohesion"),
        *(f"constraint {item.id}" for item in problem.constraints if item.related_only is not None),
    ]
    if problem.affinity is None and needs_affinity:
        errors.append(f"{', '.join(needs_affinity)} need an affinity file")
    if problem.affinity is not None:
        unknown({key for first, second, _ in problem.affinity.pairs for key in (first, second)}, "affinity")
    for topic, spec in topics.items():
        unknown_courses = sorted(set(spec.courses) - set(problem.topics.courses))
        if problem.topics.courses and unknown_courses:
            errors.append(f"topic {topic} references unknown courses: {unknown_courses}")
    if not errors:
        cycle = dependency_cycle(topics, problem.dependencies)
        if cycle is not None:
            errors.append(f"dependencies contain a cycle: {' -> '.join(cycle)}")
    if errors:
        raise ValueError("Inconsistent inputs:\n- " + "\n- ".join(errors))


def load_problem(
    topics_path: pathlib.Path,
    dependencies_path: pathlib.Path,
    constraints_path: pathlib.Path,
    objective_path: pathlib.Path,
    affinity_path: pathlib.Path | None = None,
    prefixed_path: pathlib.Path | None = None,
) -> Problem:
    problem = Problem(
        topics=_load_model(topics_path, "topics file", TopicsFile),
        dependencies=tuple(_load_model(dependencies_path, "dependencies file", DependenciesFile).dependencies),
        constraints=tuple(_load_model(constraints_path, "constraints file", ConstraintsFile).constraints),
        objective=_load_model(objective_path, "objective file", Objective),
        affinity=_load_model(affinity_path, "affinity file", AffinityFile) if affinity_path else None,
        prefixed=tuple(_load_model(prefixed_path, "pre-fixed modules file", PrefixedModulesFile).modules)
        if prefixed_path else (),
    )
    validate_problem(problem)
    return problem


def load_hint(path: pathlib.Path) -> dict[str, int]:
    """Topic -> module position in an earlier run's modules.yml (whose modules
    are listed in dependency order, as the solver numbers them)."""
    modules = load_yaml_mapping(path, "hint proposal").get("modules") or []
    if not modules:
        raise ValueError(f"Hint proposal has no modules: {path}")
    return {topic: index for index, module in enumerate(modules) for topic in module["topics"]}


def review_modules(session: dict[str, Any], topics: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """The modules of a saved review as ``{key, topics, locked}``, checked for
    format, topics in several modules and (given ``topics``) unknown topics.
    Empty modules are dropped."""
    if session.get("format") != REVIEW_FORMAT:
        raise ValueError(f"not a review file (format {session.get('format')!r}, expected {REVIEW_FORMAT})")
    if not isinstance(session.get("modules"), list):
        raise ValueError("a review file needs a modules list")
    seen: set[str] = set()
    result = []
    for item in session["modules"]:
        if not isinstance(item, dict) or not isinstance(item.get("key"), str):
            raise ValueError(f"each module needs a key: {item!r}")
        members = [str(topic) for topic in item.get("topics") or []]
        repeated = sorted((seen & set(members)) | {topic for topic in members if members.count(topic) > 1})
        if repeated:
            raise ValueError(f"topics in several modules: {repeated}")
        seen.update(members)
        if members:
            result.append({"key": item["key"], "topics": members, "locked": bool(item.get("locked"))})
    if topics is not None:
        unknown = sorted(seen - set(topics))
        if unknown:
            raise ValueError(f"unknown topics: {unknown}")
    return result


def load_review(path: pathlib.Path) -> tuple[tuple[PrefixedModule, ...], dict[str, int]]:
    """Locked modules of a saved review (as exclusive pre-fixed modules) and
    the whole arrangement as a solver hint (topic -> module position)."""
    try:
        modules = review_modules(load_yaml_mapping(path, "review file"))
        prefixed = tuple(
            PrefixedModule(key=module["key"], topics=module["topics"], exclusive=True)
            for module in modules
            if module["locked"]
        )
        PrefixedModulesFile(modules=list(prefixed))
    except (ValueError, ValidationError) as error:
        raise ValueError(f"Invalid review file {path}: {error}") from error
    return prefixed, {topic: index for index, module in enumerate(modules) for topic in module["topics"]}


def atomic_write_text(path: pathlib.Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(contents, encoding="utf-8")
    temporary.replace(path)


def dump_yaml(payload: Any, flow_leaves: bool = False) -> str:
    """With ``flow_leaves``, innermost lists are written inline (one affinity
    pair per line)."""
    return yaml.safe_dump(
        payload,
        sort_keys=False,
        allow_unicode=True,
        width=100,
        default_flow_style=None if flow_leaves else False,
    )
