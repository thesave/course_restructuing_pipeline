"""Evaluate a topic-to-module assignment independently of the solver model."""

from __future__ import annotations

import html
import itertools
import statistics
from collections import defaultdict
from typing import Any

from modules_engine.models import Problem, SolverConfig
from modules_engine.solver import (
    ECTS_SCALE,
    Solution,
    cohesion_baseline,
    pair_scores,
    scaled_ects,
    unrelated_pairs,
)


def module_keys(problem: Problem, assignment: dict[str, int]) -> dict[int, str]:
    """``mNN`` by position, or the key of the pre-fixed module a module holds."""
    keys = {module: f"m{rank:02d}" for rank, module in enumerate(sorted(set(assignment.values())), start=1)}
    for item in problem.prefixed:
        keys[assignment[item.topics[0]]] = item.key
    return keys


def module_ects(problem: Problem, assignment: dict[str, int]) -> dict[int, float]:
    loads: dict[int, int] = defaultdict(int)
    for topic, module in assignment.items():
        loads[module] += scaled_ects(problem.topics.topics[topic].ects)
    return {module: load / ECTS_SCALE for module, load in sorted(loads.items())}


def violations(problem: Problem, assignment: dict[str, int]) -> list[dict[str, Any]]:
    """Constraint violations in natural units; penalty = weight x amount for
    soft constraints (hard ones are reported with no penalty)."""
    found = []
    loads = module_ects(problem, assignment)
    keys = module_keys(problem, assignment)

    def record(constraint, amount: float, detail: Any, bound: str | None = None) -> None:
        hard = constraint.is_hard(bound)
        found.append({
            "constraint": constraint.id if bound is None else f"{constraint.id} ({bound})",
            "kind": constraint.kind,
            "hard": hard,
            "amount": round(amount, 3),
            "penalty": None if hard else round(constraint.weight * amount, 3),
            "detail": detail,
        })

    for constraint in problem.constraints:
        if constraint.module_ects is not None:
            bounds = constraint.module_ects
            for module, load in loads.items():
                if bounds.min is not None and load < bounds.min:
                    record(constraint, bounds.min - load, {keys[module]: load}, "min")
                if bounds.max is not None and load > bounds.max:
                    record(constraint, load - bounds.max, {keys[module]: load}, "max")
            continue
        if constraint.related_only is not None:
            shared: dict[int, list[list[str]]] = defaultdict(list)
            for first, second in unrelated_pairs(problem, constraint):
                if assignment[first] == assignment[second]:
                    shared[assignment[first]].append([first, second])
            if shared:
                record(
                    constraint,
                    sum(len(pairs) for pairs in shared.values()),
                    {keys[module]: pairs for module, pairs in sorted(shared.items())},
                )
            continue
        groups: dict[int, list[str]] = defaultdict(list)
        for topic in constraint.together or constraint.apart:
            groups[assignment[topic]].append(topic)
        detail = {keys[module]: topics for module, topics in sorted(groups.items())}
        if constraint.together is not None and len(groups) > 1:
            record(constraint, len(groups) - 1, detail)
        elif constraint.apart is not None:
            extra = sum(len(topics) - 1 for topics in groups.values())
            if extra:
                record(constraint, extra, {key: topics for key, topics in detail.items() if len(topics) > 1})
    return found


def cohesion(problem: Problem, assignment: dict[str, int]) -> dict[int, float]:
    """Per module, the affinity of its topic pairs minus the cohesion baseline
    for each pair (pairs missing from the affinity file score 0)."""
    if problem.affinity is None:
        return {}
    scores = pair_scores(problem)
    baseline = cohesion_baseline(problem)
    members: dict[int, list[str]] = defaultdict(list)
    for topic in sorted(assignment):
        members[assignment[topic]].append(topic)
    return {
        module: round(sum(
            scores.get(frozenset(pair), 0.0) - baseline
            for pair in itertools.combinations(topics, 2)
        ), 3)
        for module, topics in sorted(members.items())
    }


def module_dependencies(problem: Problem, assignment: dict[str, int]) -> tuple[list[dict[str, Any]], int]:
    """Module edges implied by topic dependencies, without edges already
    implied by longer paths; also returns the number of modules on the
    longest chain."""
    via: dict[tuple[int, int], list[list[str]]] = defaultdict(list)
    for edge in problem.dependencies:
        source, target = assignment[edge.requires], assignment[edge.for_]
        if source != target:
            via[source, target].append([edge.requires, edge.for_])
    successors: dict[int, set[int]] = defaultdict(set)
    for source, target in via:
        successors[source].add(target)

    def reachable(start: int, skip: tuple[int, int]) -> set[int]:
        seen, stack = set(), [start]
        while stack:
            node = stack.pop()
            for child in successors[node]:
                if (node, child) != skip and child not in seen:
                    seen.add(child)
                    stack.append(child)
        return seen

    kept = sorted(edge for edge in via if edge[1] not in reachable(edge[0], edge))
    keys = module_keys(problem, assignment)
    # Indices are a topological order, so one pass computes chain lengths.
    chain = {module: 1 for module in set(assignment.values())}
    for source, target in sorted(via):
        chain[target] = max(chain[target], chain[source] + 1)
    edges = [
        {"requires": keys[source], "for": keys[target], "via": sorted(via[source, target])}
        for source, target in kept
    ]
    return edges, max(chain.values())


def course_fragmentation(problem: Problem, assignment: dict[str, int]) -> dict[str, dict[str, Any]]:
    """For each current course, the modules its topics end up in."""
    spread: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    keys = module_keys(problem, assignment)
    for topic, spec in problem.topics.topics.items():
        for course in spec.courses:
            spread[course][keys[assignment[topic]]] += 1
    result = {}
    for course in sorted(spread):
        title = problem.topics.courses[course].title if course in problem.topics.courses else ""
        result[course] = {
            **({"title": title} if title else {}),
            "modules": len(spread[course]),
            "topics_per_module": dict(sorted(spread[course].items())),
        }
    return result


def build_report(problem: Problem, solution: Solution, config: SolverConfig, inputs: dict[str, str]) -> dict[str, Any]:
    header = {
        "status": solution.status,
        "inputs": inputs,
        "solver": {
            "time_limit_per_stage": config.time_limit,
            "workers": config.workers,
            "seed": config.seed,
            "max_modules": config.max_modules,
            "wall_time": round(solution.wall_time, 1),
        },
        "objective": {
            "mode": problem.objective.mode,
            "stages": [
                # A stage stopped by the time limit only bounds its term;
                # later stages may still improve it (see `final` below).
                {"term": stage.term, "value_at_stage": round(stage.value, 3), "optimal": stage.optimal}
                for stage in solution.stages
            ],
        },
    }
    if not solution.assignment:
        return {**header, "conflicting_hard_constraints": list(solution.conflicts)}
    assignment = solution.assignment
    loads = module_ects(problem, assignment)
    internal = cohesion(problem, assignment)
    edges, depth = module_dependencies(problem, assignment)
    fragmentation = course_fragmentation(problem, assignment)
    found = violations(problem, assignment)
    members: dict[int, list[str]] = defaultdict(list)
    for topic in sorted(assignment):
        members[assignment[topic]].append(topic)
    keys = module_keys(problem, assignment)
    prefixed = {item.key for item in problem.prefixed}
    modules = []
    for module in sorted(members):
        courses = sorted({course for topic in members[module] for course in problem.topics.topics[topic].courses})
        modules.append({
            "key": keys[module],
            **({"prefixed": True} if keys[module] in prefixed else {}),
            "ects": loads[module],
            "cohesion": internal.get(module, 0.0),
            "topics": members[module],
            "source_courses": courses,
        })
    ects = sorted(loads.values())
    spreads = sorted(item["modules"] for item in fragmentation.values())
    metrics = {
        "modules": len(modules),
        "total_ects": round(sum(ects), 2),
        "module_ects": {"min": ects[0], "median": statistics.median(ects), "max": ects[-1]},
        "cohesion": round(sum(internal.values()), 3),
        "cohesion_baseline": round(cohesion_baseline(problem), 3),
        "soft_constraint_penalty": round(sum(item["penalty"] or 0 for item in found), 3),
        "module_dependencies": len(edges),
        "longest_chain": depth,
        **({"modules_per_source_course": {"median": statistics.median(spreads), "max": spreads[-1]}} if spreads else {}),
    }
    final = {"soft_constraint_penalty": "soft_constraint_penalty", "cohesion": "cohesion", "module_count": "modules"}
    for stage in header["objective"]["stages"]:
        stage["final"] = metrics[final[stage["term"]]]
    return {
        **header,
        "metrics": metrics,
        "violations": found,
        "modules": modules,
        "module_dependencies": edges,
        "source_courses": fragmentation,
    }


def _label(value: str) -> str:
    return html.escape(value.strip(), quote=True).replace("\n", " ")


def render_mermaid(report: dict[str, Any]) -> str:
    """Modules (top to bottom in dependency order) with their topics; modules
    involved in a violation are highlighted, pre-fixed ones marked."""
    flagged = {
        key
        for item in report["violations"]
        for key in item["detail"]
    }
    # Module keys are free names, so nodes get safe aliases.
    aliases = {module["key"]: f"M{index}" for index, module in enumerate(report["modules"])}
    lines = ["%% Module proposal", "flowchart TB"]
    for module in report["modules"]:
        label = "<br/>".join([
            # No markup: mermaid-config.json disables HTML labels.
            f"{_label(module['key'])} · {module['ects']:g} ECTS" + (" · fixed" if module.get("prefixed") else ""),
            *(_label(topic) for topic in module["topics"]),
        ])
        style = "flagged" if module["key"] in flagged else "prefixed" if module.get("prefixed") else "module"
        lines.append(f'  {aliases[module["key"]]}["{label}"]:::{style}')
    lines.extend(
        f"  {aliases[edge['requires']]} --> {aliases[edge['for']]}" for edge in report["module_dependencies"]
    )
    lines.extend([
        "  classDef module fill:#e8f1fb,stroke:#24527a",
        "  classDef prefixed fill:#e9f6ec,stroke:#2d7a3e,stroke-width:2px",
        "  classDef flagged fill:#fdf0e6,stroke:#b35c00",
    ])
    return "\n".join(lines) + "\n"
