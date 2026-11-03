"""Partition topics into modules with CP-SAT.

Each topic gets a module index; dependencies require ``index(requires) <=
index(for)``, so module indices are a topological order of the module
dependency graph, which is therefore acyclic by construction. Used modules
form a prefix of the indices.
"""

from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass

from ortools.sat.python import cp_model

from modules_engine.models import TERM_SENSES, Constraint, Problem, Quantile, SolverConfig, Threshold


LOGGER = logging.getLogger(__name__)

# CP-SAT needs integers: ECTS are scaled to hundredths, soft-constraint
# weights to hundredths, affinity scores to thousandths.
ECTS_SCALE = 100
WEIGHT_SCALE = 100
AFFINITY_SCALE = 1000
PENALTY_SCALE = ECTS_SCALE * WEIGHT_SCALE
EXPLAIN_TIME_LIMIT = 10.0


def scaled_ects(ects: float) -> int:
    return max(1, round(ects * ECTS_SCALE))


def pair_scores(problem: Problem) -> dict[frozenset[str], float]:
    if problem.affinity is None:
        return {}
    return {frozenset((first, second)): score for first, second, score in problem.affinity.pairs}


def resolve_threshold(threshold: Threshold | None, problem: Problem) -> float:
    """A fixed value, or a quantile (linearly interpolated) of the scores
    listed in the affinity file."""
    if threshold is None:
        return 0.0
    if not isinstance(threshold, Quantile):
        return float(threshold)
    scores = sorted(score for *_, score in problem.affinity.pairs)
    if not scores:
        return 0.0
    position = threshold.quantile * (len(scores) - 1)
    low = int(position)
    high = min(low + 1, len(scores) - 1)
    return scores[low] + (scores[high] - scores[low]) * (position - low)


def cohesion_baseline(problem: Problem) -> float:
    term = next((item for item in problem.objective.terms if item.term == "cohesion"), None)
    return resolve_threshold(term.baseline if term else None, problem)


def unrelated_pairs(problem: Problem, constraint: Constraint) -> list[tuple[str, str]]:
    """Topic pairs a related_only constraint keeps apart: missing from the
    affinity file, scoring 0, or scoring below the constraint's min_score."""
    scores = pair_scores(problem)
    minimum = resolve_threshold(constraint.related_only.min_score, problem)
    return [
        (first, second)
        for first, second in itertools.combinations(sorted(problem.topics.topics), 2)
        if scores.get(frozenset((first, second)), 0.0) <= 0
        or scores[frozenset((first, second))] < minimum
    ]


@dataclass(frozen=True)
class Stage:
    term: str
    value: float
    optimal: bool


@dataclass(frozen=True)
class Solution:
    # OPTIMAL, FEASIBLE (time limit reached), INFEASIBLE or UNKNOWN.
    status: str
    # Topic -> module index, a topological order of module dependencies.
    assignment: dict[str, int]
    stages: tuple[Stage, ...] = ()
    # Labels of hard constraints that cannot hold together (INFEASIBLE only).
    conflicts: tuple[str, ...] = ()
    wall_time: float = 0.0


class _ModelBuilder:
    def __init__(self, problem: Problem, config: SolverConfig, explain: bool = False):
        self.problem = problem
        self.explain = explain
        self.model = cp_model.CpModel()
        self.assumptions: dict[str, cp_model.IntVar] = {}
        self.topics = sorted(problem.topics.topics)
        self.position = {topic: index for index, topic in enumerate(self.topics)}
        self.weights = [scaled_ects(problem.topics.topics[topic].ects) for topic in self.topics]
        self.total = sum(self.weights)
        count = len(self.topics)
        self.size = min(count, config.max_modules or count)
        model = self.model

        self.x = [
            [model.new_bool_var(f"x[{topic},{module}]") for module in range(self.size)]
            for topic in self.topics
        ]
        self.module = [model.new_int_var(0, self.size - 1, f"module[{topic}]") for topic in self.topics]
        self.used = [model.new_bool_var(f"used[{module}]") for module in range(self.size)]
        for index, row in enumerate(self.x):
            model.add_exactly_one(row)
            model.add(self.module[index] == sum(module * var for module, var in enumerate(row)))
        for module in range(self.size):
            column = [row[module] for row in self.x]
            for var in column:
                model.add_implication(var, self.used[module])
            model.add(sum(column) >= self.used[module])
            if module:
                model.add(self.used[module - 1] >= self.used[module])
        self.load = [
            sum(weight * row[module] for weight, row in zip(self.weights, self.x))
            for module in range(self.size)
        ]

        self.penalty: list = []
        for edge in problem.dependencies:
            self._hard(
                f"dependency {edge.requires} -> {edge.for_}",
                model.add(self._module(edge.requires) <= self._module(edge.for_)),
            )
        for constraint in problem.constraints:
            getattr(self, f"_add_{constraint.kind}")(constraint)
        self._add_prefixed()

        self.terms = {
            "soft_constraint_penalty": (sum(self.penalty), PENALTY_SCALE),
            "cohesion": (self._cohesion(), AFFINITY_SCALE),
            "module_count": (sum(self.used), 1),
        }

    def _module(self, topic: str) -> cp_model.IntVar:
        return self.module[self.position[topic]]

    def _hard(self, label: str, *constraints: cp_model.Constraint) -> None:
        if not self.explain:
            return
        literal = self.assumptions.get(label)
        if literal is None:
            literal = self.assumptions[label] = self.model.new_bool_var(label)
        for constraint in constraints:
            constraint.only_enforce_if(literal)

    def _penalty_coefficient(self, constraint: Constraint) -> int:
        return max(1, round(constraint.weight * WEIGHT_SCALE))

    def _add_together(self, constraint: Constraint) -> None:
        first, *rest = constraint.together
        if constraint.is_hard():
            self._hard(
                f"constraint {constraint.id}",
                *(self.model.add(self._module(first) == self._module(topic)) for topic in rest),
            )
            return
        # Number of modules the group is spread over, minus one.
        parts = [self.model.new_bool_var(f"parts[{constraint.id},{module}]") for module in range(self.size)]
        for topic in constraint.together:
            for module, var in enumerate(self.x[self.position[topic]]):
                self.model.add_implication(var, parts[module])
        self.penalty.append(self._penalty_coefficient(constraint) * ECTS_SCALE * (sum(parts) - 1))

    def _add_apart(self, constraint: Constraint) -> None:
        rows = [self.x[self.position[topic]] for topic in constraint.apart]
        sharing = [sum(row[module] for row in rows) for module in range(self.size)]
        if constraint.is_hard():
            self._hard(f"constraint {constraint.id}", *(self.model.add(total <= 1) for total in sharing))
            return
        for module, total in enumerate(sharing):
            extra = self.model.new_int_var(0, len(rows), f"extra[{constraint.id},{module}]")
            self.model.add(extra >= total - 1)
            self.penalty.append(self._penalty_coefficient(constraint) * ECTS_SCALE * extra)

    def _add_prefixed(self) -> None:
        """Pre-fixed modules are human decisions, so always hard."""
        anchors = []
        for item in self.problem.prefixed:
            first, *rest = item.topics
            row = self.x[self.position[first]]
            # Equal assignment rows propagate better than equal module indices.
            constraints = [
                self.model.add(var == row[module])
                for topic in rest
                for module, var in enumerate(self.x[self.position[topic]])
            ]
            if item.exclusive:
                members = set(item.topics)
                outside = [self.x[index] for index, topic in enumerate(self.topics) if topic not in members]
                # Wherever the module is, no outside topic may be.
                constraints += [
                    self.model.add(sum(other[module] for other in outside) <= len(outside) * (1 - row[module]))
                    for module in range(self.size)
                ]
            self._hard(f"pre-fixed module {item.key}", *constraints)
            anchors.append((item.key, first))
        for (first_key, first), (second_key, second) in itertools.combinations(anchors, 2):
            self._hard(
                f"pre-fixed modules {first_key} and {second_key} apart",
                self.model.add(self._module(first) != self._module(second)),
            )

    def _add_module_ects(self, constraint: Constraint) -> None:
        bounds = constraint.module_ects
        coefficient = self._penalty_coefficient(constraint)
        if bounds.min is not None:
            minimum = round(bounds.min * ECTS_SCALE)
            if constraint.is_hard("min"):
                self._hard(
                    f"constraint {constraint.id} (min)",
                    *(self.model.add(load >= minimum * used) for load, used in zip(self.load, self.used)),
                )
            else:
                for module, (load, used) in enumerate(zip(self.load, self.used)):
                    shortfall = self.model.new_int_var(0, minimum, f"shortfall[{module}]")
                    self.model.add(shortfall >= minimum * used - load)
                    self.penalty.append(coefficient * shortfall)
        if bounds.max is not None:
            maximum = round(bounds.max * ECTS_SCALE)
            if constraint.is_hard("max"):
                self._hard(
                    f"constraint {constraint.id} (max)",
                    *(self.model.add(load <= maximum) for load in self.load),
                )
            else:
                for module, load in enumerate(self.load):
                    excess = self.model.new_int_var(0, self.total, f"excess[{module}]")
                    self.model.add(excess >= load - maximum)
                    self.penalty.append(coefficient * excess)

    def _add_related_only(self, constraint: Constraint) -> None:
        pairs = unrelated_pairs(self.problem, constraint)
        if constraint.is_hard():
            self._hard(
                f"constraint {constraint.id}",
                *(self.model.add(self._module(first) != self._module(second)) for first, second in pairs),
            )
            return
        coefficient = self._penalty_coefficient(constraint) * ECTS_SCALE
        self.penalty.extend(coefficient * self._same(first, second, rewarded=False) for first, second in pairs)

    def _same(self, first: str, second: str, rewarded: bool) -> cp_model.IntVar:
        """A literal for "both topics share a module". When the objective
        rewards it, it need only force sharing; when it costs, sharing must
        force it."""
        same = self.model.new_bool_var(f"same[{first},{second}]")
        if rewarded:
            self.model.add(self._module(first) == self._module(second)).only_enforce_if(same)
        else:
            self.model.add(self._module(first) != self._module(second)).only_enforce_if(~same)
        return same

    def _cohesion(self):
        if self.problem.affinity is None:
            return 0
        scores = pair_scores(self.problem)
        baseline = cohesion_baseline(self.problem)
        # With a baseline every pair counts, missing ones at -baseline.
        pairs = (
            itertools.combinations(self.topics, 2)
            if baseline > 0
            else (sorted(pair) for pair in scores)
        )
        terms = []
        for first, second in pairs:
            coefficient = round((scores.get(frozenset((first, second)), 0.0) - baseline) * AFFINITY_SCALE)
            if coefficient:
                terms.append(coefficient * self._same(first, second, rewarded=coefficient > 0))
        return sum(terms)

    def hint(self, solver: cp_model.CpSolver) -> None:
        self.model.clear_hints()
        for row in self.x:
            for var in row:
                self.model.add_hint(var, solver.value(var))

    def hint_assignment(self, assignment: dict[str, int]) -> None:
        """Start from a known assignment (e.g. the proposal under review);
        CP-SAT repairs it where it breaks the current constraints."""
        self.model.clear_hints()
        for index, topic in enumerate(self.topics):
            module = assignment.get(topic)
            if module is None or module >= self.size:
                continue
            for column, var in enumerate(self.x[index]):
                self.model.add_hint(var, int(column == module))

    def assignment(self, solver: cp_model.CpSolver) -> dict[str, int]:
        raw = {topic: solver.value(self.module[index]) for index, topic in enumerate(self.topics)}
        compact = {module: rank for rank, module in enumerate(sorted(set(raw.values())))}
        return {topic: compact[module] for topic, module in raw.items()}


def _solver(config: SolverConfig, workers: int | None = None, time_limit: float | None = None) -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = config.time_limit if time_limit is None else time_limit
    solver.parameters.num_workers = config.workers if workers is None else workers
    solver.parameters.random_seed = config.seed
    return solver


def explain_infeasibility(problem: Problem, config: SolverConfig) -> tuple[str, ...]:
    """Return a small set of hard constraints that cannot hold together.

    CP-SAT reports a sufficient subset of the assumption literals; dropping
    each label in turn and re-solving shrinks it towards a minimal one.
    """
    builder = _ModelBuilder(problem, config, explain=True)
    literals = builder.assumptions

    def infeasible(labels: list[str]) -> list[str] | None:
        builder.model.clear_assumptions()
        builder.model.add_assumptions([literals[label] for label in labels])
        solver = _solver(config, workers=1, time_limit=min(config.time_limit, EXPLAIN_TIME_LIMIT))
        if solver.solve(builder.model) != cp_model.INFEASIBLE:
            return None
        names = {literals[label].index: label for label in labels}
        return [names[index] for index in solver.sufficient_assumptions_for_infeasibility()]

    core = infeasible(sorted(literals))
    if core is None:
        return ()
    for label in list(core):
        if label not in core:
            continue
        smaller = infeasible([item for item in core if item != label])
        if smaller is not None:
            core = smaller
    return tuple(sorted(core))


def solve(problem: Problem, config: SolverConfig, hint: dict[str, int] | None = None) -> Solution:
    started = time.monotonic()
    builder = _ModelBuilder(problem, config)
    model = builder.model
    if hint:
        builder.hint_assignment(hint)
    LOGGER.info(
        "Model: topics=%d max_modules=%d dependencies=%d constraints=%d affinity_pairs=%d",
        len(builder.topics),
        builder.size,
        len(problem.dependencies),
        len(problem.constraints),
        len(problem.affinity.pairs) if problem.affinity else 0,
    )
    terms = problem.objective.terms
    if problem.objective.mode == "weighted":
        objective = sum(
            (1 if TERM_SENSES[term.term] == "maximise" else -1)
            * term.weight / builder.terms[term.term][1]
            * builder.terms[term.term][0]
            for term in terms
        )
        model.maximize(objective)
        rounds = [None]
    else:
        rounds = list(terms)

    stages: list[Stage] = []
    assignment: dict[str, int] = {}
    status = cp_model.UNKNOWN
    all_optimal = True
    for term in rounds:
        if term is not None:
            expression, scale = builder.terms[term.term]
            model.clear_objective()
            if isinstance(expression, int):
                # Nothing to optimise (e.g. no soft constraints): only find a solution.
                pass
            elif TERM_SENSES[term.term] == "maximise":
                model.maximize(expression)
            else:
                model.minimize(expression)
        solver = _solver(config)
        status = solver.solve(model)
        LOGGER.info(
            "Stage %s: status=%s objective=%s time=%.1fs",
            "weighted" if term is None else term.term,
            solver.status_name(status),
            solver.objective_value if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else "-",
            solver.wall_time,
        )
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            break
        all_optimal = all_optimal and status == cp_model.OPTIMAL
        assignment = builder.assignment(solver)
        if term is None:
            stages.extend(
                Stage(item.term, solver.value(builder.terms[item.term][0]) / builder.terms[item.term][1], status == cp_model.OPTIMAL)
                for item in terms
            )
            continue
        if isinstance(expression, int):
            stages.append(Stage(term.term, expression / scale, True))
            continue
        value = round(solver.objective_value)
        stages.append(Stage(term.term, value / scale, status == cp_model.OPTIMAL))
        # Later stages may not worsen this term.
        model.add(expression >= value if TERM_SENSES[term.term] == "maximise" else expression <= value)
        builder.hint(solver)

    if not assignment:
        conflicts = explain_infeasibility(problem, config) if status == cp_model.INFEASIBLE else ()
        return Solution(
            status="INFEASIBLE" if status == cp_model.INFEASIBLE else "UNKNOWN",
            assignment={},
            conflicts=conflicts,
            wall_time=time.monotonic() - started,
        )
    return Solution(
        # A later stage that finds nothing keeps the previous stage's solution.
        status="OPTIMAL" if all_optimal and len(stages) == len(terms) else "FEASIBLE",
        assignment=assignment,
        stages=tuple(stages),
        wall_time=time.monotonic() - started,
    )
