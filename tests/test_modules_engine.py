from __future__ import annotations

import pathlib
import tempfile
import unittest

import yaml
from pydantic import ValidationError

from modules_engine.cli import build_parser
from modules_engine.io import (
    REVIEW_FORMAT,
    atomic_write_text,
    dependency_cycle,
    dump_yaml,
    load_hint,
    load_problem,
    load_review,
    validate_problem,
)
from modules_engine.models import (
    AffinityFile,
    ObjectiveTerm,
    Quantile,
    Constraint,
    ConstraintsFile,
    Dependency,
    Objective,
    PrefixedModule,
    PrefixedModulesFile,
    Problem,
    SolverConfig,
    TopicsFile,
)
from modules_engine.prepare import co_teaching_affinity, prepare_inputs
from modules_engine.render import MMDC, diagram_paths, render_command
from modules_engine.review import build_page, latest_run, load_run, resume_review, review_payload
from modules_engine.report import build_report, module_dependencies, render_mermaid, violations
from modules_engine.solver import resolve_threshold, solve, unrelated_pairs


CONFIG = SolverConfig(time_limit=10, workers=1, seed=0)


def make_problem(
    ects: dict[str, float],
    dependencies: list[tuple[str, str]] = (),
    constraints: list[dict] = (),
    terms: list[dict] | None = None,
    pairs: list[tuple[str, str, float]] | None = None,
    mode: str = "lexicographic",
    courses: dict[str, list[str]] | None = None,
    prefixed: list[dict] = (),
) -> Problem:
    terms = terms or [{"term": "soft_constraint_penalty"}, {"term": "module_count"}]
    return Problem(
        topics=TopicsFile.model_validate({
            "topics": {
                key: {"ects": value, "courses": (courses or {}).get(key, [])}
                for key, value in ects.items()
            },
        }),
        dependencies=tuple(Dependency.model_validate({"requires": a, "for": b}) for a, b in dependencies),
        constraints=tuple(Constraint.model_validate(item) for item in constraints),
        objective=Objective.model_validate({"mode": mode, "terms": terms}),
        affinity=AffinityFile(pairs=list(pairs)) if pairs is not None else None,
        prefixed=tuple(PrefixedModule.model_validate(item) for item in prefixed),
    )


def modules_of(assignment: dict[str, int]) -> set[frozenset[str]]:
    groups: dict[int, set[str]] = {}
    for topic, module in assignment.items():
        groups.setdefault(module, set()).add(topic)
    return {frozenset(group) for group in groups.values()}


class InputValidationTest(unittest.TestCase):
    def test_constraint_needs_exactly_one_kind(self) -> None:
        with self.assertRaises(ValidationError):
            Constraint.model_validate({"id": "c", "together": ["a", "b"], "apart": ["a", "b"]})
        with self.assertRaises(ValidationError):
            Constraint.model_validate({"id": "c"})
        with self.assertRaises(ValidationError):
            Constraint.model_validate({"id": "c", "together": ["a", "b"], "hard": {"min": True}})

    def test_constraint_ids_unique_and_single_size_constraint(self) -> None:
        with self.assertRaisesRegex(ValidationError, "duplicate constraint ids"):
            ConstraintsFile.model_validate({"constraints": [
                {"id": "c", "together": ["a", "b"]},
                {"id": "c", "apart": ["a", "b"]},
            ]})
        with self.assertRaisesRegex(ValidationError, "at most one module_ects"):
            ConstraintsFile.model_validate({"constraints": [
                {"id": "s1", "module_ects": {"min": 1}},
                {"id": "s2", "module_ects": {"max": 2}},
            ]})

    def test_cohesion_and_related_only_need_affinity(self) -> None:
        with self.assertRaises(ValidationError):
            ObjectiveTerm.model_validate({"term": "module_count", "baseline": 0.2})
        with self.assertRaises(ValidationError):
            Constraint.model_validate({"id": "r", "related_only": False})
        problem = make_problem(
            {"a": 1, "b": 1},
            constraints=[{"id": "r", "related_only": True}],
            terms=[{"term": "cohesion"}],
        )
        with self.assertRaisesRegex(ValueError, "objective term cohesion, constraint r need an affinity file"):
            validate_problem(problem)

    def test_thresholds_resolve_against_listed_scores(self) -> None:
        problem = make_problem(
            {"a": 1, "b": 1, "c": 1, "d": 1},
            pairs=[("a", "b", 0.1), ("a", "c", 0.2), ("a", "d", 0.3), ("b", "c", 0.4)],
        )
        self.assertAlmostEqual(resolve_threshold(Quantile(quantile=0.5), problem), 0.25)
        self.assertEqual(resolve_threshold(Quantile(quantile=1), problem), 0.4)
        self.assertEqual(resolve_threshold(0.7, problem), 0.7)
        self.assertEqual(resolve_threshold(None, problem), 0.0)
        constraint = Constraint.model_validate({"id": "r", "related_only": {"min_score": {"quantile": 0.5}}})
        # Below the median, or missing from the affinity file.
        self.assertEqual(
            unrelated_pairs(problem, constraint),
            [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")],
        )

    def test_unknown_topics_are_reported_together(self) -> None:
        problem = make_problem(
            {"a": 1, "b": 1},
            dependencies=[("a", "x")],
            constraints=[{"id": "c", "together": ["b", "y"]}],
            pairs=[("a", "z", 0.5)],
        )
        with self.assertRaises(ValueError) as raised:
            validate_problem(problem)
        message = str(raised.exception)
        self.assertIn("['x']", message)
        self.assertIn("constraint c", message)
        self.assertIn("['z']", message)

    def test_dependency_cycle_is_rejected(self) -> None:
        dependencies = [Dependency.model_validate({"requires": a, "for": b}) for a, b in [("a", "b"), ("b", "c"), ("c", "a")]]
        self.assertEqual(dependency_cycle(["a", "b", "c", "d"], dependencies), ["a", "b", "c", "a"])
        self.assertIsNone(dependency_cycle(["a", "b", "c"], dependencies[:2]))
        with self.assertRaisesRegex(ValueError, "cycle"):
            validate_problem(make_problem({"a": 1, "b": 1, "c": 1}, dependencies=[("a", "b"), ("b", "c"), ("c", "a")]))

    def test_load_problem_reads_optional_affinity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "topics.yml").write_text(yaml.safe_dump({"topics": {"a": {"ects": 1}, "b": {"ects": 1}}}))
            (root / "dependencies.yml").write_text(yaml.safe_dump({"dependencies": [{"requires": "a", "for": "b"}]}))
            (root / "constraints.yml").write_text(yaml.safe_dump({"constraints": []}))
            (root / "affinity.yml").write_text(yaml.safe_dump({"pairs": [["a", "b", 0.5]]}))
            (root / "objective.yml").write_text(yaml.safe_dump({"terms": [{"term": "cohesion"}]}))
            paths = [root / f"{name}.yml" for name in ("topics", "dependencies", "constraints", "objective")]
            with self.assertRaisesRegex(ValueError, "need an affinity file"):
                load_problem(*paths)
            problem = load_problem(*paths, root / "affinity.yml")
        self.assertEqual(problem.dependencies[0].for_, "b")
        self.assertEqual(problem.affinity.pairs, [("a", "b", 0.5)])


class SolverTest(unittest.TestCase):
    def test_hard_constraints_and_dependency_order(self) -> None:
        problem = make_problem(
            {"a": 1, "b": 1, "c": 1, "d": 1},
            dependencies=[("c", "a")],
            constraints=[
                {"id": "size", "module_ects": {"max": 2}},
                {"id": "ab", "together": ["a", "b"]},
                {"id": "cd", "apart": ["c", "d"]},
            ],
        )
        solution = solve(problem, CONFIG)
        self.assertEqual(solution.status, "OPTIMAL")
        assignment = solution.assignment
        self.assertEqual(assignment["a"], assignment["b"])
        self.assertNotEqual(assignment["c"], assignment["d"])
        self.assertLessEqual(assignment["c"], assignment["a"])
        # Two modules of 2 ECTS are impossible: c and d are apart and a, b fill one.
        self.assertEqual(len(modules_of(assignment)), 3)
        self.assertEqual(violations(problem, assignment), [])

    def test_cohesion_groups_high_affinity_pairs(self) -> None:
        problem = make_problem(
            {"a": 1, "b": 1, "c": 1, "d": 1},
            constraints=[{"id": "size", "module_ects": {"max": 2}}],
            terms=[{"term": "cohesion"}],
            pairs=[("a", "c", 1.0), ("b", "d", 1.0), ("a", "b", 0.2)],
        )
        solution = solve(problem, CONFIG)
        self.assertEqual(modules_of(solution.assignment), {frozenset("ac"), frozenset("bd")})
        self.assertEqual(solution.stages[0].value, 2.0)

    def test_soft_size_and_together_are_penalised_not_infeasible(self) -> None:
        problem = make_problem(
            {"a": 1, "b": 1, "c": 1, "d": 0.5},
            dependencies=[("a", "d"), ("b", "d"), ("c", "d")],
            constraints=[
                {"id": "size", "module_ects": {"min": 2, "max": 2}, "hard": {"min": False, "max": True}, "weight": 3},
                {"id": "abc", "together": ["a", "b", "c"], "hard": False, "weight": 1},
                {"id": "da", "apart": ["d", "a"], "hard": False, "weight": 10},
            ],
        )
        solution = solve(problem, CONFIG)
        found = {item["constraint"]: item for item in violations(problem, solution.assignment)}
        # abc must be split; d is kept away from a; the cheapest result splits
        # abc once and leaves modules whose shortfall totals 0.5 ECTS.
        self.assertEqual(found["abc"]["amount"], 1)
        self.assertNotIn("da", found)
        self.assertEqual(sum(item["penalty"] for item in found.values()), 1 + 3 * 0.5)
        self.assertEqual(solution.stages[0].value, 2.5)

    def test_infeasibility_names_conflicting_hard_constraints(self) -> None:
        problem = make_problem(
            {"a": 1, "b": 1, "c": 1, "d": 1},
            dependencies=[("a", "d")],
            constraints=[
                {"id": "size", "module_ects": {"min": 1, "max": 2}},
                {"id": "abc", "together": ["a", "b", "c"]},
                {"id": "cd", "apart": ["c", "d"]},
            ],
        )
        solution = solve(problem, CONFIG)
        self.assertEqual(solution.status, "INFEASIBLE")
        self.assertEqual(solution.conflicts, ("constraint abc", "constraint size (max)"))
        report = build_report(problem, solution, CONFIG, {})
        self.assertEqual(report["conflicting_hard_constraints"], ["constraint abc", "constraint size (max)"])

    def test_cohesion_baseline_makes_weak_pairs_costly(self) -> None:
        def solve_with(baseline: float | None) -> set[frozenset[str]]:
            problem = make_problem(
                {"a": 1, "b": 1, "c": 1, "d": 1},
                terms=[{"term": "cohesion", **({"baseline": baseline} if baseline is not None else {})}],
                pairs=[("a", "b", 0.9), ("c", "d", 0.9), ("a", "c", 0.1)],
            )
            solution = solve(problem, CONFIG)
            report = build_report(problem, solution, CONFIG, {})
            self.assertAlmostEqual(report["metrics"]["cohesion"], solution.stages[0].value)
            return modules_of(solution.assignment)

        # Without a baseline extra pairs never hurt, so one module scores most.
        self.assertEqual(solve_with(None), {frozenset("abcd")})
        self.assertEqual(solve_with(0.5), {frozenset("ab"), frozenset("cd")})

    def test_related_only_hard_conflicts_with_together(self) -> None:
        problem = make_problem(
            {"a": 1, "b": 1, "c": 1},
            constraints=[
                {"id": "related", "related_only": True},
                {"id": "bc", "together": ["b", "c"]},
            ],
            pairs=[("a", "b", 0.9)],
        )
        solution = solve(problem, CONFIG)
        self.assertEqual(solution.conflicts, ("constraint bc", "constraint related"))

    def test_related_only_soft_is_traded_against_size(self) -> None:
        problem = make_problem(
            {"a": 1, "b": 1, "c": 1},
            constraints=[
                {"id": "size", "module_ects": {"min": 3}, "hard": False, "weight": 1},
                {"id": "related", "related_only": True, "hard": False, "weight": 0.25},
            ],
            pairs=[("a", "b", 0.9)],
        )
        solution = solve(problem, CONFIG)
        # One module costs 2 unrelated pairs x 0.25; splitting off c costs 3 missing ECTS.
        self.assertEqual(modules_of(solution.assignment), {frozenset("abc")})
        [found] = violations(problem, solution.assignment)
        self.assertEqual((found["constraint"], found["amount"], found["penalty"]), ("related", 2, 0.5))
        self.assertEqual(found["detail"], {"m01": [["a", "c"], ["b", "c"]]})

    def test_weighted_mode_reports_every_term(self) -> None:
        problem = make_problem(
            {"a": 1, "b": 1, "c": 1},
            constraints=[{"id": "size", "module_ects": {"max": 3}}],
            terms=[
                {"term": "cohesion", "weight": 1},
                {"term": "module_count", "weight": 2},
            ],
            pairs=[("a", "b", 1.0)],
            mode="weighted",
        )
        solution = solve(problem, CONFIG)
        values = {stage.term: stage.value for stage in solution.stages}
        # One module of three topics: cohesion 1, one module.
        self.assertEqual(values, {"cohesion": 1.0, "module_count": 1.0})

    def test_max_modules_bounds_the_partition(self) -> None:
        problem = make_problem({"a": 1, "b": 1, "c": 1}, constraints=[{"id": "cd", "apart": ["a", "b", "c"]}])
        self.assertEqual(solve(problem, SolverConfig(time_limit=10, workers=1, max_modules=2)).status, "INFEASIBLE")


class PrefixedModulesTest(unittest.TestCase):
    def test_validation(self) -> None:
        with self.assertRaisesRegex(ValidationError, "kept for unnamed modules"):
            PrefixedModule.model_validate({"key": "m03", "topics": ["a"]})
        with self.assertRaisesRegex(ValidationError, "topics in several pre-fixed modules"):
            PrefixedModulesFile.model_validate({"modules": [
                {"key": "one", "topics": ["a", "b"]},
                {"key": "two", "topics": ["b"]},
            ]})
        with self.assertRaisesRegex(ValueError, "pre-fixed module one references unknown topics"):
            validate_problem(make_problem({"a": 1}, prefixed=[{"key": "one", "topics": ["a", "x"]}]))

    def test_exclusive_modules_are_kept_as_they_are(self) -> None:
        # Module count alone would merge everything into one module.
        problem = make_problem(
            {"a": 1, "b": 1, "c": 1, "d": 1},
            prefixed=[{"key": "ab", "topics": ["a", "b"]}, {"key": "lone-c", "topics": ["c"], "exclusive": False}],
        )
        solution = solve(problem, CONFIG)
        # d may join the non-exclusive lone-c, but never ab.
        self.assertEqual(modules_of(solution.assignment), {frozenset("ab"), frozenset("cd")})
        report = build_report(problem, solution, CONFIG, {})
        self.assertEqual(
            [(module["key"], module.get("prefixed", False)) for module in report["modules"]],
            [("ab", True), ("lone-c", True)],
        )
        self.assertIn(":::prefixed", render_mermaid(report))
        self.assertIn("ab · 2 ECTS · fixed", render_mermaid(report))

    def test_conflicts_with_hard_constraints_are_named(self) -> None:
        problem = make_problem(
            {"a": 2, "b": 2, "c": 2},
            constraints=[{"id": "size", "module_ects": {"max": 4}}],
            prefixed=[{"key": "big", "topics": ["a", "b", "c"]}],
        )
        solution = solve(problem, CONFIG)
        self.assertEqual(solution.conflicts, ("constraint size (max)", "pre-fixed module big"))

    def test_load_problem_reads_prefixed_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "topics.yml").write_text(yaml.safe_dump({"topics": {"a": {"ects": 1}, "b": {"ects": 1}}}))
            (root / "dependencies.yml").write_text(yaml.safe_dump({"dependencies": []}))
            (root / "constraints.yml").write_text(yaml.safe_dump({"constraints": []}))
            (root / "objective.yml").write_text(yaml.safe_dump({"terms": [{"term": "module_count"}]}))
            (root / "prefixed.yml").write_text(yaml.safe_dump({"modules": [{"key": "pair", "topics": ["a", "b"]}]}))
            paths = [root / f"{name}.yml" for name in ("topics", "dependencies", "constraints", "objective")]
            problem = load_problem(*paths, prefixed_path=root / "prefixed.yml")
        self.assertEqual([module.key for module in problem.prefixed], ["pair"])


class ReviewTest(unittest.TestCase):
    def write_run(self, root: pathlib.Path) -> pathlib.Path:
        """A solved two-module run whose modules.yml records its inputs."""
        (root / "topics.yml").write_text(yaml.safe_dump({
            "courses": {"C1": {"title": "Course one"}},
            "topics": {"a": {"ects": 1, "courses": ["C1"]}, "b": {"ects": 1, "courses": ["C1"]}, "c": {"ects": 1}},
        }))
        (root / "dependencies.yml").write_text(yaml.safe_dump({"dependencies": [{"requires": "a", "for": "c"}]}))
        (root / "constraints.yml").write_text(yaml.safe_dump({"constraints": [
            {"id": "size", "module_ects": {"min": 2, "max": 2}, "hard": {"min": False, "max": True}},
            {"id": "related", "related_only": {"min_score": {"quantile": 0.5}}, "hard": False},
        ]}))
        (root / "objective.yml").write_text(yaml.safe_dump({"terms": [{"term": "soft_constraint_penalty"}, {"term": "cohesion", "baseline": 0.1}]}))
        (root / "affinity.yml").write_text(yaml.safe_dump({"pairs": [["a", "b", 0.9], ["b", "c", 0.2]]}))
        names = ("topics", "dependencies", "constraints", "objective", "affinity")
        inputs = {name: str(root / f"{name}.yml") for name in names}
        problem = load_problem(*(pathlib.Path(inputs[name]) for name in names))
        run = root / "run"
        atomic_write_text(run / "modules.yml", dump_yaml(build_report(problem, solve(problem, CONFIG), CONFIG, inputs)))
        return run

    def test_payload_resolves_thresholds_and_suggests_resolve(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = self.write_run(pathlib.Path(directory))
            problem, report = load_run(run)
            payload = review_payload(problem, report, run)
            self.assertEqual(load_hint(run / "modules.yml"), {"a": 0, "b": 0, "c": 1})
        self.assertEqual([module["topics"] for module in payload["modules"]], [["a", "b"], ["c"]])
        self.assertEqual([module["locked"] for module in payload["modules"]], [False, False])
        self.assertEqual(payload["proposal"][0], {"key": "m01", "topics": ["a", "b"], "prefixed": False})
        self.assertEqual(payload["format"], REVIEW_FORMAT)
        related = next(item for item in payload["constraints"] if item["kind"] == "related_only")
        self.assertAlmostEqual(related["min_score"], 0.55)
        size = next(item for item in payload["constraints"] if item["kind"] == "module_ects")
        self.assertEqual(size["hard"], {"min": False, "max": True})
        self.assertEqual(payload["baseline"], 0.1)
        self.assertEqual(payload["courses"], {"C1": "Course one"})
        self.assertIn(f"--review {run / 'review.yml'}", payload["solve_command"])
        self.assertIn(f"--output {run.parent / 'run-revised'}", payload["solve_command"])

    def saved_review(self, payload: dict, modules: list[dict]) -> str:
        """A review file as the page saves it: the arrangement, then the run's data."""
        data = {key: value for key, value in payload.items() if key not in ("format", "modules", "resumed_from")}
        return yaml.safe_dump({"format": REVIEW_FORMAT, "modules": modules, **data}, sort_keys=False)

    def test_resume_review_of_the_same_proposal_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = self.write_run(pathlib.Path(directory))
            payload = review_payload(*load_run(run), run)
            review = run / "review.yml"
            review.write_text(self.saved_review(payload, [{"key": "pair", "locked": True, "topics": ["a", "c"]}]))
            warnings = resume_review(payload, review)
            # b is not placed by the review, so it is gathered in "unassigned".
            self.assertEqual(payload["modules"], [
                {"key": "pair", "topics": ["a", "c"], "locked": True},
                {"key": "unassigned", "topics": ["b"], "locked": False},
            ])
            self.assertIn("does not place 1 topic", warnings[0])
            self.assertEqual(payload["resumed_from"], str(review))
            fresh = review_payload(*load_run(run), run)
            fresh["fingerprint"] = "other"
            self.assertIn("another proposal", resume_review(fresh, review)[0])
            self.assertEqual([module["key"] for module in fresh["modules"]], ["m01", "m02"])

    def test_load_review_gives_prefixed_modules_and_hint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = self.write_run(pathlib.Path(directory))
            payload = review_payload(*load_run(run), run)
            review = run / "review.yml"
            review.write_text(self.saved_review(payload, [
                {"key": "m02", "locked": False, "topics": ["c"]},
                {"key": "pair", "locked": True, "topics": ["a", "b"]},
            ]))
            prefixed, hint = load_review(review)
            self.assertEqual([(module.key, module.topics, module.exclusive) for module in prefixed], [("pair", ["a", "b"], True)])
            self.assertEqual(hint, {"c": 0, "a": 1, "b": 1})
            review.write_text(self.saved_review(payload, [{"key": "m01", "locked": True, "topics": ["a"]}]))
            with self.assertRaisesRegex(ValueError, "kept for unnamed modules"):
                load_review(review)
            review.write_text(self.saved_review(payload, [{"key": "x", "topics": ["a"]}, {"key": "y", "topics": ["a"]}]))
            with self.assertRaisesRegex(ValueError, "several modules"):
                load_review(review)
            review.write_text(yaml.safe_dump({"modules": []}))
            with self.assertRaisesRegex(ValueError, "not a review file"):
                load_review(review)

    def test_page_inlines_data_and_escapes_script_ends(self) -> None:
        payload = {"run": "r", "topics": {"a": {"description": "x </script> y"}}}
        page = build_page(payload, "var mermaid = '</script>';", "var jsyaml = {};")
        self.assertEqual(page.count("</script>"), 4)
        self.assertIn("const INITIAL_SESSION = ", page)
        self.assertIn("var jsyaml = {};", page)
        self.assertIn("var mermaid = '<\\/script>';", page)

    def test_latest_run_skips_runs_without_modules(self) -> None:
        import os

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            for name, payload, stamp in (
                ("old", {"modules": [{"key": "m01", "topics": ["a"]}]}, 1000),
                ("failed", {"status": "INFEASIBLE"}, 3000),
                ("new", {"modules": [{"key": "m01", "topics": ["a"]}]}, 2000),
            ):
                (root / name).mkdir()
                path = root / name / "modules.yml"
                path.write_text(yaml.safe_dump(payload))
                os.utime(path, (stamp, stamp))
            self.assertEqual(latest_run(root), root / "new")
            with self.assertRaisesRegex(ValueError, "No module proposal"):
                latest_run(root / "failed")

    def test_hint_starts_from_an_earlier_assignment(self) -> None:
        problem = make_problem({"a": 1, "b": 1, "c": 1}, constraints=[{"id": "size", "module_ects": {"max": 2}}])
        solution = solve(problem, CONFIG, hint={"a": 0, "c": 0, "b": 1})
        self.assertEqual(solution.status, "OPTIMAL")
        self.assertEqual(len(modules_of(solution.assignment)), 2)


class ReportTest(unittest.TestCase):
    def test_module_dependencies_are_reduced_and_chained(self) -> None:
        problem = make_problem(
            {"a": 1, "b": 1, "c": 1, "d": 1},
            dependencies=[("a", "b"), ("b", "c"), ("a", "c"), ("c", "d")],
        )
        edges, depth = module_dependencies(problem, {"a": 0, "b": 1, "c": 2, "d": 2})
        self.assertEqual(
            edges,
            [
                {"requires": "m01", "for": "m02", "via": [["a", "b"]]},
                {"requires": "m02", "for": "m03", "via": [["b", "c"]]},
            ],
        )
        self.assertEqual(depth, 3)

    def test_report_and_mermaid(self) -> None:
        problem = make_problem(
            {"a": 1, "b": 1, "c": 2.5},
            dependencies=[("a", "c")],
            constraints=[{"id": "size", "module_ects": {"min": 2, "max": 2}, "hard": {"min": False, "max": False}}],
            courses={"a": ["X"], "b": ["X"], "c": ["X", "Y"]},
        )
        solution = solve(problem, CONFIG)
        report = build_report(problem, solution, CONFIG, {"topics": "topics.yml"})
        self.assertEqual([module["topics"] for module in report["modules"]], [["a", "b"], ["c"]])
        self.assertEqual(report["metrics"]["soft_constraint_penalty"], 0.5)
        self.assertEqual(
            report["objective"]["stages"],
            [
                {"term": "soft_constraint_penalty", "value_at_stage": 0.5, "optimal": True, "final": 0.5},
                {"term": "module_count", "value_at_stage": 2.0, "optimal": True, "final": 2},
            ],
        )
        self.assertEqual(report["source_courses"]["X"]["modules"], 2)
        self.assertEqual(report["violations"][0]["constraint"], "size (max)")
        mermaid = render_mermaid(report)
        self.assertIn("M0 --> M1", mermaid)
        self.assertIn('M1["m02 · 2.5 ECTS<br/>c"]:::flagged', mermaid)


class PrepareTest(unittest.TestCase):
    def test_co_teaching_affinity_is_jaccard(self) -> None:
        memberships = {"a": {"1", "2"}, "b": {"2", "3"}, "c": {"4"}}
        self.assertEqual(co_teaching_affinity(memberships, ["a", "b", "c"], 0.0), [["a", "b", 0.333]])
        self.assertEqual(co_teaching_affinity(memberships, ["a", "b", "c"], 0.5), [])

    def test_prepare_inputs_selects_course_topics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempt = pathlib.Path(directory) / "attempt"
            attempt.mkdir()
            (attempt / "topics-global.yml").write_text(yaml.safe_dump({
                "topics": {"a": "A.", "b": "B.", "c": "C."},
                "topic_weights": {
                    "a": {"ects": 1.0, "per_course": {"00819": 1.0, "11": 1.0}},
                    "b": {"ects": 0.5, "per_course": {"11": 0.5}},
                    "c": {"ects": 2.0, "per_course": {"22": 2.0}},
                },
                "course_credit_check": {"00819": {"credits": 6.0}, "11": {"credits": 6.0}, "22": {"credits": 3.0}},
            }))
            for course_id, name in (("00819", "One"), ("11", "Two"), ("22", "Three")):
                (attempt / f"topics-of-course-{course_id}.yml").write_text(
                    yaml.safe_dump({"course": {"id": course_id, "name": name}})
                )
            output = pathlib.Path(directory) / "inputs"
            prepare_inputs(attempt, output, ("11",))
            topics = TopicsFile.model_validate(yaml.safe_load((output / "topics.yml").read_text()))
            affinity = AffinityFile.model_validate(yaml.safe_load((output / "affinity.yml").read_text()))
            with self.assertRaisesRegex(ValueError, "Unknown course ids"):
                prepare_inputs(attempt, output, ("99",))
        self.assertEqual(sorted(topics.topics), ["a", "b"])
        # Only selected courses are listed, but affinity uses the whole corpus.
        self.assertEqual(topics.topics["a"].courses, ["11"])
        self.assertEqual(topics.courses["11"].title, "Two")
        self.assertEqual(affinity.pairs, [("a", "b", 0.5)])


class RenderTest(unittest.TestCase):
    def test_diagram_paths_from_files_and_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            for run in ("run-a", "run-b"):
                (root / run).mkdir()
                (root / run / "modules.mmd").write_text("flowchart TB\n")
            (root / "notes.txt").write_text("")
            found = diagram_paths([root, root / "run-a" / "modules.mmd"])
            self.assertEqual(found, [root / "run-a" / "modules.mmd", root / "run-b" / "modules.mmd"])
            with self.assertRaisesRegex(ValueError, "Not a .mmd file"):
                diagram_paths([root / "notes.txt"])
            (root / "empty").mkdir()
            with self.assertRaisesRegex(ValueError, "No modules.mmd"):
                diagram_paths([root / "empty"])

    def test_render_command_per_format(self) -> None:
        path = pathlib.Path("out/modules.mmd")
        pdf = render_command(path, "pdf")
        self.assertEqual((pdf[0], pdf[-1]), (str(MMDC), "out/modules.pdf"))
        self.assertEqual(render_command(path, "png")[-5:], ["out/modules.png", "--scale", "2", "-b", "white"])
        self.assertEqual(render_command(path, "svg")[-3:], ["out/modules.svg", "-b", "white"])


class CliTest(unittest.TestCase):
    def test_review_replaces_prefixed_and_hint(self) -> None:
        import contextlib
        import io as text_io
        from unittest.mock import patch

        from modules_engine import cli

        argv = ["modules_engine", "solve", "t.yml", "d.yml", "c.yml", "o.yml", "--output", "out", "--review", "r.yml", "--hint", "h.yml"]
        with patch("sys.argv", argv), contextlib.redirect_stderr(text_io.StringIO()) as stderr, self.assertRaises(SystemExit):
            cli.main()
        self.assertIn("--review replaces --prefixed and --hint", stderr.getvalue())

    def test_solve_arguments(self) -> None:
        args = build_parser().parse_args(["solve", "t.yml", "d.yml", "c.yml", "o.yml", "--output", "out", "--max-modules", "4"])
        self.assertEqual(args.command, "solve")
        self.assertEqual(args.max_modules, 4)
        self.assertEqual(args.time_limit, 60.0)


if __name__ == "__main__":
    unittest.main()
