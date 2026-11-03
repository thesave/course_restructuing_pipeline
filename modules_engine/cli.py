from __future__ import annotations

import argparse
import dataclasses
import logging
import pathlib
import sys

from modules_engine.io import atomic_write_text, dump_yaml, load_hint, load_problem, load_review, validate_problem
from modules_engine.models import SolverConfig
from modules_engine.prepare import prepare_inputs
from modules_engine.report import build_report, render_mermaid
from modules_engine.solver import solve


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer: {value}")
    return number


def _positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be positive: {value}")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Partition topics into modules under explicit constraints with a CP-SAT solver."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser(
        "prepare",
        help="Write topics.yml and affinity.yml from a restructuring attempt",
    )
    prepare.add_argument("attempt_dir", type=pathlib.Path, help="data/restructuring/attempt-* directory")
    prepare.add_argument("output_dir", type=pathlib.Path)
    prepare.add_argument(
        "--course",
        action="append",
        default=[],
        help="Keep only the topics of this course id; repeatable (default: all courses)",
    )
    prepare.add_argument(
        "--min-affinity",
        type=float,
        default=0.0,
        help="Drop affinity pairs scoring below this value",
    )

    solve_parser = commands.add_parser("solve", help="Propose modules and their dependencies")
    solve_parser.add_argument("topics", type=pathlib.Path)
    solve_parser.add_argument("dependencies", type=pathlib.Path)
    solve_parser.add_argument("constraints", type=pathlib.Path)
    solve_parser.add_argument("objective", type=pathlib.Path)
    solve_parser.add_argument(
        "--affinity",
        type=pathlib.Path,
        help="Pairwise topic affinity, required by the cohesion term and related_only constraints",
    )
    solve_parser.add_argument(
        "--review",
        type=pathlib.Path,
        help="Review file saved by the review page: keeps its locked modules and starts from its arrangement "
        "(replaces --prefixed and --hint)",
    )
    solve_parser.add_argument(
        "--hint",
        type=pathlib.Path,
        help="modules.yml of an earlier run to start from (e.g. the proposal just reviewed)",
    )
    solve_parser.add_argument(
        "--prefixed",
        type=pathlib.Path,
        help="Pre-fixed modules from human review (e.g. saved from the review page), kept as they are",
    )
    solve_parser.add_argument("--output", type=pathlib.Path, required=True, help="Directory for modules.yml and modules.mmd")
    solve_parser.add_argument(
        "--time-limit",
        type=_positive_float,
        default=60.0,
        help="Seconds per objective stage",
    )
    solve_parser.add_argument("--workers", type=_positive_int, default=8)
    solve_parser.add_argument("--seed", type=int, default=0)
    solve_parser.add_argument(
        "--max-modules",
        type=_positive_int,
        help="Upper bound on the number of modules (default: one per topic)",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            for path in prepare_inputs(args.attempt_dir, args.output_dir, tuple(args.course), args.min_affinity):
                print(path)
            return
        if args.review and (args.prefixed or args.hint):
            raise ValueError("--review replaces --prefixed and --hint")
        problem = load_problem(args.topics, args.dependencies, args.constraints, args.objective, args.affinity, args.prefixed)
        hint = load_hint(args.hint) if args.hint else None
        if args.review:
            prefixed, hint = load_review(args.review)
            problem = dataclasses.replace(problem, prefixed=prefixed)
            validate_problem(problem)
        config = SolverConfig(
            time_limit=args.time_limit,
            workers=args.workers,
            seed=args.seed,
            max_modules=args.max_modules,
        )
        solution = solve(problem, config, hint)
    except ValueError as error:
        parser.error(str(error))
    inputs = {
        name: str(getattr(args, name))
        for name in ("topics", "dependencies", "constraints", "objective", "affinity", "prefixed", "review")
        if getattr(args, name) is not None
    }
    report = build_report(problem, solution, config, inputs)
    yaml_path = args.output / "modules.yml"
    atomic_write_text(yaml_path, dump_yaml(report))
    print(yaml_path)
    if not solution.assignment:
        logging.error(
            "No proposal (%s); conflicting hard constraints: %s",
            solution.status,
            ", ".join(solution.conflicts) or "none identified",
        )
        sys.exit(1)
    mermaid_path = args.output / "modules.mmd"
    atomic_write_text(mermaid_path, render_mermaid(report))
    print(mermaid_path)


if __name__ == "__main__":
    main()
