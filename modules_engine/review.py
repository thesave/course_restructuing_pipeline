"""Interactive review page for a module proposal.

    .venv/bin/python -m modules_engine.review [RUN_DIR] [--review FILE] [--output PATH] [--open]

Without RUN_DIR, the most recent run under data/modules-engine/ is used;
--open opens the page in the default browser (review-modules.sh does both).

Reads RUN_DIR/modules.yml and the inputs it records, and writes a
self-contained HTML page (default RUN_DIR/review.html; Mermaid and js-yaml are
inlined, so it works offline). The page starts from the review file
RUN_DIR/review.yml (or --review FILE) when it was saved for the same
proposal, otherwise from the solver's proposal. In the page, topics are moved
between modules by drag and drop (or by selecting a topic and clicking a
module), new modules are created by dropping a topic on the "new module" card,
and modules are renamed and locked. ECTS, dependencies, violations, cohesion
and a Mermaid preview are recomputed after every change. The page opens and
saves review files (Save overwrites the opened file, Save as writes a new
one); `solve --review FILE` keeps a review's locked modules and re-solves the
rest, starting from its arrangement. Needs `npm install`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shlex
import sys
import webbrowser
from typing import Any

from modules_engine.io import REVIEW_FORMAT, atomic_write_text, load_problem, load_yaml_mapping, review_modules
from modules_engine.models import Problem
from modules_engine.solver import ECTS_SCALE, cohesion_baseline, resolve_threshold, scaled_ects


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parent.parent
TEMPLATE = pathlib.Path(__file__).with_name("review_template.html")
MERMAID = REPOSITORY_ROOT / "node_modules/mermaid/dist/mermaid.min.js"
JS_YAML = REPOSITORY_ROOT / "node_modules/js-yaml/dist/js-yaml.min.js"
RUNS_ROOT = REPOSITORY_ROOT / "data/modules-engine"
INPUT_NAMES = ("topics", "dependencies", "constraints", "objective", "affinity", "prefixed")


def _input_path(value: str) -> pathlib.Path:
    """Recorded input paths are relative to where `solve` ran, usually the repository root."""
    path = pathlib.Path(value)
    if path.exists() or path.is_absolute():
        return path
    return REPOSITORY_ROOT / path


def latest_run(root: pathlib.Path = RUNS_ROOT) -> pathlib.Path:
    """The most recently written run directory whose modules.yml has modules."""
    for path in sorted(root.rglob("modules.yml"), key=lambda path: path.stat().st_mtime, reverse=True):
        try:
            if load_yaml_mapping(path, "module proposal").get("modules"):
                return path.parent
        except ValueError:
            continue
    raise ValueError(f"No module proposal under {root}; run `modules_engine solve` first")


def load_run(run_dir: pathlib.Path) -> tuple[Problem, dict[str, Any]]:
    report = load_yaml_mapping(run_dir / "modules.yml", "module proposal")
    if not report.get("modules"):
        raise ValueError(f"{run_dir / 'modules.yml'} has no modules (status {report.get('status')})")
    inputs = {name: _input_path(value) for name, value in (report.get("inputs") or {}).items()}
    missing = [name for name in INPUT_NAMES[:4] if name not in inputs]
    if missing:
        raise ValueError(f"{run_dir / 'modules.yml'} does not record its inputs: {missing}")
    problem = load_problem(
        inputs["topics"],
        inputs["dependencies"],
        inputs["constraints"],
        inputs["objective"],
        inputs.get("affinity"),
        inputs.get("prefixed"),
    )
    return problem, report


def solve_command(inputs: dict[str, str], review_path: str, output: str) -> str:
    words = [".venv/bin/python", "-m", "modules_engine", "solve"]
    words += [inputs[name] for name in INPUT_NAMES[:4]]
    if "affinity" in inputs:
        words += ["--affinity", inputs["affinity"]]
    words += ["--review", review_path, "--output", output]
    return " ".join(shlex.quote(word) for word in words)


def review_payload(problem: Problem, report: dict[str, Any], run_dir: pathlib.Path) -> dict[str, Any]:
    """Everything the page needs, with thresholds already resolved so that the
    page evaluates violations and cohesion exactly as `report.py` does."""
    topics = problem.topics.topics
    constraints = []
    for item in problem.constraints:
        entry: dict[str, Any] = {"id": item.id, "kind": item.kind, "weight": item.weight}
        if item.module_ects is not None:
            entry.update(
                min=item.module_ects.min,
                max=item.module_ects.max,
                hard={"min": item.is_hard("min"), "max": item.is_hard("max")},
            )
        else:
            entry["hard"] = item.is_hard()
        if item.together is not None or item.apart is not None:
            entry["topics"] = item.together or item.apart
        if item.related_only is not None:
            entry["min_score"] = resolve_threshold(item.related_only.min_score, problem)
        constraints.append(entry)
    inputs = {name: str(value) for name, value in (report.get("inputs") or {}).items()}
    prefixed_path = str(pathlib.Path(inputs["topics"]).parent / "prefixed-modules.yml")
    review_path = str(run_dir / "review.yml")
    proposal = [
        {"key": module["key"], "topics": module["topics"], "prefixed": bool(module.get("prefixed"))}
        for module in report["modules"]
    ]
    return {
        "format": REVIEW_FORMAT,
        "run": str(run_dir),
        "status": report.get("status"),
        # Identifies the solver proposal a review file was made from.
        "fingerprint": hashlib.sha1(json.dumps(proposal, sort_keys=True).encode()).hexdigest()[:12],
        "topics": {
            key: {
                "description": spec.description,
                "ects": scaled_ects(spec.ects) / ECTS_SCALE,
                "courses": spec.courses,
            }
            for key, spec in sorted(topics.items())
        },
        "courses": {key: spec.title for key, spec in problem.topics.courses.items()},
        "dependencies": [[edge.requires, edge.for_] for edge in problem.dependencies],
        "constraints": constraints,
        "affinity": [list(pair) for pair in problem.affinity.pairs] if problem.affinity else None,
        "baseline": cohesion_baseline(problem),
        "proposal": proposal,
        # The arrangement under review; it starts as the solver's proposal.
        "modules": [
            {"key": module["key"], "topics": module["topics"], "locked": module["prefixed"]}
            for module in proposal
        ],
        "inputs": inputs,
        "review_path": review_path,
        "prefixed_path": prefixed_path,
        "solve_command": solve_command(inputs, review_path, str(run_dir.parent / f"{run_dir.name}-revised")),
    }


def resume_review(payload: dict[str, Any], session_path: pathlib.Path) -> list[str]:
    """Start the page from a saved review of the same proposal; returns warnings.

    A review saved for another proposal (the run was solved again) is ignored.
    Topics the review does not place are gathered in an unlocked module."""
    session = load_yaml_mapping(session_path, "review file")
    if session.get("fingerprint") != payload["fingerprint"]:
        return [f"{session_path} was saved for another proposal of this run; starting from the solver's proposal"]
    modules = review_modules(session, payload["topics"])
    placed = {topic for module in modules for topic in module["topics"]}
    missing = sorted(set(payload["topics"]) - placed)
    warnings = []
    if missing:
        modules.append({"key": "unassigned", "topics": missing, "locked": False})
        warnings.append(f"{session_path} does not place {len(missing)} topic(s); they are in module 'unassigned'")
    payload["modules"] = modules
    payload["resumed_from"] = str(session_path)
    return warnings


def _inline_script(source: str) -> str:
    # A literal "</script" would end the inline script early.
    return source.replace("</script", "<\\/script")


def build_page(payload: dict[str, Any], mermaid_source: str, yaml_source: str) -> str:
    template = TEMPLATE.read_text(encoding="utf-8")
    return (
        template
        .replace("/*@DATA@*/", _inline_script(f"const INITIAL_SESSION = {json.dumps(payload, ensure_ascii=False)};"))
        .replace("/*@MERMAID@*/", _inline_script(mermaid_source))
        .replace("/*@YAML@*/", _inline_script(yaml_source))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "run_dir",
        type=pathlib.Path,
        nargs="?",
        help="A `solve --output` directory containing modules.yml (default: the most recent run)",
    )
    parser.add_argument("--review", type=pathlib.Path, help="Review file to start from (default: RUN_DIR/review.yml if present)")
    parser.add_argument("--output", type=pathlib.Path, help="Default: RUN_DIR/review.html")
    parser.add_argument("--open", action="store_true", help="Open the page in the default browser")
    args = parser.parse_args()
    missing = [str(path) for path in (MERMAID, JS_YAML) if not path.exists()]
    if missing:
        parser.error(f"Not found: {', '.join(missing)}; run 'npm install' in {REPOSITORY_ROOT}")
    try:
        run_dir = args.run_dir or latest_run()
        problem, report = load_run(run_dir)
        payload = review_payload(problem, report, run_dir)
        session_path = args.review or run_dir / "review.yml"
        if args.review or session_path.exists():
            for warning in resume_review(payload, session_path):
                print(f"warning: {warning}", file=sys.stderr)
    except ValueError as error:
        parser.error(str(error))
    output = args.output or run_dir / "review.html"
    atomic_write_text(output, build_page(payload, MERMAID.read_text(encoding="utf-8"), JS_YAML.read_text(encoding="utf-8")))
    print(output)
    if args.open:
        webbrowser.open(output.resolve().as_uri())


if __name__ == "__main__":
    main()
