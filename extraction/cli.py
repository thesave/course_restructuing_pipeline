from __future__ import annotations

import argparse
import logging
import pathlib
import sys

from pydantic import ValidationError

from extraction.claude_cli import ClaudeCliEndpoint
from extraction.client import Endpoint, EndpointError, RequestError
from extraction.manual import ingest_reply, write_requests
from extraction.prompt import retry_message
from extraction.models import EndpointsFile, RunConfig
from extraction.run import Extractor, load_evidence, select_courses
from modules_engine.io import load_yaml_mapping

DEFAULT_CACHE = pathlib.Path("data/.cache/extraction")


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer: {value}")
    return number


def load_endpoints(path: pathlib.Path) -> EndpointsFile:
    try:
        return EndpointsFile.model_validate(load_yaml_mapping(path, "endpoints file"))
    except ValidationError as error:
        raise ValueError(f"Invalid endpoints file {path}: {error}") from error


def make_endpoints(settings: EndpointsFile) -> list:
    """One backend per endpoint of the file, by provider."""
    return [
        ClaudeCliEndpoint(spec, settings.model, settings.timeout, settings.request)
        if spec.provider == "claude-cli"
        else Endpoint(spec, settings.model, settings.timeout)
        for spec in settings.endpoints
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract the topics of each course from its evidence with a model on OpenAI-compatible servers or headless Claude Code."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check", help="List the models each endpoint serves")
    check.add_argument("endpoints", type=pathlib.Path, help="endpoints file (see extraction/endpoints.example.yml)")

    run = commands.add_parser("run", help="Extract topics, spreading courses over the endpoints")
    run.add_argument("evidence_dir", type=pathlib.Path, help="directory of evidence-*.yml files")
    run.add_argument("output_dir", type=pathlib.Path)
    run.add_argument("--endpoints", type=pathlib.Path, required=True, help="endpoints file")
    run.add_argument("--course", action="append", default=[], help="extract only this course; repeatable")
    run.add_argument("--service", action="store_true", help="extract only service courses")
    run.add_argument("--limit", type=_positive_int, help="extract at most this many of the selected courses")
    run.add_argument("--max-attempts", type=_positive_int, default=3, help="requests per course, retries included")
    run.add_argument("--force", action="store_true", help="redo courses that already have a topics file")
    run.add_argument("--cache-dir", type=pathlib.Path, default=DEFAULT_CACHE)

    prompt = commands.add_parser("prompt", help="Write each course's request for a model outside this program")
    prompt.add_argument("evidence_dir", type=pathlib.Path)
    prompt.add_argument("output_dir", type=pathlib.Path)
    prompt.add_argument("--course", action="append", default=[], help="only this course; repeatable")
    prompt.add_argument("--service", action="store_true", help="only service courses")
    prompt.add_argument("--limit", type=_positive_int, help="at most this many of the selected courses")

    ingest = commands.add_parser("ingest", help="Check a reply written outside this program and store its topics")
    ingest.add_argument("evidence_dir", type=pathlib.Path)
    ingest.add_argument("output_dir", type=pathlib.Path)
    ingest.add_argument("course")
    ingest.add_argument("reply", type=pathlib.Path, help="file with the JSON reply")
    ingest.add_argument("--model", required=True, help="model that wrote the reply, recorded in the topics file")
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(threadName)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = build_parser()
    args = parser.parse_args()
    if args.command in ("prompt", "ingest"):
        try:
            evidences = load_evidence(args.evidence_dir)
            if args.command == "prompt":
                selected = select_courses(evidences, tuple(args.course), args.service, args.limit)
                for path in write_requests(evidences, selected, args.output_dir):
                    print(path)
                return
            errors = ingest_reply(evidences, args.output_dir, args.course, args.reply, args.model)
        except (ValueError, OSError) as error:
            parser.error(str(error))
        if errors:
            print(retry_message(errors)["content"])
            sys.exit(1)
        print(f"ok: {args.output_dir / f'topics-{args.course}.yml'}")
        return
    try:
        settings = load_endpoints(args.endpoints)
        endpoints = make_endpoints(settings)
        if args.command == "check":
            reachable = True
            for endpoint in endpoints:
                try:
                    print(f"{endpoint.url}: {', '.join(endpoint.models()) or 'no models listed'}")
                except (EndpointError, RequestError) as error:
                    reachable = False
                    print(f"{endpoint.url}: unreachable ({error})")
            sys.exit(0 if reachable else 1)
        evidences = load_evidence(args.evidence_dir)
        selected = select_courses(evidences, tuple(args.course), args.service, args.limit)
    except ValueError as error:
        parser.error(str(error))
    extractor = Extractor(
        settings, args.output_dir, args.cache_dir, RunConfig(max_attempts=args.max_attempts, force=args.force)
    )
    results = extractor.run(
        evidences, selected, [(endpoint, spec.workers) for endpoint, spec in zip(endpoints, settings.endpoints)]
    )
    failed = sorted(course for course, result in results.items() if result.status == "failed")
    statuses = [result.status for result in results.values()]
    print(
        ", ".join(f"{statuses.count(status)} {status}" for status in ("ok", "copied", "skipped", "failed") if status in statuses)
        + f"; see {args.output_dir / 'run.yml'}"
    )
    if failed:
        print(f"failed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
