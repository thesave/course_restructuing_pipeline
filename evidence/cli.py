from __future__ import annotations

import argparse
import pathlib

from evidence.prepare import prepare_evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the evidence of each course: citable syllabus items and course facts."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Write evidence-<course>.yml files and report.md")
    prepare.add_argument("courses_dir", type=pathlib.Path, help="directory of course-*.yml files")
    prepare.add_argument("output_dir", type=pathlib.Path)
    prepare.add_argument(
        "--overrides",
        type=pathlib.Path,
        help="reviewer corrections (default: OUTPUT_DIR/overrides.yml when it exists)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    overrides = args.overrides or args.output_dir / "overrides.yml"
    if args.overrides and not args.overrides.exists():
        parser.error(f"overrides file does not exist: {args.overrides}")
    try:
        written = prepare_evidence(args.courses_dir, args.output_dir, overrides)
    except ValueError as error:
        parser.error(str(error))
    print(f"{len(written) - 1} evidence files and {written[-1]}")


if __name__ == "__main__":
    main()
