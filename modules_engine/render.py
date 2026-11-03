"""Render module proposal diagrams (modules.mmd) with mermaid-cli.

    .venv/bin/python -m modules_engine.render PATH [PATH ...] [--format pdf|png|svg]

Each PATH is a modules.mmd file or a directory searched recursively for them
(e.g. a `solve --output` directory). Writes modules.<format> next to each
input with the repository's mermaid-config.json: PDFs are cropped to the
diagram, PNGs rendered at twice the scale. Needs `npm install`.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parent.parent
MMDC = REPOSITORY_ROOT / "node_modules/.bin/mmdc"
FORMATS = ("pdf", "png", "svg")


def diagram_paths(paths: list[pathlib.Path]) -> list[pathlib.Path]:
    found: list[pathlib.Path] = []
    for path in paths:
        if path.is_dir():
            matches = sorted(path.rglob("modules.mmd"))
            if not matches:
                raise ValueError(f"No modules.mmd under {path}")
            found.extend(matches)
        elif path.suffix == ".mmd" and path.is_file():
            found.append(path)
        else:
            raise ValueError(f"Not a .mmd file or a directory: {path}")
    return list(dict.fromkeys(found))


def render_command(path: pathlib.Path, output_format: str) -> list[str]:
    command = [str(MMDC), "-q", "-c", str(REPOSITORY_ROOT / "mermaid-config.json"), "-i", str(path), "-o", str(path.with_suffix(f".{output_format}"))]
    if output_format == "png":
        command += ["--scale", "2"]
    if output_format != "pdf":
        # PDFs ignore the background colour.
        command += ["-b", "white"]
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", type=pathlib.Path, nargs="+", help="modules.mmd files or directories containing them")
    parser.add_argument("--format", choices=FORMATS, default="pdf")
    args = parser.parse_args()
    if not MMDC.exists():
        parser.error(f"mermaid-cli not found; run 'npm install' in {REPOSITORY_ROOT}")
    try:
        paths = diagram_paths(args.paths)
    except ValueError as error:
        parser.error(str(error))
    for path in paths:
        subprocess.run(render_command(path, args.format), check=True)
        print(path.with_suffix(f".{args.format}"))


if __name__ == "__main__":
    main()
