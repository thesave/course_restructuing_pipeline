#!/bin/sh
# Launch the module review page: build review.html for a `modules_engine solve`
# run and open it in the default browser. Without RUN_DIR, the most recent run
# under data/modules-engine/ is reviewed. Works from any directory.
#
#   ./review-modules.sh [RUN_DIR] [--output PATH]
set -eu

root="$(cd "$(dirname "$0")" && pwd)"

if [ ! -x "$root/.venv/bin/python" ]; then
  echo "No Python environment at $root/.venv (see AGENTS.md)" >&2
  exit 1
fi
if [ ! -f "$root/node_modules/mermaid/dist/mermaid.min.js" ]; then
  echo "Mermaid not found; run 'npm install' in $root" >&2
  exit 1
fi

# PYTHONPATH instead of cd, so that RUN_DIR may be relative to the caller.
PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}" exec "$root/.venv/bin/python" -m modules_engine.review "$@" --open
