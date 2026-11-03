#!/bin/sh
# Render the pipeline diagrams in this folder with Mermaid (mermaid-cli), using
# the repository's mermaid-config.json. Writes <name>.pdf (default, cropped to
# the diagram), <name>.png (./render.sh png, at 2x scale) or <name>.drawio
# (./render.sh drawio: an editable draw.io diagram laid out like the PDF, see
# mermaid-to-drawio.mjs) next to each <name>.mmd. Requires `npm install` at
# the repository root.
set -eu

format="${1:-pdf}"
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/.." && pwd)"
mmdc="$root/node_modules/.bin/mmdc"

case "$format" in
  pdf) options="" ;;
  png) options="--scale 2" ;;
  drawio) options="" ;;
  *) echo "usage: $0 [pdf|png|drawio]" >&2; exit 2 ;;
esac

if [ ! -x "$mmdc" ]; then
  echo "mermaid-cli not found; run 'npm install' in $root" >&2
  exit 1
fi

for input in "$here"/*.mmd; do
  output="${input%.mmd}.$format"
  if [ "$format" = drawio ]; then
    node "$here/mermaid-to-drawio.mjs" "$input" "$output" >/dev/null
  else
    # $options is intentionally unquoted: empty, or one flag and its value.
    "$mmdc" -c "$root/mermaid-config.json" -i "$input" -o "$output" -b white $options
  fi
  echo "$output"
done
