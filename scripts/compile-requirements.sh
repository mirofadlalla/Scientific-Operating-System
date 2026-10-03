#!/usr/bin/env bash
# Regenerate the fully-pinned requirements.txt from requirements.in.
#
#   ./scripts/compile-requirements.sh            # re-resolve everything
#   ./scripts/compile-requirements.sh --upgrade  # bump to the newest allowed versions
#
# torch (and everything that exists only because of it: CUDA wheels, triton,
# sympy, jinja2 …) is deliberately NOT written to requirements.txt. The Dockerfile
# installs the CPU-only torch wheel first, which keeps the image ~4 GB smaller.
# For local development install it yourself:  pip install torch  (or the CPU index).
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || { echo "uv is required: pip install uv"; exit 1; }

PY=3.11
TMP=$(mktemp)
OUT=$(mktemp)
trap 'rm -f "$TMP" "$OUT"' EXIT

# Pass 1: resolve *with* torch so we can see which CUDA packages it drags in.
uv pip compile requirements.in --python-version "$PY" --universal --no-header -q "$@" -o "$TMP"

EXCLUDES=(torch triton cuda-bindings cuda-pathfinder cuda-toolkit setuptools sympy mpmath jinja2 markupsafe)
while read -r pkg; do EXCLUDES+=("$pkg"); done < <(grep -oE '^nvidia-[a-z0-9-]+' "$TMP" | sort -u)

ARGS=()
for p in "${EXCLUDES[@]}"; do ARGS+=(--no-emit-package "$p"); done

# Pass 2: the file we actually ship. (Written to a temp file, not /dev/stdout:
# uv reads the -o path as an existing lock, which would block on a pipe.)
uv pip compile requirements.in --python-version "$PY" --universal --no-header --no-annotate -q "$@" "${ARGS[@]}" -o "$OUT"
{
  echo "# ─────────────────────────────────────────────────────────────────────────────"
  echo "# AUTO-GENERATED — do not edit by hand. Edit requirements.in, then run:"
  echo "#     ./scripts/compile-requirements.sh"
  echo "# Fully pinned for Python ${PY}+ (universal: Linux/macOS/Windows)."
  echo "# torch is intentionally absent — see scripts/compile-requirements.sh."
  echo "# ─────────────────────────────────────────────────────────────────────────────"
  grep -vE '^#' "$OUT"
} > requirements.txt
echo "Wrote requirements.txt ($(grep -cE '^[A-Za-z0-9]' requirements.txt) pinned packages)"
