#!/usr/bin/env bash
# Build paper/main.pdf from the committed logs.
#
# Two steps, in this order and never separately:
#   1. regenerate paper/figures/*.pdf from runs/ (no training, ~15 s)
#   2. typeset main.tex
#
# paper/figures/ is not committed, so step 1 is not optional -- that is the
# whole reason the PDF cannot ship a figure older than the data behind it.
# CI runs exactly this (see .github/workflows/ci.yml, job "paper"), so a build
# that only works on one laptop fails there.
#
# Requires: the repo's dev deps (matplotlib, torch) and tectonic
# (https://tectonic-typesetting.github.io). Usage:  ./paper/build.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"

echo "==> regenerating paper figures at print resolution"
# From ROOT, not from experiments/: cd-ing first would break a relative
# PYTHON (e.g. PYTHON=.venv/bin/python). Running the script by path still puts
# experiments/ on sys.path, which is how it imports its siblings.
cd "$ROOT"
"$PYTHON" experiments/paper_figures.py

echo "==> typesetting paper/main.tex"
cd "$ROOT/paper"
tectonic -X compile main.tex

echo "==> wrote paper/main.pdf"
