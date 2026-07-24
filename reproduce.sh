#!/usr/bin/env bash
#
# Regenerate every figure in figures/ from the committed logs and checkpoints.
#
#     ./reproduce.sh              # tests, mypy, then all 14 figures: ~1 min
#     PYTHON=/path/to/python ./reproduce.sh
#
# NO TRAINING happens here, and that is the point. The sweeps behind these
# figures are ~40 seeded runs of a transformer to 15-25k steps; reproducing
# them from scratch is hours of CPU. So the runs ship with the repo -- the
# CSV/JSON logs and the two model checkpoints in runs/ -- and this script turns
# those artifacts back into every figure in the README.
#
# To actually retrain instead, see experiments/run_sweep.py (the sweep grid) and
# the per-experiment scripts; each skips a cell whose log already exists, so
# delete the ones you want recomputed.
#
# Determinism: the replay is pure post-processing of committed files, so it is
# exact -- the figures come back byte-for-byte. Training itself is seeded and
# replays on the same torch build and CPU, but torch guarantees no bitwise
# determinism across versions or hardware, which is precisely why the artifacts
# are committed rather than regenerated on demand.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-}"
if [ -z "${PY}" ]; then
    if [ -x .venv/bin/python ]; then PY="$PWD/.venv/bin/python"; else PY="python3"; fi
fi

echo "=================================================================="
echo "grokking-transformer: figure reproduction (no training)"
echo "python:     $("$PY" -V 2>&1)  ($PY)"
"$PY" - <<'EOF'
import importlib
for name in ("torch", "numpy", "matplotlib"):
    try:
        m = importlib.import_module(name)
        print(f"{name+':':11s} {m.__version__}")
    except ImportError:
        print(f"{name+':':11s} MISSING")
EOF
echo "=================================================================="

started=$SECONDS

step() {  # step <label> <script> [args...]
    local label="$1"; shift
    echo
    echo "------------------------------------------------------------------"
    echo ">>> $label"
    echo "------------------------------------------------------------------"
    local t0=$SECONDS
    "$PY" "$@"
    echo "    [${label}: $((SECONDS - t0))s]"
}

# The suite includes the checks that every figure has a replay path and that
# every artifact it reads is committed (tests/test_reproduce_figures.py), so a
# failure here is exactly the failure this script exists to prevent.
step "test suite" -m pytest -q
step "static type check (mypy, grokking/)" -m mypy
step "regenerate all 14 figures from committed artifacts" experiments/reproduce_figures.py

echo
echo "=================================================================="
echo "done in $((SECONDS - started))s. figures/:"
ls -1 figures/
echo "=================================================================="
