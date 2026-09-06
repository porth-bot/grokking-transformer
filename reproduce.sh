#!/usr/bin/env bash
#
# Regenerate every figure in figures/ from the committed logs and checkpoints.
#
#     ./reproduce.sh              # tests, mypy, then all 18 figures: ~60 s
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
step "regenerate all 18 figures from committed artifacts" experiments/reproduce_figures.py

echo
echo "=================================================================="
echo "done in $((SECONDS - started))s. figures/:"
ls -1 figures/
echo "=================================================================="

# Check the byte-for-byte claim rather than repeating it. The replay above is
# pure post-processing of committed logs and checkpoints, so unlike the NumPy
# repos in this series there is no machine-dependent column to excuse: any
# figure that now differs from its committed copy means the plotting code and
# the shipped PNG have diverged. That is how gp-from-scratch found two of its
# figures sitting stale, and reading `git status` by hand is not a check.
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    # Worktree-vs-index differences only: the porcelain format is XY<space>path,
    # X the index status and Y the worktree's, so a figure that is merely newly
    # staged reads "A  path" and is not drift. Untracked ("??") ones are: a PNG
    # the replay writes but nobody committed is the same problem seen from the
    # other side.
    changed=$(git status --porcelain -- figures/ \
        | awk '{ y = substr($0, 2, 1); if (y == "M" || substr($0,1,2) == "??") print $2 }')
    echo
    if [ -z "${changed}" ]; then
        echo "reproduction: all 18 committed figures came back byte-for-byte."
    else
        echo "reproduction: DRIFT -- these differ from the committed copies:"
        for f in ${changed}; do echo "    ${f}"; done
        echo "Nothing here plots wall-clock, so this is not machine variation."
        echo "Inspect the difference, then commit the regenerated files."
    fi
    echo "=================================================================="
fi
