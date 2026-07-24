"""The replay path must cover every figure the repo ships.

``experiments/reproduce_figures.py`` is this repo's reproducibility promise: a
reader clones it and regenerates every figure from the committed CSV/JSON logs
and ``.pt`` checkpoints, with no training. The promise is worth exactly its
coverage, and coverage rots silently -- ``wd_scope.png`` (section 7) shipped
without a replay path and stayed unreproducible for eleven days, because
nothing compared the script's outputs against the figures actually committed.

So compare them here, in both directions, plus the artifacts they read.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

# The experiment modules import matplotlib at module level; it is a dev extra,
# so skip rather than fail where only the package deps are installed.
pytest.importorskip("matplotlib")

import reproduce_figures  # noqa: E402


def test_declared_figures_match_the_figures_directory():
    """Every committed figure has a replay path, and every declared figure
    exists -- no stale entries, no unreproducible extras."""
    on_disk = {p.name for p in (ROOT / "figures").glob("*.png")}
    declared = set(reproduce_figures.FIGURES)

    unreproducible = sorted(on_disk - declared)
    stale = sorted(declared - on_disk)
    assert not unreproducible, (
        "figures/ contains files reproduce_figures.py does not regenerate "
        f"(add them to FIGURES and call their producer in main()): {unreproducible}"
    )
    assert not stale, (
        "reproduce_figures.FIGURES names files that are not in figures/: "
        f"{stale}"
    )


def test_figure_list_has_no_duplicates():
    assert len(set(reproduce_figures.FIGURES)) == len(reproduce_figures.FIGURES)


def test_every_artifact_the_figures_depend_on_is_committed():
    """The run logs and checkpoints the replay reads must be in the repo, so a
    fresh clone can reproduce without retraining anything."""
    missing = reproduce_figures.check_artifacts()
    assert missing == [], f"missing committed artifacts: {missing}"
