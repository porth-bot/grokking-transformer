"""A figure must not depend on which module was imported before it.

``experiments/operations.py`` used to set no matplotlib style of its own, while
ten sibling scripts each carried their own copy of the same
``plt.rcParams.update`` block. rcParams are process-global, so the dpi
``figures/operations.png`` came out at was decided by import order: 150 through
``reproduce_figures.py`` (which imports ``plots.py`` first), 100 running
``experiments/operations.py`` alone. The committed PNG was the 100-dpi one, so
the replay path the README tells a reader to run produced a *different* file
than the repo ships -- under a README promising every PNG comes back
byte-for-byte.

``experiments/_style.py`` is the fix. These tests pin both halves of it: the
style is applied from one place everywhere (so no script can drift or be
forgotten again), and importing the script that had the bug, on its own,
already gives the committed dpi.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

pytest.importorskip("matplotlib")

import _style  # noqa: E402


def figure_scripts():
    """Every experiment module that writes a figure."""
    return sorted(
        p for p in EXPERIMENTS.glob("*.py")
        if p.name != "_style.py" and "savefig(" in p.read_text()
    )


def test_there_is_at_least_one_figure_script():
    # Guards the discovery above: an empty list would make the next test vacuous.
    assert len(figure_scripts()) >= 10


@pytest.mark.parametrize("path", figure_scripts(), ids=lambda p: p.name)
def test_figure_scripts_apply_the_shared_style(path):
    src = path.read_text()
    assert "apply_style()" in src, (
        f"{path.name} saves a figure but never calls apply_style(), so its "
        "output depends on whether some other module set rcParams first"
    )
    assert not re.search(r"rcParams\s*(\.update|\[)", src), (
        f"{path.name} sets rcParams directly; put shared style in _style.STYLE "
        "so every entry point draws the same figure"
    )


def test_style_is_the_dpi_the_committed_figures_were_drawn_at():
    assert _style.STYLE["savefig.dpi"] == 150
    assert _style.STYLE["figure.dpi"] == 150


def test_importing_a_figure_script_alone_already_sets_the_style():
    """The regression itself: no sibling import, still 150 dpi.

    A subprocess, because rcParams are global and the rest of this suite has
    already imported half the plotting stack.
    """
    code = (
        "import operations, matplotlib.pyplot as plt;"
        "print(plt.rcParams['savefig.dpi'], plt.rcParams['figure.dpi'])"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=EXPERIMENTS, capture_output=True, text=True, check=True,
    )
    assert out.stdout.split() == ["150.0", "150.0"], out.stdout
