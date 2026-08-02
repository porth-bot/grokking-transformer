"""One figure style for every script in this directory, applied explicitly.

Before this module existed the same six-line ``plt.rcParams.update`` block was
copy-pasted into ten scripts and *missing* from an eleventh, ``operations.py``.
That made the appearance of ``figures/operations.png`` depend on import order:

- through ``reproduce_figures.py``, ``plots.py`` was imported first, its copy of
  the block had already set ``figure.dpi = 150`` process-wide, and the figure
  came out at 150 dpi;
- running ``python experiments/operations.py`` on its own, nothing had set it,
  so matplotlib's default 100 dpi applied.

The committed PNG was the 100-dpi one, so ``./reproduce.sh`` -- the command the
README hands a fresh reader -- regenerated a *different* file than the one in
the repo, against a README that promises every PNG comes back byte-for-byte. A
shared style makes the output a property of the code rather than of which
module happened to be imported first; ``tests/test_figure_style.py`` pins both
halves of that.

Importing this module also selects the non-interactive Agg backend, so figure
scripts run headless (CI, a remote shell) without a display.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# The canonical style. 150 dpi is what every committed figure is drawn at; the
# despined axes and frameless legend are the house look across the repo.
STYLE = {
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
}


def apply_style() -> None:
    """Set the repo-wide figure style. Idempotent; call it before plotting."""
    plt.rcParams.update(STYLE)
