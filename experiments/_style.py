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
from matplotlib.figure import Figure  # noqa: E402

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


def _strip_png_software_tag() -> None:
    """Stop matplotlib stamping its own version into every PNG it writes.

    By default a matplotlib PNG carries a ``Software`` tEXt chunk reading
    "Matplotlib version3.11.0, ...". Nothing renders it, and it makes the
    committed figures *byte*-unstable across a matplotlib patch release while
    leaving them pixel-identical: regenerating all 18 figures from a fresh
    clone on 3.11.1 produced 18 files that ``cmp`` called different and whose
    decoded pixels agreed exactly, the only differing bytes being that string.

    That matters because the README promises the figures come back
    byte-for-byte and the fresh-clone check is how the repo keeps that promise
    -- a check that reports 18 false positives on a routine dependency bump is
    a check nobody can read. Passing ``metadata={"Software": None}`` drops the
    chunk, so a byte difference means a pixel difference again.

    Done by wrapping ``Figure.savefig`` once rather than editing fourteen call
    sites, for the reason this module exists at all: a rule copied into every
    producer is a rule that will be missing from the fifteenth. PDF and SVG
    output is left alone -- their metadata dictionaries take a different set of
    keys, and ``paper_figures.render_to`` (which wraps this wrapper to redirect
    the paper build to PDFs) must keep working.
    """
    if getattr(Figure.savefig, "_strips_png_software", False):
        return

    original = Figure.savefig

    def savefig(self, fname=None, *args, **kwargs):
        is_png = str(getattr(fname, "name", fname)).lower().endswith(".png")
        if is_png and kwargs.get("format", "png") == "png" and "metadata" not in kwargs:
            kwargs["metadata"] = {"Software": None}
        return original(self, fname, *args, **kwargs)

    savefig._strips_png_software = True  # type: ignore[attr-defined]
    Figure.savefig = savefig  # type: ignore[method-assign]


def apply_style() -> None:
    """Set the repo-wide figure style. Idempotent; call it before plotting."""
    plt.rcParams.update(STYLE)
    _strip_png_software_tag()
