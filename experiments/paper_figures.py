"""Render the paper's figures at print resolution, from the committed logs.

``paper/main.tex`` needs the same eighteen figures the README shows, but a
150 dpi PNG sized for a browser is not a print figure: at the ~3.3 in column
width of a two-column page it lands around 500 dpi worth of pixels for the
line art and then throws the rest away on any zoom. So the paper gets vector
PDFs, drawn by *exactly* the same code, and nothing about ``figures/*.png``
changes.

The mechanism, and why it is a monkeypatch
------------------------------------------
Fifteen call sites across eleven scripts hard-code their own output path
(``fig.savefig(ROOT / "figures" / "x.png", ...)``), and two of them --
``head_count.py`` and ``swap_equivariance.py`` -- also pass an explicit
``dpi=150``, which silently wins over any ``rcParams`` change. So neither
"set a different rcParam" nor "set a different output directory" is available
without editing eleven files.

Editing them was the alternative and it is the worse one: it would put a second
code path between the logs and the paper, which is the exact drift this repo
keeps finding (``figures/sparse.png`` in the sibling gp repo shipped stale for
a day for want of one). Instead ``render_to`` swaps ``Figure.savefig`` for the
duration of one call to ``reproduce_figures.regenerate_all()``, rewriting the
destination and forcing the dpi. The producers are untouched, so a figure whose
code changes changes in both places at once, and a figure added to
``reproduce_figures.FIGURES`` reaches the paper with no edit here at all.

The explicit-dpi override is the part worth testing rather than assuming, so
``tests/test_paper.py`` asserts that a ``savefig(..., dpi=150)`` under
``render_to`` comes out at the paper's dpi and not at 150.

Two known cosmetic consequences: some producers print "wrote figures/x.png"
from a path they computed themselves rather than from the one savefig used, so
their stdout is wrong under interception (this module prints the authoritative
list at the end), and ``dpi`` on a vector backend only governs the rasterized
elements plus the point-to-pixel conversion, not the line art.

Output goes to ``paper/figures/``, which is *not* committed -- CI regenerates
it from the committed CSV/JSON logs and the two committed checkpoints before
every paper build, so a stale figure cannot reach the PDF.

Run:  python experiments/paper_figures.py
"""

import sys
from contextlib import contextmanager
from pathlib import Path

from matplotlib.figure import Figure

import reproduce_figures

ROOT = Path(__file__).resolve().parent.parent
PAPER_FIGS = ROOT / "paper" / "figures"

# Vector, so the figures are resolution-independent in print; the dpi below
# still matters for the rasterized parts (imshow panels) and for how
# points-to-inches conversions land.
FORMAT = "pdf"
DPI = 300


def paper_figure_names(fmt: str = FORMAT):
    """The paper's figure filenames: reproduce_figures.FIGURES, re-suffixed.

    Single source of truth in both directions -- ``main.tex`` may include a
    subset, but it may not include anything that is not here.
    """
    return tuple(Path(name).with_suffix("." + fmt).name
                 for name in reproduce_figures.FIGURES)


@contextmanager
def render_to(outdir: Path, dpi: int = DPI, fmt: str = FORMAT):
    """Redirect every ``Figure.savefig`` into ``outdir`` at ``dpi``/``fmt``.

    Only the basename of the requested path survives; the directory, the
    extension and the dpi are replaced. Yields the list of files written, in
    the order they were written, so the caller can report and check it.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    original = Figure.savefig
    written: list[Path] = []

    def patched(self, fname=None, *args, **kwargs):
        stem = Path(str(fname)).stem
        target = outdir / f"{stem}.{fmt}"
        # Positional dpi is not part of savefig's signature (everything after
        # fname is keyword-only), so overriding the keyword is sufficient and
        # beats an explicit dpi= at the call site.
        kwargs["dpi"] = dpi
        result = original(self, target, *args, **kwargs)
        written.append(target)
        return result

    Figure.savefig = patched  # type: ignore[method-assign]
    try:
        yield written
    finally:
        Figure.savefig = original  # type: ignore[method-assign]


def main():
    missing = reproduce_figures.check_artifacts()
    if missing:
        print("ERROR: missing committed artifacts required to build the paper "
              "figures:")
        for m in missing:
            print(f"  - {m}")
        return 1

    with render_to(PAPER_FIGS) as written:
        reproduce_figures.regenerate_all()

    names = sorted(p.name for p in written)
    expected = sorted(paper_figure_names())
    print(f"\nWrote {len(names)} figures to paper/figures/ at {DPI} dpi "
          f"({FORMAT}):")
    for n in names:
        size_kb = (PAPER_FIGS / n).stat().st_size / 1024
        print(f"  {n:28s} {size_kb:8.1f} KB")

    if names != expected:
        print("\nERROR: what was written does not match "
              "reproduce_figures.FIGURES:")
        print(f"  unexpected: {sorted(set(names) - set(expected))}")
        print(f"  missing:    {sorted(set(expected) - set(names))}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
