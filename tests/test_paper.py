"""The paper must agree with the repo it is built from.

``paper/main.tex`` is typeset in CI, so a LaTeX error cannot ship. What CI's
build alone would *not* catch is the paper drifting away from the code around
it: a figure included that nothing regenerates, a figure regenerated that the
paper silently stopped showing, a ``\\ref`` to a label that was renamed, or a
citation key that is not in the bibliography (LaTeX prints "??" for the first
two of those and carries on with exit status 0).

So the text checks here are deliberately pure -- no matplotlib, no torch -- and
run in CI's ordinary test job, where matplotlib is not installed. The four
tests that need the rendering machinery ask for it per test, not at module
scope -- see the note above them.
"""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
TEX = (PAPER / "main.tex").read_text()

sys.path.insert(0, str(ROOT / "experiments"))


def included_figures():
    """Figure filenames the paper includes, in order."""
    return re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", TEX)


def repo_figure_names(suffix=".pdf"):
    """The repo's figure set, named the way the paper's copies are named.

    Read off ``figures/`` rather than imported from
    ``reproduce_figures.FIGURES``, so this check still runs where matplotlib is
    not installed. ``test_reproduce_figures.py`` is what pins the two together;
    ``test_paper_figure_names_match_the_repo_figures`` below closes the loop.
    """
    return {p.stem + suffix for p in (ROOT / "figures").glob("*.png")}


def test_paper_includes_every_figure_the_repo_ships_and_nothing_else():
    included = set(included_figures())
    expected = repo_figure_names()

    orphaned = sorted(included - expected)
    unused = sorted(expected - included)
    assert not orphaned, (
        "paper/main.tex includes figures nothing regenerates "
        f"(the build will fail on a fresh clone): {orphaned}"
    )
    assert not unused, (
        "figures/ has figures the paper does not show. Either include them or "
        f"say in main.tex's header why they are left out: {unused}"
    )


def test_no_figure_is_included_twice():
    included = included_figures()
    assert len(included) == len(set(included)), "duplicate \\includegraphics"


def test_every_ref_has_a_label():
    """LaTeX renders an unresolved \\ref as "??" and still exits 0, so the CI
    build would not fail on one. This does."""
    labels = set(re.findall(r"\\label\{([^}]+)\}", TEX))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", TEX))
    assert refs <= labels, f"\\ref to missing label(s): {sorted(refs - labels)}"


def test_every_citation_key_is_in_the_bibliography():
    """Same reason: a missing bib entry is a "[?]" in the PDF, not an error."""
    keys = set(re.findall(r"@\w+\{([^,]+),", (PAPER / "refs.bib").read_text()))
    cited = set()
    for group in re.findall(r"\\cite[tp]?\{([^}]+)\}", TEX):
        cited.update(k.strip() for k in group.split(","))
    assert cited <= keys, f"cited but not in refs.bib: {sorted(cited - keys)}"


def test_the_bibliography_has_no_unused_entries():
    keys = set(re.findall(r"@\w+\{([^,]+),", (PAPER / "refs.bib").read_text()))
    cited = set()
    for group in re.findall(r"\\cite[tp]?\{([^}]+)\}", TEX):
        cited.update(k.strip() for k in group.split(","))
    assert keys <= cited, f"in refs.bib but never cited: {sorted(keys - cited)}"


def test_paper_figures_directory_is_not_committed():
    """The build regenerates paper/figures/ from the committed logs every time.
    Committing it would reintroduce exactly the stale-artifact failure that
    building in CI is meant to make impossible."""
    ignore = (ROOT / ".gitignore").read_text()
    assert "paper/figures/" in ignore


# --- the rendering machinery ------------------------------------------------
#
# Gated per test, not at module scope. A module-level importorskip would skip
# this whole file where matplotlib is absent -- which is CI's test job -- and
# take the six text checks above with it, silently. That is the same mistake
# the repo made once already with a module-level skip in an attention test.


def _renderer():
    """Import the print-resolution renderer, or skip this test without it."""
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import paper_figures

    return paper_figures


def test_paper_figure_names_match_the_repo_figures():
    """Ties the text checks above to the renderer's own idea of the file set."""
    paper_figures = _renderer()
    assert set(paper_figures.paper_figure_names()) == repo_figure_names()


def test_render_to_redirects_the_path_and_overrides_an_explicit_dpi(tmp_path):
    """The reason paper_figures exists rather than an rcParams tweak.

    Two producers (``head_count.py``, ``swap_equivariance.py``) pass an
    explicit ``dpi=150`` to savefig, which beats any rcParams change. If the
    interception ever stops overriding it, those two figures would quietly ship
    into the paper at 150 dpi while the other sixteen were at 300 -- a
    difference no build error and no eyeball would report.
    """
    paper_figures = _renderer()
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(1, 1))
    with paper_figures.render_to(tmp_path, dpi=300, fmt="pdf") as written:
        fig.savefig(ROOT / "figures" / "somewhere_else.png", dpi=150)
    plt.close(fig)

    assert written == [tmp_path / "somewhere_else.pdf"]
    assert written[0].exists()

    # A 1x1-inch figure saved as PNG records its dpi in the file, which is the
    # only place the override is directly observable; re-run in that format.
    fig = plt.figure(figsize=(1, 1))
    with paper_figures.render_to(tmp_path, dpi=300, fmt="png"):
        fig.savefig(ROOT / "figures" / "somewhere_else.png", dpi=150)
    plt.close(fig)
    from PIL import Image
    with Image.open(tmp_path / "somewhere_else.png") as im:
        assert im.size == (300, 300), (
            f"explicit dpi=150 was not overridden: got {im.size}, want (300, 300)"
        )


def test_render_to_restores_savefig_even_when_the_body_raises(tmp_path):
    paper_figures = _renderer()
    from matplotlib.figure import Figure

    original = Figure.savefig
    with pytest.raises(ValueError):
        with paper_figures.render_to(tmp_path):
            raise ValueError("boom")
    assert Figure.savefig is original


def test_render_to_leaves_the_committed_figures_alone(tmp_path):
    """Interception must not write into figures/ -- the README's PNGs are
    byte-compared by the reproducibility promise and must not move."""
    paper_figures = _renderer()
    import matplotlib.pyplot as plt

    before = {p: p.stat().st_mtime_ns for p in (ROOT / "figures").glob("*.png")}
    fig = plt.figure(figsize=(1, 1))
    with paper_figures.render_to(tmp_path):
        fig.savefig(ROOT / "figures" / "grokking_main.png")
    plt.close(fig)
    after = {p: p.stat().st_mtime_ns for p in (ROOT / "figures").glob("*.png")}
    assert before == after
