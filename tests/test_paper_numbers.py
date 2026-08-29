"""The paper's tables must agree with the runs they claim to summarize.

``test_paper.py`` checks the paper against the *repo* -- figures, refs, cites.
This checks it against the *data*: every grok-step cell the paper tabulates is
recomputed here from ``runs/*.json`` and compared against the typeset text.

That closes the loop the CI build leaves open. Building the PDF proves the
LaTeX compiles; regenerating the figures proves no figure is staler than its
logs. Neither says anything about a number typed into a table, and the numbers
are what a reader takes away. Every prose correction made while these sections
were written was a number or a shape that had drifted from the logs behind it,
in a repository that already regenerates its figures -- so a table typed by
hand is exactly where the next one would hide.

Pure stdlib + NumPy on purpose: no matplotlib, no torch, so this runs in CI's
ordinary test job alongside the text checks.
"""

import json
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
TEX = (ROOT / "paper" / "main.tex").read_text()

BUDGET = 25_000


# -- reading the paper -------------------------------------------------------

def table_rows(label):
    """Body rows of the tabular carrying ``\\label{label}``, as cell lists."""
    block = re.search(
        r"\\begin\{table\}.*?\\label\{" + re.escape(label) + r"\}.*?"
        r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}",
        TEX,
        re.S,
    )
    assert block, f"no tabular found for label {label!r}"
    body = block.group(1)
    body = body.split(r"\midrule", 1)[1].split(r"\bottomrule", 1)[0]
    rows = []
    for line in body.split(r"\\"):
        line = line.strip()
        if not line:
            continue
        rows.append([c.strip() for c in line.split("&")])
    return rows


# -- reading the runs --------------------------------------------------------

def grok_steps(pattern):
    """Per-seed grok step for every run matching ``pattern`` (None = never)."""
    names = sorted(p.stem for p in RUNS.glob(pattern + ".json"))
    assert names, f"no runs matched {pattern!r}"
    return [json.loads((RUNS / f"{n}.json").read_text())["grok_step"] for n in names]


def latex_int(x):
    """1300 -> '1{,}300', the way the paper's tables write it."""
    return f"{int(x):,}".replace(",", "{,}")


def cell(values):
    """A grok-step cell as the paper writes it: median [min--max]."""
    reached = [v for v in values if v is not None]
    assert reached, "every seed censored; the caller should special-case that"
    med, lo, hi = np.median(reached), min(reached), max(reached)
    body = latex_int(med)
    if lo != hi:
        body += f" [{latex_int(lo)}--{latex_int(hi)}]"
    if len(reached) < len(values):
        body += f" ({len(reached)} of {len(values)} seeds)"
    return body


def censored_median_is_a_bound(values):
    """True when at least half a cell's seeds never grokked."""
    n_censored = sum(v is None for v in values)
    return n_censored * 2 >= len(values)


# -- the tables --------------------------------------------------------------

def test_weight_decay_table_matches_the_runs():
    rows = table_rows("tab:wd")
    assert [r[0] for r in rows] == ["$0.0$", "$0.1$", "$1.0$"]

    for row, wd in zip(rows, ["0", "0.1", "1"]):
        steps = grok_steps(f"p97_frac0.30_wd{wd}_seed[0-9]")
        assert len(steps) == 5
        if wd == "0":
            assert all(s is None for s in steps)
            assert "never" in row[2] and "5/5" in row[2]
        else:
            assert row[2] == cell(steps), f"wd={wd}: table says {row[2]!r}"
        # Every arm memorizes at step 100, which the table asserts in column 1.
        assert row[1] == "100"


def test_fraction_table_matches_the_runs():
    rows = table_rows("tab:frac")
    for row, frac in zip(rows, ["0.25", "0.30", "0.40", "0.60"]):
        steps = grok_steps(f"p97_frac{frac}_wd1_seed[0-9]")
        if len(steps) == 1:
            assert row[1] == f"{latex_int(steps[0])} (1 seed)"
        else:
            assert row[1] == cell(steps), f"frac={frac}: table says {row[1]!r}"


def test_the_surface_table_matches_the_runs_including_its_censoring():
    rows = table_rows("tab:surface")
    fracs = ["0.25", "0.30", "0.40"]
    wds = ["0", "0.1", "0.3", "1"]

    for row, frac in zip(rows, fracs):
        for text, wd in zip(row[1:], wds):
            steps = grok_steps(f"p97_frac{frac}_wd{wd}_seed[0-2]")
            assert len(steps) == 3
            if censored_median_is_a_bound(steps):
                assert text == f"$> {latex_int(BUDGET)}$", (
                    f"({frac}, {wd}) is censored; table says {text!r}"
                )
            else:
                assert text == cell(steps), f"({frac}, {wd}): table says {text!r}"


def test_head_count_table_matches_the_runs():
    rows = table_rows("tab:heads")
    for row, suffix in zip(rows, ["_h1", "_h2", ""]):
        steps = grok_steps(f"p97_frac0.30_wd1_seed[0-9]{suffix}")
        assert len(steps) == 5
        assert row[2] == cell(steps), f"heads {row[0]}: table says {row[2]!r}"
        # The seed-spread column, which is what makes the ordering readable.
        spread = max(steps) / min(steps)
        assert row[3] == f"${spread:.1f}\\times$"


def test_operations_table_matches_the_runs():
    rows = table_rows("tab:ops")
    wanted = [("", "1"), ("_opsub", "1"), ("_opmul", "1"),
              ("", "0.1"), ("_opsub", "0.1"), ("_opmul", "0.1")]
    for row, (op, wd) in zip(rows, wanted):
        # Three seeds per cell, including the addition rows, which reuse the
        # first three seeds of the main sweep rather than all five. Reading
        # them at five would put a different range in the addition rows than
        # in the sub/mul rows they exist to be compared against.
        steps = grok_steps(f"p97_frac0.30_wd{wd}_seed[0-2]{op}")
        assert len(steps) == 3
        assert row[3] == cell(steps), f"{op or 'add'} wd={wd}: table says {row[3]!r}"


# -- the claims the tables are quoted for ------------------------------------

def test_the_separations_the_text_claims_are_actually_disjoint():
    """Two orderings the paper calls complete, checked as set inequalities."""
    wd1 = grok_steps("p97_frac0.30_wd1_seed[0-9]")
    wd01 = grok_steps("p97_frac0.30_wd0.1_seed[0-9]")
    assert max(wd1) < min(wd01), "Sec. 3.1's arms are supposed to be disjoint"

    h1 = grok_steps("p97_frac0.30_wd1_seed[0-9]_h1")
    h2 = grok_steps("p97_frac0.30_wd1_seed[0-9]_h2")
    h4 = grok_steps("p97_frac0.30_wd1_seed[0-9]")
    assert max(h1) < min(h2) < max(h2) < min(h4), "Sec. 5.1 claims no overlap"

    sub = grok_steps("p97_frac0.30_wd1_seed[0-2]_opsub")
    mul = grok_steps("p97_frac0.30_wd1_seed[0-2]_opmul")
    assert min(sub) > max(h4) and min(sub) > max(mul), (
        "Sec. 5.2 claims subtraction's range overlaps neither of the others"
    )


def test_seed_0_is_the_slowest_seed_in_every_head_count_arm():
    """Sec. 5.1's explanation of why the published table was pessimistic."""
    for suffix in ["_h1", "_h2", ""]:
        steps = grok_steps(f"p97_frac0.30_wd1_seed[0-9]{suffix}")
        assert steps[0] == max(steps), f"arm {suffix or '_h4'}: {steps}"


def test_the_seeds_being_averaged_did_not_run_the_same_length():
    """Sec. 4.5's confound: seed 0 got ~9,000 more steps than the others."""
    lengths = [
        json.loads((RUNS / f"p97_frac0.30_wd1_seed{s}.json").read_text())["steps_run"]
        for s in range(5)
    ]
    assert lengths[0] == 11_100
    assert max(lengths[1:]) <= 2_100 and min(lengths[1:]) >= 1_800
    assert lengths[0] - max(lengths[1:]) == pytest.approx(9_000, abs=100)
