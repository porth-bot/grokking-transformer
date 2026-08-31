"""The README's tables must agree with the runs they claim to summarize.

Three modules already do this for the paper -- ``test_paper_numbers.py`` for
its tables, ``test_paper_setup.py`` for its configuration prose,
``test_paper_prose.py`` for its single-run numbers and captions -- and between
them they caught eleven claims that had drifted from the logs. Every one of
those lived in the paper. The README carries its *own* copy of most of the same
numbers, typed independently, and nothing checked it.

That gap is not hypothetical: writing the paper corrected "ends at 0.36 test
accuracy" to the 0.3545 its log says, and the README's Section 7 table went on
saying 0.36 for ten days afterwards, because a correction made in one document
does not walk across to the other. So the README gets the same instrument the
paper has -- its tables recomputed from ``runs/*.json`` and ``runs/*.csv``, and
its prose numbers where those are the point of a section.

Pure stdlib on purpose: no matplotlib, no torch, so this runs in CI's ordinary
test job.
"""

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
README = (ROOT / "README.md").read_text()

LN2 = 0.6931471805599453


# -- reading the README ------------------------------------------------------

def section(heading_starts_with):
    """The text of the `###` section whose heading starts with the given text."""
    parts = re.split(r"^### ", README, flags=re.M)
    hits = [p for p in parts if p.startswith(heading_starts_with)]
    assert len(hits) == 1, f"{heading_starts_with!r} matched {len(hits)} sections"
    return hits[0]


def plain(cell):
    """A table cell without markdown emphasis or code ticks."""
    return cell.replace("**", "").replace("*", "").replace("`", "").strip()


def table(text, index=0):
    """Body rows of the ``index``-th markdown table in ``text``, as cell lists."""
    rows, tables, in_table = [], [], False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [plain(c) for c in stripped.strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):   # the |---|---| rule
                in_table = True
                rows = []
                continue
            if in_table:
                rows.append(cells)
            continue
        if in_table:
            tables.append(rows)
            in_table = False
        rows = []
    if in_table:
        tables.append(rows)
    assert len(tables) > index, f"wanted table {index}, found {len(tables)}"
    return tables[index]


# -- reading the runs --------------------------------------------------------

def summary(name):
    return json.loads((RUNS / f"{name}.json").read_text())


def history(name):
    with open(RUNS / f"{name}.csv") as f:
        rows = list(csv.DictReader(f))
    out = {k: [float(r[k]) for r in rows] for k in rows[0]}
    out["step"] = [int(s) for s in out["step"]]
    return out


def summaries(pattern):
    names = sorted(p.stem for p in RUNS.glob(pattern + ".json"))
    assert names, f"no runs matched {pattern!r}"
    return [summary(n) for n in names]


def grok_steps(pattern):
    return [s["grok_step"] for s in summaries(pattern)]


def cell(values):
    """A grok-step cell the way the README writes it: `1,300 [1,200-1,900]`,
    with an en dash and thousands separators, and no range when it is one seed
    or every seed agrees."""
    reached = [v for v in values if v is not None]
    assert reached, "every seed censored; the caller should special-case that"
    med = sorted(reached)[len(reached) // 2]
    lo, hi = min(reached), max(reached)
    body = f"{med:,}"
    if lo != hi:
        body += f" [{lo:,}–{hi:,}]"
    return body


# -- the tables --------------------------------------------------------------

def test_the_weight_decay_table_matches_the_runs():
    rows = table(section("1. Weight decay controls"))
    assert [r[0] for r in rows] == ["0.0", "0.1", "1.0"]
    for row, wd in zip(rows, ["0", "0.1", "1"]):
        steps = grok_steps(f"p97_frac0.30_wd{wd}_seed[0-9]")
        assert len(steps) == 5
        assert row[1] == "step 100"
        if wd == "0":
            assert all(s is None for s in steps)
            assert "never" in row[2] and "all 5 seeds" in row[2]
            assert row[3] == "∞"
        else:
            assert row[2] == cell(steps), f"wd={wd}: README says {row[2]!r}"
            # the delay column is the median over the memorization step
            median = int(cell(steps).split(" ")[0].replace(",", ""))
            assert row[3] == f"{median // 100}×"


def test_the_fraction_table_matches_the_runs():
    rows = table(section("2. Less data"))
    for row, frac in zip(rows, ["0.25", "0.30", "0.40", "0.60"]):
        steps = grok_steps(f"p97_frac{frac}_wd1_seed[0-9]")
        if len(steps) == 1:
            assert row[1] == f"{steps[0]:,} (1 seed)"
            median = steps[0]
        else:
            assert len(steps) == 5
            assert row[1] == cell(steps), f"frac={frac}: README says {row[1]!r}"
            median = int(cell(steps).split(" ")[0].replace(",", ""))
        assert row[2] == f"{median // 100}×"


def test_the_head_count_table_matches_the_runs():
    rows = table(section("9. Does grokking need multiple heads"))
    for row, suffix in zip(rows, ["_h1", "_h2", ""]):
        runs = summaries(f"p97_frac0.30_wd1_seed[0-9]{suffix}")
        assert len(runs) == 5
        steps = [r["grok_step"] for r in runs]
        assert row[2] == f"{sum(s is not None for s in steps)}/5"
        assert row[3] == str(runs[0]["memorize_step"]) == "100"
        assert row[4] == cell(steps), f"heads {row[0]}: README says {row[4]!r}"
        assert row[5] == f"{max(steps) / min(steps):.1f}×"
        assert row[6] == f"{min(r['final_test_acc'] for r in runs):.3f}"


def test_the_weight_decay_scope_table_matches_the_runs():
    """Section 7's three arms, and the one that was wrong: the embeddings-only
    run ends at 0.3545, which the paper reports as 0.35 and the README reported
    as 0.36 until this test was written."""
    rows = table(section("7. *Where* does the norm pressure"))
    for row, name in zip(rows, ["p97_frac0.30_wd1_seed0",
                                "p97_frac0.30_wd1_seed0_wdsnon_embeddings",
                                "p97_frac0.30_wd1_seed0_wdsembeddings"]):
        s = summary(name)
        assert row[1] == f"step {s['memorize_step']}"
        if s["grok_step"] is None:
            assert row[2].startswith("never")
            assert f"{s['steps_run'] // 1000}k steps" in row[2]
        else:
            assert row[2] == f"step {s['grok_step']}", f"{name}: {row[2]!r}"
        assert row[3] == f"{s['final_test_acc']:.2f}", f"{name}: {row[3]!r}"


def test_the_norm_the_unconstrained_readout_reaches():
    """Section 7's headline for the embeddings-only arm is a norm, not an
    accuracy: with the readout undecayed the parameter norm climbs instead of
    turning over."""
    h = history("p97_frac0.30_wd1_seed0_wdsembeddings")
    assert round(h["weight_norm"][0], 1) == 21.7      # not "21": it rounds up
    assert round(h["weight_norm"][-1]) == 287
    assert h["weight_norm"][-1] == max(h["weight_norm"])   # climbs, never peaks
    text = section("7. *Where* does the norm pressure")
    assert "balloons from **21.7 to 287**" in text


def test_the_wd0_final_accuracies_are_the_range_the_text_quotes():
    finals = [s["final_test_acc"] for s in summaries("p97_frac0.30_wd0_seed[0-9]")]
    assert len(finals) == 5
    lo, hi = min(finals), max(finals)
    assert f"final test accuracy {lo:.2f}–{hi:.2f} across the five" in \
        section("1. Weight decay controls")


def test_the_attention_read_out_lags_the_transition():
    """The appendix's one timing claim, and the definition it depends on: the
    first step within 1% of ln 2 is step 0 (a fresh model is symmetric on
    average), so 'restored' is searched forward from peak asymmetry."""
    h = history("attention_p97_frac0.30_wd1_seed0")
    peak = max(range(len(h["attn_asymmetry"])), key=lambda i: h["attn_asymmetry"][i])
    restored = next(step for step, e in zip(h["step"][peak:],
                                            h["attn_operand_entropy"][peak:])
                    if e >= 0.999 * LN2)
    half = next(step for step, a in zip(h["step"], h["test_acc"]) if a >= 0.5)
    naive = next(step for step, e in zip(h["step"], h["attn_operand_entropy"])
                 if e >= 0.999 * LN2)
    assert (restored, half, naive) == (2300, 1500, 0)
    assert restored > summary("p97_frac0.30_wd1_seed0")["grok_step"] > half
    text = README
    assert "restored at step 2300" in text and "passes 0.5 (1500)" in text
    assert "grok step (1900)" in text
    # and the definition that number depends on is stated where it is quoted
    assert "runs forward from the step of peak asymmetry" in text


def test_the_operations_table_matches_the_runs():
    """Three seeds per cell, addition included -- reading the addition rows at
    five seeds would put a different range in them than in the sub/mul rows
    they exist to be compared against."""
    rows = table(section("11. Other operations"))
    wanted = [("", "1"), ("_opsub", "1"), ("_opmul", "1"),
              ("", "0.1"), ("_opsub", "0.1"), ("_opmul", "0.1")]
    for row, (op, wd) in zip(rows, wanted):
        runs = summaries(f"p97_frac0.30_wd{wd}_seed[0-2]{op}")
        assert len(runs) == 3
        steps = [r["grok_step"] for r in runs]
        reached = [x for x in steps if x is not None]
        if len(reached) < 3:
            assert row[3] == f"{reached[0]:,} ({len(reached)} of 3 seeds)"
            # a cell where two seeds never grokked reports every seed's
            # accuracy rather than a summary that would hide the disagreement
            assert row[4] == " / ".join(f"{r['final_test_acc']:.3f}" for r in runs)
        else:
            assert row[3] == cell(steps), f"{op or 'add'} wd={wd}: {row[3]!r}"
            assert row[4] == f"{min(r['final_test_acc'] for r in runs):.3f}"


def test_the_surface_table_matches_the_runs_including_its_censoring():
    rows = table(section("13. Does more data offset"))
    for row, frac in zip(rows, ["0.25", "0.30", "0.40"]):
        for text, wd in zip(row[1:], ["0", "0.1", "0.3", "1"]):
            steps = grok_steps(f"p97_frac{frac}_wd{wd}_seed[0-2]")
            assert len(steps) == 3
            if sum(x is None for x in steps) * 2 >= 3:
                assert text == "> 25,000", f"({frac}, {wd}) is censored: {text!r}"
            else:
                assert text == cell(steps), f"({frac}, {wd}): {text!r}"
