"""Tests for the (wd, frac) interaction surface (Sec. 13).

Three things here are worth testing beyond "the code runs", and they are the
three ways this particular figure could ship a wrong claim:

- **The censoring arithmetic.** ``censored_median`` is the only place that
  decides whether a cell has a number or a lower bound, and both of its failure
  modes are silent: imputing the budget for a censored cell invents a fast
  measurement, and refusing to report a median when one *is* identified throws
  away a real one. Its contract is checked against hand-computed cases,
  including the two-of-three-censored boundary.
- **The grid's identity with the sweep.** Five of the twelve cells are runs
  ``run_sweep.py`` already trained, and they are reused by *name*. If
  ``cfg_for`` drifted from the sweep's config in any field that enters
  ``run_name`` -- or in one that does not, which would be worse -- the surface
  would silently mix two configurations. Checked directly against
  ``run_sweep``'s own job list.
- **The additive fit.** It is the quantitative form of the section's headline,
  so it is pinned on data where the answer is known in closed form: an exactly
  additive surface must fit with zero residual, and a surface built with a
  planted interaction must show it at the planted size.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

import run_sweep  # noqa: E402
import wd_frac_surface as W  # noqa: E402


# --------------------------------------------------------------------------
# censoring
# --------------------------------------------------------------------------

def test_censored_median_all_observed():
    assert W.censored_median([300.0, 100.0, 200.0]) == (200.0, False)


def test_censored_median_one_censored_is_still_identified():
    # [100, 300, inf] -> the middle order statistic is observed, so the median
    # is exact no matter how large the censored seed would have been.
    assert W.censored_median([100.0, 300.0, None]) == (300.0, False)


def test_censored_median_two_of_three_censored_is_a_lower_bound():
    value, censored = W.censored_median([100.0, None, None])
    assert censored
    assert value == float(W.MAX_STEPS)


def test_censored_median_all_censored():
    value, censored = W.censored_median([None, None, None])
    assert censored and value == float(W.MAX_STEPS)


def test_censored_median_never_imputes_the_budget_as_an_observation():
    """A censored cell must not be reported as if it grokked at the budget.

    The tempting shortcut -- substitute ``MAX_STEPS`` for every ``None`` and
    take an ordinary median -- gives the same *number* here but loses the flag,
    and the flag is what keeps the figure from painting a colour over a region
    where nothing was measured. So the flag, not the number, is the assertion.
    """
    _, censored = W.censored_median([None, None, 100.0])
    assert censored is True


def test_censored_median_even_sample_upper_middle_censored():
    # [100, 200, 300, inf]: the midpoint of the two middle order statistics is
    # observed, so it is identified.
    assert W.censored_median([100.0, 200.0, 300.0, None]) == (250.0, False)
    # [100, 200, inf, inf]: the upper middle is censored, so it is not.
    assert W.censored_median([100.0, 200.0, None, None])[1] is True


def test_censored_median_empty_is_an_error():
    with pytest.raises(ValueError):
        W.censored_median([])


# --------------------------------------------------------------------------
# grid identity
# --------------------------------------------------------------------------

def test_grid_shape():
    assert len(W.cells()) == len(W.WDS) * len(W.FRACS) == 12
    assert len(set(W.cells())) == 12


def test_training_order_starts_at_the_cheap_corner():
    """Strongest regularization first, so an interrupted pass leaves signal."""
    wds_in_order = [w for _, w in W.cells()]
    assert wds_in_order == sorted(wds_in_order, reverse=True)


def test_overlapping_cells_reuse_the_sweep_runs_exactly():
    """The five shared cells must resolve to the sweep's own run names.

    Not just equal strings: the configs are compared field by field, because a
    difference in something ``run_name`` does not encode (``patience``, the
    model config, the betas) would give two different experiments one filename
    and the surface would be a mix of both.
    """
    sweep = {
        (c.train_frac, c.weight_decay, c.seed): c
        for c in (
            run_sweep.TrainConfig(
                p=97, train_frac=f, weight_decay=w,
                max_steps=run_sweep.MAX_STEPS, eval_every=100, seed=s,
            )
            for f, w, s in run_sweep.jobs()
        )
    }
    shared = 0
    for frac, wd in W.cells():
        for seed in W.SEEDS:
            key = (frac, wd, seed)
            if key not in sweep:
                continue
            shared += 1
            mine, theirs = W.cfg_for(frac, wd, seed), sweep[key]
            assert mine.run_name() == theirs.run_name()
            assert mine == theirs, f"{key}: config differs from the sweep's"
    assert shared == 15, f"expected 5 shared cells x 3 seeds, got {shared}"


def test_budget_matches_the_sweep():
    """The censoring bound is §1's "25k budget"; the two must not drift."""
    assert W.MAX_STEPS == run_sweep.MAX_STEPS


def test_run_names_are_distinct_across_seeds():
    for cell, names in W.run_names().items():
        assert len(set(names)) == len(W.SEEDS), cell


# --------------------------------------------------------------------------
# the additive (no-interaction) null
# --------------------------------------------------------------------------

def _exact_additive(a, b):
    """Surface with ``log10 T = a_i + b_j`` exactly, and nothing censored."""
    med = np.array([[10.0 ** (ai + bj) for bj in b] for ai in a])
    return med, np.zeros(med.shape, dtype=bool)


def test_additive_fit_is_exact_on_an_additive_surface():
    med, cen = _exact_additive([3.0, 2.5, 2.0], [4.0, 3.0, 2.5, 2.0])
    fit = W.additive_fit(med, cen)
    assert np.nanmax(np.abs(fit["resid"])) < 1e-9
    assert np.allclose(fit["ratio"][np.isfinite(fit["ratio"])], 1.0, atol=1e-8)


def test_additive_effects_are_sum_to_zero_and_recover_the_differences():
    a, b = [3.0, 2.5, 2.0], [4.0, 3.0, 2.5, 2.0]
    fit = W.additive_fit(*_exact_additive(a, b))
    assert abs(float(np.sum(fit["a"]))) < 1e-6
    assert abs(float(np.sum(fit["b"]))) < 1e-6
    # Only differences are identified, so compare centred effects.
    assert np.allclose(fit["a"], np.array(a) - np.mean(a), atol=1e-6)
    assert np.allclose(fit["b"], np.array(b) - np.mean(b), atol=1e-6)
    assert abs(fit["mu"] - (np.mean(a) + np.mean(b))) < 1e-6


def test_additive_fit_detects_a_planted_interaction():
    """A planted 10x bump has to show up in the residual, not be absorbed."""
    med, cen = _exact_additive([3.0, 2.5, 2.0], [4.0, 3.0, 2.5, 2.0])
    med[0, 0] *= 10.0
    fit = W.additive_fit(med, cen)
    assert fit["ratio"][0, 0] > 3.0, "the bump was absorbed into the row/col effects"
    assert np.nanmax(fit["ratio"]) == pytest.approx(fit["ratio"][0, 0])


def test_additive_fit_ignores_censored_cells():
    med, cen = _exact_additive([3.0, 2.5, 2.0], [4.0, 3.0, 2.5, 2.0])
    med[2, 0] = 1e9          # a value that would wreck the fit if it were used
    cen[2, 0] = True
    fit = W.additive_fit(med, cen)
    assert math.isnan(fit["resid"][2, 0])
    assert np.nanmax(np.abs(fit["resid"])) < 1e-9


def test_additive_fit_refuses_an_underdetermined_grid():
    med, cen = _exact_additive([3.0, 2.5, 2.0], [4.0, 3.0, 2.5, 2.0])
    cen[:] = True
    cen[0, 0] = cen[1, 1] = False
    with pytest.raises(ValueError, match="identified cells"):
        W.additive_fit(med, cen)


def test_additive_dof_counts_identified_cells():
    med, cen = _exact_additive([3.0, 2.5, 2.0], [4.0, 3.0, 2.5, 2.0])
    fit = W.additive_fit(med, cen)
    assert fit["dof"] == 12 - (1 + 2 + 3)


# --------------------------------------------------------------------------
# the committed artifacts
# --------------------------------------------------------------------------

def test_every_cell_has_its_committed_logs():
    for cell, names in W.run_names().items():
        for name in names:
            assert (W.RUNS / f"{name}.json").exists(), f"{cell}: {name}.json"
            assert (W.RUNS / f"{name}.csv").exists(), f"{cell}: {name}.csv"


def test_load_cell_rejects_a_truncated_run(tmp_path):
    """A run that stopped early without grokking is a job failure, not a datum.

    This is the guard that keeps an interrupted training pass from showing up
    on the surface as a censored -- i.e. slow -- cell; the two are visually
    identical in the figure and only the step count tells them apart.
    """
    frac, wd = W.FRACS[0], W.WDS[0]
    for seed in W.SEEDS:
        name = W.cfg_for(frac, wd, seed).run_name()
        (tmp_path / f"{name}.json").write_text(json.dumps(
            {"grok_step": None, "steps_run": 4200}))
    with pytest.raises(ValueError, match="truncated run"):
        W.load_cell(frac, wd, runs_dir=tmp_path)


def test_load_cell_accepts_a_genuine_censored_run(tmp_path):
    frac, wd = W.FRACS[0], W.WDS[0]
    for seed in W.SEEDS:
        name = W.cfg_for(frac, wd, seed).run_name()
        (tmp_path / f"{name}.json").write_text(json.dumps(
            {"grok_step": None, "steps_run": W.MAX_STEPS}))
    assert W.load_cell(frac, wd, runs_dir=tmp_path) == [None] * len(W.SEEDS)


def test_surface_matches_the_committed_summaries():
    """The arrays the figure draws are the JSONs on disk, cell by cell."""
    s = W.surface()
    assert s["median"].shape == (len(W.FRACS), len(W.WDS))
    for i, frac in enumerate(W.FRACS):
        for j, wd in enumerate(W.WDS):
            vals = W.load_cell(frac, wd)
            med, cen = W.censored_median(vals)
            assert s["median"][i, j] == med
            assert bool(s["censored"][i, j]) == cen
            assert s["n_censored"][i, j] == sum(v is None for v in vals)


def test_wd_one_row_reproduces_the_frac_sweep_ordering():
    """The wd=1.0 row of this grid is §2's frac sweep, so it must agree.

    Not a tautology: §2 reports five seeds and this grid three, so agreement on
    the *ordering* (more data groks sooner) is a real check that the three-seed
    subset did not land on a non-representative corner of the five.
    """
    s = W.surface()
    j = W.WDS.index(1.0)
    col = [s["median"][i, j] for i in range(len(W.FRACS))]
    assert not s["censored"][:, j].any()
    assert col == sorted(col, reverse=True), f"frac ordering broke at wd=1: {col}"


def test_figure_replays_without_training(tmp_path):
    out = tmp_path / "surface.png"
    assert W.figure(out=out) == out
    assert out.stat().st_size > 10_000


def test_table_marks_censored_cells_rather_than_numbering_them():
    md = W.table()
    s = W.surface()
    n_censored_cells = int(s["censored"].sum())
    assert md.count(f"> {W.MAX_STEPS:,}") == n_censored_cells
    assert md.count("|") > 0


# --------------------------------------------------------------------------
# what the censored cells can and cannot establish
# --------------------------------------------------------------------------

def test_additive_predict_inverts_the_fit_on_open_cells():
    med, cen = _exact_additive([3.0, 2.5, 2.0], [4.0, 3.0, 2.5, 2.0])
    fit = W.additive_fit(med, cen)
    for i in range(med.shape[0]):
        for j in range(med.shape[1]):
            assert W.additive_predict(fit, i, j) == pytest.approx(med[i, j], rel=1e-8)


def test_censored_evidence_is_silent_when_the_bound_is_above_the_prediction():
    """A cell censored *below* what additivity predicts says nothing.

    If the null already expects the cell to run past the budget, then "it ran
    past the budget" is a confirmation of nothing -- and reporting it as
    evidence of a super-additive delay would be manufacturing a result out of
    where the budget happens to sit.
    """
    med, cen = _exact_additive([3.0, 2.5, 2.0], [4.0, 3.0, 2.5, 2.0])
    lo, hi = med.copy(), med.copy()          # zero seed spread -> no envelope
    cen[0, 0] = True                          # predicted 10^7, budget 25k
    ev = W.censored_evidence(med, cen, lo, hi)
    assert len(ev) == 1
    assert ev[0]["bound_ratio"] < 1.0
    assert ev[0]["refutes_additivity"] is False


def test_censored_evidence_flags_a_cell_that_really_is_too_slow():
    """With no seed spread, a cell censored well past its prediction refutes."""
    a, b = [1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]   # every cell predicts 100
    med, cen = _exact_additive(a, b)
    lo, hi = med.copy(), med.copy()
    cen[0, 0] = True
    ev = W.censored_evidence(med, cen, lo, hi)
    assert ev[0]["prediction"] == pytest.approx(100.0)
    assert ev[0]["bound_ratio"] == pytest.approx(W.MAX_STEPS / 100.0)
    assert ev[0]["refutes_additivity"] is True


def test_censored_evidence_envelope_straddling_one_withdraws_the_claim():
    """Seed spread wide enough to move the prediction past the budget mutes it.

    This is the guard that matters on the real grid: the wd=0 column effect
    rests on one open cell, so its seed range propagates into a very wide
    envelope, and a point estimate above 1 is not evidence on its own.
    """
    a, b = [1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]
    med, cen = _exact_additive(a, b)
    cen[0, 0] = True
    lo = np.full_like(med, 10.0)
    hi = np.full_like(med, 1e6)              # a decade either side of every cell
    ev = W.censored_evidence(med, cen, lo, hi)
    assert ev[0]["ratio_lo"] < 1.0 < ev[0]["ratio_hi"]
    assert ev[0]["refutes_additivity"] is False


def test_censored_evidence_envelope_brackets_the_point_estimate():
    s = W.surface()
    for e in W.censored_evidence(s["median"], s["censored"], s["lo"], s["hi"]):
        assert e["ratio_lo"] <= e["bound_ratio"] <= e["ratio_hi"]


def test_censored_evidence_guards_against_an_exponential_envelope():
    med = np.ones((5, 5)) * 100.0
    cen = np.zeros(med.shape, dtype=bool)
    cen[0, 0] = True
    with pytest.raises(ValueError, match="too many to enumerate"):
        W.censored_evidence(med, cen, med.copy(), med.copy())


def test_substitution_check_needs_a_monotone_split():
    """A non-monotone blocked/grokking split is a different finding.

    "More data substitutes for regularization" means the blocked fractions sit
    *below* the grokking ones. If a middle fraction grokked and a larger one
    did not, that is something else entirely and must not be reported under
    this heading.
    """
    cen = np.array([[True], [False], [True]])
    n_cen = np.array([[3], [0], [3]])
    assert W.substitution_check(cen, n_cen)["substitutes"] is False


def test_substitution_check_on_the_committed_grid():
    s = W.surface()
    sub = W.substitution_check(s["censored"], s["n_censored"])
    assert sub["wd"] == 0.0, "the check must look down the least-regularized column"
    assert sub["substitutes"] is True
    assert sub["blocked"] == [0.25, 0.30]
    assert sub["grokking"] == [0.40]


def test_substitution_check_is_all_or_nothing():
    cen = np.array([[True], [True], [True]])
    n_cen = np.array([[3], [3], [3]])
    assert W.substitution_check(cen, n_cen)["substitutes"] is False
    cen = np.zeros((3, 1), dtype=bool)
    assert W.substitution_check(cen, np.zeros((3, 1), int))["substitutes"] is False


def test_the_wd_zero_grok_is_a_real_generalization():
    """The headline cell has to actually generalize, not just cross 0.99 once.

    ``grok_step`` fires at test accuracy 0.99, and a cell that touched it and
    fell back would make "wd=0 groks at 40% data" a claim about one eval. Every
    seed's *final* test accuracy is checked instead.
    """
    for seed in W.SEEDS:
        name = W.cfg_for(0.40, 0.0, seed).run_name()
        with open(W.RUNS / f"{name}.json") as fh:
            summary = json.load(fh)
        assert summary["grok_step"] is not None, f"seed {seed} did not grok"
        assert summary["final_test_acc"] > 0.99, (
            f"seed {seed} ended at {summary['final_test_acc']}")


def test_the_wd_zero_grok_happens_with_a_monotone_weight_norm():
    """Sec. 5's norm decline is a weight-decay signature, not a grok signature.

    The wd=0 runs reach ~0.999 test accuracy with a parameter norm that never
    once falls below its running peak, which is the whole content of the §5
    caveat this grid adds. Asserted here so the claim cannot drift from the
    committed trajectories.
    """
    import csv as _csv
    for seed in W.SEEDS:
        name = W.cfg_for(0.40, 0.0, seed).run_name()
        with open(W.RUNS / f"{name}.csv", newline="") as fh:
            norms = [float(r["weight_norm"]) for r in _csv.DictReader(fh)]
        peak = norms[0]
        for value in norms:
            assert value >= peak - 1e-9, f"seed {seed} norm fell from {peak}"
            peak = max(peak, value)


def test_readme_grid_table_is_the_generated_one():
    """The §13 table in the README must be what ``table()`` prints, verbatim.

    Every stale number this repo has shipped got there the same way: a table
    hand-copied into the README, then the code underneath it changed. The grid
    is the biggest such table in the repo, so it is pinned rather than trusted.
    """
    generated = W.table().split("\n\nAdditive")[0].strip()
    readme = (ROOT / "README.md").read_text()
    assert generated in readme, (
        "README §13's grid table has drifted from wd_frac_surface.table():\n"
        + generated)


def test_readme_reports_the_substitution_result_it_measured():
    """The headline sentence and the measurement must agree on the fractions."""
    s = W.surface()
    sub = W.substitution_check(s["censored"], s["n_censored"])
    readme = (ROOT / "README.md").read_text()
    assert sub["substitutes"] is True
    # The claim in prose is "25% and 30% never grok, 40% does" at wd = 0.
    assert sub["blocked"] == [0.25, 0.30] and sub["grokking"] == [0.40]
    assert "the §1 sentence" in readme and "30%-data* statement" in readme


def test_readme_does_not_claim_the_censored_cells_refute_additivity():
    """The envelope withdrew that claim; the prose must not smuggle it back.

    This is the assertion guarding the section's one genuine near-miss: the
    (30%, 0) bound ratio is 2.29x, larger than the fit's worst open-cell
    residual, and it would have read as a refutation if the seed-range envelope
    had not been computed. If a later edit ever makes a censored cell genuinely
    refute additivity, this test fails and the prose has to be rewritten -- which
    is the point.
    """
    s = W.surface()
    evidence = W.censored_evidence(s["median"], s["censored"], s["lo"], s["hi"])
    assert not any(e["refutes_additivity"] for e in evidence)
    for e in evidence:
        assert e["ratio_lo"] < 1.0 < e["ratio_hi"], (
            f"cell frac={e['frac']} wd={e['wd']} no longer straddles 1")
    readme = (ROOT / "README.md").read_text()
    assert "every one straddles 1" in readme
