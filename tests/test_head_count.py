"""Section 9's head-count ablation, now five seeds per arm (issue #4, part 1).

Two kinds of check. The claims the README makes are asserted against the
committed artifacts -- the 15 run logs and the read-out CSV -- so a rerun that
moves a number fails here instead of leaving the prose stale. And because the
read-out CSV is a *cache* (``seeds.fill_table`` only measures runs it does not
already have), one test re-measures the single run whose checkpoint is in the
repo and requires the cached row to agree; that is what stops the cache from
silently outliving the weights it came from.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

pytest.importorskip("matplotlib")

import head_count  # noqa: E402

from grokking.aggregate import rank_sum_test  # noqa: E402

MAIN = "p97_frac0.30_wd1_seed0"


def _grok():
    return {h: [s["grok_step"] for _, s in runs]
            for h, runs in head_count.collect().items()}


# -- what the ablation varies, and what it must not --------------------------

def test_only_the_head_count_varies_across_arms():
    """A timing comparison across arms is only about heads if nothing else
    moves -- including the parameter count, which d_model/n_heads keeps fixed."""
    cfgs = {h: head_count.cfg_for(h, 0) for h in head_count.HEADS}
    base = cfgs[4]
    for h, c in cfgs.items():
        assert (c.p, c.train_frac, c.weight_decay, c.lr, c.max_steps,
                c.seed) == (base.p, base.train_frac, base.weight_decay,
                            base.lr, base.max_steps, base.seed)
        assert c.model.d_model == base.model.d_model
        assert c.model.d_mlp == base.model.d_mlp
        assert c.model.n_layers == base.model.n_layers
        assert c.model.n_heads == h


def test_the_four_head_arm_is_the_main_sweep_runs_not_retrained_copies():
    assert head_count.run_names()[4][0] == MAIN
    assert all("_h" not in n for n in head_count.run_names()[4])
    assert all(n.endswith("_h1") for n in head_count.run_names()[1])


def test_every_arm_has_five_seeds_and_every_run_is_committed():
    names = head_count.run_names()
    assert {h: len(v) for h, v in names.items()} == {1: 5, 2: 5, 4: 5}
    assert len({n for v in names.values() for n in v}) == 15
    for v in names.values():
        for n in v:
            for ext in (".csv", ".json"):
                assert (ROOT / "runs" / f"{n}{ext}").exists(), f"{n}{ext}"


# -- the timing result -------------------------------------------------------

def test_every_run_in_every_arm_grokked():
    """The comparison is of grok *times*; an arm that failed to generalize
    would make the medians meaningless."""
    for h, runs in head_count.collect().items():
        for _, s in runs:
            assert s["final_test_acc"] >= head_count.GROKKED_ACC, (h, s)
            assert s["grok_step"] is not None


def test_fewer_heads_grok_sooner_with_complete_separation():
    """The README's ordering: 1 < 2 < 4 heads, and no seed of a smaller-head
    arm is slower than any seed of a larger one."""
    grok = _grok()
    assert [np.median(grok[h]) for h in (1, 2, 4)] == [300, 700, 1300]
    for a, b in ((1, 2), (2, 4), (1, 4)):
        r = rank_sum_test(grok[a], grok[b])
        assert r["superiority"] == 0.0, (a, b)          # complete separation
        assert r["p_two_sided"] == pytest.approx(2 / 252)


def test_the_between_arm_gaps_exceed_the_within_arm_spread():
    """Why the arms separate at all: the seed spread inside an arm (1.3-1.6x)
    is smaller than the step between arms (1.9-2.3x on medians)."""
    grok = _grok()
    within = max(max(grok[h]) / min(grok[h]) for h in head_count.HEADS)
    between = min(np.median(grok[b]) / np.median(grok[a])
                  for a, b in ((1, 2), (2, 4)))
    assert within < 1.7 < 1.8 < between


def test_seed_zero_was_the_slowest_seed_in_every_arm():
    """Worth pinning because §9 used to report seed 0 alone: its 400/900/1900
    is the *maximum* of each arm, so the shipped numbers were uniformly the
    pessimistic end -- while the ratios between them survived averaging."""
    grok = _grok()
    for h, seed0 in ((1, 400), (2, 900), (4, 1900)):
        assert grok[h][0] == seed0
        assert seed0 == max(grok[h])


# -- the mechanistic read-out ------------------------------------------------

def test_the_readout_csv_covers_exactly_the_fifteen_runs():
    rows = head_count.load_readouts()
    names = {n for v in head_count.run_names().values() for n in v}
    assert {r["run"] for r in rows} == names
    assert len(rows) == 15


def test_every_arm_reaches_the_algorithmic_symmetry():
    """ln 2 operand entropy on every head count: the grokked circuit reads the
    two operands evenly whether it has one head or four."""
    for r in head_count.load_readouts():
        assert r["attn_operand_entropy"] == pytest.approx(np.log(2), abs=2e-4)
        assert r["attn_asymmetry"] < 0.02


def test_the_end_state_does_not_separate_the_arms():
    """The counterweight to the timing result, and the reason §9's mechanistic
    guess is not supported: no attention read-out distinguishes the head counts
    at the same sample size where grok step separates all three pairs."""
    rows = head_count.load_readouts()
    for field in head_count.READOUTS:
        for a, b in ((1, 2), (2, 4), (1, 4)):
            r = rank_sum_test([x[field] for x in rows if x["n_heads"] == a],
                              [x[field] for x in rows if x["n_heads"] == b])
            assert r["p_two_sided"] > 0.05, (field, a, b, r["p_two_sided"])


def test_the_committed_csv_still_matches_a_live_measurement():
    """The cache against the code, on the one run whose weights are in the
    repo. Without this the CSV is only ever tested against itself."""
    from grokking.attention import measure_attention
    from grokking.checkpoints import load_model
    from grokking.data import modular_dataset

    tokens, _ = modular_dataset(97, "add")
    model, _ = load_model(MAIN, which="final")
    model.eval()
    live = measure_attention(model, tokens)
    row = next(r for r in head_count.load_readouts() if r["run"] == MAIN)
    for field, value in live.items():
        assert row[field] == pytest.approx(value, abs=1e-6), field
