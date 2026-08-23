"""Sec. 12's seed sweep: what the five seeds support, asserted (issue #4).

Three kinds of check.

The claims Sec. 12 and the rewritten Sec. 5 / Sec. 8 / appendix make are pinned
against the committed read-out CSV, so a rerun that moves a number fails here
rather than leaving the prose behind. That includes the *negative* ones -- the
operand-weight contrast that does not separate completely, and the
embedding/logit frequency overlap that does not distinguish the checkpoints --
because those are exactly the claims a future edit would be tempted to smooth
over.

The CSV is a cache (``seeds.fill_table`` only measures keys it lacks), so one
test re-measures the two rows whose checkpoint is committed and requires the
cache to agree.

And one test pins a *definition* rather than a result: the ring statistics in
the table use each checkpoint's own dominant frequency while the figure uses
the final model's for both panels, and the two give 0.399 and 0.412 for the
same checkpoint. Left untested, that reads as a contradiction between a table
and a figure caption.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import mechanistic_seeds as ms  # noqa: E402

from grokking.aggregate import rank_sum_test  # noqa: E402

MAIN = "p97_frac0.30_wd1_seed0"


@pytest.fixture(scope="module")
def rows():
    return ms.load_readouts()


def _arms(rows, field):
    return ms.arm(rows, "memorize", field), ms.arm(rows, "final", field)


# -- what is in the table ----------------------------------------------------

def test_the_csv_covers_five_seeds_at_both_checkpoints(rows):
    assert len(rows) == 10
    assert {r["key"] for r in rows} == set(ms.keys())
    assert sorted(r["seed"] for r in rows) == sorted(list(ms.SEEDS) * 2)
    assert ms.run_names()[0] == MAIN


def test_every_final_checkpoint_actually_grokked(rows):
    """A read-out of a run that did not generalize would not be a read-out of
    the generalizing circuit."""
    assert all(a == 1.0 for a in ms.arm(rows, "final", "logit_acc_full"))
    assert max(ms.arm(rows, "memorize", "logit_acc_full")) < 0.35


# -- the twelve read-outs that separate completely ---------------------------

@pytest.mark.parametrize("field", [f for f, _, _ in ms.READOUTS
                                   if f != "attn_operand_frac"])
def test_the_read_outs_separate_the_checkpoints_completely(rows, field):
    """No memorization checkpoint on the wrong side of any final one, i.e. the
    smallest p-value five vs five can produce."""
    mem, fin = _arms(rows, field)
    mem = [v for v in mem if not np.isnan(v)]
    fin = [v for v in fin if not np.isnan(v)]
    if not mem:                       # "99% is never reached at memorization"
        assert field == "logit_freqs_for_99" and len(fin) == 5
        return
    r = rank_sum_test(mem, fin)
    assert r["superiority"] in (0.0, 1.0), (field, r["superiority"])
    assert r["p_two_sided"] == pytest.approx(2 / 252)


def test_the_operand_weight_is_the_one_contrast_that_does_not_separate(rows):
    """The appendix's 99.7% -> 83.7% is the extreme pair, not the effect.

    Two of the five grokked runs keep more operand weight than the *least*
    concentrated memorizing one, so the arms overlap; the difference is real
    (p = 0.032) but it is 98.1% -> 90.9%, not 99.7% -> 83.7%.
    """
    mem, fin = _arms(rows, "attn_operand_frac")
    r = rank_sum_test(mem, fin)
    assert 0.0 < r["p_two_sided"] <= 0.05
    assert r["superiority"] not in (0.0, 1.0)      # not complete separation
    assert max(fin) > min(mem)                     # the ranges overlap
    assert np.mean(mem) == pytest.approx(0.981, abs=5e-3)
    assert np.mean(fin) == pytest.approx(0.909, abs=5e-3)


# -- the headline: where seed 0 sits -----------------------------------------

def test_seed_zero_is_the_most_flattering_seed_on_most_read_outs(rows):
    """The reason the issue existed. Sec. 9 found seed 0 was the *slowest* run
    in all three arms -- pessimistic on timing. On the mechanistic read-outs it
    is the opposite: it holds the extreme in the direction the README's story
    points on 8 of the 13, and lies outside the other four's whole range on 7,
    which is why the published contrasts are the largest available rather than
    typical."""
    best, total = ms.seed_zero_summary(rows)
    assert (best, total) == (8, 13)
    assert set(ms.seed_zero_outside(rows)) == {
        "emb_top5_energy", "emb_freqs_90", "emb_var_in_plane",
        "attn_operand_entropy", "attn_asymmetry", "eq_attn_defect",
        "eq_logit_swap",
    }


def test_the_five_runs_are_not_the_same_length(rows):
    """The confound behind the line above, and the reason it is not read as
    "seed 0 is a lucky seed": seed 0's run was extended to 11,100 steps (Sec. 5
    explains why) and the others early-stop on patience at ~2,000, so its final
    checkpoint has ~9,000 extra steps of decay after grokking -- over which
    Sec. 5's own numbers show the circuit continuing to sparsify."""
    steps = {r["seed"]: r["steps_run"] for r in rows if r["which"] == "final"}
    assert steps[0] == 11_100
    assert all(1_800 <= steps[s] <= 2_100 for s in ms.MATCHED_SEEDS)
    assert steps[0] - max(steps[s] for s in ms.MATCHED_SEEDS) == 9_000
    # Both checkpoints of a run are the same run, so the columns must agree.
    for seed in ms.SEEDS:
        pair = [r["steps_run"] for r in rows if r["seed"] == seed]
        assert pair[0] == pair[1]
    # Memorization, by contrast, is at step 100 in every seed -- that arm is
    # matched, which is why only the "final" arm carries the caveat.
    assert {r["memorize_step"] for r in rows} == {100.0}


def test_dropping_the_long_run_leaves_the_conclusions_standing(rows):
    """The robustness check the confound demands. Four seeds per arm still
    permits a complete separation (C(8,4) = 70, floor p = 0.029), and 12 of 13
    read-outs still get one -- so the long run carries the effect *sizes* and
    none of the conclusions. The exception is the same read-out that was
    already the weakest at five seeds."""
    checks = ms.matched_length_check(rows)
    incomplete = [(f, p) for f, p, complete in checks if not complete]
    assert len(checks) - len(incomplete) == 12
    assert [f for f, _ in incomplete] == ["attn_operand_frac"]
    assert incomplete[0][1] == pytest.approx(0.114, abs=1e-3)
    for field, p, complete in checks:
        if complete and not np.isnan(p):
            assert p == pytest.approx(2 / 70), field


def test_the_published_symmetry_numbers_are_the_extremes_of_their_arms(rows):
    """0.189 -> 0.00017 is min-to-max: the mean contrast is 28x, not 1100x."""
    mem, fin = _arms(rows, "eq_attn_defect")
    assert mem[0] == max(mem) and fin[0] == min(fin)
    assert mem[0] / fin[0] > 1000
    assert np.mean(mem) / np.mean(fin) == pytest.approx(28.7, rel=0.05)


def test_one_seed_moves_the_operand_weight_the_wrong_way(rows):
    """The concrete form of "it does not separate", and the reason a complete
    separation is worth asserting elsewhere: for the twelve read-outs that
    separate, no seed can move against the story (that is what separation
    *means*, so those per-seed directions are already covered). Here seed 2's
    operand weight *rises* through grokking, 0.960 -> 0.966, against a
    published contrast that is a fall."""
    mem, fin = _arms(rows, "attn_operand_frac")
    reversed_seeds = [s for s, m, f in zip(ms.SEEDS, mem, fin) if f > m]
    assert reversed_seeds == [2]
    assert fin[2] - mem[2] == pytest.approx(0.006, abs=1e-3)


# -- Sec. 8's restricted accuracy --------------------------------------------

def test_three_frequencies_is_seed_zero_and_it_takes_three_to_five(rows):
    m99 = ms.arm(rows, "final", "logit_freqs_for_99")
    assert m99[0] == 3 and min(m99) == 3 and max(m99) == 5
    assert np.mean(m99) == pytest.approx(3.8)
    acc3 = ms.arm(rows, "final", "logit_acc_m3")
    assert acc3[0] == 1.0                        # the published 1.00
    assert min(acc3) == pytest.approx(0.928, abs=1e-3)


def test_memorization_never_rebuilds_the_task_from_a_few_frequencies(rows):
    for m in ms.RESTRICT_MS:
        assert max(ms.arm(rows, "memorize", f"logit_acc_m{m}")) < 0.9
    assert all(np.isnan(v) for v in ms.arm(rows, "memorize", "logit_freqs_for_99"))


def test_the_circuit_is_already_there_under_the_memorization_in_every_seed(rows):
    """Sec. 8's most interesting claim, and one that got *stronger* with seeds:
    projecting the memorizing logits onto the a+b subspace recovers far more
    test accuracy than the model itself expresses -- 0.82 against 0.25."""
    top10 = ms.arm(rows, "memorize", "logit_acc_m10")
    own = ms.arm(rows, "memorize", "logit_acc_full")
    assert all(t > o for t, o in zip(top10, own))
    assert min(top10) > max(own)                 # complete separation
    assert np.mean(top10) == pytest.approx(0.817, abs=5e-3)
    assert np.mean(own) == pytest.approx(0.247, abs=5e-3)


# -- the two claims that turned out not to be grokking signatures ------------

def test_the_frequency_overlap_is_above_chance_but_says_nothing_about_grokking(rows):
    """Sec. 8 offered the embedding/logit frequency overlap as evidence that
    the logits are written in the basis the embeddings carry. It is far above
    chance -- and just as high at memorization, so it does not separate the
    checkpoints and cannot be evidence *about grokking*."""
    mem, fin = _arms(rows, "emb_diag_overlap")
    chance = 25 / ms.N_FREQS
    assert min(mem + fin) > 3 * chance
    assert rank_sum_test(mem, fin)["p_two_sided"] > 0.05
    assert np.mean(mem) >= np.mean(fin)


def test_the_memorization_ring_statistics_sit_at_the_unstructured_baseline(rows):
    """"Diffuse" understates it: at memorization the variance in the ring plane
    is the level pure Gaussian embeddings reach (0.043 against 0.040), so that
    statistic carries no signal there at all. The radial CV does carry some
    (0.343 against 0.428), and it is the one that separates."""
    from grokking.mechanistic import random_ring_baseline

    base = random_ring_baseline(128, 97, n_draws=8)
    mem_var, fin_var = _arms(rows, "emb_var_in_plane")
    mem_cv, fin_cv = _arms(rows, "emb_radial_cv")
    assert np.mean(mem_var) == pytest.approx(base["var_in_plane"], rel=0.15)
    assert np.mean(fin_var) > 2.0 * base["var_in_plane"]
    assert 0.7 < np.mean(mem_cv) / base["radial_cv"] < 1.0
    assert np.mean(fin_cv) < 0.4 * base["radial_cv"]


# -- the cache, against the code ---------------------------------------------

def test_the_committed_csv_still_matches_a_live_measurement(rows):
    """Both of seed 0's rows re-measured from the weights in the repo. Without
    this the other eight rows are only ever tested against themselves."""
    from grokking.checkpoints import load_model
    from grokking.mechanistic import measure_checkpoint

    for which in ms.WHICH:
        model, summary = load_model(MAIN, which=which)
        cfg = summary["config"]
        live = measure_checkpoint(model, int(cfg["p"]), float(cfg["train_frac"]),
                                  int(cfg["seed"]))
        row = next(r for r in rows if r["key"] == f"{MAIN}@{which}")
        for field, value in live.items():
            if np.isnan(value):
                assert np.isnan(row[field]), field
            else:
                assert row[field] == pytest.approx(value, rel=1e-6), (which, field)


def test_the_table_and_the_figure_measure_the_ring_in_different_planes(rows):
    """A definition, pinned so it cannot become a contradiction.

    ``embedding_circle.py`` projects both checkpoints onto the *final* model's
    dominant frequency (so the two panels are comparable); this table gives each
    checkpoint its own. Own-plane can only flatter the memorizing model, so the
    table's 0.399 is the conservative number and the figure's 0.412 the
    generous one -- and that ordering is what is asserted, not the digits.
    """
    from grokking.checkpoints import load_model
    from grokking.mechanistic import dominant_frequency, frequency_projection

    mem, _ = load_model(MAIN, which="memorize")
    fin, _ = load_model(MAIN, which="final")
    E_mem = mem.tok_emb.weight.detach().cpu().numpy()[:97]
    k_final = dominant_frequency(fin.tok_emb.weight.detach().cpu().numpy()[:97], 97)
    _, _, _, shared_cv = frequency_projection(E_mem, k_final, 97)
    own_cv = ms.arm(rows, "memorize", "emb_radial_cv")[0]
    assert own_cv < shared_cv
    assert own_cv == pytest.approx(0.399, abs=1e-3)
    assert shared_cv == pytest.approx(0.412, abs=1e-3)
