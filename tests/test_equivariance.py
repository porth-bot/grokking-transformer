"""The swap read-outs, and the control Sec. 11 was missing.

Two kinds of test here, and the distinction matters. The first kind checks the
*statistics* against constructions whose symmetry is known by hand -- an exactly
commutative function must score 0, an exactly anti-equivariant one must score 0
on the other statistic, and the index permutations must be the permutations they
claim to be. Those cannot fail for an interesting reason; they fail when the
code is wrong.

The second kind pins what the committed checkpoints actually measure, which is
Sec. 11's hypothesis under test: grokking on addition buys commutativity, and
grokking on subtraction buys the symmetry subtraction has instead. If a future
change to the model or the runs moves those, the claim in the README moves with
them and should not do so quietly.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

pytest.importorskip("matplotlib")
import swap_equivariance  # noqa: E402

from grokking.checkpoints import load_model
from grokking.data import modular_dataset
from grokking.equivariance import (
    logit_defect,
    measure_equivariance,
    negate_index,
    shuffle_baseline,
    swap_index,
)

P = 97
GROKKED = 0.8


# -- the index permutations --------------------------------------------------


def test_the_swap_index_is_the_swap():
    """It has to send the row for (a, b) to the row for (b, a), and be an
    involution -- the property Exercise 3's "the dataset is closed under the
    swap" argument rests on."""
    tokens, _ = modular_dataset(P, "add")
    swap = swap_index(P)
    assert bool((tokens[swap][:, 0] == tokens[:, 1]).all())
    assert bool((tokens[swap][:, 1] == tokens[:, 0]).all())
    assert bool((swap[swap] == torch.arange(P * P)).all())


def test_the_negate_index_negates_the_answer():
    """Indexing the class axis with it must send class c to class -c mod p, and
    it must fix 0 (the only self-negating residue for odd p)."""
    neg = negate_index(P)
    assert int(neg[0]) == 0
    for c in (1, 5, 96):
        assert int(neg[c]) == (-c) % P
    assert bool((neg[neg] == torch.arange(P)).all())
    assert len(set(neg.tolist())) == P                 # a permutation


# -- the statistics, against functions whose symmetry is known ---------------


def _one_hot_logits(answers: torch.Tensor, scale: float = 10.0) -> torch.Tensor:
    return torch.nn.functional.one_hot(answers, P).float() * scale


def test_an_exactly_commutative_function_has_zero_invariance_defect():
    """(a + b) mod p as a hard-coded logit table: invariance defect exactly 0,
    and -- the part worth checking -- anti-equivariance defect *large*, because
    negating addition's answer is wrong. Without the second assertion the two
    statistics could be the same statistic and this would not notice.

    The wrong-symmetry defect is compared against this table's *own* shuffle
    baseline rather than an absolute level: a one-hot table's logits have a
    very different spread from a trained model's, so its no-symmetry level is
    0.20 rather than ~1.1, and a hardcoded threshold here would be measuring
    the encoding rather than the symmetry."""
    _, answers = modular_dataset(P, "add")
    logits = _one_hot_logits(answers)
    assert logit_defect(logits, P, negate=False) == pytest.approx(0.0, abs=1e-7)
    assert logit_defect(logits, P, negate=True) == pytest.approx(
        shuffle_baseline(logits), rel=0.05)


def test_an_exactly_anti_equivariant_function_has_zero_anti_defect():
    """The mirror image, on (a - b) mod p."""
    _, answers = modular_dataset(P, "sub")
    logits = _one_hot_logits(answers)
    assert logit_defect(logits, P, negate=True) == pytest.approx(0.0, abs=1e-7)
    assert logit_defect(logits, P, negate=False) == pytest.approx(
        shuffle_baseline(logits), rel=0.05)


def test_the_shuffle_baseline_is_the_no_symmetry_level():
    """A random logit table respects nothing, so its swap defect and its
    shuffle baseline must agree -- which is what licenses reading the baseline
    as "this is what no symmetry looks like"."""
    torch.manual_seed(0)
    logits = torch.randn(P * P, P)
    assert logit_defect(logits, P) == pytest.approx(shuffle_baseline(logits),
                                                    rel=0.05)


# -- what the committed read-outs say ----------------------------------------
#
# The sweep's checkpoints are gitignored (12 MB of weights, and .gitignore
# commits only the two the Fourier figure needs), so the claims below are
# tested against the committed CSV -- the same artifact the figure replays
# from. `test_the_committed_csv_still_matches_a_live_measurement` is what stops
# that being a test of a file against itself: it re-measures the one run whose
# checkpoints *are* committed and requires the CSV to agree.

CSV = ROOT / "runs" / f"{swap_equivariance.NAME}.csv"


def _rows():
    return swap_equivariance.collect(CSV)


def _cell(op, wd, which, field, grokked_only=False):
    return np.array([r[field] for r in _rows()
                     if r["op"] == op and r["wd"] == wd and r["which"] == which
                     and (not grokked_only or r["test_acc"] >= GROKKED)])


def test_grokking_on_addition_buys_invariance_and_not_anti_equivariance():
    mem = _cell("add", "1", "memorize", "logit_swap_defect")
    fin = _cell("add", "1", "final", "logit_swap_defect", grokked_only=True)
    assert len(fin) == 3
    # commutativity is acquired, by well over an order of magnitude
    assert mem.min() > 0.4
    assert fin.max() < 0.05
    # ...and the symmetry addition does NOT have is not acquired
    assert _cell("add", "1", "final", "logit_anti_defect",
                 grokked_only=True).min() > 0.9


def test_grokking_on_subtraction_buys_the_symmetry_subtraction_has():
    """Sec. 11's control. Subtraction cannot be commutative, so if the addition
    result were a fact about grokking rather than about the task, this would
    look the same. It does not: the invariance defect *rises* to the
    no-symmetry level while the anti-equivariance defect falls.
    """
    base = np.median([r["shuffle_baseline"] for r in _rows()])
    mem_sw = _cell("sub", "1", "memorize", "logit_swap_defect")
    fin_sw = _cell("sub", "1", "final", "logit_swap_defect", grokked_only=True)
    mem_an = _cell("sub", "1", "memorize", "logit_anti_defect")
    fin_an = _cell("sub", "1", "final", "logit_anti_defect", grokked_only=True)
    assert len(fin_sw) == 3

    # no commutativity, and grokking moves away from it, not toward it
    assert np.median(fin_sw) > np.median(mem_sw)
    assert fin_sw.min() > 0.9 * base
    # the licensed symmetry is acquired instead
    assert mem_an.min() > 0.9
    assert fin_an.max() < 0.6
    assert np.median(fin_an) < np.median(mem_an) / 1.7


def test_subtraction_acquires_its_symmetry_far_less_cleanly_than_addition():
    """The honest asymmetry in the result, pinned so it is not rounded away.

    Addition's defect lands at 0.010 across a 0.03 spread; subtraction's at
    0.271 across 0.15-0.51. Both are far below the no-symmetry level and only
    one is close to exact, and that gap is consistent with subtraction being
    the slowest of the three operations (Sec. 11).
    """
    add = _cell("add", "1", "final", "logit_swap_defect", grokked_only=True)
    sub = _cell("sub", "1", "final", "logit_anti_defect", grokked_only=True)
    assert add.max() < 0.05
    assert np.median(sub) > 10 * np.median(add)
    assert sub.max() / sub.min() > 2          # and it is seed-unstable


def test_the_wd_0_1_subtraction_cell_is_one_run_and_is_reported_as_one():
    """Two of its three seeds never generalized (Sec. 11 measured the same),
    so anything read off that cell is a single run. The grokked-only filter has
    to actually drop them -- averaging a failed run into a claim about what
    grokking does would miss the entire point."""
    all_three = _cell("sub", "0.1", "final", "logit_anti_defect")
    grokked = _cell("sub", "0.1", "final", "logit_anti_defect",
                    grokked_only=True)
    assert len(all_three) == 3 and len(grokked) == 1
    # and the two that failed sit near the no-symmetry level, as they should
    failed = sorted(set(all_three.tolist()) - set(grokked.tolist()))
    assert min(failed) > 0.8


def test_the_committed_csv_still_matches_a_live_measurement():
    """The artifact against the code, on the one run whose weights are in the
    repo. This is the check that keeps the CSV from going stale the way
    figures have twice in this repo's history."""
    tokens, _ = modular_dataset(P, "add")
    for which in ("memorize", "final"):
        model, _ = load_model("p97_frac0.30_wd1_seed0", which=which)
        model.eval()
        live = measure_equivariance(model, tokens, P)
        row = next(r for r in _rows()
                   if r["run"] == "p97_frac0.30_wd1_seed0"
                   and r["which"] == which)
        for field, value in live.items():
            assert row[field] == pytest.approx(value, abs=1e-6), (which, field)


def test_the_attention_statistic_agrees_with_the_published_addition_numbers():
    """Sec. 11's own read-out, on the run the appendix reports: 0.189 at
    memorization and 0.00017 after grokking."""
    tokens, _ = modular_dataset(P, "add")
    d = {}
    for which in ("memorize", "final"):
        model, _ = load_model("p97_frac0.30_wd1_seed0", which=which)
        model.eval()
        d[which] = measure_equivariance(model, tokens, P)["attn_equivariance_defect"]
    assert d["memorize"] == pytest.approx(0.189, abs=0.005)
    assert d["final"] < 0.001
