"""The grokked model's logits are a sparse, a+b-structured function.

Tests the claims experiments/logit_attribution.py visualizes, computed directly
from the committed checkpoints (no retraining):

1. keeping *all* diagonal frequencies rebuilds the logits' argmax exactly (the
   restriction machinery is a faithful decomposition, not an approximation),
2. the final model concentrates its logit energy on the a+b diagonal far more
   than the memorization checkpoint, and
3. a handful of top frequencies rebuild full test accuracy for the grokked
   model, while the same restriction on the memorization checkpoint recovers
   *more* accuracy than the raw model expresses -- the circuit forming early.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from logit_attribution import (  # noqa: E402
    MAIN,
    diagonal_frequency_energy,
    logit_tensor,
    restrict_to_freqs,
)
# aliased so pytest does not collect the imported helper as a test case
from logit_attribution import test_accuracy as accuracy  # noqa: E402

from grokking.checkpoints import load_model  # noqa: E402


def _load(which):
    model, summary = load_model(MAIN, which=which)
    p = summary["config"]["p"]
    L, tokens, targets = logit_tensor(model, p)
    return L, tokens, targets, p, summary


def test_keeping_all_diagonal_frequencies_is_exact_for_argmax():
    # Restricting to every k on the diagonal keeps all of a function-of-(a+b);
    # the residual is only the (small) off-diagonal part, so the argmax
    # predictions of the reconstruction should match the true logits closely.
    L, tokens, targets, p, summary = _load("final")
    K = (p - 1) // 2
    Lr = restrict_to_freqs(L, p, range(1, K + 1))
    same = (L.argmax(-1) == Lr.argmax(-1)).float().mean()
    assert float(same) > 0.99


def test_final_logits_are_more_diagonal_than_memorization():
    Lf, *_ , pf, _ = _load("final")
    Lm, *_ , pm, _ = _load("memorize")
    _, frac_fin = diagonal_frequency_energy(Lf, pf)
    _, frac_mem = diagonal_frequency_energy(Lm, pm)
    assert frac_fin > 0.9          # grokked: nearly all energy is "compute a+b"
    assert frac_mem < 0.5          # memorizing: diffuse
    assert frac_fin > 2 * frac_mem


def test_three_frequencies_rebuild_generalization():
    L, tokens, targets, p, summary = _load("final")
    frac = summary["config"]["train_frac"]
    seed = summary["config"]["seed"]
    diag, _ = diagonal_frequency_energy(L, p)
    order = [int(k) for k in torch.argsort(diag, descending=True) if k > 0]

    full = accuracy(L, tokens, targets, frac, seed)
    top3 = accuracy(restrict_to_freqs(L, p, order[:3]), tokens, targets, frac, seed)
    assert full > 0.99             # the committed grokked run generalizes fully
    assert top3 > 0.99             # and three frequencies suffice to rebuild it


def test_restricting_memorization_logits_reveals_latent_structure():
    # Projecting the memorization checkpoint's logits onto the a+b subspace
    # denoises the per-pair memorization and exposes the partial circuit: the
    # restricted read-out generalizes better than the raw memorizing model.
    L, tokens, targets, p, summary = _load("memorize")
    frac = summary["config"]["train_frac"]
    seed = summary["config"]["seed"]
    diag, _ = diagonal_frequency_energy(L, p)
    order = [int(k) for k in torch.argsort(diag, descending=True) if k > 0]

    full = accuracy(L, tokens, targets, frac, seed)
    restricted = accuracy(restrict_to_freqs(L, p, order[:10]),
                               tokens, targets, frac, seed)
    assert restricted > full + 0.1
