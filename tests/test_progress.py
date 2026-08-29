"""The progress-measure machinery is a faithful Fourier decomposition.

These check the reusable helpers in ``grokking/progress.py`` -- the functions the
Day-19 trajectory experiment logs at every eval -- against exact properties, so
the trajectory it produces means what the README claims:

1. restrict + exclude is an exact partition of the logits (keeping all diagonal
   frequencies plus the complement reconstructs L exactly);
2. restricting to only the true frequency of a synthetic ``f(a+b)`` logit tensor
   is loss-free, while excluding that frequency destroys it -- the two measures
   are on opposite sides of the key structure;
3. embedding sparsity is 1.0 for a pure few-frequency embedding and small for
   white noise -- the sparsity statistic actually measures concentration;
4. the final committed checkpoint has a higher embedding top-5 fraction and a
   lower restricted loss than the memorization checkpoint (the trajectory's
   endpoints move the way the story needs).
"""

import sys
from pathlib import Path

import numpy as np
import torch

from grokking.data import modular_addition_dataset
from grokking.model import ModelConfig, Transformer
from grokking.progress import (
    diagonal_key_frequencies,
    embedding_spectrum,
    embedding_top_k_fraction,
    exclude_freqs,
    excluded_loss,
    logit_tensor,
    measure_all,
    restrict_to_freqs,
    restricted_loss,
)
from grokking.checkpoints import load_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

P = 17  # small prime -> fast exact FFT checks


def _fake_logits_from_ab_freq(p, k):
    """Logit tensor L[a,b,c] = cos(2 pi k (a+b-c)/p): a pure single-frequency a+b."""
    idx = torch.arange(p)
    a = idx.view(p, 1, 1)
    b = idx.view(1, p, 1)
    c = idx.view(1, 1, p)
    return torch.cos(2 * np.pi * k * (a + b - c) / p)


def test_restrict_plus_exclude_reconstructs_logits():
    torch.manual_seed(0)
    L = torch.randn(P, P, P)
    keep = [2, 5]
    mean = L.mean(dim=(0, 1), keepdim=True)
    recon = restrict_to_freqs(L, P, keep) + (exclude_freqs(L, P, keep) - mean)
    assert torch.allclose(recon, L, atol=1e-5)


def test_keeping_all_diagonal_frequencies_is_identity_for_a_sum_function():
    """For any function of ``a+b``, all 2D-DFT energy is on the diagonal, so
    keeping every diagonal frequency reconstructs it exactly (an arbitrary L
    would not -- off-diagonal per-pair structure is what restriction drops)."""
    torch.manual_seed(1)
    h = torch.randn(P, P)                     # h[(a+b) mod p, c]
    idx = (torch.arange(P).view(P, 1) + torch.arange(P).view(1, P)) % P
    L = h[idx]                                # L[a, b, c] = h[(a+b)%p, c]
    all_ks = list(range(1, (P - 1) // 2 + 1))
    assert torch.allclose(restrict_to_freqs(L, P, all_ks), L, atol=1e-5)


def test_restrict_and_exclude_are_opposite_on_a_pure_frequency():
    k = 3
    L = _fake_logits_from_ab_freq(P, k)
    key = diagonal_key_frequencies(L, P, k=1)
    assert key == [k]
    # Restricting to the true frequency changes almost nothing; excluding it
    # removes essentially all the non-mean structure.
    restricted = restrict_to_freqs(L, P, [k])
    excluded = exclude_freqs(L, P, [k])
    mean = L.mean(dim=(0, 1), keepdim=True)
    assert torch.allclose(restricted, L, atol=1e-5)
    assert torch.allclose(excluded, mean.expand_as(L), atol=1e-5)


def test_embedding_top_k_fraction_pure_vs_noise():
    cfg = ModelConfig(p=P, vocab_size=P + 1)
    model = Transformer(cfg)
    # Pure: embed each token on a few cosine/sine frequencies -> all energy there.
    n = torch.arange(P).float()
    cols = []
    for k in (1, 3):
        cols += [torch.cos(2 * np.pi * k * n / P), torch.sin(2 * np.pi * k * n / P)]
    pure = torch.stack(cols, dim=1)                       # (P, 4)
    pad = torch.zeros(P, cfg.d_model - pure.shape[1])
    with torch.no_grad():
        model.tok_emb.weight[:P] = torch.cat([pure, pad], dim=1)
    assert embedding_top_k_fraction(model, P, k=5) > 0.999

    torch.manual_seed(2)
    with torch.no_grad():
        model.tok_emb.weight[:P] = torch.randn(P, cfg.d_model)
    # White noise spreads energy over all (P-1)//2 = 8 frequencies; top-1 should
    # be far from concentrated.
    assert embedding_top_k_fraction(model, P, k=1) < 0.6


def test_spectrum_length_and_nonnegative():
    cfg = ModelConfig(p=P, vocab_size=P + 1)
    spec = embedding_spectrum(Transformer(cfg), P)
    assert spec.shape == ((P - 1) // 2 + 1,)
    assert (spec >= 0).all()


def test_measure_all_matches_individual_calls():
    torch.manual_seed(3)
    cfg = ModelConfig(p=P, vocab_size=P + 1)
    model = Transformer(cfg)
    tokens, targets = modular_addition_dataset(P)
    key = diagonal_key_frequencies(logit_tensor(model, tokens, P), P, k=5)
    m = measure_all(model, tokens, targets, P, key, k_emb=5)
    assert m["restricted_loss"] == restricted_loss(model, tokens, targets, P, key)
    assert m["excluded_loss"] == excluded_loss(model, tokens, targets, P, key)
    assert m["emb_top_frac"] == embedding_top_k_fraction(model, P, 5)


def test_committed_checkpoints_move_the_right_way():
    """Endpoints of the real trajectory: final is sparser + lower restricted loss."""
    mem, summary = load_model("p97_frac0.30_wd1_seed0", which="memorize")
    fin, _ = load_model("p97_frac0.30_wd1_seed0", which="final")
    p = summary["config"]["p"]
    tokens, targets = modular_addition_dataset(p)
    key = diagonal_key_frequencies(logit_tensor(fin, tokens, p), p, k=5)

    assert embedding_top_k_fraction(fin, p, 5) > embedding_top_k_fraction(mem, p, 5)
    assert restricted_loss(fin, tokens, targets, p, key) < restricted_loss(
        mem, tokens, targets, p, key
    )


# -- what the trajectory actually looks like ---------------------------------
#
# Sec. 10 described the delay as a "flat test-accuracy plateau" with accuracy
# "still stuck near 15%" at step 1,400, and Sec. 4 contrasted p = 113 against
# p = 97 "sitting near chance". The committed logs refute all three, so the
# corrected statements are asserted here rather than left to prose: they are
# exactly the kind a later edit would smooth back toward the tidier story.

def _trajectory(name, runs="runs"):
    import csv

    root = Path(__file__).resolve().parents[1]
    with open(root / runs / f"{name}.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    return {int(r["step"]): {k: float(v) for k, v in r.items()} for r in rows}


def test_the_delay_is_not_a_near_chance_plateau_in_any_seed():
    """Sec. 4: after memorization the model is 16-29x chance and climbing."""
    import json

    root = Path(__file__).resolve().parents[1]
    chance = 1.0 / 97
    at_memorize, before_jump = [], []
    for seed in range(5):
        name = f"p97_frac0.30_wd1_seed{seed}"
        traj = _trajectory(name)
        summary = json.loads((root / "runs" / f"{name}.json").read_text())
        acc = {s: r["test_acc"] for s, r in traj.items()}

        at_memorize.append(acc[summary["memorize_step"]])
        # The last eval before test accuracy first clears 0.6, i.e. the last
        # look at the model while the delay is still running.
        jump = min(s for s, a in acc.items() if a > 0.6)
        before_jump.append(acc[max(s for s in acc if s < jump)])

    assert min(at_memorize) > 15 * chance, (
        f"memorization accuracies {at_memorize} are supposed to be far above "
        f"the {chance:.4f} chance level"
    )
    # The range Sec. 4 quotes, to the precision it quotes it at.
    assert round(min(at_memorize), 3) == 0.163
    assert round(max(at_memorize), 3) == 0.302
    assert round(sum(at_memorize) / len(at_memorize), 3) == 0.247
    # And every seed is materially higher again just before the jump.
    assert all(b > m for b, m in zip(before_jump, at_memorize))
    assert min(before_jump) > 0.37


def test_p113s_memorization_accuracy_sits_inside_p97s_seed_range():
    """Sec. 4's "softer at p = 113" was one seed against one seed."""
    import json

    root = Path(__file__).resolve().parents[1]
    p97 = []
    for seed in range(5):
        name = f"p97_frac0.30_wd1_seed{seed}"
        summary = json.loads((root / "runs" / f"{name}.json").read_text())
        p97.append(_trajectory(name)[summary["memorize_step"]]["test_acc"])

    name = "p113_frac0.30_wd1_seed0"
    summary = json.loads((root / "runs" / f"{name}.json").read_text())
    p113 = _trajectory(name)[summary["memorize_step"]]["test_acc"]

    assert min(p97) < p113 < max(p97), (
        f"p=113 reads {p113:.3f}, p=97's five seeds span "
        f"[{min(p97):.3f}, {max(p97):.3f}] -- if that stops being an overlap "
        "the Sec. 4 comparison can be made again"
    )
    # And the seed the old comparison used was the weakest of the five.
    assert p97[0] == min(p97)


def test_the_restricted_loss_leads_early_and_is_overtaken_before_the_jump():
    """Sec. 10: restricted < full through step 1,000, full ahead from 1,100."""
    traj = _trajectory("progress_p97_frac0.30_wd1_seed0")
    lead = {s: r["restricted_loss"] < r["test_loss"] for s, r in traj.items()}

    assert all(lead[s] for s in range(100, 1001, 100))
    assert not any(lead[s] for s in range(1100, 1601, 100))
    # The lead it opens with, and the margin it is overtaken by.
    assert traj[100]["test_loss"] - traj[100]["restricted_loss"] > 0.9
    assert 0 < traj[1100]["restricted_loss"] - traj[1100]["test_loss"] < 0.05
