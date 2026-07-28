"""The claims made in `theory/notes.md` Sec. 6 (exercises), checked.

The other repos in this series put their exercises' answers under test rather
than leaving them as prose (mcmc's Sec. 7, gp's Sec. 9). Same rule here: an
exercise whose solution asserts a number should have somewhere that the number
is computed and compared.
"""

from __future__ import annotations

import math

import pytest
import torch

from grokking.checkpoints import load_model
from grokking.data import modular_addition_dataset
from grokking.model import ModelConfig, Transformer
from grokking.progress import embedding_top_k_fraction

P = 97
# The trained model's top-5 a+b frequencies (README Sec. 10).
KEY_FREQS = [5, 14, 20, 36, 38]
RUN = "p97_frac0.30_wd1_seed0"


# --------------------------------------------------------------------------- #
# Exercise 1: a sparse frequency set still argmaxes correctly
# --------------------------------------------------------------------------- #
def _cosine_sum(freqs: list[int], p: int = P) -> torch.Tensor:
    n = torch.arange(p, dtype=torch.float64)
    return sum(torch.cos(2 * math.pi * k * n / p) for k in freqs)


def test_all_frequencies_give_the_exact_delta():
    """Sec. 3's identity, as arithmetic: the full sum is p at 0 and 0 elsewhere."""
    s = _cosine_sum(list(range(P)))
    assert float(s[0]) == pytest.approx(P)
    assert float(s[1:].abs().max()) < 1e-10


def test_a_sparse_set_keeps_the_peak_and_the_off_target_spread_is_the_predicted_size():
    """Peak ``m``, off-target sd ``sqrt(m/2)``, so the margin grows like ``sqrt(2m)``.

    Each off-target term is a cosine at an effectively arbitrary phase, hence
    variance 1/2, and the ``m`` of them are close enough to independent that
    the sum's sd is ``sqrt(m/2)``. What makes the algorithm work is not that
    the off-target values are small but that the peak is ``m`` while they grow
    only like ``sqrt(m)``.
    """
    m = len(KEY_FREQS)
    s = _cosine_sum(KEY_FREQS)
    off = s[1:]
    assert float(s[0]) == pytest.approx(float(m))
    assert float(off.std()) == pytest.approx(math.sqrt(m / 2), rel=0.1)
    # And the argmax is right with room to spare: the largest off-target value
    # is well under the peak.
    assert float(s[0] / off.max()) > 1.5


def test_the_sparse_logits_pick_the_right_answer_for_every_pair():
    """Five frequencies out of 97 suffice, over all p^2 pairs."""
    a = torch.arange(P)
    A, B, C = torch.meshgrid(a, a, a, indexing="ij")
    n = (A + B - C) % P
    logits = sum(torch.cos(2 * math.pi * k * n / P) for k in KEY_FREQS)
    assert bool((logits.argmax(dim=2) == (A[:, :, 0] + B[:, :, 0]) % P).all())


# --------------------------------------------------------------------------- #
# Exercise 2: angle addition puts all the energy on the a+b diagonal
# --------------------------------------------------------------------------- #
def test_the_ideal_logit_tensor_lives_exactly_on_the_diagonal_modes():
    """``cos(omega(a+b-c))`` is a sum of rank-1 products of pure-frequency
    vectors in ``a`` and ``b``, so its 2D DFT over ``(a, b)`` is supported on
    ``(k, k)`` and ``(p-k, p-k)`` and nowhere else.

    This is the *reference* that Sec. 8's measurement is a measurement of:
    the ideal tensor scores 1.000 here, the grokked model 0.98, the memorizing
    one 0.12.
    """
    a = torch.arange(P)
    A, B, C = torch.meshgrid(a, a, a, indexing="ij")
    n = (A + B - C) % P
    L = sum(torch.cos(2 * math.pi * k * n / P) for k in KEY_FREQS).double()

    F = torch.fft.fft2(L - L.mean(dim=(0, 1), keepdim=True), dim=(0, 1))
    energy = F.abs().pow(2).sum(dim=2)
    on_diagonal = sum(float(energy[k, k] + energy[P - k, P - k]) for k in KEY_FREQS)
    assert on_diagonal / float(energy.sum()) == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Exercise 3: attention at initialization
# --------------------------------------------------------------------------- #
def _preattention_logits(model: Transformer, tokens: torch.Tensor) -> torch.Tensor:
    """Pre-softmax attention logits at the "=" row: ``(B, H, 3)``."""
    block = model.blocks[0]
    attn = block.attn
    with torch.no_grad():
        h = model.tok_emb(tokens) + model.pos_emb[: tokens.shape[1]]
        x = block.ln1(h)
        B, T, C = x.shape
        q, k, _ = attn.qkv(x).split(C, dim=2)
        q = q.view(B, T, attn.n_heads, attn.d_head).transpose(1, 2)
        k = k.view(B, T, attn.n_heads, attn.d_head).transpose(1, 2)
        z = (q @ k.transpose(-2, -1)) / math.sqrt(attn.d_head)
    return z[:, :, 2, :]


@pytest.mark.parametrize("s", [0.05, 0.2])
def test_the_delta_method_softmax_variances_are_right_for_independent_logits(s):
    """``Var(A_i) = s^2 (n-1)/n^3`` and ``Var(A_i - A_j) = 2 s^2/n^2``.

    From ``A_i ~= 1/n + (z_i - zbar)/n`` for small ``s``. Checked where the
    derivation's own assumption holds exactly -- i.i.d. logits -- so that any
    disagreement is the expansion, not the model. It is a *small-s* expansion:
    at s = 0.5 the same formulas are already ~11% off, which is why the check
    is parameterized only over small values.
    """
    n = 3
    torch.manual_seed(0)
    z = s * torch.randn(400_000, n, dtype=torch.float64)
    a = torch.softmax(z, dim=-1)
    d = a[:, 0] - a[:, 1]
    assert float(a.var()) == pytest.approx(s * s * (n - 1) / n**3, rel=0.05)
    assert float(d.var()) == pytest.approx(2 * s * s / n**2, rel=0.05)
    assert float(d.abs().mean()) == pytest.approx(
        math.sqrt(2 * s * s / n**2) * math.sqrt(2 / math.pi), rel=0.05
    )


def test_the_initial_attention_logit_scale_matches_the_hand_computation():
    """``sd(z) = sigma_W^2 d_model`` -- and the head split cancels out of it.

    LayerNorm hands the projection unit-variance rows, so ``Var(q_i) =
    sigma_W^2 d_model``; then ``Var(q.k) = d_head sigma_W^4 d_model^2`` and
    the ``1/sqrt(d_head)`` leaves ``sd = sigma_W^2 d_model``, with no
    ``d_head`` in it. So one head and four heads start at the same attention
    temperature -- worth knowing beside Sec. 9's head-count ablation, since it
    says the head count is not changing where training *starts*.
    """
    tokens, _ = modular_addition_dataset(P)
    predicted = 0.02**2 * 128  # sigma_W^2 * d_model
    for n_heads in (1, 2, 4):
        torch.manual_seed(0)
        model = Transformer(ModelConfig(p=P, n_heads=n_heads)).eval()
        z = _preattention_logits(model, tokens[:4000])
        assert float(z.std()) == pytest.approx(predicted, rel=0.25)


def _eq_attention(model: Transformer, tokens: torch.Tensor) -> torch.Tensor:
    """The "=" query's attention over ``[a, b, =]``: ``(B, H, 3)``.

    Same arithmetic as ``experiments/attention_pattern.py`` (which averages
    over the dataset before taking the difference -- the distinction this
    exercise is about).
    """
    x = model.tok_emb(tokens) + model.pos_emb[: tokens.shape[1]]
    block = model.blocks[0]
    with torch.no_grad():
        return block.attn.attn_weights(block.ln1(x))[:, :, -1, :]


def _swap_index(p: int = P) -> torch.Tensor:
    """Row-major ``(a, b)`` -> the row holding ``(b, a)``."""
    idx = torch.arange(p * p)
    return (idx % p) * p + (idx // p)


def test_the_initial_operand_asymmetry_is_the_baseline_grokking_is_measured_against():
    """At init the "=" token reads both operands equally to within ~0.01.

    The dataset-averaged asymmetry that Sec. 4's appendix reports is
    0.004-0.017 across seeds at initialization. That is the number that makes
    the published trajectory readable: the memorizing checkpoint's 0.19 is an
    order of magnitude *above* random init, and the grokked model's 0.0001 is
    two orders *below* it.
    """
    tokens, _ = modular_addition_dataset(P)
    values = []
    for seed in range(4):
        torch.manual_seed(seed)
        model = Transformer(ModelConfig(p=P)).eval()
        mean_attention = _eq_attention(model, tokens).mean(0)  # (H, 3)
        values.append(float((mean_attention[:, 0] - mean_attention[:, 1]).abs().mean()))
    init = sum(values) / len(values)
    assert 0.002 < init < 0.05, values

    measured = {}
    for which in ("memorize", "final"):
        model, _ = load_model(RUN, which)
        mean_attention = _eq_attention(model, tokens).mean(0)
        measured[which] = float((mean_attention[:, 0] - mean_attention[:, 1]).abs().mean())
    assert measured["memorize"] > 10 * init
    assert measured["final"] < init / 10


def test_the_published_asymmetry_statistic_measures_swap_equivariance_not_per_input_symmetry():
    """What the 0.19 -> 0.00 number is actually a number about.

    Write ``D(a,b) = A[(a,b) -> pos 0] - A[(a,b) -> pos 1]``. If the "="
    token's attention follows the *token* rather than the position -- i.e. the
    computation is equivariant under swapping the operands -- then
    ``A[(b,a) -> pos 0] = A[(a,b) -> pos 1]``, so ``D(b,a) = -D(a,b)``: ``D``
    is *odd* under the swap. The dataset is every ordered pair, hence closed
    under the swap, so its mean of ``D`` is exactly zero.

    The consequence, which is easy to state wrongly: the statistic does **not**
    say each individual input is read symmetrically, and the grokked model
    does not read them symmetrically -- its per-example ``|D|`` is 0.15,
    eight times the value at init. It says the model has learned commutativity
    as a symmetry of its whole computation.
    """
    tokens, _ = modular_addition_dataset(P)
    swap = _swap_index()
    assert bool((tokens[swap][:, 0] == tokens[:, 1]).all())  # the index is the swap

    defect, per_example, logit_defect = {}, {}, {}
    for which in ("memorize", "final"):
        model, _ = load_model(RUN, which)
        a = _eq_attention(model, tokens)
        # equivariance defect: weight on token a, computed both ways round
        defect[which] = float((a[:, :, 0] - a[swap][:, :, 1]).abs().mean())
        per_example[which] = float((a[:, :, 0] - a[:, :, 1]).abs().mean())
        with torch.no_grad():
            logits = model(tokens)[:, -1, :]
        logit_defect[which] = float((logits - logits[swap]).abs().mean() / logits.std())

    # Grokking buys equivariance, by three orders of magnitude.
    assert defect["memorize"] > 0.1
    assert defect["final"] < 0.01
    # ...while the grokked model is still strongly asymmetric per example.
    assert per_example["final"] > 0.1
    # And the symmetry reaches the output: the memorizing model computes a
    # materially non-commutative function, the grokked one does not.
    assert logit_defect["memorize"] > 0.4
    assert logit_defect["final"] < 0.02


# --------------------------------------------------------------------------- #
# Exercise 4: Parseval, and why the energy fraction is well defined
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("which", ["memorize", "final"])
def test_the_embedding_spectrum_accounts_for_exactly_the_centred_energy(which):
    """``sum_n ||E_n - Ebar||^2 = (2/p) sum_{k>=1} ||F_k||^2`` for real ``E``, odd ``p``.

    Parseval with the conjugate doubling that a real FFT's half-spectrum
    implies. This is what makes "fraction of embedding energy in the top 5
    frequencies" a statement about the embedding rather than about the
    transform: the spectrum's non-DC part is the centred embedding's energy,
    exactly, with nothing unaccounted for.
    """
    model, _ = load_model(RUN, which)
    E = model.tok_emb.weight.detach()[:P].double()
    centred = float(((E - E.mean(dim=0, keepdim=True)) ** 2).sum())
    modal = torch.fft.rfft(E, dim=0).abs().pow(2).sum(dim=1)
    assert centred == pytest.approx(float(2 * modal[1:].sum() / P), rel=1e-10)


def test_the_top_k_fraction_is_unchanged_by_the_normalization_it_omits():
    """``embedding_top_k_fraction`` drops the ``2/p`` -- and it cancels in a ratio.

    Worth pinning because it cuts both ways: the *fraction* is correct without
    the factor, while any absolute modal energy read off
    ``embedding_spectrum`` is half the true one. The measure is right; a
    number lifted out of it would not be.
    """
    model, _ = load_model(RUN, "final")
    E = model.tok_emb.weight.detach()[:P].double()
    modal = torch.fft.rfft(E, dim=0).abs().pow(2).sum(dim=1)[1:]
    scaled = 2 * modal / P
    top = 5
    unscaled_fraction = float(modal.sort(descending=True).values[:top].sum() / modal.sum())
    scaled_fraction = float(scaled.sort(descending=True).values[:top].sum() / scaled.sum())
    assert unscaled_fraction == pytest.approx(scaled_fraction, rel=1e-12)
    assert unscaled_fraction == pytest.approx(embedding_top_k_fraction(model, P, top), rel=1e-5)


def test_weight_decay_is_visible_in_the_embedding_energy_alone():
    """The norm story of Sec. 4, in one number the spectrum already computes."""
    energies = {}
    for which in ("memorize", "final"):
        model, _ = load_model(RUN, which)
        E = model.tok_emb.weight.detach()[:P].double()
        energies[which] = float(((E - E.mean(dim=0, keepdim=True)) ** 2).sum())
    assert energies["memorize"] > 4 * energies["final"]
