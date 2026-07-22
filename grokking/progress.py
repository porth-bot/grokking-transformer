"""Nanda et al. (2023) progress measures, as reusable package functions.

``experiments/fourier.py`` and ``experiments/logit_attribution.py`` read the
*Fourier structure* of two committed checkpoints (memorization and final). The
progress measures here compute the same quantities so they can be logged along
the **whole training trajectory** (``experiments/progress_measures.py``, Day 19),
turning the two-point before/after story into a continuous one: the generalizing
circuit forms *gradually*, well before the test-accuracy jump.

Three measures, all Nanda et al.'s (arXiv:2301.05217), each a function of the
model at one training step:

- **Embedding sparsity** (:func:`embedding_top_k_fraction`). Fraction of the
  digit-embedding Fourier energy carried by its top ``k`` frequencies. Rises
  from diffuse (memorization) toward concentrated (grokked) as the circuit forms.

- **Restricted loss** (:func:`restricted_loss`). Cross-entropy of the logits with
  everything *except* a fixed set of key ``a+b`` frequencies projected out --
  i.e. the loss the model *would* have if it used only the generalizing circuit.
  It falls smoothly toward the true loss as that circuit takes over.

- **Excluded loss** (:func:`excluded_loss`). Cross-entropy with exactly those key
  frequencies *removed*. Low while the model does not yet depend on them, then
  rising as the generalizing solution becomes load-bearing.

Frequency bases (shared with the two checkpoint scripts, kept here as the
package-level source):

- Embeddings: 1D rFFT of the ``(p, d_model)`` token-embedding matrix along the
  token axis (the DC term is dropped from sparsity statistics).
- Logits: 2D FFT of the logit tensor ``L[a, b, c]`` over the two input axes. A
  function of ``a+b`` lives entirely on the diagonal ``k_a = k_b``; the "key
  frequencies" are the top diagonal frequencies of a reference (final) model.

All losses are computed over *all* ``p^2`` pairs -- these are mechanism measures,
deliberately decoupled from the particular train/test split.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .data import modular_addition_dataset
from .model import Transformer

# --------------------------------------------------------------------------- #
# Embedding sparsity
# --------------------------------------------------------------------------- #


def embedding_spectrum(model: Transformer, p: int) -> torch.Tensor:
    """Per-frequency L2 norm of the digit embeddings, ``(p//2 + 1,)``.

    The digit-token rows ``E`` (the "=" row excluded) are rFFT'd along the token
    axis; we report ``||F_k||_2`` over the model dimension for ``k = 0 .. (p-1)/2``.
    """
    E = model.tok_emb.weight.detach()[:p]      # (p, d_model)
    Fk = torch.fft.rfft(E, dim=0)              # (p//2 + 1, d_model), complex
    return Fk.abs().pow(2).sum(dim=1).sqrt()


def embedding_top_k_fraction(model: Transformer, p: int, k: int = 5) -> float:
    """Share of (non-DC) embedding spectral energy in the top ``k`` frequencies."""
    energy = embedding_spectrum(model, p)[1:].pow(2)   # drop DC (k=0)
    top = energy.sort(descending=True).values[:k].sum()
    return float(top / energy.sum())


# --------------------------------------------------------------------------- #
# Logit Fourier structure
# --------------------------------------------------------------------------- #


def logit_tensor(model: Transformer, tokens: torch.Tensor, p: int) -> torch.Tensor:
    """Logits at the "=" position for every ordered pair, shaped ``[a, b, c]``.

    The dataset is all ``p^2`` pairs in row-major (a outer, b inner) order, so a
    reshape to ``(p, p, p)`` indexes cleanly as ``L[a, b, c]``.
    """
    with torch.no_grad():
        logits = model(tokens)[:, -1, :]       # (p^2, p)
    return logits.reshape(p, p, p)


def diagonal_key_frequencies(L: torch.Tensor, p: int, k: int = 5) -> list[int]:
    """The top ``k`` ``a+b`` frequencies by diagonal logit energy.

    2D-FFT the (input-mean-removed) logits over ``(a, b)``; energy at ``(j, j)``
    plus its conjugate partner ``(p-j, p-j)``, summed over the answer axis, is the
    contribution of frequency ``j`` of ``a+b``. Returns the ``k`` largest ``j>0``.
    """
    Lc = L - L.mean(dim=(0, 1), keepdim=True)
    Fk = torch.fft.fft2(Lc, dim=(0, 1))
    E = Fk.abs().pow(2)
    K = (p - 1) // 2
    diag = torch.zeros(K + 1)
    for j in range(1, K + 1):
        diag[j] = E[j, j, :].sum() + E[p - j, p - j, :].sum()
    order = [int(j) for j in torch.argsort(diag, descending=True) if j > 0]
    return order[:k]


def _diagonal_mask(p: int, ks: list[int]) -> torch.Tensor:
    """Boolean ``(p, p)`` mask true at the ``(k, k)`` / ``(p-k, p-k)`` modes."""
    mask = torch.zeros(p, p, dtype=torch.bool)
    for k in ks:
        mask[k, k] = True
        mask[p - k, p - k] = True
    return mask


def restrict_to_freqs(L: torch.Tensor, p: int, keep_ks: list[int]) -> torch.Tensor:
    """Rebuild the logits keeping only the ``a+b`` structure at ``keep_ks``.

    Keeps the input-mean (constant-in-``(a,b)``) part plus the diagonal modes of
    every ``k`` in ``keep_ks``; all other 2D-DFT coefficients are zeroed before
    the inverse transform.
    """
    mean = L.mean(dim=(0, 1), keepdim=True)
    Fk = torch.fft.fft2(L - mean, dim=(0, 1))
    mask = _diagonal_mask(p, keep_ks)
    return torch.fft.ifft2(Fk * mask[:, :, None], dim=(0, 1)).real + mean


def exclude_freqs(L: torch.Tensor, p: int, drop_ks: list[int]) -> torch.Tensor:
    """Rebuild the logits with the ``a+b`` structure at ``drop_ks`` removed.

    Keeps the mean and *every* non-DC mode except the diagonal modes of the
    dropped frequencies -- the complement of :func:`restrict_to_freqs`. (For any
    key set, ``restrict + (exclude - mean)`` reconstructs ``L`` exactly.)
    """
    mean = L.mean(dim=(0, 1), keepdim=True)
    Fk = torch.fft.fft2(L - mean, dim=(0, 1))
    keep = ~_diagonal_mask(p, drop_ks)
    return torch.fft.ifft2(Fk * keep[:, :, None], dim=(0, 1)).real + mean


# --------------------------------------------------------------------------- #
# Losses over all pairs
# --------------------------------------------------------------------------- #


def _ce_over_pairs(L: torch.Tensor, targets: torch.Tensor, p: int) -> float:
    """Cross-entropy of a logit tensor ``L[a,b,c]`` against ``targets`` (all pairs)."""
    return float(F.cross_entropy(L.reshape(p * p, p), targets))


def restricted_loss(
    model: Transformer, tokens: torch.Tensor, targets: torch.Tensor,
    p: int, key_ks: list[int],
) -> float:
    """Loss of the logits projected onto only the key ``a+b`` frequencies."""
    L = logit_tensor(model, tokens, p)
    return _ce_over_pairs(restrict_to_freqs(L, p, key_ks), targets, p)


def excluded_loss(
    model: Transformer, tokens: torch.Tensor, targets: torch.Tensor,
    p: int, key_ks: list[int],
) -> float:
    """Loss of the logits with the key ``a+b`` frequencies ablated."""
    L = logit_tensor(model, tokens, p)
    return _ce_over_pairs(exclude_freqs(L, p, key_ks), targets, p)


def measure_all(
    model: Transformer, tokens: torch.Tensor, targets: torch.Tensor,
    p: int, key_ks: list[int], k_emb: int = 5,
) -> dict[str, float]:
    """All three progress measures for one model (one forward pass for the logits).

    ``key_ks`` is the fixed reference frequency set (the final model's top
    frequencies); ``k_emb`` is how many embedding frequencies define sparsity.
    """
    L = logit_tensor(model, tokens, p)
    return {
        "emb_top_frac": embedding_top_k_fraction(model, p, k_emb),
        "restricted_loss": _ce_over_pairs(restrict_to_freqs(L, p, key_ks), targets, p),
        "excluded_loss": _ce_over_pairs(exclude_freqs(L, p, key_ks), targets, p),
    }


def full_pairs(p: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Convenience: all ``p^2`` (tokens, targets) for the mechanism-measure losses."""
    return modular_addition_dataset(p)
