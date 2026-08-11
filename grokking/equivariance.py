"""Swap symmetries of the learned function, measured at the output.

The appendix's attention read-out found that grokking on ``(a + b) mod p``
*symmetrizes* how the "=" position reads the two operands, and Exercise 3 of
``theory/notes.md`` worked out what that statistic is really about: not
per-input symmetry, but **swap-equivariance** of the whole computation. Sec. 11
then leaned on it, proposing commutativity as the reason subtraction is the
hardest of the three operations -- and flagged, correctly, that the proposal was
untested, because the read-out had only ever been run on addition.

This module is the read-out, written so it can be pointed at any operation, and
generalized in the one way subtraction requires.

**The two symmetries are different symmetries.** Addition is commutative, so a
correct model is *invariant* under swapping the operands:

    L(a, b) = L(b, a).

Subtraction is not commutative, and demanding invariance of it would be
demanding that the model be wrong. What ``a - b = -(b - a)`` licenses instead is
*anti*-equivariance: swapping the operands negates the answer, so a correct
model's logit vector comes back permuted by ``c -> -c mod p``:

    L(a, b)[c] = L(b, a)[-c].

Both are one line apart in code and they measure opposite things, so both are
computed for both operations. The invariance defect is the appendix's published
statistic; the anti-equivariance defect is what a subtraction model can actually
satisfy. Reporting only the first on subtraction would find "no symmetry" and
conclude, wrongly, that grokking there is unstructured.

Normalization. Every defect is divided by the logits' own standard deviation, so
it is in units of the spread of the thing being compared and is comparable
across checkpoints whose logit scales differ by weight decay. A defect of 1.0
means the swapped logits differ from the originals by about as much as the
logits vary at all, i.e. no symmetry; the reference for "none" is measured
rather than assumed by :func:`shuffle_baseline`.

The attention-level version (:func:`attention_equivariance_defect`) is kept
alongside because it is the exact statistic Sec. 11's hypothesis was stated in
terms of, and because it localizes the symmetry to the "=" row rather than the
whole circuit.
"""

from __future__ import annotations

from typing import cast

import torch

from .model import Block, Transformer


def swap_index(p: int) -> torch.Tensor:
    """Row-major ``(a, b)`` -> the row holding ``(b, a)``.

    The dataset from ``modular_dataset`` is every *ordered* pair in row-major
    order, so this permutation is an involution of the dataset onto itself --
    which is what makes a dataset mean of an odd statistic exactly zero
    (Exercise 3) rather than approximately so.
    """
    idx = torch.arange(p * p)
    return (idx % p) * p + (idx // p)


def negate_index(p: int) -> torch.Tensor:
    """Output classes reordered by ``c -> (-c) mod p``.

    Indexing a logit vector's class axis with this maps the model's score for
    every answer ``c`` onto the slot for ``-c``, which is what subtraction's
    swap does to the correct answer.
    """
    return (-torch.arange(p)) % p


def final_logits(model: Transformer, tokens: torch.Tensor) -> torch.Tensor:
    """Logits at the "=" position: ``(p^2, p)``."""
    with torch.no_grad():
        return model(tokens)[:, -1, :]


def logit_defect(logits: torch.Tensor, p: int, negate: bool = False) -> float:
    """Mean ``|L - L_swapped|`` in units of the logits' own sd.

    With ``negate=False`` this is the invariance defect ``L(a,b) vs L(b,a)``
    (zero iff the model is commutative). With ``negate=True`` it is the
    anti-equivariance defect ``L(a,b) vs L(b,a)[-c]`` (zero iff swapping the
    operands negates the answer).
    """
    swapped = logits[swap_index(p)]
    if negate:
        swapped = swapped[:, negate_index(p)]
    return float((logits - swapped).abs().mean() / logits.std())


def eq_attention_per_example(model: Transformer,
                             tokens: torch.Tensor) -> torch.Tensor:
    """The "=" query's attention over ``{a, b, =}``, per input: ``(p^2, H, 3)``.

    ``attention.eq_attention`` averages this over the batch, which is the right
    object for the entropy read-outs but destroys exactly what is needed here:
    the swap statistic pairs each input with a *different* input, so the two
    rows have to still exist separately.
    """
    if len(model.blocks) != 1:
        raise ValueError(
            f"the '=' read-out is defined for the 1-layer runs here; "
            f"this model has {len(model.blocks)} layers"
        )
    block = cast(Block, model.blocks[0])
    with torch.no_grad():
        x = model.tok_emb(tokens) + model.pos_emb[: tokens.shape[1]]
        return block.attn.attn_weights(block.ln1(x))[:, :, -1, :]


def attention_equivariance_defect(model: Transformer, tokens: torch.Tensor,
                                  p: int) -> float:
    """``E |A[(a,b) -> a] - A[(b,a) -> b]|``: the appendix's statistic.

    Zero when the "=" row's weight follows the *token* rather than the slot,
    which is what a commutative circuit needs and a non-commutative one cannot
    have. Note this is not the same as the per-example asymmetry
    ``E|A[(a,b) -> a] - A[(a,b) -> b]|``, which stays large even for the
    grokked addition model (Exercise 3).
    """
    a = eq_attention_per_example(model, tokens)
    return float((a[:, :, 0] - a[swap_index(p)][:, :, 1]).abs().mean())


def shuffle_baseline(logits: torch.Tensor, seed: int = 0) -> float:
    """The defect of a permutation that respects no symmetry at all.

    Pairing each input with a *random* other input instead of its swap gives
    the scale a defect takes when there is nothing to find. Without it, "1.0"
    is a number with no referent: it is only meaningful to say a defect is at
    the no-symmetry level if that level has been measured.
    """
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(logits.shape[0], generator=g)
    return float((logits - logits[perm]).abs().mean() / logits.std())


def measure_equivariance(model: Transformer, tokens: torch.Tensor,
                         p: int) -> dict[str, float]:
    """Every read-out above, from one forward pass."""
    logits = final_logits(model, tokens)
    return {
        "logit_swap_defect": logit_defect(logits, p, negate=False),
        "logit_anti_defect": logit_defect(logits, p, negate=True),
        "attn_equivariance_defect": attention_equivariance_defect(model, tokens, p),
        "shuffle_baseline": shuffle_baseline(logits),
    }
