"""How the "=" query spreads its attention, as numbers that can be tracked.

``experiments/attention_pattern.py`` reads the "=" row out of two committed
checkpoints and shows that grokking *symmetrizes* it: the memorizing model puts
lopsided weight on the two operands, the grokked model splits them almost
evenly. That is a two-point story. These are the same read-outs written as
functions of a model, so they can be logged along a whole training trajectory
(``experiments/attention_entropy.py``) the way ``progress.py`` does for the
Fourier measures.

Why the "=" row and not the whole matrix. The sequence is ``[a, b, =]`` under a
causal mask, so position 0 can only attend to itself (entropy exactly 0) and
position 1 to two slots. Averaging entropy over all query positions would
therefore be mostly a measurement of the mask. The "=" query is the one row
that carries the computation -- it is where the answer is written, and it is
the only position that can see both operands.

Four statistics, all averaged over all ``p^2`` inputs:

- **Entropy** (:func:`eq_attention_entropy`), in nats, of the full "=" row over
  ``{a, b, =}``. Reference levels: ``ln 3 = 1.0986`` is uniform over all three
  positions, ``ln 2 = 0.6931`` is half on ``a``, half on ``b``, nothing on
  ``=``.
- **Operand entropy** (:func:`eq_operand_entropy`), the same thing computed on
  the ``{a, b}`` weights *renormalized to sum to one*. This is the measure of
  commutativity alone, with the self-attention channel divided out; it is
  bounded by ``ln 2`` and hits it exactly when a head is symmetric.
- **Operand fraction** (:func:`eq_operand_fraction`), the weight on ``a`` and
  ``b`` together -- i.e. how much of the row is *not* self-attention.
- **Asymmetry** (:func:`eq_operand_asymmetry`), ``mean_h |A[=->a] - A[=->b]|``,
  the appendix's statistic. Zero exactly when every head is commutative.

The reason for both entropies. Attention entropy is usually read as sharpness,
and here that reading fails twice over. Commutativity makes an *even* operand
split correct, so the algorithmic optimum is ``ln 2``, not 0 -- sharpening would
be wrong. And the grokked model measures 1.02 nats, above ``ln 2`` and close to
uniform, which looks like a diffuse read and is not: it comes from ~16% of the
row going to the "=" position itself, whose value vector is *identical for every
input* (position 2 is always the "=" token), so that channel carries a constant
vector -- a learned bias, not information. Divide it out and the operand entropy
is ``ln 2`` to five decimals on every head. Reporting only the full entropy
would say "diffuse"; reporting only the renormalized one would hide a real 16%
of the row. Both, and the fraction that relates them, is the honest set.

All of these read ``CausalSelfAttention.attn_weights``, the same softmax the
forward pass computes, so they cannot drift from the model's actual behavior.
"""

from __future__ import annotations

import math
from typing import cast

import torch

from .model import Block, Transformer

# The levels every entropy number here should be read against.
UNIFORM_3 = math.log(3.0)      # 1.0986: flat over {a, b, =}
SYMMETRIC_2 = math.log(2.0)    # 0.6931: half on a, half on b, none on =


def eq_attention(model: Transformer, tokens: torch.Tensor) -> torch.Tensor:
    """Per-head "=" query attention, averaged over the batch: ``(n_heads, T)``.

    Row ``h`` is head ``h``'s mean attention from the final position to
    ``[a, b, =]``. One layer is assumed (every run in this repo is 1-layer), and
    that assumption is checked rather than silently indexed past.
    """
    if len(model.blocks) != 1:
        raise ValueError(
            f"the '=' read-out is defined for the 1-layer runs here; "
            f"this model has {len(model.blocks)} layers"
        )
    with torch.no_grad():
        x = model.tok_emb(tokens) + model.pos_emb[: tokens.shape[1]]
        block = cast(Block, model.blocks[0])
        att = block.attn.attn_weights(block.ln1(x))        # (B, H, T, T)
        return att[:, :, -1, :].mean(0)                    # (H, T)


def row_entropy(rows: torch.Tensor) -> torch.Tensor:
    """Shannon entropy in nats along the last axis, with 0 log 0 = 0.

    Attention rows routinely contain exact-zero-to-float entries (a saturated
    softmax), and ``p log p`` at ``p = 0`` is a NaN rather than the 0 the limit
    gives. Clamping inside the log and multiplying by the *unclamped* ``p`` is
    the standard fix, and leaves every other entry bit-exact.
    """
    return -(rows * torch.log(rows.clamp_min(1e-30))).sum(-1)


def eq_attention_entropy(model: Transformer, tokens: torch.Tensor) -> float:
    """Mean over heads of the full "=" row's entropy (nats).

    This is the entropy of the *batch-averaged* row, not the mean of per-example
    entropies. The two differ (entropy is concave, so the averaged row is at
    least as high) and the averaged row is the right object: the claim under
    test is about the circuit's fixed read of the operand slots, not about
    per-example sharpness. Exercise 3 of ``theory/notes.md`` makes the same
    distinction for the asymmetry statistic and it applies verbatim here.
    """
    return float(row_entropy(eq_attention(model, tokens)).mean())


def eq_operand_entropy(model: Transformer, tokens: torch.Tensor) -> float:
    """Entropy of the ``{a, b}`` weights renormalized to sum to one (max ln 2).

    Commutativity alone, with the constant self-attention channel divided out.
    """
    row = eq_attention(model, tokens)[:, :2]
    return float(row_entropy(row / row.sum(-1, keepdim=True).clamp_min(1e-30)).mean())


def eq_operand_fraction(model: Transformer, tokens: torch.Tensor) -> float:
    """Mean over heads of the weight the "=" row puts on ``a`` and ``b``."""
    return float(eq_attention(model, tokens)[:, :2].sum(-1).mean())


def eq_operand_asymmetry(model: Transformer, tokens: torch.Tensor) -> float:
    """``mean_h |A[= -> a] - A[= -> b]|``: 0 iff every head is commutative."""
    row = eq_attention(model, tokens)
    return float((row[:, 0] - row[:, 1]).abs().mean())


def measure_attention(model: Transformer, tokens: torch.Tensor) -> dict[str, float]:
    """All four statistics from a single attention read-out."""
    row = eq_attention(model, tokens)
    operands = row[:, :2]
    renormalized = operands / operands.sum(-1, keepdim=True).clamp_min(1e-30)
    return {
        "attn_entropy": float(row_entropy(row).mean()),
        "attn_operand_entropy": float(row_entropy(renormalized).mean()),
        "attn_operand_frac": float(operands.sum(-1).mean()),
        "attn_asymmetry": float((operands[:, 0] - operands[:, 1]).abs().mean()),
    }
