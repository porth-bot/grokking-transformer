"""Grokking symmetrizes the "=" token's attention over the two operands.

Tests the claim experiments/attention_pattern.py visualizes, straight from the
committed checkpoints (no retraining):

- the "=" attention weights are a valid causal distribution (rows sum to 1,
  no weight leaks to future positions);
- the "=" token reads the operands, not itself (operand weight dominates);
- the per-head a/b asymmetry collapses toward zero after grokking, reflecting
  the commutativity a + b = b + a, while the memorizing model stays lopsided.
"""

import numpy as np
import torch

from grokking.checkpoints import load_model
from grokking.data import modular_addition_dataset

MAIN = "p97_frac0.30_wd1_seed0"


def _eq_attention(model, tokens):
    """(n_heads, seq_len) mean "=" -> [a, b, =] attention over the dataset."""
    x = model.tok_emb(tokens) + model.pos_emb[: tokens.shape[1]]
    block = model.blocks[0]
    att = block.attn.attn_weights(block.ln1(x))       # (B, H, T, T)
    return att, att[:, :, -1, :].mean(0).cpu().numpy()


def test_eq_attention_is_a_valid_causal_distribution():
    model, summary = load_model(MAIN, which="final")
    tokens, _ = modular_addition_dataset(summary["config"]["p"])
    att, _ = _eq_attention(model, tokens)             # (B, H, 3, 3)

    # every query row is a probability distribution over allowed positions
    assert torch.allclose(att.sum(-1), torch.ones_like(att.sum(-1)), atol=1e-5)
    # causal mask: position 0 sees only itself; position 1 cannot see "="
    assert att[:, :, 0, 1:].abs().max().item() == 0.0
    assert att[:, :, 1, 2:].abs().max().item() == 0.0


def test_grokking_symmetrizes_operand_attention():
    mem, summary = load_model(MAIN, which="memorize")
    fin, _ = load_model(MAIN, which="final")
    tokens, _ = modular_addition_dataset(summary["config"]["p"])
    _, eq_mem = _eq_attention(mem, tokens)
    _, eq_fin = _eq_attention(fin, tokens)

    # the "=" token reads the operands (a, b), not itself, in both regimes
    assert eq_mem[:, :2].sum(1).mean() > 0.75
    assert eq_fin[:, :2].sum(1).mean() > 0.75

    # commutativity shows up as symmetric a/b attention only after grokking
    asym_mem = np.abs(eq_mem[:, 0] - eq_mem[:, 1]).mean()
    asym_fin = np.abs(eq_fin[:, 0] - eq_fin[:, 1]).mean()
    assert asym_mem > 0.10          # memorizing heads are lopsided
    assert asym_fin < 0.02          # grokked heads split a and b evenly
    assert asym_fin < asym_mem
