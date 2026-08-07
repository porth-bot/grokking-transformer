"""The attention entropy read-outs, and the decomposition they exist to make.

`test_attention_pattern.py` covers the two-checkpoint symmetry claim. These
cover the entropy statistics `grokking/attention.py` adds on top of it, and in
particular the one thing that view got wrong: the grokked model's "=" row is
NOT concentrated on the operands. About a sixth of it returns to the "="
position, whose value vector is constant across every input, so the full-row
entropy climbs toward uniform while the operand-only entropy sits at ln 2.

The load-bearing test is `test_self_attention_channel_carries_a_constant_vector`
-- without it, "the extra entropy is a learned bias, not a diffuse read" is an
interpretation rather than a measurement.
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch

from grokking.attention import (
    SYMMETRIC_2,
    UNIFORM_3,
    eq_attention,
    eq_attention_entropy,
    eq_operand_asymmetry,
    eq_operand_entropy,
    eq_operand_fraction,
    measure_attention,
    row_entropy,
)
from grokking.checkpoints import load_model
from grokking.data import modular_addition_dataset

MAIN = "p97_frac0.30_wd1_seed0"


def _loaded(which):
    model, summary = load_model(MAIN, which=which)
    tokens, _ = modular_addition_dataset(summary["config"]["p"])
    return model, tokens


# -- the entropy primitive ---------------------------------------------------


def test_row_entropy_matches_the_closed_forms_it_will_be_compared_against():
    """Uniform over k gives ln k, a point mass gives 0, and an exact zero in
    the row must not produce a NaN -- the case a saturated softmax reaches and
    the reason for the clamp."""
    rows = torch.tensor([
        [1 / 3, 1 / 3, 1 / 3],       # uniform over 3
        [0.5, 0.5, 0.0],             # uniform over 2, with a hard zero
        [1.0, 0.0, 0.0],             # point mass
        [0.7, 0.2, 0.1],             # generic
    ])
    expected = [UNIFORM_3, SYMMETRIC_2, 0.0,
                -sum(p * math.log(p) for p in (0.7, 0.2, 0.1))]
    np.testing.assert_allclose(row_entropy(rows).numpy(), expected, atol=1e-12)
    assert torch.isfinite(row_entropy(rows)).all()


def test_measure_attention_agrees_with_the_individual_functions():
    """One read-out or four, the numbers have to be the same -- measure_attention
    exists only to avoid recomputing the softmax four times."""
    model, tokens = _loaded("final")
    bundle = measure_attention(model, tokens)
    assert bundle["attn_entropy"] == eq_attention_entropy(model, tokens)
    assert bundle["attn_operand_entropy"] == eq_operand_entropy(model, tokens)
    assert bundle["attn_operand_frac"] == eq_operand_fraction(model, tokens)
    assert bundle["attn_asymmetry"] == eq_operand_asymmetry(model, tokens)


def test_eq_attention_row_is_a_distribution_and_rejects_multilayer_models():
    model, tokens = _loaded("final")
    row = eq_attention(model, tokens)
    assert row.shape == (4, 3)
    np.testing.assert_allclose(row.sum(-1).numpy(), 1.0, atol=1e-6)
    assert (row >= 0).all()

    model.blocks = torch.nn.ModuleList([model.blocks[0], model.blocks[0]])
    try:
        eq_attention(model, tokens)
    except ValueError as exc:
        assert "1-layer" in str(exc)
    else:                                       # pragma: no cover
        raise AssertionError("a 2-layer model should have been refused")


# -- what grokking does to the read-out --------------------------------------


def test_full_row_entropy_rises_through_grokking():
    """The direction that contradicts the usual "attention sharpens" reading.

    It rises for two independent reasons, both measured below: the operand
    split becomes even (which raises entropy, because commutativity makes an
    even split correct rather than sloppy), and a self-attention channel opens.
    """
    mem_model, tokens = _loaded("memorize")
    fin_model, _ = _loaded("final")
    h_mem = eq_attention_entropy(mem_model, tokens)
    h_fin = eq_attention_entropy(fin_model, tokens)
    assert h_fin > h_mem
    assert h_mem < SYMMETRIC_2 < h_fin < UNIFORM_3      # 0.677 -> 1.024


def test_operand_entropy_reaches_its_ceiling_exactly_after_grokking():
    """With the self-attention channel divided out, every grokked head sits at
    ln 2 to five decimals -- the commutativity a + b = b + a, read directly off
    the softmax. The memorizing model is measurably below it."""
    mem_model, tokens = _loaded("memorize")
    fin_model, _ = _loaded("final")
    assert abs(eq_operand_entropy(fin_model, tokens) - SYMMETRIC_2) < 1e-5
    assert eq_operand_entropy(mem_model, tokens) < SYMMETRIC_2 - 0.01

    # per head, not just on average -- an average can hide two opposite errors
    row = eq_attention(fin_model, tokens)[:, :2]
    per_head = row_entropy(row / row.sum(-1, keepdim=True))
    np.testing.assert_allclose(per_head.numpy(), SYMMETRIC_2, atol=1e-4)


def test_the_grokked_row_is_not_concentrated_on_the_operands():
    """The correction to the appendix's original wording.

    "~all of its attention on the operands in both checkpoints" is right at
    memorization (0.997) and wrong at the end (0.837). Both numbers are pinned
    here so the README's prose cannot drift from them again.
    """
    mem_model, tokens = _loaded("memorize")
    fin_model, _ = _loaded("final")
    frac_mem = eq_operand_fraction(mem_model, tokens)
    frac_fin = eq_operand_fraction(fin_model, tokens)
    assert frac_mem > 0.99
    assert 0.80 < frac_fin < 0.87
    assert frac_mem - frac_fin > 0.1          # the self-attention channel opens


def test_self_attention_channel_carries_a_constant_vector():
    """Why the extra entropy is a bias and not lost information.

    Position 2 is the "=" token in every single example, so the value vector
    the "=" query pulls from itself does not depend on the input at all -- its
    across-batch standard deviation is exactly zero. Whatever weight the row
    puts there adds the same vector to every example: a learned bias term, not
    a diffuse read of the operands. Measured, because the claim is the whole
    reason the two entropies are reported separately.
    """
    model, tokens = _loaded("final")
    assert torch.unique(tokens[:, -1]).numel() == 1        # always "="

    with torch.no_grad():
        block = model.blocks[0]
        x = model.tok_emb(tokens) + model.pos_emb[: tokens.shape[1]]
        _, v = block.attn._attention(block.ln1(x))          # (B, H, T, d_head)

    assert float(v[:, :, -1, :].std(0).max()) == 0.0        # constant, exactly
    assert float(v[:, :, 0, :].std(0).max()) > 0.0          # operands are not


def test_asymmetry_and_entropy_are_not_the_same_measurement():
    """They can disagree, which is why both are logged: the grokked model has
    essentially zero asymmetry AND above-ln-2 entropy at the same time. A run
    that reported only one of them would call the same model either perfectly
    symmetric or suspiciously diffuse."""
    model, tokens = _loaded("final")
    assert eq_operand_asymmetry(model, tokens) < 1e-3
    assert eq_attention_entropy(model, tokens) > SYMMETRIC_2 + 0.25


# -- the committed trajectory ------------------------------------------------
#
# The figure and the README both read this CSV, so the shape of the trajectory
# is worth pinning: the four regimes are the result, and a silent regression in
# the read-out would show up here as a flat or monotone curve.


def _trajectory():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    import attention_entropy

    return attention_entropy.summarize()


def test_committed_trajectory_shows_the_four_regimes():
    """Symmetric at init, broken while memorizing, restored at the jump, and a
    self-attention channel that opens only afterwards."""
    s = _trajectory()

    # 1. init is already at the algorithmic symmetry (a near-uniform softmax is)
    assert abs(s["operand_entropy_at_init"] - SYMMETRIC_2) < 1e-4
    assert abs(s["entropy_at_init"] - UNIFORM_3) < 1e-3
    assert abs(s["operand_frac_at_init"] - 2 / 3) < 0.02
    assert s["asymmetry_at_init"] < 0.01

    # 2. memorization breaks it, and concentrates the row onto the operands
    assert s["operand_entropy_min"] < SYMMETRIC_2 - 0.03
    assert s["asymmetry_peak"] > 0.15
    assert s["operand_frac_at_memorize"] > 0.99

    # 3. grokking restores it
    assert abs(s["operand_entropy_final"] - SYMMETRIC_2) < 1e-4
    assert s["asymmetry_final"] < 1e-3

    # 4. and the constant channel opens only after the jump, then fluctuates
    assert s["operand_frac_postgrok_min"] < 0.90 < s["operand_frac_postgrok_max"]


def test_the_symmetry_is_restored_after_the_jump_not_before_it():
    """The claim the experiment was built to test, and it came out the boring
    way: unlike §10's restricted loss, this read-out does not anticipate
    grokking. Stated as a test so a later run that reverses it has to say so.
    """
    s = _trajectory()
    assert s["step_symmetry_restored"] > s["step_test_acc_half"]
    assert s["step_symmetry_restored"] > s["grok_step"]
