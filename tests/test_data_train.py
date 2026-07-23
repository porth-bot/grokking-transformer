"""Dataset correctness and end-to-end trainability."""

import pytest
import torch

from grokking.data import (
    OPERATIONS,
    modular_addition_dataset,
    modular_dataset,
    train_test_split,
)
from grokking.model import ModelConfig, Transformer
from grokking.train import (
    EMBEDDING_PARAMS,
    TrainConfig,
    train,
    weight_decay_groups,
)


def test_dataset_is_exhaustive_and_correct():
    p = 11
    tokens, targets = modular_addition_dataset(p)
    assert tokens.shape == (p * p, 3)
    # every ordered pair exactly once
    pairs = {(int(a), int(b)) for a, b, _ in tokens}
    assert len(pairs) == p * p
    # labels actually implement modular addition; '=' token is id p
    assert torch.all(tokens[:, 2] == p)
    assert torch.all(targets == (tokens[:, 0] + tokens[:, 1]) % p)


# --- multi-operation datasets (sub / mul, the Day-20 comparison) -----------


@pytest.mark.parametrize("operation", sorted(OPERATIONS))
def test_every_operation_is_exhaustive_and_labels_are_correct(operation):
    p = 11
    tokens, targets = modular_dataset(p, operation)
    assert tokens.shape == (p * p, 3)
    # every ordered pair exactly once, '=' token is id p, all labels in range
    assert len({(int(a), int(b)) for a, b, _ in tokens}) == p * p
    assert torch.all(tokens[:, 2] == p)
    assert torch.all((0 <= targets) & (targets < p))
    a, b = tokens[:, 0], tokens[:, 1]
    expected = {"add": (a + b), "sub": (a - b), "mul": (a * b)}[operation] % p
    assert torch.all(targets == expected)


def test_subtraction_labels_include_the_wrapped_negatives():
    # a - b is negative for a < b; the modulo must map it into 0..p-1, not
    # leave a raw negative (torch's % already does this -- pin it).
    _, targets = modular_dataset(7, "sub")
    assert torch.all((0 <= targets) & (targets < 7))
    assert int(modular_dataset(7, "sub")[1][7 * 2 + 5]) == (2 - 5) % 7  # == 4


def test_multiplication_by_zero_row_and_column_is_all_zero():
    # the 2p-1 pairs with a=0 or b=0 collapse to 0 -- the trivial residue
    # outside the multiplicative group (documented in the module + README).
    p = 13
    tokens, targets = modular_dataset(p, "mul")
    zero_mask = (tokens[:, 0] == 0) | (tokens[:, 1] == 0)
    assert int(zero_mask.sum()) == 2 * p - 1
    assert torch.all(targets[zero_mask] == 0)


def test_addition_wrapper_matches_the_general_dataset():
    for p in (5, 13):
        wa, wb = modular_addition_dataset(p)
        ga, gb = modular_dataset(p, "add")
        assert torch.equal(wa, ga) and torch.equal(wb, gb)


def test_unknown_operation_is_rejected():
    with pytest.raises(ValueError):
        modular_dataset(11, "div")


def test_operation_only_tags_run_name_when_not_addition():
    add = TrainConfig(p=97, train_frac=0.30, weight_decay=1.0, seed=0)
    assert add.run_name() == "p97_frac0.30_wd1_seed0"
    mul = TrainConfig(p=97, train_frac=0.30, weight_decay=1.0, seed=0,
                      operation="mul")
    assert mul.run_name() == "p97_frac0.30_wd1_seed0_opmul"


def test_multiplication_memorizes_a_small_problem_end_to_end():
    # sanity that the non-addition path trains: mul on a small modulus must
    # still reach 100% train accuracy in a few hundred full-batch steps.
    cfg = TrainConfig(
        p=13, train_frac=0.6, weight_decay=0.0, operation="mul",
        max_steps=600, eval_every=50, seed=0, device="cpu",
        model=ModelConfig(d_model=64, n_heads=4, d_mlp=128),
    )
    _, summary = train(cfg, out_dir="runs_test", verbose=False)
    assert summary["final_train_acc"] == 1.0
    assert summary["config"]["operation"] == "mul"


def test_split_is_disjoint_and_exhaustive():
    tokens, targets = modular_addition_dataset(13)
    (tr_x, _), (te_x, _) = train_test_split(tokens, targets, 0.37, seed=5)
    tr = {tuple(r.tolist()) for r in tr_x}
    te = {tuple(r.tolist()) for r in te_x}
    assert not tr & te
    assert len(tr) + len(te) == 13 * 13
    # deterministic in the seed
    (tr_x2, _), _ = train_test_split(tokens, targets, 0.37, seed=5)
    assert torch.equal(tr_x, tr_x2)


def test_training_loop_memorizes_small_problem():
    """End-to-end sanity on CPU: with no weight decay and plenty of capacity,
    a few hundred full-batch steps must drive train accuracy to 100% on a
    small modulus (this is the memorization phase grokking starts from)."""
    cfg = TrainConfig(
        p=13,
        train_frac=0.5,
        weight_decay=0.0,
        max_steps=600,
        eval_every=50,
        seed=0,
        device="cpu",
        model=ModelConfig(d_model=64, n_heads=4, d_mlp=128),
    )
    history, summary = train(cfg, out_dir="runs_test", verbose=False)
    assert summary["final_train_acc"] == 1.0
    assert summary["memorize_step"] is not None


# --- weight-decay scoping (the "which params' norm pressure?" ablation) -----


def _small_model():
    return Transformer(ModelConfig(p=13, vocab_size=14, d_model=32, n_heads=4,
                                   d_mlp=64))


def test_wd_scope_all_uses_a_single_default_group():
    # "all" is signalled by None so train() takes the untouched single-group
    # AdamW path -- the standard run stays bit-for-bit unchanged.
    assert weight_decay_groups(_small_model(), 1.0, "all") is None


def test_wd_scope_embeddings_decays_only_the_two_embedding_tensors():
    model = _small_model()
    named = dict(model.named_parameters())
    emb_ids = {id(named[n]) for n in EMBEDDING_PARAMS}

    groups = weight_decay_groups(model, 1.0, "embeddings")
    decayed, free = groups
    assert decayed["weight_decay"] == 1.0 and free["weight_decay"] == 0.0
    assert {id(p) for p in decayed["params"]} == emb_ids
    # partition is exhaustive and disjoint
    assert len(decayed["params"]) + len(free["params"]) == len(named)
    assert emb_ids.isdisjoint({id(p) for p in free["params"]})


def test_wd_scope_non_embeddings_is_the_complementary_partition():
    model = _small_model()
    named = dict(model.named_parameters())
    emb_ids = {id(named[n]) for n in EMBEDDING_PARAMS}

    groups = weight_decay_groups(model, 1.0, "non_embeddings")
    decayed, free = groups
    assert decayed["weight_decay"] == 1.0 and free["weight_decay"] == 0.0
    # now the embeddings are the *free* group
    assert {id(p) for p in free["params"]} == emb_ids
    assert emb_ids.isdisjoint({id(p) for p in decayed["params"]})
    assert len(decayed["params"]) + len(free["params"]) == len(named)


def test_wd_scope_rejects_unknown_scope():
    with pytest.raises(ValueError):
        weight_decay_groups(_small_model(), 1.0, "just_the_mlp")


def test_wd_scope_only_tags_run_name_when_restricted():
    base = TrainConfig(p=97, train_frac=0.30, weight_decay=1.0, seed=0)
    assert base.run_name() == "p97_frac0.30_wd1_seed0"
    scoped = TrainConfig(p=97, train_frac=0.30, weight_decay=1.0, seed=0,
                         wd_scope="embeddings")
    assert scoped.run_name() == "p97_frac0.30_wd1_seed0_wdsembeddings"


def test_scoped_weight_decay_still_trains():
    # A scoped optimizer must still be a valid AdamW that memorizes a small
    # problem (both groups receive gradients and step).
    cfg = TrainConfig(
        p=13, train_frac=0.5, weight_decay=1.0, wd_scope="non_embeddings",
        max_steps=600, eval_every=50, seed=0, device="cpu",
        model=ModelConfig(d_model=64, n_heads=4, d_mlp=128),
    )
    _, summary = train(cfg, out_dir="runs_test", verbose=False)
    assert summary["final_train_acc"] == 1.0
