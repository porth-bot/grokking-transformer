"""Dataset correctness and end-to-end trainability."""

import pytest
import torch

from grokking.data import modular_addition_dataset, train_test_split
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
