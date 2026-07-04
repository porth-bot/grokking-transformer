"""Dataset correctness and end-to-end trainability."""

import torch

from grokking.data import modular_addition_dataset, train_test_split
from grokking.model import ModelConfig
from grokking.train import TrainConfig, train


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
