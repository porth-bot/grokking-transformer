"""The checkpoint loader must rebuild a run's model and load its real weights.

The strong check is behavioral: reconstruct the committed main run's exact
test split, load its final checkpoint, and confirm it reproduces the summary's
test accuracy -- if the config were rebuilt wrongly or the weights not loaded,
accuracy would collapse. The memorization checkpoint is the control: same
train accuracy, near-chance test accuracy.
"""

import json
import tempfile
from pathlib import Path

import torch

from grokking.checkpoints import (
    load_model,
    load_summary,
    model_config_from_summary,
)
from grokking.data import modular_addition_dataset, train_test_split
from grokking.model import ModelConfig, Transformer

MAIN = "p97_frac0.30_wd1_seed0"


@torch.no_grad()
def _accuracy(model, x, y):
    return float((model(x)[:, -1, :].argmax(-1) == y).float().mean())


def test_config_reconstructed_from_summary():
    summary = load_summary(MAIN)
    cfg = model_config_from_summary(summary)
    assert cfg.p == summary["config"]["p"]
    assert cfg.vocab_size == summary["config"]["p"] + 1
    # a derived (non-field) key in the summary must not break reconstruction
    poisoned = json.loads(json.dumps(summary))
    poisoned["config"]["model"]["d_head"] = 999
    assert model_config_from_summary(poisoned).d_model == cfg.d_model


def test_final_checkpoint_reproduces_reported_accuracy():
    model, summary = load_model(MAIN, which="final")
    assert model.n_params() == summary["n_params"]
    p = summary["config"]["p"]
    tokens, targets = modular_addition_dataset(p)
    (tr_x, tr_y), (te_x, te_y) = train_test_split(
        tokens, targets, summary["config"]["train_frac"], summary["config"]["seed"]
    )
    # weights loaded correctly => the reported final test accuracy is reproduced
    assert _accuracy(model, te_x, te_y) == summary["final_test_acc"]
    assert _accuracy(model, tr_x, tr_y) >= 0.999


def test_memorize_checkpoint_is_a_distinct_earlier_model():
    """The memorization checkpoint fits train but not test -- and is genuinely a
    different set of weights than the final one (not the same file loaded twice)."""
    mem, summary = load_model(MAIN, which="memorize")
    fin, _ = load_model(MAIN, which="final")
    p = summary["config"]["p"]
    tokens, targets = modular_addition_dataset(p)
    (tr_x, tr_y), (te_x, te_y) = train_test_split(
        tokens, targets, summary["config"]["train_frac"], summary["config"]["seed"]
    )
    assert _accuracy(mem, tr_x, tr_y) >= 0.999      # memorized the train set
    assert _accuracy(mem, te_x, te_y) < 0.5         # but has not grokked
    # the two checkpoints hold different embeddings
    assert not torch.equal(mem.tok_emb.weight, fin.tok_emb.weight)


def test_load_round_trip_recovers_state_dict():
    """A model saved with a matching JSON summary reloads bit-for-bit."""
    torch.manual_seed(0)
    cfg = ModelConfig(p=17, vocab_size=18, d_model=32, n_heads=4, d_mlp=64)
    model = Transformer(cfg)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        run = "p17_frac0.50_wd1_seed0"
        torch.save(model.state_dict(), d / f"{run}.pt")
        from dataclasses import asdict
        summary = {"config": {"p": 17, "model": asdict(cfg)}, "n_params": model.n_params()}
        (d / f"{run}.json").write_text(json.dumps(summary))
        reloaded, _ = load_model(run, which="final", runs_dir=d)
    for k, v in model.state_dict().items():
        assert torch.equal(v, reloaded.state_dict()[k])
