"""Save and reload trained models from a run's committed artifacts.

A checkpoint on disk is just a ``state_dict`` (``<run>.pt`` for the final model,
``<run>_memorize.pt`` for the memorization point). The architecture needed to
*receive* those weights lives in the run's JSON summary, under
``config.model`` -- ``train.py`` writes it there. Rebuilding the model from that
saved config, instead of hardcoding ``p``/``d_model`` at each call site, is what
lets every figure be regenerated from committed files alone and keeps the
analysis correct for runs that used a different modulus or width (e.g. the
p=113 scaling run, or a wider model).

Typical use::

    model, summary = load_model("p97_frac0.30_wd1_seed0", which="final")
    model, _       = load_model("p97_frac0.30_wd1_seed0", which="memorize")
"""

import json
from dataclasses import fields
from pathlib import Path

import torch

from .model import ModelConfig, Transformer

DEFAULT_RUNS = Path(__file__).resolve().parent.parent / "runs"


def load_summary(run_name, runs_dir=DEFAULT_RUNS):
    """Read the committed ``<run_name>.json`` summary (config + results)."""
    with open(Path(runs_dir) / f"{run_name}.json") as f:
        return json.load(f)


def model_config_from_summary(summary):
    """Reconstruct the exact ModelConfig a run was trained with.

    Only keys that are real ModelConfig fields are used (``d_head`` is a derived
    property and never stored), so this is robust to extra keys in the summary.
    """
    saved = summary["config"]["model"]
    valid = {f.name for f in fields(ModelConfig)}
    return ModelConfig(**{k: v for k, v in saved.items() if k in valid})


def checkpoint_path(run_name, which="final", runs_dir=DEFAULT_RUNS):
    if which not in ("final", "memorize"):
        raise ValueError("which must be 'final' or 'memorize'")
    suffix = "" if which == "final" else "_memorize"
    return Path(runs_dir) / f"{run_name}{suffix}.pt"


def load_model(run_name, which="final", runs_dir=DEFAULT_RUNS, map_location="cpu"):
    """Rebuild the model for a run and load its weights.

    Parameters
    ----------
    run_name : str          e.g. "p97_frac0.30_wd1_seed0"
    which : {"final", "memorize"}
        Which checkpoint to load -- the end of training or the memorization
        point (first eval with train acc >= 99.9%).

    Returns
    -------
    (model, summary) : the eval-mode Transformer and the run's summary dict.
    """
    summary = load_summary(run_name, runs_dir)
    model = Transformer(model_config_from_summary(summary))
    state = torch.load(checkpoint_path(run_name, which, runs_dir), map_location=map_location)
    model.load_state_dict(state)
    model.eval()
    return model, summary
