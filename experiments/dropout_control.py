"""Does *any* regularizer grok, or specifically weight-norm pressure?

Weight decay is what turns memorization into generalization in the main runs.
But weight decay is a very specific regularizer -- it pushes the parameter
norm down, and the Omnigrok picture (Liu et al. 2023) argues that norm
reduction is the mechanism, not regularization in general. This control tests
that claim directly: swap weight decay for **dropout** (a regularizer that
does *not* target the weight norm) and hold everything else fixed.

Configuration: train fraction 0.30, seed 0, lr 1e-3, **dropout 0.1, weight
decay 0** -- 30% of the data, the only regularizer changed from wd to dropout.
Compared against two runs already in ``runs/`` at the same frac/seed/lr:

    wd 0   (no regularizer)   -> never groks
    wd 1.0 (the main run)     -> groks

If dropout groks, "any regularizer works"; if it does not, grokking here is
specifically about the weight norm, and dropout -- which leaves the norm free
to stay large -- cannot buy generalization no matter how long it trains.

Reuses runs/p97_frac0.30_wd0_seed0.* and runs/p97_frac0.30_wd1_seed0.* for the
two baselines; only the dropout run is computed here. Resumable: an existing
dropout summary is not recomputed. Produces ``figures/dropout_control.png``
(test accuracy and weight norm over training for the three runs) from the
committed CSVs.

Run:  python experiments/dropout_control.py   (~4 min on MPS for the one run)
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from grokking.model import ModelConfig  # noqa: E402
from grokking.train import TrainConfig, train  # noqa: E402

MAX_STEPS = 25_000   # same budget as the never-grokking wd=0 baseline
ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
FIGS = ROOT / "figures"

plt.rcParams.update(
    {
        "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False,
    }
)


def run_dropout():
    cfg = TrainConfig(
        p=97, train_frac=0.30, weight_decay=0.0, lr=1e-3,
        max_steps=MAX_STEPS, eval_every=100, seed=0,
        model=ModelConfig(dropout=0.1),
    )
    if not RUNS.joinpath(cfg.run_name() + ".json").exists():
        print(f"=== {cfg.run_name()} on {cfg.device} ===", flush=True)
        train(cfg, out_dir=str(RUNS))
    else:
        print(f"skip {cfg.run_name()} (already done)", flush=True)
    return cfg.run_name()


def _load(name):
    with open(RUNS / f"{name}.csv") as f:
        rows = list(csv.DictReader(f))
    hist = {k: [float(r[k]) for r in rows] for k in rows[0]}
    with open(RUNS / f"{name}.json") as f:
        summary = json.load(f)
    return hist, summary


def figure_and_table(dropout_name):
    """Comparison table + the two-panel figure, from committed CSVs."""
    runs = [
        ("wd 0, no reg", "p97_frac0.30_wd0_seed0", "C3"),
        ("dropout 0.1, wd 0", dropout_name, "C2"),
        ("wd 1.0 (main)", "p97_frac0.30_wd1_seed0", "C0"),
    ]

    fig, (ax_acc, ax_norm) = plt.subplots(
        1, 2, figsize=(10, 3.6), constrained_layout=True
    )
    rows = []
    for label, name, color in runs:
        h, s = _load(name)
        steps = [max(st, 1) for st in h["step"]]  # log-x can't show step 0
        ax_acc.plot(steps, h["test_acc"], lw=1.6, color=color, label=label)
        ax_norm.plot(steps, h["weight_norm"], lw=1.6, color=color, label=label)
        rows.append((label, s["memorize_step"], s["grok_step"],
                     s["final_train_acc"], s["final_test_acc"]))
    for ax in (ax_acc, ax_norm):
        ax.set_xscale("log")
        ax.set_xlabel("step (log scale)")
        ax.legend(loc="best", fontsize=8)
    ax_acc.set_ylabel("test accuracy")
    ax_acc.set_title("Dropout groks too", loc="left")
    ax_norm.set_yscale("log")
    ax_norm.set_ylabel("weight norm $\\|\\theta\\|$ (log)")
    ax_norm.set_title("...while its weight norm grows, not shrinks", loc="left")
    fig.suptitle(
        "Dropout (wd 0) generalizes like weight decay does, but through a "
        "different mechanism: the norm rises the whole time.", fontsize=9,
    )
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "dropout_control.png", bbox_inches="tight")
    plt.close(fig)
    print("saved figures/dropout_control.png")

    print(f"\n{'regularizer':>20} {'memorize':>9} {'grok':>7} "
          f"{'final train':>12} {'final test':>11}")
    for label, mem, grok, tr, te in rows:
        grok_s = str(grok) if grok is not None else "never"
        print(f"{label:>20} {str(mem):>9} {grok_s:>7} {tr:>12.3f} {te:>11.3f}")


if __name__ == "__main__":
    name = run_dropout()
    figure_and_table(name)
