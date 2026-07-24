"""*Which* parameters' norm pressure drives grokking?

The main runs decay every weight, and the Omnigrok picture (Liu et al. 2023)
says the falling parameter norm is what turns memorization into
generalization. But "the norm" is the norm of *all* the weights. This ablation
asks a sharper question: is the pressure that matters on the **embeddings**
(the token/position lookup tables, where the Fourier structure of the learned
representation lives -- Nanda et al. 2023) or on the **rest** of the network
(attention + MLP + unembed, which read that representation out)?

We hold everything fixed at the main config (train fraction 0.30, weight decay
1.0, lr 1e-3, seed 0) and change only *where* the decay is applied:

    all             decay every parameter          (the main run, reused)
    embeddings      decay only tok_emb + pos_emb    (rest trains at wd 0)
    non_embeddings  decay everything except those   (embeddings at wd 0)

The untargeted group trains with plain Adam (wd 0), so each run isolates the
effect of shrinking one subset. If one scope groks near the all-decay time and
the other does not, the norm pressure that matters is localized; if both lag
the full-decay run, grokking needs pressure on the whole network at once.

Reuses runs/p97_frac0.30_wd1_seed0.* for the all-decay baseline; computes the
two scoped runs (resumable -- an existing summary is not recomputed).
Produces figures/wd_scope.png (test accuracy + weight norm over training) from
the committed CSVs.

Run:  python experiments/wd_scope.py
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from grokking.train import TrainConfig, train  # noqa: E402

MAX_STEPS = 15_000   # a bit above the all-decay run's natural ~11k stop
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


SCOPES = ("embeddings", "non_embeddings")


def cfg_for(scope):
    """The config for one scope arm -- the single source of truth for its
    run_name, shared with reproduce_figures.py so the artifact check and the
    training call cannot drift apart."""
    return TrainConfig(
        p=97, train_frac=0.30, weight_decay=1.0, wd_scope=scope, lr=1e-3,
        max_steps=MAX_STEPS, eval_every=100, seed=0,
    )


def run_scope(scope):
    cfg = cfg_for(scope)
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


def figure_and_table(emb_name, non_name):
    runs = [
        ("decay all (main)", "p97_frac0.30_wd1_seed0", "C0"),
        ("decay embeddings only", emb_name, "C2"),
        ("decay non-embeddings only", non_name, "C3"),
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
    ax_acc.set_title("Where weight decay is applied", loc="left")
    ax_norm.set_ylabel("weight norm $\\|\\theta\\|$")
    ax_norm.set_title("...and the total parameter norm", loc="left")
    fig.suptitle(
        "Grokking vs the scope of weight decay (frac 0.30, wd 1.0, seed 0)",
        fontsize=9,
    )
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "wd_scope.png", bbox_inches="tight")
    plt.close(fig)
    print("saved figures/wd_scope.png")

    print(f"\n{'weight-decay scope':>26} {'memorize':>9} {'grok':>7} "
          f"{'final train':>12} {'final test':>11}")
    for label, mem, grok, tr, te in rows:
        grok_s = str(grok) if grok is not None else "never"
        print(f"{label:>26} {str(mem):>9} {grok_s:>7} {tr:>12.3f} {te:>11.3f}")


if __name__ == "__main__":
    emb, non = (run_scope(scope) for scope in SCOPES)
    figure_and_table(emb, non)
