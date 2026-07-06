"""Learning-rate sensitivity of the grok time (frac=0.30, wd=1.0, seed 0).

Delayed generalization is reported at one AdamW learning rate (1e-3). Is the
grok time an artifact of that tuned lr, or does the memorize-then-grok
phenomenon survive an order-of-magnitude change? We rerun the main config at
lr in {3e-4, 1e-3, 3e-3}, holding everything else fixed, and compare the
memorization and grokking steps.

Runs go to ``runs_lr/`` (resumable; the .pt checkpoints are not committed --
this experiment needs only the CSV trajectories -- see .gitignore). Produces
``figures/lr_sweep.png`` and prints a summary table.

Run:  python experiments/lr_sweep.py    (~20 min on MPS: three full runs)
"""

import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from grokking.train import TrainConfig, train  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs_lr"
FIGS = ROOT / "figures"
LRS = [3e-4, 1e-3, 3e-3]
COLORS = {3e-4: "C3", 1e-3: "C0", 3e-3: "C1"}
MAX_STEPS = 25_000

plt.rcParams.update(
    {
        "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False,
    }
)


def cfg_for(lr):
    return TrainConfig(
        p=97, train_frac=0.30, weight_decay=1.0, lr=lr,
        max_steps=MAX_STEPS, eval_every=100, seed=0,
    )


def run_all():
    for lr in LRS:
        cfg = cfg_for(lr)
        if (RUNS / (cfg.run_name() + ".json")).exists():
            print(f"skip {cfg.run_name()} (already done)", flush=True)
            continue
        print(f"=== {cfg.run_name()} on {cfg.device} ===", flush=True)
        train(cfg, out_dir=str(RUNS))


def load(name):
    with open(RUNS / f"{name}.csv") as f:
        rows = list(csv.DictReader(f))
    hist = {k: [float(r[k]) for r in rows] for k in rows[0]}
    with open(RUNS / f"{name}.json") as f:
        summary = json.load(f)
    return hist, summary


def figure_and_table():
    fig, ax = plt.subplots(figsize=(5.8, 3.4), constrained_layout=True)
    rows = []
    for lr in LRS:
        h, s = load(cfg_for(lr).run_name())
        steps = [max(st, 1) for st in h["step"]]  # log-x can't show step 0
        grok = s["grok_step"]
        ax.plot(steps, h["test_acc"], lw=1.6, color=COLORS[lr],
                label=f"lr = {lr:g}" + (f"  (grok @ {grok})" if grok else "  (never)"))
        if grok:
            ax.axvline(grok, color=COLORS[lr], ls=":", lw=1, alpha=0.6)
        rows.append((lr, s["memorize_step"], grok, s["final_test_acc"], s["steps_run"]))
    ax.set_xscale("log")
    ax.set_xlabel("step (log scale)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Grokking survives a 10x lr change; the grok step tracks lr",
                 loc="left")
    ax.legend(loc="center left", fontsize=8)
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(FIGS / "lr_sweep.png", bbox_inches="tight")
    plt.close(fig)
    print("saved figures/lr_sweep.png")

    print("\n   lr   memorize   grok   final_test_acc   steps_run")
    print("  " + "-" * 52)
    for lr, mem, grok, acc, steps in rows:
        print(f"  {lr:>5g}   {str(mem):>7}   {str(grok):>5}   "
              f"{acc:>12.3f}   {steps:>8}")


if __name__ == "__main__":
    run_all()
    figure_and_table()
