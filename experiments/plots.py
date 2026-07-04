"""Generate all figures from the sweep's CSV logs. Rerunnable without retraining.

Run:  python experiments/plots.py
"""

import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update(
    {
        "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False,
    }
)

RUNS = Path(__file__).resolve().parent.parent / "runs"
FIGS = Path(__file__).resolve().parent.parent / "figures"


def load(name):
    with open(RUNS / f"{name}.csv") as f:
        rows = list(csv.DictReader(f))
    hist = {k: [float(r[k]) for r in rows] for k in rows[0]}
    with open(RUNS / f"{name}.json") as f:
        summary = json.load(f)
    return hist, summary


def savefig(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(FIGS / name, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figures/{name}")


def nonzero_steps(h):
    """Log-x plots can't show step 0; shift it to the first eval interval."""
    return [max(s, 1) for s in h["step"]]


def main_grokking_figure():
    h, s = load("p97_frac0.30_wd1_seed0")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)

    ax = axes[0]
    ax.plot(nonzero_steps(h), h["train_acc"], label="train", lw=1.6)
    ax.plot(nonzero_steps(h), h["test_acc"], label="test", lw=1.6)
    ax.set_xscale("log")
    ax.set_xlabel("step (log scale)")
    ax.set_ylabel("accuracy")
    if s["memorize_step"]:
        ax.axvline(s["memorize_step"], color="gray", ls=":", lw=1)
        ax.annotate("memorized", (s["memorize_step"], 0.55), rotation=90,
                    fontsize=7, color="gray", ha="right")
    if s["grok_step"]:
        ax.axvline(s["grok_step"], color="gray", ls="--", lw=1)
        ax.annotate("grokked", (s["grok_step"], 0.45), rotation=90,
                    fontsize=7, color="gray", ha="right")
    ax.set_title(f"(a+b) mod 97, 30% train data, wd=1.0", loc="left")
    ax.legend(loc="center left")

    ax = axes[1]
    ax.plot(nonzero_steps(h), h["weight_norm"], color="C2", lw=1.6)
    ax.set_xscale("log")
    ax.set_xlabel("step (log scale)")
    ax.set_ylabel(r"$\|\theta\|_2$ (all parameters)")
    if s["grok_step"]:
        ax.axvline(s["grok_step"], color="gray", ls="--", lw=1)
    ax.set_title("Norm rises to the transition, then decay takes over", loc="left")
    savefig(fig, "grokking_main.png")


def loss_figure():
    h, _ = load("p97_frac0.30_wd1_seed0")
    fig, ax = plt.subplots(figsize=(5.2, 3.2), constrained_layout=True)
    ax.plot(nonzero_steps(h), h["train_loss"], label="train loss", lw=1.4)
    ax.plot(nonzero_steps(h), h["test_loss"], label="test loss", lw=1.4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("step (log scale)")
    ax.set_ylabel("cross-entropy (log scale)")
    ax.set_title("Test loss rises while the model memorizes, then collapses",
                 loc="left")
    ax.legend()
    savefig(fig, "grokking_loss.png")


def wd_sweep_figure():
    fig, ax = plt.subplots(figsize=(5.6, 3.4), constrained_layout=True)
    for wd, color in [(0.0, "C3"), (0.1, "C1"), (1.0, "C0")]:
        h, s = load(f"p97_frac0.30_wd{wd:g}_seed0")
        ax.plot(nonzero_steps(h), h["test_acc"], lw=1.6, color=color,
                label=f"wd = {wd:g}" + (f"  (grok @ {s['grok_step']})" if s["grok_step"] else "  (never)"))
    ax.set_xscale("log")
    ax.set_xlabel("step (log scale)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Weight decay is the difference between memorizing and learning",
                 loc="left")
    ax.legend(loc="upper left", fontsize=8)
    savefig(fig, "wd_sweep.png")


def frac_sweep_figure():
    fracs = [0.25, 0.30, 0.40, 0.60]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    ax = axes[0]
    grok_steps = []
    for frac in fracs:
        h, s = load(f"p97_frac{frac:.2f}_wd1_seed0")
        ax.plot(nonzero_steps(h), h["test_acc"], lw=1.4, label=f"{int(frac*100)}% train")
        grok_steps.append(s["grok_step"])
    ax.set_xscale("log")
    ax.set_xlabel("step (log scale)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Less data, longer trance", loc="left")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot([f * 100 for f in fracs], grok_steps, "o-", color="C0")
    ax.set_yscale("log")
    ax.set_xlabel("train fraction (%)")
    ax.set_ylabel("steps to 99% test acc (log)")
    ax.set_title("Time-to-grok vs data fraction", loc="left")
    savefig(fig, "frac_sweep.png")


if __name__ == "__main__":
    main_grokking_figure()
    loss_figure()
    wd_sweep_figure()
    frac_sweep_figure()
