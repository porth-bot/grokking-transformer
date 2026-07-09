"""Generate all figures from the sweep's CSV logs. Rerunnable without retraining.

Run:  python experiments/plots.py
"""

import csv
import json
import os
import re
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from grokking.aggregate import align_and_aggregate, fmt_median_range  # noqa: E402

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


def seed_names(frac, wd):
    """Every completed seed for a (frac, wd) cell, ordered by seed index.

    Matches only the plain ``..._seed<N>`` runs -- not the dropout/lr variants
    that share the prefix (e.g. ``_seed0_do0.1``) -- so aggregation never mixes
    a regularizer control into the wd/frac error bars.
    """
    stem = f"p97_frac{frac:.2f}_wd{wd:g}_seed"
    pat = re.compile(re.escape(stem) + r"(\d+)\.csv$")
    found = []
    for p in RUNS.glob(f"{stem}*.csv"):
        m = pat.search(p.name)
        if m:
            found.append((int(m.group(1)), p.stem))
    return [name for _, name in sorted(found)]


def aggregate_cell(frac, wd, key="test_acc"):
    """Median + IQR band of a metric across all seeds of a (frac, wd) cell."""
    hs = [load(name)[0] for name in seed_names(frac, wd)]
    steps = [[max(s, 1) for s in h["step"]] for h in hs]
    return align_and_aggregate(steps, [h[key] for h in hs])


def grok_steps(frac, wd):
    """Per-seed grok step (None where a seed never grokked)."""
    return [load(name)[1]["grok_step"] for name in seed_names(frac, wd)]


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
    """Test accuracy vs step for three weight decays, median line + IQR band
    over the 5 seeds of each cell."""
    fig, ax = plt.subplots(figsize=(5.6, 3.4), constrained_layout=True)
    for wd, color in [(0.0, "C3"), (0.1, "C1"), (1.0, "C0")]:
        grid, med, lo, hi = aggregate_cell(0.30, wd)
        gs = grok_steps(0.30, wd)
        grid = [max(g, 1) for g in grid]
        ax.plot(grid, med, lw=1.7, color=color,
                label=f"wd = {wd:g}  (grok @ {fmt_median_range(gs)})")
        ax.fill_between(grid, lo, hi, color=color, alpha=0.18, lw=0)
    ax.set_xscale("log")
    ax.set_xlabel("step (log scale)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Weight decay is the difference between memorizing and learning\n"
                 "(median over 5 seeds, band = IQR)", loc="left", fontsize=9)
    ax.legend(loc="upper left", fontsize=7.5)
    savefig(fig, "wd_sweep.png")


def frac_sweep_figure():
    """Left: test-accuracy median+IQR bands for the multi-seed fractions.
    Right: time-to-grok median with min-max whiskers vs data fraction."""
    band_fracs = [0.25, 0.30, 0.40]     # 5-seed cells
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)

    ax = axes[0]
    for frac, color in zip(band_fracs, ["C0", "C1", "C2"]):
        grid, med, lo, hi = aggregate_cell(frac, 1.0)
        grid = [max(g, 1) for g in grid]
        ax.plot(grid, med, lw=1.5, color=color, label=f"{int(frac*100)}% train")
        ax.fill_between(grid, lo, hi, color=color, alpha=0.18, lw=0)
    ax.set_xscale("log")
    ax.set_xlabel("step (log scale)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Less data, longer trance (median + IQR)", loc="left")
    ax.legend(fontsize=8)

    ax = axes[1]
    # frac 0.60 stays a single-seed context point at the "delay vanishes" end.
    all_fracs = band_fracs + [0.60]
    xs, meds, los, his = [], [], [], []
    for frac in all_fracs:
        gs = grok_steps(frac, 1.0)
        vals = [g for g in gs if g is not None]
        if not vals:
            continue
        xs.append(frac * 100)
        meds.append(float(np.median(vals)))
        los.append(min(vals))
        his.append(max(vals))
    yerr = [[m - l for m, l in zip(meds, los)], [h - m for m, h in zip(meds, his)]]
    ax.errorbar(xs, meds, yerr=yerr, fmt="o-", color="C0", capsize=3, lw=1.3)
    ax.set_yscale("log")
    ax.set_xlabel("train fraction (%)")
    ax.set_ylabel("steps to 99% test acc (log)")
    ax.set_title("Time-to-grok vs data fraction\n(median, whiskers = min–max)",
                 loc="left", fontsize=9)
    savefig(fig, "frac_sweep.png")


def print_tables():
    """Emit the multi-seed numbers for the README tables (median [min–max])."""
    print("\nWeight-decay sweep (frac 0.30):")
    for wd in (0.0, 0.1, 1.0):
        gs = grok_steps(0.30, wd)
        print(f"  wd={wd:<4g} n={len(gs)}  grok: {fmt_median_range(gs)}")
    print("Data-fraction sweep (wd 1.0):")
    for frac in (0.25, 0.30, 0.40, 0.60):
        gs = grok_steps(frac, 1.0)
        print(f"  frac={frac:<4g} n={len(gs)}  grok: {fmt_median_range(gs)}")


if __name__ == "__main__":
    main_grokking_figure()
    loss_figure()
    wd_sweep_figure()
    frac_sweep_figure()
    print_tables()
