"""Does grokking need multiple attention heads?

The main runs use 4 heads (d_model 128 -> d_head 32). Modular addition has a
known mechanistic solution -- embed each input on a circle at a few Fourier
frequencies, add the angles in the attention/MLP, read the sum off by
interference (Nanda et al. 2023) -- and nothing in that circuit obviously needs
the representation split across several heads. This ablation asks the question
directly: hold the main config fixed (frac 0.30, wd 1.0, seed 0, lr 1e-3, one
layer) and vary only ``n_heads`` in {1, 2, 4}. With d_model fixed at 128 the
head width tracks the count (128 / 64 / 32), so this is genuinely "how finely is
attention partitioned", not "how much total width".

Reuses the 4-head main run already in ``runs/`` (p97_frac0.30_wd1_seed0); only
the 1- and 2-head runs are computed here (tagged ``_h1``/``_h2`` in the run
name). Resumable: existing summaries are not recomputed. Produces
``figures/head_count.png`` (test accuracy over training for the three counts)
from the committed CSVs.

Run:  python experiments/head_count.py   (~5-15 min on MPS for the two new runs)
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from grokking.model import ModelConfig  # noqa: E402
from grokking.train import TrainConfig, train  # noqa: E402

MAX_STEPS = 25_000   # same budget as the main run
ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
FIGS = ROOT / "figures"
HEADS = [1, 2, 4]

plt.rcParams.update(
    {
        "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False,
    }
)


def run_head(n_heads):
    """Train (or reuse) the main config with the given head count; return name."""
    cfg = TrainConfig(
        p=97, train_frac=0.30, weight_decay=1.0, lr=1e-3,
        max_steps=MAX_STEPS, eval_every=100, seed=0,
        model=ModelConfig(n_heads=n_heads),
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


def figure_and_table(names):
    """Comparison table + the test-accuracy figure, from committed CSVs."""
    colors = {1: "C3", 2: "C1", 4: "C0"}
    fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
    rows = []
    for n_heads in HEADS:
        h, s = _load(names[n_heads])
        steps = [max(st, 1) for st in h["step"]]
        ax.plot(steps, h["test_acc"], lw=1.6, color=colors[n_heads],
                label=f"{n_heads} head{'s' if n_heads > 1 else ''} "
                      f"(d_head {128 // n_heads})")
        rows.append((n_heads, s["memorize_step"], s["grok_step"],
                     s["final_train_acc"], s["final_test_acc"]))
    ax.set_xscale("log")
    ax.set_xlabel("step (log scale)")
    ax.set_ylabel("test accuracy")
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Grokking on (a+b) mod 97 across attention head counts",
                 loc="left")
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "head_count.png", bbox_inches="tight")
    plt.close(fig)
    print("saved figures/head_count.png")

    print(f"\n{'n_heads':>8} {'d_head':>7} {'memorize':>9} {'grok':>7} "
          f"{'final train':>12} {'final test':>11}")
    for n_heads, mem, grok, tr, te in rows:
        grok_s = str(grok) if grok is not None else "never"
        print(f"{n_heads:>8} {128 // n_heads:>7} {str(mem):>9} {grok_s:>7} "
              f"{tr:>12.3f} {te:>11.3f}")


if __name__ == "__main__":
    names = {n: run_head(n) for n in HEADS}
    figure_and_table(names)
