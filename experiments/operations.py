"""Does grokking depend on the operation, or just its group structure?

The canonical task is (a+b) mod p. This experiment asks whether the *same*
delayed generalization appears for two other binary operations on the same
digit vocabulary, holding everything else fixed (frac 0.30, one layer, 4 heads,
lr 1e-3), at both a strong and a weak weight decay (wd in {1.0, 0.1}), over
three seeds:

- **(a - b) mod p** is still the additive group of Z/pZ; negating the second
  operand is a relabelling of the answer, so the Fourier-addition circuit
  transfers unchanged. Prediction: groks, on the same timescale as addition.

- **(a * b) mod p** is the interesting one. On the *nonzero* residues it is the
  cyclic MULTIPLICATIVE group (Z/pZ)^x of order p-1, and the discrete logarithm
  to a primitive root g (a = g^i, b = g^j => a*b = g^((i+j) mod (p-1))) makes it
  ISOMORPHIC to addition mod (p-1). So multiplication should still grok -- it is
  addition in disguise -- but in a p-1 = 96 element group, and the 2p-1 = 193
  pairs that involve a 0 (product 0) sit outside that group and can only be
  memorized. Prediction: groks; the writeup states the isomorphism explicitly.

Seeds. Day 20 ran seed 0 only and flagged the comparison as directional. Day 22
adds seeds 1 and 2, so every cell is a median over three runs with a min-max
range, and "does the ordering survive seed noise?" gets a measured answer
instead of a caveat. Three seeds is still not a distribution -- the same honest
framing as the wd/frac error bars in section 3.

The addition rows reuse the committed sweep CSVs (p97_frac0.30_wd{1,0.1}_seedN,
already present for seeds 0-4 from the multi-seed error-bar runs); only the
sub/mul runs are computed here, tagged ``_opsub`` / ``_opmul`` in the run name
so no existing artifact is touched. Resumable: existing summaries are skipped.

Produces ``figures/operations.png`` (median test accuracy with a min-max band,
one panel per weight decay) from the committed CSVs.

Run:  python experiments/operations.py   (~1 h on MPS for the eight new runs)
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from grokking.aggregate import align_and_aggregate, fmt_median_range  # noqa: E402
from grokking.train import TrainConfig, train  # noqa: E402

MAX_STEPS = 25_000   # same budget as the main run
ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
FIGS = ROOT / "figures"

OPERATIONS = ["add", "sub", "mul"]
WEIGHT_DECAYS = [1.0, 0.1]
SEEDS = [0, 1, 2]

# Human-readable labels for the figure/table (the answer's group, and its order).
GROUP = {
    "add": "(a+b) mod 97   [Z/97, order 97]",
    "sub": "(a−b) mod 97   [Z/97, order 97]",
    "mul": "(a×b) mod 97   [(Z/97)ˣ, order 96]",
}


def cfg_for(operation, wd, seed=0):
    return TrainConfig(
        p=97, train_frac=0.30, weight_decay=wd, operation=operation, lr=1e-3,
        max_steps=MAX_STEPS, eval_every=100, seed=seed,
    )


def run(operation, wd, seed):
    """Train (or reuse) one (operation, wd, seed) cell; return its run name."""
    cfg = cfg_for(operation, wd, seed)
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


def figure_and_table():
    """Comparison table + a two-panel test-accuracy figure, committed CSVs only.

    Each (operation, wd) cell is aggregated across ``SEEDS`` the same way the
    wd/frac error bars are: forward-fill every seed onto the union step grid,
    then plot the median with a band. The band is min-max here rather than the
    IQR -- with three seeds the quartiles carry no more information than the
    extremes, and the extremes are the honest thing to show.
    """
    colors = {"add": "C0", "sub": "C2", "mul": "C3"}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True,
                             constrained_layout=True)
    rows = []
    for ax, wd in zip(axes, WEIGHT_DECAYS):
        for op in OPERATIONS:
            loaded = [_load(cfg_for(op, wd, s).run_name()) for s in SEEDS]
            grid, med, lo, hi = align_and_aggregate(
                [h["step"] for h, _ in loaded],
                [h["test_acc"] for h, _ in loaded],
                lo_pct=0.0, hi_pct=100.0,
            )
            steps = [max(int(st), 1) for st in grid]
            ax.plot(steps, med, lw=1.6, color=colors[op],
                    label=GROUP[op].split("  ")[0])
            ax.fill_between(steps, lo, hi, color=colors[op], alpha=0.15, lw=0)
            rows.append((
                op, wd,
                [s["memorize_step"] for _, s in loaded],
                [s["grok_step"] for _, s in loaded],
                [s["final_test_acc"] for _, s in loaded],
            ))
        ax.set_xscale("log")
        ax.set_xlabel("step (log scale)")
        ax.set_title(f"weight decay {wd:g}", loc="left")
    axes[0].set_ylabel("test accuracy")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"Grokking across modular operations "
                 f"(frac 0.30, median of {len(SEEDS)} seeds, band = min–max)",
                 x=0.01, ha="left", fontsize=11)
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "operations.png", bbox_inches="tight")
    plt.close(fig)
    print("saved figures/operations.png")

    # sort the printed table by (wd desc, op order) for a stable, readable block
    order = {op: i for i, op in enumerate(OPERATIONS)}
    rows.sort(key=lambda r: (-r[1], order[r[0]]))
    print(f"\nmedian [min–max] over seeds {SEEDS}")
    print(f"{'operation':>26} {'wd':>4} {'memorize':>16} {'grok':>24} "
          f"{'final test':>16}")
    for op, wd, mem, grok, test in rows:
        acc = "  ".join(f"{v:.3f}" for v in test)
        print(f"{GROUP[op]:>26} {wd:>4g} {fmt_median_range(mem):>16} "
              f"{fmt_median_range(grok):>24} {acc:>16}")


if __name__ == "__main__":
    for wd in WEIGHT_DECAYS:
        for op in ("sub", "mul"):   # add reuses the committed sweep CSVs
            for seed in SEEDS:
                run(op, wd, seed)
    figure_and_table()
