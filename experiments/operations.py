"""Does grokking depend on the operation, or just its group structure?

The canonical task is (a+b) mod p. This experiment asks whether the *same*
delayed generalization appears for two other binary operations on the same
digit vocabulary, holding everything else fixed (frac 0.30, seed 0, one layer,
4 heads, lr 1e-3), at both a strong and a weak weight decay (wd in {1.0, 0.1}):

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

The addition rows reuse the committed main-run CSVs (p97_frac0.30_wd1_seed0 and
p97_frac0.30_wd0.1_seed0); only the four sub/mul runs are computed here, tagged
``_opsub`` / ``_opmul`` in the run name so no existing artifact is touched.
Resumable: existing summaries are skipped. Single seed (0) -- like the head-count
ablation, this is a directional comparison, not an error-bar study; Day 22 adds
two more seeds to the table.

Produces ``figures/operations.png`` (test accuracy over training, one panel per
weight decay) from the committed CSVs.

Run:  python experiments/operations.py   (~10-20 min on MPS for the four runs)
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from grokking.train import TrainConfig, train  # noqa: E402

MAX_STEPS = 25_000   # same budget as the main run
ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
FIGS = ROOT / "figures"

OPERATIONS = ["add", "sub", "mul"]
WEIGHT_DECAYS = [1.0, 0.1]

# Human-readable labels for the figure/table (the answer's group, and its order).
GROUP = {
    "add": "(a+b) mod 97   [Z/97, order 97]",
    "sub": "(a−b) mod 97   [Z/97, order 97]",
    "mul": "(a×b) mod 97   [(Z/97)ˣ, order 96]",
}


def cfg_for(operation, wd):
    return TrainConfig(
        p=97, train_frac=0.30, weight_decay=wd, operation=operation, lr=1e-3,
        max_steps=MAX_STEPS, eval_every=100, seed=0,
    )


def run(operation, wd):
    """Train (or reuse) one (operation, wd) cell; return its run name."""
    cfg = cfg_for(operation, wd)
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
    """Comparison table + a two-panel test-accuracy figure, committed CSVs only."""
    colors = {"add": "C0", "sub": "C2", "mul": "C3"}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True,
                             constrained_layout=True)
    rows = []
    for ax, wd in zip(axes, WEIGHT_DECAYS):
        for op in OPERATIONS:
            h, s = _load(cfg_for(op, wd).run_name())
            steps = [max(st, 1) for st in h["step"]]
            ax.plot(steps, h["test_acc"], lw=1.6, color=colors[op],
                    label=GROUP[op].split("  ")[0])
            rows.append((op, wd, s["memorize_step"], s["grok_step"],
                         s["final_train_acc"], s["final_test_acc"]))
        ax.set_xscale("log")
        ax.set_xlabel("step (log scale)")
        ax.set_title(f"weight decay {wd:g}", loc="left")
    axes[0].set_ylabel("test accuracy")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("Grokking across modular operations (frac 0.30, seed 0)",
                 x=0.01, ha="left", fontsize=11)
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "operations.png", bbox_inches="tight")
    plt.close(fig)
    print("saved figures/operations.png")

    # sort the printed table by (wd desc, op order) for a stable, readable block
    order = {op: i for i, op in enumerate(OPERATIONS)}
    rows.sort(key=lambda r: (-r[1], order[r[0]]))
    print(f"\n{'operation':>26} {'wd':>4} {'memorize':>9} {'grok':>7} "
          f"{'final train':>12} {'final test':>11}")
    for op, wd, mem, grok, tr, te in rows:
        grok_s = str(grok) if grok is not None else "never"
        print(f"{GROUP[op]:>26} {wd:>4g} {str(mem):>9} {grok_s:>7} "
              f"{tr:>12.3f} {te:>11.3f}")


if __name__ == "__main__":
    for wd in WEIGHT_DECAYS:
        for op in ("sub", "mul"):   # add reuses the committed main-run CSVs
            run(op, wd)
    figure_and_table()
