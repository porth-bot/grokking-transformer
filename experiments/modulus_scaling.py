"""Does time-to-grok scale with the modulus p?

The headline runs all use p = 97. This experiment repeats the main
configuration (train fraction 0.30, weight decay 1.0, seed 0, lr 1e-3) at a
larger prime, p = 113, changing nothing else. The comparison is fair because
grok steps are only comparable at fixed learning rate (see the lr-sensitivity
section of the README) and lr is held at 1e-3 here.

Two forces pull in opposite directions as p grows: the task has more residue
classes and more Fourier frequencies for the circuit to represent (harder),
but frac 0.30 of the larger p^2 grid is more absolute training pairs (easier).
The run settles which dominates on this axis.

Reuses runs/p97_frac0.30_wd1_seed0.* for the p = 97 point (produced by
run_sweep.py); only the p = 113 run is computed here. Resumable: an existing
p = 113 summary is not recomputed.

Run:  python experiments/modulus_scaling.py
"""

import json
from pathlib import Path

from grokking.train import TrainConfig, train

MAX_STEPS = 25_000
RUNS = Path("runs")


def run_p113():
    cfg = TrainConfig(
        p=113, train_frac=0.30, weight_decay=1.0,
        max_steps=MAX_STEPS, eval_every=100, seed=0,
    )
    if not RUNS.joinpath(cfg.run_name() + ".json").exists():
        print(f"=== {cfg.run_name()} on {cfg.device} ===", flush=True)
        train(cfg, out_dir="runs")
    else:
        print(f"skip {cfg.run_name()} (already done)", flush=True)
    return cfg.run_name()


def summarize():
    """Print the p = 97 vs p = 113 comparison from the two run summaries."""
    rows = []
    for name in ("p97_frac0.30_wd1_seed0", "p113_frac0.30_wd1_seed0"):
        with open(RUNS / f"{name}.json") as f:
            s = json.load(f)
        p = s["config"]["p"]
        n_train = int(round(0.30 * p * p))
        rows.append((p, n_train, s["memorize_step"], s["grok_step"]))

    print(f"\n{'p':>5} {'train pairs':>12} {'memorize':>9} {'grok':>7} {'delay x':>8}")
    for p, n, mem, grok in rows:
        delay = grok / mem if mem else float("nan")
        print(f"{p:>5} {n:>12} {mem:>9} {grok:>7} {delay:>7.1f}x")


if __name__ == "__main__":
    run_p113()
    summarize()
