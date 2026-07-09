"""Run the full grokking sweep. Resumable: completed runs are skipped.

Design of the sweep (single-variable slices through config space, each now
repeated over several seeds so the figures can carry error bars):

- WD     wd in {0, 0.1, 1.0} at frac=0.30: is weight decay what converts
         memorization into generalization? wd=0 is the negative control.
- FRAC   frac in {0.25, 0.30, 0.40} at wd=1.0: how does the delay scale as the
         train set shrinks toward the critical fraction? (frac=0.30 is shared
         with the WD slice, so it is the headline delayed-generalization run.)

Each of those five (frac, wd) cells is run for SEEDS = {0..4}; the medians and
inter-quartile bands are what the README tables and figures report. A single
extra single-seed point (frac=0.60, seed 0) is kept for the frac-sweep figure's
"grokking degenerates into ordinary learning" end of the curve.

Every run logs its full trajectory to runs/<name>.csv and saves two
checkpoints (memorization point + final) for the Fourier analysis. Runs are
keyed by name (which includes the seed), so re-running skips finished cells and
only fills gaps -- safe to interrupt and resume.

Run:  python experiments/run_sweep.py
"""

from grokking.train import TrainConfig, train
from pathlib import Path

MAX_STEPS = 25_000
SEEDS = [0, 1, 2, 3, 4]

# (train_frac, weight_decay) cells swept over every seed in SEEDS.
MULTI_SEED_CELLS = [
    (0.30, 1.0),   # main run (shared by the WD and FRAC slices)
    (0.30, 0.0),   # wd negative control
    (0.30, 0.1),
    (0.25, 1.0),
    (0.40, 1.0),
]

# Single-seed points kept for figure context only (not part of the error-bar
# set): frac=0.60 anchors the "delay nearly vanishes" end of the frac curve.
SINGLE_SEED_CELLS = [
    (0.60, 1.0),
]


def jobs():
    """(frac, wd, seed) tuples for the whole sweep, main run first."""
    for frac, wd in MULTI_SEED_CELLS:
        for seed in SEEDS:
            yield frac, wd, seed
    for frac, wd in SINGLE_SEED_CELLS:
        yield frac, wd, 0


def main():
    for frac, wd, seed in jobs():
        cfg = TrainConfig(
            p=97, train_frac=frac, weight_decay=wd,
            max_steps=MAX_STEPS, eval_every=100, seed=seed,
        )
        if Path("runs", cfg.run_name() + ".json").exists():
            print(f"skip {cfg.run_name()} (already done)", flush=True)
            continue
        print(f"=== {cfg.run_name()} on {cfg.device} ===", flush=True)
        train(cfg, out_dir="runs")


if __name__ == "__main__":
    main()
