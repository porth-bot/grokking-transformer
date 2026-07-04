"""Run the full grokking sweep. Resumable: completed runs are skipped.

Design of the sweep (one seed, single-variable slices through config space):

- MAIN   frac=0.30, wd=1.0  -- the headline delayed-generalization run.
- WD     wd in {0, 0.1, 1.0} at frac=0.30: is weight decay what converts
         memorization into generalization? wd=0 is the negative control.
- FRAC   frac in {0.25, 0.30, 0.40, 0.60} at wd=1.0: how does the delay
         scale as the train set shrinks toward the critical fraction?

Every run logs its full trajectory to runs/<name>.csv and saves two
checkpoints (memorization point + final) for the Fourier analysis.

Run:  python experiments/run_sweep.py
"""

from grokking.train import TrainConfig, train
from pathlib import Path

MAX_STEPS = 25_000

CONFIGS = [
    # (train_frac, weight_decay) -- main run first so its artifacts exist early
    (0.30, 1.0),
    (0.30, 0.0),
    (0.30, 0.1),
    (0.25, 1.0),
    (0.40, 1.0),
    (0.60, 1.0),
]


def main():
    for frac, wd in CONFIGS:
        cfg = TrainConfig(
            p=97, train_frac=frac, weight_decay=wd,
            max_steps=MAX_STEPS, eval_every=100, seed=0,
        )
        if Path("runs", cfg.run_name() + ".json").exists():
            print(f"skip {cfg.run_name()} (already done)", flush=True)
            continue
        print(f"=== {cfg.run_name()} on {cfg.device} ===", flush=True)
        train(cfg, out_dir="runs")


if __name__ == "__main__":
    main()
