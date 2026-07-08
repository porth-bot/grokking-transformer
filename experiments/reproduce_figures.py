"""Regenerate every committed figure from committed logs and checkpoints.

No training. This is the single entry point that turns the files already in the
repo -- the sweep CSV/JSON logs in ``runs/`` and the model checkpoints
(``.pt`` / ``_memorize.pt``) -- back into the figures in ``figures/``:

    CSV logs      -> grokking_main, grokking_loss, wd_sweep, frac_sweep   (plots.py)
    CSV logs      -> lr_sweep                                        (lr_sweep.py)
    checkpoints   -> fourier_spectrum                                     (fourier.py)
    checkpoints   -> embedding_circle                            (embedding_circle.py)
    checkpoints   -> attention_pattern                         (attention_pattern.py)

It first checks that the artifacts each figure depends on are present, so a
missing or renamed file fails loudly here rather than with a cryptic error deep
in a plotting call. Run:  python experiments/reproduce_figures.py
"""

import sys
from pathlib import Path

import attention_pattern
import dropout_control
import embedding_circle
import fourier
import lr_sweep
import plots

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"

# Runs whose CSV/JSON the plot functions read, and whose checkpoints the
# Fourier analysis reads. Keep in sync with plots.py / fourier.py.
CSV_RUNS = [
    "p97_frac0.30_wd1_seed0",
    "p97_frac0.30_wd0_seed0",
    "p97_frac0.30_wd0_seed0_do0.1",
    "p97_frac0.30_wd0.1_seed0",
    "p97_frac0.25_wd1_seed0",
    "p97_frac0.40_wd1_seed0",
    "p97_frac0.60_wd1_seed0",
]
CKPT_RUNS = [(fourier.MAIN, ["", "_memorize"])]
# The lr-sensitivity sweep logs live in runs_lr/ (CSV/JSON only).
LR_RUNS = [lr_sweep.cfg_for(lr).run_name() for lr in lr_sweep.LRS]


def check_artifacts():
    """Return the list of missing files the figures depend on (empty if OK)."""
    missing = []
    for name in CSV_RUNS:
        for ext in (".csv", ".json"):
            if not (RUNS / f"{name}{ext}").exists():
                missing.append(f"runs/{name}{ext}")
    for name in LR_RUNS:
        for ext in (".csv", ".json"):
            if not (lr_sweep.RUNS / f"{name}{ext}").exists():
                missing.append(f"runs_lr/{name}{ext}")
    for name, suffixes in CKPT_RUNS:
        for s in suffixes:
            if not (RUNS / f"{name}{s}.pt").exists():
                missing.append(f"runs/{name}{s}.pt")
    return missing


def main():
    missing = check_artifacts()
    if missing:
        print("ERROR: missing committed artifacts required to reproduce figures:")
        for m in missing:
            print(f"  - {m}")
        print("Run experiments/run_sweep.py to (re)generate them.")
        return 1

    print("Regenerating CSV-based figures ...")
    plots.main_grokking_figure()
    plots.loss_figure()
    plots.wd_sweep_figure()
    plots.frac_sweep_figure()
    lr_sweep.figure_and_table()  # from committed runs_lr/ CSVs, no retraining
    dropout_control.figure_and_table("p97_frac0.30_wd0_seed0_do0.1")

    print("Regenerating checkpoint-based figures ...")
    fourier.main()
    embedding_circle.main()
    attention_pattern.main()

    print("All figures reproduced into figures/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
