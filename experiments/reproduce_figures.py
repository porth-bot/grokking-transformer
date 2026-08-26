"""Regenerate every committed figure from committed logs and checkpoints.

No training. This is the single entry point that turns the files already in the
repo -- the sweep CSV/JSON logs in ``runs/`` and the model checkpoints
(``.pt`` / ``_memorize.pt``) -- back into the figures in ``figures/``:

    CSV logs      -> grokking_main, grokking_loss, wd_sweep, frac_sweep   (plots.py)
    CSV logs      -> lr_sweep                                        (lr_sweep.py)
    CSV logs      -> dropout_control                          (dropout_control.py)
    CSV logs      -> wd_scope                                        (wd_scope.py)
    CSV logs      -> wd_frac_surface                          (wd_frac_surface.py)
    CSV logs (+ read-out CSV) -> head_count                        (head_count.py)
    read-out CSV  -> mechanistic_seeds                    (mechanistic_seeds.py)
    CSV logs      -> progress_measures                      (progress_measures.py)
    CSV logs      -> attention_entropy                      (attention_entropy.py)
    CSV logs      -> operations                                    (operations.py)
    CSV logs      -> swap_equivariance                      (swap_equivariance.py)
    checkpoints   -> fourier_spectrum                                     (fourier.py)
    checkpoints   -> embedding_circle                            (embedding_circle.py)
    checkpoints   -> attention_pattern                         (attention_pattern.py)
    checkpoints   -> logit_attribution                       (logit_attribution.py)

That list is ``FIGURES`` below, and a test asserts it matches the contents of
``figures/`` exactly -- so a figure added without a replay path here fails the
suite instead of silently becoming unreproducible (which is how wd_scope.png
went missing for eleven days).

It first checks that the artifacts each figure depends on are present, so a
missing or renamed file fails loudly here rather than with a cryptic error deep
in a plotting call. Run:  python experiments/reproduce_figures.py
"""

import sys
from pathlib import Path

import attention_entropy
import attention_pattern
import dropout_control
import embedding_circle
import head_count
import fourier
import logit_attribution
import lr_sweep
import mechanistic_seeds
import operations
import plots
import progress_measures
import run_sweep
import swap_equivariance
import wd_frac_surface
import wd_scope

from grokking.train import TrainConfig

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
FIGS = ROOT / "figures"

# Every figure this script regenerates, i.e. every figure the repo ships.
# tests/test_reproduce_figures.py asserts this equals the actual contents of
# figures/, in both directions.
FIGURES = (
    "grokking_main.png",       # plots.main_grokking_figure
    "grokking_loss.png",       # plots.loss_figure
    "wd_sweep.png",            # plots.wd_sweep_figure
    "frac_sweep.png",          # plots.frac_sweep_figure
    "lr_sweep.png",            # lr_sweep.figure_and_table
    "dropout_control.png",     # dropout_control.figure_and_table
    "wd_scope.png",            # wd_scope.figure_and_table
    "wd_frac_surface.png",     # wd_frac_surface.figure
    "head_count.png",          # head_count.figure_and_table
    "mechanistic_seeds.png",   # mechanistic_seeds.figure
    "progress_measures.png",   # progress_measures.figure
    "attention_entropy.png",   # attention_entropy.figure
    "operations.png",          # operations.figure_and_table
    "swap_equivariance.png",   # swap_equivariance.figure
    "fourier_spectrum.png",    # fourier.main
    "embedding_circle.png",    # embedding_circle.main
    "attention_pattern.png",   # attention_pattern.main
    "logit_attribution.png",   # logit_attribution.main
)

# Runs whose CSV/JSON the plot functions read, and whose checkpoints the
# Fourier analysis reads. The wd/frac error-bar figures consume every seed of
# every sweep cell, so derive the list from run_sweep itself (single source of
# truth) rather than restating it; the dropout control (§6) is the one extra.
CSV_RUNS = [
    TrainConfig(p=97, train_frac=f, weight_decay=w, seed=s).run_name()
    for f, w, s in run_sweep.jobs()
] + [
    "p97_frac0.30_wd0_seed0_do0.1",
] + [
    # head-count ablation (§9): 1/2/4 heads x five seeds. The 4-head arm is the
    # main sweep runs already listed above; asking head_count for the names
    # keeps this list from drifting when the seed list changes.
    name for names in head_count.run_names().values() for name in names
] + [
    # weight-decay scope ablation (§7): embeddings-only and non-embeddings-only;
    # the all-parameters arm is the main run already listed above.
    wd_scope.cfg_for(scope).run_name() for scope in wd_scope.SCOPES
] + [
    # (wd, frac) interaction surface (§13): the twelve-cell grid at three seeds.
    # Five of its cells are the sweep runs already listed above, so those names
    # repeat here and are simply checked twice; the other seven are new.
    name for names in wd_frac_surface.run_names().values() for name in names
] + [
    # operations comparison (§11): sub/mul at wd {1.0, 0.1} over three seeds;
    # add reuses the main-run and wd0.1 sweep CSVs already listed above.
    operations.cfg_for(op, wd, seed).run_name()
    for wd in operations.WEIGHT_DECAYS
    for op in ("sub", "mul")
    for seed in operations.SEEDS
]
CKPT_RUNS = [(fourier.MAIN, ["", "_memorize"])]
# The lr-sensitivity sweep logs live in runs_lr/ (CSV/JSON only).
LR_RUNS = [lr_sweep.cfg_for(lr).run_name() for lr in lr_sweep.LRS]
# Four read-outs are single committed CSVs (+ JSON meta) rather than sweeps.
# Two are trajectories from rerunning the main config with its own
# instrumentation -- the progress measures (§10) and the attention read-out
# (appendix). The third is the swap-equivariance table (§11): one row per
# checkpoint of the operations sweep, committed because those checkpoints are
# not (.gitignore keeps .pt files out) and 3 KB of read-outs reproduces the
# figure where 12 MB of weights would be needed to recompute them. The fourth
# is the head-count read-out table (§9), committed for the same reason: 15
# runs' attention statistics, from 15 checkpoints that are not in the repo.
# The fifth is the mechanistic seed sweep (Sec. 12): every read-out of ten
# checkpoints, nine of which are not in the repo either.
TRAJECTORY_CSVS = [progress_measures.NAME, attention_entropy.NAME,
                   swap_equivariance.NAME, head_count.NAME,
                   mechanistic_seeds.NAME]


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
    for name in TRAJECTORY_CSVS:
        for ext in (".csv", ".json"):
            if not (RUNS / f"{name}{ext}").exists():
                missing.append(f"runs/{name}{ext}")
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
    head_count.figure_and_table()  # §9, from the 15 run logs + the read-out CSV
    mechanistic_seeds.figure()  # §12, from the committed read-out CSV
    wd_scope.figure_and_table(*(wd_scope.cfg_for(s).run_name()
                                for s in wd_scope.SCOPES))  # §7
    wd_frac_surface.figure()  # §13, the (wd, frac) grid from its run logs
    progress_measures.figure()  # §10, from the committed progress trajectory CSV
    attention_entropy.figure()  # appendix, from the committed attention trajectory
    operations.figure_and_table()  # §11, sub/mul vs add from committed CSVs
    swap_equivariance.figure(swap_equivariance.collect())  # §11 control

    print("Regenerating checkpoint-based figures ...")
    fourier.main()
    embedding_circle.main()
    attention_pattern.main()
    logit_attribution.main()

    print("All figures reproduced into figures/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
