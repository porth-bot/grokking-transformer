"""Progress measures along the trajectory: the circuit forms before the jump.

Day 19. ``fourier.py`` (§5) and ``logit_attribution.py`` (§8) show the *grokked*
model is sparse in frequency space, from two committed checkpoints. This upgrades
that to a **training trajectory**: rerun the main config (frac 0.30, wd 1.0, seed
0) with per-eval instrumentation and log Nanda et al.'s progress measures at every
eval step -- watching the generalizing circuit form *gradually*, well before the
test-accuracy jump.

How it stays a single honest run
--------------------------------
The restricted/excluded losses need a *fixed* reference frequency set -- the
grokked model's key ``a+b`` frequencies -- applied across the whole trajectory,
so we track the SAME circuit forming rather than a moving target. But those
frequencies are only known at the end. So we train once, snapshotting the (tiny,
~0.2M-param) model at each eval via the ``on_eval`` hook, read the key
frequencies off the FINAL snapshot, then replay every snapshot against that fixed
set. One deterministic training run; the replay adds no training.

The trajectory CSV is committed, so the figure reproduces from committed files
with no retraining (like every other figure). Run:

    python experiments/progress_measures.py            # train + figure
    python experiments/progress_measures.py --figure   # figure from committed CSV
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from grokking.data import modular_addition_dataset
from grokking.model import Transformer
from grokking.progress import diagonal_key_frequencies, logit_tensor, measure_all
from grokking.train import TrainConfig, train

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
NAME = "progress_p97_frac0.30_wd1_seed0"
K_EMB = 5      # embedding frequencies defining sparsity
K_KEY = 5      # logit a+b frequencies held fixed for restricted/excluded loss

COLUMNS = [
    "step", "train_loss", "test_loss", "train_acc", "test_acc",
    "weight_norm", "emb_top_frac", "restricted_loss", "excluded_loss",
]


def generate(out_dir: Path = RUNS) -> None:
    """Train the main config with instrumentation; write the trajectory CSV/JSON.

    Runs on CPU for a deterministic, committable trajectory (the mechanistic
    story is device-independent; CPU keeps it reproducible in CI).
    """
    cfg = TrainConfig(
        p=97, train_frac=0.30, weight_decay=1.0, lr=1e-3,
        max_steps=8000, eval_every=100, patience=25, seed=0, device="cpu",
    )
    p = cfg.p

    snapshots: list[dict] = []

    def snapshot(step: int, model: Transformer) -> None:
        snapshots.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

    # train() writes <run_name>.{csv,json,pt} using the config's run_name, which
    # for the main config is exactly "p97_frac0.30_wd1_seed0" -- the committed
    # main run every other figure depends on. Send those side-artifacts to a
    # throwaway dir (gitignored) so this instrumented run can never clobber them;
    # the only committed output is the progress CSV/JSON written below.
    print("Training main config with progress-measure instrumentation (CPU) ...", flush=True)
    history, summary = train(cfg, out_dir="runs_progress", on_eval=snapshot)

    tokens, targets = modular_addition_dataset(p)
    model = Transformer(cfg.model)

    # Key frequencies: the top a+b diagonal frequencies of the FINAL model, held
    # fixed for the whole trajectory.
    model.load_state_dict(snapshots[-1])
    model.eval()
    key_ks = diagonal_key_frequencies(logit_tensor(model, tokens, p), p, K_KEY)
    print(f"key a+b frequencies (final model, top {K_KEY}): {sorted(key_ks)}", flush=True)

    rows = []
    for row, snap in zip(history, snapshots):
        model.load_state_dict(snap)
        model.eval()
        m = measure_all(model, tokens, targets, p, key_ks, k_emb=K_EMB)
        rows.append({**row, **m})

    out_dir.mkdir(exist_ok=True)
    with open(out_dir / f"{NAME}.csv", "w") as f:
        f.write(",".join(COLUMNS) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in COLUMNS) + "\n")
    meta = {
        "key_freqs": sorted(key_ks),
        "k_emb": K_EMB,
        "memorize_step": summary["memorize_step"],
        "grok_step": summary["grok_step"],
        "steps_run": summary["steps_run"],
    }
    with open(out_dir / f"{NAME}.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote runs/{NAME}.csv ({len(rows)} evals) and .json", flush=True)


def overtake_step(steps, restricted, test_loss, after=0):
    """First step after ``after`` where the full model's test loss goes below
    the restricted loss, or ``None`` if it never does.

    The restricted loss starts well below the test loss -- that is the whole
    point of the measure -- but the test loss is falling too, and on this run it
    catches up *before* the accuracy jumps. The left panel's title used to
    assert the lead held "before the jump"; it holds through step 1,000 and not
    after, so the title now reports where the two cross instead of describing a
    shape the data does not have.
    """
    for step, r, t in zip(steps, restricted, test_loss):
        if step > after and t < r:
            return int(step)
    return None


def accuracy_jump_step(steps, acc, memorize_step):
    """First eval at which test accuracy clears the midpoint of its own rise.

    Not the grok step, which is the 99% crossing and lands *after* the
    accuracy has already moved: on the committed run the grok step is 1,900
    while accuracy goes 0.53 -> 0.86 at 1,600. Anything asking "did this
    read-out move before the accuracy did" has to compare against the latter,
    or it credits the read-out with rises that happened after the jump.
    """
    at_mem = next(a for s, a in zip(steps, acc) if s == memorize_step)
    final = acc[-1]
    midpoint = 0.5 * (at_mem + final)
    return next(int(s) for s, a in zip(steps, acc)
                if s > memorize_step and a > midpoint)


def pre_jump_rise_share(steps, values, jump, memorize_step):
    """Fraction of a read-out's rise since memorization that precedes ``jump``.

    "Begins to climb before the accuracy step" is true of almost any rising
    quantity and says nothing about how much of the climb is pre-jump. Measured
    from the memorization value to the run's maximum, the embedding top-5 share
    completes about a tenth of its rise before the accuracy jumps -- a much
    weaker claim than the panel used to make, and the one the curve supports.
    """
    start = next(v for s, v in zip(steps, values) if s == memorize_step)
    at_jump = max(v for s, v in zip(steps, values) if s < jump)
    total = max(values)
    if total == start:
        return 0.0
    return (at_jump - start) / (total - start)


def _read_csv(path: Path) -> dict[str, list[float]]:
    with open(path) as f:
        header = f.readline().strip().split(",")
        cols: dict[str, list[float]] = {h: [] for h in header}
        for line in f:
            for h, v in zip(header, line.strip().split(",")):
                cols[h].append(float(v))
    return cols


def figure(csv_path: Path = RUNS / f"{NAME}.csv") -> None:
    """Render the trajectory figure from the committed CSV (no retraining)."""
    from _style import apply_style  # selects the Agg backend on import

    import matplotlib.pyplot as plt

    apply_style()

    d = _read_csv(csv_path)
    with open(csv_path.with_suffix(".json")) as f:
        meta = json.load(f)
    grok = meta["grok_step"]
    steps = d["step"]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4), constrained_layout=True)

    # Left: the mechanism losses vs the true test loss. At memorization the
    # RESTRICTED loss (only the 5 key a+b frequencies) is already ~0.9 nats
    # BELOW the full model's test loss -- the generalizing circuit is the better
    # predictor there. It does not stay ahead: both fall through the delay and
    # the full model overtakes it well before the accuracy jumps, which is what
    # the title reports. Post-grok the EXCLUDED loss (key freqs removed) holds a
    # median near 1 while the test loss collapses to ~1e-2: the model depends on
    # exactly those frequencies.
    ax = axes[0]
    ax.plot(steps, d["test_loss"], color="k", lw=1.5, label="test loss (full model)")
    ax.plot(steps, d["restricted_loss"], color="C0", lw=1.8,
            label="restricted: 5 key a+b freqs only")
    ax.plot(steps, d["excluded_loss"], color="C3", lw=1.8,
            label="excluded: those 5 freqs removed")
    ax.axvline(grok, color="C2", ls="--", lw=1, alpha=0.8)
    ax.text(grok, 0.13, "  grok", color="C2", va="bottom", fontsize=8)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylim(0.1, 8)
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss (all pairs)")
    over = overtake_step(steps, d["restricted_loss"], d["test_loss"],
                         after=meta["memorize_step"])
    ax.set_title("The a+b-circuit-only loss starts below the test loss"
                 + (f"\nand is overtaken at step {over:,}, before the jump"
                    if over is not None else "\nand keeps the lead throughout"),
                 loc="left", fontsize=9)
    ax.legend(fontsize=7.5, loc="lower left")

    # Right: embedding sparsity against test accuracy. It does start climbing
    # before the accuracy jumps, but only 13% of its rise is pre-jump, which
    # is why the title reports the share instead of the direction.
    ax = axes[1]
    ax.plot(steps, d["emb_top_frac"], color="C0", lw=1.8,
            label=f"embedding top-{meta['k_emb']} energy")
    ax.axvline(grok, color="C2", ls="--", lw=1, alpha=0.8)
    ax.text(grok, 0.02, "  grok", color="C2", va="bottom", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("top-5 embedding energy fraction", color="C0")
    ax.set_ylim(0, 1.02)
    ax.tick_params(axis="y", labelcolor="C0")
    ax2 = ax.twinx()
    ax2.plot(steps, d["test_acc"], color="k", lw=1.2, alpha=0.8, label="test accuracy")
    ax2.set_ylabel("test accuracy")
    ax2.set_ylim(0, 1.02)
    ax2.spines["top"].set_visible(False)
    jump = accuracy_jump_step(steps, d["test_acc"], meta["memorize_step"])
    share = pre_jump_rise_share(steps, d["emb_top_frac"], jump,
                                meta["memorize_step"])
    ax.set_title("Embedding structure forms gradually;\n"
                 f"{share:.0%} of the rise comes before the accuracy jump",
                 loc="left", fontsize=9)

    fig.suptitle("Progress measures: the generalizing circuit forms gradually, "
                 "then the accuracy jumps", y=1.07)
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures" / "progress_measures.png", bbox_inches="tight")
    print("saved figures/progress_measures.png")


def main() -> None:
    if "--figure" not in sys.argv:
        generate()
    figure()


if __name__ == "__main__":
    main()
