"""The attention read-out along the trajectory, not at two points.

The appendix (``attention_pattern.py``) compares the "=" query's attention at
the memorization checkpoint and at the end. That tells you the endpoints and
nothing about the path -- in particular it cannot say whether the operand read
symmetrizes *before* the test-accuracy jump (like the Fourier progress measures
of §10, which form gradually) or *at* it. This logs the same read-out at every
eval step, the same way ``progress_measures.py`` does, and answers that.

What comes out is four regimes, and the middle two are the ones two
checkpoints cannot show:

1. **Initialization** is already at the algorithmic symmetry -- entropy ln 3,
   operand entropy exactly ln 2, two thirds of the row on the operands. That is
   just what a near-uniform softmax gives, but it sets the baseline everything
   after is measured against.
2. **Memorization breaks it.** The row concentrates hard on the operands
   (0.67 -> 0.997) and goes lopsided: asymmetry 0.004 -> 0.189, operand entropy
   0.693 -> 0.658. The model does not start non-commutative and become
   commutative; it starts symmetric, *destroys* the symmetry to memorize, and
   then recovers it.
3. **Grokking restores it**, to ln 2 to five decimals on every head.
4. **After the jump**, the constant self-attention channel of
   ``grokking/attention.py`` opens and then fluctuates (operand weight wandering
   between 0.83 and 0.98 from eval to eval) while the operand entropy stays
   pinned. The full-row entropy inherits that noise; the operand entropy does
   not, which is the clearest demonstration that they are measuring different
   things.

Two things this does NOT show, stated because the temptation is to claim them.
The symmetry is restored at step 2300, *after* test accuracy passes 0.5 (1500)
and after the grok step (1900) -- so on this run the attention read-out is a
lagging indicator, not an early-warning signal like §10's restricted loss. And
this is seed 0 only, like every other mechanistic read-out in the repo that has
not yet been through issue #4.

Method, and why it is one honest run
------------------------------------
Train the main config (frac 0.30, wd 1.0, seed 0) once on CPU with the
``on_eval`` hook snapshotting the ~0.2M-parameter model at every eval, then
replay the attention read-out over the snapshots. No retraining in the replay;
the training side-artifacts go to a throwaway directory so this can never
clobber the committed main run that every other figure depends on. Only the
trajectory CSV/JSON are committed, and the figure regenerates from those alone.

Note the trajectory ends where patience stops it, around 4500 steps, well
short of the 25000-step committed main run. So its last row is not the
committed final checkpoint: 0.93 nats and 0.91 operand weight here against 1.02
and 0.84 there. Both sit inside the post-grok fluctuation band above, which is
the point -- there is no single "final" value of these two, and quoting one
without the band would be misleading.

    python experiments/attention_entropy.py            # train + figure (~3 min CPU)
    python experiments/attention_entropy.py --figure   # figure from committed CSV
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from grokking.attention import SYMMETRIC_2, UNIFORM_3, measure_attention
from grokking.data import modular_addition_dataset
from grokking.model import Transformer
from grokking.train import TrainConfig, train

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
NAME = "attention_p97_frac0.30_wd1_seed0"

COLUMNS = [
    "step", "train_loss", "test_loss", "train_acc", "test_acc", "weight_norm",
    "attn_entropy", "attn_operand_entropy", "attn_operand_frac", "attn_asymmetry",
]


def generate(out_dir: Path = RUNS) -> None:
    """Train the main config with attention instrumentation; write the CSV/JSON."""
    cfg = TrainConfig(
        p=97, train_frac=0.30, weight_decay=1.0, lr=1e-3,
        max_steps=8000, eval_every=100, patience=25, seed=0, device="cpu",
    )
    snapshots: list[dict] = []

    def snapshot(step: int, model: Transformer) -> None:
        snapshots.append(
            {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        )

    print("Training main config with attention instrumentation (CPU) ...", flush=True)
    history, summary = train(cfg, out_dir="runs_attention", on_eval=snapshot)

    tokens, _ = modular_addition_dataset(cfg.p)
    model = Transformer(cfg.model)
    rows = []
    for row, snap in zip(history, snapshots):
        model.load_state_dict(snap)
        model.eval()
        rows.append({**row, **measure_attention(model, tokens)})

    out_dir.mkdir(exist_ok=True)
    with open(out_dir / f"{NAME}.csv", "w") as f:
        f.write(",".join(COLUMNS) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in COLUMNS) + "\n")
    meta = {
        "memorize_step": summary["memorize_step"],
        "grok_step": summary["grok_step"],
        "steps_run": summary["steps_run"],
        "n_heads": cfg.model.n_heads,
        "uniform_entropy": UNIFORM_3,
        "symmetric_entropy": SYMMETRIC_2,
    }
    with open(out_dir / f"{NAME}.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote runs/{NAME}.csv ({len(rows)} evals) and .json", flush=True)


def _read_csv(path: Path) -> dict[str, list[float]]:
    with open(path) as f:
        header = f.readline().strip().split(",")
        cols: dict[str, list[float]] = {h: [] for h in header}
        for line in f:
            for h, v in zip(header, line.strip().split(",")):
                cols[h].append(float(v))
    return cols


def summarize(csv_path: Path = RUNS / f"{NAME}.csv") -> dict[str, float]:
    """The numbers the README quotes, read off the committed trajectory.

    Four points, because the trajectory has four regimes and quoting only the
    endpoints would miss the middle two: initialization, the memorization
    plateau, the grokking step, and the end.

    "When is the symmetry restored" needs care. The obvious spelling -- first
    step where the operand entropy is within 1% of ln 2 -- returns step 0,
    because a randomly initialized model is *already* symmetric on average. The
    question only means anything after the memorization dip, so the search
    starts from the step of peak asymmetry. Getting that wrong is how one
    reports "the symmetry was there from the beginning" about a model that
    spent 1500 steps being lopsided.
    """
    d = _read_csv(csv_path)
    with open(csv_path.with_suffix(".json")) as f:
        meta = json.load(f)
    steps = d["step"]

    def at(step: float) -> int:
        return min(range(len(steps)), key=lambda i: abs(steps[i] - step))

    peak = max(range(len(steps)), key=lambda i: d["attn_asymmetry"][i])
    restored = [steps[i] for i in range(peak, len(steps))
                if d["attn_operand_entropy"][i] >= 0.999 * SYMMETRIC_2]
    half_acc = [s for s, a in zip(steps, d["test_acc"]) if a >= 0.5]
    post = [f for s, f in zip(steps, d["attn_operand_frac"]) if s > meta["grok_step"]]
    mem, grok = at(meta["memorize_step"]), at(meta["grok_step"])

    return {
        "memorize_step": meta["memorize_step"],
        "grok_step": meta["grok_step"],
        "step_test_acc_half": half_acc[0] if half_acc else float("nan"),
        "entropy_at_init": d["attn_entropy"][0],
        "entropy_at_memorize": d["attn_entropy"][mem],
        "entropy_at_grok": d["attn_entropy"][grok],
        "entropy_final": d["attn_entropy"][-1],
        "operand_entropy_at_init": d["attn_operand_entropy"][0],
        "operand_entropy_min": min(d["attn_operand_entropy"]),
        "operand_entropy_final": d["attn_operand_entropy"][-1],
        "step_symmetry_restored": restored[0] if restored else float("nan"),
        "operand_frac_at_init": d["attn_operand_frac"][0],
        "operand_frac_at_memorize": d["attn_operand_frac"][mem],
        "operand_frac_postgrok_min": min(post) if post else float("nan"),
        "operand_frac_postgrok_max": max(post) if post else float("nan"),
        "asymmetry_at_init": d["attn_asymmetry"][0],
        "asymmetry_peak": d["attn_asymmetry"][peak],
        "asymmetry_final": d["attn_asymmetry"][-1],
        "steps_run": steps[-1],
    }


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

    # Left: the two entropies against their two reference levels. The gap
    # between them IS the self-attention channel, so plotting both on the same
    # axes makes the decomposition visible rather than asserted.
    #
    # symlog rather than log on x: step 0 is not a throwaway point here, it is
    # the baseline the whole four-regime reading is measured against (a random
    # softmax is already at ln 3 / ln 2), and a log axis would drop it.
    ax = axes[0]
    # Labels go above their line at the right edge, where both curves are
    # clear of it (the renormalized operand curve sits exactly ON the symmetric
    # line there, so anything below it would be unreadable).
    for level, label in ((UNIFORM_3, "uniform over {a, b, =}"),
                         (SYMMETRIC_2, "symmetric on {a, b}")):
        ax.axhline(level, color="0.6", ls=":", lw=1)
        ax.text(steps[-1], level + 0.009, label, fontsize=7.5, color="0.35",
                va="bottom", ha="right")
    ax.plot(steps, d["attn_entropy"], color="C0", lw=1.8,
            label='full "=" row, over {a, b, =}')
    ax.plot(steps, d["attn_operand_entropy"], color="C3", lw=1.8,
            label="operands only, renormalized")
    ax.axvline(grok, color="C2", ls="--", lw=1, alpha=0.8)
    ax.text(grok, 0.615, "  grok", color="C2", va="bottom", fontsize=8)
    ax.set_xscale("symlog", linthresh=100)
    ax.set_xlim(0, steps[-1])
    ax.set_ylim(0.61, 1.15)
    ax.set_xlabel("step (linear below 100, log above)")
    ax.set_ylabel("attention entropy (nats)")
    ax.set_title("Memorization breaks the symmetry the init had;\n"
                 "grokking puts it back, exactly", loc="left", fontsize=9)
    ax.legend(fontsize=7.5, loc="center left")

    # Right: what the gap is made of, and the symmetry statistic that motivated
    # the whole read-out, against test accuracy.
    ax = axes[1]
    ax.plot(steps, d["attn_operand_frac"], color="C0", lw=1.8,
            label="operand weight (1 - self-attention)")
    ax.plot(steps, d["attn_asymmetry"], color="C3", lw=1.8,
            label=r"per-head $|A_{=\to a} - A_{=\to b}|$")
    ax.axvline(grok, color="C2", ls="--", lw=1, alpha=0.8)
    ax.text(grok, 0.02, "  grok", color="C2", va="bottom", fontsize=8)
    ax.set_xscale("symlog", linthresh=100)
    ax.set_xlim(0, steps[-1])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("step (linear below 100, log above)")
    ax.set_ylabel("attention weight")
    ax2 = ax.twinx()
    ax2.plot(steps, d["test_acc"], color="k", lw=1.2, alpha=0.8)
    ax2.set_ylabel("test accuracy")
    ax2.set_ylim(0, 1.02)
    ax2.spines["top"].set_visible(False)
    ax.set_title("Asymmetry peaks while memorizing, collapses at\n"
                 "the jump; then the self-attention channel opens",
                 loc="left", fontsize=9)
    ax.legend(fontsize=7.5, loc="center left")

    fig.suptitle('Attention entropy along the trajectory: the "=" read is '
                 "symmetric, then is not, then is again", y=1.07)
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures" / "attention_entropy.png", bbox_inches="tight")
    print("saved figures/attention_entropy.png")


def main() -> None:
    if "--figure" not in sys.argv:
        generate()
    figure()
    for k, v in summarize().items():
        print(f"{k:38s} {v:.5g}")


if __name__ == "__main__":
    main()
