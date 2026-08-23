"""Does grokking need multiple attention heads?  -- five seeds, not one.

The main runs use 4 heads (d_model 128 -> d_head 32). Modular addition has a
known mechanistic solution -- embed each input on a circle at a few Fourier
frequencies, add the angles in the attention/MLP, read the sum off by
interference (Nanda et al. 2023) -- and nothing in that circuit obviously needs
the representation split across several heads. This ablation asks the question
directly: hold the main config fixed (frac 0.30, wd 1.0, lr 1e-3, one layer)
and vary only ``n_heads`` in {1, 2, 4}. With d_model fixed at 128 the head
width tracks the count (128 / 64 / 32), so this is genuinely "how finely is
attention partitioned", not "how much total width".

**This section used to be one seed** and said so: seed 0 gave grok steps of
400 / 900 / 1900 for 1 / 2 / 4 heads, a clean monotone ordering, with a caveat
noting that the 4-head main run alone spans 1200-1900 across seeds and that a
multi-seed sweep is what would settle it. This is that sweep -- five seeds per
head count, the same five the §1 tables use -- and it is the first half of
issue #4.

What five seeds change, and why the change is the point: a grok step is a
*heavy-tailed* quantity here (the early-stopped runs differ by more than 2x
within one arm), so an ordering read off one seed per arm is a comparison of
three single draws. The arms are compared with an exact rank-sum permutation
test over the C(10,5) = 252 relabelings (``grokking.aggregate.rank_sum_test``),
which is the strongest statement five seeds per arm can support: the smallest
p-value obtainable is 1/252 = 0.004, and complete separation is what it takes
to get it.

Two artifacts, following the split ``swap_equivariance.py`` uses:

- the 15 runs' CSV/JSON logs are committed, so the trajectory figure and the
  timing table replay from the repo with no training and no weights;
- the attention read-outs (appendix statistics, computed from each run's final
  checkpoint) go in a committed CSV, because ``.gitignore`` keeps ``.pt`` files
  out. Regenerating *that* needs the checkpoints, i.e. the training pass.

The read-out is here because the seed-0 story came with a mechanistic guess --
"more heads must coordinate the same computation across a partitioned residual
stream" -- and the appendix's ``|A[=->a] - A[=->b]|`` measures exactly whether
the heads end up doing the same symmetric read at every head count.

Run:  python experiments/head_count.py             (table + figure, no training)
      python experiments/head_count.py --train     (fill in missing seeds)
      python experiments/head_count.py --generate  (re-measure the read-out CSV
                                                    from local checkpoints)
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from grokking.aggregate import (  # noqa: E402
    align_and_aggregate,
    fmt_mean_range,
    fmt_median_range,
    rank_sum_test,
    spread_ratio,
)
from grokking.model import ModelConfig  # noqa: E402
from grokking.seeds import ensure_runs, fill_table  # noqa: E402
from grokking.train import TrainConfig  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import apply_style  # noqa: E402

MAX_STEPS = 25_000   # same budget as the main run
ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
FIGS = ROOT / "figures"
HEADS = (1, 2, 4)
SEEDS = (0, 1, 2, 3, 4)
NAME = "head_count_readouts"
# Checkpoint-derived read-outs (grokking/attention.py), one row per run.
READOUTS = ("attn_entropy", "attn_operand_entropy", "attn_operand_frac",
            "attn_asymmetry")
COLUMNS = ("run", "n_heads", "seed", "final_test_acc") + READOUTS
# A run counts as grokked if it actually generalized. Every cell here does, but
# the tables say so rather than assuming it.
GROKKED_ACC = 0.99


def cfg_for(n_heads: int, seed: int) -> TrainConfig:
    """The main config with only the head count (and seed) varied."""
    return TrainConfig(
        p=97, train_frac=0.30, weight_decay=1.0, lr=1e-3,
        max_steps=MAX_STEPS, eval_every=100, seed=seed,
        model=ModelConfig(n_heads=n_heads),
    )


def run_names() -> dict[int, list[str]]:
    """{n_heads: [run name per seed]}. Pure -- trains and reads nothing.

    The 4-head arm's names are the main sweep runs (no ``_h`` suffix), which is
    what lets this ablation reuse five runs it does not have to train.
    """
    return {h: [cfg_for(h, s).run_name() for s in SEEDS] for h in HEADS}


def train_all(verbose: bool = True) -> dict[int, list[str]]:
    """Train whatever is missing. Resumable: existing runs are skipped."""
    return {h: ensure_runs(lambda s, h=h: cfg_for(h, s), SEEDS,
                           out_dir=RUNS, verbose=verbose)
            for h in HEADS}


def generate(force: bool = False) -> list[dict]:
    """Measure each run's final checkpoint; write ``runs/head_count_readouts.csv``.

    Needs the ``.pt`` files (gitignored), i.e. a local training pass. Only runs
    missing from the CSV are measured, so an interrupted pass resumes -- and
    ``tests/test_head_count.py`` re-measures the one run whose checkpoint *is*
    committed and requires the cached row to match, which is what keeps a
    resumable cache from quietly going stale.
    """
    from grokking.attention import measure_attention
    from grokking.checkpoints import load_model
    from grokking.data import modular_dataset

    tokens, _ = modular_dataset(97, "add")
    names = run_names()
    seed_of = {n: (h, s) for h, ns in names.items()
               for n, s in zip(ns, SEEDS)}

    def measure(name: str) -> dict:
        n_heads, seed = seed_of[name]
        model, summary = load_model(name, which="final", runs_dir=RUNS)
        return {"n_heads": n_heads, "seed": seed,
                "final_test_acc": float(summary["final_test_acc"]),
                **measure_attention(model, tokens)}

    flat = [n for h in HEADS for n in names[h]]
    rows = fill_table(flat, measure, RUNS / f"{NAME}.csv", COLUMNS,
                      force=force)
    with open(RUNS / f"{NAME}.json", "w") as f:
        json.dump({"p": 97, "train_frac": 0.30, "weight_decay": 1.0,
                   "heads": list(HEADS), "seeds": list(SEEDS),
                   "which": "final", "n_runs": len(rows)}, f, indent=2)
    print(f"wrote runs/{NAME}.csv ({len(rows)} runs) and .json")
    return rows


def load_readouts(path: Path | None = None) -> list[dict]:
    """The committed attention read-outs. No checkpoints, no torch."""
    with open(path or RUNS / f"{NAME}.csv") as f:
        rows = []
        for r in csv.DictReader(f):
            row = dict(r)
            row["n_heads"] = int(row["n_heads"])
            row["seed"] = int(row["seed"])
            for k in ("final_test_acc",) + READOUTS:
                row[k] = float(row[k])
            rows.append(row)
    return rows


def _load(name: str) -> tuple[dict, dict]:
    with open(RUNS / f"{name}.csv") as f:
        rows = list(csv.DictReader(f))
    hist = {k: [float(r[k]) for r in rows] for k in rows[0]}
    with open(RUNS / f"{name}.json") as f:
        summary = json.load(f)
    return hist, summary


def collect() -> dict[int, list[tuple[dict, dict]]]:
    """{n_heads: [(history, summary) per seed]} from the committed logs."""
    return {h: [_load(n) for n in names]
            for h, names in run_names().items()}


def pairwise(vals: dict[int, list[float]], indent: str = "  ") -> None:
    """Every arm against every other, by the exact rank-sum test.

    ``P(a < b)`` is ``1 - superiority``: the chance a random seed of the
    smaller-head arm lands below a random seed of the larger one, ties split.
    Printed next to the p-value because with five seeds a *complete* separation
    (P = 1.00) is only worth p = 0.008 -- the floor -- so the two numbers say
    different things and neither alone is the result.
    """
    for i, a in enumerate(sorted(vals)):
        for b in sorted(vals)[i + 1:]:
            r = rank_sum_test(vals[a], vals[b])
            verdict = ("separates" if r["p_two_sided"] <= 0.05
                       else "does NOT separate")
            print(f"{indent}{a} vs {b} heads: P({a} < {b}) = "
                  f"{1 - r['superiority']:.2f}, p = {r['p_two_sided']:.3f} "
                  f"-- {verdict} at 0.05")


def table(runs: dict[int, list[tuple[dict, dict]]]) -> None:
    """Timing across seeds, plus the pairwise tests that say what it supports."""
    print(f"\n{'n_heads':>8} {'d_head':>7} {'grokked':>8} {'memorize':>10} "
          f"{'grok step (median [min-max])':>30} {'spread':>7} {'final test':>11}")
    grok: dict[int, list[float]] = {}
    for h in HEADS:
        summaries = [s for _, s in runs[h]]
        grok[h] = [s["grok_step"] for s in summaries]
        n_grokked = sum(1 for s in summaries
                        if s["final_test_acc"] >= GROKKED_ACC)
        ratio = spread_ratio([g for g in grok[h] if g is not None])
        print(f"{h:>8} {128 // h:>7} {f'{n_grokked}/{len(SEEDS)}':>8} "
              f"{fmt_median_range([s['memorize_step'] for s in summaries]):>10} "
              f"{fmt_median_range(grok[h]):>30} {ratio:>6.1f}x "
              f"{fmt_mean_range([s['final_test_acc'] for s in summaries]):>11}")

    print(f"\nexact rank-sum permutation test on grok step "
          f"(C(10,5) = 252 relabelings, so p >= 0.008 two-sided):")
    pairwise({h: [g for g in grok[h] if g is not None] for h in HEADS})


def readout_table(rows: list[dict]) -> None:
    """The mechanistic read-out the seed-0 story guessed at."""
    print(f"\nattention read-out at the '=' position, final checkpoint "
          f"(mean [min-max] over {len(SEEDS)} seeds):")
    print(f"{'n_heads':>8} {'entropy':>22} {'operand entropy':>22} "
          f"{'operand frac':>22} {'asymmetry':>22}")
    for h in HEADS:
        vals = [r for r in rows if r["n_heads"] == h]
        cells = [fmt_mean_range([v[k] for v in vals], "{:.4f}")
                 for k in READOUTS]
        print(f"{h:>8} " + " ".join(f"{c:>22}" for c in cells))
    print(f"   reference levels: ln 3 = 1.0986 (uniform over a, b, =), "
          f"ln 2 = 0.6931 (symmetric over a, b)")
    print("\nsame test, on each read-out -- does the *end state* differ by "
          "head count?")
    for field in READOUTS:
        print(f"  {field}:")
        pairwise({h: [r[field] for r in rows if r["n_heads"] == h]
                  for h in HEADS}, indent="    ")


def figure_and_table(runs=None, rows=None) -> None:
    """The figure the README shows, from committed logs and the read-out CSV."""
    apply_style()
    runs = collect() if runs is None else runs
    rows = load_readouts() if rows is None else rows
    colors = {1: "C3", 2: "C1", 4: "C0"}

    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.7), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.5, 1.0, 1.0]})

    ax = axes[0]
    for h in HEADS:
        steps_list = [[max(int(s), 1) for s in hist["step"]] for hist, _ in runs[h]]
        vals = [hist["test_acc"] for hist, _ in runs[h]]
        grid, med, lo, hi = align_and_aggregate(steps_list, vals)
        ax.plot(grid, med, lw=1.7, color=colors[h],
                label=f"{h} head{'s' if h > 1 else ''} (d_head {128 // h})")
        ax.fill_between(grid, lo, hi, color=colors[h], alpha=0.18, lw=0)
    ax.set_xscale("log")
    ax.set_xlabel("step (log scale)")
    ax.set_ylabel("test accuracy")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"Median over {len(SEEDS)} seeds, IQR band", loc="left",
                 fontsize=10)

    ax = axes[1]
    for i, h in enumerate(HEADS):
        g = [s["grok_step"] for _, s in runs[h] if s["grok_step"] is not None]
        jitter = np.linspace(-0.13, 0.13, len(g))
        ax.plot(np.full(len(g), i) + jitter, g, "o", color=colors[h], ms=5,
                alpha=0.85)
        ax.plot([i - 0.25, i + 0.25], [np.median(g)] * 2, "-", color="0.2",
                lw=1.8, zorder=3)
    ax.set_xticks(range(len(HEADS)))
    ax.set_xticklabels([f"{h}" for h in HEADS])
    ax.set_xlabel("n_heads")
    ax.set_ylabel("grok step")
    ax.set_yscale("log")
    ax.set_title("Every seed (bar = median)", loc="left", fontsize=10)

    ax = axes[2]
    for i, h in enumerate(HEADS):
        v = [r["attn_asymmetry"] for r in rows if r["n_heads"] == h]
        jitter = np.linspace(-0.13, 0.13, len(v))
        ax.plot(np.full(len(v), i) + jitter, v, "o", color=colors[h], ms=5,
                alpha=0.85)
        ax.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, "-", color="0.2",
                lw=1.8, zorder=3)
    ax.set_xticks(range(len(HEADS)))
    ax.set_xticklabels([f"{h}" for h in HEADS])
    ax.set_xlabel("n_heads")
    ax.set_ylabel(r"$|A_{=\to a} - A_{=\to b}|$")
    ax.set_yscale("log")
    ax.set_title("Operand asymmetry, grokked", loc="left", fontsize=10)

    fig.suptitle("Grokking on (a+b) mod 97 across attention head counts "
                 f"({len(SEEDS)} seeds each)", fontsize=11)
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "head_count.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved figures/head_count.png")


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if "--train" in argv:
        train_all()
    if "--generate" in argv:
        generate(force="--force" in argv)
    runs, rows = collect(), load_readouts()
    table(runs)
    readout_table(rows)
    figure_and_table(runs, rows)


if __name__ == "__main__":
    main()
