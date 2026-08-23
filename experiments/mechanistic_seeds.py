"""Every mechanistic read-out, five seeds instead of one (issue #4, part 2).

Part 1 (``head_count.py``, Sec. 9) put five seeds behind a *timing* claim. This
is the other half of the issue and the harder half: Sec. 5, Sec. 8 and the
appendix all read structure out of two checkpoints -- the memorization point
and the end of training -- of **one run**, seed 0, because seed 0 is the run
whose weights are committed. Nine numbers in the README come from that one
pair of files.

The question is not whether the story is true. It is whether the *numbers* are
typical, and the two ways they can fail to be are different. A claim can be
qualitative and survive averaging with a different level (Sec. 9's grok steps
moved 400/900/1900 -> 300/700/1300 and the ordering held). Or a contrast can
turn out to rest on the gap between one lucky checkpoint and one unlucky one,
in which case the direction may survive while the *size* does not -- and a
reader who takes "0.189 -> 0.00017" as the effect size has been misled even
though every digit is real.

So each read-out is measured on all five seeds at both checkpoints, and the two
arms are compared with the same exact rank-sum permutation test Sec. 9 uses
(``grokking.aggregate.rank_sum_test``, C(10,5) = 252 relabelings, so p >= 0.008
two-sided). What that instrument gives here is the ability to say which
contrasts are *complete separations* -- no memorization checkpoint on the wrong
side of any final one -- and which merely have a mean difference.

Artifacts, following the split ``swap_equivariance.py`` and ``head_count.py``
use: ``.gitignore`` keeps checkpoints out of the repo except seed 0's, so the
read-outs go in a committed CSV and the figure replays from that CSV alone.
Regenerating the CSV needs the other four seeds' ``.pt`` files, i.e. a local
training pass (``experiments/run_sweep.py``); ``tests/test_mechanistic_seeds.py``
re-measures seed 0's two rows from the committed weights and requires the cache
to agree, which is what stops it outliving the code that produced it.

One definitional footnote, because it is the sort of thing that silently
becomes a discrepancy. The embedding ring is measured in the plane of *each
checkpoint's own* dominant frequency, while ``embedding_circle.py``'s figure
projects both checkpoints onto the plane of the **final** model's dominant
frequency (which is what makes the two panels comparable by eye). The figure's
memorization radial CV is therefore 0.412 where this table's is 0.399: asking
each checkpoint for its best plane is the conservative choice here, since it can
only flatter the memorizing model, and a test pins that ordering rather than
leaving the two numbers to look like a contradiction.

Run:  python experiments/mechanistic_seeds.py             (table + figure)
      python experiments/mechanistic_seeds.py --generate  (re-measure the CSV
                                                           from local .pt files)
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

from grokking.aggregate import rank_sum_test, stats  # noqa: E402
from grokking.mechanistic import RESTRICT_MS, random_ring_baseline  # noqa: E402
from grokking.seeds import fill_table  # noqa: E402
from grokking.train import TrainConfig  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
FIGS = ROOT / "figures"
NAME = "mechanistic_readouts"

SEEDS = (0, 1, 2, 3, 4)
WHICH = ("memorize", "final")
MAX_STEPS = 25_000          # the main sweep's budget; these are its runs
N_FREQS = 48                # (p-1)/2 non-DC frequencies at p = 97

# Read-outs, with the sign of the change the README reports from memorization
# to grokking. The direction is not decoration: it is what lets the table say
# whether seed 0 sits at the flattering or the sober end of its arm, which is
# the whole question this experiment exists to answer. (For the operand weight
# the "flattering" end is the *low* one only because the published contrast is
# a fall, 99.7% -> 83.7%; nothing about the circuit says less operand weight is
# better, and the appendix explains the fall as a constant-bias channel
# opening.)
READOUTS: tuple[tuple[str, str, int], ...] = (
    ("emb_top5_energy",     "top-5 embedding energy share",      +1),
    ("emb_freqs_90",        "embedding freqs for 90% energy",     -1),
    ("emb_radial_cv",       "embedding ring radial CV",           -1),
    ("emb_var_in_plane",    "variance in the ring plane",         +1),
    ("logit_diag_energy",   "logit energy on the a+b diagonal",   +1),
    ("logit_diag_freqs_90", "diagonal freqs for 90% energy",      -1),
    ("logit_freqs_for_99",  "freqs to rebuild 99% test acc",      -1),
    ("attn_entropy",        'full "=" row entropy (max ln 3)',    +1),
    ("attn_operand_frac",   "attention weight on the operands",   -1),
    ("attn_operand_entropy", "operand entropy (max ln 2)",        +1),
    ("attn_asymmetry",      "per-head |A[=->a] - A[=->b]|",       -1),
    ("eq_attn_defect",      "attention equivariance defect",      -1),
    ("eq_logit_swap",       "logit swap defect",                  -1),
)
# Extra columns that are per-run facts or controls rather than comparable
# read-outs: the ring's frequency differs per seed, the anti-equivariance defect
# is the symmetry addition must *not* have, and the shuffle baseline is the
# measured level of "no symmetry at all" that the defects are read against.
EXTRA = ("emb_dominant_k", "emb_diag_overlap", "logit_acc_full",
         "eq_logit_anti", "eq_shuffle_baseline")
COLUMNS = (("key", "run", "seed", "which")
           + tuple(k for k, _, _ in READOUTS) + EXTRA
           + tuple(f"logit_acc_m{m}" for m in RESTRICT_MS))

# Panels of the figure, in reading order: embeddings, then logits and
# attention, then the two symmetry defects. ``log`` for the defects, which span
# three decades. The ninth cell of the 3x3 grid is the restricted-accuracy
# curve, so this tuple has eight entries and not nine.
PANELS: tuple[tuple[str, str, bool], ...] = (
    ("emb_top5_energy",     "top-5 embedding\nenergy share",         False),
    ("emb_freqs_90",        "embedding freqs\nfor 90% energy",       False),
    ("emb_radial_cv",       "embedding ring\nradial CV",             False),
    ("logit_diag_energy",   "logit energy on\nthe $a{+}b$ diagonal", False),
    ("logit_diag_freqs_90", "diagonal freqs\nfor 90% energy",        False),
    ("attn_operand_frac",   'attention weight\non the operands',     False),
    ("eq_attn_defect",      "attention equivariance\ndefect",        True),
    ("eq_logit_swap",       "logit swap\ndefect",                    True),
)


def cfg_for(seed: int) -> TrainConfig:
    """The main run (Sec. 1's headline config) at one seed."""
    return TrainConfig(p=97, train_frac=0.30, weight_decay=1.0,
                       max_steps=MAX_STEPS, eval_every=100, seed=seed)


def run_names() -> list[str]:
    """The five run names, in seed order. Pure -- reads nothing."""
    return [cfg_for(s).run_name() for s in SEEDS]


def keys() -> list[str]:
    """``<run>@<checkpoint>`` for all ten rows, memorization before final."""
    return [f"{n}@{w}" for n in run_names() for w in WHICH]


def generate(force: bool = False) -> list[dict]:
    """Measure all ten checkpoints; write ``runs/mechanistic_readouts.csv``.

    Needs the four non-seed-0 ``.pt`` files, which are gitignored; run
    ``experiments/run_sweep.py`` first if they are missing. Resumable, since
    ``fill_table`` only measures keys the CSV does not already carry.
    """
    from grokking.checkpoints import load_model
    from grokking.mechanistic import measure_checkpoint

    def measure(key: str) -> dict:
        name, which = key.split("@")
        model, summary = load_model(name, which=which, runs_dir=RUNS)
        cfg = summary["config"]
        return {"run": name, "seed": int(cfg["seed"]), "which": which,
                **measure_checkpoint(model, int(cfg["p"]),
                                     float(cfg["train_frac"]),
                                     int(cfg["seed"]))}

    rows = fill_table(keys(), measure, RUNS / f"{NAME}.csv", COLUMNS,
                      key="key", force=force)
    with open(RUNS / f"{NAME}.json", "w") as f:
        json.dump({"p": 97, "train_frac": 0.30, "weight_decay": 1.0,
                   "max_steps": MAX_STEPS, "seeds": list(SEEDS),
                   "checkpoints": list(WHICH), "n_rows": len(rows),
                   "restrict_ms": list(RESTRICT_MS)}, f, indent=2)
    print(f"wrote runs/{NAME}.csv ({len(rows)} rows) and .json")
    return rows


def load_readouts(path: Path | None = None) -> list[dict]:
    """The committed read-outs. No checkpoints, no torch."""
    with open(path or RUNS / f"{NAME}.csv") as f:
        rows = []
        for r in csv.DictReader(f):
            row: dict = dict(r)
            row["seed"] = int(row["seed"])
            for k in COLUMNS[4:]:
                row[k] = float(row[k])
            rows.append(row)
    return rows


def arm(rows: list[dict], which: str, field: str) -> list[float]:
    """One checkpoint's five values of one read-out, in seed order."""
    by_seed = {r["seed"]: r[field] for r in rows if r["which"] == which}
    return [by_seed[s] for s in SEEDS]


# -- tables ------------------------------------------------------------------

def _fmt(v: float) -> str:
    if np.isnan(v):
        return "never"
    if v == 0 or 1e-3 <= abs(v) < 1e4:
        return f"{v:.4g}"
    return f"{v:.1e}"


def _span(vals: list[float]) -> str:
    finite = [v for v in vals if not np.isnan(v)]
    if not finite:
        return "never"
    s = stats(finite)
    body = f"{_fmt(s['mean'])} [{_fmt(s['min'])}–{_fmt(s['max'])}]"
    if len(finite) < len(vals):
        body += f" ({len(finite)}/{len(vals)})"
    return body


def _rank_from_favourable(vals: list[float], seed0: float, direction: int) -> str:
    """Where seed 0 sits in its own arm, counting from the flattering end.

    "1 of 5" means no seed looks more like the generalizing circuit than the
    one whose numbers are in the README.
    """
    finite = [v for v in vals if not np.isnan(v)]
    if np.isnan(seed0):
        return "n/a"
    better = sum(1 for v in finite if (v > seed0 if direction > 0 else v < seed0))
    return f"{better + 1} of {len(finite)}"


def table(rows: list[dict]) -> None:
    """Both arms of every read-out, the test between them, and seed 0's rank."""
    print(f"\n{'read-out':32s} {'at memorization':26s} {'after grokking':26s} "
          f"{'p':>6}  {'separation':>10}  seed 0 (final)")
    for field, label, direction in READOUTS:
        mem, fin = arm(rows, "memorize", field), arm(rows, "final", field)
        finite_mem = [v for v in mem if not np.isnan(v)]
        finite_fin = [v for v in fin if not np.isnan(v)]
        if finite_mem and finite_fin:
            r = rank_sum_test(finite_mem, finite_fin)
            p = f"{r['p_two_sided']:.3f}"
            sep = ("complete" if r["superiority"] in (0.0, 1.0)
                   else "partial" if r["p_two_sided"] <= 0.05 else "none")
        else:                       # e.g. "99% is never reached at memorization"
            p, sep = "n/a", "complete" if finite_fin else "n/a"
        print(f"{label:32s} {_span(mem):26s} {_span(fin):26s} {p:>6}  {sep:>10}  "
              f"{_fmt(fin[0])} ({_rank_from_favourable(fin, fin[0], direction)})")

    print(f"\nrestricted accuracy: keep the top m frequencies of the a+b "
          f"diagonal and rebuild the logits")
    print(f"{'m':>3}  {'at memorization':26s} {'after grokking':26s}")
    for m in RESTRICT_MS:
        f = f"logit_acc_m{m}"
        print(f"{m:>3}  {_span(arm(rows, 'memorize', f)):26s} "
              f"{_span(arm(rows, 'final', f)):26s}")
    print(f"{'all':>3}  {_span(arm(rows, 'memorize', 'logit_acc_full')):26s} "
          f"{_span(arm(rows, 'final', 'logit_acc_full')):26s}   "
          f"<- the model's own test accuracy")

    chance = 25 / N_FREQS
    print(f"\ntop-5 embedding freqs shared with top-5 diagonal logit freqs "
          f"(chance {chance:.2f} of 5):")
    for w in WHICH:
        print(f"  {w:10s} {_span(arm(rows, w, 'emb_diag_overlap'))}")
    print(f"\nreferences for the two 'is this structure' read-outs:")
    print(f"  unstructured logits put {2 * N_FREQS / (97 * 97 - 1):.3f} of their "
          f"energy on the a+b diagonal (closed form: 2 of p^2-1 modes per k)")
    base = random_ring_baseline(128, 97)
    print(f"  unstructured embeddings read radial CV {base['radial_cv']:.3f} and "
          f"{base['var_in_plane']:.3f} variance in the best-of-48 plane "
          f"({int(base['n_draws'])} draws)")

    print(f"\ndominant embedding frequency per seed: "
          f"{ {s: int(v) for s, v in zip(SEEDS, arm(rows, 'final', 'emb_dominant_k'))} }")


def seed_zero_summary(rows: list[dict]) -> tuple[int, int]:
    """(read-outs where seed 0 is the most flattering seed, total).

    The headline of this experiment, computed rather than eyeballed.
    """
    best = 0
    for field, _, direction in READOUTS:
        fin = arm(rows, "final", field)
        finite = [v for v in fin if not np.isnan(v)]
        if np.isnan(fin[0]) or not finite:
            continue
        extreme = max(finite) if direction > 0 else min(finite)
        best += fin[0] == extreme
    return best, len(READOUTS)


# -- figure ------------------------------------------------------------------

def figure(rows: list[dict] | None = None) -> None:
    """Per-seed dot plots of the headline read-outs, plus the restricted curve."""
    import matplotlib.pyplot as plt

    from _style import apply_style

    apply_style()
    rows = rows if rows is not None else load_readouts()

    fig, axes = plt.subplots(3, 3, figsize=(10.2, 9.4), constrained_layout=True)
    for ax, (field, label, logy) in zip(axes.flat, PANELS):
        mem, fin = arm(rows, "memorize", field), arm(rows, "final", field)
        for j, (vals, colour) in enumerate(((mem, "C3"), (fin, "C0"))):
            x = np.full(len(vals), j) + np.linspace(-0.13, 0.13, len(vals))
            ax.scatter(x[1:], vals[1:], s=26, color=colour, zorder=3,
                       label="seeds 1-4" if j == 1 else None)
            ax.scatter(x[:1], vals[:1], s=64, marker="*", color=colour,
                       edgecolors="k", linewidths=0.5, zorder=4,
                       label="seed 0 (published)" if j == 1 else None)
            ax.hlines(np.mean(vals), j - 0.22, j + 0.22, color=colour, lw=1.4)
        r = rank_sum_test(mem, fin)
        ax.set_xticks([0, 1], ["memorize", "final"])
        ax.set_xlim(-0.5, 1.5)
        if logy:
            ax.set_yscale("log")
        ax.set_title(f"{label}\n$p = {r['p_two_sided']:.3f}$", fontsize=8.5,
                     loc="left")
    axes[0, 0].legend(fontsize=7, loc="upper left")

    ax = axes[2, 2]
    for w, colour in (("memorize", "C3"), ("final", "C0")):
        for i, seed in enumerate(SEEDS):
            row = next(r for r in rows if r["which"] == w and r["seed"] == seed)
            ax.plot(RESTRICT_MS, [row[f"logit_acc_m{m}"] for m in RESTRICT_MS],
                    color=colour, lw=2.0 if seed == 0 else 0.9,
                    alpha=1.0 if seed == 0 else 0.55,
                    label=f"{w} (seed 0)" if seed == 0 else None)
    ax.set_xlabel("top $a{+}b$ frequencies kept")
    ax.set_ylabel("restricted test accuracy")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Restricted accuracy\n(thick = seed 0)", fontsize=8.5, loc="left")
    ax.legend(fontsize=7, loc="lower right")

    fig.suptitle("Every mechanistic read-out across five seeds: the story holds, "
                 "the published numbers are the flattering end", y=1.02)
    fig.savefig(FIGS / "mechanistic_seeds.png", bbox_inches="tight")
    print("saved figures/mechanistic_seeds.png")


def figure_and_table(rows: list[dict] | None = None) -> None:
    rows = rows if rows is not None else load_readouts()
    table(rows)
    best, total = seed_zero_summary(rows)
    print(f"\nseed 0 is the most flattering of the five seeds on {best} of "
          f"{total} read-outs at the final checkpoint.")
    figure(rows)


def main() -> None:
    if "--generate" in sys.argv:
        generate(force="--force" in sys.argv)
    figure_and_table()


if __name__ == "__main__":
    main()
