"""Does grokking buy commutativity, or does it buy the task's own symmetry?

Section 11 found subtraction the hardest of the three operations and proposed
an explanation: `a - b` is the only non-commutative one, and the appendix's
attention read-out shows grokking on addition *symmetrizes* how the "=" position
reads the two operands (per-head `|A[=->a] - A[=->b]|` falling 0.19 -> 0.00). A
commutative target lets that symmetric circuit serve; a non-commutative one
forbids it. Section 11 flagged the obvious problem with that argument -- the
read-out had only ever been run on addition, so "grokking symmetrizes" might
just as well be a fact about grokking as a fact about commutativity. The
"Next" list has asked for this measurement since.

This is that measurement, on the committed subtraction checkpoints, and it is
built as a *control*: if the symmetrization is about the task, it must not
appear where the task forbids it.

The one thing that has to be right for the control to be fair is what symmetry
subtraction is even allowed to have. Demanding invariance `L(a,b) = L(b,a)` of a
subtraction model is demanding that it be wrong, so a defect there is not
evidence of anything. What `a - b = -(b - a)` licenses is *anti*-equivariance:
swapping the operands negates the answer, so the logit vector should come back
permuted by `c -> -c mod p`. Both statistics are computed for both operations
(`grokking/equivariance.py`), which turns one number into a 2x2 with a
prediction in every cell:

                          invariance defect     anti-equivariance defect
    addition, grokked     ~0  (commutative)     large (negation is wrong)
    subtraction, grokked  large (must order)    ~0  (the licensed symmetry)

Both operations at their memorization checkpoints should fail both.

The "large" cells need a referent, or "1.0" is a number about nothing. Every
defect is normalized by the logits' own sd, and a shuffle baseline -- each input
paired with a *random* other input rather than with its swap -- measures what
the statistic reads when there is no symmetry to find.

Trains nothing: it reads the checkpoints the operations sweep already produced.
Those are not committed -- ``.gitignore`` keeps ``.pt`` files out of the repo
except the two the Fourier figure needs -- so this follows the same split the
progress-measure and attention trajectories use: ``generate()`` reads the local
checkpoints and writes a small committed CSV of the read-outs, and ``figure()``
replays the figure from that CSV alone. A fresh clone can reproduce the figure
and check every number in the README without 12 MB of weights or a retrain;
regenerating the CSV itself needs the sweep (``experiments/operations.py``).

Run:  python experiments/swap_equivariance.py            (figure, from the CSV)
      python experiments/swap_equivariance.py --generate (re-measure, needs
                                                          the sweep checkpoints)
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import apply_style  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
NAME = "swap_equivariance_p97"
P = 97
FIELDS = ("logit_swap_defect", "logit_anti_defect",
          "attn_equivariance_defect", "shuffle_baseline")
COLUMNS = ("op", "wd", "seed", "which", "run", "test_acc") + FIELDS
SEEDS = (0, 1, 2)
WEIGHT_DECAYS = ("1", "0.1")
OPS = ("add", "sub")
# A run counts as grokked if it actually generalized; the wd 0.1 subtraction
# cell mostly did not (Sec. 11 measured one seed of three), and averaging a
# failed run into a claim about what grokking does would be the whole point
# missed.
GROKKED_ACC = 0.8


def run_name(op: str, wd: str, seed: int) -> str:
    base = f"p97_frac0.30_wd{wd}_seed{seed}"
    return base if op == "add" else f"{base}_opsub"


def generate(out_dir: Path = RUNS) -> None:
    """Measure every read-out from the sweep's checkpoints; write the CSV.

    Needs the ``.pt`` files ``experiments/operations.py`` produces, which are
    gitignored. This is the only step that does, and it is why the CSV is
    committed.
    """
    from grokking.checkpoints import load_model
    from grokking.data import modular_dataset
    from grokking.equivariance import measure_equivariance

    rows = []
    for op in OPS:
        tokens, _ = modular_dataset(P, op)
        for wd in WEIGHT_DECAYS:
            for seed in SEEDS:
                name = run_name(op, wd, seed)
                for which in ("memorize", "final"):
                    model, summary = load_model(name, which=which)
                    model.eval()
                    rows.append({
                        "op": op, "wd": wd, "seed": seed, "which": which,
                        "run": name,
                        "test_acc": float(summary["final_test_acc"]),
                        **measure_equivariance(model, tokens, P),
                    })
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / f"{NAME}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(COLUMNS))
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / f"{NAME}.json", "w") as f:
        json.dump({"p": P, "seeds": list(SEEDS), "weight_decays":
                   list(WEIGHT_DECAYS), "operations": list(OPS),
                   "grokked_threshold": GROKKED_ACC,
                   "n_checkpoints": len(rows)}, f, indent=2)
    print(f"wrote runs/{NAME}.csv ({len(rows)} checkpoints) and .json")


def collect(csv_path: Path = RUNS / f"{NAME}.csv") -> list[dict]:
    """The committed read-outs. No checkpoints, no torch."""
    with open(csv_path) as f:
        rows = []
        for r in csv.DictReader(f):
            row = dict(r)
            row["seed"] = int(row["seed"])
            for k in ("test_acc",) + FIELDS:
                row[k] = float(row[k])
            rows.append(row)
    return rows


def cell(rows, op, wd, which, field, grokked_only=False):
    vals = [r[field] for r in rows
            if r["op"] == op and r["wd"] == wd and r["which"] == which
            and (not grokked_only or r["test_acc"] >= GROKKED_ACC)]
    return np.array(vals)


def fmt(v):
    if len(v) == 0:
        return "     --      "
    return f"{np.median(v):.3f} [{v.min():.2f},{v.max():.2f}]"


def report(rows) -> None:
    base = np.array([r["shuffle_baseline"] for r in rows])
    print(f"shuffle baseline (no symmetry to find), over all "
          f"{len(base)} checkpoints: median {np.median(base):.3f} "
          f"[{base.min():.2f}, {base.max():.2f}]\n")

    for field, title in (
            ("logit_swap_defect", "invariance defect   L(a,b) vs L(b,a)"),
            ("logit_anti_defect", "anti-equivariance   L(a,b) vs L(b,a)[-c]"),
            ("attn_equivariance_defect",
             "attention defect    E|A[(a,b)->a] - A[(b,a)->b]|  (Sec. 11's statistic)")):
        print(f"{title}   -- median [min, max] over {len(SEEDS)} seeds")
        print(f"{'':10s} {'add, memorize':>16s} {'add, grokked':>16s} "
              f"{'sub, memorize':>16s} {'sub, grokked':>16s}")
        for wd in WEIGHT_DECAYS:
            cells = [fmt(cell(rows, op, wd, which, field,
                              grokked_only=(which == "final")))
                     for op in OPS for which in ("memorize", "final")]
            # column order is add/mem, add/final, sub/mem, sub/final
            order = [cells[0], cells[1], cells[2], cells[3]]
            print(f"  wd={wd:<5s} " + " ".join(f"{c:>16s}" for c in order))
        print()

    ngrok = {(op, wd): int(sum(1 for r in rows
                               if r["op"] == op and r["wd"] == wd
                               and r["which"] == "final"
                               and r["test_acc"] >= GROKKED_ACC))
             for op in OPS for wd in WEIGHT_DECAYS}
    print("runs counted as grokked (test acc >= "
          f"{GROKKED_ACC}), of {len(SEEDS)}: " +
          ", ".join(f"{op} wd={wd}: {n}" for (op, wd), n in ngrok.items()))


def figure(rows) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9), constrained_layout=True)
    base = float(np.median([r["shuffle_baseline"] for r in rows]))

    for ax, (field, title) in zip(axes, (
            ("logit_swap_defect",
             "Invariance:  $L(a,b)$ vs $L(b,a)$"),
            ("logit_anti_defect",
             "Anti-equivariance:  $L(a,b)$ vs $L(b,a)[-c]$"))):
        labels, groups, colors = [], [], []
        for op in OPS:
            for which in ("memorize", "final"):
                v = cell(rows, op, "1", which, field,
                         grokked_only=(which == "final"))
                labels.append(f"{op}\n{'memorized' if which == 'memorize' else 'grokked'}")
                groups.append(v)
                colors.append("C1" if which == "memorize" else "C0")
        xs = np.arange(len(groups))
        ax.bar(xs, [np.median(v) for v in groups], color=colors, alpha=0.75,
               width=0.62)
        for x, v in zip(xs, groups):
            ax.plot(np.full(len(v), x), v, "o", color="0.2", ms=3.5, zorder=3)
        ax.axhline(base, color="0.45", ls=":", lw=1.2)
        ax.annotate("no symmetry (shuffle baseline)", xy=(-0.42, base),
                    xytext=(0, 4), textcoords="offset points", ha="left",
                    va="bottom", fontsize=7.5, color="0.35")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, max(base, max(v.max() for v in groups)) * 1.22)
        ax.set_ylabel("defect / logit sd")
        ax.set_title(title, loc="left", fontsize=10)

    fig.suptitle("Grokking acquires the symmetry the operation has, not "
                 "commutativity  ($wd=1$, 3 seeds, dots are seeds)",
                 fontsize=10)
    out = ROOT / "figures" / "swap_equivariance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out.relative_to(ROOT)}")


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if "--generate" in argv:
        generate()
    rows = collect()
    report(rows)
    figure(rows)


if __name__ == "__main__":
    main()
