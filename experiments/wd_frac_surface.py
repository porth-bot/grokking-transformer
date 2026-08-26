"""Does more data offset less regularization?  -- the (wd, frac) surface.

Sections 1 and 2 vary weight decay and training fraction *one at a time*: a wd
slice at frac 0.30, a frac slice at wd 1.0. Both are monotone, and both are
consistent with a story neither of them can test -- that the two knobs trade off
against each other, so that enough data would buy grokking at a weight decay
that cannot produce it on its own. Reading that off two orthogonal slices
through a surface is exactly the mistake a one-variable-at-a-time design invites
(issue #3). This is the coarse grid that answers it: wd in {0, 0.1, 0.3, 1.0}
crossed with frac in {0.25, 0.30, 0.40}, every cell at three seeds.

Three things about the design are load-bearing.

**Censoring, not missing data.** A cell that does not grok inside the 25,000-step
budget has not produced a large number; it has produced *no* number, bounded
below by 25,000. The distinction matters because half of the wd=0 row is like
that, and the two wrong ways to handle it are both tempting: dropping those
cells biases the surface toward the fast corner, and imputing 25,000 for them
invents a measurement. What is done here instead is the standard right-censored
treatment -- ``censored_median`` sorts a cell's seeds with the censored ones at
``+inf`` and takes the median, which is an exact observed value whenever fewer
than half the seeds are censored and otherwise reports "> 25,000" honestly. The
figure marks those cells rather than colouring them, so no reader can take a
gradient across a region where the measurement stops existing.

**Three seeds, and what three seeds can and cannot say.** Sections 9 and 12 are
the repo's own argument against single-seed timing claims -- the issue asks for
seed 0, but a grok step is heavy-tailed here (§9's arms span 1.3-1.6x within a
cell) and issue #4 was filed precisely because seed-0 numbers had been shipped
as if they were the quantity. Three seeds give each cell a median and a range.
They do *not* give the separation §9 gets: the exact rank-sum test's floor at
three vs three is 1/C(6,3) = 0.05, so this grid can rank cells and measure how
much bigger a cell-to-cell step is than the within-cell spread, and cannot
certify a single pairwise comparison at the level §9 does. That is stated here
rather than papered over.

**The interaction is a quantitative claim, so it gets a quantitative null.**
"The two knobs do not interact" has a precise meaning on this surface: in logs,
the delay is additive, ``log10 T(f, w) = mu + a_f + b_w``, i.e. the frac curves
are parallel and shifting wd multiplies the grok step by the same factor at
every training fraction. ``additive_fit`` fits that null by least squares over
the cells whose median is identified and reports each cell's residual as a
*ratio* (10^residual), which is directly comparable to the cell's own seed
spread. An interaction is only worth naming if it is larger than the noise it
has to beat.

Artifacts: every run's CSV/JSON trajectory is committed, so the table and figure
replay from the repo with no training and no checkpoints -- the surface needs
only ``grok_step`` and ``steps_run`` from each summary JSON. Nothing extra is
cached, which also means nothing extra can go stale.

Run:  python experiments/wd_frac_surface.py            (table + figure)
      python experiments/wd_frac_surface.py --train    (fill in missing cells)
      python experiments/wd_frac_surface.py --train --budget-seconds 540
"""

from __future__ import annotations

import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _style import apply_style, plt  # noqa: E402

from grokking.aggregate import fmt_median_range  # noqa: E402
from grokking.seeds import ensure_runs  # noqa: E402
from grokking.train import TrainConfig  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
FIGS = ROOT / "figures"

# The grid. wd 0.0/0.1/1.0 and frac 0.25/0.30/0.40 are the values §1 and §2
# already use, so five of the twelve cells are the sweep runs those sections
# report and need no retraining; wd 0.3 is the new column, placed where §1's
# 108x -> 13x jump between 0.1 and 1.0 is widest.
WDS = (0.0, 0.1, 0.3, 1.0)
FRACS = (0.25, 0.30, 0.40)
SEEDS = (0, 1, 2)

# Shared with run_sweep.py's cells: the budget a non-grokking run exhausts, and
# therefore the value every censored observation is bounded below by. §1's
# "never (25k budget)" is this number, so the two must not drift apart.
MAX_STEPS = 25_000
EVAL_EVERY = 100

NAME = "wd_frac_surface"


def cfg_for(frac: float, wd: float, seed: int) -> TrainConfig:
    """The configuration for one grid cell and seed.

    Everything except ``train_frac``/``weight_decay``/``seed`` is the main
    run's config, so the five overlapping cells produce byte-identical run
    names to the ones ``run_sweep.py`` already trained -- which is what lets
    this grid reuse them instead of retraining a third of itself.
    """
    return TrainConfig(
        p=97, train_frac=frac, weight_decay=wd,
        max_steps=MAX_STEPS, eval_every=EVAL_EVERY, seed=seed,
    )


def cells() -> list[tuple[float, float]]:
    """Every ``(frac, wd)`` cell of the grid, strongest regularization first.

    Order is the *training* order, and it runs from the cheap corner to the
    expensive one: grok time falls steeply in wd, so a wd=1.0 cell early-stops
    in a couple of thousand steps while a wd=0 cell spends the whole 25,000-step
    budget. Filling the cheap rows first means an interrupted pass leaves a
    usable partial surface rather than three finished controls and no signal.
    """
    return [(f, w) for w in sorted(WDS, reverse=True) for f in FRACS]


def run_names() -> dict[tuple[float, float], list[str]]:
    """``(frac, wd) -> [run name per seed]``, in ``SEEDS`` order."""
    return {
        (f, w): [cfg_for(f, w, s).run_name() for s in SEEDS]
        for f, w in cells()
    }


def train_grid(budget_seconds: float | None = None, runs_dir: Path = RUNS) -> bool:
    """Train whatever the grid is missing; return True when the grid is full.

    ``budget_seconds`` time-boxes the pass: this host suspends a process that is
    not holding the foreground, and a foreground call here cannot outlive its
    own timeout, so a full grid is filled by repeated bounded passes. The check
    happens between cells, and a cell is atomic -- an interrupted run leaves no
    JSON summary and is simply redone, which is ``ensure_runs``'s contract.
    """
    t0 = time.time()
    todo = [
        (f, w) for f, w in cells()
        if not all(runs_dir.joinpath(f"{n}.json").exists()
                   for n in run_names()[(f, w)])
    ]
    for frac, wd in todo:
        if budget_seconds is not None and time.time() - t0 > budget_seconds:
            print(f"budget spent; {len(todo)} cell(s) were outstanding", flush=True)
            return False
        print(f"=== cell frac={frac} wd={wd} ===", flush=True)
        ensure_runs(lambda s, f=frac, w=wd: cfg_for(f, w, s), SEEDS, out_dir=runs_dir)
    return True


def load_cell(
    frac: float, wd: float, runs_dir: Path = RUNS
) -> list[float | None]:
    """Each seed's grok step for one cell; ``None`` where it never grokked.

    A ``None`` is only meaningful if the run actually spent the whole budget --
    a run that stopped early *and* never grokked would be a truncated job, not a
    censored observation, and averaging the two together is how a surface grows
    a fast region that is really an interrupted one. So that is checked here
    rather than assumed.
    """
    out: list[float | None] = []
    for seed in SEEDS:
        name = cfg_for(frac, wd, seed).run_name()
        with open(runs_dir / f"{name}.json") as fh:
            summary = json.load(fh)
        grok = summary["grok_step"]
        if grok is None and summary["steps_run"] < MAX_STEPS:
            raise ValueError(
                f"{name}: no grok step but only {summary['steps_run']} of "
                f"{MAX_STEPS} steps -- that is a truncated run, not a censored "
                "observation; delete its artifacts and retrain"
            )
        out.append(None if grok is None else float(grok))
    return out


def censored_median(values: list[float | None]) -> tuple[float, bool]:
    """Median of a cell, treating ``None`` as right-censored at ``MAX_STEPS``.

    Returns ``(value, censored)``. Every censored seed is known only to exceed
    every observed one, so sorting with ``+inf`` in their place puts the sample
    in the true order whatever the unobserved values are -- and the median is
    then an *exact* statistic of the cell whenever it lands on an observed seed,
    which happens iff fewer than half the seeds are censored. When it does not,
    the median is itself censored and the returned value is ``MAX_STEPS`` with
    ``censored=True``, meaning "at least this", never "equal to this".

    With an even number of seeds the midpoint of the two middle order statistics
    is identified only if the upper one is; that case is handled the same way.
    """
    if not values:
        raise ValueError("empty cell")
    ordered = sorted(math.inf if v is None else v for v in values)
    n = len(ordered)
    mid = ordered[n // 2] if n % 2 else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])
    if math.isinf(mid):
        return float(MAX_STEPS), True
    return float(mid), False


def surface(runs_dir: Path = RUNS) -> dict[str, np.ndarray]:
    """The grid as arrays indexed ``[frac, wd]``.

    ``median`` is the censored median, ``censored`` flags the cells where it is
    a lower bound, ``n_censored`` counts the seeds that never grokked, and
    ``lo``/``hi`` are the observed extremes (``nan`` where nothing was observed).
    """
    shape = (len(FRACS), len(WDS))
    med = np.zeros(shape)
    cen = np.zeros(shape, dtype=bool)
    n_cen = np.zeros(shape, dtype=int)
    lo = np.full(shape, np.nan)
    hi = np.full(shape, np.nan)
    for i, frac in enumerate(FRACS):
        for j, wd in enumerate(WDS):
            vals = load_cell(frac, wd, runs_dir)
            med[i, j], cen[i, j] = censored_median(vals)
            n_cen[i, j] = sum(v is None for v in vals)
            seen = [v for v in vals if v is not None]
            if seen:
                lo[i, j], hi[i, j] = min(seen), max(seen)
    return {"median": med, "censored": cen, "n_censored": n_cen,
            "lo": lo, "hi": hi}


def additive_fit(
    median: np.ndarray, censored: np.ndarray
) -> dict[str, np.ndarray | float]:
    """Least-squares fit of ``log10 T = mu + a_frac + b_wd`` to the open cells.

    This is the no-interaction null: under it, changing wd multiplies the grok
    step by the same factor at every training fraction, so the frac curves in
    §2's log axes are parallel translates. The fit is over the cells whose
    median is identified (``censored`` False) -- a censored cell contributes an
    inequality, not a value, and least squares has no way to use one.

    Effects are reported with sum-to-zero constraints so ``mu`` is the grand
    mean of the fitted log delays and ``a``/``b`` are readable as "this row is
    10^a times the average row".

    Returns ``mu``, ``a`` (per frac), ``b`` (per wd), ``resid`` (log10 units,
    ``nan`` on censored cells), ``ratio`` (10^resid), and ``dof``.
    """
    n_f, n_w = median.shape
    rows, ys = [], []
    for i in range(n_f):
        for j in range(n_w):
            if censored[i, j]:
                continue
            row = np.zeros(1 + n_f + n_w)
            row[0] = 1.0
            row[1 + i] = 1.0
            row[1 + n_f + j] = 1.0
            rows.append(row)
            ys.append(math.log10(median[i, j]))
    if len(rows) < 1 + (n_f - 1) + (n_w - 1) + 1:
        raise ValueError(
            f"{len(rows)} identified cells cannot support an additive fit with "
            f"{(n_f - 1) + (n_w - 1)} free effects plus a residual"
        )
    # Sum-to-zero constraints on each effect group, appended as exact rows with
    # a large weight: they fix the (otherwise rank-deficient) parameterization
    # without changing the fitted surface.
    big = 1e6
    for group_start, size in ((1, n_f), (1 + n_f, n_w)):
        row = np.zeros(1 + n_f + n_w)
        row[group_start:group_start + size] = big
        rows.append(row)
        ys.append(0.0)
    beta, *_ = np.linalg.lstsq(np.array(rows), np.array(ys), rcond=None)
    mu, a, b = beta[0], beta[1:1 + n_f], beta[1 + n_f:]
    resid = np.full(median.shape, np.nan)
    n_obs = 0
    for i in range(n_f):
        for j in range(n_w):
            if censored[i, j]:
                continue
            resid[i, j] = math.log10(median[i, j]) - (mu + a[i] + b[j])
            n_obs += 1
    return {
        "mu": float(mu), "a": a, "b": b, "resid": resid,
        "ratio": np.power(10.0, resid),
        "dof": n_obs - (1 + (n_f - 1) + (n_w - 1)),
    }


def additive_predict(fit: dict, i: int, j: int) -> float:
    """The no-interaction fit's prediction for cell ``[i, j]``, in steps."""
    return float(10.0 ** (fit["mu"] + fit["a"][i] + fit["b"][j]))


def censored_evidence(
    median: np.ndarray, censored: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> list[dict[str, float | bool | int]]:
    """What the censored cells say about the additive null, which the fit cannot.

    ``additive_fit`` drops the censored cells because least squares has no way
    to use an inequality -- and on this grid those are the three cells the
    section is *about*, so reporting only the fit's residuals would answer the
    interaction question using nothing but the cells where the interesting thing
    did not happen.

    A censored cell still carries a testable statement. The null predicts a
    number for it; the cell says the truth exceeds ``MAX_STEPS``. So
    ``bound_ratio = MAX_STEPS / prediction`` is a **lower bound** on how much
    slower the cell is than additivity allows: above 1 it is evidence of a
    super-additive delay (the two knobs hurting together more than their
    product), at or below 1 the censoring is consistent with the null and says
    nothing at all.

    The bound is only as good as the fit behind it, and on this grid that fit is
    fragile in a specific way: the wd=0 column effect rests on the single open
    cell in that column, whose three seeds span 16.5x. So each bound comes with
    an envelope -- the fit is redone with every open cell moved to each end of
    its own observed seed range, and the extreme bound ratios over those refits
    are reported. The fitted log delay is linear in the open cells' log medians
    and the ratio is monotone in it, so the extremes are attained at a vertex of
    that box and enumerating the vertices gives the exact envelope.

    Returns one dict per censored cell, in row-major order.
    """
    n_f, n_w = median.shape
    open_idx = [(i, j) for i in range(n_f) for j in range(n_w) if not censored[i, j]]
    if len(open_idx) > 20:
        raise ValueError(
            f"{len(open_idx)} open cells is too many to enumerate the envelope "
            "over (2^n refits); widen this guard deliberately if the grid grows"
        )
    targets = [(i, j) for i in range(n_f) for j in range(n_w) if censored[i, j]]
    base = additive_fit(median, censored)
    span = {t: [math.inf, -math.inf] for t in targets}
    for corner in itertools.product(*[(lo[i, j], hi[i, j]) for i, j in open_idx]):
        perturbed = median.copy()
        for (i, j), value in zip(open_idx, corner):
            perturbed[i, j] = value
        fit = additive_fit(perturbed, censored)
        for i, j in targets:
            ratio = MAX_STEPS / additive_predict(fit, i, j)
            span[(i, j)][0] = min(span[(i, j)][0], ratio)
            span[(i, j)][1] = max(span[(i, j)][1], ratio)
    out: list[dict[str, float | bool | int]] = []
    for i, j in targets:
        pred = additive_predict(base, i, j)
        out.append({
            "frac": FRACS[i], "wd": WDS[j], "i": i, "j": j,
            "prediction": pred,
            "bound": float(MAX_STEPS),
            "bound_ratio": MAX_STEPS / pred,
            "ratio_lo": span[(i, j)][0],
            "ratio_hi": span[(i, j)][1],
            # "This cell refutes additivity" is only safe if even the most
            # favourable refit still puts the cell past the prediction.
            "refutes_additivity": span[(i, j)][0] > 1.0,
        })
    return out


def substitution_check(
    censored: np.ndarray, n_censored: np.ndarray
) -> dict[str, object]:
    """Does more data buy grokking where weight decay alone cannot?

    This is issue #3's actual question, and it is answerable without any fit at
    all: look down the least-regularized column and count. If some training
    fractions are censored there and a larger one is not, data has substituted
    for regularization -- a statement about which cells produced a grok step and
    which produced none, so no model, no interpolation, and nothing a seed
    spread can move.
    """
    j = int(np.argmin(WDS))
    column = [(FRACS[i], bool(censored[i, j]), int(n_censored[i, j]))
              for i in range(len(FRACS))]
    blocked = [f for f, cen, _ in column if cen]
    grokking = [f for f, cen, _ in column if not cen]
    return {
        "wd": WDS[j],
        "column": column,
        "blocked": blocked,
        "grokking": grokking,
        # True only if the split is monotone in frac: every blocked fraction
        # below every grokking one. A non-monotone split would be a different
        # (and much stranger) finding, so it is not quietly reported as this one.
        "substitutes": bool(blocked and grokking and max(blocked) < min(grokking)),
    }


def spread_ratios(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Per-cell ``max/min`` over the seeds that grokked (``nan`` if none did)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return hi / lo


def table(runs_dir: Path = RUNS) -> str:
    """The markdown table §13 reports, plus the interaction summary."""
    s = surface(runs_dir)
    fit = additive_fit(s["median"], s["censored"])
    spread = spread_ratios(s["lo"], s["hi"])
    lines = ["| train frac | " + " | ".join(f"wd {w:g}" for w in WDS) + " |",
             "|---" * (len(WDS) + 1) + "|"]
    for i, frac in enumerate(FRACS):
        cellstrs = []
        for j, _ in enumerate(WDS):
            vals = load_cell(frac, WDS[j], runs_dir)
            cellstrs.append(
                f"**> {MAX_STEPS:,}**" if s["censored"][i, j]
                else fmt_median_range(vals)
            )
        lines.append(f"| {frac:.0%} | " + " | ".join(cellstrs) + " |")
    lines.append("")
    lines.append(
        f"Additive (no-interaction) fit over the {int(np.isfinite(fit['resid']).sum())} "
        f"identified cells, {fit['dof']} residual dof:"
    )
    lines.append("")
    lines.append("| train frac | " + " | ".join(f"wd {w:g}" for w in WDS)
                 + " | seed spread (max/min) |")
    lines.append("|---" * (len(WDS) + 2) + "|")
    for i, frac in enumerate(FRACS):
        cellstrs = []
        for j in range(len(WDS)):
            r = fit["ratio"][i, j]
            cellstrs.append("—" if not np.isfinite(r) else f"{r:.2f}×")
        worst = np.nanmax(spread[i])
        lines.append(f"| {frac:.0%} | " + " | ".join(cellstrs)
                     + f" | up to {worst:.2f}× |")
    return "\n".join(lines)


def figure(runs_dir: Path = RUNS, out: Path | None = None) -> Path:
    """Two panels: the censored surface, and whether the frac curves are parallel."""
    apply_style()
    s = surface(runs_dir)
    med, cen = s["median"], s["censored"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.6))

    # Panel A -- discrete cells, never interpolated. pcolormesh with explicit
    # edges draws one flat rectangle per cell, so nothing between two measured
    # configurations is ever painted as if it had been measured; the censored
    # cells are masked out of the colour map entirely and hatched instead.
    shown = np.where(cen, np.nan, np.log10(med))
    xe = np.arange(len(WDS) + 1) - 0.5
    ye = np.arange(len(FRACS) + 1) - 0.5
    mesh = ax.pcolormesh(xe, ye, np.ma.masked_invalid(shown),
                         cmap="viridis_r", shading="flat")
    for i in range(len(FRACS)):
        for j in range(len(WDS)):
            if cen[i, j]:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, facecolor="0.90",
                    edgecolor="0.55", hatch="///", linewidth=0.8))
                ax.text(j, i, f"> {MAX_STEPS // 1000}k\n({s['n_censored'][i, j]}"
                              f"/{len(SEEDS)} never)",
                        ha="center", va="center", fontsize=7.5, color="0.25")
            else:
                dark = shown[i, j] > np.nanmean(shown)
                ax.text(j, i, f"{med[i, j]:,.0f}\n[{s['lo'][i, j]:,.0f}–"
                              f"{s['hi'][i, j]:,.0f}]",
                        ha="center", va="center", fontsize=7.5,
                        color="white" if dark else "black")
    ax.set_xticks(range(len(WDS)), [f"{w:g}" for w in WDS])
    ax.set_yticks(range(len(FRACS)), [f"{f:.0%}" for f in FRACS])
    ax.set_xlabel("weight decay")
    ax.set_ylabel("train fraction")
    ax.set_title("median grok step over 3 seeds [min–max]")
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    fig.colorbar(mesh, ax=ax, label="log$_{10}$ grok step")

    # Panel B -- the interaction, read as parallelism. Under the additive null
    # the three curves are vertical translates of one another on this log axis;
    # a real interaction bends them apart. Two details keep the panel from
    # overstating what three seeds can see: the min-max seed range is drawn on
    # every median, because the flattest-looking row is also the noisiest and
    # the bars are what let a reader check the bend against the spread; and the
    # censored cells are drawn as up-arrows at the budget, nudged apart in x so
    # that two rows censored at the same wd do not hide one another.
    x = np.arange(len(WDS), dtype=float)
    offsets = np.linspace(-0.06, 0.06, len(FRACS))
    for i, frac in enumerate(FRACS):
        open_cells = ~cen[i]
        xo = x + offsets[i]
        yerr = np.vstack([med[i] - s["lo"][i], s["hi"][i] - med[i]])
        ax2.errorbar(xo[open_cells], med[i][open_cells],
                     yerr=yerr[:, open_cells], fmt="o-", capsize=2.5,
                     elinewidth=1.0, markersize=4, label=f"frac {frac:.0%}")
        colour = ax2.lines[-1].get_color()
        for j in np.flatnonzero(cen[i]):
            ax2.errorbar(xo[j], MAX_STEPS, yerr=[[0], [MAX_STEPS * 0.6]],
                         lolims=True, color=colour, capsize=3, elinewidth=1.0)
    ax2.axhline(MAX_STEPS, color="0.6", lw=0.8, ls=":")
    ax2.text(len(WDS) - 1, MAX_STEPS * 1.08, "budget", ha="right",
             fontsize=7.5, color="0.4")
    ax2.set_yscale("log")
    ax2.set_xticks(x, [f"{w:g}" for w in WDS])
    ax2.set_xlabel("weight decay")
    ax2.set_ylabel("grok step (median)")
    ax2.set_title("parallel curves = no interaction")
    ax2.legend(loc="lower left")

    fig.tight_layout()
    path = out or (FIGS / f"{NAME}.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_and_table(runs_dir: Path = RUNS) -> Path:
    print(table(runs_dir))
    path = figure(runs_dir)
    print(f"\nwrote {path.relative_to(ROOT)}")
    return path


def main() -> int:
    args = sys.argv[1:]
    if "--train" in args:
        budget = None
        if "--budget-seconds" in args:
            budget = float(args[args.index("--budget-seconds") + 1])
        done = train_grid(budget_seconds=budget)
        if not done:
            print("grid incomplete -- rerun --train to continue", flush=True)
            return 0
    figure_and_table()
    return 0


if __name__ == "__main__":
    sys.exit(main())
