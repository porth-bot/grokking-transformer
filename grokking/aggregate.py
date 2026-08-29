"""Aggregate a metric across seeds onto a common step grid.

The sweeps run several seeds per configuration; each seed's trajectory can end
at a different step (a run early-stops once test accuracy holds at ~1.0, and a
non-grokking control runs the full budget). To draw a median line with an
inter-quartile band we need every seed's curve sampled on the *same* step axis.

The rule here is a forward-fill: past a seed's last logged eval its value is
held constant. That is the honest extension for these runs -- a seed that
early-stopped did so *because* it had grokked and would stay at ~1.0, and a
seed still logging simply has more steps. All runs share the eval stride, so
the union of their step lists is itself a regular grid.

Pure NumPy, no I/O, so the aggregation logic is unit-tested directly.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def align_and_aggregate(
    steps_list: Sequence[Sequence[int]],
    values_list: Sequence[Sequence[float]],
    lo_pct: float = 25.0,
    hi_pct: float = 75.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Median and percentile band of a per-seed metric on a shared step grid.

    Parameters
    ----------
    steps_list : list of 1D int sequences (one per seed), each sorted ascending
        and sharing the same stride (they all start at 0).
    values_list : list of 1D float sequences, ``values_list[i]`` aligned to
        ``steps_list[i]``.
    lo_pct, hi_pct : band percentiles (default the inter-quartile 25/75).

    Returns
    -------
    grid : (T,) int array -- the union step axis.
    median, lo, hi : (T,) float arrays -- across-seed median and band, each
        seed forward-filled to ``grid``.
    """
    if not steps_list:
        raise ValueError("need at least one seed")
    grid = np.array(sorted(set().union(*(set(map(int, s)) for s in steps_list))))
    mat = np.empty((len(values_list), grid.size))
    for i, (s, v) in enumerate(zip(steps_list, values_list)):
        steps = np.asarray(s)
        vals = np.asarray(v, dtype=float)
        if steps.size != vals.size:
            raise ValueError(f"seed {i}: steps/values length mismatch")
        # For each grid step, take the most recent logged eval at or before it;
        # clip holds the last value for grid steps beyond this seed's end.
        idx = np.clip(np.searchsorted(steps, grid, side="right") - 1, 0, vals.size - 1)
        mat[i] = vals[idx]
    median = np.median(mat, axis=0)
    lo = np.percentile(mat, lo_pct, axis=0)
    hi = np.percentile(mat, hi_pct, axis=0)
    return grid, median, lo, hi


def summarize(
    values: Sequence[float | None],
) -> tuple[float | None, float | None, float | None]:
    """(median, min, max) of a per-seed scalar (e.g. grok step), ignoring None.

    Returns ``(None, None, None)`` if every seed is ``None`` (e.g. a control
    that never grokked in any seed).
    """
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None, None
    arr = np.asarray(vals, dtype=float)
    return float(np.median(arr)), float(arr.min()), float(arr.max())


def fmt_median_range(
    values: Sequence[float | None], none_label: str = "never"
) -> str:
    """Human-readable ``median [min–max]`` for a table cell.

    ``never`` seeds (``None``) are counted: an all-``None`` cell renders as the
    label, and a partially-``None`` cell notes how many seeds reached it.
    """
    n = len(values)
    reached = [v for v in values if v is not None]
    if not reached:
        return none_label
    med, lo, hi = summarize(values)
    body = f"{med:,.0f}" if lo == hi else f"{med:,.0f} [{lo:,.0f}–{hi:,.0f}]"
    if len(reached) < n:
        body += f" ({len(reached)}/{n} seeds)"
    return body


def stats(values: Sequence[float]) -> dict[str, float]:
    """``mean``/``sd``/``min``/``max``/``n`` of a per-seed scalar read-out.

    ``sd`` is the sample standard deviation (ddof=1), which is undefined for a
    single seed and reported as ``nan`` rather than 0.0 -- a lone run has no
    measured spread, and printing 0.0 would claim it has none.
    """
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        raise ValueError("no values")
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else float("nan"),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": float(arr.size),
    }


def fmt_mean_range(values: Sequence[float], fmt: str = "{:.3f}") -> str:
    """``mean [min–max]`` for a continuous read-out (issue #4's format).

    Distinct from :func:`fmt_median_range`, which is for grok *steps*: those
    are counts on a 100-step eval grid where the median is the robust summary
    and thousands separators help. A read-out like an entropy or an energy
    fraction wants its mean and the raw extent of the seeds.
    """
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 1:
        return fmt.format(float(arr[0]))
    return (f"{fmt.format(float(arr.mean()))} "
            f"[{fmt.format(float(arr.min()))}–{fmt.format(float(arr.max()))}]")


def spread_ratio(values: Sequence[float]) -> float:
    """``max/min`` -- how many times the worst seed is the best seed.

    The slate's usual way of saying "is this effect bigger than the seed
    noise": an effect is only readable if the gap between arms exceeds the
    spread within them. Returns ``inf`` if the minimum is 0 and the maximum is
    not (an unbounded ratio, which is the honest answer), and ``nan`` if the
    values straddle zero, where a ratio means nothing.
    """
    arr = np.asarray(list(values), dtype=float)
    lo, hi = float(arr.min()), float(arr.max())
    if lo <= 0.0 <= hi and not (lo == 0.0 and hi == 0.0):
        return float("inf") if lo == 0.0 else float("nan")
    if lo == 0.0 and hi == 0.0:
        return 1.0
    return hi / lo


def rank_sum_test(
    a: Sequence[float], b: Sequence[float], max_splits: int = 200_000
) -> dict[str, float]:
    """Exact two-sample permutation test on the Mann-Whitney statistic.

    Five seeds per arm is far too few for a normal approximation and the
    read-outs here are not Gaussian (grok steps are counts on a 100-step grid,
    with ties), so the p-value is enumerated rather than approximated.

    The statistic is ``U_a = #{a_i > b_j} + 0.5 #{a_i == b_j}``, counted over
    all ``n*m`` pairs; ``U_a / (n*m)`` is the probability that a randomly drawn
    ``a`` exceeds a randomly drawn ``b`` (ties split), reported as
    ``superiority``. Under the null hypothesis that the two arms are draws from
    one distribution, the labels are exchangeable: every way of calling ``n`` of
    the ``n+m`` pooled values "a" is equally likely. So the exact null
    distribution of ``U`` is obtained by walking all ``C(n+m, n)`` splits, and

        p_greater = P(U >= U_obs),   p_less = P(U <= U_obs)

    each counting the observed split itself, which is why no p-value here can
    fall below ``1 / C(n+m, n)`` (1/252 for 5 vs 5). The two-sided p-value is
    ``min(1, 2 min(p_less, p_greater))``. Enumerating the *values* rather than
    their ranks makes ties exact for free: a tied pair contributes 0.5 in every
    split it appears in, with no midrank correction to get wrong.

    Returns ``u``, ``superiority``, ``p_less``, ``p_greater``, ``p_two_sided``,
    and two resolution floors, so a caller can tell "not significant" from
    "cannot be significant at this sample size". Which floor to compare
    against depends on which p-value is being read, and they differ by a
    factor of two: ``min_p = 1 / C(n+m, n)`` bounds ``p_less`` and
    ``p_greater``, while ``min_p_two_sided = min(1, 2 / C(n+m, n))`` bounds
    ``p_two_sided`` -- the one every caller in this repo actually quotes.
    Comparing a two-sided p against ``min_p`` is a check that can never fire:
    complete separation at five vs five reads 0.0079, not 0.0040.
    """
    from itertools import combinations
    from math import comb

    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    n, m = x.size, y.size
    if n == 0 or m == 0:
        raise ValueError("both arms need at least one value")
    n_splits = comb(n + m, n)
    if n_splits > max_splits:
        raise ValueError(
            f"C({n + m}, {n}) = {n_splits} splits exceeds max_splits="
            f"{max_splits}; this routine is exact-only by design"
        )

    def u_of(first: np.ndarray, second: np.ndarray) -> float:
        d = first[:, None] - second[None, :]
        return float((d > 0).sum() + 0.5 * (d == 0).sum())

    u_obs = u_of(x, y)
    pooled = np.concatenate([x, y])
    idx = np.arange(n + m)
    ge = le = 0
    for pick in combinations(idx, n):
        mask = np.zeros(n + m, dtype=bool)
        mask[list(pick)] = True
        u = u_of(pooled[mask], pooled[~mask])
        ge += u >= u_obs
        le += u <= u_obs
    p_greater = ge / n_splits
    p_less = le / n_splits
    return {
        "u": u_obs,
        "superiority": u_obs / (n * m),
        "p_less": p_less,
        "p_greater": p_greater,
        "p_two_sided": min(1.0, 2.0 * min(p_less, p_greater)),
        "min_p": 1.0 / n_splits,
        "min_p_two_sided": min(1.0, 2.0 / n_splits),
    }
