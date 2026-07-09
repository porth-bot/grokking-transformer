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

import numpy as np


def align_and_aggregate(steps_list, values_list, lo_pct=25.0, hi_pct=75.0):
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
        s = np.asarray(s)
        v = np.asarray(v, dtype=float)
        if s.size != v.size:
            raise ValueError(f"seed {i}: steps/values length mismatch")
        # For each grid step, take the most recent logged eval at or before it;
        # clip holds the last value for grid steps beyond this seed's end.
        idx = np.clip(np.searchsorted(s, grid, side="right") - 1, 0, v.size - 1)
        mat[i] = v[idx]
    median = np.median(mat, axis=0)
    lo = np.percentile(mat, lo_pct, axis=0)
    hi = np.percentile(mat, hi_pct, axis=0)
    return grid, median, lo, hi


def summarize(values):
    """(median, min, max) of a per-seed scalar (e.g. grok step), ignoring None.

    Returns ``(None, None, None)`` if every seed is ``None`` (e.g. a control
    that never grokked in any seed).
    """
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None, None
    arr = np.asarray(vals, dtype=float)
    return float(np.median(arr)), float(arr.min()), float(arr.max())


def fmt_median_range(values, none_label="never"):
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
