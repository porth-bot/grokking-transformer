"""The seed-aggregation helpers that turn per-seed trajectories into the median
lines and IQR bands the multi-seed figures/tables report."""

import numpy as np
import pytest

from grokking.aggregate import (
    align_and_aggregate,
    fmt_mean_range,
    fmt_median_range,
    rank_sum_test,
    spread_ratio,
    stats,
    summarize,
)


def test_equal_length_seeds_reduce_to_elementwise_stats():
    steps = [[0, 100, 200], [0, 100, 200], [0, 100, 200]]
    vals = [[0.0, 0.5, 1.0], [0.2, 0.6, 0.9], [0.1, 0.4, 0.8]]
    grid, med, lo, hi = align_and_aggregate(steps, vals)
    np.testing.assert_array_equal(grid, [0, 100, 200])
    np.testing.assert_allclose(med, np.median(vals, axis=0))
    np.testing.assert_allclose(lo, np.percentile(vals, 25, axis=0))
    np.testing.assert_allclose(hi, np.percentile(vals, 75, axis=0))
    assert np.all(lo <= med) and np.all(med <= hi)


def test_shorter_seed_is_forward_filled_not_dropped():
    """A seed that early-stopped (having grokked) holds its last value across
    the union grid rather than shrinking the axis to the shortest run."""
    steps = [[0, 100, 200, 300], [0, 100]]
    vals = [[0.1, 0.4, 0.7, 1.0], [0.2, 0.5]]     # second seed ends at step 100
    grid, med, lo, hi = align_and_aggregate(steps, vals)
    np.testing.assert_array_equal(grid, [0, 100, 200, 300])
    # at steps 200 and 300 the short seed contributes its last value (0.5)
    np.testing.assert_allclose(med, np.median([[0.1, 0.4, 0.7, 1.0],
                                               [0.2, 0.5, 0.5, 0.5]], axis=0))


def test_exact_step_match_takes_that_steps_value_not_the_prior_one():
    """Forward-fill must not lag: a grid step equal to a logged step uses that
    step's value (searchsorted 'right' boundary), not the previous eval's."""
    steps = [[0, 100, 200]]
    vals = [[0.3, 0.6, 0.9]]
    grid, med, _, _ = align_and_aggregate(steps, vals)
    np.testing.assert_allclose(med, [0.3, 0.6, 0.9])


def test_length_mismatch_is_rejected():
    import pytest
    with pytest.raises(ValueError):
        align_and_aggregate([[0, 100, 200]], [[0.1, 0.2]])


def test_summarize_ignores_none_and_reports_median_min_max():
    assert summarize([1900, 1200, 1300, None, 1500]) == (1400.0, 1200.0, 1900.0)
    assert summarize([None, None]) == (None, None, None)


def test_fmt_median_range_cases():
    assert fmt_median_range([1300, 1300, 1300]) == "1,300"       # no spread
    assert fmt_median_range([1200, 1500, 1900]) == "1,500 [1,200–1,900]"
    assert fmt_median_range([None, None]) == "never"
    # a partially-reached cell notes the count
    assert "3/5 seeds" in fmt_median_range([1000, 1200, 1400, None, None])


# -- scalar read-out summaries -----------------------------------------------

def test_stats_reports_sample_sd_and_nan_for_a_lone_seed():
    import math
    s = stats([1.0, 2.0, 3.0])
    assert (s["mean"], s["min"], s["max"], s["n"]) == (2.0, 1.0, 3.0, 3.0)
    assert s["sd"] == np.std([1.0, 2.0, 3.0], ddof=1)
    # one seed has no measured spread; saying 0.0 would claim it has none
    assert math.isnan(stats([2.0])["sd"])


def test_fmt_mean_range_drops_the_range_for_a_single_value():
    assert fmt_mean_range([0.1, 0.2, 0.3]) == "0.200 [0.100–0.300]"
    assert fmt_mean_range([0.5]) == "0.500"
    assert fmt_mean_range([1.0, 2.0], "{:.1f}") == "1.5 [1.0–2.0]"


def test_spread_ratio_edge_cases():
    import math
    assert spread_ratio([2.0, 4.0, 8.0]) == 4.0
    assert spread_ratio([3.0]) == 1.0
    assert spread_ratio([0.0, 0.0]) == 1.0
    assert spread_ratio([0.0, 2.0]) == float("inf")
    assert math.isnan(spread_ratio([-1.0, 2.0]))


# -- the exact rank-sum test -------------------------------------------------
#
# Five seeds per arm is the sample size these ablations can afford, which is
# far too small for a normal approximation, so the p-value is enumerated. The
# checks below are hand-computed: with n = m = 2 the whole null distribution
# fits in a comment.

def test_complete_separation_of_five_versus_five_hits_the_resolution_floor():
    r = rank_sum_test([300, 300, 300, 300, 400], [700, 700, 700, 700, 900])
    assert r["superiority"] == 0.0          # no pair goes the other way
    assert r["u"] == 0.0
    assert r["min_p"] == pytest.approx(1 / 252)
    assert r["p_less"] == pytest.approx(1 / 252)
    assert r["p_two_sided"] == pytest.approx(2 / 252)


def test_the_null_distribution_is_enumerated_correctly_by_hand():
    """a = [1, 2], b = [3, 4]. The C(4,2) = 6 splits give U = 0, 1, 2, 2, 3, 4,
    so P(U <= 0) = 1/6, P(U >= 0) = 1, two-sided 2/6."""
    r = rank_sum_test([1, 2], [3, 4])
    assert r["u"] == 0.0
    assert r["p_less"] == pytest.approx(1 / 6)
    assert r["p_greater"] == pytest.approx(1.0)
    assert r["p_two_sided"] == pytest.approx(1 / 3)


def test_ties_count_half_and_identical_arms_are_maximally_unsurprising():
    r = rank_sum_test([1.0, 1.0], [1.0, 1.0])
    assert r["u"] == 2.0                     # 4 tied pairs at 0.5 each
    assert r["superiority"] == 0.5
    assert r["p_two_sided"] == 1.0


def test_swapping_the_arms_mirrors_the_statistic():
    a, b = [1, 5, 6, 9], [2, 3, 4, 10]
    fwd, rev = rank_sum_test(a, b), rank_sum_test(b, a)
    assert fwd["superiority"] == pytest.approx(1 - rev["superiority"])
    assert fwd["p_less"] == pytest.approx(rev["p_greater"])
    assert fwd["p_two_sided"] == pytest.approx(rev["p_two_sided"])


def test_superiority_is_the_fraction_of_pairs_the_first_arm_wins():
    a, b = [1.0, 2.0], [1.5, 3.0]            # only 2 > 1.5, so 1 of 4 pairs
    assert rank_sum_test(a, b)["superiority"] == pytest.approx(0.25)
    # and half the pairs when the arms interleave the other way
    assert rank_sum_test([1, 4], [2, 3])["superiority"] == pytest.approx(0.5)


def test_an_oversized_comparison_is_refused_rather_than_approximated():
    with pytest.raises(ValueError, match="exact-only"):
        rank_sum_test(list(range(12)), list(range(12)), max_splits=1000)
