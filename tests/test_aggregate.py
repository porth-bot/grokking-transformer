"""The seed-aggregation helpers that turn per-seed trajectories into the median
lines and IQR bands the multi-seed figures/tables report."""

import numpy as np

from grokking.aggregate import align_and_aggregate, fmt_median_range, summarize


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
