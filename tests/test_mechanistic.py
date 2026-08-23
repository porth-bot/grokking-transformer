"""The mechanistic statistics, against cases whose answers are known in advance.

``grokking/mechanistic.py`` is the shared measurement layer behind Sec. 5,
Sec. 8, Sec. 12 and the appendix. The figures pin it against the committed
checkpoints, which says the numbers are stable but not that they are *right*.
These tests build inputs whose read-out is derivable by hand -- a logit tensor
that is exactly a function of ``a + b``, an embedding matrix that is exactly a
circle, a spectrum whose top-k share can be counted -- and require the answers
the algebra gives.

The one that matters most is the chance level: a diagonal energy fraction of
19% at memorization only means something once you know that shuffling the
logits gives 1%.
"""

import numpy as np
import pytest
import torch

from grokking.mechanistic import (
    diagonal_frequency_energy,
    dominant_frequency,
    first_m_reaching,
    frequency_projection,
    n_freqs_to_reach,
    overlap,
    random_ring_baseline,
    restrict_to_freqs,
    top_indices,
    top_k_energy_fraction,
)

P = 97
K_NONDC = (P - 1) // 2       # 48 non-DC frequencies


# -- counting sparsity -------------------------------------------------------

def test_n_freqs_to_reach_counts_the_components_that_get_there():
    """[4,3,2,1]/10 reaches exactly 0.9 at three components, and a flat
    spectrum needs all of them."""
    assert n_freqs_to_reach([4, 3, 2, 1], 0.9) == 3
    assert n_freqs_to_reach([1, 1, 1, 1], 0.9) == 4
    assert n_freqs_to_reach([1, 0, 0], 0.9) == 1
    assert n_freqs_to_reach([1, 0, 0], 1.0) == 1


def test_n_freqs_to_reach_never_claims_more_concentration_than_there_is():
    """A tie at the threshold rounds toward *more* components, not fewer."""
    assert n_freqs_to_reach([1, 1, 1, 1, 1], 0.2) == 1
    assert n_freqs_to_reach([1, 1, 1, 1, 1], 0.4) == 2


def test_n_freqs_to_reach_refuses_an_all_zero_spectrum():
    with pytest.raises(ValueError, match="undefined"):
        n_freqs_to_reach([0.0, 0.0])


def test_top_k_energy_fraction_reads_squared_norms_and_drops_dc():
    """The spectrum argument is ``||F_k||_2``, so the share is of *squared*
    entries, and index 0 (the mean embedding) is excluded."""
    spec = torch.tensor([100.0, 2.0, 1.0, 1.0, 1.0, 1.0])
    assert top_k_energy_fraction(spec, 1) == pytest.approx(4 / 8)
    assert top_k_energy_fraction(spec, 5) == pytest.approx(1.0)


def test_top_indices_and_overlap():
    assert top_indices([0.1, 9.0, 5.0], 2) == [1, 2]
    assert top_indices([0.1, 9.0, 5.0], 2, offset=1) == [2, 3]
    assert overlap([1, 2, 3], [3, 4, 5]) == 1
    assert overlap([1, 2], [3, 4]) == 0


def test_first_m_reaching_says_never_rather_than_the_last_grid_point():
    assert first_m_reaching((1, 2, 3), (0.1, 0.5, 0.99), 0.99) == 3
    assert np.isnan(first_m_reaching((1, 2, 3), (0.1, 0.5, 0.9), 0.99))


# -- the logit read-out, on logits whose structure is known ------------------

def _cosine_logits(k: int) -> torch.Tensor:
    """``L[a,b,c] = cos(2 pi k (a + b - c) / p)`` -- the angle-addition circuit
    with a single frequency and nothing else."""
    n = torch.arange(P, dtype=torch.float64)
    arg = (n[:, None, None] + n[None, :, None] - n[None, None, :]) * (2 * np.pi * k / P)
    return torch.cos(arg)


def test_a_pure_a_plus_b_cosine_puts_all_its_energy_on_the_diagonal():
    """The definition's whole point: energy on ``k_a = k_b`` is the share of
    the computation that is a function of ``a + b``, so a function of ``a + b``
    scores 1."""
    diag, frac = diagonal_frequency_energy(_cosine_logits(7), P)
    assert frac == pytest.approx(1.0, abs=1e-6)
    assert int(diag.argmax()) == 7
    assert float(diag[7] / diag[1:].sum()) == pytest.approx(1.0, abs=1e-9)


def test_restricting_to_the_frequency_that_is_there_changes_nothing():
    L = _cosine_logits(7)
    assert float((restrict_to_freqs(L, P, [7]) - L).abs().max()) < 1e-9


def test_restricting_to_a_frequency_that_is_absent_leaves_only_the_mean():
    """The restriction keeps the constant-in-(a,b) part, so dropping every
    frequency present leaves exactly that -- and a pure cosine's mean over
    ``(a, b)`` is zero."""
    L = _cosine_logits(7)
    assert float(restrict_to_freqs(L, P, [9]).abs().max()) < 1e-9


def test_a_lookup_table_scores_the_chance_level_on_the_diagonal():
    """The reference the README's 19% and 98% have to be read against. The
    diagonal is 2 of the p^2 - 1 non-DC 2D frequencies per k, so unstructured
    logits put 2*48/(p^2-1) = 1.0% there -- not 0."""
    g = torch.Generator().manual_seed(0)
    _, frac = diagonal_frequency_energy(
        torch.randn(P, P, P, generator=g, dtype=torch.float64), P)
    chance = 2 * K_NONDC / (P * P - 1)
    assert chance == pytest.approx(0.0102, abs=1e-4)
    assert frac == pytest.approx(chance, rel=0.15)


# -- the embedding geometry, on an embedding that is exactly a ring ----------

def _ring(k: int) -> np.ndarray:
    n = np.arange(P)
    return np.stack([np.cos(2 * np.pi * k * n / P),
                     np.sin(2 * np.pi * k * n / P),
                     np.zeros(P)], axis=1)


def test_a_perfect_ring_has_zero_radial_spread_and_all_its_variance_in_plane():
    E = _ring(5)
    assert dominant_frequency(E, P) == 5
    _, _, var_frac, radial_cv = frequency_projection(E, 5, P)
    assert var_frac == pytest.approx(1.0)
    assert radial_cv == pytest.approx(0.0, abs=1e-12)


def test_the_ring_statistics_ignore_a_constant_offset():
    """``frequency_projection`` centres first, so translating every embedding
    by one vector -- which changes no angle -- must not move either number."""
    E = _ring(5) + np.array([3.0, -1.0, 2.0])
    _, _, var_frac, radial_cv = frequency_projection(E, 5, P)
    assert var_frac == pytest.approx(1.0)
    assert radial_cv == pytest.approx(0.0, abs=1e-12)


def test_the_ring_baseline_is_not_the_number_you_would_guess():
    """Why ``random_ring_baseline`` is measured rather than asserted.

    A plane fixed in advance holds ``2/d_model`` of an unstructured
    embedding's variance -- 1.6% at d_model=128. The dominant frequency's plane
    is the best of 48, and that selection alone lifts it to ~4%, which is
    2.6x the naive floor and right where the memorization checkpoint sits. A
    reader told "16% of the variance is in the ring plane, up from 3.4%" would
    otherwise read the second number as a small signal instead of no signal.
    """
    base = random_ring_baseline(128, P, n_draws=8, seed=0)
    assert base["var_in_plane"] == pytest.approx(0.040, abs=0.006)
    assert base["var_in_plane"] > 2.0 * (2 / 128)
    assert base["radial_cv"] == pytest.approx(0.43, abs=0.05)


def test_an_unstructured_embedding_is_not_a_ring():
    """Whatever the level, noise is nowhere near a circle: the radial CV is the
    statistic that actually separates, which is why it is the one Sec. 12
    reports and the variance share is only context."""
    rng = np.random.default_rng(0)
    E = rng.normal(size=(P, 128))
    k = dominant_frequency(E, P)
    _, _, var_frac, radial_cv = frequency_projection(E, k, P)
    assert var_frac < 0.1
    assert radial_cv > 0.3         # against 0.0 for a circle, 0.13 grokked
