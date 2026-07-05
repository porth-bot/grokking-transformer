"""The grokked embedding is measurably more circular than the memorized one.

This tests the claim experiments/embedding_circle.py visualizes: at the dominant
Fourier frequency, the final embeddings lie on a ring (low radial spread, large
share of variance in the cos/sin plane) while the memorization-point embeddings
do not. Computed directly from the committed checkpoints -- no retraining.
"""

import numpy as np

from grokking.checkpoints import load_model

MAIN = "p97_frac0.30_wd1_seed0"


def _dominant_frequency(E, p):
    F = np.fft.rfft(E - E.mean(0, keepdims=True), axis=0)
    power = (F.real**2 + F.imag**2).sum(axis=1)
    return int(power[1:].argmax()) + 1


def _projection_stats(E, k, p):
    """(variance fraction in the k-plane, radial coefficient of variation)."""
    n = np.arange(p)
    c, s = np.cos(2 * np.pi * k * n / p), np.sin(2 * np.pi * k * n / p)
    Ec = E - E.mean(axis=0, keepdims=True)
    u = c @ Ec
    v = s @ Ec
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)
    x, y = Ec @ u, Ec @ v
    var_frac = (x @ x + y @ y) / (Ec * Ec).sum()
    r = np.sqrt(x**2 + y**2)
    return float(var_frac), float(r.std() / r.mean())


def test_grokked_embedding_forms_a_ring_at_top_frequency():
    mem, summary = load_model(MAIN, which="memorize")
    fin, _ = load_model(MAIN, which="final")
    p = summary["config"]["p"]
    E_mem = mem.tok_emb.weight.detach().cpu().numpy()[:p]
    E_fin = fin.tok_emb.weight.detach().cpu().numpy()[:p]

    k = _dominant_frequency(E_fin, p)
    var_fin, cv_fin = _projection_stats(E_fin, k, p)
    var_mem, cv_mem = _projection_stats(E_mem, k, p)

    # the final embedding concentrates far more variance in the frequency plane
    assert var_fin > 3 * var_mem
    # and the points are much closer to a constant radius (a rounder ring)
    assert cv_fin < 0.25 < cv_mem
