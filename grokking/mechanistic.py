"""The mechanistic read-outs of one checkpoint, in one place.

Every "what changed inside the model" claim in this repo -- the embedding
Fourier spectrum (Sec. 5), the logit attribution (Sec. 8), the attention
pattern and the embedding ring (appendix) -- was written as a script that
loaded seed 0's two checkpoints, measured one thing, and drew one figure. That
was fine while each was a separate story. Issue #4 asks for all of them across
seeds, which turns "measure one thing" into "measure the same eighteen things
on ten checkpoints", and at that point the statistics have to live somewhere
importable rather than one per plotting script.

So this module owns the measurements and the experiment scripts own the
figures. Two consequences worth having:

- ``experiments/fourier.py``, ``experiments/embedding_circle.py`` and
  ``experiments/logit_attribution.py`` now import their statistics from here
  instead of defining them, so "the seed sweep measures the same quantity the
  published figure does" is a property of the code and not of my memory. The
  duplicated copies in ``tests/test_embedding_circle.py`` stay duplicated on
  purpose -- an independent reimplementation is worth more as a test than a
  second call to the function under test.
- Nothing here imports matplotlib, so the seed-sweep tests run in CI, where
  the plotting dependency is not installed.

Everything takes an already-loaded model and returns floats; loading, caching
and aggregation are ``grokking.checkpoints`` and ``grokking.seeds``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from .attention import measure_attention
from .data import modular_addition_dataset, train_test_split
from .equivariance import measure_equivariance
from .model import Transformer

# Restricted-accuracy grid: how many top ``a+b`` frequencies to keep. Small
# values are where the interesting resolution is (seed 0 reaches 1.00 at three),
# so the grid is dense there and sparse after.
RESTRICT_MS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8, 10)


# -- embeddings --------------------------------------------------------------

def embedding_spectrum(model: Transformer, p: int) -> torch.Tensor:
    """Per-frequency L2 norm of the digit embeddings.

    ``E`` is ``(p, d_model)`` -- the "=" token row is excluded, since only digit
    tokens participate in the periodic structure. ``rfft`` along the token axis
    gives coefficients for frequencies ``k = 0 .. (p-1)/2``; we report
    ``||F_k||_2`` over the model dimension, and the DC term ``k=0`` (the mean
    embedding) is dropped from sparsity statistics.
    """
    E = model.tok_emb.weight.detach()[:p]          # (p, d_model)
    F = torch.fft.rfft(E, dim=0)                   # (p//2 + 1, d_model), complex
    return F.abs().pow(2).sum(dim=1).sqrt()        # (p//2 + 1,)


def top_k_energy_fraction(spec: torch.Tensor, k: int = 5) -> float:
    """Fraction of squared spectral norm captured by the top-k frequencies
    (excluding DC)."""
    energy = spec[1:].pow(2)
    top = energy.sort(descending=True).values[:k].sum()
    return float(top / energy.sum())


def n_freqs_to_reach(energy: Sequence[float] | np.ndarray | torch.Tensor,
                     frac: float = 0.9) -> int:
    """How many of the largest components hold ``frac`` of the total energy.

    The seed-invariant way to say "sparse". ``top_k_energy_fraction`` fixes k
    and reads off a share, which is the natural summary for one model but a
    poor one across seeds: the share at k=5 depends on *how many* frequencies a
    seed happened to use, so a seed that solves the task with six frequencies
    scores worse than one that uses four while being no less sparse. Fixing the
    share and reading off the count inverts that, and the count is directly
    comparable to the 48 available frequencies.

    Ties are broken by taking the larger count (``searchsorted`` on the
    cumulative share), so the answer never claims more concentration than the
    data has.
    """
    e = np.sort(np.asarray(_to_numpy(energy), dtype=float))[::-1]
    total = e.sum()
    if total <= 0:
        raise ValueError("energy is all zero; 'frequencies to reach' is undefined")
    return int(np.searchsorted(np.cumsum(e) / total, frac) + 1)


def top_indices(energy: Sequence[float] | np.ndarray | torch.Tensor,
                m: int, offset: int = 0) -> list[int]:
    """The ``m`` largest components' indices, shifted by ``offset``.

    ``offset`` exists because the two spectra here are indexed differently: the
    embedding spectrum's array position *is* the frequency (with DC at 0 and
    dropped by slicing, hence offset 1), while the diagonal logit energy is
    already stored per frequency.
    """
    e = np.asarray(_to_numpy(energy), dtype=float)
    return [int(i) + offset for i in np.argsort(e)[::-1][:m]]


def overlap(a: Sequence[int], b: Sequence[int]) -> int:
    """How many frequencies two "top-m" lists share.

    Sec. 8 claims the dominant logit frequencies overlap the dominant embedding
    frequencies. Under independence two lists of m out of K would share
    ``m^2/K`` by chance (0.52 for m=5, K=48), which is the number this has to be
    read against.
    """
    return len(set(int(x) for x in a) & set(int(x) for x in b))


def random_ring_baseline(d_model: int, p: int, n_draws: int = 20,
                         seed: int = 0) -> dict[str, float]:
    """What the ring statistics read on embeddings with no structure at all.

    The appendix calls the memorization checkpoint's radial CV "diffuse" and
    the seed sweep can say how diffuse, but only against a measured floor. Two
    reasons this floor is not guessable:

    - The dominant frequency is chosen as the argmax over 48 of them, so the
      variance in "its" plane is inflated by the selection. A plane picked in
      advance would hold 2/d_model of the variance (1.6% at d_model=128);
      picked as the best of 48 it holds around 4% of pure Gaussian noise. Read
      against 1.6% the memorizing model looks structured; it is not.
    - The radial CV of a 2D Gaussian's radius has no reason to be 1.

    Returns the mean over draws of ``var_in_plane`` and ``radial_cv``.
    """
    rng = np.random.default_rng(seed)
    vs, cs = [], []
    for _ in range(n_draws):
        E = rng.normal(size=(p, d_model))
        _, _, v, c = frequency_projection(E, dominant_frequency(E, p), p)
        vs.append(v)
        cs.append(c)
    return {"var_in_plane": float(np.mean(vs)), "radial_cv": float(np.mean(cs)),
            "n_draws": float(n_draws)}


def dominant_frequency(E: np.ndarray, p: int) -> int:
    """Frequency (excluding DC) carrying the most embedding energy."""
    F = np.fft.rfft(E - E.mean(0, keepdims=True), axis=0)
    power = (F.real**2 + F.imag**2).sum(axis=1)
    return int(power[1:].argmax()) + 1


def frequency_projection(E: np.ndarray, k: int,
                         p: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Project centered embeddings onto the frequency-k (cos, sin) plane.

    Returns ``(x, y)`` scores, the fraction of total embedding variance captured
    by the plane, and the radial coefficient of variation (0 = perfect circle).
    """
    n = np.arange(p)
    c = np.cos(2 * np.pi * k * n / p)
    s = np.sin(2 * np.pi * k * n / p)
    Ec = E - E.mean(axis=0, keepdims=True)
    u = c @ Ec
    v = s @ Ec
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)
    x, y = Ec @ u, Ec @ v
    var_frac = float((x @ x + y @ y) / (Ec * Ec).sum())
    r = np.sqrt(x**2 + y**2)
    return x, y, var_frac, float(r.std() / r.mean())


# -- logits ------------------------------------------------------------------

def logit_tensor(model: Transformer,
                 p: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Logits at the "=" position for every ordered pair, shaped ``[a, b, c]``.

    The dataset is all ``p^2`` pairs in row-major (a outer, b inner) order, so a
    reshape to ``(p, p, p)`` indexes cleanly as ``L[a, b, c]``.
    """
    tokens, targets = modular_addition_dataset(p)
    with torch.no_grad():
        logits = model(tokens)[:, -1, :]        # (p^2, p)
    return logits.reshape(p, p, p), tokens, targets


def diagonal_frequency_energy(L: torch.Tensor,
                              p: int) -> tuple[torch.Tensor, float]:
    """Per-frequency energy of the logits on the ``k_a = k_b`` (a+b) diagonal.

    Returns ``(diag, diag_fraction)`` where ``diag[k]`` is the squared-magnitude
    energy of the 2D DFT of the (input-mean-removed) logits at ``(k, k)`` plus
    its conjugate partner ``(p-k, p-k)``, summed over the answer axis, for
    ``k = 0 .. (p-1)/2``; ``diag_fraction`` is the share of all non-DC logit
    energy that lives on this diagonal (i.e. is explained by ``a + b``).
    """
    Lc = L - L.mean(dim=(0, 1), keepdim=True)   # drop the constant-in-(a,b) part
    F = torch.fft.fft2(Lc, dim=(0, 1))          # (p, p, p) complex
    E = F.abs().pow(2)
    K = (p - 1) // 2
    diag = torch.zeros(K + 1)
    for k in range(1, K + 1):
        diag[k] = E[k, k, :].sum() + E[p - k, p - k, :].sum()
    non_dc_total = E.sum() - E[0, 0, :].sum()
    diag_fraction = float(diag[1:].sum() / non_dc_total)
    return diag, diag_fraction


def restrict_to_freqs(L: torch.Tensor, p: int,
                      keep_ks: Sequence[int]) -> torch.Tensor:
    """Rebuild the logits keeping only the a+b structure at ``keep_ks``.

    Keeps the input-mean (the constant-in-(a,b) part) plus, for each ``k`` in
    ``keep_ks``, the diagonal modes ``(k, k)`` and ``(p-k, p-k)``; every other
    2D-DFT coefficient is zeroed before the inverse transform. The result is the
    logits as they would be if the model computed *only* those frequencies of
    ``a + b`` -- the restricted-accuracy control.
    """
    mean = L.mean(dim=(0, 1), keepdim=True)
    F = torch.fft.fft2(L - mean, dim=(0, 1))
    mask = torch.zeros(p, p, dtype=torch.bool)
    for k in keep_ks:
        mask[k, k] = True
        mask[p - k, p - k] = True
    Fm = F * mask[:, :, None]
    return torch.fft.ifft2(Fm, dim=(0, 1)).real + mean


def test_accuracy(L: torch.Tensor, tokens: torch.Tensor, targets: torch.Tensor,
                  train_frac: float, seed: int) -> float:
    """Test-split accuracy of an argmax read-out of a logit tensor ``L``."""
    (_, _), (te_tok, te_tgt) = train_test_split(tokens, targets, train_frac, seed)
    a, b = te_tok[:, 0], te_tok[:, 1]
    pred = L[a, b, :].argmax(dim=-1)
    return float((pred == te_tgt).float().mean())


def first_m_reaching(ms: Sequence[int], accs: Sequence[float],
                     threshold: float) -> float:
    """Smallest ``m`` whose restricted accuracy reaches ``threshold``.

    ``nan`` when no ``m`` on the grid does, which is a different statement from
    "a large number": the memorization checkpoints never reach 0.99 at any
    ``m``, and quoting the grid's last value there would read as though ten
    frequencies had sufficed.
    """
    for m, a in zip(ms, accs):
        if a >= threshold:
            return float(m)
    return float("nan")


# -- everything, on one checkpoint -------------------------------------------

def measure_checkpoint(model: Transformer, p: int, train_frac: float,
                       split_seed: int) -> dict[str, float]:
    """Every mechanistic read-out of one checkpoint, from one set of weights.

    ``train_frac`` and ``split_seed`` are the run's own, so the restricted
    accuracies are measured on the split that run was actually held out from.
    """
    tokens, _ = modular_addition_dataset(p)

    E = model.tok_emb.weight.detach().cpu().numpy()[:p]
    spec = embedding_spectrum(model, p)
    emb_energy = spec[1:].pow(2)
    k_dom = dominant_frequency(E, p)
    _, _, var_in_plane, radial_cv = frequency_projection(E, k_dom, p)

    L, toks, targets = logit_tensor(model, p)
    diag, diag_frac = diagonal_frequency_energy(L, p)
    order = top_indices(diag[1:], m=(p - 1) // 2, offset=1)
    accs = [test_accuracy(restrict_to_freqs(L, p, order[:m]), toks, targets,
                          train_frac, split_seed) for m in RESTRICT_MS]

    out: dict[str, float] = {
        "emb_top5_energy": top_k_energy_fraction(spec, 5),
        "emb_freqs_90": float(n_freqs_to_reach(emb_energy, 0.9)),
        "emb_dominant_k": float(k_dom),
        "emb_var_in_plane": var_in_plane,
        "emb_radial_cv": radial_cv,
        "logit_diag_energy": diag_frac,
        "logit_diag_freqs_90": float(n_freqs_to_reach(diag[1:], 0.9)),
        "logit_acc_full": test_accuracy(L, toks, targets, train_frac, split_seed),
        "logit_freqs_for_99": first_m_reaching(RESTRICT_MS, accs, 0.99),
        "emb_diag_overlap": float(overlap(top_indices(emb_energy, 5, offset=1),
                                          order[:5])),
    }
    out.update({f"logit_acc_m{m}": a for m, a in zip(RESTRICT_MS, accs)})
    out.update(measure_attention(model, tokens))   # already ``attn_``-prefixed
    eq = measure_equivariance(model, tokens, p)
    out.update({
        "eq_attn_defect": eq["attn_equivariance_defect"],
        "eq_logit_swap": eq["logit_swap_defect"],
        "eq_logit_anti": eq["logit_anti_defect"],
        "eq_shuffle_baseline": eq["shuffle_baseline"],
    })
    return out


def _to_numpy(x: Any) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
