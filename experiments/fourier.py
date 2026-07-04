"""Fourier analysis of the learned embeddings: what did grokking change?

The generalizing algorithm for modular addition (Nanda et al. 2023) works in
frequency space. If the embedding of digit token n contains components
cos(2 pi k n / p) and sin(2 pi k n / p) for a few frequencies k, downstream
layers can combine a and b via the angle-addition identity

    cos(w a)cos(w b) - sin(w a)sin(w b) = cos(w(a + b)),

and score answer c by accumulating cos(w(a + b - c)) over its frequencies --
maximized exactly at c = (a + b) mod p (see theory/notes.md, Sec. 3 for why).

That predicts a *measurable* signature: the digit-embedding matrix, Fourier-
transformed along the token axis, should be sparse -- energy concentrated in
a handful of frequencies -- once the model generalizes, and diffuse while it
is merely memorizing. This script tests that prediction by comparing the
checkpoint saved at the memorization point against the final checkpoint of
the same run.

Run:  python experiments/fourier.py   (after run_sweep.py)
"""

import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pathlib import Path

from grokking.model import ModelConfig, Transformer

plt.rcParams.update(
    {
        "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False,
    }
)

ROOT = Path(__file__).resolve().parent.parent
MAIN = "p97_frac0.30_wd1_seed0"
P = 97


def embedding_spectrum(state_path):
    """Per-frequency L2 norm of the digit embeddings.

    E is (p, d_model) -- the '=' token row is excluded, since only digit
    tokens participate in the periodic structure. rfft along the token axis
    gives coefficients for frequencies k = 0 .. (p-1)/2; we report
    ||F_k||_2 over the model dimension, and the DC term k=0 (the mean
    embedding) is dropped from sparsity statistics.
    """
    cfg = ModelConfig(p=P, vocab_size=P + 1)
    model = Transformer(cfg)
    model.load_state_dict(torch.load(state_path, map_location="cpu"))
    E = model.tok_emb.weight.detach()[:P]          # (p, d_model)
    F = torch.fft.rfft(E, dim=0)                   # (p//2 + 1, d_model), complex
    return F.abs().pow(2).sum(dim=1).sqrt()        # (p//2 + 1,)


def top_k_energy_fraction(spec, k=5):
    """Fraction of squared spectral norm captured by the top-k frequencies
    (excluding DC)."""
    energy = spec[1:].pow(2)
    top = energy.sort(descending=True).values[:k].sum()
    return float(top / energy.sum())


def main():
    spec_mem = embedding_spectrum(ROOT / "runs" / f"{MAIN}_memorize.pt")
    spec_fin = embedding_spectrum(ROOT / "runs" / f"{MAIN}.pt")

    frac_mem = top_k_energy_fraction(spec_mem)
    frac_fin = top_k_energy_fraction(spec_fin)
    dominant = (spec_fin[1:].argsort(descending=True)[:5] + 1).tolist()
    print(f"top-5 frequency energy fraction  memorization: {frac_mem:.3f}   final: {frac_fin:.3f}")
    print(f"dominant frequencies (final): k = {sorted(dominant)}")

    ks = range(len(spec_mem))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True, constrained_layout=True)
    for ax, spec, title, frac in [
        (axes[0], spec_mem, "at memorization (100% train, ~1% test)", frac_mem),
        (axes[1], spec_fin, "after grokking (100% train, ~100% test)", frac_fin),
    ]:
        ax.bar(ks, spec, width=0.8)
        ax.set_xlabel("frequency $k$")
        ax.set_title(f"{title}\ntop-5 energy: {frac:.0%}", loc="left", fontsize=9)
    axes[0].set_ylabel(r"$\|\hat E_k\|_2$")
    fig.suptitle("Embedding Fourier spectrum: grokking = discovering sparse structure",
                 y=1.06)
    fig.savefig(ROOT / "figures" / "fourier_spectrum.png", bbox_inches="tight")
    print("saved figures/fourier_spectrum.png")


if __name__ == "__main__":
    main()
