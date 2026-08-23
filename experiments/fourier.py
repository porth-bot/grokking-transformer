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

The statistics themselves live in ``grokking.mechanistic`` (this figure and
the seed sweep of Sec. 12 must measure the same thing, so they call the same
function); this script is the figure.

Run:  python experiments/fourier.py   (after run_sweep.py)
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pathlib import Path

from grokking.checkpoints import load_model
from grokking.mechanistic import embedding_spectrum, top_k_energy_fraction

from _style import apply_style  # noqa: E402

apply_style()

ROOT = Path(__file__).resolve().parent.parent
MAIN = "p97_frac0.30_wd1_seed0"


def main():
    model_mem, summary = load_model(MAIN, which="memorize")
    model_fin, _ = load_model(MAIN, which="final")
    p = summary["config"]["p"]
    spec_mem = embedding_spectrum(model_mem, p)
    spec_fin = embedding_spectrum(model_fin, p)

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
