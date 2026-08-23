"""Embedding geometry before vs after grokking: the digit ring.

Companion to fourier.py. The Fourier view shows the digit embeddings become
spectrally sparse when the model groks; this shows the same fact geometrically.
For a frequency k the generalizing solution wants
E[n] ~ u_k cos(2 pi k n / p) + v_k sin(2 pi k n / p) with fixed d_model vectors
u_k, v_k, which places token n on a circle at angle 2 pi k n / p. Projecting the
embeddings onto that 2D (cos, sin) subspace should therefore trace a ring once
the circuit forms, and nothing before it.

Why a *frequency* projection and not a plain PCA of the full embedding: the
grokked embedding is a superposition of several frequencies (here k in
{5,14,20,36,37}), each occupying its own ~2D subspace, so no single PCA plane is
a clean circle -- it mixes them. Isolating one frequency's plane is what makes
the ring legible. We report, for each checkpoint, the share of embedding
variance living in that plane and the radial spread of the points.

Points are coloured by the residue class r = (k n) mod p, which is exactly the
token's angular position around the ring (gcd(k, p) = 1, so n -> k n mod p is a
bijection) -- the colour then winds smoothly around the circle. Both checkpoints
load through grokking.checkpoints; no retraining.

The projection statistics live in ``grokking.mechanistic``, shared with the
seed sweep of Sec. 12.

Run:  python experiments/embedding_circle.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from grokking.checkpoints import load_model
from grokking.mechanistic import dominant_frequency, frequency_projection

from _style import apply_style  # noqa: E402

apply_style()

ROOT = Path(__file__).resolve().parent.parent
MAIN = "p97_frac0.30_wd1_seed0"


def main():
    model_mem, summary = load_model(MAIN, which="memorize")
    model_fin, _ = load_model(MAIN, which="final")
    p = summary["config"]["p"]
    E_mem = model_mem.tok_emb.weight.detach().cpu().numpy()[:p]
    E_fin = model_fin.tok_emb.weight.detach().cpu().numpy()[:p]

    k = dominant_frequency(E_fin, p)
    residue = (k * np.arange(p)) % p
    print(f"dominant embedding frequency (final): k = {k}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.1), constrained_layout=True)
    for ax, E, title in [
        (axes[0], E_mem, "at memorization (~1% test)"),
        (axes[1], E_fin, "after grokking (~100% test)"),
    ]:
        x, y, var_frac, r_cv = frequency_projection(E, k, p)
        sc = ax.scatter(x, y, c=residue, cmap="twilight", s=24, edgecolors="none")
        ax.set_xlabel(rf"$\cos$-direction of $k={k}$")
        ax.set_ylabel(rf"$\sin$-direction of $k={k}$")
        ax.set_aspect("equal", "datalim")
        ax.set_title(
            f"{title}\nvariance in plane {var_frac:.0%}, radial CV {r_cv:.2f}",
            loc="left", fontsize=9,
        )
        print(f"  {title:28s}: var_in_plane={var_frac:.3f}  radial_cv={r_cv:.3f}")
    cbar = fig.colorbar(sc, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label(r"residue $(k\,n)\ \mathrm{mod}\ p$")
    fig.suptitle(
        f"Grokking arranges the digit embeddings on a circle (frequency $k={k}$)",
        x=0.02, ha="left",
    )
    fig.savefig(ROOT / "figures" / "embedding_circle.png", bbox_inches="tight")
    print("saved figures/embedding_circle.png")


if __name__ == "__main__":
    main()
