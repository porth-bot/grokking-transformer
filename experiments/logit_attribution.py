"""Per-frequency logit attribution: the grokked model's *output* is a sparse
sum over a few frequencies (a preview of Day 19's progress measures).

``experiments/fourier.py`` shows grokking sparsifies the *embeddings*. This
asks the same question one step downstream, at the logits the model actually
scores answers with. The generalizing algorithm (Nanda et al. 2023) computes

    logit(a, b, c)  ~=  sum_k  A_k cos( w_k (a + b - c) ),     w_k = 2 pi k / p,

so two things should be true of a grokked model's logits and *not* of a
memorizing one:

1. **They depend on ``a + b``.** Fourier-transform the logit tensor
   ``L[a, b, c]`` over the two input axes: a function of ``a + b`` has all its
   energy on the diagonal ``k_a = k_b``. We report the fraction of (non-DC)
   logit energy sitting on that diagonal -- how much of what the model does is
   "compute the sum" versus per-pair lookup.

2. **A handful of frequencies suffice.** Keep only the top-``m`` diagonal
   frequencies, inverse-transform to rebuild the logits, and measure test
   accuracy. This is a hand-built *restricted accuracy* -- the same idea as
   Nanda's restricted loss (Day 19), here as a static read of two checkpoints.

The striking read-out (committed p=97 run): the final model puts ~98% of its
logit energy on the ``a + b`` diagonal and just its top *three* frequencies
rebuild 100% test accuracy; the memorization checkpoint is diffuse (~12% on the
diagonal). And projecting the *memorization* logits onto the clean ``a + b``
subspace already recovers far more test accuracy than the raw model expresses --
the generalizing circuit is forming under the memorization, before the test-acc
jump. ``tests/test_logit_attribution.py`` pins these against the checkpoints.

Run:  python experiments/logit_attribution.py   (after run_sweep.py)
"""

import numpy as np
import torch

from pathlib import Path

from grokking.checkpoints import load_model
from grokking.data import modular_addition_dataset, train_test_split

# matplotlib is imported lazily inside _figure() (it is an experiments/dev dep,
# absent from the numpy+torch CI job) so that tests can reuse the analysis
# helpers below without pulling in a plotting dependency.

ROOT = Path(__file__).resolve().parent.parent
MAIN = "p97_frac0.30_wd1_seed0"


def logit_tensor(model, p):
    """Logits at the "=" position for every ordered pair, shaped ``[a, b, c]``.

    The dataset is all ``p^2`` pairs in row-major (a outer, b inner) order, so a
    reshape to ``(p, p, p)`` indexes cleanly as ``L[a, b, c]``.
    """
    tokens, targets = modular_addition_dataset(p)
    with torch.no_grad():
        logits = model(tokens)[:, -1, :]        # (p^2, p)
    return logits.reshape(p, p, p), tokens, targets


def diagonal_frequency_energy(L, p):
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


def restrict_to_freqs(L, p, keep_ks):
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


def test_accuracy(L, tokens, targets, train_frac, seed):
    """Test-split accuracy of an argmax read-out of a logit tensor ``L``."""
    (_, _), (te_tok, te_tgt) = train_test_split(tokens, targets, train_frac, seed)
    a, b = te_tok[:, 0], te_tok[:, 1]
    pred = L[a, b, :].argmax(dim=-1)
    return float((pred == te_tgt).float().mean())


def main():
    mem, summary = load_model(MAIN, which="memorize")
    fin, _ = load_model(MAIN, which="final")
    p = summary["config"]["p"]
    frac = summary["config"]["train_frac"]
    seed = summary["config"]["seed"]

    out = {}
    for tag, model in (("memorize", mem), ("final", fin)):
        L, tokens, targets = logit_tensor(model, p)
        diag, diag_frac = diagonal_frequency_energy(L, p)
        order = [int(k) for k in torch.argsort(diag, descending=True) if k > 0]
        ms = [1, 2, 3, 4, 5, 6, 8, 10]
        accs = [test_accuracy(restrict_to_freqs(L, p, order[:m]),
                              tokens, targets, frac, seed) for m in ms]
        full = test_accuracy(L, tokens, targets, frac, seed)
        out[tag] = dict(diag=diag, diag_frac=diag_frac, order=order,
                        ms=ms, accs=accs, full=full)
        print(f"{tag:9s}: {diag_frac:.1%} of non-DC logit energy on the a+b "
              f"diagonal; full test acc {full:.3f}; top-3 freqs "
              f"{sorted(order[:3])} -> restricted acc "
              f"{test_accuracy(restrict_to_freqs(L, p, order[:3]), tokens, targets, frac, seed):.3f}")

    _figure(out, p)


def _figure(out, p):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
            "axes.titlesize": 10, "axes.labelsize": 9,
            "axes.spines.top": False, "axes.spines.right": False,
            "legend.frameon": False,
        }
    )

    fin, mem = out["final"], out["memorize"]
    K = (p - 1) // 2
    ks = np.arange(1, K + 1)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3), constrained_layout=True)

    # Left: per-frequency diagonal logit energy (share of diagonal energy).
    fin_share = (fin["diag"][1:] / fin["diag"][1:].sum()).numpy()
    mem_share = (mem["diag"][1:] / mem["diag"][1:].sum()).numpy()
    ax = axes[0]
    ax.bar(ks, fin_share, width=0.9, color="C0", label="after grokking")
    ax.plot(ks, mem_share, color="C3", lw=1.0, alpha=0.8, label="at memorization")
    ax.set_xlabel("frequency $k$ (on the $a{+}b$ diagonal)")
    ax.set_ylabel("share of diagonal logit energy")
    ax.set_title(f"Per-frequency logit attribution\n"
                 f"on $a{{+}}b$ diagonal: {mem['diag_frac']:.0%} (mem) "
                 f"$\\to$ {fin['diag_frac']:.0%} (final)", loc="left", fontsize=9)
    ax.legend(fontsize=8)

    # Right: restricted test accuracy vs number of top diagonal freqs kept.
    ax = axes[1]
    ax.plot(fin["ms"], fin["accs"], "o-", color="C0", label="after grokking")
    ax.plot(mem["ms"], mem["accs"], "s-", color="C3", label="at memorization")
    ax.axhline(fin["full"], color="C0", ls=":", lw=1, alpha=0.7)
    ax.axhline(mem["full"], color="C3", ls=":", lw=1, alpha=0.7)
    ax.set_xlabel("number of top $a{+}b$ frequencies kept")
    ax.set_ylabel("test accuracy of restricted logits")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Restricted accuracy: a few frequencies\nrebuild the whole "
                 "generalizing solution", loc="left", fontsize=9)
    ax.legend(fontsize=8)

    fig.suptitle("The grokked model's logits are a sparse sum over $a{+}b$ "
                 "frequencies", y=1.08)
    fig.savefig(ROOT / "figures" / "logit_attribution.png", bbox_inches="tight")
    print("saved figures/logit_attribution.png")


if __name__ == "__main__":
    main()
