"""What the "=" token attends to, before vs after grokking.

The model reads the sequence [a, b, =] and must write (a + b) mod p at the
final ("=") position. So the one attention row that carries the computation is
the "=" query: its distribution over {a, b, =} says which operands it pulls in
before the MLP combines them. This script reads that row out of the committed
checkpoints (no retraining) and averages it over all p^2 examples.

Two facts come out, and the second is the grokking story:

1. In BOTH the memorization-point and the grokked model, the "=" token spends
   almost all of its weight on the two operand positions a and b, and almost
   none on itself -- there is nothing at the "=" slot to read except the
   operands, and the causal mask forbids looking ahead anyway.

2. Grokking *symmetrizes* that read. Addition is commutative (a + b = b + a),
   so the algorithmic solution should treat the two operands interchangeably.
   The grokked model does: every head splits its "=" attention almost exactly
   evenly between a and b. The memorizing model does not -- its heads are
   lopsided (one may put 0.74 on a and 0.25 on b), a fingerprint of a solution
   that has keyed on particular (a, b) pairs rather than the symmetric rule.

We quantify (2) with the per-head operand asymmetry, mean_h |A[=->a] - A[=->b]|,
which collapses toward zero once the circuit forms. Attention weights come from
``CausalSelfAttention.attn_weights`` -- the exact softmax the forward pass uses.
Run:  python experiments/attention_pattern.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from grokking.checkpoints import load_model  # noqa: E402
from grokking.data import modular_addition_dataset  # noqa: E402

plt.rcParams.update(
    {
        "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
    }
)

ROOT = Path(__file__).resolve().parent.parent
MAIN = "p97_frac0.30_wd1_seed0"
POS_LABELS = ["a", "b", "="]


def eq_attention(model, tokens):
    """Mean over the dataset of the "=" query's attention, per head.

    Returns an (n_heads, seq_len) array: row h is head h's average attention
    from the final ("=") position to [a, b, =].
    """
    x = model.tok_emb(tokens) + model.pos_emb[: tokens.shape[1]]
    block = model.blocks[0]                       # the run is a 1-layer model
    att = block.attn.attn_weights(block.ln1(x))   # (B, H, T, T)
    return att[:, :, -1, :].mean(0).cpu().numpy()  # (H, T), "=" query row


def operand_stats(eq_row):
    """(operand fraction a+b, mean per-head |a - b| asymmetry) for an (H, 3) row."""
    operand_frac = float(eq_row[:, :2].sum(1).mean())        # weight on a and b
    asymmetry = float(np.abs(eq_row[:, 0] - eq_row[:, 1]).mean())
    return operand_frac, asymmetry


def main():
    model_mem, summary = load_model(MAIN, which="memorize")
    model_fin, _ = load_model(MAIN, which="final")
    p = summary["config"]["p"]
    tokens, _ = modular_addition_dataset(p)

    panels = [
        (eq_attention(model_mem, tokens), "at memorization (~1% test)"),
        (eq_attention(model_fin, tokens), "after grokking (~100% test)"),
    ]
    vmax = max(panel[0].max() for panel in panels)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for ax, (eq_row, title) in zip(axes, panels):
        H = eq_row.shape[0]
        frac, asym = operand_stats(eq_row)
        im = ax.imshow(eq_row, cmap="magma", vmin=0.0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(POS_LABELS)), POS_LABELS)
        ax.set_yticks(range(H), [f"head {h}" for h in range(H)])
        ax.set_xlabel('"=" attends to position')
        for h in range(H):
            for j in range(len(POS_LABELS)):
                val = eq_row[h, j]
                ax.text(j, h, f"{val:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if val < 0.6 * vmax else "black")
        ax.set_title(f"{title}\noperand weight {frac:.0%}, a/b asymmetry {asym:.2f}",
                     loc="left", fontsize=9)
        print(f"{title:28s}: operand_frac={frac:.3f}  asymmetry={asym:.3f}")
    fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02, label="attention weight")
    fig.suptitle(
        'Grokking symmetrizes the "=" token\'s read of the two operands',
        x=0.02, ha="left",
    )
    fig.savefig(ROOT / "figures" / "attention_pattern.png", bbox_inches="tight")
    print("saved figures/attention_pattern.png")


if __name__ == "__main__":
    main()
