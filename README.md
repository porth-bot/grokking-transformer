# grokking-transformer

![ci](https://github.com/porth-bot/grokking-transformer/actions/workflows/ci.yml/badge.svg)

A decoder-only transformer implemented from scratch (the attention arithmetic
is written out and tested against PyTorch's fused reference) and used to
reproduce and dissect **grokking**: on modular addition, the model reaches
100% *training* accuracy at step 100 and stays near ~20% *test* accuracy for
~1,300 steps (median over 5 seeds) — then jumps to 100%. The repo measures
what controls the delay
(weight decay, data fraction) and inspects what changes inside the network
(weight norm, the Fourier structure of its embeddings) when it finally
generalizes.

![grokking](figures/grokking_main.png)

## Problem

Train a 223k-parameter transformer on 30% of all pairs $(a, b)$ to predict
$(a + b) \bmod 97$, supervised at the "=" position of the sequence
$[a, b, =]$. The dataset is noiseless and exhaustive, so test accuracy has an
unambiguous meaning: either the network recovered *the algorithm*, or it
memorized. With ~2.8k training examples against 223k parameters,
memorization is easy — the scientific question is why the network ever
prefers the general solution, and what schedule it finds it on.

The theory ([`theory/notes.md`](theory/notes.md)) covers the attention
derivation, the frequency-space algorithm for modular addition (via the DFT
delta identity $\sum_{k=0}^{p-1} \cos(2\pi k n / p) = p\,\delta_{n \equiv 0}$
and the angle-addition identities), and the norm/efficiency account of *why*
generalization is delayed rather than absent.

## What's implemented

| Piece | Where | Verified how |
|---|---|---|
| Causal multi-head attention, by hand | [`grokking/model.py`](grokking/model.py) | equal to `F.scaled_dot_product_attention` given the same weights; zero attention mass on the future; changing a future token provably cannot change past logits |
| LayerNorm, by hand | [`grokking/model.py`](grokking/model.py) | equal to `F.layer_norm` |
| Modular-addition dataset + splits | [`grokking/data.py`](grokking/data.py) | exhaustiveness, label correctness, disjoint & deterministic splits |
| Full-batch AdamW harness | [`grokking/train.py`](grokking/train.py) | end-to-end memorization sanity run on CPU |
| Sweeps / plots / Fourier analysis | [`experiments/`](experiments/) | all figures regenerate from committed CSV logs |

Design choices that matter for the science: **full batch** (no minibatch
noise confound), **AdamW's decoupled decay** (the regularizer under study —
L2-through-Adam is a different object), **no dropout by default** (so weight
decay is the only regularizer in the main runs — though dropout is an
available knob, used only for the regularizer control in §6), and **two
checkpoints per run** (memorization point
and final) so "before vs after" is a comparison within a single trajectory.

## Results

All runs: $p = 97$, 1 layer, $d_{\text{model}} = 128$, 4 heads, lr $10^{-3}$,
full-batch AdamW. The weight-decay and data-fraction sweeps below run **5 seeds
per cell** and report the median with the min–max range; the mechanistic
single-run analyses (§3–6 and the Fourier/attention/embedding read-outs) stay
on seed 0, which is what the hero figure shows. Logs in [`runs/`](runs/),
regenerate figures with `python experiments/plots.py`.

### 1. Weight decay controls whether — and when — grokking happens

30% training data, three values of weight decay (median grok step over 5 seeds,
`[min–max]`; memorization is at step 100 in every seed):

| weight decay | memorized (100% train) | grokked (99% test), median [range] | delay |
|---|---|---|---|
| 0.0 | step 100 | **never** (25k budget, all 5 seeds) | ∞ |
| 0.1 | step 100 | 10,800 [7,600–13,900] | 108× |
| 1.0 | step 100 | 1,300 [1,200–1,900] | 13× |

![wd sweep](figures/wd_sweep.png)

The wd = 0 control memorizes identically fast, then stays memorized — no seed
transitions within budget (final test accuracy 0.29–0.42 across the five, some
implicit regularization but no grok). The seed spread never comes close to
closing the gap between the three cells: even the slowest wd = 1 seed (1,900)
groks before the *fastest* wd = 0.1 seed (7,600), so weight decay's ordering is
not a seed artifact. This is the cleanest evidence in the repo that the delayed
generalization is *driven by the regularizer*, not by more gradient steps on
the task loss: after step ~100 the training loss
is nearly zero and almost all subsequent change in test accuracy is the
norm-pressure term reorganizing the network's internals.

### 2. Less data, longer trance

Weight decay 1.0, four training fractions (median grok step over 5 seeds,
`[min–max]`; 60% is a single-seed context point):

| train fraction | grokked at step, median [range] | delay over memorization |
|---|---|---|
| 25% | 2,700 [2,000–3,100] | 27× |
| 30% | 1,300 [1,200–1,900] | 13× |
| 40% | 300 [300–700] | 3× |
| 60% | 200 (1 seed) | 2× |

![frac sweep](figures/frac_sweep.png)

Monotone in the medians, roughly log-linear: as the training set shrinks,
memorization gets
relatively cheaper (fewer pairs to store) while the general circuit's cost is
fixed — so the phase in which memorization dominates stretches. At 60% data
the "delay" nearly vanishes and grokking degenerates into ordinary learning;
grokking is a *small-data* phenomenon.

### 3. Robustness: grokking survives a 10× learning-rate change

Is the grok time an artifact of one tuned learning rate? Rerunning the main
config (30%, wd = 1, seed 0) at lr spanning an order of magnitude
([`lr_sweep.py`](experiments/lr_sweep.py)):

| lr | memorized at | grokked at | delay |
|---|---|---|---|
| 3e-4 | 200 | 5,500 | 27× |
| 1e-3 | 100 | 1,700 | 17× |
| 3e-3 | 100 | 800 | 8× |

![lr sweep](figures/lr_sweep.png)

The phenomenon is robust — the network memorizes fast and generalizes late at
every learning rate — but the grok *step* is not a physical constant: it
scales roughly inversely with lr (a 10× larger lr groks ~7× sooner), because
the grok step counts optimizer steps, and a larger step covers more of the
same path per iteration. Memorization is already near-instant at all three
lrs, so the delay multiple shrinks as lr grows while never vanishing. The
takeaway for the rest of this repo: grok steps are only comparable **at fixed
lr** (all other sweeps here hold lr = 1e-3), and "1,900 steps" is a property
of the optimizer schedule, not just the task.

### 4. Does the delay grow with the modulus? (No — data size wins)

Every run above uses $p = 97$. Repeating the main configuration (30%,
wd = 1, seed 0, same lr) at a larger prime $p = 113$ changes two things at
once: more residue classes and Fourier frequencies for the circuit to
represent (harder), but 30% of the larger $p^2$ grid is more absolute
training pairs (easier). [`modulus_scaling.py`](experiments/modulus_scaling.py):

| $p$ | train pairs (30%) | memorized at | grokked at | delay |
|---|---|---|---|---|
| 97 | 2,823 | step 100 | step 1,900 | 19× |
| 113 | 3,831 | step 100 | **step 600** | 6× |

The larger modulus groks **sooner**, not later: memorization is instant in
both, but generalization arrives 3× earlier at $p = 113$. The absolute
training-set size dominates — this is the same lever as §2 (grokking is a
small-data phenomenon), and 3,831 pairs sit further from the critical
fraction than 2,823 do. The transition is also softer at $p = 113$: test
accuracy is already 29% at the memorization point and climbs steadily, rather
than sitting near chance through a long plateau. So "time-to-grok" is not a
clean increasing function of $p$; at fixed data *fraction*, the data-quantity
effect wins on this axis. (One seed, one extra modulus — a direction, not a
scaling law.)

### 5. What changes inside: norm and Fourier structure

Two measurements on the main run (30%, wd = 1), same seed, same trajectory:

- **Weight norm** (right panel of the hero figure): rises while the
  loss-gradient dominates, peaks around the transition, then falls once
  train loss is pinned at ~0 and decay is the only force left. (Our first
  version of this run early-stopped 500 steps after grokking and *missed*
  the decline — the run was extended to 11k steps precisely so the plot
  shows the dynamics rather than an artifact of the stopping rule.)
- **Embedding Fourier spectrum** — the algorithm's fingerprint. At the
  memorization checkpoint, spectral energy is spread across all 48
  frequencies (top-5 share: **13.6%**, indistinguishable from unstructured).
  At the final checkpoint, five frequencies ($k = 5, 14, 20, 36, 37$)
  dominate with a top-5 share of **56.7%**:

![fourier](figures/fourier_spectrum.png)

Consistent with Nanda et al.'s progress-measures picture: the general
circuit is sparse in frequency space, and it keeps *consolidating after*
the accuracy jump (our early-stopped checkpoint showed 40%; 3k steps later,
57%) — the "sudden" jump is a thresholding artifact of accuracy, not a
discontinuity in the weights.

### 6. Is it the norm specifically, or any regularizer? (A dropout control)

Section 5 shows weight decay generalizing by pulling the weight norm down, and
the Omnigrok picture (Liu et al. 2023) makes *norm reduction* the mechanism.
That invites a control: swap weight decay for **dropout** — a regularizer that
does not target the norm at all — holding frac = 0.30, seed, and lr fixed
(**dropout 0.1, weight decay 0**).

| regularizer | memorized | grokked | final test acc |
|---|---|---|---|
| none (wd 0) | step 100 | never | 0.29 |
| **dropout 0.1** (wd 0) | step 200 | **step 3500** | **0.999** |
| weight decay 1.0 | step 100 | step 1900 | 1.00 |

![dropout control](figures/dropout_control.png)

Dropout groks — so it is **not** weight decay specifically that is required.
But the mechanism is visibly different: weight decay generalizes while driving
the norm *down* (section 5), whereas under dropout the norm **rises
monotonically** the entire time (21 → 55) and the model generalizes anyway.
Norm reduction is therefore *sufficient but not necessary* here; a regularizer
that instead penalizes co-adapted, memorization-friendly features reaches the
same generalizing circuit by a different route. What the two share — and what
the unregularized run lacks — is simply *pressure against the pure-memorization
solution*, not a particular way of applying it.

### 7. *Where* does the norm pressure need to be? (Not the embeddings)

Section 6 says pressure against memorization is what matters. But "the weight
norm" is the norm of *every* parameter — so does the decay need to act on the
**embeddings** (the token/position tables, where the Fourier structure lives),
on the **rest** of the network (attention + MLP + unembed, which read that
structure out), or on both at once? This ablation holds the main config fixed
(frac 0.30, wd 1.0, seed 0) and changes only *which* parameters weight decay is
applied to; the untargeted group trains at wd 0.

| weight-decay scope | memorized | grokked (99% test) | final test |
|---|---|---|---|
| decay everything (main) | step 100 | step 1900 | 1.00 |
| decay **non-embeddings only** | step 100 | **step 1800** | 1.00 |
| decay **embeddings only** | step 100 | **never** (15k steps) | 0.36 |

![weight-decay scope](figures/wd_scope.png)

The pressure that matters is on the **non-embedding** weights. Decaying them
alone reproduces the full-decay run almost exactly (grok step 1800 vs 1900, the
two curves overlap). Decaying only the embeddings does essentially nothing: the
model never groks in 15k steps, because the rest of the network — now
unconstrained — keeps its large memorization weights, and the total norm climbs
without bound ($\|\theta\|$ balloons from 21 to **287**, the green curve, while
both grokking runs hold it near 20–40). Weight decay drives grokking by
shrinking the *readout* circuit's parameters; pinning the embeddings' norm is
neither sufficient nor the operative lever. (The embeddings supply the Fourier
basis, but their *scale* is not what memorization exploits.)

### Appendix: attention and embedding geometry

The same before/after story is visible in two more read-outs of the
committed checkpoints (both regenerated by `reproduce_figures.py`, no
retraining):

- **Attention pattern** ([`attention_pattern.py`](experiments/attention_pattern.py)).
  The "=" token — where the answer is written — spends ~all of its attention
  on the two operand positions `a` and `b` in *both* checkpoints (it has
  nothing else to read, and the causal mask forbids looking ahead). What
  grokking changes is the *symmetry*: the grokked heads split their operand
  attention almost exactly evenly (per-head $|A_{=\to a} - A_{=\to b}|$ falls
  from **0.19** to **0.00**), matching the commutativity $a + b = b + a$ that
  the general algorithm must respect, whereas the memorizing heads are
  lopsided (one puts 0.74 on `a`, 0.25 on `b`).

  ![attention](figures/attention_pattern.png)

- **Embedding ring** ([`embedding_circle.py`](experiments/embedding_circle.py)).
  Projected onto the dominant frequency's (cos, sin) plane, the grokked digit
  embeddings trace a clean circle (radial CV 0.13, up from a diffuse 0.41 at
  memorization) — the geometric face of the Fourier sparsification above.

  ![embedding ring](figures/embedding_circle.png)

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
pytest                              # 22 tests
python experiments/run_sweep.py     # 26 runs (5 seeds x 5 cells + 1), ~2 h on Apple Silicon (MPS) — resumable
python experiments/plots.py         # figures from committed CSVs (no training needed)
python experiments/fourier.py       # needs the checkpoints from run_sweep.py
python experiments/dropout_control.py  # §6 regularizer control (~4 min: one run)
python experiments/wd_scope.py         # §7 weight-decay scope ablation (2 runs; reuses the main baseline)
python experiments/reproduce_figures.py  # every figure from committed logs, no training
```

Committed CSV logs mean the figures are reproducible without retraining;
checkpoints (`runs/*.pt`) are gitignored.

## Honest limitations

- **Five seeds, not a distribution.** The wd and frac sweeps now carry
  min–max ranges over 5 seeds (§1–2), enough to show the between-cell gaps
  survive seed noise but too few to trust the range as a real spread — treat
  it as a rough error bar, not a confidence interval. The *mechanistic*
  read-outs (Fourier spectrum, attention, embedding ring, §5 and appendix)
  are still single-run (seed 0); their qualitative claims are not yet
  seed-averaged.
- **Architecture differs from Nanda et al.** (we use LayerNorm + GELU;
  their interp model was LN-free ReLU), which is likely part of why our
  final spectrum is sparse-but-not-extremely-sparse rather than >90%
  concentrated. Training far past the transition sharpens it.
- **Thresholds are conventions** (99.9% "memorized", 99% "grokked"); the
  underlying weight-space transition is gradual.

## Next

- wd × frac interaction surface (a coarse 2D grid); seed-averaged versions of
  the mechanistic read-outs.
- Progress measures *during* training (restricted/excluded loss à la Nanda)
  rather than two-checkpoint snapshots.
- Other operations: subtraction and multiplication grok; division's
  structure differs — a natural comparative study.

## References

Power et al. (2022) arXiv:2201.02177 (grokking); Nanda et al. (2023) ICLR,
arXiv:2301.05217 (Fourier circuit, progress measures); Liu et al. (2023)
"Omnigrok", ICLR (norm dynamics); Varma et al. (2023) arXiv:2309.02390
(circuit efficiency); Vaswani et al. (2017) (transformer); Loshchilov &
Hutter (2019) (AdamW). Roles and derivations in
[`theory/notes.md`](theory/notes.md).

## Provenance

Built as a study resource: implemented from scratch with AI assistance
(Claude), with the theory written out in [`theory/notes.md`](theory/notes.md)
and every structural claim about the implementation pinned by a test.
MIT license.

*Suggested GitHub topics:* `grokking` `transformer` `mechanistic-interpretability`
`deep-learning` `pytorch` `from-scratch` `attention`
