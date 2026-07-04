# Theory notes

What the model is, why the task is the right probe, the algorithm the
network is believed to learn, and what each experiment is designed to test.

## 1. Task

Learn $(a + b) \bmod p$ for $p = 97$ from the token sequence $[a, b, =]$,
supervised at the "=" position. The dataset is the full universe of $p^2 = 9409$
ordered pairs; training sees a random fraction $f$, and the test set is the
exact complement. Three properties make this the canonical grokking probe:

1. **Noiseless and exhaustive** — generalization can only mean *recovering
   the algorithm*, not interpolating noise. Chance is $1/p \approx 1\%$.
2. **A capacity gap that permits memorization** — our model has ~420k
   parameters against ~2.8k training examples at $f = 0.3$: memorizing is
   easy, so the interesting question is what makes the network ever prefer
   the general solution.
3. **Known algorithmic structure** (Sec. 3), so "did it generalize?" can be
   asked mechanically, by inspecting weights, not just behaviorally.

## 2. Model

Decoder-only transformer, 1 layer, $d_{\text{model}} = 128$, 4 heads,
$d_{\text{mlp}} = 512$, learned positional embeddings, pre-LayerNorm
residual blocks, no dropout.

**Attention.** With $Q = XW_Q$, $K = XW_K$, $V = XW_V$ per head:

$$\text{Attn}(X) = \text{softmax}\!\Big(\frac{QK^\top}{\sqrt{d_{\text{head}}}} + M\Big) V,$$

where $M_{ij} = -\infty$ for $j > i$ (causal mask) and $0$ otherwise. The
$\sqrt{d_{\text{head}}}$ keeps pre-softmax logits at unit variance at init:
for $q, k$ with i.i.d. zero-mean unit-variance entries,
$\operatorname{Var}(q \cdot k) = \sum_{i=1}^{d_{\text{head}}} \operatorname{Var}(q_i k_i) = d_{\text{head}}$,
and softmax saturates (vanishing gradients through it) when its inputs are
large. Multi-head = running $H$ attentions on $d_{\text{head}}$-dimensional
projections and concatenating; heads can attend to different token relations
in parallel. Masked positions receive exactly zero weight and each row of
the attention matrix remains a probability distribution over the visible
prefix (asserted in tests).

**Readout.** Logits over the $p$ possible answers come from the final
residual stream at the "=" position through LayerNorm and an unembedding
$W_U \in \mathbb{R}^{d_{\text{model}} \times p}$. Loss is cross-entropy at
that position only.

## 3. The Fourier multiplication algorithm

The clean way to compute $(a + b) \bmod p$ with continuous machinery is in
frequency space (Nanda et al. 2023). Fix a frequency $k$ and let
$\omega = 2\pi k / p$. If token embeddings encode
$\cos(\omega a), \sin(\omega a)$ (a 2-dimensional subspace per frequency),
then the angle-addition identities

$$\cos(\omega(a+b)) = \cos\omega a \cos\omega b - \sin\omega a \sin\omega b,
\qquad
\sin(\omega(a+b)) = \sin\omega a\cos\omega b + \cos\omega a \sin\omega b$$

turn *addition of tokens* into *multiplication of features* — exactly what
attention (bilinear in the residual stream) and the MLP nonlinearity can
implement. Scoring answer $c$ by

$$\text{logit}(c) \;\propto\; \sum_{k \in K} \cos\!\big(\tfrac{2\pi k}{p}(a + b - c)\big)$$

is maximized at $c \equiv a + b \pmod p$. With **all** frequencies this is
the DFT delta identity — the geometric series
$\sum_{k=0}^{p-1} e^{2\pi i k n / p}$ sums to $p$ when $n \equiv 0 \pmod p$
and to $\frac{1 - e^{2\pi i n}}{1 - e^{2\pi i n/p}} = 0$ otherwise, so

$$\sum_{k=0}^{p-1} \cos\!\big(\tfrac{2\pi k n}{p}\big) = p\,\delta_{n \equiv 0 \bmod p}.$$

A *sparse* subset $K$ of frequencies keeps the peak at $|K|$ while off-target
values, sums of cosines at distinct nonzero phases, stay well below it —
sufficient for argmax. Trained networks empirically use $|K| \approx 4$–$6$.

**Testable signature.** Take the digit-embedding matrix
$E \in \mathbb{R}^{p \times d}$ and Fourier-transform along the token axis.
If the network implements the algorithm, $\|\hat E_k\|$ should be
concentrated on a few frequencies after generalization and diffuse during
pure memorization. `experiments/fourier.py` compares exactly this between
the memorization-point checkpoint and the final checkpoint of the same run.

## 4. Grokking: the phenomenon and the competing stories

**The phenomenon** (Power et al. 2022): with a small training fraction and
weight decay, train accuracy reaches 100% early while test accuracy sits at
chance for thousands of further steps — then jumps to ~100%. The gap between
memorization and generalization can span orders of magnitude in step count.

**Why would a fully-memorized network keep changing?** The training loss is
not the only force: with decoupled weight decay the update also shrinks
every parameter. Among solutions with zero training error, the dynamics
therefore drift toward *small-norm* ones. The memorization solution stores
~2.8k arbitrary input–output pairs and pays for each; the Fourier circuit is
one reusable algorithm whose cost doesn't scale with the training set. So
the general solution is the norm-efficient one, and the trajectory —
memorize fast (steepest descent on loss), then slowly rotate weight mass
into the efficient circuit while loss stays pinned near zero — predicts both
the delay and the weight-norm decline during the transition (Liu et al.
2023 "Omnigrok"; Varma et al. 2023 frame it as circuit efficiency;
Nanda et al. 2023 measure the circuit forming *gradually* before the
accuracy jump, so the "sudden" jump is a thresholding artifact of accuracy,
not a discontinuity in the weights).

**What each sweep tests:**

- **Weight-decay sweep** (wd ∈ {0, 0.1, 1.0} at $f = 0.3$). If the norm
  pressure is the driver, wd = 0 should memorize and then *stay* memorized
  within budget, and time-to-grok should fall as wd rises.
- **Data-fraction sweep** ($f$ ∈ {0.25, 0.30, 0.40, 0.60} at wd = 1). The
  smaller the training set, the cheaper pure memorization (fewer pairs to
  store) relative to the fixed-cost general circuit — so the delay should
  grow as $f$ shrinks, diverging near a critical fraction where the general
  circuit stops being reachable/preferred.
- **Fourier + norm instrumentation.** Generalization should co-occur with
  (i) the global parameter norm falling and (ii) the embedding spectrum
  sparsifying. Both are measured on the same run, same seed.

## 5. Honest limitations

- **One seed per configuration.** Time-to-grok is known to vary across
  seeds; our sweep shows trends, not error bars. (The qualitative
  wd = 0 vs wd = 1 contrast is robust in the literature.)
- **One task, one architecture.** Grokking appears across tasks and even in
  MLPs; nothing here distinguishes transformer-specific stories.
- **Accuracy thresholds** (99% "grokked", 99.9% "memorized") are
  conventions; the underlying transition is gradual in the weights.

## References

- Power, Burda, Edwards, Babuschkin & Misra (2022), "Grokking:
  Generalization beyond overfitting on small algorithmic datasets",
  arXiv:2201.02177.
- Nanda, Chan, Lieberum, Smith & Steinhardt (2023), "Progress measures for
  grokking via mechanistic interpretability", ICLR 2023, arXiv:2301.05217.
- Liu, Michaud & Tegmark (2023), "Omnigrok: Grokking beyond algorithmic
  data", ICLR 2023, arXiv:2210.01117.
- Varma, Shah, Kenton, Kramár & Kumar (2023), "Explaining grokking through
  circuit efficiency", arXiv:2309.02390.
- Vaswani et al. (2017), "Attention is all you need", NeurIPS 2017.
- Loshchilov & Hutter (2019), "Decoupled weight decay regularization",
  ICLR 2019 (AdamW).
- Xiong et al. (2020), "On layer normalization in the transformer
  architecture", ICML 2020 (pre-LN).
