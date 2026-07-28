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

- **Five seeds, not a distribution.** The wd and frac cells are run at seeds
  0–4 and reported as median [min–max]; the between-cell gaps survive that
  spread (the slowest wd = 1 seed groks before the fastest wd = 0.1 one), but
  five draws bound nothing. The later single-seed studies — head count
  (README §9) and the operation sweep at wd 0.1 — say so where they appear,
  and the operations table was subsequently re-run at three seeds, which
  retracted one of its seed-0 readings.
- **One task, one architecture.** Grokking appears across tasks and even in
  MLPs; nothing here distinguishes transformer-specific stories.
- **Accuracy thresholds** (99% "grokked", 99.9% "memorized") are
  conventions; the underlying transition is gradual in the weights.

## 6. Exercises

Five things worth deriving rather than reading. Solutions are collapsed;
each ends with where the answer is checked, because a number stated in prose
and a number a test computes should not be able to drift apart. Same format
as [mcmc-from-scratch's Sec. 7](https://github.com/porth-bot/mcmc-from-scratch)
and [gp-from-scratch's Sec. 9](https://github.com/porth-bot/gp-from-scratch).

---

**Exercise 1.** Sec. 3 shows that *all* $p$ frequencies give an exact delta.
A trained network uses about five. Show that the peak survives, and work out
by how much: for a set $K$ of $m$ frequencies, what are the peak and the
typical off-target size of $\sum_{k \in K} \cos(2\pi k n/p)$?

<details><summary>Solution</summary>

At $n = 0$ every cosine is 1, so the sum is exactly $m$ — the peak is
unaffected by dropping frequencies. For $n \neq 0$ the phases $2\pi k n/p$
are spread around the circle, and a cosine of an (effectively) uniform angle
has mean 0 and variance $1/2$, so a sum of $m$ near-independent such terms has
mean 0 and standard deviation $\sqrt{m/2}$. The margin is therefore

$$\frac{\text{peak}}{\text{off-target sd}} = \frac{m}{\sqrt{m/2}} = \sqrt{2m},$$

which is the whole content of "a sparse subset is enough": the peak grows like
$m$ and the interference only like $\sqrt m$, so the gap opens with $m$ and
five frequencies already give $\sqrt{10} \approx 3.2$ standard deviations.

For the trained model's own $K = \{5, 14, 20, 36, 38\}$ at $p = 97$: peak
5.000, off-target sd 1.512 against a predicted 1.581, largest off-target value
2.677 — so the correct answer wins by a factor of 1.87, and
`logits.argmax` is right for all $97^2$ pairs. Note what the margin is *not*:
the off-target values are not small (the largest is more than half the peak).
Argmax does not need them to be.

Checked in `tests/test_theory_exercises.py`
(`test_all_frequencies_give_the_exact_delta`,
`test_a_sparse_set_keeps_the_peak_...`,
`test_the_sparse_logits_pick_the_right_answer_for_every_pair`).
</details>

---

**Exercise 2.** Sec. 3 asserts that the algorithm makes the logit tensor
$L[a,b,c]$ a function of $a+b$. Show that this puts *all* of its Fourier
energy on the diagonal $k_a = k_b$ of the 2D DFT over the input axes — the
measurement `experiments/logit_attribution.py` reports.

<details><summary>Solution</summary>

Expand the target logit with the angle-addition identities, writing
$\omega = 2\pi k/p$:

$$\cos(\omega(a+b-c)) = \big[\cos\omega a\,\cos\omega b - \sin\omega a\,\sin\omega b\big]\cos\omega c + \big[\sin\omega a\,\cos\omega b + \cos\omega a\,\sin\omega b\big]\sin\omega c.$$

Every bracket is a sum of **products of a pure frequency-$k$ function of $a$
with a pure frequency-$k$ function of $b$** — this is exactly why the
architecture can compute it, since attention and the MLP supply the
multiplication, and it is also the answer. A product $f(a)g(b)$ has 2D DFT
$\hat f \otimes \hat g$, and $\hat f, \hat g$ are supported on $\{\pm k\}$, so
each term contributes only at $(\pm k, \pm k)$. Summing over $k \in K$ leaves
support exactly on $\{(k,k), (p-k,p-k) : k \in K\}$ and nothing else.

The converse is the reason the measurement is meaningful in the other
direction: any function of $a+b$ alone is constant along $a+b = \text{const}$,
and such functions are precisely those whose 2D DFT lives on the diagonal.

The ideal tensor puts $1.0000000000$ of its non-DC energy on those modes. That
is the reference the grokked model's **0.98** and the memorizing model's
**0.12** (README §8) are read against — the trained network is not
approximately doing this, it is doing it to two digits.

Checked in `test_the_ideal_logit_tensor_lives_exactly_on_the_diagonal_modes`.
</details>

---

**Exercise 3.** Compute the attention statistics at initialization: the
standard deviation of the pre-softmax logits, and the resulting spread of the
attention weights. Then use them to decide what the appendix's "operand
asymmetry" number is actually measuring.

<details><summary>Solution</summary>

*(a) Logit scale.* Pre-LN hands the projection rows of unit variance, so with
$W_Q$ entries i.i.d. $\mathcal N(0, \sigma_W^2)$ each query coordinate has
variance $\sigma_W^2 d_{\text{model}}$, and likewise for keys. Then
$\operatorname{Var}(q \cdot k) = d_{\text{head}}\, \sigma_W^4 d_{\text{model}}^2$,
and the $1/\sqrt{d_{\text{head}}}$ leaves

$$\operatorname{sd}(z) = \sigma_W^2 d_{\text{model}} = 0.02^2 \times 128 = 0.0512,$$

with **no $d_{\text{head}}$ in it** — the scaling cancels the head split
exactly. So 1, 2 and 4 heads all start at the same attention temperature,
which is worth knowing next to README §9's head-count ablation: whatever
makes more heads grok more slowly, it is not a different starting point.
Measured across seeds: 0.039–0.052.

*(b) Attention spread.* For small $s = \operatorname{sd}(z)$, expand the
softmax about equal logits: $A_i \approx \frac1n + \frac{z_i - \bar z}{n}$.
Hence

$$\operatorname{Var}(A_i) = \frac{s^2 (n-1)}{n^3}, \qquad \operatorname{Var}(A_i - A_j) = \frac{2s^2}{n^2},$$

the second using $\operatorname{Cov}(A_i, A_j) = -s^2/n^3$. With $n = 3$ and
$s \approx 0.044$ that gives $\mathbb{E}|A_i - A_j| = \frac{s\sqrt2}{n}\sqrt{2/\pi} \approx 0.018$
per example. The formulas are a *small-$s$* expansion and are exact to three
digits at $s = 0.05$ but already 11% off at $s = 0.5$; the real model also
violates their independence assumption, since the three logits share one query
vector (measured $\operatorname{corr}(z_a, z_b)$ from $-0.04$ to $+0.29$), so
per-seed agreement is only within ~30%.

*(c) What the appendix statistic measures.* The appendix reports the
asymmetry **after averaging over the dataset**: init 0.004–0.017, memorization
0.189, grokked 0.0001. Write $D(a,b) = A[(a,b) \to \text{pos }0] - A[(a,b) \to \text{pos }1]$.
If the "=" token's attention follows the *token* rather than the position —
i.e. the computation is equivariant under swapping the operands — then
$A[(b,a) \to \text{pos }0] = A[(a,b) \to \text{pos }1]$, so $D(b,a) = -D(a,b)$.
$D$ is **odd** under the swap, and the dataset is every ordered pair and hence
closed under it, so the average is exactly zero.

So the statistic is a measure of *swap-equivariance*, not of "each input is
read symmetrically" — and the distinction is not academic, because the grokked
model does **not** read individual inputs symmetrically: its per-example
$\mathbb{E}|D|$ is 0.15, eight times the value at initialization. Measuring
the equivariance defect directly,
$\mathbb{E}\big|A[(a,b) \to a] - A[(b,a) \to b]\big|$, gives 0.189 at
memorization and 0.00017 after grokking, and the symmetry reaches the output:
the memorizing model's logits change by 0.61 of their own standard deviation
when the operands are swapped — it genuinely computes a non-commutative
function — against 0.004 for the grokked one. Commutativity is learned, and it
is learned at grokking.

Checked in `test_the_delta_method_softmax_variances_are_right_for_independent_logits`,
`test_the_initial_attention_logit_scale_matches_the_hand_computation`,
`test_the_initial_operand_asymmetry_...`, and
`test_the_published_asymmetry_statistic_measures_swap_equivariance_...`.
</details>

---

**Exercise 4.** `grokking/progress.py` reports "the fraction of embedding
Fourier energy in the top $k$ frequencies". Show that this is a statement
about the embedding and not about the transform — and find the one thing in
that module that a factor of 2 *would* make wrong.

<details><summary>Solution</summary>

Parseval for the DFT of a real $p \times d$ matrix $E$ along the token axis:

$$\sum_{n=0}^{p-1} \\|E_n - \bar E\\|^2 = \frac1p \sum_{k=1}^{p-1} \\|\hat E_k\\|^2 = \frac2p \sum_{k=1}^{(p-1)/2} \\|\hat E_k\\|^2$$

for odd $p$, the last step by conjugate symmetry $\hat E_{p-k} = \overline{\hat E_k}$
— which is exactly the half-spectrum a real FFT returns. Subtracting the mean
is the same as dropping $k = 0$. So the non-DC spectrum accounts for precisely
the centred embedding energy, nothing more and nothing missing, and a
*fraction* of it is a fraction of a real quantity. Verified to 1e-10 on both
committed checkpoints.

The thing the factor would break: `embedding_spectrum` returns
$\\|\hat E_k\\|$ without the $2/p$, so `embedding_top_k_fraction` is correct
(the constant cancels in a ratio) while any **absolute** modal energy read off
that function is half the true one. The measure is right; a number lifted out
of it would not be.

Aside, since the spectrum already computes it: the centred embedding energy
falls from 7.59 at memorization to 1.14 at the end — Sec. 4's norm pressure,
visible in the embedding table alone.

Checked in `test_the_embedding_spectrum_accounts_for_exactly_the_centred_energy`,
`test_the_top_k_fraction_is_unchanged_by_the_normalization_it_omits`, and
`test_weight_decay_is_visible_in_the_embedding_energy_alone`.
</details>

---

**Exercise 5.** README §11 finds that $(a \cdot b) \bmod p$ groks too. Explain
why that is not a separate phenomenon — and then say what the explanation does
*not* account for.

<details><summary>Solution</summary>

The nonzero residues $(\mathbb{Z}/97)^\times$ form a cyclic group of order
$p - 1 = 96$ under multiplication. Fixing a generator $g$, the discrete
logarithm $\log_g : (\mathbb{Z}/97)^\times \to \mathbb{Z}/96$ is a group
isomorphism, so $\log_g(ab) = \log_g a + \log_g b \bmod 96$: multiplication
*is* addition, in a relabelling of the tokens. A network that can learn the
Fourier-addition circuit can learn this one, over $\mathbb{Z}/96$ rather than
$\mathbb{Z}/97$. The $2p - 1 = 193$ pairs with a zero operand (2.05% of the
$97^2$ grid) are outside the group and are simply memorized.

Now the part that does not follow. The seed-0 run made multiplication look
*fastest*, and the tempting story was that 96 is highly composite ($2^5 \cdot 3$)
while 97 is prime, so the multiplicative task has more usable subgroup
structure. Three seeds retracted it: mul groks at 1,100 [1,000–1,500] and add
at 1,300 [1,200–1,900] — overlapping ranges, no effect to explain. And the
isomorphism cannot be the operative variable anyway, because addition and
subtraction share a group exactly and differ decisively (sub 3,700
[2,500–3,900], and at wd 0.1 it groks in only one seed of three).

So: the isomorphism explains why multiplication groks **at all**, and explains
nothing about speed. The surviving hypothesis for the subtraction gap is
commutativity — subtraction is the only non-commutative operation of the
three, and Exercise 3 shows the addition model spends its grokking transition
acquiring exactly that symmetry, which a subtraction model cannot use. README
§11's "Next" asks for the measurement that would test it: the swap-equivariance
read-out of Exercise 3, run on the subtraction checkpoints.
</details>

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
