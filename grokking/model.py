"""Decoder-only transformer with attention written out by hand.

Scaled dot-product attention (Vaswani et al. 2017): tokens are projected to
queries, keys, and values; token i attends to token j with weight

    A_ij = softmax_j( q_i . k_j / sqrt(d_head) )   for j <= i (causal mask),

and its new representation is sum_j A_ij v_j. The 1/sqrt(d_head) keeps the
logits' variance ~1 at initialization: for q, k with i.i.d. unit-variance
entries, Var(q . k) = d_head, and softmax saturates (killing gradients) if
its inputs grow with dimension.

Design choices, and why:

- **Learned positional embeddings**, added to token embeddings. With
  sequences of fixed length 3 ([a, b, =]) there is nothing for sinusoidal
  extrapolation to buy.
- **Pre-LayerNorm** residual blocks, x + f(LN(x)): the residual stream keeps
  an identity path from input to logits, which trains stably without warmup
  (Xiong et al. 2020).
- **LayerNorm written out** (mean/variance over the model dimension, then a
  learned affine): it is the one normalization this repo relies on, so it is
  implemented, not imported.
- **No dropout**: grokking experiments are full-batch and the phenomenon
  under study *is* the regularization story -- weight decay must be the only
  regularizer in play.
- ``nn.Linear`` / ``nn.Embedding`` are used as parameter containers; the
  attention arithmetic itself never calls a fused/library attention op
  (tests verify equivalence against PyTorch's reference implementation).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    p: int = 97            # modulus; answers live in {0, ..., p-1}
    vocab_size: int = 98   # p digit tokens + 1 "=" token (id p)
    seq_len: int = 3       # [a, b, =]
    d_model: int = 128
    n_heads: int = 4
    d_mlp: int = 512
    n_layers: int = 1

    @property
    def d_head(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads


class LayerNorm(nn.Module):
    """y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta, stats over d_model.

    Normalizing per token (not per batch) makes the op independent of batch
    composition -- essential when train and eval batch sizes differ.
    Biased variance (1/N) matches torch.nn.functional.layer_norm.
    """

    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps) * self.gamma + self.beta


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention, spelled out.

    Shapes: input (B, T, C) -> qkv projection -> (B, n_heads, T, d_head)
    per q/k/v -> attention weights (B, n_heads, T, T) -> values -> merge
    heads -> output projection (B, T, C).

    The causal mask sets logits for j > i to -inf *before* softmax, so
    masked positions get exactly zero weight and each row still sums to 1
    over the allowed prefix.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads, self.d_head = cfg.n_heads, cfg.d_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        mask = torch.triu(torch.ones(cfg.seq_len, cfg.seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask)  # True where attention is forbidden

    def _attention(self, x):
        """Shared core: return the attention weights (B, H, T, T) and values.

        Kept as one place so ``forward`` and ``attn_weights`` (used by the
        attention-pattern analysis) can never disagree about the arithmetic.
        """
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)   # (B, H, T, T)
        att = att.masked_fill(self.causal_mask[:T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        return att, v

    def forward(self, x):
        B, T, C = x.shape
        att, v = self._attention(x)
        y = att @ v                                                # (B, H, T, d_head)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

    @torch.no_grad()
    def attn_weights(self, x):
        """Attention weight matrices (B, H, T, T) for analysis/plotting.

        Row i of head h is token i's distribution over the positions it
        attends to (j <= i under the causal mask). Not used in the forward
        pass -- purely a read-out of the same softmax ``forward`` computes.
        """
        return self._attention(x)[0]


class MLP(nn.Module):
    """Position-wise 2-layer network: d_model -> d_mlp -> GELU -> d_model.

    In the grokking-circuits picture this is where the "multiply the
    frequency components" nonlinearity lives (Nanda et al. 2023).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.up = nn.Linear(cfg.d_model, cfg.d_mlp)
        self.down = nn.Linear(cfg.d_mlp, cfg.d_model)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    """Token + position embeddings -> n_layers blocks -> LN -> unembed to p logits.

    The unembedding maps to p classes (the possible answers), not to
    vocab_size: the "=" token is never a valid answer. Loss and accuracy are
    computed from the logits at the final position (the "=" slot), where the
    model must write the sum.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(cfg.seq_len, cfg.d_model))
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = LayerNorm(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.p, bias=False)
        self.apply(self._init)
        nn.init.normal_(self.pos_emb, std=0.02)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens):
        """tokens: (B, T) int64 -> logits (B, T, p)."""
        x = self.tok_emb(tokens) + self.pos_emb[: tokens.shape[1]]
        for block in self.blocks:
            x = block(x)
        return self.unembed(self.ln_f(x))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
