"""Structural guarantees of the from-scratch transformer."""

import torch
import torch.nn.functional as F

from grokking.model import CausalSelfAttention, LayerNorm, ModelConfig, Transformer

CFG = ModelConfig(p=97, vocab_size=98, seq_len=3, d_model=64, n_heads=4, d_mlp=128)


def test_output_shape_and_class_count():
    torch.manual_seed(0)
    model = Transformer(CFG)
    tokens = torch.randint(0, 98, (5, 3))
    logits = model(tokens)
    assert logits.shape == (5, 3, 97)  # p answer classes, not vocab_size


def test_layernorm_matches_torch_reference():
    torch.manual_seed(1)
    ln = LayerNorm(32)
    with torch.no_grad():
        ln.gamma.copy_(torch.randn(32))
        ln.beta.copy_(torch.randn(32))
    x = torch.randn(7, 5, 32)
    expected = F.layer_norm(x, (32,), ln.gamma, ln.beta, ln.eps)
    torch.testing.assert_close(ln(x), expected, rtol=1e-5, atol=1e-6)


def test_attention_matches_pytorch_reference():
    """The hand-written attention must equal PyTorch's fused reference
    (scaled_dot_product_attention with is_causal=True) given identical
    projection weights. The oracle is used only in this test."""
    torch.manual_seed(2)
    attn = CausalSelfAttention(CFG)
    x = torch.randn(4, 3, CFG.d_model)

    ours = attn(x)

    q, k, v = attn.qkv(x).split(CFG.d_model, dim=2)
    B, T = x.shape[:2]
    q = q.view(B, T, CFG.n_heads, CFG.d_head).transpose(1, 2)
    k = k.view(B, T, CFG.n_heads, CFG.d_head).transpose(1, 2)
    v = v.view(B, T, CFG.n_heads, CFG.d_head).transpose(1, 2)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    ref = attn.proj(ref.transpose(1, 2).contiguous().view(B, T, CFG.d_model))

    torch.testing.assert_close(ours, ref, rtol=1e-5, atol=1e-6)


def test_causal_mask_blocks_future_influence():
    """Changing the final token must not change logits at earlier positions;
    changing the FIRST token must change logits everywhere downstream."""
    torch.manual_seed(3)
    model = Transformer(CFG).eval()
    base = torch.tensor([[10, 20, 97]])

    changed_last = torch.tensor([[10, 20, 30]])
    with torch.no_grad():
        l_base, l_last = model(base), model(changed_last)
    torch.testing.assert_close(l_base[:, :2], l_last[:, :2])  # past unaffected
    assert not torch.allclose(l_base[:, 2], l_last[:, 2])

    changed_first = torch.tensor([[11, 20, 97]])
    with torch.no_grad():
        l_first = model(changed_first)
    assert not torch.allclose(l_base[:, 2], l_first[:, 2])  # info does flow forward


def test_dropout_is_identity_at_zero_and_active_when_on():
    """The dropout knob (added for the regularizer control) must not touch the
    default architecture: at dropout=0 the network is deterministic in train
    mode. With dropout>0 it perturbs outputs in train mode but is again the
    identity in eval mode -- so evaluation always measures the clean model."""
    tokens = torch.randint(0, 98, (5, 3))

    torch.manual_seed(0)
    plain = Transformer(ModelConfig(**{**CFG.__dict__, "dropout": 0.0})).train()
    torch.testing.assert_close(plain(tokens), plain(tokens))  # deterministic

    torch.manual_seed(0)
    dropped = Transformer(ModelConfig(**{**CFG.__dict__, "dropout": 0.1})).train()
    assert not torch.allclose(dropped(tokens), dropped(tokens))  # train: stochastic
    dropped.eval()
    torch.testing.assert_close(dropped(tokens), dropped(tokens))  # eval: identity


def test_attention_rows_are_distributions():
    """Masked softmax rows must sum to 1 over the allowed prefix and put
    exactly zero mass on the future."""
    torch.manual_seed(4)
    attn = CausalSelfAttention(CFG)
    x = torch.randn(2, 3, CFG.d_model)
    q, k, _ = attn.qkv(x).split(CFG.d_model, dim=2)
    B, T = x.shape[:2]
    q = q.view(B, T, CFG.n_heads, CFG.d_head).transpose(1, 2)
    k = k.view(B, T, CFG.n_heads, CFG.d_head).transpose(1, 2)
    att = (q @ k.transpose(-2, -1)) / (CFG.d_head ** 0.5)
    att = att.masked_fill(attn.causal_mask[:T, :T], float("-inf")).softmax(dim=-1)
    torch.testing.assert_close(att.sum(-1), torch.ones(B, CFG.n_heads, T))
    assert float(att[..., 0, 1:].abs().max().detach()) == 0.0  # position 0 sees only itself
