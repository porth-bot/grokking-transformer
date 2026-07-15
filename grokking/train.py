"""Full-batch AdamW training with the instrumentation grokking needs.

Grokking is a *trajectory* phenomenon, so the training loop's job is mostly
measurement: at every eval step we record train/test loss and accuracy plus
the global parameter L2 norm. The weight norm is the key auxiliary signal --
the memorizing solution found first has large norm; weight decay then slowly
trades pure-memorization weight mass for the general (Fourier) circuit, and
the test-accuracy jump co-occurs with the norm's decline
(Nanda et al. 2023; Liu et al. 2023 "Omnigrok" makes the norm story central).

Choices that matter:

- **Full batch.** The dataset fits in one tensor (p^2 <= ~10k rows). This
  removes minibatch noise as a confound and matches the standard setup.
- **AdamW, not Adam-with-L2.** Decoupled weight decay (Loshchilov & Hutter
  2019) shrinks weights directly in the update; L2-in-the-loss folded through
  Adam's preconditioner decays adaptively-scaled and is NOT the same
  regularizer. Grokking's dependence on wd is the experiment, so the
  regularizer must be the clean one.
- **Two checkpoints.** At the *memorization point* (first eval with train
  acc >= 99.9%) and at the end. The Fourier analysis compares them: same
  train accuracy, completely different internals.
- **Early stop with grace.** Stop only after test acc >= 99.9% holds for
  `patience` consecutive evals, so a lucky eval can't truncate the run.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import modular_addition_dataset, train_test_split
from .model import ModelConfig, Transformer


@dataclass
class TrainConfig:
    p: int = 97
    train_frac: float = 0.4
    weight_decay: float = 1.0
    wd_scope: str = "all"      # {"all", "embeddings", "non_embeddings"}
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.98)
    max_steps: int = 30_000
    eval_every: int = 100
    patience: int = 5          # consecutive >=99.9% test evals before stopping
    seed: int = 0
    device: str = ""           # "" -> auto: cuda > mps > cpu
    model: ModelConfig = field(default_factory=ModelConfig)

    def __post_init__(self) -> None:
        self.model.p = self.p
        self.model.vocab_size = self.p + 1
        if not self.device:
            self.device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )

    def run_name(self) -> str:
        base = f"p{self.p}_frac{self.train_frac:.2f}_wd{self.weight_decay:g}_seed{self.seed}"
        # lr joins the run identity only when it differs from the 1e-3 default,
        # so existing default-lr run artifacts keep their names.
        if abs(self.lr - 1e-3) > 1e-12:
            base += f"_lr{self.lr:g}"
        # dropout likewise only tags the name when it is actually on (the
        # regularizer control), leaving every wd-only run's artifacts untouched.
        if self.model.dropout > 0:
            base += f"_do{self.model.dropout:g}"
        # weight-decay scope tags the name only when decay is restricted to a
        # parameter subset, so the default "decay everything" runs keep their
        # existing artifact names.
        if self.wd_scope != "all":
            base += f"_wds{self.wd_scope}"
        return base


@torch.no_grad()
def evaluate(
    model: Transformer, tokens: torch.Tensor, targets: torch.Tensor
) -> tuple[float, float]:
    """Loss and accuracy from the logits at the '=' position."""
    logits = model(tokens)[:, -1, :]
    loss = F.cross_entropy(logits, targets)
    acc = (logits.argmax(dim=-1) == targets).float().mean()
    return float(loss), float(acc)


@torch.no_grad()
def weight_norm(model: nn.Module) -> float:
    return math.sqrt(sum(float(p.pow(2).sum()) for p in model.parameters()))


# The two embedding tensors -- the token lookup table and the learned
# positional embeddings. Everything else (attention, MLP, LayerNorm affines,
# unembed) is "non-embedding". Used to scope weight decay in the ablation
# that asks *which* parameters' norm pressure drives grokking.
EMBEDDING_PARAMS = ("tok_emb.weight", "pos_emb")


def weight_decay_groups(
    model: nn.Module, weight_decay: float, scope: str
) -> list[dict[str, Any]] | None:
    """AdamW parameter groups that apply ``weight_decay`` only to ``scope``.

    ``scope`` is one of:
      - ``"all"``            -> ``None`` (caller uses a single default group;
                                behavior is bit-for-bit the standard run),
      - ``"embeddings"``     -> decay only ``tok_emb.weight`` and ``pos_emb``,
      - ``"non_embeddings"`` -> decay everything *except* those two.

    The untargeted group gets ``weight_decay=0`` so it trains with plain Adam
    dynamics, isolating the effect of norm pressure on the targeted subset.
    """
    if scope == "all":
        return None
    if scope not in ("embeddings", "non_embeddings"):
        raise ValueError(f"unknown wd_scope {scope!r}")
    emb: list[nn.Parameter] = []
    rest: list[nn.Parameter] = []
    for name, p in model.named_parameters():
        (emb if name in EMBEDDING_PARAMS else rest).append(p)
    decayed, free = (emb, rest) if scope == "embeddings" else (rest, emb)
    return [
        {"params": decayed, "weight_decay": weight_decay},
        {"params": free, "weight_decay": 0.0},
    ]


def train(
    cfg: TrainConfig, out_dir: str = "runs", verbose: bool = True
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one grokking experiment; returns the history and summary dict.

    Writes ``<out_dir>/<run_name>.csv`` (the training trajectory),
    ``.json`` (config + summary), and ``.pt`` / ``_memorize.pt``
    (final / memorization-point checkpoints).
    """
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    tokens, targets = modular_addition_dataset(cfg.p)
    (tr_x, tr_y), (te_x, te_y) = train_test_split(tokens, targets, cfg.train_frac, cfg.seed)
    tr_x, tr_y = tr_x.to(device), tr_y.to(device)
    te_x, te_y = te_x.to(device), te_y.to(device)

    model = Transformer(cfg.model).to(device)
    groups = weight_decay_groups(model, cfg.weight_decay, cfg.wd_scope)
    if groups is None:
        opt = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, betas=cfg.betas,
            weight_decay=cfg.weight_decay,
        )
    else:
        opt = torch.optim.AdamW(groups, lr=cfg.lr, betas=cfg.betas)

    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    name = cfg.run_name()

    history = []
    memorize_step, grok_step = None, None
    streak = 0
    t0 = time.time()

    for step in range(cfg.max_steps + 1):
        if step % cfg.eval_every == 0:
            model.eval()
            tr_loss, tr_acc = evaluate(model, tr_x, tr_y)
            te_loss, te_acc = evaluate(model, te_x, te_y)
            history.append(
                {
                    "step": step,
                    "train_loss": tr_loss,
                    "test_loss": te_loss,
                    "train_acc": tr_acc,
                    "test_acc": te_acc,
                    "weight_norm": weight_norm(model),
                }
            )
            if memorize_step is None and tr_acc >= 0.999:
                memorize_step = step
                torch.save(model.state_dict(), out / f"{name}_memorize.pt")
            if grok_step is None and te_acc >= 0.99:
                grok_step = step
            streak = streak + 1 if te_acc >= 0.999 else 0
            if verbose and step % (10 * cfg.eval_every) == 0:
                print(
                    f"[{name}] step {step:6d}  train {tr_acc:6.3f}  "
                    f"test {te_acc:6.3f}  |w| {history[-1]['weight_norm']:8.1f}",
                    flush=True,
                )
            if streak >= cfg.patience:
                break
            model.train()

        logits = model(tr_x)[:, -1, :]
        loss = F.cross_entropy(logits, tr_y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    torch.save(model.state_dict(), out / f"{name}.pt")
    summary = {
        "config": {**asdict(cfg), "model": asdict(cfg.model)},
        "memorize_step": memorize_step,
        "grok_step": grok_step,
        "final_train_acc": history[-1]["train_acc"],
        "final_test_acc": history[-1]["test_acc"],
        "steps_run": history[-1]["step"],
        "wall_seconds": round(time.time() - t0, 1),
        "n_params": model.n_params(),
    }
    with open(out / f"{name}.csv", "w") as f:
        cols = list(history[0].keys())
        f.write(",".join(cols) + "\n")
        for row in history:
            f.write(",".join(str(row[c]) for c in cols) + "\n")
    with open(out / f"{name}.json", "w") as f:
        json.dump(summary, f, indent=2)
    if verbose:
        print(f"[{name}] done: {summary['steps_run']} steps, "
              f"memorize@{memorize_step}, grok@{grok_step}, "
              f"{summary['wall_seconds']}s", flush=True)
    return history, summary
