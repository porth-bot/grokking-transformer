"""Modular addition dataset: the standard grokking task (Power et al. 2022).

Learn (a + b) mod p from examples. The dataset is *all* p^2 ordered pairs,
encoded as the token sequence [a, b, =] with the answer supervised at the
"=" position. Two properties make this task the canonical grokking probe:

- **Exhaustive and noiseless**: train/test split is a random partition of a
  finite universe, so "generalization" means recovering the algorithm, not
  smoothing over noise. Chance accuracy is 1/p (~1%), memorization gets you
  exactly the train set, and the gap between the two is unambiguous.
- **Algorithmically structured**: addition mod p has a Fourier structure
  (see theory/notes.md) that trained networks demonstrably exploit, which is
  what makes the delayed generalization *inspectable* rather than mysterious.
"""

from __future__ import annotations

import torch

EQ_OFFSET = 0  # "=" token id is p (digits occupy 0..p-1)


def modular_addition_dataset(p: int) -> tuple[torch.Tensor, torch.Tensor]:
    """All p^2 examples of (a + b) mod p.

    Returns
    -------
    tokens : (p^2, 3) int64 -- rows [a, b, p] where p is the "=" token id.
    targets : (p^2,) int64 -- (a + b) mod p.
    """
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    eq = torch.full_like(a, p)
    tokens = torch.stack([a, b, eq], dim=1)
    targets = (a + b) % p
    return tokens, targets


def train_test_split(
    tokens: torch.Tensor, targets: torch.Tensor, train_frac: float, seed: int
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Deterministic random partition of the full example universe.

    The split is a function of (seed, train_frac) only, so every sweep
    configuration trains on a reproducible subset and is evaluated on its
    exact complement (disjointness is asserted in tests).
    """
    n = tokens.shape[0]
    n_train = int(round(train_frac * n))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    tr, te = perm[:n_train], perm[n_train:]
    return (tokens[tr], targets[tr]), (tokens[te], targets[te])
