"""Modular arithmetic datasets: the standard grokking task (Power et al. 2022).

Learn ``(a op b) mod p`` from examples. The dataset is *all* p^2 ordered pairs,
encoded as the token sequence [a, b, =] with the answer supervised at the
"=" position. Two properties make this task the canonical grokking probe:

- **Exhaustive and noiseless**: train/test split is a random partition of a
  finite universe, so "generalization" means recovering the algorithm, not
  smoothing over noise. Chance accuracy is 1/p (~1%), memorization gets you
  exactly the train set, and the gap between the two is unambiguous.
- **Algorithmically structured**: addition mod p has a Fourier structure
  (see theory/notes.md) that trained networks demonstrably exploit, which is
  what makes the delayed generalization *inspectable* rather than mysterious.

Three binary operations are supported (all map onto the same abelian-group
Fourier story, which is exactly why the comparison is interesting -- see
``experiments/operations.py`` and README section 11):

- ``"add"``: ``(a + b) mod p``. The canonical task.
- ``"sub"``: ``(a - b) mod p``. Still the additive group of Z/pZ -- negating
  the second operand is a relabelling, so the same circuit applies.
- ``"mul"``: ``(a * b) mod p``. On the *nonzero* residues this is the cyclic
  MULTIPLICATIVE group (Z/pZ)^x, order p-1, isomorphic to addition mod (p-1)
  via the discrete logarithm to a primitive root (a = g^i, b = g^j =>
  a*b = g^((i+j) mod (p-1))). The 2p-1 pairs with a=0 or b=0 map to 0 and sit
  outside that group -- a trivial constant the network can only memorize.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

EQ_OFFSET = 0  # "=" token id is p (digits occupy 0..p-1)

# Each operation is the elementwise map (a, b) -> target BEFORE the mod p; the
# caller applies ``% p``. Kept as a table so run configs and tests share one
# source of truth for the set of supported operations.
OPERATIONS: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
}


def modular_dataset(
    p: int, operation: str = "add"
) -> tuple[torch.Tensor, torch.Tensor]:
    """All p^2 examples of ``(a op b) mod p`` for ``op`` in ``OPERATIONS``.

    Returns
    -------
    tokens : (p^2, 3) int64 -- rows [a, b, p] where p is the "=" token id.
    targets : (p^2,) int64 -- ``(a op b) mod p`` (in ``0..p-1``; the modulo
        maps the negatives that subtraction produces back into range).
    """
    if operation not in OPERATIONS:
        raise ValueError(
            f"unknown operation {operation!r}; choose from {sorted(OPERATIONS)}"
        )
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    eq = torch.full_like(a, p)
    tokens = torch.stack([a, b, eq], dim=1)
    targets = OPERATIONS[operation](a, b) % p
    return tokens, targets


def modular_addition_dataset(p: int) -> tuple[torch.Tensor, torch.Tensor]:
    """All p^2 examples of ``(a + b) mod p`` (``modular_dataset(p, "add")``).

    Kept as a named wrapper so the many call sites and checkpoints built around
    the canonical addition task are untouched by the multi-operation support.
    """
    return modular_dataset(p, "add")


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
