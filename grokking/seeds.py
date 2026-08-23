"""Running one configuration across several seeds, and caching what each
seed's checkpoint says.

Every mechanistic read-out in this repo started life as a measurement of one
run -- seed 0 -- because that is the run whose checkpoints are committed. The
timing claims outgrew that a while ago (the sweep carries five seeds and the
tables report medians and ranges), but the read-outs computed *from weights*
did not, and issue #4 is the bill for that. This module is the shared
machinery: train a configuration over a list of seeds without retraining what
is already on disk, then measure each seed's checkpoint once and keep the
numbers in a small committed CSV.

Two things here are guards rather than convenience, and both exist because the
failure they catch is invisible in the output:

- ``ensure_runs`` takes a *factory* ``seed -> TrainConfig`` and checks that the
  config it gets back actually carries the seed it asked for, and that the
  seeds produce distinct run names. A factory that ignores its argument trains
  one seed five times and writes it to one file; the resulting "five-seed"
  table is five copies of one number with a spread of exactly zero, which reads
  as a *strong* result. Nothing downstream can tell the difference, so it is
  checked here.
- ``fill_table`` only measures runs whose read-outs are missing from the cache,
  which makes ``--generate`` resumable -- but it therefore cannot notice that a
  cached row is stale. The convention the repo already uses
  (``swap_equivariance``) is the answer: the CSV is committed, and a test
  re-measures the one run whose weights are committed and requires the cached
  row to match. ``fill_table``'s job is to make that row cheap to keep, not to
  decide it is fresh.

Checkpoints are keyed by seed already -- ``TrainConfig.run_name`` puts the seed
in the filename -- so "checkpoints keyed by seed" needs no new naming scheme,
only code that walks the keys.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .train import TrainConfig, train


def ensure_runs(
    make_cfg: Callable[[int], TrainConfig],
    seeds: Iterable[int],
    out_dir: str | Path = "runs",
    verbose: bool = True,
) -> list[str]:
    """Train the seeds of one configuration that are not on disk yet.

    Parameters
    ----------
    make_cfg : ``seed -> TrainConfig``
        Builds the configuration for one seed. Must set ``cfg.seed`` to the
        seed it is handed (checked).
    seeds : the seeds to cover.
    out_dir : where ``<run_name>.json`` / ``.csv`` / ``.pt`` live.

    Returns
    -------
    The run name for every seed, in the order given. A run counts as done iff
    its JSON summary exists -- that file is written last, after the CSV and
    both checkpoints, so a half-written run is never mistaken for a finished
    one.
    """
    out = Path(out_dir)
    # Build and check the whole plan before training anything: a guard that
    # fires on the second seed has already spent an hour on the first.
    plan: list[tuple[TrainConfig, str]] = []
    for seed in seeds:
        cfg = make_cfg(seed)
        if cfg.seed != seed:
            raise ValueError(
                f"make_cfg({seed}) returned a config with seed {cfg.seed}: "
                "the factory must pass the seed through, or the 'seeds' are "
                "one run measured repeatedly"
            )
        name = cfg.run_name()
        if any(name == n for _, n in plan):
            raise ValueError(
                f"run name {name!r} repeats across seeds -- the seed is not "
                "part of the artifact name, so the runs would overwrite "
                "each other"
            )
        plan.append((cfg, name))

    for cfg, name in plan:
        if out.joinpath(f"{name}.json").exists():
            if verbose:
                print(f"skip {name} (already done)", flush=True)
        else:
            if verbose:
                print(f"=== {name} on {cfg.device} ===", flush=True)
            train(cfg, out_dir=out, verbose=verbose)
    return [name for _, name in plan]


def load_table(
    path: str | Path, key: str = "run"
) -> dict[str, dict[str, str]]:
    """Read a tidy read-out CSV into ``{key value: row}``; ``{}`` if absent.

    Values stay strings -- this layer does not know which columns are floats.
    Callers cast (or use :func:`fill_table`, which hands back the rows it was
    given for new runs and the parsed CSV rows for cached ones).
    """
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, newline="") as f:
        rows = list(csv.DictReader(f))
    if rows and key not in rows[0]:
        raise ValueError(f"{p} has no {key!r} column (has {list(rows[0])})")
    table = {r[key]: dict(r) for r in rows}
    if len(table) != len(rows):
        raise ValueError(f"{p} has duplicate {key!r} values")
    return table


def write_table(
    path: str | Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]
) -> None:
    """Write rows as a CSV with exactly ``columns``, in that order.

    Extra keys are an error rather than being dropped silently: a read-out the
    measurement produces and the schema forgets would otherwise vanish between
    the run and the committed file.
    """
    for i, row in enumerate(rows):
        missing = set(columns) - set(row)
        extra = set(row) - set(columns)
        if missing or extra:
            raise ValueError(
                f"row {i} does not match the schema "
                f"(missing {sorted(missing)}, unexpected {sorted(extra)})"
            )
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(columns))
        w.writeheader()
        w.writerows([{c: row[c] for c in columns} for row in rows])


def fill_table(
    names: Sequence[str],
    measure: Callable[[str], dict[str, Any]],
    path: str | Path,
    columns: Sequence[str],
    key: str = "run",
    force: bool = False,
) -> list[dict[str, Any]]:
    """Measure the runs missing from the cache; rewrite it; return every row.

    ``measure(name)`` returns that run's read-outs (without the ``key``
    column, which is filled in from ``name``). Rows come back in ``names``
    order, so the caller's loop order -- not the CSV's -- decides the table.
    Runs already in the cache are not re-measured unless ``force``; that is
    what lets a generate pass resume after an interrupted session, and it is
    also why a committed cache needs a test that re-measures at least one row.
    """
    cached = {} if force else load_table(path, key=key)
    rows: list[dict[str, Any]] = []
    for name in names:
        if name in cached:
            rows.append(cached[name])
        else:
            row = measure(name)
            if key in row:
                raise ValueError(
                    f"measure({name!r}) returned a {key!r} column; that is "
                    "filled in here so a row cannot claim to be another run"
                )
            rows.append({key: name, **row})
    write_table(path, rows, columns)
    return rows
