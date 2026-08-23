"""The multi-seed machinery: the guards, and the read-out cache.

``grokking/seeds.py`` exists so a mechanistic read-out can be measured across
seeds without retraining what is on disk and without re-measuring what is
already cached. Both of those shortcuts can fail *silently* -- a factory that
ignores the seed produces five copies of one run, and a cache that is never
invalidated produces numbers from an older model -- so the guards against them
are what is tested here, alongside the plain behavior.
"""

import csv

import pytest

from grokking.model import ModelConfig
from grokking.seeds import ensure_runs, fill_table, load_table, write_table
from grokking.train import TrainConfig


def cfg(seed, n_heads=4):
    return TrainConfig(p=97, train_frac=0.30, weight_decay=1.0, seed=seed,
                       model=ModelConfig(n_heads=n_heads))


# -- ensure_runs -------------------------------------------------------------

def test_existing_runs_are_skipped_and_named_in_order(tmp_path):
    """A run counts as done iff its JSON summary exists; nothing is trained."""
    for seed in (0, 1, 2):
        (tmp_path / f"{cfg(seed).run_name()}.json").write_text("{}")
    names = ensure_runs(cfg, [0, 1, 2], out_dir=tmp_path, verbose=False)
    assert names == [cfg(s).run_name() for s in (0, 1, 2)]


def test_a_missing_run_is_trained_with_the_config_the_factory_returned(tmp_path, monkeypatch):
    import grokking.seeds as seeds

    trained = []
    monkeypatch.setattr(seeds, "train",
                        lambda c, out_dir, verbose: trained.append((c, out_dir)))
    (tmp_path / f"{cfg(0).run_name()}.json").write_text("{}")
    ensure_runs(lambda s: cfg(s, n_heads=2), [0, 1], out_dir=tmp_path,
                verbose=False)
    # seed 0's h2 name differs from the h4 file that exists, so both train
    assert [c.seed for c, _ in trained] == [0, 1]
    assert all(c.model.n_heads == 2 for c, _ in trained)
    assert all(out == tmp_path for _, out in trained)


def test_a_factory_that_ignores_the_seed_is_rejected(tmp_path):
    """The failure this guard exists for: five 'seeds' that are one run, whose
    table would report a spread of exactly zero and read as a strong result."""
    with pytest.raises(ValueError, match="must pass the seed through"):
        ensure_runs(lambda s: cfg(0), [0, 1, 2], out_dir=tmp_path, verbose=False)


def test_seeds_that_collide_in_the_run_name_are_rejected(tmp_path):
    """A config whose name drops the seed would have the runs overwrite each
    other on disk; caught before the second one trains."""
    class Unseeded(TrainConfig):
        def run_name(self):
            return "fixed_name"

    def make(seed):
        c = Unseeded(p=97, train_frac=0.30, weight_decay=1.0, seed=seed)
        return c

    with pytest.raises(ValueError, match="repeats across seeds"):
        ensure_runs(make, [0, 1], out_dir=tmp_path, verbose=False)


# -- the read-out table ------------------------------------------------------

COLS = ("run", "seed", "value")


def test_write_then_load_round_trips(tmp_path):
    p = tmp_path / "t.csv"
    write_table(p, [{"run": "a", "seed": 0, "value": 1.5}], COLS)
    assert load_table(p) == {"a": {"run": "a", "seed": "0", "value": "1.5"}}
    assert list(csv.DictReader(open(p)).fieldnames) == list(COLS)


def test_load_table_of_a_missing_file_is_empty(tmp_path):
    assert load_table(tmp_path / "nope.csv") == {}


def test_write_table_rejects_a_row_that_does_not_match_the_schema(tmp_path):
    p = tmp_path / "t.csv"
    with pytest.raises(ValueError, match="unexpected"):
        write_table(p, [{"run": "a", "seed": 0, "value": 1.0, "extra": 2}], COLS)
    with pytest.raises(ValueError, match="missing"):
        write_table(p, [{"run": "a", "seed": 0}], COLS)


def test_duplicate_keys_in_a_cached_file_are_rejected(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("run,seed,value\na,0,1\na,1,2\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_table(p)


def test_fill_table_measures_only_what_is_missing(tmp_path):
    p = tmp_path / "t.csv"
    write_table(p, [{"run": "a", "seed": 0, "value": 1.5}], COLS)
    seen = []

    def measure(name):
        seen.append(name)
        return {"seed": 9, "value": 2.5}

    rows = fill_table(["a", "b"], measure, p, COLS)
    assert seen == ["b"]                       # "a" came from the cache
    assert [r["run"] for r in rows] == ["a", "b"]
    assert load_table(p)["b"]["value"] == "2.5"
    # and the cached row survived the rewrite untouched
    assert load_table(p)["a"]["value"] == "1.5"


def test_fill_table_force_remeasures_everything(tmp_path):
    p = tmp_path / "t.csv"
    write_table(p, [{"run": "a", "seed": 0, "value": 1.5}], COLS)
    rows = fill_table(["a"], lambda n: {"seed": 0, "value": 9.0}, p, COLS,
                      force=True)
    assert rows[0]["value"] == 9.0


def test_fill_table_returns_rows_in_the_order_asked_for(tmp_path):
    p = tmp_path / "t.csv"
    write_table(p, [{"run": "b", "seed": 1, "value": 2.0},
                    {"run": "a", "seed": 0, "value": 1.0}], COLS)
    rows = fill_table(["a", "b"], lambda n: {"seed": 0, "value": 0.0}, p, COLS)
    assert [r["run"] for r in rows] == ["a", "b"]


def test_a_measurement_may_not_relabel_itself(tmp_path):
    """``measure`` returning its own 'run' column could attribute one model's
    read-out to another run; the key is filled in from the name instead."""
    p = tmp_path / "t.csv"
    with pytest.raises(ValueError, match="filled in here"):
        fill_table(["a"], lambda n: {"run": "other", "seed": 0, "value": 1.0},
                   p, COLS)
