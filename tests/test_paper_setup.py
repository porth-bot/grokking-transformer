"""The paper's Setup section must agree with the code and configs it describes.

Three files now check the paper, against three different things:

* ``test_paper.py``         -- against the *repo*: figures, refs, citations.
* ``test_paper_numbers.py`` -- against the *logs*: every grok-step cell in the
  five tables, recomputed from ``runs/*.json``.
* this file                 -- against the *implementation and the run
  configs*: the model shape, dataset sizes, optimizer settings, thresholds,
  budgets and seed counts that Section 2 states in prose.

Section 2 is where a reader learns what was actually run, and until now not one
number in it was checked by anything. It is also the section most likely to
drift, because it describes the setup rather than a result: change a default in
``ModelConfig`` or add a seed to a sweep and no figure moves, no table moves,
and the prose quietly stops being true. Every assertion here reads the paper
first and the code second, so the failure message names the sentence that lied.

Unlike ``test_paper_numbers.py`` this file imports torch, because the claims it
checks are claims about the model and the split rather than about the logs. CI
installs torch, so it runs there.
"""

import json
import re
from pathlib import Path

import pytest

from grokking.data import modular_dataset, train_test_split
from grokking.model import ModelConfig, Transformer
from grokking.train import EARLY_STOP_ACC, GROK_ACC, MEMORIZE_ACC, TrainConfig

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
TEX = (ROOT / "paper" / "main.tex").read_text()

# Section 2 runs from its own heading to the next \section.
SETUP = TEX.split(r"\label{sec:setup}", 1)[1].split(r"\section{", 1)[0]


def setup_says(pattern):
    """Assert Section 2 contains ``pattern`` (a regex), and return the match."""
    m = re.search(pattern, SETUP)
    assert m, f"Section 2 no longer contains /{pattern}/"
    return m


def run_config(name):
    return json.loads((RUNS / f"{name}.json").read_text())["config"]


def training_runs():
    """The ``runs/p*.json`` files that are actually a training run.

    ``progress_*.json`` sits in the same directory and is an analysis artifact
    with no ``config``, so membership is decided by the key rather than by a
    glob that would silently start including the next such file.
    """
    for f in sorted(RUNS.glob("p*.json")):
        payload = json.loads(f.read_text())
        if "config" in payload:
            yield f.stem, payload["config"]


def tex_int(n):
    """``9409`` -> the regex for how the paper types it: ``9{,}409``."""
    return re.escape(f"{n:,}".replace(",", "{,}"))


# -- the model ---------------------------------------------------------------

def test_the_model_shape_the_setup_states_is_the_one_the_code_builds():
    cfg = ModelConfig()
    setup_says(r"\$d_\{\\text\{model\}\} = " + str(cfg.d_model) + r"\$")
    setup_says(rf"{cfg.n_heads} heads of width {cfg.d_head}\b")
    setup_says(rf"an MLP of width {cfg.d_mlp}\b")
    setup_says(rf"a vocabulary\s+of {cfg.vocab_size} ")
    setup_says(rf"the length-{cfg.seq_len} sequence")
    assert cfg.n_layers == 1, "Section 2 calls the model one-layer"
    setup_says(r"one-layer decoder-only transformer")


def test_the_parameter_count_the_setup_states_is_the_model_s_own():
    n = sum(p.numel() for p in Transformer(ModelConfig()).parameters())
    setup_says(r"for " + tex_int(n) + r"\s+parameters")
    # Section 5.1 quotes the same count as the invariant across head-count arms,
    # so the two statements have to stay one number.
    assert f"{n:,}".replace(",", "{,}") in TEX.split(r"\label{sec:heads}", 1)[1]


def test_dropout_is_off_by_default_as_the_setup_claims():
    setup_says(r"Dropout is \$0\.0\$")
    assert ModelConfig().dropout == 0.0


# -- the data ----------------------------------------------------------------

def test_the_dataset_sizes_the_setup_states_are_the_ones_the_split_produces():
    p = TrainConfig().p
    tokens, targets = modular_dataset(p)
    setup_says(r"All \$p\^2 = " + tex_int(len(tokens)) + r"\$ ordered pairs")

    (tr_x, _), (te_x, _) = train_test_split(tokens, targets, 0.30, seed=0)
    n_tr, n_te = len(tr_x), len(te_x)
    assert n_tr + n_te == len(tokens)
    setup_says(
        r"at\s+30\\% it is " + tex_int(n_tr) + r" training pairs "
        r"against " + tex_int(n_te) + r" test"
    )


def test_the_chance_level_the_setup_states_is_one_over_the_modulus():
    p = TrainConfig().p
    m = setup_says(r"chance is \$1/(\d+) = ([\d.]+)\$")
    assert int(m.group(1)) == p
    assert float(m.group(2)) == pytest.approx(1 / p, abs=5e-5)


# -- the definitions ---------------------------------------------------------

@pytest.mark.parametrize(
    "pattern, value",
    [
        (r"training accuracy reaches\s+([\d.]+)\\%", MEMORIZE_ACC),
        (r"test accuracy reaches\s+([\d.]+)\\%", GROK_ACC),
        (r"holds \$\\ge ([\d.]+)\\%\$", EARLY_STOP_ACC),
    ],
)
def test_the_thresholds_the_setup_defines_are_the_ones_training_applies(pattern, value):
    """The prose gives percentages; ``train`` compares fractions."""
    assert float(setup_says(pattern).group(1)) / 100 == pytest.approx(value)


def test_the_budgets_the_setup_states_are_the_experiments_own():
    """25,000 steps everywhere, 15,000 for the decay-scope arm."""
    setup_says(r"a budget of 25\{,\}000 steps")
    setup_says(r"\(15\{,\}000 for the decay-scope arm")
    budgets = {}
    for src in sorted((ROOT / "experiments").glob("*.py")):
        m = re.search(r"^MAX_STEPS = ([\d_]+)", src.read_text(), re.M)
        if m:
            budgets[src.stem] = int(m.group(1).replace("_", ""))
    assert budgets.pop("wd_scope") == 15_000
    assert set(budgets.values()) == {25_000}, budgets


# -- the runs behind it ------------------------------------------------------

def test_the_eval_cadence_the_setup_states_is_the_one_every_run_used():
    setup_says(r"evaluated every 100 steps")
    odd = {name: c["eval_every"] for name, c in training_runs()
           if c["eval_every"] != 100}
    assert not odd, odd


def test_the_optimizer_settings_the_setup_states_match_every_committed_run():
    setup_says(r"Full-batch AdamW .*learning\s*rate \$10\^\{-3\}\$")
    setup_says(r"\\beta = \(0\.9, 0\.98\)")
    for name, c in training_runs():
        assert tuple(c["betas"]) == (0.9, 0.98), name
        # The learning-rate sweep is the one arm that varies lr, and it tags
        # its run names accordingly (TrainConfig.run_name), so "lr 1e-3
        # everywhere else" is checkable off the filename.
        if "_lr" not in name:
            assert c["lr"] == pytest.approx(1e-3), name


def test_the_seed_protocol_the_setup_claims_is_the_one_the_runs_show():
    """Five / three / one, per the paragraph that lists which is which."""
    setup_says(r"five for the weight-decay and data-fraction sweeps")
    setup_says(r"three for the \$wd \\times \\mathrm\{frac\}\$ grid")
    setup_says(r"and one for the learning-rate, dropout,\s+modulus and decay-scope controls")

    def seeds(pattern):
        return sorted(
            int(re.search(r"_seed(\d+)", f.stem).group(1))
            for f in RUNS.glob(pattern + ".json")
        )

    five = {
        "wd sweep": "p97_frac0.30_wd[01]*_seed[0-9]",
        "frac 25%": "p97_frac0.25_wd1_seed[0-9]",
        "frac 40%": "p97_frac0.40_wd1_seed[0-9]",
        "heads 1": "p97_frac0.30_wd1_seed[0-9]_h1",
        "heads 2": "p97_frac0.30_wd1_seed[0-9]_h2",
    }
    for arm, pattern in five.items():
        got = set(seeds(pattern))
        assert got == {0, 1, 2, 3, 4}, (arm, sorted(got))

    three = {
        "grid, 25% x wd 0.3": "p97_frac0.25_wd0.3_seed[0-9]",
        "subtraction": "p97_frac0.30_wd1_seed[0-9]_opsub",
        "multiplication": "p97_frac0.30_wd1_seed[0-9]_opmul",
    }
    for arm, pattern in three.items():
        assert seeds(pattern) == [0, 1, 2], (arm, seeds(pattern))

    one = ["p113_frac0.30_wd1_seed[0-9]", "p97_frac0.60_wd1_seed[0-9]",
           "p97_frac0.30_wd1_seed[0-9]_wdsembeddings",
           "p97_frac0.30_wd0_seed[0-9]_do0.1"]
    for pattern in one:
        assert seeds(pattern) == [0], (pattern, seeds(pattern))


def test_the_run_length_asymmetry_the_setup_states_has_the_cause_it_names():
    """Sec. 2's last paragraph blames ``patience``, not luck. Check the knob.

    ``test_paper_numbers.py`` already checks the *lengths* (11,100 against
    1,800-2,100). What is checked here is the sentence's causal claim: seed 0
    was configured to keep going for 30 evaluations past the threshold and the
    others for 5. If someone re-ran seed 0 at patience 5 the lengths test would
    fail loudly; this one fails if the explanation drifts from the config while
    the artifacts stay put.
    """
    setup_says(r"keep training for 30 evaluations\s+past the threshold rather than 5")
    patience = {s: run_config(f"p97_frac0.30_wd1_seed{s}")["patience"] for s in range(5)}
    assert patience[0] == 30
    assert set(patience[s] for s in range(1, 5)) == {5}
