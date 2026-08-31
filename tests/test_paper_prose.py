"""The paper's single-run prose numbers must agree with the trajectories.

Three tests already guard three classes of number, and they leave one gap.
``test_paper.py`` checks the paper against the repo (figures, refs, cites).
``test_paper_numbers.py`` recomputes every grok step the paper *tabulates*.
``test_paper_setup.py`` checks the Setup section against the code and configs.
None of them touches the numbers typed into the results prose and the figure
captions -- the losses, accuracies, norms and crossing points read off a single
committed trajectory -- and that is where the last four corrections came from:

* a caption asserting that the parameter norm "rises to the transition" when
  the plotted series peaks at 1.9x the grok step,
* the same shape asserted in the section that cites the caption,
* "ends at 0.36 test accuracy" for a run whose log says 0.3545,
* an abstract that turned "groks at step 1,300" into "1,300 steps after
  memorizing", which is 100 steps too many.

Every one of them is a number that no figure regeneration and no table check
could have caught, because none of them lives in a figure or a table. So the
prose gets its own instrument: the claims below are recomputed from
``runs/*.csv`` and matched against the typeset text. Where the paper states a
*shape* rather than a value ("rises through the transition", "climbs at nearly
every evaluation"), the shape is asserted here as an inequality, since a shape
is exactly the kind of claim that survives a figure regeneration unchallenged.

Pure stdlib on purpose -- no matplotlib, no torch -- so this runs in CI's
ordinary test job alongside the other text checks.
"""

import csv
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
TEX = (ROOT / "paper" / "main.tex").read_text()


# -- reading the runs --------------------------------------------------------

def history(name):
    """A run's per-eval log as {column: [values]}, steps as ints."""
    with open(RUNS / f"{name}.csv") as f:
        rows = list(csv.DictReader(f))
    out = {k: [float(r[k]) for r in rows] for k in rows[0]}
    out["step"] = [int(s) for s in out["step"]]
    return out


def summary(name):
    return json.loads((RUNS / f"{name}.json").read_text())


def at(h, column, step):
    """The value of ``column`` at exactly ``step``."""
    return h[column][h["step"].index(step)]


MAIN = "p97_frac0.30_wd1_seed0"
PROGRESS = "progress_p97_frac0.30_wd1_seed0"
ATTENTION = "attention_p97_frac0.30_wd1_seed0"


# -- reading the paper -------------------------------------------------------

FLAT = re.sub(r"\s+", " ", TEX)


def says(*fragments):
    """Assert each fragment appears in main.tex, reporting the one that does not.

    Numbers are quoted exactly as the paper typesets them, thousands separator
    and all, so a corrected log fails the test that quotes it rather than
    passing a laxer match. Whitespace is the one thing normalized away, on both
    sides: a fragment must not stop matching because the sentence around it
    rewrapped.
    """
    for fragment in fragments:
        flat = re.sub(r"\s+", " ", fragment)
        assert flat in FLAT, f"main.tex no longer says {fragment!r}"


def tex_number(pattern):
    """The single number matched by ``pattern``'s one capture group."""
    found = re.findall(pattern, TEX)
    assert len(found) == 1, f"{pattern!r} matched {len(found)} times, want 1"
    return float(found[0].replace("{,}", "").replace(",", ""))


# -- the norm trajectory -----------------------------------------------------
#
# The correction this module was written for. The claim under test is a shape:
# the norm keeps rising *past* the transition, so the peak is well after the
# grok step and the decline is entirely post-transition.

def test_the_norm_peaks_after_the_transition_not_at_it():
    h, s = history(MAIN), summary(MAIN)
    peak = max(range(len(h["weight_norm"])), key=lambda i: h["weight_norm"][i])
    peak_step = h["step"][peak]

    assert peak_step > s["grok_step"], (
        "the paper and the figure title both say the norm peaks after the "
        f"transition; peak is at {peak_step}, grok step {s['grok_step']}"
    )
    says("peak is at step 3{,}700", "$1.9\\times$ the")
    assert peak_step == 3700
    assert round(peak_step / s["grok_step"], 1) == 1.9


def test_the_norm_levels_the_paper_quotes_are_the_run_s():
    h = history(MAIN)
    assert round(at(h, "weight_norm", 100), 1) == 24.1
    assert round(at(h, "weight_norm", 1900), 1) == 27.8
    assert round(max(h["weight_norm"]), 1) == 32.9
    says("norm is $24.1$ at", "$27.8$ at the grok step",
         "a peak of $32.9$ at step 3{,}700")


def test_the_drawdown_the_paper_quotes_is_peak_to_final():
    h = history(MAIN)
    peak = max(h["weight_norm"])
    drawdown = 1 - h["weight_norm"][-1] / peak
    assert round(100 * drawdown) == 22
    says("$22\\%$ over the remaining")


def test_the_peak_survives_smoothing_as_the_paper_claims():
    """The peak is one eval of a noisy series, so the paper says it is stable
    under smoothing and names the bin. That is the claim checked here."""
    h = history(MAIN)
    bins = {}
    for step, norm in zip(h["step"], h["weight_norm"]):
        bins.setdefault(step // 1000, []).append(norm)
    medians = {b: sorted(v)[len(v) // 2] for b, v in bins.items()}
    best = max(medians, key=lambda b: medians[b])
    assert best == 3, f"highest 1000-step median is in bin {best}000, not 3000"
    says("$3{,}000$--$4{,}000$ bin")


def test_only_the_long_run_shows_a_sustained_decline():
    """Section 5's qualifier: two of the four short seeds peak at their own
    last evaluation, and none of the four falls more than 6.8% from its peak,
    so their drawdowns are dips rather than a trend."""
    peaks_at_end = 0
    finals = []
    for seed in (1, 2, 3, 4):
        h = history(f"p97_frac0.30_wd1_seed{seed}")
        norm = h["weight_norm"]
        peak = max(range(len(norm)), key=lambda i: norm[i])
        peaks_at_end += peak == len(norm) - 1
        finals.append(100 * (1 - norm[-1] / norm[peak]))
    assert peaks_at_end == 2
    assert round(min(finals), 1) == 0.0 and round(max(finals), 1) == 6.8
    says("two of them peaking at their own last evaluation", "0.0--6.8\\%")


def test_the_maximum_drawdowns_are_the_range_the_paper_gives():
    lows = []
    for seed in range(5):
        h = history(f"p97_frac0.30_wd1_seed{seed}")
        peak, worst = 0.0, 0.0
        for norm in h["weight_norm"]:
            peak = max(peak, norm)
            worst = max(worst, 1 - norm / peak)
        lows.append(100 * worst)
    assert round(min(lows), 1) == 7.1 and round(max(lows), 1) == 23.2
    says("maximum drawdowns of 7.1--23.2\\%")


def test_the_long_wd0_run_the_comparison_rests_on_ran_longer_than_seed_0():
    """The wd=0 arm's monotone norm is only the stronger claim if the run that
    carries it is at least as long as the one it is compared against."""
    longest = max((summary(f"p97_frac0.40_wd0_seed{s}") for s in range(3)),
                  key=lambda j: j["steps_run"])
    assert longest["steps_run"] == 19300
    assert longest["steps_run"] > summary(MAIN)["steps_run"]
    says("19{,}300 steps, nearly twice as many")


def test_the_training_loss_refutes_the_tidier_mechanism():
    """Section 5 says the "decay is the only force left" account puts the
    decline two thousand steps early. That rests on the training loss already
    being negligible long before the norm turns over."""
    h = history(MAIN)
    assert 1e-1 < at(h, "train_loss", 100) < 2e-1
    assert 1e-2 < at(h, "train_loss", 200) < 3e-2
    assert 1e-4 < at(h, "train_loss", 1500) < 3e-4
    peak_step = h["step"][max(range(len(h["weight_norm"])),
                              key=lambda i: h["weight_norm"][i])]
    assert peak_step - 1500 > 2000
    says("${\\sim}10^{-1}$ at step 100, ${\\sim}2 \\times 10^{-2}$ by step 200",
         "${\\sim}10^{-4}$ by step 1{,}500")


# -- the loss figure ---------------------------------------------------------

def test_the_loss_figure_s_caption_numbers_are_the_run_s():
    h = history(MAIN)
    peak = max(h["test_loss"])
    assert h["step"][h["test_loss"].index(peak)] == summary(MAIN)["memorize_step"]
    assert round(peak, 2) == 5.03
    assert round(at(h, "test_loss", 1500), 2) == 1.86
    assert round(peak / at(h, "test_loss", 1500), 1) == 2.7
    says("maximum of $5.03$ at the memorization step",
         "falls by $2.7\\times$", "to $1.86$ by step\n  1{,}500")


# -- the progress measures ---------------------------------------------------

def test_the_restricted_loss_starts_ahead_by_the_margin_quoted():
    h = history(PROGRESS)
    full, restricted = at(h, "test_loss", 100), at(h, "restricted_loss", 100)
    assert round(full, 2) == 5.03 and round(restricted, 2) == 4.10
    assert round(full - restricted, 1) == 0.9
    says("test loss is $5.03$ while the restricted", "loss is $4.10$")


def test_the_full_model_overtakes_the_restricted_one_where_the_paper_says():
    h = history(PROGRESS)
    crossings = [step for step, full, restricted
                 in zip(h["step"], h["test_loss"], h["restricted_loss"])
                 if step >= 100 and full < restricted]
    assert crossings[0] == 1100, f"first crossing at {crossings[0]}"
    assert round(at(h, "test_loss", 1100), 2) == 3.29
    assert round(at(h, "restricted_loss", 1100), 2) == 3.31
    says("overtakes the restricted", "at step 1{,}100 ---\n$3.29$ against $3.31$")


def test_the_accuracy_jump_is_where_the_paper_puts_it():
    """The jump is the midpoint crossing, not the 99% crossing that defines the
    grok step -- 500 steps after the loss crossing and 300 before the grok.

    "Midpoint" is the midpoint of the accuracy's *own* rise, from its value at
    memorization to its final value, which is 0.58 here and not 0.5. Spelling
    it out rather than importing ``progress_measures.accuracy_jump_step``
    keeps this module free of matplotlib, and re-deriving it is the point: a
    threshold copied from the prose would agree with the prose by
    construction.
    """
    h, s = history(PROGRESS), summary(MAIN)
    memorize = s["memorize_step"]
    midpoint = 0.5 * (at(h, "test_acc", memorize) + h["test_acc"][-1])
    jump = next(step for step, acc in zip(h["step"], h["test_acc"])
                if step > memorize and acc > midpoint)
    assert round(midpoint, 2) == 0.58
    assert jump == 1600
    assert jump - 1100 == 500
    assert jump < s["grok_step"]
    says("500 steps before the\naccuracy jump at 1{,}600")


def test_the_delay_is_a_climb_and_not_a_plateau():
    h = history(PROGRESS)
    window = [(step, acc) for step, acc in zip(h["step"], h["test_acc"])
              if 100 <= step <= 1500]
    assert round(window[0][1], 3) == 0.163
    assert round(window[-1][1], 3) == 0.527
    falls = sum(b[1] <= a[1] for a, b in zip(window, window[1:]))
    assert falls == 1, f"{falls} non-increasing evals, the paper says one"
    says("$0.163$ at memorization to $0.527$ at step", "rising at every eval but one")


def test_the_pre_jump_share_of_the_embedding_rise_is_the_generous_one():
    """The share is taken at the read-out's *best* pre-jump value, because that
    is the number hardest for the "it rises early" reading, not the kindest."""
    h = history(PROGRESS)
    start = at(h, "emb_top_frac", 100)
    best_pre = max(v for s, v in zip(h["step"], h["emb_top_frac"]) if s < 1600)
    top = max(h["emb_top_frac"])
    assert (round(start, 3), round(best_pre, 3), round(top, 3)) == (0.136, 0.181, 0.477)
    assert round(100 * (best_pre - start) / (top - start)) == 13
    says("$0.181$ at step 900, against $0.136$ and $0.477$", "only $13\\%$ of the rise")


def test_removing_five_frequencies_collapses_the_grokked_model():
    h = history(PROGRESS)
    post = [(full, excluded) for step, full, excluded
            in zip(h["step"], h["test_loss"], h["excluded_loss"]) if step >= 1900]
    fulls = sorted(f for f, _ in post)
    excludeds = sorted(e for _, e in post)
    assert fulls[len(fulls) // 2] < 1e-1
    assert round(excludeds[len(excludeds) // 2], 2) == 0.97
    says("excluded loss\nholds a median of $0.97$")


# -- attention ---------------------------------------------------------------

def test_the_attention_read_out_lags_the_transition():
    h, s = history(ATTENTION), summary(MAIN)
    peak = max(range(len(h["attn_asymmetry"])),
               key=lambda i: h["attn_asymmetry"][i])
    ln2 = 0.6931471805599453
    restored = next(step for step, ent in
                    zip(h["step"][peak:], h["attn_operand_entropy"][peak:])
                    if ent >= 0.999 * ln2)
    half = next(step for step, acc in zip(h["step"], h["test_acc"]) if acc >= 0.5)
    assert restored == 2300 and half == 1500
    assert restored > s["grok_step"] > half
    says("symmetry is restored at step 2{,}300")


def test_the_entropy_rises_through_grokking():
    h = history(ATTENTION)
    assert round(at(h, "attn_entropy", 100), 3) == 0.677
    assert round(h["attn_entropy"][-1], 3) == 0.935
    says("$0.677 \\to\n0.935$")


def test_memorization_breaks_a_symmetry_initialization_already_had():
    h = history(ATTENTION)
    assert round(h["attn_asymmetry"][0], 3) == 0.004
    assert round(at(h, "attn_asymmetry", 100), 3) == 0.189
    says("from $0.004$ at\ninitialization to $0.189$ at the memorization checkpoint")


# -- the single-seed control arms --------------------------------------------
#
# Stated as directions in the paper, but the levels are still typed in, and a
# typed level is a claim.

@pytest.mark.parametrize("name, grok, final_acc", [
    ("runs_lr/p97_frac0.30_wd1_seed0_lr0.0003", 5500, None),
    ("runs_lr/p97_frac0.30_wd1_seed0", 1700, None),
    ("runs_lr/p97_frac0.30_wd1_seed0_lr0.003", 800, None),
])
def test_the_learning_rate_arm_groks_where_the_paper_says(name, grok, final_acc):
    got = json.loads((ROOT / f"{name}.json").read_text())["grok_step"]
    assert got == grok
    says("5{,}500 / 1{,}700 / 800")


def test_the_dropout_control_groks_with_a_norm_that_only_rises():
    h, s = history("p97_frac0.30_wd0_seed0_do0.1"), summary("p97_frac0.30_wd0_seed0_do0.1")
    assert s["grok_step"] == 3500
    assert round(s["final_test_acc"], 3) == 0.999
    norm = h["weight_norm"]
    assert max(norm) == norm[-1], "the norm must end at its own maximum"
    assert round(norm[0], 1) == 21.7 and round(norm[-1], 1) == 55.2
    says("groks, at step 3{,}500", "$21.7 \\to 55.2$")


def test_the_modulus_control_groks_sooner_and_is_not_softer():
    s = summary("p113_frac0.30_wd1_seed0")
    h = history("p113_frac0.30_wd1_seed0")
    assert s["grok_step"] == 600
    assert round(at(h, "test_acc", s["memorize_step"]), 3) == 0.288
    assert round(113 * 113 * 0.30) == 3831
    says("at step 600 against 1{,}900", "accuracy at memorization is $0.288$",
         "3{,}831 against 2{,}823")


def test_the_scope_arms_land_where_the_paper_says():
    non_emb = summary("p97_frac0.30_wd1_seed0_wdsnon_embeddings")
    emb = summary("p97_frac0.30_wd1_seed0_wdsembeddings")
    h = history("p97_frac0.30_wd1_seed0_wdsembeddings")
    assert non_emb["grok_step"] == 1800
    assert emb["grok_step"] is None and emb["steps_run"] == 15000
    assert round(emb["final_test_acc"], 2) == 0.35
    assert round(h["weight_norm"][0], 1) == 21.7
    assert round(h["weight_norm"][-1]) == 287
    says("1{,}800 against 1{,}900", "ends at $0.35$ test accuracy",
         "climbs from $21.7$ to $287$")


# -- the abstract ------------------------------------------------------------

def test_the_abstract_s_delay_is_the_grok_step_and_not_the_gap():
    """The abstract used to say the model "does not reach 99% test accuracy for
    another 1,300 steps" after memorizing at 100, which is the grok step read
    as a gap. The two differ by exactly the memorization step."""
    steps = sorted(summary(f"p97_frac0.30_wd1_seed{s}")["grok_step"] for s in range(5))
    median = steps[len(steps) // 2]
    assert median == 1300
    assert median // summary(MAIN)["memorize_step"] == 13
    says("until step 1{,}300 (median over five seeds), a delay of",
         "$13\\times$ the time it took to memorize")
    assert "for another 1{,}300 steps" not in TEX


def test_the_abstract_s_order_of_magnitude_is_the_measured_ratio():
    def median_grok(pattern, seeds):
        got = sorted(summary(pattern.format(s=s))["grok_step"] for s in seeds)
        return got[len(got) // 2]

    wd = median_grok("p97_frac0.30_wd0.1_seed{s}", range(5)) / \
        median_grok("p97_frac0.30_wd1_seed{s}", range(5))
    frac = median_grok("p97_frac0.25_wd1_seed{s}", range(5)) / \
        median_grok("p97_frac0.40_wd1_seed{s}", range(5))
    assert round(wd, 1) == 8.3 and round(frac, 1) == 9.0
    says("$8.3\\times$ from $wd = 0.1$ to $wd = 1$, $9.0\\times$ from")
