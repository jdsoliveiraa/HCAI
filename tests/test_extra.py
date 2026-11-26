"""The post-hoc analyses (`hcai.extra`) on synthetic results with planted effects: a planted bias is found, and
nothing survives Holm when none is planted."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hcai.analysis import analyse, load_results
from hcai.extra import (
    extra_tables,
    near_cutoff_units,
    prepare,
    rank_adjusted,
    reference_free_gaps,
    repeat_noise,
    subset_gaps,
)
from hcai.runner import ResultRow, write_results
from test_analysis import one, synthetic_results

BORDERLINE = [f"{role}/weak/cv{k:02d}" for role in ("nurse", "accountant") for k in range(1, 6)]


def frame(tmp_path: Path, rows: list[ResultRow], name: str = "results.csv") -> pd.DataFrame:
    path = tmp_path / name
    write_results(rows, path)
    return load_results(path)


def shift(rows: list[ResultRow], identity: str, points: float, base_cvs: list[str] | None = None) -> list[ResultRow]:
    """Add `points` to the Evaluator score of `identity` (on `base_cvs` only, when given)."""
    return [replace(r, score=int(np.clip(r.score + points, 0, 100)))
            if r.agent == "evaluator" and r.score is not None and r.identity == identity
            and (base_cvs is None or r.base_cv in base_cvs) else r for r in rows]


def repeats_of(rows: list[ResultRow], sigma: float, seed: int = 0) -> list[ResultRow]:
    """One repeat of every Evaluator row: the score plus noise of SD sigma·√2, so one score has noise SD sigma."""
    rng = np.random.default_rng(seed)
    return [replace(r, repeat=1, score=int(np.clip(round(r.score + rng.normal(0, sigma * np.sqrt(2))), 0, 100)))
            for r in rows if r.agent == "evaluator" and r.score is not None]


# --- 1. reference-free contrast ----------------------------------------------------------------


def test_a_planted_shift_of_the_reference_is_found_against_the_mean_and_does_not_move_the_others(tmp_path):
    # The C1 artefact: only the reference (pt_m) scores 3 points higher, so every gap against it is about -3.
    df = prepare(frame(tmp_path, shift(synthetic_results(), "pt_m", 3.0)), "pt_m")
    free = reference_free_gaps(df, 500, np.random.default_rng(0))
    ev = free[free["agent"] == "evaluator"]
    for condition in ("C0", "C1"):
        ref = one(ev, condition=condition, identity="pt_m")
        assert ref["estimate"] == pytest.approx(3.0 * 7 / 8, abs=0.6)
        assert ref["p_holm"] < 0.05
        others = ev[(ev["condition"] == condition) & (ev["identity"] != "pt_m")]
        # their true deviation is -3/8 (the mean moved up by 3/8), not the -3 a gap against pt_m shows
        assert ((others["estimate"] + 3.0 / 8).abs() < 1.0).all()
        assert (others["estimate"].abs() < ref["estimate"]).all()
        assert (others["reference"] == "mean of identities").all()
    assert ev.groupby("condition", observed=True)["estimate"].sum().abs().max() < 1e-9  # deviations sum to 0


def test_without_a_planted_effect_no_reference_free_gap_survives_and_equivalence_passes(tmp_path):
    df = prepare(frame(tmp_path, synthetic_results(base_per_cell=20)), "pt_m")
    free = reference_free_gaps(df, 500, np.random.default_rng(0))
    assert (free["p_holm"] >= 0.05).all()
    ev = free[free["agent"] == "evaluator"]
    assert ev["equivalent"].astype(bool).all()


# --- 2. borderline and near-cutoff subsets -------------------------------------------------------


def test_a_bias_planted_on_borderline_cvs_only_is_found_in_the_borderline_subset(tmp_path):
    df = prepare(frame(tmp_path, shift(synthetic_results(), "br_f", -6.0, BORDERLINE)), "pt_m")
    table = subset_gaps(df, "pt_m", BORDERLINE, 500, np.random.default_rng(0))
    border = table[(table["subset"] == "borderline") & (table["agent"] == "evaluator")]
    for contrast in ("vs_reference", "vs_mean"):
        hit = one(border, contrast=contrast, condition="C0", identity="br_f")
        assert hit["estimate"] < -4.0
        assert hit["p_holm"] < 0.05
    # 10 borderline CVs × 2 languages per condition (1 model)
    assert one(border, contrast="vs_reference", condition="C0", identity="br_m")["n_pairs"] == 20


def test_without_a_planted_effect_nothing_survives_in_either_subset(tmp_path):
    df = prepare(frame(tmp_path, synthetic_results()), "pt_m")
    table = subset_gaps(df, "pt_m", BORDERLINE, 500, np.random.default_rng(0))
    assert set(table["subset"]) == {"borderline", "near_cutoff"}
    assert (table["p_holm"] >= 0.05).all(), table[table["p_holm"] < 0.05]


def test_the_near_cutoff_subset_holds_the_units_whose_mean_is_near_the_median(tmp_path):
    df = prepare(frame(tmp_path, shift(synthetic_results(), "br_f", -6.0)), "pt_m")
    # The synthetic CVs sit at two quality levels 20 points apart, far from their median: a wider margin than the
    # core run's 5 points gives the subset enough units.
    units = near_cutoff_units(df, margin=12.0)
    assert units.groupby("condition", observed=True).size().min() >= 10
    assert ((units["unit_mean"] - units["median"]).abs() <= 12.0).all()
    table = subset_gaps(df, "pt_m", [], 500, np.random.default_rng(0), near_margin=12.0)
    near = table[(table["subset"] == "near_cutoff") & (table["contrast"] == "vs_mean")]
    for condition, n in units.groupby("condition", observed=True).size().items():
        assert one(near, condition=condition, identity="br_m")["n_pairs"] == n
    # the planted bias is found there too
    assert one(near, contrast="vs_mean", condition="C0", identity="br_f")["p_holm"] < 0.05


# --- 3. position-adjusted Ranker ----------------------------------------------------------------


def test_a_planted_identity_effect_on_rank_is_found_after_position(tmp_path):
    rows = synthetic_results(gaps={("br_f", "C0"): -5.0, ("br_f", "C1"): -5.0}, position_effect=8.0, seed=2)
    table, joint = rank_adjusted(prepare(frame(tmp_path, rows), "pt_m"))
    for condition in ("C0", "C1"):
        rank = one(table, condition=condition, outcome="rank", identity="br_f")
        assert rank["estimate"] > 0.5  # ranked worse than the identity mean
        assert rank["p_holm"] < 0.05
        assert one(table, condition=condition, outcome="top3_pp", identity="br_f")["estimate"] < 0
        assert one(joint, condition=condition, outcome="rank")["p_holm"] < 0.05


def test_a_position_effect_alone_gives_no_identity_effect(tmp_path):
    table, joint = rank_adjusted(prepare(frame(tmp_path, synthetic_results(position_effect=8.0, seed=2)), "pt_m"))
    assert len(table) == 2 * 2 * 8  # conditions × outcomes × identities
    assert (table["p_holm"] >= 0.05).all(), table[table["p_holm"] < 0.05]
    assert (joint["p_holm"] >= 0.05).all()
    assert (joint["position_wald_chi2"] > 0).all()


# --- 4. noise against the gaps -------------------------------------------------------------------


def test_the_repeat_noise_is_recovered_and_a_planted_gap_larger_than_it_is_not_inside_noise(tmp_path):
    rows = synthetic_results(gaps={("br_f", "C0"): -6.0, ("br_f", "C1"): -6.0})
    results = frame(tmp_path, rows)
    repeats = frame(tmp_path, repeats_of(rows, sigma=2.0), "repeats.csv")
    noise = repeat_noise(results, repeats, 500, np.random.default_rng(0))
    for condition in ("C0", "C1"):
        n = one(noise, condition=condition, model="all")
        assert n["sigma_ci95_low"] < 2.0 < n["sigma_ci95_high"]
    a = analyse(results, "pt_m", n_boot=500, n_model_boot=0, n_perm=99)
    t = extra_tables(results, repeats, a.gaps, "pt_m", [], n_boot=500)
    table = t["extra_noise_vs_gap"]
    hit = one(table, contrast="vs_reference", condition="C0", identity="br_f")
    assert not hit["inside_noise"]
    assert hit["gap_over_noise_sd"] == pytest.approx(-6.0 / (2.0 * np.sqrt(2)), abs=0.6)
    assert hit["noise_se_mean"] == pytest.approx(hit["noise_sd_pair"] / np.sqrt(hit["n_pairs"]))
    others = table[(table["identity"] != "br_f") & (table["contrast"] == "vs_reference")]
    assert others["inside_noise"].all()


def test_without_a_planted_effect_every_gap_is_inside_noise(tmp_path):
    rows = synthetic_results(base_per_cell=20)
    results = frame(tmp_path, rows)
    repeats = frame(tmp_path, repeats_of(rows, sigma=3.0), "repeats.csv")
    a = analyse(results, "pt_m", n_boot=500, n_model_boot=0, n_perm=99)
    t = extra_tables(results, repeats, a.gaps, "pt_m", [], n_boot=500)
    assert t["extra_noise_vs_gap"]["inside_noise"].all()
    assert set(t["extra_noise_vs_gap"]["contrast"]) == {"vs_reference", "vs_mean"}
