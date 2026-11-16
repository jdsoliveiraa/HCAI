"""Seam 3, analysis: ranking quality (RQ3) from synthetic results tables with known Evaluator scores and Ranker orders."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd
import pytest

from hcai.analysis import results_frame
from hcai.quality import ranking_quality
from hcai.runner import ResultRow
from test_analysis import _row, one, run, synthetic_results

QUALITY_SCORE = {"strong": 80, "average": 60, "weak": 40}
IDENTITIES = [("pt_m", "male", "portuguese"), ("br_f", "female", "brazilian")]

Scorer = Callable[[str, np.random.Generator], float]


def evaluator_table(scorers: dict[str, Scorer], *, n_cvs: int = 8, models: tuple[str, ...] = ("model-a",),
                    seed: int = 0) -> list[ResultRow]:
    """Evaluator rows of `n_cvs` base CVs per quality level, scored per condition by `scorers[condition]`."""
    rng = np.random.default_rng(seed)
    rows = []
    for quality in QUALITY_SCORE:
        for k in range(1, n_cvs + 1):
            for condition, scorer in scorers.items():
                for model in models:
                    for ident, gender, origin in IDENTITIES:
                        rows.append(_row(base_cv=f"nurse/{quality}/cv{k:02d}", identity=ident, gender=gender,
                                         origin=origin, name=f"{ident}-name", language="en", condition=condition,
                                         model=model, agent="evaluator", score=int(round(scorer(quality, rng)))))
    return rows


def mixed_pools(orders: dict[str, list[str]], *, n_pools: int = 6, rejected: str | None = None) -> list[ResultRow]:
    """Ranker rows of mixed-quality pools (one strong, one average, one weak base CV), ranked per condition in
    the quality order `orders[condition]` (best first). A `rejected` quality level is screened out of every pool."""
    rows = []
    for k in range(1, n_pools + 1):
        for condition, order in orders.items():
            members = [q for q in order if q != rejected]
            for position, quality in enumerate(QUALITY_SCORE, start=1):
                ident, gender, origin = IDENTITIES[position % 2]
                common = dict(base_cv=f"nurse/{quality}/cv{k:02d}", identity=ident, gender=gender, origin=origin,
                              name=f"{ident}-name", language="en", condition=condition, model="model-a",
                              agent="ranker", pool=f"nurse/mixed{k:02d}", pool_kind="mixed", order_seed=0)
                if quality == rejected:
                    rows.append(_row(**common, status="rejected"))
                    continue
                rank = members.index(quality) + 1
                rows.append(_row(**common, input_position=position, rank=rank, top3=rank <= 3))
    return rows


def by_quality(scores: Mapping[str, float]) -> Scorer:
    return lambda quality, rng: scores[quality]


def noisy(sd: float) -> Scorer:
    return lambda quality, rng: QUALITY_SCORE[quality] + rng.normal(0, sd)


def quality(rows: list[ResultRow], **kwargs) -> pd.DataFrame:
    return ranking_quality(results_frame(rows), **{"n_boot": 300, "seed": 0, **kwargs})


def test_spearman_is_one_for_scores_that_follow_quality_and_minus_one_for_reversed_scores():
    table = quality(evaluator_table({
        "C0": by_quality(QUALITY_SCORE),
        "C1": by_quality({"strong": 40, "average": 60, "weak": 80}),
    }))

    c0 = one(table, metric="spearman", condition="C0", model="all")
    c1 = one(table, metric="spearman", condition="C1", model="all")
    assert c0["agent"] == "evaluator" and c0["reference"] == "C0"
    assert c0["estimate"] == pytest.approx(1.0) and c1["estimate"] == pytest.approx(-1.0)
    assert c1["delta"] == pytest.approx(-2.0)
    assert c1["delta_ci95_low"] == pytest.approx(-2.0) and c1["delta_ci95_high"] == pytest.approx(-2.0)
    assert c0["n"] == 3 * 8 * 2 and c0["n_clusters"] == 3 * 8
    assert c0["delta"] == 0.0


def test_a_mitigation_that_blurs_the_scores_loses_spearman_with_a_ci_below_zero():
    table = quality(evaluator_table({"C0": noisy(3), "C1": noisy(3), "C2": noisy(25)}, n_cvs=10))

    c0, c1, c2 = (one(table, metric="spearman", condition=c, model="all") for c in ("C0", "C1", "C2"))
    assert c0["estimate"] > 0.85 and c0["ci95_low"] - 0.01 < c0["estimate"] < c0["ci95_high"] + 0.01
    assert c2["estimate"] < 0.75 and c2["delta_ci95_high"] < 0
    assert c1["delta_ci95_low"] < 0 < c1["delta_ci95_high"]  # same noise as C0: no change


def test_pairwise_accuracy_counts_the_ranker_pairs_ordered_by_true_quality_on_mixed_pools():
    rows = mixed_pools({
        "C0": ["strong", "average", "weak"],   # 3 of 3 pairs right
        "C1": ["average", "strong", "weak"],   # strong < average wrong: 2 of 3
        "C2": ["strong", "average", "weak"],
    })

    table = quality(rows)

    c0, c1, c2 = (one(table, metric="pairwise_accuracy", condition=c, model="all") for c in ("C0", "C1", "C2"))
    assert c0["agent"] == "ranker"
    assert c0["estimate"] == 1.0 and c0["n"] == 6 * 3 and c0["n_clusters"] == 6
    assert c1["estimate"] == pytest.approx(2 / 3) and c1["delta"] == pytest.approx(-1 / 3)
    assert not c1["within_tolerance"]  # loses more than the pre-registered 5 percentage points
    assert c2["delta"] == 0.0 and c2["within_tolerance"]


def test_pairwise_accuracy_skips_screened_out_cvs_and_equal_quality_pools():
    rows = mixed_pools({"C0": ["weak", "average", "strong"]}, rejected="average")
    rows += synthetic_results(conditions=("C0",), languages=("en",), roles=("nurse",), base_per_cell=2)

    c0 = one(quality(rows), metric="pairwise_accuracy", condition="C0", model="all")

    assert c0["estimate"] == 0.0 and c0["n"] == 6  # only weak vs strong is left in each pool, ranked wrong


def test_ranking_quality_is_reported_per_model_and_pooled():
    rows = evaluator_table({"C0": by_quality(QUALITY_SCORE), "C1": by_quality(QUALITY_SCORE)},
                           models=("model-a", "model-b"))

    table = quality(rows)

    assert set(zip(table["condition"], table["model"])) == {
        (c, m) for c in ("C0", "C1") for m in ("all", "model-a", "model-b")}
    assert one(table, metric="spearman", condition="C1", model="model-b")["n"] == 3 * 8 * 2


def test_quality_levels_outside_the_order_are_left_out_and_the_order_is_configurable():
    rows = evaluator_table({"C0": by_quality(QUALITY_SCORE)})

    assert quality(rows, quality_order=("high", "low")).empty
    reversed_order = quality(rows, quality_order=("weak", "average", "strong"))
    assert one(reversed_order, metric="spearman", condition="C0", model="all")["estimate"] == pytest.approx(-1.0)


def test_the_analysis_writes_the_ranking_quality_table(tmp_path):
    a = run(tmp_path, synthetic_results(conditions=("C0", "C1"), base_per_cell=4))

    spearman = one(a.ranking_quality, metric="spearman", condition="C0", model="all")
    assert spearman["estimate"] > 0.5  # synthetic scores: strong 80 vs weak 60 plus noise
    assert "ranking_quality" in {p.stem for p in a.write(tmp_path / "out")}


def test_the_spearman_ci_is_centred_on_the_estimate_not_biased_by_resampling_the_quality_mix():
    table = quality(evaluator_table({"C0": noisy(12)}, n_cvs=9, models=("model-a", "model-b")), n_boot=1000)

    c0 = one(table, metric="spearman", condition="C0", model="all")
    width = c0["ci95_high"] - c0["ci95_low"]
    assert c0["estimate"] - c0["ci95_low"] > 0.3 * width and c0["ci95_high"] - c0["estimate"] > 0.3 * width


def test_the_core_design_builds_mixed_quality_pools_for_pairwise_accuracy():
    from conftest import core_config
    from hcai.config import load_design, pilot_design

    core = load_design(core_config())

    assert core.pools.mixed_quality > 0 and pilot_design(core).pools.mixed_quality > 0
