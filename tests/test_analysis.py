"""Seam 3, analysis: synthetic results tables with planted effects in, fairness tables out.

Every test writes a synthetic results table with the runner's own writer and reads it back through
`load_results`, so the analysis is exercised on exactly the CSV contract the runner produces.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hcai.analysis import Analysis, analyse, load_results, main
from hcai.runner import ResultRow, write_results

GENDERS = {"m": "male", "f": "female"}
ORIGINS = {"pt": "portuguese", "br": "brazilian", "palop": "palop", "ee": "eastern_european"}
IDENTITIES = [(f"{o}_{g}", gender, origin) for o, origin in ORIGINS.items() for g, gender in GENDERS.items()]
QUALITY_MEAN = {"strong": 80.0, "weak": 60.0}


def synthetic_results(
    *,
    gaps: dict[tuple[str, str], float] | None = None,
    position_effect: float = 0.0,
    conditions: tuple[str, ...] = ("C0", "C1"),
    languages: tuple[str, ...] = ("en", "pt"),
    models: tuple[str, ...] = ("model-a",),
    roles: tuple[str, ...] = ("nurse", "accountant"),
    base_per_cell: int = 10,
    screen_threshold: int = 45,
    seed: int = 0,
) -> list[ResultRow]:
    """A results table from a known generative model.

    Evaluator score = quality mean + CV effect + name effect + noise + the planted gap of
    (identity, condition). The Screener passes scores >= `screen_threshold`. The Ranker orders each equal-quality pool
    by score + noise + a bonus of `position_effect` points for input position 1, falling linearly to 0
    at the last position. Two names per identity, alternating over base CVs (so `base_per_cell` is even).
    """
    gaps = gaps or {}
    rng = np.random.default_rng(seed)
    names = {ident: [f"{ident}-name{k}" for k in (1, 2)] for ident, _, _ in IDENTITIES}
    # Name effects are centred within each identity, so an identity's true effect is its planted gap.
    name_effect = {}
    for first, second in names.values():
        e = rng.normal(0, 1)
        name_effect[first], name_effect[second] = e, -e
    rows: list[ResultRow] = []
    for role in roles:
        for quality, quality_mean in QUALITY_MEAN.items():
            for k in range(1, base_per_cell + 1):
                base_cv = f"{role}/{quality}/cv{k:02d}"
                cv_effect = rng.normal(0, 4)
                for language in languages:
                    for condition in conditions:
                        for model in models:
                            key = dict(language=language, condition=condition, model=model)
                            scores: dict[str, int] = {}
                            for ident, gender, origin in IDENTITIES:
                                name = names[ident][k % 2]
                                latent = quality_mean + cv_effect + name_effect[name] + rng.normal(0, 3)
                                score = int(round(np.clip(latent + gaps.get((ident, condition), 0.0), 0, 100)))
                                passed = score >= screen_threshold
                                common = dict(base_cv=base_cv, identity=ident, gender=gender, origin=origin, name=name, **key)
                                rows.append(_row(**common, agent="screener", screen_pass=passed))
                                rows.append(_row(**common, agent="evaluator", score=score if passed else None,
                                                 status="ok" if passed else "rejected"))
                                scores[ident] = score
                            rows += _rank(rng, base_cv, scores, names, k, position_effect, key, screen_threshold)
    return rows


def _rank(rng, base_cv, scores, names, k, position_effect, key, screen_threshold) -> list[ResultRow]:
    eligible = [i for i, s in scores.items() if s >= screen_threshold]
    order = list(rng.permutation(eligible))
    n = len(order)
    utility = {
        ident: scores[ident] + rng.normal(0, 1) + (position_effect * (n - pos) / (n - 1) if n > 1 else 0.0)
        for pos, ident in enumerate(order, start=1)
    }
    ranked = sorted(order, key=lambda i: -utility[i])
    by_id = {ident: (gender, origin) for ident, gender, origin in IDENTITIES}
    rows = []
    for ident, (gender, origin) in by_id.items():
        common = dict(base_cv=base_cv, identity=ident, gender=gender, origin=origin, name=names[ident][k % 2],
                      agent="ranker", pool=base_cv, pool_kind="equal", order_seed=0, **key)
        if ident in order:
            rank = ranked.index(ident) + 1
            rows.append(_row(**common, input_position=order.index(ident) + 1, rank=rank, top3=rank <= 3))
        else:
            rows.append(_row(**common, status="rejected"))
    return rows


def _row(**values) -> ResultRow:
    defaults = dict(score=None, rationale="", prompt_tokens=0, completion_tokens=0, cost=0.0, cached=False,
                    status="ok", error="")
    return ResultRow(**{**defaults, **values})


def run(tmp_path: Path, rows: list[ResultRow], **kwargs) -> Analysis:
    path = tmp_path / "results.csv"
    write_results(rows, path)
    return analyse(load_results(path), **{"reference": "pt_m", "n_boot": 500, "n_model_boot": 0, "n_perm": 999, **kwargs})


def one(table: pd.DataFrame, **match) -> pd.Series:
    hits = table
    for column, value in match.items():
        hits = hits[hits[column] == value]
    assert len(hits) == 1, f"expected one row for {match}, got {len(hits)}"
    return hits.iloc[0]


def test_a_planted_five_point_gap_is_detected_with_its_direction_magnitude_and_impact_ratio(tmp_path):
    a = run(tmp_path, synthetic_results(gaps={("br_f", "C0"): -5.0, ("br_f", "C1"): -5.0}))

    for condition in ("C0", "C1"):
        gap = one(a.gaps, agent="evaluator", condition=condition, identity="br_f")
        assert gap["outcome"] == "score"
        assert gap["estimate"] == pytest.approx(-5.0, abs=1.0)
        assert gap["ci95_high"] < 0
        assert gap["p_holm"] < 0.05
        assert not gap["equivalent"]

        ranker = one(a.impact_ratios, agent="ranker", condition=condition, identity="br_f")
        assert ranker["outcome"] == "top3"
        assert ranker["impact_ratio"] < 0.8
        assert ranker["below_threshold"]

    unaffected = one(a.gaps, agent="evaluator", condition="C0", identity="br_m")
    assert unaffected["estimate"] == pytest.approx(0.0, abs=1.0)


def test_with_no_planted_effect_nothing_survives_holm_and_the_equivalence_check_passes(tmp_path):
    # Holm holds the error rate at 5% per family, so with ~10 families an occasional null table has a
    # survivor (1 of seeds 0-9 at this size); seed 0 is not one. 20 base CVs per cell keep the
    # equivalence check from being marginal (it passes on all of seeds 0-9; at 10 per cell, 2 fail).
    a = run(tmp_path, synthetic_results(base_per_cell=20, seed=0))

    tested = {
        "gaps": a.gaps,
        "model terms": a.model_terms[a.model_terms["identity_term"]],
        "identity contrasts": a.identity_contrasts,
        "rank by identity": a.rank_identity,
        "rank by position": a.rank_position,
    }
    for name, table in tested.items():
        assert len(table) > 0, name
        assert (table["p_holm"].dropna() >= 0.05).all(), f"{name}: {table[table['p_holm'] < 0.05]}"
    assert set(a.equivalence["condition"]) == {"C0", "C1"}
    assert a.equivalence["equivalent"].all()
    assert (a.equivalence["margin"] == 2.0).all()


def test_a_planted_position_effect_is_attributed_to_position_not_identity(tmp_path):
    a = run(tmp_path, synthetic_results(position_effect=8.0, seed=2))

    for condition in ("C0", "C1"):
        first = one(a.rank_position, condition=condition, level=1)
        last = one(a.rank_position, condition=condition, level=8)
        assert first["diff"] < -1.0  # shown first, ranked better than the expected 4.5
        assert first["p_holm"] < 0.05
        assert last["diff"] > 0
    assert (a.rank_identity["p_holm"] >= 0.05).all()
    assert (a.gaps.loc[a.gaps["agent"] == "ranker", "p_holm"] >= 0.05).all()


def test_a_planted_mitigation_effect_shows_up_in_the_identity_by_condition_terms(tmp_path):
    a = run(tmp_path, synthetic_results(gaps={("br_f", "C0"): -5.0}, seed=3), n_model_boot=5)

    contrasts = a.identity_contrasts[a.identity_contrasts["agent"] == "evaluator"]
    assert one(contrasts, contrast="identity_gap", identity="br_f", condition="C0")["estimate"] == pytest.approx(-5.0, abs=1.0)
    assert one(contrasts, contrast="identity_gap", identity="br_f", condition="C1")["estimate"] == pytest.approx(0.0, abs=1.0)
    change = one(contrasts, contrast="identity_x_condition", identity="br_f", condition="C1")
    assert change["estimate"] == pytest.approx(5.0, abs=1.0)
    assert change["p_holm"] < 0.05
    assert change["boot_ci95_low"] > 0
    others = contrasts[(contrasts["contrast"] == "identity_x_condition") & (contrasts["identity"] != "br_f")]
    assert (others["p_holm"] >= 0.05).all()

    # Only br_f changed, so the change sits in the three-way gender × origin × condition term.
    terms = a.model_terms[a.model_terms["agent"] == "evaluator"]
    three_way = one(terms, term="gender[T.female]:origin[T.brazilian]:condition[T.C1]")
    assert three_way["estimate"] == pytest.approx(5.0, abs=1.5)
    assert three_way["p_holm"] < 0.05
    assert three_way["n_boot"] == 5
    assert three_way["random_effects"] == "(1|CV) + (1|name)"

    gaps = a.gaps[(a.gaps["agent"] == "evaluator") & (a.gaps["identity"] == "br_f")].set_index("condition")
    assert gaps.loc["C0", "estimate"] == pytest.approx(-5.0, abs=1.0)
    assert gaps.loc["C1", "equivalent"]


def test_every_table_is_written_tidy_per_agent_and_condition(tmp_path, capsys):
    results = tmp_path / "results.csv"
    write_results(synthetic_results(conditions=("C0", "C1", "C2"), models=("model-a", "model-b"),
                                    screen_threshold=55, seed=4), results)

    assert main([str(results), "--reference", "pt_m", "--out", str(tmp_path / "out"),
                 "--n-boot", "200", "--n-model-boot", "0", "--n-perm", "199"]) == 0

    tables = {p.stem: pd.read_csv(p) for p in (tmp_path / "out").glob("*.csv")}
    assert set(tables) == {"gaps", "equivalence", "impact_ratios", "model_terms", "identity_contrasts",
                           "rank_identity", "rank_position"}
    conditions = {"C0", "C1", "C2"}
    for name in ("gaps", "impact_ratios", "identity_contrasts"):
        t = tables[name]
        assert {(ag, c) for ag, c in zip(t["agent"], t["condition"])} == {
            (ag, c) for ag in ("screener", "evaluator", "ranker") for c in conditions}, name
    assert set(tables["model_terms"]["agent"]) == {"screener", "evaluator", "ranker"}
    assert set(tables["model_terms"]["model_kind"]) == {"mixed_lm", "logit_cluster"}
    assert "model[T.model-b]" in set(tables["model_terms"]["term"])
    for name in ("rank_identity", "rank_position"):
        assert set(tables[name]["condition"]) == conditions
    assert set(tables["equivalence"]["condition"]) == conditions
    assert "paired gaps" in capsys.readouterr().out


def test_an_unknown_reference_identity_is_refused(tmp_path):
    with pytest.raises(ValueError, match="xx_m"):
        run(tmp_path, synthetic_results(conditions=("C0",), languages=("en",), base_per_cell=2), reference="xx_m")


def test_a_reference_identity_screened_out_everywhere_shows_in_the_screener_without_breaking_later_agents(tmp_path):
    a = run(tmp_path, synthetic_results(gaps={("pt_m", "C0"): -100.0, ("pt_m", "C1"): -100.0}, base_per_cell=4))

    screener = one(a.gaps, agent="screener", condition="C0", identity="br_f")
    assert screener["estimate"] == pytest.approx(100.0)  # percentage points above the reference
    assert one(a.impact_ratios, agent="screener", condition="C0", identity="pt_m")["impact_ratio"] == 0.0
    assert set(a.gaps["agent"]) == {"screener"}  # no Evaluator scores or complete pools hold pt_m
    assert "evaluator" not in set(a.model_terms["agent"])
