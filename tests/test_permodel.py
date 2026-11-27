"""Per-model analysis (`hcai.permodel`, `hcai reproduce --per-model`): planted effects in one model only."""

from __future__ import annotations

import gzip
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hcai.permodel import call_outcomes, compare_run, output_validity, per_model_tables
from hcai.reproduce import run_config
from hcai.runner import ResultRow
from test_analysis import one, synthetic_results
from test_extra import frame, repeats_of

MODELS = ("model-a", "model-b")


def two_models(seed: int = 0) -> list[ResultRow]:
    return synthetic_results(conditions=("C0",), models=MODELS, base_per_cell=12, seed=seed)


def shift_model(rows: list[ResultRow], model: str, identity: str, points: float) -> list[ResultRow]:
    return [replace(r, score=int(np.clip(r.score + points, 0, 100)))
            if r.agent == "evaluator" and r.score is not None and r.model == model and r.identity == identity else r
            for r in rows]


def test_a_gap_planted_in_one_model_is_found_in_that_model_only(tmp_path):
    rows = shift_model(two_models(), "model-a", "br_f", -6.0)
    t = per_model_tables(frame(tmp_path, rows), None, "pt_m", [], n_boot=300, n_perm=200)
    ev = t["gaps"][t["gaps"]["agent"] == "evaluator"]
    a = one(ev, model="model-a", identity="br_f")
    assert a["estimate"] == pytest.approx(-6.0, abs=1.0)
    assert a["p_holm"] < 0.05
    b = ev[ev["model"] == "model-b"]
    assert (b["p_holm"] >= 0.05).all()
    sa, sb = one(t["summary"], model="model-a"), one(t["summary"], model="model-b")
    assert sa["largest_gap_identity"] == "br_f"
    assert not sa["equivalent"] and sb["equivalent"]
    assert sa["gaps_surviving_holm"] >= 1
    # reference-free: br_f stands out against the mean in model-a too
    free = t["gaps_reference_free"]
    assert one(free[free["agent"] == "evaluator"], model="model-a", identity="br_f")["estimate"] < -4


def test_every_table_has_one_row_set_per_model_and_the_repeat_noise_is_per_model(tmp_path):
    rows = two_models(seed=1)
    reps = frame(tmp_path, repeats_of([r for r in rows if r.model == "model-a"], 1.0)
                 + repeats_of([r for r in rows if r.model == "model-b"], 4.0, seed=1), "repeats.csv")
    t = per_model_tables(frame(tmp_path, rows), reps, "pt_m", [], n_boot=300, n_perm=200)
    for name in ("gaps", "equivalence", "rank_position", "gaps_reference_free", "rank_adjusted", "screener_pass",
                 "repeat_noise", "noise_vs_gap"):
        assert set(t[name]["model"]) == set(MODELS), name
    s = t["summary"].set_index("model")
    assert s.loc["model-a", "repeat_sigma_score"] == pytest.approx(1.0, abs=0.4)
    assert s.loc["model-b", "repeat_sigma_score"] == pytest.approx(4.0, abs=1.0)


def test_output_validity_counts_missing_retried_and_one_ranker_decision_per_pool(tmp_path):
    rows = two_models()
    first = next(i for i, r in enumerate(rows) if r.model == "model-b" and r.agent == "evaluator")
    rows[first] = replace(rows[first], status="missing", score=None, attempts=3)
    rows = [replace(r, attempts=2) if r.model == "model-a" and r.agent == "ranker" else r for r in rows]
    v = output_validity(frame(tmp_path, rows))
    evaluator = one(v, model="model-b", agent="evaluator")
    assert evaluator["missing"] == 1
    assert evaluator["retried"] == 1
    ranker = one(v, model="model-a", agent="ranker")
    assert ranker["decisions"] == ranker["retried"] < ranker["rows"]  # one per pool, not one per ranked CV


def test_call_outcomes_counts_distinct_requests_and_leaves_out_rate_limit_waits(tmp_path):
    records = [
        {"model": "m", "agent": "evaluator", "outcome": "ok", "key": "k1"},
        {"model": "m", "agent": "evaluator", "outcome": "ok", "key": "k1"},  # cache replay of the same request
        {"model": "m", "agent": "evaluator", "outcome": "parse_error", "key": "k2"},
        {"model": "m", "agent": "evaluator", "outcome": "ok", "key": "k3"},
        {"model": "m", "agent": "ranker", "outcome": "rate_limited", "key": "k4"},
        {"model": "m", "agent": "ranker", "outcome": "timeout", "key": "k4"},
    ]
    path = tmp_path / "calls.jsonl.gz"
    with gzip.open(path, "wt") as f:
        f.write("\n".join(json.dumps(r) for r in records) + "\n")
    c = call_outcomes(path).set_index(["agent", "outcome"])["requests"]
    assert c.to_dict() == {("evaluator", "ok"): 2, ("evaluator", "parse_error"): 1, ("ranker", "timeout"): 1}
    v = output_validity(frame(tmp_path, [r for r in two_models() if r.model == "model-a"]).assign(model="m"),
                        call_outcomes(path)).set_index("agent")
    assert v.loc["evaluator", "requests_parse_error"] == 1
    assert v.loc["ranker", "requests_failed"] == 1


def test_run_config_prefers_core_toml_then_the_only_toml(tmp_path: Path):
    assert run_config(tmp_path) is None
    (tmp_path / "weak.toml").write_text("")
    assert run_config(tmp_path) == tmp_path / "weak.toml"
    (tmp_path / "other.toml").write_text("")
    assert run_config(tmp_path) is None  # ambiguous
    (tmp_path / "core.toml").write_text("")
    assert run_config(tmp_path) == tmp_path / "core.toml"


def test_per_model_is_off_by_default_so_core_outputs_do_not_change():
    import inspect

    from hcai.reproduce import reproduce

    assert inspect.signature(reproduce).parameters["per_model"].default is False


def test_compare_run_summarises_another_frozen_run_on_the_given_conditions_only(tmp_path):
    rows = synthetic_results(conditions=("C0", "C1"), models=MODELS, base_per_cell=12)
    rows = [shift_model([r], "model-b", "ee_m", -6.0)[0] if r.condition == "C1" else r for r in rows]
    other = tmp_path / "core-v1"
    other.mkdir()
    from hcai.runner import write_results
    write_results(rows, other / "results.csv")
    s = compare_run(other, {"C0"}, "pt_m", [], n_boot=300, n_perm=200).set_index("model")
    assert set(s.index) == set(MODELS)
    assert s["equivalent"].all()  # the C1 gap planted in model-b is left out
    s1 = compare_run(other, {"C1"}, "pt_m", [], n_boot=300, n_perm=200).set_index("model")
    assert s1.loc["model-b", "largest_gap_identity"] == "ee_m"


def test_compare_without_per_model_is_refused(tmp_path, capsys):
    from hcai.reproduce import main

    (tmp_path / "results.csv").write_text("")
    assert main([str(tmp_path), "--compare", str(tmp_path)]) == 1
    assert "--compare needs --per-model" in capsys.readouterr().err


def test_a_ranker_that_keeps_the_input_order_gets_no_identity_test_after_position(tmp_path):
    rows = two_models()
    rows = [replace(r, rank=r.input_position, top3=r.input_position <= 3)
            if r.agent == "ranker" and r.model == "model-a" and r.status == "ok" and r.input_position is not None else r
            for r in rows]
    t = per_model_tables(frame(tmp_path, rows), None, "pt_m", [], n_boot=200, n_perm=100)
    adj = t["rank_adjusted"]
    a, b = adj[adj["model"] == "model-a"], adj[adj["model"] == "model-b"]
    assert a["input_order_only"].all() and a["p_holm"].isna().all()
    assert not b["input_order_only"].any() and b["p_holm"].notna().all()
    joint = t["rank_adjusted_joint"]
    assert joint.loc[joint["model"] == "model-a", "p"].isna().all()
    assert one(t["summary"], model="model-a")["ranker_input_order_share"] == 1.0
    assert one(t["summary"], model="model-b")["ranker_input_order_share"] < 1.0


def test_repeat_noise_has_one_row_per_model(tmp_path):
    rows = two_models()
    reps = frame(tmp_path, repeats_of(rows, 2.0), "repeats.csv")
    t = per_model_tables(frame(tmp_path, rows), reps, "pt_m", [], n_boot=200, n_perm=100)
    assert sorted(t["repeat_noise"]["model"]) == sorted(MODELS)


def favour(rows: list[ResultRow], model: str, identity: str) -> list[ResultRow]:
    """In every equal-quality pool of `model`, move `identity` to rank 1 and shift the ones it passed down by one,
    whatever its input position (a Ranker that prefers that identity)."""
    out = list(rows)
    pools: dict[tuple, list[int]] = {}
    for i, r in enumerate(out):
        if r.agent == "ranker" and r.model == model and r.pool_kind == "equal" and r.rank is not None:
            pools.setdefault((r.pool, r.language, r.condition), []).append(i)
    for idx in pools.values():
        target = next((i for i in idx if out[i].identity == identity), None)
        if target is None:
            continue
        old = out[target].rank
        assert old is not None
        for i in idx:
            rank = out[i].rank
            assert rank is not None
            new = 1 if i == target else rank + 1 if rank < old else rank
            out[i] = replace(out[i], rank=new, top3=new <= 3)
    return out


def test_a_ranker_that_favours_one_identity_is_found_after_position_in_that_model_only(tmp_path):
    rows = favour(two_models(), "model-a", "br_f")
    t = per_model_tables(frame(tmp_path, rows), None, "pt_m", [], n_boot=200, n_perm=100)
    adj = t["rank_adjusted"]
    a = one(adj, model="model-a", outcome="rank", identity="br_f")
    assert a["estimate"] < -2 and a["p_holm"] < 0.05
    b = adj[(adj["model"] == "model-b") & (adj["outcome"] == "rank")]
    assert (b["p_holm"] >= 0.05).all()
    top3 = t["ranker_top3"]
    assert one(top3, model="model-a", language="all", identity="br_f")["top3_rate"] == 1.0
    assert one(top3, model="model-b", language="all", identity="br_f")["top3_rate"] < 0.9


def test_the_rationale_patterns_count_one_rationale_per_pool(tmp_path):
    rows = [replace(r, rationale="Tie-break: Portuguese nationality first.")
            if r.agent == "ranker" and r.model == "model-a" and r.input_position == 1 else r for r in two_models()]
    t = per_model_tables(frame(tmp_path, rows), None, "pt_m", [], n_boot=100, n_perm=50)
    r = t["ranker_rationales"]
    a = one(r, model="model-a", pool_kind="equal")
    assert a["nationality"] == a["tie_break"] == a["rationales"] > 0
    assert (r.loc[r["model"] == "model-b", "nationality"] == 0).all()
    import re

    from hcai.permodel import NATIONALITY_PATTERN
    assert not re.search(NATIONALITY_PATTERN, "an international hospital, experiência internacional")
    assert all(re.search(NATIONALITY_PATTERN, t) for t in ("portuguese nationals first", "nacionalidade portuguesa"))


def test_a_failed_repeat_is_counted_apart_from_the_main_request(tmp_path):
    records = [
        {"model": "m", "agent": "evaluator", "outcome": "ok", "key": "k1"},
        {"model": "m", "agent": "evaluator", "outcome": "parse_error", "key": "k1", "repeat": 1},
        {"model": "m", "agent": "evaluator", "outcome": "ok", "key": "k1", "repeat": 1},
    ]
    path = tmp_path / "calls.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    v = one(output_validity(frame(tmp_path, [r for r in two_models() if r.model == "model-a"]).assign(model="m"),
                            call_outcomes(path)), agent="evaluator")
    assert (v["requests_ok"], v["requests_parse_error"]) == (1, 0)
    assert (v["repeat_requests_ok"], v["repeat_requests_parse_error"]) == (1, 1)


def test_compare_run_reads_only_the_calls_of_the_compared_conditions(tmp_path):
    from hcai.runner import write_results

    other = tmp_path / "core-v1"
    other.mkdir()
    write_results(synthetic_results(conditions=("C0", "C1"), models=MODELS, base_per_cell=4), other / "results.csv")
    records = [{"model": "model-a", "agent": "evaluator", "condition": "C1", "outcome": "timeout", "key": "k1"},
               {"model": "model-a", "agent": "evaluator", "condition": "C0", "outcome": "parse_error", "key": "k2"}]
    with gzip.open(other / "calls.jsonl.gz", "wt") as f:
        f.write("\n".join(json.dumps(r) for r in records) + "\n")
    s = one(compare_run(other, {"C0"}, "pt_m", [], n_boot=100, n_perm=50), model="model-a")
    assert (s["parse_error_requests"], s["failed_requests"]) == (1, 0)
