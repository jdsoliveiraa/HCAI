"""Runner hardening (#6): retries, missing data, cost totals, pilot slice, stability sample. Fake client only."""

import json
from dataclasses import replace

import pytest

from conftest import scores_by_name
from hcai.config import RunSettings, load_design, pilot_design
from hcai.llm import FakeLLMClient, LLMRequest

from hcai.runner import execute, run_experiment

GOOD = scores_by_name({"João Silva": 70, "Ana Souza": 65})


def is_evaluator(request: LLMRequest) -> bool:
    return "score" in (request.response_schema or {}).get("properties", {})


def evaluator_fails_first(n_failures: int, bad: str | dict = "not json"):
    """The Evaluator answers malformed `n_failures` times per CV, then correctly."""
    seen: dict[str, int] = {}

    def respond(request: LLMRequest):
        if is_evaluator(request):
            text = request.messages[-1]["content"]
            cv = "João Silva" if "João Silva" in text else "Ana Souza"
            seen[cv] = seen.get(cv, 0) + 1
            if seen[cv] <= n_failures:
                return bad
        return GOOD(request)

    return respond


def evaluator_rows(rows):
    return [r for r in rows if r.agent == "evaluator"]


def test_malformed_output_is_retried_and_the_valid_answer_kept(design, tmp_path):
    design = replace(design, run=RunSettings(retries=2))
    client = FakeLLMClient(evaluator_fails_first(2))

    rows = evaluator_rows(run_experiment(design, client, cache_dir=tmp_path / "cache"))

    assert {r.identity: r.score for r in rows} == {"pt_m": 70, "br_f": 65}
    assert all(r.status == "ok" and r.attempts == 3 for r in rows)
    assert sum(is_evaluator(c) for c in client.calls) == 6


def test_output_still_malformed_after_the_retries_is_recorded_as_missing(design, tmp_path):
    design = replace(design, run=RunSettings(retries=2))
    client = FakeLLMClient(evaluator_fails_first(99, bad={"score": 250, "rationale": "too keen"}))

    rows = evaluator_rows(run_experiment(design, client, cache_dir=tmp_path / "cache"))

    assert len(rows) == 2
    assert all(r.status == "missing" and r.score is None and r.attempts == 3 for r in rows)
    assert all("out of range" in r.error for r in rows)
    assert sum(is_evaluator(c) for c in client.calls) == 6


def test_a_failing_call_is_retried_then_recorded_as_missing(design, tmp_path):
    def respond(request):
        if is_evaluator(request):
            raise TimeoutError("provider timed out")
        return GOOD(request)

    design = replace(design, run=RunSettings(retries=1))
    client = FakeLLMClient(respond)

    rows = evaluator_rows(run_experiment(design, client, cache_dir=tmp_path / "cache"))

    assert all(r.status == "missing" and "provider timed out" in r.error for r in rows)
    assert sum(is_evaluator(c) for c in client.calls) == 4


def test_a_rerun_after_retries_is_served_from_the_cache(design, tmp_path):
    # A parseable but invalid answer is cached; the retry asks again with a note, so it has its own cache key.
    design = replace(design, run=RunSettings(retries=2))
    bad = {"score": 250, "rationale": "too keen"}
    first = run_experiment(design, FakeLLMClient(evaluator_fails_first(1, bad=bad)), cache_dir=tmp_path / "cache")
    client = FakeLLMClient(GOOD)

    second = run_experiment(design, client, cache_dir=tmp_path / "cache")

    assert client.calls == []
    assert [replace(r, cached=False) for r in second] == first


class PricedClient(FakeLLMClient):
    """Every call costs $0.25."""

    def complete(self, request):
        r = super().complete(request)
        return replace(r, usage=replace(r.usage, cost=0.25))


def test_run_writes_token_and_cost_totals_alongside_the_results(design, tmp_path):
    design = replace(design, run=RunSettings(retries=2))
    bad = {"score": 250, "rationale": "too keen"}
    client = PricedClient(evaluator_fails_first(1, bad=bad))

    first = execute(design, client, out_dir=tmp_path / "out", cache_dir=tmp_path / "cache")
    second = execute(design, PricedClient(GOOD), out_dir=tmp_path / "out", cache_dir=tmp_path / "cache")

    on_disk = json.loads((tmp_path / "out" / "run_summary.json").read_text())
    assert (tmp_path / "out" / "results.csv").exists()
    # 2 Screener + 2 x 2 Evaluator attempts + 1 Ranker
    assert first.totals["calls"] == 7 and first.totals["cache_hits"] == 0
    assert first.totals["cost_spent"] == 1.75 and first.totals["cost_total"] == 1.75
    assert first.totals["prompt_tokens"] == sum(len(m["content"].split()) for c in client.calls for m in c.messages)
    assert first.totals["by_model"]["model-a"]["cost_total"] == 1.75
    assert first.totals["by_agent"]["evaluator"]["calls"] == 4
    assert second.totals["calls"] == 0 and second.totals["cache_hits"] == 7
    assert second.totals["cost_spent"] == 0 and second.totals["cost_total"] == 1.75
    assert on_disk == second.totals


def test_missing_rows_are_counted_in_the_totals(design, tmp_path):
    design = replace(design, run=RunSettings(retries=1))

    run = execute(design, FakeLLMClient(evaluator_fails_first(99)), out_dir=tmp_path / "out", cache_dir=tmp_path / "c")

    assert run.totals["rows"] == len(run.rows) == 6
    assert run.totals["status"] == {"ok": 4, "missing": 2}


@pytest.fixture
def wide_design(design, corpus_dir):
    """2 roles x 2 base CVs x 2 identities x 2 models."""
    (corpus_dir / "data_analyst" / "high" / "cv2.en.md").write_text("$name\nNationality: $nationality\nSQL, R.")
    nurse = corpus_dir / "nurse" / "high"
    nurse.mkdir(parents=True)
    (corpus_dir / "nurse" / "job_ad.en.md").write_text("Nurse. Mandatory: licence.")
    for n in (1, 2):
        (nurse / f"cv{n}.en.md").write_text(f"$name\nNationality: $nationality\nLicensed nurse, {n} years.")
    return replace(design, roles=["data_analyst", "nurse"], base_cvs_per_role=2, models=["model-a", "model-b"])


def test_pilot_runs_a_small_slice_of_the_same_design(wide_design, tmp_path):
    pilot = pilot_design(wide_design)

    rows = run_experiment(pilot, FakeLLMClient(GOOD), cache_dir=tmp_path / "cache")

    # default slice: the first role and one base CV per quality level, every identity, language, condition and model
    assert {r.base_cv for r in rows} == {"data_analyst/high/cv1"}
    assert {r.identity for r in rows} == {"pt_m", "br_f"}
    assert {r.model for r in rows} == {"model-a", "model-b"}
    assert len(rows) == 2 * 2 * 3  # 2 identities x 2 models x 3 agents


def test_pilot_slice_is_set_in_the_config_without_touching_the_design(wide_design, tmp_path):
    config = tmp_path / "design.toml"
    config.write_text(f"""
[design]
roles = ["data_analyst", "nurse"]
quality_levels = ["high"]
base_cvs_per_role = 2
languages = ["en"]
conditions = ["C0"]
models = ["model-a", "model-b"]
corpus_dir = "{wide_design.corpus_dir}"

[[identities]]
id = "pt_m"
gender = "male"
origin = "portuguese"
names = ["João Silva"]
nationality = "Portuguese"

[pilot]
roles = 2
models = 1
""")
    design = load_design(config)

    rows = run_experiment(pilot_design(design), FakeLLMClient(GOOD), cache_dir=tmp_path / "cache")

    assert design.models == ["model-a", "model-b"] and design.base_cvs_per_role == 2
    assert {r.base_cv for r in rows} == {"data_analyst/high/cv1", "nurse/high/cv1"}
    assert {r.model for r in rows} == {"model-a"}


def test_pilot_summary_extrapolates_the_cost_of_the_full_run(wide_design, tmp_path):
    run = execute(pilot_design(wide_design), PricedClient(GOOD), out_dir=tmp_path / "out", cache_dir=tmp_path / "c")

    assert run.totals["cost_total"] == 2.5  # per model: 2 Screener + 2 Evaluator + 1 Ranker
    assert run.totals["pilot"] == {"full_design_scale": 4.0, "estimated_full_cost": 10.0}
