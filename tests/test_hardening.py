"""Runner hardening (#6): retries, missing data, cost totals, pilot slice, stability sample. Fake client only."""

import json
from dataclasses import replace

import pytest

from conftest import scores_by_name
from hcai.config import PilotSlice, Pools, RunSettings, load_design, pilot_design
from hcai.llm import FakeLLMClient, LLMRequest

from hcai.runner import execute, read_results, run_experiment, write_results

GOOD = scores_by_name({"João Silva": 70, "Ana Souza": 65})


def is_evaluator(request: LLMRequest) -> bool:
    return "score" in (request.response_schema or {}).get("properties", {})


def evaluator_fails_first(n_failures: int, bad: str | dict = "not json"):
    """The Evaluator answers malformed `n_failures` times per CV, then correctly."""
    seen: dict[str, int] = {}

    def respond(request: LLMRequest):
        if is_evaluator(request):
            text = " ".join(m["content"] for m in request.messages)
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

    design = replace(design, run=RunSettings(retries=1, retry_backoff=0.0))
    client = FakeLLMClient(respond)

    rows = evaluator_rows(run_experiment(design, client, cache_dir=tmp_path / "cache"))

    assert all(r.status == "missing" and "provider timed out" in r.error for r in rows)
    assert sum(is_evaluator(c) for c in client.calls) == 4


def test_a_rerun_after_retries_is_served_from_the_cache(design, tmp_path):
    # Malformed answers are cached too; the retry asks again with a note, so it has its own cache key.
    design = replace(design, run=RunSettings(retries=2))
    first = run_experiment(design, FakeLLMClient(evaluator_fails_first(1)), cache_dir=tmp_path / "cache")
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
    assert first.totals["by_agent"]["evaluator"]["attempts"] == 4
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
    (corpus_dir / "data_analyst" / "high" / "cv2.en.md").write_text("$name\nNationality: $nationality\n$work_eligibility\nSQL, R.")
    nurse = corpus_dir / "nurse" / "high"
    nurse.mkdir(parents=True)
    (corpus_dir / "nurse" / "job_ad.en.md").write_text("Nurse. Mandatory: licence.")
    for n in (1, 2):
        (nurse / f"cv{n}.en.md").write_text(f"$name\nNationality: $nationality\n$work_eligibility\nLicensed nurse, {n} years.")
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
    assert run.totals["pilot"] == {"cv_scale": 4.0, "ranker_scale": 4.0, "estimated_full_cost": 10.0}


def test_pilot_estimate_scales_the_ranker_by_the_cvs_it_ranks(wide_design, tmp_path):
    design = replace(wide_design, pools=Pools(equal_quality=True, mixed_quality=2, mixed_size=8))

    run = execute(pilot_design(design), PricedClient(GOOD), out_dir=tmp_path / "out", cache_dir=tmp_path / "c")

    # Ranked CV slots per role x model: full 4 (equal) + 2 mixed pools of 2 CVs = 8 over 2 roles; pilot 2 + 1 x 1 = 3.
    assert run.totals["pilot"]["ranker_scale"] == round(32 / 6, 4)
    # Screener + Evaluator cost 2.0 x 4, Ranker cost 1.0 (2 pools x 2 models) x 32/6
    assert run.totals["pilot"]["estimated_full_cost"] == round(8 + 32 / 6, 4)


def drifting_scores():
    """João Silva scores 70, 72, 74, ... on successive calls; Ana Souza always 65."""
    calls = {"João Silva": 0}

    def respond(request: LLMRequest):
        if is_evaluator(request) and any("João Silva" in m["content"] for m in request.messages):
            calls["João Silva"] += 1
            return {"score": 68 + 2 * calls["João Silva"], "rationale": "drifts"}
        return GOOD(request)

    return respond


def test_repeat_sample_produces_a_stability_report(design, tmp_path):
    design = replace(design, run=RunSettings(repeat_fraction=1.0, repeats=2))

    run = execute(design, FakeLLMClient(drifting_scores()), out_dir=tmp_path / "out", cache_dir=tmp_path / "cache")

    assert [r.score for r in run.rows if r.agent == "evaluator"] == [70, 65]  # the results keep the first answer
    assert {r.repeat for r in run.repeats} == {1, 2}
    stability = {(s["model"], s["condition"], s["agent"]): s for s in run.stability}
    evaluator = stability[("model-a", "C0", "evaluator")]
    assert evaluator["items"] == 2 and evaluator["pairs"] == 4
    assert evaluator["agreement"] == 0.5  # João's two repeats differ from his original score, Ana's match
    assert evaluator["mean_abs_diff"] == 1.5  # (2 + 4 + 0 + 0) / 4
    assert evaluator["mean_sd"] == 1.0  # SD of [70, 72, 74] is 2, of [65, 65, 65] is 0
    assert stability[("model-a", "C0", "screener")]["agreement"] == 1.0
    assert stability[("model-a", "C0", "ranker")]["agreement"] == 1.0
    assert json.loads((tmp_path / "out" / "stability.json").read_text()) == run.stability
    assert (tmp_path / "out" / "repeats.csv").exists()


def test_repeats_are_paid_for_once_and_counted_in_the_totals(design, tmp_path):
    design = replace(design, run=RunSettings(repeat_fraction=1.0, repeats=1))
    first = execute(design, PricedClient(drifting_scores()), out_dir=tmp_path / "out", cache_dir=tmp_path / "cache")
    client = PricedClient(drifting_scores())

    second = execute(design, client, out_dir=tmp_path / "out", cache_dir=tmp_path / "cache")

    assert first.totals["repeat_sample"]["attempts"] == 5 and first.totals["calls"] == 10
    assert first.totals["cost_total"] == 2.5  # 5 calls for the results, 5 for one repeat of each
    assert client.calls == []
    assert second.stability == first.stability


def test_no_repeat_sample_by_default(design, tmp_path):
    client = FakeLLMClient(GOOD)

    run = execute(design, client, out_dir=tmp_path / "out", cache_dir=tmp_path / "cache")

    assert run.repeats == [] and run.stability == []
    assert len(client.calls) == 5


def test_spend_counts_only_the_calls_this_run_made(design, tmp_path):
    out, cache = tmp_path / "out", tmp_path / "cache"
    execute(replace(design, run=RunSettings(retries=0)), PricedClient(evaluator_fails_first(1)), out, cache)

    # The Evaluator's first, malformed answer replays from the cache; only its retry is a new call.
    run = execute(replace(design, run=RunSettings(retries=1)), PricedClient(GOOD), out, cache)

    assert run.totals["calls"] == 2 and run.totals["cache_hits"] == 5
    assert run.totals["cost_spent"] == 0.5 and run.totals["cost_total"] == 1.75
    assert run.totals["by_model"]["model-a"]["cost_spent"] == 0.5


def ranker_answers(*answers):
    """The Ranker gives these answers in turn (the last one from then on); the other agents answer well."""
    given = []

    def respond(request: LLMRequest):
        if "ranking" in (request.response_schema or {}).get("properties", {}):
            given.append(1)
            return answers[min(len(given), len(answers)) - 1]
        return GOOD(request)

    return respond


@pytest.mark.parametrize("bad", [[1, "2"], [True, 2], [1, 1], "not json"])
def test_an_invalid_ranking_is_retried(design, tmp_path, bad):
    good = {"ranking": [2, 1], "rationale": "second first"}

    rows = run_experiment(design, FakeLLMClient(ranker_answers({"ranking": bad, "rationale": "x"}, good)),
                          cache_dir=tmp_path / "cache")

    ranker = [r for r in rows if r.agent == "ranker"]
    assert sorted(r.rank for r in ranker if r.rank is not None) == [1, 2]
    assert all(r.status == "ok" for r in ranker)
    assert [r.attempts for r in sorted(ranker, key=lambda r: r.input_position or 0)] == [2, 0]


def test_a_ranking_still_invalid_after_the_retries_marks_the_pool_missing(design, tmp_path):
    design = replace(design, run=RunSettings(retries=1))

    rows = run_experiment(design, FakeLLMClient(ranker_answers({"ranking": [1], "rationale": "x"})),
                          cache_dir=tmp_path / "cache")

    ranker = [r for r in rows if r.agent == "ranker"]
    assert len(ranker) == 2 and all(r.status == "missing" and r.rank is None and "permutation" in r.error for r in ranker)


def test_a_boolean_score_is_not_a_score(design, tmp_path):
    design = replace(design, run=RunSettings(retries=0))

    def respond(request):
        return {"score": True, "rationale": "yes"} if is_evaluator(request) else GOOD(request)

    rows = evaluator_rows(run_experiment(design, FakeLLMClient(respond), cache_dir=tmp_path / "cache"))

    assert all(r.status == "missing" and r.score is None for r in rows)


def test_a_call_that_fails_once_then_succeeds_is_kept(design, tmp_path):
    failed = set()

    def respond(request):
        if is_evaluator(request) and not failed:
            failed.add(1)
            raise ConnectionError("blip")
        return GOOD(request)

    design = replace(design, run=RunSettings(retries=1, retry_backoff=0.0))

    rows = evaluator_rows(run_experiment(design, FakeLLMClient(respond), cache_dir=tmp_path / "cache"))

    assert {r.identity: r.score for r in rows} == {"pt_m": 70, "br_f": 65}
    assert all(r.status == "ok" for r in rows)


def test_retries_wait_by_default_before_repeating_a_failed_call():
    assert RunSettings().retry_backoff > 0


def test_results_written_before_the_attempts_column_still_load(design, fake_client, tmp_path):
    import csv

    rows = run_experiment(design, fake_client, cache_dir=tmp_path / "cache")
    path = tmp_path / "results.csv"
    write_results(rows, path)
    with path.open() as f:
        old = [{k: v for k, v in raw.items() if k not in ("attempts", "repeat")} for raw in csv.DictReader(f)]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(old[0]))
        writer.writeheader()
        writer.writerows(old)

    assert [r.status for r in read_results(path)] == [r.status for r in rows]


def test_pilot_estimate_adds_the_expected_share_of_the_stability_sample(wide_design, tmp_path):
    design = replace(wide_design, run=RunSettings(repeat_fraction=0.5, repeats=2))

    run = execute(pilot_design(design), PricedClient(GOOD), out_dir=tmp_path / "out", cache_dir=tmp_path / "c")

    # results cost 2.5, x 4 for the full design, x (1 + 0.5 x 2) for the repeats, whatever the pilot drew
    assert run.totals["pilot"]["estimated_full_cost"] == 20.0


def test_pilot_estimate_counts_the_variants_the_full_run_will_render_whatever_the_names(wide_design, tmp_path):
    from hcai.variants import build_variants

    many_names = [replace(i, names=[f"{i.names[0]} {k}" for k in range(4)]) for i in wide_design.identities]
    design = replace(wide_design, identities=many_names)
    pilot = pilot_design(design)

    run = execute(pilot, PricedClient(GOOD), out_dir=tmp_path / "out", cache_dir=tmp_path / "c")

    # one name per base CV x identity: more names spread over the CVs, they don't multiply the calls
    assert run.totals["pilot"]["cv_scale"] == len(build_variants(design)) / len(build_variants(pilot)) == 4.0


def test_a_pilot_sized_results_table_can_be_analysed(design, tmp_path):
    from hcai.analysis import analyse, results_frame

    def respond(request):  # the Screener rejects one CV, so screen-pass varies within the single base CV
        if "passed" in (request.response_schema or {}).get("properties", {}):
            return {"passed": "João Silva" in " ".join(m["content"] for m in request.messages), "reason": "r"}
        return GOOD(request)

    rows = run_experiment(design, FakeLLMClient(respond), cache_dir=tmp_path / "cache")

    analysis = analyse(results_frame(rows), "pt_m", n_boot=50, n_model_boot=0, n_perm=99)

    assert analysis is not None


def test_the_variants_config_run_can_be_analysed(tmp_path):
    from pathlib import Path

    from hcai.analysis import analyse, load_results
    from hcai.cli import _fake_response, run_command

    config = Path(__file__).parent.parent / "configs" / "variants.toml"
    run_command(config, out_dir=tmp_path, cache_dir=tmp_path / "cache", client=FakeLLMClient(_fake_response))

    analysis = analyse(load_results(tmp_path / "results.csv"), "pt_m", n_boot=50, n_model_boot=0, n_perm=99)

    assert analysis is not None


@pytest.mark.parametrize("pilot_slice", [None, PilotSlice(identities=3, languages=1)])
def test_pilot_scales_match_the_variants_and_pools_the_runner_builds(pilot_slice):
    from pathlib import Path

    from hcai.runner import build_pools
    from hcai.variants import build_variants

    full = load_design(Path(__file__).parent.parent / "configs" / "variants.toml")  # EN + PT, name-bank identities
    full = replace(full, conditions=["C0", "C1"], models=["model-a", "model-b"],
                   pools=Pools(equal_quality=True, mixed_quality=2, mixed_size=8))
    if pilot_slice is not None:
        full = pilot_design(replace(full, pilot=pilot_slice))

    variants = build_variants(full)
    ranked = sum(
        len(pool.members)
        for role in full.roles for language in full.languages
        for pool in build_pools(full, role, language, [v for v in variants if v.role == role and v.language == language])
    )
    runs = len(full.conditions) * len(full.models)

    assert full.size == len(variants) * runs
    assert full.ranked_slots == ranked * runs


@pytest.mark.parametrize("n", [2, 4])
def test_a_pilot_slice_of_identities_spreads_over_origins_and_genders(n):
    from pathlib import Path

    full = load_design(Path(__file__).parent.parent / "configs" / "variants.toml")  # name bank: 4 origins x 2 genders

    kept = pilot_design(replace(full, pilot=PilotSlice(identities=n))).identities

    assert len(kept) == n
    assert len({i.origin for i in kept}) == n and len({i.gender for i in kept}) == 2


def test_a_pilot_keeping_every_identity_keeps_their_order(wide_design):
    assert pilot_design(wide_design).identities == wide_design.identities
