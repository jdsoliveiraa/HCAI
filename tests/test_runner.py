import csv
from dataclasses import replace

from conftest import scores_by_name
from hcai.llm import FakeLLMClient
from hcai.runner import read_results, run_experiment, write_results


def test_one_evaluator_row_per_cv_identity_with_score_and_tokens(design, fake_client, tmp_path):
    rows = [r for r in run_experiment(design, fake_client, cache_dir=tmp_path / "cache") if r.agent == "evaluator"]

    assert len(rows) == 2
    by_identity = {r.identity: r for r in rows}
    assert by_identity["pt_m"].score == 70
    assert by_identity["br_f"].score == 65
    assert by_identity["br_f"].name == "Ana Souza"
    for r in rows:
        assert (r.base_cv, r.language, r.condition, r.model, r.agent) == ("data_analyst/high/cv1", "en", "C0", "model-a", "evaluator")
        assert r.status == "ok"
        assert r.rationale
        assert r.prompt_tokens > 0 and r.completion_tokens > 0


def test_second_run_against_same_cache_makes_no_calls(design, tmp_path):

    responder = scores_by_name({"João Silva": 70, "Ana Souza": 65})
    first_client, second_client = FakeLLMClient(responder), FakeLLMClient(responder)

    first = run_experiment(design, first_client, cache_dir=tmp_path / "cache")
    second = run_experiment(design, second_client, cache_dir=tmp_path / "cache")

    assert len(first_client.calls) == 5  # 2 Screener + 2 Evaluator + 1 Ranker
    assert second_client.calls == []
    assert [replace(r, cached=False) for r in second] == first
    assert all(r.cached for r in second) and not any(r.cached for r in first)


def test_changing_any_request_field_misses_the_cache(design, tmp_path):
    from dataclasses import replace


    responder = scores_by_name({"João Silva": 70, "Ana Souza": 65})
    run_experiment(design, FakeLLMClient(responder), cache_dir=tmp_path / "cache")
    client = FakeLLMClient(responder)
    run_experiment(replace(design, seed=7), client, cache_dir=tmp_path / "cache")

    assert len(client.calls) == 5


def test_malformed_evaluator_output_is_recorded_as_missing(design, tmp_path):
    crew = scores_by_name({"João Silva": 70, "Ana Souza": 65})

    def respond(request):
        if "score" in (request.response_schema or {})["properties"]:
            return "sorry, I cannot score this"
        return crew(request)

    rows = [r for r in run_experiment(design, FakeLLMClient(respond), cache_dir=tmp_path / "cache") if r.agent == "evaluator"]

    assert len(rows) == 2
    assert all(r.status == "missing" and r.score is None and r.error for r in rows)


def test_results_table_round_trips_through_csv(design, fake_client, tmp_path):
    rows = run_experiment(design, fake_client, cache_dir=tmp_path / "cache")
    path = tmp_path / "results.csv"

    write_results(rows, path)

    with path.open() as f:
        header = next(csv.reader(f))
    for column in ["base_cv", "identity", "name", "language", "condition", "model", "agent",
                   "score", "rationale", "prompt_tokens", "completion_tokens", "cost", "status"]:
        assert column in header
    assert read_results(path) == rows


def test_a_rerun_replays_malformed_answers_and_more_retries_ask_again(design, tmp_path):
    from hcai.config import RunSettings

    design = replace(design, run=RunSettings(retries=0))
    run_experiment(design, FakeLLMClient(lambda request: "not json"), cache_dir=tmp_path / "cache")
    replay = FakeLLMClient(scores_by_name({"João Silva": 70, "Ana Souza": 65}))
    replayed = run_experiment(design, replay, cache_dir=tmp_path / "cache")
    again = FakeLLMClient(scores_by_name({"João Silva": 70, "Ana Souza": 65}))

    rows = run_experiment(replace(design, run=RunSettings(retries=1)), again, cache_dir=tmp_path / "cache")

    assert replay.calls == [] and all(r.status == "missing" for r in replayed)
    assert len(again.calls) == 5  # 2 Screener retries, then the Evaluator and Ranker that had not run
    assert all(r.status == "ok" for r in rows)
