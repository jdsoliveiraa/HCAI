"""Condition C2 (#8): rubric Evaluator with quoted evidence, LLM judge and escalation. Runner seam, fake client only."""

import csv
import json
from dataclasses import replace

import pytest

from hcai.config import Identity, Oversight, load_design
from hcai.llm import FakeLLMClient, LLMRequest
from hcai.runner import execute, read_results, run_experiment

JOB_AD = """# Data Analyst

## Mandatory requirements
- SQL
- Python

## Desirable requirements
- Dashboards
- Logistics experience
"""

REAL_QUOTE = "SQL, Python, 3 years of experience."
FABRICATED = "Led the Power BI dashboard team at DHL"


def rubric(scores: list[int], quotes: list[str] | None = None) -> dict:
    quotes = quotes or [REAL_QUOTE if s else "" for s in scores]
    return {"criteria": [{"criterion": i, "score": s, "quote": q}
                         for i, (s, q) in enumerate(zip(scores, quotes), start=1)],
            "rationale": "per-criterion evidence"}


def fake_crew(rubrics: dict[str, dict], judge_fails: bool = False):
    """Screener passes all, Ranker keeps the order; the rubric depends on the name in the CV.

    The fake judge marks a criterion unfaithful when its quote is the fabricated one.
    """

    def respond(request: LLMRequest):
        props = (request.response_schema or {}).get("properties", {})
        text = " ".join(m["content"] for m in request.messages)
        if "passed" in props:
            return {"passed": True, "reason": "meets the mandatory requirements"}
        if "ranking" in props:
            return {"ranking": props["ranking"]["items"]["enum"], "rationale": "kept the order shown"}
        if "verdicts" in props:
            if judge_fails:
                return "no idea"
            n = len(props["verdicts"]["items"]["properties"]["criterion"]["enum"])
            fabricated = FABRICATED in text
            return {"verdicts": [
                {"criterion": i, "verdict": "unfaithful" if fabricated and i == 3 else "faithful",
                 "reason": "quote not in the CV" if fabricated and i == 3 else "quote found and supports the score"}
                for i in range(1, n + 1)]}
        if "criteria" in props:
            for name, answer in rubrics.items():
                if name in text:
                    return answer
        raise AssertionError(f"unexpected request: {list(props)}")

    return respond


@pytest.fixture
def c2_design(design, corpus_dir):
    (corpus_dir / "data_analyst" / "job_ad.en.md").write_text(JOB_AD)
    return replace(design, conditions=["C2"],
                   oversight=Oversight(judge_model="judge-model", cutoff=70, margin=5))


def by_agent(rows, agent):
    return {r.identity: r for r in rows if r.agent == agent}


def test_rubric_evaluator_scores_each_criterion_with_a_quote_and_a_total(c2_design, tmp_path):
    client = FakeLLMClient(fake_crew({"João Silva": rubric([10, 10, 8, 4]), "Ana Souza": rubric([10, 8, 0, 0])}))

    evaluator = by_agent(run_experiment(c2_design, client, cache_dir=tmp_path / "cache"), "evaluator")

    joao = evaluator["pt_m"]
    assert joao.status == "ok" and joao.condition == "C2"
    assert joao.score == 80  # 32 of 40 rubric points
    evidence = json.loads(joao.evidence)
    assert [c["criterion"] for c in evidence] == ["SQL", "Python", "Dashboards", "Logistics experience"]
    assert [c["score"] for c in evidence] == [10, 10, 8, 4]
    assert all(c["quote"] == REAL_QUOTE for c in evidence)
    assert evaluator["br_f"].score == 45  # 18 of 40


def test_a_fabricated_quote_is_flagged_by_the_judge_and_escalated(c2_design, tmp_path):
    client = FakeLLMClient(fake_crew({
        "João Silva": rubric([10, 10, 10, 6]),
        "Ana Souza": rubric([10, 10, 10, 6], [REAL_QUOTE, REAL_QUOTE, FABRICATED, REAL_QUOTE]),
    }))

    rows = run_experiment(c2_design, client, cache_dir=tmp_path / "cache")

    judge, evaluator = by_agent(rows, "judge"), by_agent(rows, "evaluator")
    verdicts = json.loads(judge["br_f"].judge_verdicts)
    assert [v["verdict"] for v in verdicts] == ["faithful", "faithful", "unfaithful", "faithful"]
    assert verdicts[2]["criterion"] == "Dashboards" and verdicts[2]["reason"]
    assert judge["br_f"].judge_flag is True and judge["br_f"].judge_model == "judge-model"
    assert evaluator["br_f"].judge_flag is True
    assert evaluator["br_f"].escalated is True and "flagged quote" in evaluator["br_f"].escalation_reason


def test_a_clean_score_far_from_the_cutoff_is_not_escalated(c2_design, tmp_path):
    client = FakeLLMClient(fake_crew({"João Silva": rubric([10, 10, 10, 6]), "Ana Souza": rubric([4, 4, 2, 0])}))

    rows = run_experiment(c2_design, client, cache_dir=tmp_path / "cache")

    evaluator = by_agent(rows, "evaluator")
    assert {i: r.score for i, r in evaluator.items()} == {"pt_m": 90, "br_f": 25}
    for r in evaluator.values():
        assert r.judge_flag is False and r.escalated is False and r.escalation_reason == ""
    assert all(r.judge_flag is False and r.status == "ok" for r in by_agent(rows, "judge").values())


def test_a_score_within_the_margin_of_the_cutoff_is_escalated(c2_design, tmp_path):
    client = FakeLLMClient(fake_crew({"João Silva": rubric([8, 8, 7, 4]), "Ana Souza": rubric([10, 10, 10, 6])}))

    evaluator = by_agent(run_experiment(c2_design, client, cache_dir=tmp_path / "cache"), "evaluator")

    assert evaluator["pt_m"].score == 68  # 27 of 40 = 67.5, within 5 of the cutoff 70
    assert evaluator["pt_m"].escalated is True and evaluator["pt_m"].escalation_reason == "near cutoff"
    assert evaluator["pt_m"].judge_flag is False
    assert evaluator["br_f"].escalated is False


def test_the_margin_is_configurable(c2_design, tmp_path):
    client = FakeLLMClient(fake_crew({"João Silva": rubric([8, 8, 7, 4]), "Ana Souza": rubric([10, 10, 10, 6])}))
    design = replace(c2_design, oversight=replace(c2_design.oversight, margin=1))

    evaluator = by_agent(run_experiment(design, client, cache_dir=tmp_path / "cache"), "evaluator")

    assert evaluator["pt_m"].escalated is False


def test_the_default_cutoff_is_the_median_score_of_the_condition_and_model(c2_design, tmp_path):
    identities = c2_design.identities + [
        Identity(id="cv_m", gender="male", origin="palop", names=["Carlos Tavares"], nationality="Cape Verdean")]
    design = replace(c2_design, identities=identities, oversight=Oversight(judge_model="judge-model", margin=5))
    client = FakeLLMClient(fake_crew({
        "João Silva": rubric([4, 4, 4, 4]),  # 40
        "Ana Souza": rubric([6, 6, 6, 6]),  # 60, the median
        "Carlos Tavares": rubric([9, 9, 9, 9]),  # 90
    }))

    evaluator = by_agent(run_experiment(design, client, cache_dir=tmp_path / "cache"), "evaluator")

    assert {i: r.escalated for i, r in evaluator.items()} == {"pt_m": False, "br_f": True, "cv_m": False}


def test_a_judge_that_gives_no_valid_verdict_escalates_the_decision(c2_design, tmp_path):
    client = FakeLLMClient(fake_crew({"João Silva": rubric([10, 10, 10, 6]), "Ana Souza": rubric([4, 4, 2, 0])},
                                     judge_fails=True))

    rows = run_experiment(c2_design, client, cache_dir=tmp_path / "cache")

    assert all(r.status == "missing" and r.judge_flag is None and r.error for r in by_agent(rows, "judge").values())
    assert all(r.escalated and r.escalation_reason == "judge missing" for r in by_agent(rows, "evaluator").values())


def test_c0_rows_are_not_judged_or_escalated(c2_design, tmp_path):
    from conftest import scores_by_name

    client = FakeLLMClient(scores_by_name({"João Silva": 70, "Ana Souza": 65}))
    rows = run_experiment(replace(c2_design, conditions=["C0"]), client, cache_dir=tmp_path / "cache")

    assert {r.agent for r in rows} == {"screener", "evaluator", "ranker"}
    assert all(r.judge_flag is None and r.escalated is None and r.evidence == "" for r in rows)
    assert all(c.model == "model-a" for c in client.calls)


def test_execute_counts_escalations_writes_the_review_queue_and_bills_the_judge(c2_design, tmp_path):
    client = FakeLLMClient(fake_crew({
        "João Silva": rubric([10, 10, 10, 6]),
        "Ana Souza": rubric([10, 10, 10, 6], [REAL_QUOTE, REAL_QUOTE, FABRICATED, REAL_QUOTE]),
    }))

    run = execute(c2_design, client, out_dir=tmp_path / "out", cache_dir=tmp_path / "cache")

    oversight = run.totals["oversight"]
    assert oversight["C2"]["model-a"] == dict(decisions=2, judged=2, flagged=1, near_cutoff=0, judge_missing=0,
                                              evaluator_missing=0, escalated=1)
    assert run.totals["by_model"]["judge-model"]["calls"] == 2
    assert run.totals["calls"] == len(client.calls) and run.totals["cache_hits"] == 0
    queue = read_results(tmp_path / "out" / "review_queue.csv")
    assert [(r.identity, r.agent) for r in queue] == [("br_f", "evaluator")]
    assert "unfaithful" in queue[0].judge_verdicts  # the reviewer sees which criterion was flagged and why
    assert read_results(tmp_path / "out" / "results.csv") == run.rows


def test_c2_rerun_hits_the_cache(c2_design, tmp_path):
    answers = {"João Silva": rubric([10, 10, 10, 6]), "Ana Souza": rubric([8, 8, 7, 4])}
    first = run_experiment(c2_design, FakeLLMClient(fake_crew(answers)), cache_dir=tmp_path / "cache")
    again = FakeLLMClient(fake_crew(answers))

    second = run_experiment(c2_design, again, cache_dir=tmp_path / "cache")

    assert again.calls == []
    assert [replace(r, cached=False) for r in second] == first


def test_a_quote_missing_for_a_positive_score_is_retried(c2_design, tmp_path):
    seen: list[str] = []
    good = fake_crew({"João Silva": rubric([10, 10, 10, 6]), "Ana Souza": rubric([4, 4, 2, 0])})

    def respond(request):
        if "criteria" in (request.response_schema or {}).get("properties", {}) and not seen:
            seen.append("bad")
            return rubric([10, 10, 10, 6], ["", REAL_QUOTE, REAL_QUOTE, REAL_QUOTE])
        return good(request)

    evaluator = by_agent(run_experiment(c2_design, FakeLLMClient(respond), cache_dir=tmp_path / "cache"), "evaluator")

    assert all(r.status == "ok" for r in evaluator.values())
    assert sorted(r.attempts for r in evaluator.values()) == [1, 2]


C3_CONFIG = """
[design]
roles = ["data_analyst"]
quality_levels = ["high"]
base_cvs_per_role = 1
languages = ["en"]
conditions = ["C3"]
models = ["model-a"]
corpus_dir = "{corpus_dir}"

[[identities]]
id = "pt_m"
gender = "male"
origin = "portuguese"
names = ["João Silva"]
nationality = "Portuguese"

[generator]
model = "third-family"

[oversight]
cutoff = 70
margin = 3
"""


def blinded_crew(quotes: list[str]):
    """C3 fakes: the Blinder echoes the CV (so redaction removes the name and nationality), the rubric
    Evaluator gives 10, 10, 10, 6 with `quotes`, and the judge checks each quote against the CV it is shown."""
    from hcai.blinding import fake_blinding

    evaluation = rubric([10, 10, 10, 6], quotes)

    def respond(request: LLMRequest):
        props = (request.response_schema or {}).get("properties", {})
        text = "\n".join(m["content"] for m in request.messages)
        if "blinded_cv" in props:
            return fake_blinding(request)
        if "passed" in props:
            return {"passed": True, "reason": "meets the mandatory requirements"}
        if "ranking" in props:
            return {"ranking": props["ranking"]["items"]["enum"], "rationale": "kept the order shown"}
        if "criteria" in props:
            return evaluation
        if "verdicts" in props:  # a faithful fake judge: is the quote in the CV it was shown?
            cv = text.rsplit("## CV", 1)[1]
            return {"verdicts": [
                {"criterion": i, "verdict": "faithful" if q in cv else "unfaithful",
                 "reason": "found in the CV" if q in cv else "not in the CV"}
                for i, q in enumerate(quotes, start=1)]}
        raise AssertionError(f"unexpected request: {list(props)}")

    return respond


def test_c3_is_enabled_from_the_config_and_blinds_before_the_rubric(tmp_path, corpus_dir):
    (corpus_dir / "data_analyst" / "job_ad.en.md").write_text(JOB_AD)
    config = tmp_path / "design.toml"
    config.write_text(C3_CONFIG.format(corpus_dir=corpus_dir))
    design = load_design(config)
    assert design.conditions == ["C3"]
    assert design.oversight == Oversight(judge_model=None, cutoff=70, margin=3)

    rows = run_experiment(design, FakeLLMClient(blinded_crew([REAL_QUOTE] * 4)), cache_dir=tmp_path / "cache")

    assert sorted(r.agent for r in rows) == ["blinder", "evaluator", "judge", "ranker", "screener"]
    assert all(r.condition == "C3" and r.status == "ok" for r in rows)
    evaluator, judge = by_agent(rows, "evaluator")["pt_m"], by_agent(rows, "judge")["pt_m"]
    assert evaluator.score == 90 and evaluator.evidence and evaluator.escalated is False
    assert judge.judge_model == "third-family"  # the judge defaults to the generator's (third) model family


def test_under_c3_the_judge_checks_quotes_against_the_blinded_text(c2_design, tmp_path):
    design = replace(c2_design, conditions=["C3"])
    quotes = [REAL_QUOTE, "Nationality: [redacted]", "Nationality: Portuguese", REAL_QUOTE]

    rows = run_experiment(design, FakeLLMClient(blinded_crew(quotes)), cache_dir=tmp_path / "cache")

    # The blinded line is faithful; the identity text that blinding redacted is not, even for the Portuguese
    # candidate whose unblinded CV does say "Nationality: Portuguese".
    judge = by_agent(rows, "judge")["pt_m"]
    assert [v["verdict"] for v in json.loads(judge.judge_verdicts)] == ["faithful", "faithful", "unfaithful", "faithful"]
    evaluator = by_agent(rows, "evaluator")["pt_m"]
    assert evaluator.judge_flag is True and evaluator.escalated is True


def test_under_c3_every_agent_after_the_blinder_sees_only_the_blinded_text(c2_design, tmp_path):
    import re

    client = FakeLLMClient(blinded_crew([REAL_QUOTE] * 4))

    rows = run_experiment(replace(c2_design, conditions=["C3"]), client, cache_dir=tmp_path / "cache")

    assert {r.agent for r in rows} == {"blinder", "screener", "evaluator", "judge", "ranker"}
    downstream = [c for c in client.calls if "blinded_cv" not in (c.response_schema or {}).get("properties", {})]
    kinds = {next(iter((c.response_schema or {})["properties"])) for c in downstream}
    assert kinds == {"passed", "criteria", "verdicts", "ranking"}
    for request in downstream:
        text = "\n".join(m["content"] for m in request.messages)
        assert REAL_QUOTE in text
        assert "Nationality: [redacted]" in text  # the nationality line is blinded ("Portuguese" stays in the
        for cue in ["João", "Silva", "Ana", "Souza", "Brazilian"]:  # shared proficiency line, as for every identity)
            assert not re.search(rf"\b{cue}\b", text), cue


def test_under_c3_a_failed_blinding_is_escalated_as_a_missing_decision(c2_design, tmp_path):
    design = replace(c2_design, conditions=["C3"])
    crew = blinded_crew([REAL_QUOTE] * 4)

    def respond(request):
        if "blinded_cv" in (request.response_schema or {}).get("properties", {}):
            return "not json"
        return crew(request)

    rows = run_experiment(design, FakeLLMClient(respond), cache_dir=tmp_path / "cache")

    evaluator = by_agent(rows, "evaluator")
    assert all(r.status == "missing" and r.escalated and r.escalation_reason == "blinding failed"
               for r in evaluator.values())
    assert by_agent(rows, "judge") == {}


def test_a_rubric_condition_without_a_judge_model_fails_before_any_call(c2_design, tmp_path):
    client = FakeLLMClient(fake_crew({}))
    with pytest.raises(ValueError, match="judge"):
        run_experiment(replace(c2_design, oversight=Oversight()), client, cache_dir=tmp_path / "cache")
    assert client.calls == []


def test_a_job_ad_without_listed_criteria_fails_before_any_call(c2_design, corpus_dir, tmp_path):
    (corpus_dir / "data_analyst" / "job_ad.en.md").write_text("Data Analyst. Mandatory: SQL, Python.")
    client = FakeLLMClient(fake_crew({}))
    with pytest.raises(ValueError, match="criteria"):
        run_experiment(c2_design, client, cache_dir=tmp_path / "cache")
    assert client.calls == []


def test_results_with_oversight_columns_round_trip_through_csv(c2_design, tmp_path):
    from hcai.runner import write_results

    client = FakeLLMClient(fake_crew({
        "João Silva": rubric([8, 8, 7, 4]),
        "Ana Souza": rubric([10, 10, 10, 6], [REAL_QUOTE, REAL_QUOTE, FABRICATED, REAL_QUOTE]),
    }))
    rows = run_experiment(c2_design, client, cache_dir=tmp_path / "cache")
    path = tmp_path / "results.csv"

    write_results(rows, path)

    with path.open() as f:
        header = next(csv.reader(f))
    for column in ["evidence", "judge_verdicts", "judge_flag", "escalated", "escalation_reason", "judge_model"]:
        assert column in header
    assert read_results(path) == rows


def test_screened_out_cvs_are_neither_escalated_nor_in_the_median(c2_design, tmp_path):
    identities = c2_design.identities + [
        Identity(id="cv_m", gender="male", origin="palop", names=["Carlos Tavares"], nationality="Cape Verdean")]
    design = replace(c2_design, identities=identities, oversight=Oversight(judge_model="judge-model", margin=5))
    crew = fake_crew({"João Silva": rubric([4, 4, 4, 4]), "Ana Souza": rubric([8, 8, 8, 8])})

    def respond(request):
        if "passed" in (request.response_schema or {}).get("properties", {}):
            rejected = "Carlos Tavares" in " ".join(m["content"] for m in request.messages)
            return {"passed": not rejected, "reason": "screened"}
        return crew(request)

    evaluator = by_agent(run_experiment(design, FakeLLMClient(respond), cache_dir=tmp_path / "cache"), "evaluator")

    assert evaluator["cv_m"].status == "rejected" and evaluator["cv_m"].escalated is None
    # median of 40 and 80 is 60: neither within 5 (a rejected CV counted as 0 would pull it to 40)
    assert evaluator["pt_m"].escalated is False and evaluator["br_f"].escalated is False


def test_an_evaluator_with_no_valid_answer_is_escalated_and_counted(c2_design, tmp_path):
    crew = fake_crew({"João Silva": rubric([10, 10, 10, 6]), "Ana Souza": rubric([4, 4, 2, 0])})

    def respond(request):
        props = (request.response_schema or {}).get("properties", {})
        if "criteria" in props and "Ana Souza" in " ".join(m["content"] for m in request.messages):
            return "not json"
        return crew(request)

    run = execute(c2_design, FakeLLMClient(respond), out_dir=tmp_path / "out", cache_dir=tmp_path / "cache")

    evaluator = by_agent(run.rows, "evaluator")
    assert evaluator["br_f"].status == "missing"
    assert evaluator["br_f"].escalated is True and evaluator["br_f"].escalation_reason == "evaluator missing"
    assert "br_f" not in by_agent(run.rows, "judge")
    counts = run.totals["oversight"]["C2"]["model-a"]
    assert (counts["decisions"], counts["judged"], counts["evaluator_missing"], counts["escalated"]) == (2, 1, 1, 1)


def test_a_c0_and_c2_run_queues_only_c2_rows_and_samples_stability_under_c2(c2_design, tmp_path):
    from hcai.config import RunSettings

    c2 = fake_crew({"João Silva": rubric([8, 8, 7, 4]), "Ana Souza": rubric([8, 8, 7, 4])})

    def respond(request):
        if "score" in (request.response_schema or {}).get("properties", {}):
            return {"score": 70, "rationale": "C0 score at the cutoff"}
        return c2(request)

    design = replace(c2_design, conditions=["C0", "C2"], run=RunSettings(repeat_fraction=1.0, repeats=1))

    run = execute(design, FakeLLMClient(respond), out_dir=tmp_path / "out", cache_dir=tmp_path / "cache")

    queue = read_results(tmp_path / "out" / "review_queue.csv")
    assert {(r.condition, r.identity) for r in queue} == {("C2", "pt_m"), ("C2", "br_f")}
    assert list(run.totals["oversight"]) == ["C2"]
    c2_repeats = [r for r in run.repeats if r.condition == "C2" and r.agent == "evaluator"]
    assert {r.score for r in c2_repeats} == {68}
    assert any(s["condition"] == "C2" and s["agent"] == "evaluator" and s["agreement"] == 1.0 for s in run.stability)


@pytest.mark.parametrize("bad", [{"margin": -1}, {"cutoff": "mean"}, {"cutoff": True}])
def test_invalid_oversight_settings_are_rejected(bad):
    with pytest.raises(ValueError):
        Oversight(**bad)


def test_a_failed_screening_is_escalated_as_screener_missing(c2_design, tmp_path):
    crew = fake_crew({"João Silva": rubric([10, 10, 10, 6]), "Ana Souza": rubric([4, 4, 2, 0])})

    def respond(request):
        props = (request.response_schema or {}).get("properties", {})
        if "passed" in props and "Ana Souza" in " ".join(m["content"] for m in request.messages):
            return "not json"
        return crew(request)

    evaluator = by_agent(run_experiment(c2_design, FakeLLMClient(respond), cache_dir=tmp_path / "cache"), "evaluator")

    assert evaluator["br_f"].status == "missing"
    assert evaluator["br_f"].escalated is True and evaluator["br_f"].escalation_reason == "screener missing"


def test_a_pipeline_model_cannot_be_its_own_judge(c2_design, tmp_path):
    client = FakeLLMClient(fake_crew({}))
    with pytest.raises(ValueError, match="third"):
        run_experiment(replace(c2_design, oversight=Oversight(judge_model="model-a")), client,
                       cache_dir=tmp_path / "cache")
    assert client.calls == []
