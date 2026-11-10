"""Runner seam for the 3-agent crew (Screener -> Evaluator -> Ranker), driven by the fake LLM client."""

from collections import Counter
from dataclasses import replace

import pytest

from hcai.config import Design, Identity, Pools, load_design
from hcai.llm import FakeLLMClient, LLMRequest
from hcai.runner import ResultRow, read_results, run_experiment, write_results

NAMES = ["Tiago Martins", "Ana Souza", "Rui Costa", "Joana Lopes", "Kwame Tavares", "Inês Mendes", "Andriy Koval", "Olena Petrenko"]


@pytest.fixture
def pool_design(design: Design, corpus_dir) -> Design:
    """8 identities, 2 base CVs, 2 models: two equal-quality pools of 8 per model."""
    (corpus_dir / "data_analyst" / "high" / "cv2.en.md").write_text(
        "$name\nNationality: $nationality\n$work_eligibility\nSQL, R."
    )
    identities = [
        Identity(id=f"id{i}", gender=("male", "female")[i % 2], origin="y", names=[n], nationality="Z")
        for i, n in enumerate(NAMES)
    ]
    return Design(**{**design.__dict__, "identities": identities, "base_cvs_per_role": 2, "models": ["model-a", "model-b"]})


def prompt(request: LLMRequest) -> str:
    return "\n".join(m["content"] for m in request.messages)


def crew_responder(reject: frozenset[str] | set[str] = frozenset(), prefer: list[str] | None = None):
    """Fake responses per agent, recognised by the response schema.

    Screener rejects CVs naming anyone in `reject`; Evaluator gives 60; Ranker orders the candidates
    by `prefer` (names), or keeps the order it was shown when `prefer` is None.
    """

    def respond(request: LLMRequest) -> dict:
        props = (request.response_schema or {})["properties"]
        text = prompt(request)
        if "passed" in props:
            return {"passed": not any(n in text for n in reject), "reason": "screened"}
        if "score" in props:
            return {"score": 60, "rationale": "fine"}
        labels = props["ranking"]["items"]["enum"]
        if prefer is None:
            return {"ranking": labels, "rationale": "as shown"}
        blocks = text.split("### Candidate ")[1:]
        order = {name: i for i, name in enumerate(prefer)}
        shown = {label: next(n for n in NAMES if n in blocks[label - 1]) for label in labels}
        return {"ranking": sorted(labels, key=lambda label: order[shown[label]]), "rationale": "preferred"}

    return respond


def positions(rows: list[ResultRow]) -> list[int]:
    """The recorded Ranker input positions, sorted."""
    return sorted(r.input_position for r in rows if r.input_position is not None)


def by_agent(rows: list[ResultRow]) -> dict[str, list[ResultRow]]:
    out: dict[str, list[ResultRow]] = {}
    for r in rows:
        out.setdefault(r.agent, []).append(r)
    return out


def test_each_agent_gets_its_own_rows_for_every_cv_and_pool_member(pool_design, tmp_path):
    rows = run_experiment(pool_design, FakeLLMClient(crew_responder()), cache_dir=tmp_path / "cache")

    counts = Counter(r.agent for r in rows)
    # 2 base CVs x 8 identities x 2 models; one equal-quality pool per base CV.
    assert counts == {"screener": 32, "evaluator": 32, "ranker": 32}
    assert all(r.status == "ok" for r in rows)
    agents = by_agent(rows)
    assert all(r.screen_pass is True and r.rationale for r in agents["screener"])
    assert all(r.score == 60 for r in agents["evaluator"])
    assert all(r.rank is not None and r.top3 is not None and r.pool for r in agents["ranker"])


def test_ranker_input_order_is_shuffled_recorded_and_a_permutation_per_pool(pool_design, tmp_path):
    # The fake Ranker returns candidates in the order shown, so rank == the recorded input position
    # exactly when the recorded position is the one the Ranker saw.
    rows = run_experiment(pool_design, FakeLLMClient(crew_responder()), cache_dir=tmp_path / "cache")

    pools: dict[tuple[str, str], list[ResultRow]] = {}
    for r in by_agent(rows)["ranker"]:
        pools.setdefault((r.pool, r.model), []).append(r)
    assert len(pools) == 4
    for members in pools.values():
        assert positions(members) == list(range(1, 9))
        assert all(r.rank == r.input_position for r in members)
        assert len({r.order_seed for r in members}) == 1 and members[0].order_seed is not None
        assert sum(bool(r.top3) for r in members) == 3
    identity_orders = {tuple(r.identity for r in sorted(m, key=lambda r: r.input_position or 0)) for m in pools.values()}
    assert len(identity_orders) == 4  # every pool and model saw its own order
    assert tuple(f"id{i}" for i in range(8)) not in identity_orders


def test_rank_follows_the_candidate_not_the_position(pool_design, tmp_path):
    rows = run_experiment(pool_design, FakeLLMClient(crew_responder(prefer=NAMES[::-1])), cache_dir=tmp_path / "cache")

    for r in by_agent(rows)["ranker"]:
        assert r.rank == 8 - NAMES.index(r.name)
        assert r.top3 == (r.name in NAMES[-3:])


def test_rejected_cvs_are_marked_and_neither_scored_nor_ranked(pool_design, tmp_path):
    client = FakeLLMClient(crew_responder(reject={"Ana Souza", "Rui Costa"}))

    rows = run_experiment(pool_design, client, cache_dir=tmp_path / "cache")

    agents = by_agent(rows)
    assert len(agents["screener"]) == len(agents["evaluator"]) == len(agents["ranker"]) == 32
    rejected = {"Ana Souza", "Rui Costa"}
    assert all((r.screen_pass is False) == (r.name in rejected) for r in agents["screener"])
    for r in agents["evaluator"] + agents["ranker"]:
        if r.name in rejected:
            assert (r.status, r.score, r.rank, r.input_position) == ("rejected", None, None, None)
        else:
            assert r.status == "ok"
    ranked = [r for r in agents["ranker"] if r.status == "ok"]
    assert positions([r for r in ranked if r.pool.endswith("cv1") and r.model == "model-a"]) == list(range(1, 7))
    evaluator_calls = [c for c in client.calls if "score" in (c.response_schema or {})["properties"]]
    assert len(evaluator_calls) == 24


def test_mixed_quality_pools_come_from_the_config(pool_design, corpus_dir, tmp_path):
    for quality, n in [("medium", 3), ("low", 3)]:
        (corpus_dir / "data_analyst" / quality).mkdir()
        for i in range(n):
            (corpus_dir / "data_analyst" / quality / f"cv{i}.en.md").write_text(f"$name\n$work_eligibility\n{quality} CV {i}")
    design = replace(pool_design, quality_levels=["high", "medium", "low"], base_cvs_per_role=3, models=["model-a"],
                     pools=Pools(equal_quality=False, mixed_quality=2, mixed_size=8))

    rows = run_experiment(design, FakeLLMClient(crew_responder()), cache_dir=tmp_path / "cache")

    ranker = by_agent(rows)["ranker"]
    assert {r.pool for r in ranker} == {"data_analyst/mixed01", "data_analyst/mixed02"}
    assert {r.pool_kind for r in ranker} == {"mixed"}
    for pool in ["data_analyst/mixed01", "data_analyst/mixed02"]:
        members = [r for r in ranker if r.pool == pool]
        assert len(members) == 8
        assert len({r.base_cv for r in members}) == 8  # distinct base CVs
        assert {r.base_cv.split("/")[1] for r in members} == {"high", "medium", "low"}
        assert positions(members) == list(range(1, 9))
    # Screener and Evaluator rows stay one per CV, however many pools a CV is in.
    assert len(by_agent(rows)["screener"]) == 8 * 8


def test_pool_composition_is_read_from_the_toml_config(tmp_path, corpus_dir):
    config = tmp_path / "design.toml"
    config.write_text(f"""
[design]
roles = ["data_analyst"]
quality_levels = ["high"]
base_cvs_per_role = 1
languages = ["en"]
conditions = ["C0"]
models = ["m"]
corpus_dir = "{corpus_dir}"

[pools]
equal_quality = false
mixed_quality = 3
mixed_size = 6

[[identities]]
id = "a"
gender = "male"
origin = "portuguese"
names = ["A"]
nationality = "Portuguese"
""")

    assert load_design(config).pools == Pools(equal_quality=False, mixed_quality=3, mixed_size=6)


def test_second_run_makes_no_calls_and_reproduces_the_order(pool_design, tmp_path):
    first = run_experiment(pool_design, FakeLLMClient(crew_responder()), cache_dir=tmp_path / "cache")
    client = FakeLLMClient(crew_responder())

    second = run_experiment(pool_design, client, cache_dir=tmp_path / "cache")

    assert client.calls == []
    assert [replace(r, cached=False) for r in second] == first


def test_malformed_ranker_output_marks_the_pool_missing(pool_design, tmp_path):
    def respond(request: LLMRequest):
        if "ranking" in (request.response_schema or {})["properties"]:
            return {"ranking": [1, 1, 2], "rationale": "bad"}
        return crew_responder()(request)

    rows = run_experiment(pool_design, FakeLLMClient(respond), cache_dir=tmp_path / "cache")

    ranker = by_agent(rows)["ranker"]
    assert all(r.status == "missing" and r.rank is None and r.error for r in ranker)
    assert all(r.input_position is not None for r in ranker)  # the order shown is still recorded


def test_results_with_pipeline_columns_round_trip_through_csv(pool_design, tmp_path):
    rows = run_experiment(pool_design, FakeLLMClient(crew_responder(reject={"Ana Souza"})), cache_dir=tmp_path / "c")

    write_results(rows, tmp_path / "results.csv")

    assert read_results(tmp_path / "results.csv") == rows
