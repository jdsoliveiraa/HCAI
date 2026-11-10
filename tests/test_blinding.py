"""Condition C1: a blinding agent rewrites each CV before the crew sees it; runner seam with the fake client."""

import re
from collections import Counter
from dataclasses import replace

import pytest

from hcai.blinding import fake_blinding
from hcai.config import Design, Identity
from hcai.llm import FakeLLMClient, LLMRequest
from hcai.runner import run_experiment

BLIND_CV = "[Candidate]\nRight to work in Portugal: yes\nSQL, Python, 3 years of experience."


def text_of(request: LLMRequest) -> str:
    return "\n".join(m["content"] for m in request.messages)


def is_blinder(request: LLMRequest) -> bool:
    return "blinded_cv" in (request.response_schema or {}).get("properties", {})


def crew(blinded: str | None = BLIND_CV):
    """Fake responses: the blinder returns `blinded` (or echoes the CV it was given when None); the Screener
    passes, the Evaluator gives 60 and the Ranker keeps the order shown."""

    def respond(request: LLMRequest) -> dict:
        props = (request.response_schema or {})["properties"]
        if "blinded_cv" in props:
            return fake_blinding(request) if blinded is None else {"blinded_cv": blinded}
        if "passed" in props:
            return {"passed": True, "reason": "ok"}
        if "score" in props:
            return {"score": 60, "rationale": "fine"}
        return {"ranking": props["ranking"]["items"]["enum"], "rationale": "as shown"}

    return respond


@pytest.fixture
def c1_design(design: Design) -> Design:
    return replace(design, conditions=["C0", "C1"])


def test_c1_rows_are_tagged_and_each_cv_gets_a_blinder_row(c1_design, tmp_path):
    rows = run_experiment(c1_design, FakeLLMClient(crew()), cache_dir=tmp_path / "cache")

    counts = Counter((r.condition, r.agent) for r in rows)
    assert counts == {
        ("C0", "screener"): 2, ("C0", "evaluator"): 2, ("C0", "ranker"): 2,
        ("C1", "blinder"): 2, ("C1", "screener"): 2, ("C1", "evaluator"): 2, ("C1", "ranker"): 2,
    }
    blinder = [r for r in rows if r.agent == "blinder"]
    assert all(r.status == "ok" and r.rationale == BLIND_CV and r.prompt_tokens > 0 for r in blinder)
    assert {r.identity for r in blinder} == {"pt_m", "br_f"}


def test_downstream_agents_only_receive_the_blinded_text(c1_design, tmp_path):
    client = FakeLLMClient(crew())

    run_experiment(replace(c1_design, conditions=["C1"]), client, cache_dir=tmp_path / "cache")

    downstream = [r for r in client.calls if not is_blinder(r)]
    # Both identities blind to the same text, so their Screener and Evaluator requests are one cached call each.
    agents = {next(iter((r.response_schema or {})["properties"])) for r in downstream}
    assert agents == {"passed", "score", "ranking"}
    for request in downstream:
        text = text_of(request)
        assert "SQL, Python, 3 years of experience." in text
        for cue in ["João", "Silva", "Ana", "Souza", "Portuguese", "Brazilian", "Nationality"]:
            assert not re.search(rf"\b{cue}\b", text)


def screener_texts(client: FakeLLMClient) -> list[str]:
    return [text_of(r) for r in client.calls if "passed" in (r.response_schema or {}).get("properties", {})]


@pytest.fixture
def gendered_design(c1_design: Design, corpus_dir) -> Design:
    """The base CV with EN pronouns in slots, a gendered word that is the same for both, and a third identity
    whose name only appears in the name bank."""
    (corpus_dir / "data_analyst" / "high" / "cv1.en.md").write_text(
        "# $name\nLisbon, Portugal · Nationality: $nationality\n$work_eligibility\n\n"
        "{{He|She}} reported to {{his|her}} manager. {{Ms|Ms}} Costa, {{her|her}} mentor, praised {{him|her}}.\n"
        "SQL, Python, 3 years of experience.\n"
    )
    extra = Identity(id="ua_m", gender="male", origin="ukrainian", names=["Andriy Koval"], nationality="Ukrainian")
    return replace(c1_design, conditions=["C1"], identities=[*c1_design.identities, extra])


def test_cues_the_blinder_leaves_in_are_redacted_before_the_crew_sees_them(gendered_design, tmp_path):
    client = FakeLLMClient(crew(blinded=None))  # the blinder echoes the CV unchanged

    rows = run_experiment(gendered_design, client, cache_dir=tmp_path / "cache")

    texts = screener_texts(client)
    assert len(texts) == 1  # every identity's redacted CV is the same text, so one Screener call served all three
    text = texts[0]
    shared_line = "Portuguese: native-level proficiency (C2)"
    for cue in ["João", "Silva", "Ana", "Souza", "Andriy", "Koval", "Portuguese", "Brazilian", "Ukrainian",
                "He", "She", "his", "him", "her"]:
        assert not re.search(rf"(?<!\w){cue}(?!\w)", text.replace(shared_line, "")), cue
    # Lines that are the same for every identity are kept, even where they name a nationality's language.
    assert shared_line in text
    assert "Right to work in Portugal: yes" in text
    assert "SQL, Python, 3 years of experience." in text
    blinder = [r for r in rows if r.agent == "blinder"]
    assert all(r.status == "ok" and "[redacted]" in r.rationale for r in blinder)


def test_a_name_from_the_name_bank_written_by_the_blinder_is_redacted(gendered_design, tmp_path):
    client = FakeLLMClient(crew(blinded="Candidate: Andriy Koval (Ukrainian)\nSQL, Python, 3 years of experience."))

    run_experiment(gendered_design, client, cache_dir=tmp_path / "cache")

    for text in screener_texts(client):
        assert "Andriy" not in text and "Koval" not in text and "Ukrainian" not in text
        assert "SQL, Python, 3 years of experience." in text


PT_CV = """# $name

Lisboa, Portugal · Nacionalidade: $nationality
$work_eligibility
Outras línguas: Inglês (C1), Francês (B2)

## Perfil
{{Licenciado|Licenciada}} em Estatística, com 4 anos de experiência.

## Formação
Licenciatura em Estatística, NOVA IMS, Lisboa, 2020
Pós-graduação em Ciência de Dados, Universidade de Coimbra, Coimbra, 2022
"""

# What a Blinder might return: the name and nationality are gone, but it wrote two feminine words, kept one
# university and one of the other languages, and dropped the location.
PT_BLINDED = """Candidata dedicada, com 4 anos de experiência.
Línguas: Inglês (C1)
Licenciatura em Estatística, NOVA IMS, 2020"""


@pytest.fixture
def pt_design(design: Design, corpus_dir) -> Design:
    (corpus_dir / "data_analyst" / "high" / "cv1.pt.md").write_text(PT_CV)
    (corpus_dir / "data_analyst" / "job_ad.pt.md").write_text("Analista de Dados. Obrigatório: SQL, Python.")
    identities = [
        replace(i, nationalities={"pt": n}) for i, n in zip(design.identities, ["Portuguesa", "Brasileira"])
    ]
    return replace(design, languages=["pt"], conditions=["C1"], identities=identities)


def test_proxy_leak_report_counts_surviving_gendered_words_and_origin_linked_fields(pt_design, tmp_path):
    from hcai.blinding import proxy_leaks

    rows = run_experiment(pt_design, FakeLLMClient(crew(PT_BLINDED)), cache_dir=tmp_path / "cache")

    leaks = {leak.identity: leak for leak in proxy_leaks(pt_design, rows)}
    assert set(leaks) == {"pt_m", "br_f"}
    female, male = leaks["br_f"], leaks["pt_m"]
    assert (female.own_gender_words, female.other_gender_words) == (2, 0)  # candidata, dedicada
    assert (male.own_gender_words, male.other_gender_words) == (0, 2)
    for leak in (female, male):
        assert leak.redacted == 0
        assert (leak.universities_kept, leak.universities) == (1, 2)
        assert (leak.languages_kept, leak.languages) == (1, 2)
        assert leak.location_kept is False
        assert (leak.language, leak.condition, leak.model) == ("pt", "C1", "model-a")


def test_proxy_leak_report_counts_the_cues_the_blinder_left_in(pt_design, tmp_path):
    from hcai.blinding import proxy_leaks

    rows = run_experiment(pt_design, FakeLLMClient(crew(blinded=None)), cache_dir=tmp_path / "cache")

    leaks = {leak.identity: leak for leak in proxy_leaks(pt_design, rows)}
    # The echoed CV still has the name, the nationality and the gendered word: 3 redactions each.
    assert {i: leak.redacted for i, leak in leaks.items()} == {"pt_m": 3, "br_f": 3}
    assert all(leak.location_kept and leak.universities_kept == 2 and leak.languages_kept == 2
               for leak in leaks.values())
    assert all(leak.own_gender_words == leak.other_gender_words == 0 for leak in leaks.values())


def test_a_cv_whose_blinding_fails_is_missing_for_every_agent_and_never_sent_unblinded(c1_design, tmp_path):
    from hcai.config import RunSettings

    def respond(request: LLMRequest):
        return "no JSON here" if is_blinder(request) else crew()(request)

    client = FakeLLMClient(respond)
    design = replace(c1_design, conditions=["C1"], run=RunSettings(retries=1))

    rows = run_experiment(design, client, cache_dir=tmp_path / "cache")

    assert all(is_blinder(r) for r in client.calls) and len(client.calls) == 4  # 2 CVs x (1 try + 1 retry)
    assert Counter((r.agent, r.status) for r in rows) == {
        ("blinder", "missing"): 2, ("screener", "missing"): 2, ("evaluator", "missing"): 2, ("ranker", "missing"): 2,
    }
    assert all(r.error for r in rows)


def test_execute_writes_the_proxy_leak_report(pt_design, tmp_path):
    import csv
    import json

    from hcai.runner import execute

    run = execute(pt_design, FakeLLMClient(crew(PT_BLINDED)), out_dir=tmp_path / "out", cache_dir=tmp_path / "cache")

    with (tmp_path / "out" / "proxy_leaks.csv").open() as f:
        table = list(csv.DictReader(f))
    assert {r["identity"] for r in table} == {"pt_m", "br_f"}
    summary = json.loads((tmp_path / "out" / "proxy_leaks.json").read_text())
    assert summary == run.proxy_leaks
    [group] = summary
    assert (group["cvs"], group["universities_kept"], group["languages_kept"], group["location_kept"]) == (2, 0.5, 0.5, 0.0)
    assert group["with_own_gender_words"] == 1  # the female variant: candidata, dedicada


def test_a_c0_only_run_writes_no_proxy_leak_report(design, fake_client, tmp_path):
    from hcai.runner import execute

    execute(design, fake_client, out_dir=tmp_path / "out", cache_dir=tmp_path / "cache")

    assert not (tmp_path / "out" / "proxy_leaks.csv").exists()


def test_fake_llm_run_of_a_c1_config_reports_proxy_leaks(tmp_path, capsys):
    from pathlib import Path

    from hcai.cli import main

    repo = Path(__file__).parent.parent
    config = tmp_path / "c1.toml"
    config.write_text(
        (repo / "configs" / "variants.toml").read_text()
        .replace('conditions = ["C0"]', 'conditions = ["C0", "C1"]')
        .replace('"../data/', f'"{repo}/data/')
    )

    exit_code = main(["run", str(config), "--out", str(tmp_path / "out"), "--cache", str(tmp_path / "cache"), "--fake-llm"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Proxy leaks" in out
    assert (tmp_path / "out" / "proxy_leaks.csv").exists()
