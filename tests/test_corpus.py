"""Seam 1 (corpus generator): job ads and base CVs from a fake generator client, frozen on disk."""

from dataclasses import replace
from pathlib import Path

import pytest

from hcai.cli import main
from hcai.config import Design, Generator, Identity, load_design
from hcai.corpus import CorpusFrozen, GenerationFailed, generate_corpus, load_corpus
from hcai.llm import FakeLLMClient, LLMRequest
from hcai.runner import render_cv, run_experiment

MANDATORY = ["Degree in a relevant field", "At least 2 years of experience", "Fluent Portuguese"]
DESIRABLE = ["Team leadership", "A professional certification"]


def job_ad_response(role: str = "role") -> dict:
    return {
        "title": f"Job for {role}",
        "location": "Lisbon",
        "summary": "A company in Lisbon is hiring.",
        "mandatory": MANDATORY,
        "desirable": DESIRABLE,
    }


def cv_response(headline: str = "Professional with 3 years of experience", cover: list[int] | None = None) -> dict:
    """A structured base CV whose evidence quotes its own text, covering the given mandatory requirements."""
    cover = [1, 2, 3] if cover is None else cover
    quotes = {1: "BSc in Something, University of Lisbon", 2: "2021–present", 3: "header"}
    return {
        "headline": headline,
        "summary": "Delivers results and costs below $5k per project.",
        "experience": [{
            "title": "Specialist", "employer": "Empresa Lda", "location": "Lisbon", "start": "2021", "end": "present",
            "bullets": ["Did the job well."],
        }],
        "education": [{"degree": "BSc in Something", "institution": "University of Lisbon", "year": "2020"}],
        "skills": ["Skill A", "Skill B"],
        "certifications": [],
        "mandatory_evidence": [{"requirement": i, "evidence": quotes[i]} for i in cover],
        "desirable_met": [1],
        "borderline_note": "",
    }


def generator_responder(cv: dict | None = None):
    """Fake generator: answers job-ad requests and base-CV requests by their response schema."""

    def respond(request: LLMRequest) -> dict:
        assert request.response_schema is not None
        if request.response_schema.get("title") == "job_ad":
            return job_ad_response()
        return cv if cv is not None else cv_response()

    return respond


def brazilian() -> tuple[Identity, str]:
    return Identity(id="br_f", gender="female", origin="brazilian", names=["Ana Souza"], nationality="Brazilian"), "Ana Souza"


@pytest.fixture
def corpus_design(tmp_path: Path, design: Design) -> Design:
    return replace(
        design,
        roles=["nurse", "accountant"],
        quality_levels=["strong", "average", "weak"],
        base_cvs_per_role=2,
        corpus_dir=tmp_path / "corpus" / "v1",
        generator=Generator(model="generator/model"),
    )


def test_corpus_has_one_job_ad_per_role_and_the_configured_base_cvs_per_role_and_quality(corpus_design, tmp_path):

    corpus = generate_corpus(corpus_design, FakeLLMClient(generator_responder()), cache_dir=tmp_path / "cache")

    assert sorted(corpus.job_ads) == ["accountant", "nurse"]
    for ad in corpus.job_ads.values():
        assert ad.mandatory == MANDATORY and ad.desirable == DESIRABLE
    assert [cv.id for cv in corpus.base_cvs] == [
        f"{role}/{quality}/cv{n:02d}"
        for role in ["nurse", "accountant"] for quality in ["strong", "average", "weak"] for n in (1, 2)
    ]
    assert all(cv.id.split("/")[:2] == [cv.role, cv.quality] for cv in corpus.base_cvs)


def test_quality_and_borderline_tags_are_stored_as_ground_truth_in_the_frozen_corpus(corpus_design, tmp_path):

    design = replace(corpus_design, generator=Generator(model="generator/model", borderline={"weak": 1, "average": 1}))
    borderline_cv = cv_response()
    borderline_cv["borderline_note"] = "exactly the minimum 2 years"
    generate_corpus(design, FakeLLMClient(generator_responder(borderline_cv)), cache_dir=tmp_path / "cache")

    corpus = load_corpus(design.corpus_dir)

    assert corpus.version == "v1" and corpus.generator.model == "generator/model"
    tagged = {cv.id for cv in corpus.base_cvs if cv.borderline}
    assert tagged == {f"{role}/{q}/cv02" for role in ["nurse", "accountant"] for q in ["average", "weak"]}
    assert {cv.id: cv.quality for cv in corpus.base_cvs}["accountant/weak/cv01"] == "weak"
    assert corpus.job_ads["nurse"].mandatory == MANDATORY


def test_a_base_cv_missing_a_mandatory_requirement_is_regenerated(corpus_design, tmp_path):

    answers = iter([cv_response(cover=[1, 2])])  # the first CV skips mandatory requirement 3

    def respond(request: LLMRequest) -> dict:
        if request.response_schema and request.response_schema.get("title") == "job_ad":
            return job_ad_response()
        return next(answers, cv_response())

    client = FakeLLMClient(respond)
    corpus = generate_corpus(corpus_design, client, cache_dir=tmp_path / "cache")

    assert len(corpus.base_cvs) == 12
    for cv in corpus.base_cvs:
        assert sorted(e["requirement"] for e in cv.mandatory_evidence) == [1, 2, 3]


def test_a_generator_that_never_meets_the_mandatory_requirements_fails_and_writes_nothing(corpus_design, tmp_path):

    client = FakeLLMClient(generator_responder(cv_response(cover=[1])))

    with pytest.raises(GenerationFailed, match="mandatory"):
        generate_corpus(corpus_design, client, cache_dir=tmp_path / "cache")
    assert not corpus_design.corpus_dir.exists()


def test_evidence_that_is_not_quoted_from_the_cv_does_not_count(corpus_design, tmp_path):

    invented = cv_response()
    invented["mandatory_evidence"][0]["evidence"] = "PhD from a university not in the CV"

    with pytest.raises(GenerationFailed):
        generate_corpus(corpus_design, FakeLLMClient(generator_responder(invented)), cache_dir=tmp_path / "cache")


def test_regenerating_a_frozen_corpus_is_refused_unless_forced(corpus_design, tmp_path):

    generate_corpus(corpus_design, FakeLLMClient(generator_responder()), cache_dir=tmp_path / "cache")
    frozen = (corpus_design.corpus_dir / "nurse" / "strong" / "cv01.en.md").read_text()
    rewritten = FakeLLMClient(generator_responder(cv_response(headline="A different candidate")))

    with pytest.raises(CorpusFrozen):
        generate_corpus(corpus_design, rewritten, cache_dir=tmp_path / "other-cache")
    assert rewritten.calls == []
    assert (corpus_design.corpus_dir / "nurse" / "strong" / "cv01.en.md").read_text() == frozen

    generate_corpus(corpus_design, rewritten, cache_dir=tmp_path / "other-cache", force=True)
    corpus = load_corpus(corpus_design.corpus_dir)
    assert corpus.base_cvs[0].content["headline"] == "A different candidate"
    assert [cv.id for cv in corpus.base_cvs][:2] == ["nurse/strong/cv01", "nurse/strong/cv02"]


def test_a_directory_that_is_not_a_generated_corpus_is_never_overwritten(corpus_design, tmp_path):

    corpus_design.corpus_dir.mkdir(parents=True)
    (corpus_design.corpus_dir / "notes.md").write_text("hand-written")

    with pytest.raises(CorpusFrozen):
        generate_corpus(corpus_design, FakeLLMClient(generator_responder()), cache_dir=tmp_path / "c", force=True)
    assert (corpus_design.corpus_dir / "notes.md").read_text() == "hand-written"


def test_the_generator_model_is_configured_apart_from_the_pipeline_models(tmp_path):

    config = tmp_path / "design.toml"
    config.write_text(
        '[design]\nroles = ["nurse"]\nquality_levels = ["strong", "weak"]\nbase_cvs_per_role = 1\n'
        'languages = ["en"]\nconditions = ["C0"]\nmodels = ["pipeline/a", "pipeline/b"]\ncorpus_dir = "corpus/v2"\n\n'
        '[generator]\nmodel = "third-family/generator"\nborderline = { weak = 1 }\n'
    )
    design = load_design(config)
    assert design.generator == Generator(model="third-family/generator", borderline={"weak": 1})

    client = FakeLLMClient(generator_responder({**cv_response(), "borderline_note": "just meets requirement 2"}))
    generate_corpus(design, client, cache_dir=tmp_path / "cache")

    assert {c.model for c in client.calls} == {"third-family/generator"}
    assert (tmp_path / "corpus" / "v2" / "manifest.json").exists()


def test_a_design_without_a_generator_cannot_generate_a_corpus(corpus_design, tmp_path):

    with pytest.raises(ValueError, match="generator"):
        generate_corpus(replace(corpus_design, generator=None), FakeLLMClient(generator_responder()), cache_dir=tmp_path)


def test_the_runner_evaluates_the_generated_corpus_with_identities_filled_in(corpus_design, fake_client, tmp_path):

    generate_corpus(corpus_design, FakeLLMClient(generator_responder()), cache_dir=tmp_path / "gen-cache")

    rows = run_experiment(corpus_design, fake_client, cache_dir=tmp_path / "run-cache")

    evaluator_rows = [r for r in rows if r.agent == "evaluator"]
    assert len(evaluator_rows) == 12 * 2
    assert all(r.status == "ok" for r in rows)
    variant = render_cv(load_corpus(corpus_design.corpus_dir).base_cvs[0].template(), *brazilian())
    assert "Ana Souza" in variant and "Nationality: Brazilian" in variant
    assert "$name" not in variant and "$5k" in variant  # a literal $ in generated text survives rendering


CORE = Path(__file__).parent.parent / "configs" / "core.toml"


def test_core_config_generates_a_corpus_from_the_command_line_and_refuses_to_overwrite_it(tmp_path, monkeypatch, capsys):

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    core = load_design(CORE)
    assert core.generator is not None and core.generator.model not in core.models
    config = tmp_path / "core.toml"
    config.write_text(CORE.read_text().replace('corpus_dir = "../data/corpus/v1"', 'corpus_dir = "corpus/v1"'))
    args = ["corpus", str(config), "--cache", str(tmp_path / "cache"), "--fake-llm"]

    assert main(args) == 0
    assert not (tmp_path / "corpus" / "v1").exists()  # fake corpora never land on the real version
    corpus = load_corpus(tmp_path / "corpus" / "v1-fake")
    assert len(corpus.base_cvs) == len(core.roles) * len(core.quality_levels) * core.base_cvs_per_role
    assert any(cv.borderline for cv in corpus.base_cvs)
    assert "base CVs" in capsys.readouterr().out

    assert main(args) == 1
    assert "frozen" in capsys.readouterr().err
    assert main([*args, "--force"]) == 0


def test_the_generator_model_may_not_be_one_of_the_audited_pipeline_models(corpus_design, tmp_path):
    design = replace(corpus_design, generator=Generator(model=corpus_design.models[0]))

    with pytest.raises(ValueError, match="third family"):
        generate_corpus(design, FakeLLMClient(generator_responder()), cache_dir=tmp_path / "cache")
