"""Seam 1 (corpus generator): job ads and base CVs from a fake generator client, frozen on disk."""

from dataclasses import replace
from pathlib import Path

import pytest

from hcai.cli import main
from hcai.config import Design, Generator, load_design
from conftest import scores_by_name
from hcai.corpus import CorpusFrozen, GenerationFailed, generate_corpus, load_corpus, translate_corpus
from hcai.llm import FakeLLMClient, LLMRequest
from hcai.config import identities_from_name_bank
from hcai.names import load_name_bank
from hcai.runner import run_experiment
from hcai.variants import WORK_ELIGIBILITY, build_variants, render_variant

MANDATORY = ["Degree in a relevant field", "At least 2 years of experience", "Fluent Portuguese"]
DESIRABLE = ["Team leadership", "A professional certification"]
BANK = Path(__file__).parents[1] / "data" / "names" / "name_bank.toml"


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
    template = load_corpus(corpus_design.corpus_dir).base_cvs[0].template()
    variant = render_variant(template, name="Ana Souza", nationality="Brazilian", gender="female", language="en")
    assert "Ana Souza" in variant and "Nationality: Brazilian" in variant
    assert "$name" not in variant and "$5k" in variant  # a literal $ in generated text survives rendering


def test_the_generated_corpus_renders_through_the_slot_templates_for_every_name_bank_identity(corpus_design, tmp_path):

    generate_corpus(corpus_design, FakeLLMClient(generator_responder()), cache_dir=tmp_path / "gen-cache")
    design = replace(corpus_design, identities=identities_from_name_bank(load_name_bank(BANK)))

    variants = build_variants(design)  # raises if the demographic-only diff check fails

    assert len(variants) == 12 * 8
    assert all(line in v.text.splitlines() for v in variants for line in WORK_ELIGIBILITY["en"])


def test_a_base_cv_with_a_pronoun_outside_a_slot_is_regenerated(corpus_design, tmp_path):

    gendered = cv_response()
    gendered["summary"] = "She delivers results."
    answers = iter([gendered])

    def respond(request: LLMRequest) -> dict:
        if request.response_schema and request.response_schema.get("title") == "job_ad":
            return job_ad_response()
        return next(answers, cv_response())

    corpus = generate_corpus(corpus_design, FakeLLMClient(respond), cache_dir=tmp_path / "cache")

    assert all("She " not in cv.template() for cv in corpus.base_cvs)
    build_variants(corpus_design)


CORE = Path(__file__).parent.parent / "configs" / "core.toml"


def test_core_config_generates_a_corpus_from_the_command_line_and_refuses_to_overwrite_it(tmp_path, monkeypatch, capsys):

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    core = load_design(CORE)
    assert core.generator is not None and core.generator.model not in core.models
    config = tmp_path / "core.toml"
    config.write_text(
        CORE.read_text()
        .replace('corpus_dir = "../data/corpus/v1"', 'corpus_dir = "corpus/v1"')
        .replace('name_bank = "../data/names/name_bank.toml"', f'name_bank = "{BANK}"')
    )
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


def test_a_rerun_draws_fresh_answers_after_unparseable_generations(corpus_design, tmp_path):
    with pytest.raises(GenerationFailed):
        generate_corpus(corpus_design, FakeLLMClient(lambda request: "not json"), cache_dir=tmp_path / "cache")

    corpus = generate_corpus(corpus_design, FakeLLMClient(generator_responder()), cache_dir=tmp_path / "cache")

    assert corpus.base_cvs


# --- PT-PT translations (#14) ------------------------------------------------------------------------------


def pt_job_ad_response() -> dict:
    return {
        "title": "Emprego", "location": "Lisboa", "summary": "Uma empresa em Lisboa está a contratar.",
        "mandatory": ["Licenciatura numa área relevante", "Pelo menos 2 anos de experiência", "Português fluente"],
        "desirable": ["Liderança de equipas", "Uma certificação profissional"],
    }


def pt_cv_response(**changes) -> dict:
    """The PT-PT translation of `cv_response()`: same structure and dates, gendered words in slots."""
    cv = {
        "headline": "{{Profissional licenciado|Profissional licenciada}} com 3 anos de experiência",
        "summary": "Apresenta resultados com custos abaixo de $5k por projeto; {{inscrito|inscrita}} na Ordem.",
        "experience": [{
            "title": "{{Técnico Especialista|Técnica Especialista}}", "employer": "Empresa Lda", "location": "Lisboa",
            "start": "2021", "end": "presente", "bullets": ["Fez bem o seu trabalho com a equipa."],
        }],
        "education": [{"degree": "Licenciatura em Algo", "institution": "Universidade de Lisboa", "year": "2020"}],
        "skills": ["Competência A", "Competência B"],
        "certifications": [],
    }
    return {**cv, **changes}


def bilingual_responder(pt_cvs=None, pt_ad=None):
    """Fake generator: EN job ads and base CVs, and their PT-PT translations (answers from `pt_cvs` first)."""
    answers = iter(pt_cvs or [])

    def respond(request: LLMRequest) -> dict:
        title = (request.response_schema or {}).get("title")
        if title == "job_ad":
            return job_ad_response()
        if title == "job_ad_translation":
            return pt_ad or pt_job_ad_response()
        if title == "base_cv_translation":
            return next(answers, pt_cv_response())
        return cv_response()

    return respond


@pytest.fixture
def bilingual_design(corpus_design: Design) -> Design:
    return replace(corpus_design, languages=["en", "pt"], identities=identities_from_name_bank(load_name_bank(BANK)))


def test_every_base_cv_and_job_ad_has_a_pt_counterpart_with_the_same_id_and_quality(bilingual_design, tmp_path):

    generate_corpus(bilingual_design, FakeLLMClient(bilingual_responder()), cache_dir=tmp_path / "cache")
    corpus = load_corpus(bilingual_design.corpus_dir)

    assert corpus.languages == ["en", "pt"]
    assert sorted(corpus.translated_job_ads["pt"]) == ["accountant", "nurse"]
    for role in ["nurse", "accountant"]:
        ad = (bilingual_design.corpus_dir / role / "job_ad.pt.md").read_text()
        assert "## Requisitos obrigatórios" in ad and "Pelo menos 2 anos de experiência" in ad
    for cv in corpus.base_cvs:
        path = bilingual_design.corpus_dir / f"{cv.id}.pt.md"
        assert path.read_text() == cv.template("pt")
        assert "## Experiência" in path.read_text() and "Nacionalidade: $nationality" in path.read_text()
    by_language = {lang: {(v.base_cv, v.quality) for v in build_variants(replace(bilingual_design, languages=[lang]))}
                   for lang in ["en", "pt"]}
    assert by_language["pt"] == by_language["en"] == {(cv.id, cv.quality) for cv in corpus.base_cvs}


def test_pt_variants_pass_the_diff_check_and_inflect_by_gender_for_all_eight_identities(bilingual_design, tmp_path):

    generate_corpus(bilingual_design, FakeLLMClient(bilingual_responder()), cache_dir=tmp_path / "cache")

    variants = [v for v in build_variants(bilingual_design) if v.language == "pt"]  # raises if the check fails

    assert len(variants) == 12 * 8
    assert len({v.identity for v in variants}) == 8
    for v in variants:
        assert all(line in v.text.splitlines() for line in WORK_ELIGIBILITY["pt"])
        assert "Autorização para trabalhar em Portugal" in v.text and "Português: proficiência" in v.text
        feminine = v.gender == "female"
        assert ("Profissional licenciada" in v.text) == feminine and ("Técnica Especialista" in v.text) == feminine
        assert ("inscrito" in v.text) != feminine
        assert "$5k" in v.text


@pytest.mark.parametrize("bad, reason", [
    (pt_cv_response(summary="Gerenciamento de projetos abaixo de $5k com a equipe."), "Brazilian"),
    (pt_cv_response(summary="Enfermeiro com custos abaixo de $5k."), "slot"),
    (pt_cv_response(summary="{{Analista escolhido|Analista escolhida}} pela direção; escolhido abaixo de $5k."), "gendered"),
    (pt_cv_response(skills=["Competência A"]), "skills"),
    (pt_cv_response(experience=[{**pt_cv_response()["experience"][0], "start": "2019"}]), "start"),
    (pt_cv_response(headline="Profissional com 5 anos de experiência"), "number"),
])
def test_a_translation_that_is_not_faithful_pt_pt_with_slotted_gendered_words_is_regenerated(
    bilingual_design, tmp_path, bad, reason,
):
    client = FakeLLMClient(bilingual_responder(pt_cvs=[bad]))

    corpus = generate_corpus(bilingual_design, client, cache_dir=tmp_path / "cache")

    translations = [c for c in client.calls if (c.response_schema or {}).get("title") == "base_cv_translation"]
    assert len(translations) == 12 + 1  # the bad answer was retried once
    assert all(cv.translations["pt"] == pt_cv_response() for cv in corpus.base_cvs)
    build_variants(bilingual_design)

    always_bad = FakeLLMClient(bilingual_responder(pt_cvs=[bad] * 3))
    with pytest.raises(GenerationFailed, match=reason):
        generate_corpus(bilingual_design, always_bad, cache_dir=tmp_path / "other-cache", force=True)


def test_an_invariant_title_holding_the_other_gender_s_form_is_not_a_leak(bilingual_design, tmp_path):
    head_nurse = pt_cv_response(
        summary="{{Enfermeiro|Enfermeira}} com custos abaixo de $5k; reporta à {{Enfermeira-Chefe|Enfermeira-Chefe}}.",
    )
    client = FakeLLMClient(bilingual_responder(pt_cvs=[head_nurse] * 12))

    corpus = generate_corpus(bilingual_design, client, cache_dir=tmp_path / "cache")

    assert all(cv.translations["pt"] == head_nurse for cv in corpus.base_cvs)
    male = next(v for v in build_variants(bilingual_design) if v.language == "pt" and v.gender == "male")
    assert "Enfermeiro com custos" in male.text and "reporta à Enfermeira-Chefe" in male.text


def test_a_generator_that_keeps_writing_brazilian_portuguese_fails_and_writes_nothing(bilingual_design, tmp_path):

    brazilian = pt_job_ad_response() | {"summary": "Uma empresa procura um desenvolvedor para o time."}
    client = FakeLLMClient(bilingual_responder(pt_ad=brazilian))

    with pytest.raises(GenerationFailed, match="Brazilian"):
        generate_corpus(bilingual_design, client, cache_dir=tmp_path / "cache")
    assert not bilingual_design.corpus_dir.exists()


def test_translations_are_made_by_the_third_family_generator_not_an_audited_model(bilingual_design, tmp_path):

    client = FakeLLMClient(bilingual_responder())
    generate_corpus(bilingual_design, client, cache_dir=tmp_path / "cache")

    translations = [c for c in client.calls if "translation" in (c.response_schema or {}).get("title", "")]
    assert len(translations) == 2 + 12  # one per job ad and base CV
    assert {c.model for c in translations} == {"generator/model"}
    assert load_corpus(bilingual_design.corpus_dir).translated_by["pt"].model == "generator/model"

    audited = replace(bilingual_design, generator=Generator(model=bilingual_design.models[0]))
    with pytest.raises(ValueError, match="third family"):
        translate_corpus(audited, FakeLLMClient(bilingual_responder()), cache_dir=tmp_path / "cache", force=True)


def test_translating_a_frozen_english_corpus_adds_pt_and_leaves_the_english_files_unchanged(bilingual_design, tmp_path):

    english_only = replace(bilingual_design, languages=["en"])
    generate_corpus(english_only, FakeLLMClient(bilingual_responder()), cache_dir=tmp_path / "cache")
    root = bilingual_design.corpus_dir
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file() and p.name != "manifest.json"}

    translate_corpus(bilingual_design, FakeLLMClient(bilingual_responder()), cache_dir=tmp_path / "cache")

    assert {p: p.read_bytes() for p in before} == before
    corpus = load_corpus(root)
    assert corpus.languages == ["en", "pt"]
    assert [cv.id for cv in corpus.base_cvs] == [cv.id for cv in load_corpus(root).base_cvs]
    assert all((root / f"{cv.id}.pt.md").exists() for cv in corpus.base_cvs)
    build_variants(bilingual_design)

    again = FakeLLMClient(bilingual_responder())
    with pytest.raises(CorpusFrozen):
        translate_corpus(bilingual_design, again, cache_dir=tmp_path / "other-cache")
    assert again.calls == []
    translator = replace(bilingual_design, generator=Generator(model="generator/other"))
    translate_corpus(translator, again, cache_dir=tmp_path / "other-cache", force=True)
    assert len(again.calls) == 2 + 12
    retranslated = load_corpus(root)
    assert retranslated.translated_by["pt"].model == "generator/other"
    assert retranslated.generator.model == "generator/model"  # the EN provenance is unchanged


def test_an_unsupported_language_is_refused_before_any_llm_call(bilingual_design, tmp_path):
    client = FakeLLMClient(bilingual_responder())

    with pytest.raises(ValueError, match="'fr'"):
        generate_corpus(replace(bilingual_design, languages=["en", "fr"]), client, cache_dir=tmp_path / "cache")
    assert client.calls == []


def test_core_config_runs_in_en_and_pt_with_rows_for_every_agent(tmp_path, monkeypatch):

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert load_design(CORE).languages == ["en", "pt"]
    config = tmp_path / "core.toml"
    config.write_text(
        CORE.read_text()
        .replace('corpus_dir = "../data/corpus/v1"', 'corpus_dir = "corpus/v1"')
        .replace('name_bank = "../data/names/name_bank.toml"', f'name_bank = "{BANK}"')
    )
    assert main(["corpus", str(config), "--cache", str(tmp_path / "gen-cache"), "--fake-llm"]) == 0
    config.write_text(config.read_text().replace('corpus_dir = "corpus/v1"', 'corpus_dir = "corpus/v1-fake"'))
    core = load_design(config)
    # A slice of the core design, so the test stays fast: one base CV per role x quality level, one model.
    design = replace(core, base_cvs_per_role=1, models=core.models[:1])

    rows = run_experiment(design, FakeLLMClient(scores_by_name({}, default=50)), cache_dir=tmp_path / "run-cache")

    assert {(r.language, r.agent) for r in rows} == {(lang, agent) for lang in ["en", "pt"]
                                                     for agent in ["screener", "evaluator", "ranker"]}
    assert all(r.status == "ok" for r in rows)
    per_language = {lang: sum(r.agent == "evaluator" and r.language == lang for r in rows) for lang in ["en", "pt"]}
    assert per_language == {"en": 3 * 3 * 8, "pt": 3 * 3 * 8}


def test_the_fake_generator_survives_a_retried_translation(bilingual_design, tmp_path):
    from hcai.corpus import fake_generator_response

    first = iter([True])

    def respond(request: LLMRequest) -> dict:
        answer = fake_generator_response(request)
        if (request.response_schema or {}).get("title") == "base_cv_translation" and next(first, False):
            return {**answer, "skills": []}  # rejected, so the next request carries a retry note
        return answer

    corpus = generate_corpus(bilingual_design, FakeLLMClient(respond), cache_dir=tmp_path / "cache")

    assert all("pt" in cv.translations for cv in corpus.base_cvs)
    build_variants(bilingual_design)
