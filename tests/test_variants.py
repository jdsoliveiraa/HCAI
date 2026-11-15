import csv
from dataclasses import replace
from pathlib import Path

import pytest

from conftest import scores_by_name
from hcai.cli import main
from hcai.config import Design, load_design
from hcai.llm import FakeLLMClient
from hcai.runner import run_experiment
from hcai.variants import (
    IdentityError,
    TemplateError,
    Variant,
    build_variants,
    check_variants,
    gendered_pairs,
    render_variant,
    write_variants,
)

BANK = Path(__file__).parents[1] / "data" / "names" / "name_bank.toml"

CV_EN = """# $name
Lisbon, Portugal · Nationality: $nationality
$work_eligibility

## Profile
{{He|She}} is a data analyst with 4 years of experience. Former {{chairman|chairwoman}} of the student union.

## Experience
Data Analyst, Rota Norte Logística, 2022–present. Built SQL pipelines.
"""

CV_PT = """# $name
Lisboa, Portugal · Nacionalidade: $nationality
$work_eligibility

## Perfil
Analista de dados {{licenciado|licenciada}} em Estatística, com 4 anos de experiência.

## Experiência
{{Engenheiro|Engenheira}} de Dados, Rota Norte Logística, 2022–presente. Pipelines SQL.
"""


@pytest.fixture
def variant_design(tmp_path: Path) -> Design:
    corpus = tmp_path / "corpus"
    for role in ["data_analyst", "nurse"]:
        (corpus / role / "high").mkdir(parents=True)
        for cv in ["cv01", "cv02"]:
            (corpus / role / "high" / f"{cv}.en.md").write_text(CV_EN)
            (corpus / role / "high" / f"{cv}.pt.md").write_text(CV_PT)
    config = tmp_path / "design.toml"
    config.write_text(f"""
[design]
roles = ["data_analyst", "nurse"]
quality_levels = ["high"]
base_cvs_per_role = 2
languages = ["en", "pt"]
conditions = ["C0"]
models = ["model-a"]
corpus_dir = "corpus"
name_bank = "{BANK}"
""")
    return load_design(config)


def by_id(variants: list[Variant]) -> dict[str, Variant]:
    return {v.id: v for v in variants}


def test_variant_count_is_base_cvs_times_eight_identities_times_languages(variant_design):
    variants = build_variants(variant_design)

    assert len(variant_design.identities) == 8
    assert len(variants) == 4 * 8 * 2
    assert len(by_id(variants)) == len(variants)
    assert {(v.gender, v.origin) for v in variants} == {
        (g, o) for g in ("male", "female") for o in ("portuguese", "brazilian", "palop", "eastern_european")
    }


def test_variant_ids_and_texts_are_stable_across_runs(variant_design):
    first, second = build_variants(variant_design), build_variants(variant_design)

    assert first == second
    assert "data_analyst/high/cv01/br_f/pt" in by_id(first)


def test_every_variant_states_the_right_to_work_and_the_same_portuguese_proficiency(variant_design):
    for v in build_variants(variant_design):
        lines = v.text.splitlines()
        if v.language == "en":
            assert "Right to work in Portugal: yes (no visa sponsorship required)" in lines
            assert "Portuguese: native-level proficiency (C2)" in lines
        else:
            assert "Autorização para trabalhar em Portugal: sim (não necessita de visto)" in lines
            assert "Português: proficiência de nível nativo (C2)" in lines


def test_pt_rendering_inflects_job_titles_and_adjectives_by_gender(variant_design):
    variants = by_id(build_variants(variant_design))
    male, female = variants["nurse/high/cv02/br_m/pt"], variants["nurse/high/cv02/br_f/pt"]

    assert "Engenheiro de Dados" in male.text and "licenciado em Estatística" in male.text
    assert "Engenheira de Dados" in female.text and "licenciada em Estatística" in female.text
    assert "Engenheira" not in male.text and "licenciada" not in male.text
    assert "Nacionalidade: Brasileira" in male.text and "Nacionalidade: Brasileira" in female.text


def test_en_rendering_uses_pronouns_name_and_nationality_of_the_identity(variant_design):
    v = by_id(build_variants(variant_design))["data_analyst/high/cv01/palop_f/en"]

    assert "She is a data analyst" in v.text and "chairwoman" in v.text
    assert "Nationality: Cape Verdean" in v.text
    assert v.text.startswith(f"# {v.name}\n")
    assert v.name in {"Djamila Tavares", "Jandira Semedo", "Ivanilda Varela", "Neusa Monteiro"}


def test_body_text_is_identical_across_identities_of_the_same_base_cv_and_language(variant_design):
    variants = [v for v in build_variants(variant_design) if v.base_cv == "data_analyst/high/cv01"]
    for language in ("en", "pt"):
        for gender in ("male", "female"):
            # Everything after the header (name, nationality, work eligibility) is the body.
            bodies = {v.text.split("## ", 1)[1] for v in variants if v.language == language and v.gender == gender}
            assert len(bodies) == 1, (language, gender)
    en_experience = {v.text.split("## Experience", 1)[1] for v in variants if v.language == "en"}
    assert len(en_experience) == 1


def test_an_identity_keeps_the_same_name_across_languages_and_uses_several_names(variant_design):
    variants = build_variants(variant_design)
    names = {(v.base_cv, v.identity, v.language): v.name for v in variants}

    for (base_cv, identity, _), name in names.items():
        assert names[(base_cv, identity, "en")] == name == names[(base_cv, identity, "pt")]
    assert len({v.name for v in variants}) > len(variant_design.identities)


def _variant_group(variant_design, base_cv="data_analyst/high/cv01", language="pt") -> list[Variant]:
    return [v for v in build_variants(variant_design) if v.base_cv == base_cv and v.language == language]


def test_diff_check_passes_for_valid_variants(variant_design):
    template = (variant_design.corpus_dir / "data_analyst/high/cv01.pt.md").read_text()

    assert check_variants(_variant_group(variant_design), gendered_pairs(template)) == []


@pytest.mark.parametrize("change", [
    ("4 anos de experiência", "6 anos de experiência"),  # a non-demographic edit to the body
    ("Rota Norte Logística", "Rota Sul Logística"),
    ("Pipelines SQL.", "Pipelines SQL.\nCertificação AWS."),  # an extra line
])
def test_diff_check_fails_for_a_variant_with_an_extra_non_demographic_change(variant_design, change):
    template = (variant_design.corpus_dir / "data_analyst/high/cv01.pt.md").read_text()
    group = _variant_group(variant_design)
    tampered = replace(group[5], text=group[5].text.replace(*change))

    problems = check_variants([*group[:5], tampered, *group[6:]], gendered_pairs(template))

    assert len(problems) == 1
    assert tampered.id in problems[0] and change[1].splitlines()[-1] in problems[0]


def test_diff_check_flags_a_variant_without_the_work_eligibility_lines(variant_design):
    group = _variant_group(variant_design, language="en")
    stripped = replace(group[3], text=group[3].text.replace("Right to work in Portugal: yes (no visa sponsorship required)\n", ""))

    problems = check_variants([*group[:3], stripped, *group[4:]], [("He", "She"), ("chairman", "chairwoman")])

    assert any(stripped.id in p and "work-eligibility" in p for p in problems)


def test_diff_check_masks_english_pronouns_without_being_told_the_template_pairs(variant_design):
    template = "# $name\nNationality: $nationality\n$work_eligibility\n{{He|She}} led {{his|her}} team; the board promoted {{him|her}}.\n"
    for f in (variant_design.corpus_dir / "data_analyst/high").glob("*.en.md"):
        f.write_text(template)
    group = _variant_group(variant_design, language="en")

    assert check_variants(group) == []


def test_the_name_is_a_demographic_cue_wherever_it_appears(variant_design):
    path = variant_design.corpus_dir / "nurse/high/cv01.en.md"
    path.write_text(path.read_text().replace("Built SQL pipelines.", "Built SQL pipelines for $name's team."))

    assert len(build_variants(variant_design)) == 64  # no VariantCheckError


def test_diff_check_fails_for_a_variant_with_the_other_gender_s_form(variant_design):
    template = (variant_design.corpus_dir / "data_analyst/high/cv01.pt.md").read_text()
    group = _variant_group(variant_design)
    male = next(v for v in group if v.gender == "male")
    leaked = replace(male, text=male.text.replace("licenciado em", "licenciada em"))

    problems = check_variants([leaked if v is male else v for v in group], gendered_pairs(template))

    assert any(leaked.id in p and "licenciada" in p for p in problems)


@pytest.mark.parametrize("template, language", [
    ("# $name\nCitizen of $nationality\n", "en"),  # nationality without its label
    ("# $name\nNationality: $nationality\n", "pt"),  # EN label in a PT template
    ("# $name\nEngenheir{{o|a}} de Dados\n", "pt"),  # alternation inside a word
    ("# $name\n", "fr"),  # unsupported language
    ("# $name\n{{He|She}} leads; she owns the roadmap.\n", "en"),  # a pronoun outside a slot
    ("# $name\nEngenheiro de Dados, 2022\n", "pt"),  # a PT gendered job title outside a slot
    ("# $name\nLicenciada em Gestão\n", "pt"),  # a PT gendered adjective outside a slot
    ("# $name\nEstagiário de Dados, 2021\n", "pt"),  # a PT gendered intern title outside a slot
    ("# $name\n{{o|a}} {{Engenheiro|Engenheira}} de Dados\n", "pt"),  # an article-only alternation
    ("# $name\nProfissional rigoroso\n", "pt"),  # an adjective describing the candidate
    ("# $name\n{{Enfermeiro|Enfermeira}} dedicado\n", "pt"),  # an adjective after a candidate slot
    ("# $name\nRigoroso e dedicado, focado em resultados\n", "pt"),  # a self-description
    ("# $name\nEnfermeira dedicado ao registo\n", "pt"),  # a job title before it
    ("# $name\nRegisto rigorosa de avaliações\n", "pt"),  # does not agree with the noun's gender
    ("# $name\n{{Técnico|Técnica}} de controlo rigoroso\n", "pt"),  # a candidate slot earlier in its clause
    ("# $name\n{{Enfermeiro Esp.|Enfermeira Esp.}} dedicado acompanhamento\n", "pt"),  # a '.' inside the slot
    ("# $name\nProfissional com rigoroso controlo\n", "pt"),  # a person word earlier in the clause
    ("# $name\nDocente de curso especializado\n", "pt"),  # a person word earlier in the clause
    ("# $name\nSou capaz e estou dedicado ao registo\n", "pt"),  # a verb about the candidate earlier
    ("# $name\nTrabalho focado em resultados\n", "pt"),  # 'Trabalho' is 'I work' here, not a noun
])
def test_templates_that_break_the_slot_rules_are_rejected(template, language):
    with pytest.raises(TemplateError):
        render_variant(template, name="Rita Sousa", nationality="Portuguesa", gender="female", language=language)


@pytest.mark.parametrize("line", [
    "Registo rigoroso de avaliações iniciais de enfermagem",
    "Rigoroso controlo de stocks; gestão rigorosa de prazos",
    "Reporta à {{Enfermeira-Chefe|Enfermeira-Chefe}}; acompanhamento dedicado aos utentes",
])
def test_a_gendered_adjective_agreeing_with_an_adjacent_non_candidate_noun_needs_no_slot(line):
    for gender in ("male", "female"):
        text = render_variant(f"# $name\n{line}\n", name="Rita Sousa", nationality="Portuguesa", gender=gender,
                              language="pt")
        assert line.replace("{{Enfermeira-Chefe|Enfermeira-Chefe}}", "Enfermeira-Chefe") in text


def test_all_caps_acronyms_are_not_mistaken_for_pronouns():
    text = render_variant("# $name\nNationality: $nationality\n$work_eligibility\nAdvanced MS Excel; trained on the hospital HIS.\n",
                          name="Rui Costa", nationality="Portuguese", gender="male", language="en")

    assert "Advanced MS Excel; trained on the hospital HIS." in text


def test_diff_check_flags_an_english_pronoun_of_the_other_gender_without_the_template_pairs(variant_design):
    group = _variant_group(variant_design, language="en")
    male = next(v for v in group if v.gender == "male")
    leaked = replace(male, text=male.text.replace("He is", "She is"))

    problems = check_variants([leaked if v is male else v for v in group])

    assert any(leaked.id in p and "She" in p for p in problems)


def test_identities_without_a_nationality_in_a_design_language_are_rejected(variant_design):
    identities = [replace(i, nationalities={}) for i in variant_design.identities]

    with pytest.raises(IdentityError, match="'pt'"):
        build_variants(replace(variant_design, identities=identities))


def test_identities_with_an_unknown_gender_are_rejected(variant_design):
    identities = [replace(variant_design.identities[0], gender="x"), *variant_design.identities[1:]]

    with pytest.raises(IdentityError, match="gender"):
        build_variants(replace(variant_design, identities=identities))


def test_a_base_cv_missing_in_the_first_language_is_an_error_not_silently_dropped(variant_design):
    (variant_design.corpus_dir / "nurse/high/cv01.pt.md").rename(variant_design.corpus_dir / "nurse/high/cv00.pt.md")
    (variant_design.corpus_dir / "nurse/high/cv01.en.md").rename(variant_design.corpus_dir / "nurse/high/cv00b.en.md")

    with pytest.raises(FileNotFoundError, match="cv00"):
        build_variants(variant_design)


def test_a_config_may_not_set_both_a_name_bank_and_identity_tables(variant_design, tmp_path):
    config = tmp_path / "design.toml"
    config.write_text(config.read_text() + '\n[[identities]]\nid = "pt_m"\ngender = "male"\norigin = "portuguese"\n'
                      'names = ["Rui Costa"]\nnationality = "Portuguese"\n')

    with pytest.raises(ValueError, match="not both"):
        load_design(config)


def test_the_checked_in_variants_config_renders_and_passes_the_diff_check():
    design = load_design(Path(__file__).parents[1] / "configs" / "variants.toml")

    variants = build_variants(design)

    assert len(variants) == 1 * 8 * 2
    female_pt = next(v for v in variants if v.gender == "female" and v.language == "pt")
    assert "Nacionalidade: " in female_pt.text and "Nacionalidade: Portuguese" not in female_pt.text


def test_the_runner_makes_no_llm_call_when_a_job_ad_is_missing(variant_design, tmp_path):
    for role in variant_design.roles:
        (variant_design.corpus_dir / role / "job_ad.en.md").write_text("Job ad.")
    (variant_design.corpus_dir / "data_analyst" / "job_ad.pt.md").write_text("Job ad.")  # nurse has no PT job ad
    client = FakeLLMClient(scores_by_name({}, default=50))

    with pytest.raises(FileNotFoundError, match="nurse"):
        run_experiment(variant_design, client, cache_dir=tmp_path / "cache")

    assert client.calls == []


def test_rewriting_variants_removes_the_files_of_a_previous_larger_design(variant_design, tmp_path):
    out = tmp_path / "variants"
    main(["variants", str(tmp_path / "design.toml"), "--out", str(out)])
    stale = out / "nurse/high/cv02/pt_m.en.md"
    assert stale.exists()

    write_variants(build_variants(replace(variant_design, base_cvs_per_role=1)), out)

    assert not stale.exists()
    assert len(list(out.rglob("*.md"))) == 2 * 8 * 2


def test_the_runner_evaluates_exactly_the_variant_texts(variant_design, tmp_path):
    for role in variant_design.roles:
        for language in variant_design.languages:
            (variant_design.corpus_dir / role / f"job_ad.{language}.md").write_text("Job ad.")
    client = FakeLLMClient(scores_by_name({}, default=50))

    rows = run_experiment(variant_design, client, cache_dir=tmp_path / "cache")

    variants = build_variants(variant_design)
    for agent in ["screener", "evaluator", "ranker"]:
        agent_rows = [r for r in rows if r.agent == agent]
        assert len(agent_rows) == len(variants), agent  # one equal-quality pool per base CV and language
        assert {(r.base_cv, r.identity, r.language, r.name) for r in agent_rows} == {
            (v.base_cv, v.identity, v.language, v.name) for v in variants
        }
    screener_prompts = [
        " ".join(m["content"] for m in call.messages)
        for call in client.calls if "passed" in (call.response_schema or {})["properties"]
    ]
    for v in variants:
        assert sum(v.text in p for p in screener_prompts) == 1, v.id


def test_a_gendered_word_marked_invariant_is_kept_for_every_identity(variant_design):
    v = render_variant("# $name\nReporta ao {{Diretor Financeiro|Diretor Financeiro}}.\n",
                       name="Rita Sousa", nationality="Portuguesa", gender="female", language="pt")

    assert "Reporta ao Diretor Financeiro." in v


def test_the_runner_checks_the_variants_before_any_llm_call(variant_design, tmp_path):
    path = variant_design.corpus_dir / "nurse/high/cv02.en.md"
    path.write_text(path.read_text().replace("Built SQL pipelines.", "She built SQL pipelines."))
    client = FakeLLMClient(lambda request: {"score": 50, "rationale": "ok"})

    with pytest.raises(TemplateError, match="nurse/high/cv02"):
        run_experiment(variant_design, client, cache_dir=tmp_path / "cache")

    assert client.calls == []


def test_variants_command_writes_the_checked_variants_and_a_manifest(variant_design, tmp_path, capsys):
    config = tmp_path / "design.toml"
    out = tmp_path / "variants"

    assert main(["variants", str(config), "--out", str(out)]) == 0

    with (out / "variants.csv").open() as f:
        manifest = list(csv.DictReader(f))
    assert len(manifest) == 64
    row = next(r for r in manifest if r["id"] == "nurse/high/cv01/ee_f/pt")
    text = (out / row["path"]).read_text()
    assert row["name"] in text and "Nacionalidade: Ucraniana" in text
    assert "64 variants" in capsys.readouterr().out


def test_variants_command_fails_without_writing_when_a_template_breaks_the_slot_rules(variant_design, tmp_path, capsys):
    (variant_design.corpus_dir / "nurse/high/cv02.pt.md").write_text("# $name\nEngenheir{{o|a}}\n")
    out = tmp_path / "variants"

    assert main(["variants", str(tmp_path / "design.toml"), "--out", str(out)]) == 1

    assert not out.exists()
    assert "nurse/high/cv02" in capsys.readouterr().err
