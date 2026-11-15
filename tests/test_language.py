"""The language check: function words of English vs Portuguese, proper nouns tolerated."""

import pytest

from hcai.language import count, language_problem


def test_portuguese_text_from_the_real_corpus_is_not_english():
    fields = [
        ("summary", "A Lusitânia Tech Solutions está a recrutar um Engenheiro de Software para a sua equipa em Lisboa."),
        ("mandatory[1]", "Licenciatura em Engenharia Informática ou área afim"),
        ("headline", "Contabilista Certificado"),
    ]
    problem = language_problem(fields, "en")
    assert "must be written in English" in problem and "summary" in problem


def test_english_text_with_portuguese_proper_nouns_is_english():
    fields = [
        ("summary", "Certified accountant registered with the Ordem dos Contabilistas Certificados (OCC), with "
                    "5 years of experience in tax compliance, e.g. IRC and IVA returns."),
        ("education[1].degree", "Licenciatura (bachelor's degree) em Contabilidade e Fiscalidade"),
        ("experience[1].bullets[1]", "Prepared the monthly accounts for 40 clients of the Porto e Norte office."),
    ]
    assert language_problem(fields, "en") == ""


def test_a_single_portuguese_field_in_an_english_document_is_named():
    fields = [("summary", "Nurse with 6 years of experience in the emergency department of a large hospital."),
              ("experience[1].bullets[2]", "Prestação de cuidados de enfermagem a doentes críticos com uma equipa")]
    problem = language_problem(fields, "en")
    assert "experience[1].bullets[2]" in problem and "summary" not in problem.split("text in:")[1]


def test_a_translation_left_in_english_is_not_portuguese():
    fields = [("summary", "Delivers results and costs below $5k per project."),
              ("experience[1].bullets[1]", "Did the job well with the team.")]
    assert "must be written in Portuguese" in language_problem(fields, "pt")
    assert language_problem([("summary", "Apresenta resultados com custos abaixo de $5k por projeto.")], "pt") == ""


@pytest.mark.parametrize("text, en, pt", [
    ("University of Lisbon", 0, 0),  # inside a name
    ("Universidade de Lisboa", 0, 0),
    ("SQL Server, NA region, OS upgrades", 0, 0),  # acronyms and words common to both languages
    ("e.g. budgets", 0, 0),
    ("gestão de equipas e de projetos", 0, 3),
])
def test_count_skips_proper_nouns_acronyms_and_shared_words(text, en, pt):
    assert count(text) == {"en": en, "pt": pt}
