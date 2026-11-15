"""The language check: marker words of English vs Portuguese, proper nouns tolerated."""

import pytest

from hcai.language import Field, count, language_problem


def test_portuguese_text_from_the_real_corpus_is_not_english():
    fields = [
        Field("summary", "A Lusitânia Tech Solutions está a recrutar um Engenheiro de Software para a sua equipa em Lisboa."),
        Field("mandatory[1]", "Licenciatura em Engenharia Informática ou área afim"),
        Field("headline", "Contabilista Certificado", "title"),
    ]
    problem = language_problem(fields, "en")
    assert "must be written in English" in problem and "summary" in problem


ENGLISH_ACCOUNTANT_CV = [
    Field("headline", "Contabilista Certificado (Certified Accountant) with 8 years in audit and tax", "title"),
    Field("summary", "Certified accountant registered with the Ordem dos Contabilistas Certificados (OCC), with "
                     "8 years of experience in tax compliance, e.g. IRC and IVA returns."),
    Field("experience[1].title", "Senior Accountant", "title"),
    Field("experience[1].bullets[1]", "Prepared the monthly accounts for 40 clients of the Porto e Norte office."),
    Field("experience[1].bullets[2]", "Rolled out e-invoicing via e-Fatura and e-commerce payments, e. g. for retail."),
    Field("education[1].degree", "Licenciatura (bachelor's degree) em Contabilidade e Fiscalidade", "name"),
    Field("skills[1]", "e-commerce, e-payments and e-invoicing", "name"),
    Field("skills[2]", "SNC (Sistema de Normalização Contabilística)", "name"),
]
ENGLISH_NURSE_CV = [
    Field("headline", "Registered Nurse (Enfermeiro) in emergency care", "title"),
    Field("summary", "Nurse registered with the Ordem dos Enfermeiros, with 6 years in the emergency department."),
    Field("experience[1].title", "Staff Nurse, Emergency Department", "title"),
    Field("experience[1].bullets[1]", "Triaged patients using the Manchester Triage System at Hospital de Santa Maria."),
    Field("certifications[1]", "Suporte Avançado de Vida (Advanced Life Support), Conselho Português de Ressuscitação", "name"),
]
ENGLISH_SWE_JOB_AD = [
    Field("title", "Software Engineer (Backend)", "title"),
    Field("summary", "Lusitânia Tech Solutions is hiring a backend engineer for its team in Lisbon."),
    Field("mandatory[1]", "Licenciatura or Mestrado in Computer Science or a related field"),
    Field("mandatory[2]", "At least 3 years of experience with Java or Kotlin"),
    Field("desirable[1]", "Experience with e-commerce platforms"),
]


@pytest.mark.parametrize("fields", [ENGLISH_ACCOUNTANT_CV, ENGLISH_NURSE_CV, ENGLISH_SWE_JOB_AD])
def test_realistic_english_documents_with_portuguese_proper_nouns_are_english(fields):
    assert language_problem(fields, "en") == ""


def test_a_single_portuguese_field_in_an_english_document_is_named():
    fields = [Field("summary", "Nurse with 6 years of experience in the emergency department of a large hospital."),
              Field("experience[1].bullets[2]", "Prestação de cuidados de enfermagem a doentes críticos com uma equipa")]
    problem = language_problem(fields, "en")
    assert "experience[1].bullets[2]" in problem and "summary" not in problem.split("text in:")[1]


def test_portuguese_titles_and_skills_in_an_english_cv_are_caught():
    fields = [
        Field("headline", "Contabilista Certificado", "title"),
        Field("experience[1].title", "Técnico de Contabilidade", "title"),
        Field("experience[1].bullets[1]", "Prepared the monthly accounts for clients."),
        Field("skills[1]", "Contabilidade financeira", "name"),
        Field("skills[2]", "Reconciliações bancárias", "name"),
    ]
    problem = language_problem(fields, "en")
    assert "headline" in problem and "experience[1].title" in problem


PT_TRANSLATION = [
    Field("headline", "{{Enfermeiro|Enfermeira}} com 6 anos de experiência em urgência", "title"),
    Field("summary", "{{Inscrito|Inscrita}} na Ordem dos Enfermeiros, com experiência no serviço de urgência."),
    Field("experience[1].bullets[1]", "Triagem de doentes com o Sistema de Triagem de Manchester."),
    Field("skills[1]", "Epic EHR", "name"),
    Field("certifications[1]", "Introduction to the IFRS for SMEs (online course by the IFRS Foundation)", "name"),
    Field("certifications[2]", "Basic Life Support (BLS) for healthcare providers", "name"),
]


def test_a_portuguese_translation_keeping_english_certification_names_is_portuguese():
    assert language_problem(PT_TRANSLATION, "pt") == ""


def test_a_translation_left_in_english_is_not_portuguese():
    fields = [Field("summary", "Delivers results and costs below $5k per project."),
              Field("experience[1].bullets[1]", "Did the job well with the team.")]
    assert "must be written in Portuguese" in language_problem(fields, "pt")
    assert language_problem([Field("summary", "Apresenta resultados com custos abaixo de $5k por projeto.")], "pt") == ""


@pytest.mark.parametrize("text, en, pt", [
    ("University of Lisbon", 0, 0),  # inside a name
    ("Universidade de Lisboa", 0, 0),
    ("SQL Server, NA region, OS upgrades", 0, 0),  # acronyms and words common to both languages
    ("e.g. budgets, e. g. audits", 0, 0),
    ("e-commerce and e-Fatura", 1, 0),
    ("gestão de equipas e de projetos", 0, 4),  # 'gestão': a letter English never uses
    ("{{Enfermeiro|Enfermeira}} de serviço", 0, 2),  # a slot does not hide the words after it
])
def test_count_skips_proper_nouns_acronyms_and_shared_words(text, en, pt):
    assert count(text) == {"en": en, "pt": pt}
