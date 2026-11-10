"""The PT-PT check flags common PT-BR terms, and PT-PT templates that use them are rejected."""

import pytest

from hcai.ptpt import ptbr_terms
from hcai.variants import TemplateError, render_variant

PT_HEADER = "$name\nNacionalidade: $nationality\n$work_eligibility\n\n"


@pytest.mark.parametrize("text, flagged", [
    ("Formação de usuários finais", ["usuários"]),
    ("Liderou uma equipe de 5 pessoas", ["equipe"]),
    ("Gerenciamento de projetos", ["Gerenciamento"]),
    ("Concluiu o ensino médio em 2010", ["ensino médio"]),
    ("Graduação em Enfermagem", ["Graduação"]),
    ("Registro de dados no banco de dados", ["Registro", "banco de dados"]),
    ("Experiência em UTI", ["UTI"]),
    ("Atualmente está trabalhando num hospital", ["está trabalhando"]),
    ("Prêmio de mérito acadêmico", ["Prêmio", "acadêmico"]),
])
def test_common_ptbr_terms_are_flagged(text, flagged):
    assert ptbr_terms(text) == flagged


@pytest.mark.parametrize("text", [
    "Formação de utilizadores finais; liderou uma equipa de 5 pessoas",
    "Gestão de projetos; concluiu o ensino secundário em 2010",
    "Pós-graduação em Gestão; licenciatura em Enfermagem",
    "Registo de dados na base de dados; contacto com clientes; secção de contabilidade",
    "Prémio de mérito académico; experiência em UCI; está a trabalhar num hospital",
    "Utilização de MS Excel e SQL",
])
def test_european_portuguese_is_not_flagged(text):
    assert ptbr_terms(text) == []


def test_a_pt_template_with_a_ptbr_term_is_rejected():
    template = PT_HEADER + "Liderou uma equipe de enfermagem."

    with pytest.raises(TemplateError, match="PT-BR.*equipe.*equipa"):
        render_variant(template, name="X", nationality="X", gender="female", language="pt")


def test_the_ptbr_check_only_applies_to_pt_templates():
    en = "$name\nNationality: $nationality\n$work_eligibility\n\nWorked with the equipe tool."

    assert "equipe" in render_variant(en, name="X", nationality="X", gender="male", language="en")
