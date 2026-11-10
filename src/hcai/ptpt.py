"""European-Portuguese (PT-PT) checks: flag common Brazilian-Portuguese (PT-BR) terms in a PT-PT text.

The core design compares EN with PT-PT; Brazilian dialect in the CV body would add a cue the identities do
not share (spec #1: dialect bias is out of scope), so PT-PT base CVs and job ads must avoid PT-BR vocabulary,
spelling and grammar. The lists below are the common, unambiguous cases for CVs and job ads in the three core
roles, not a full dialect classifier; a term that is also normal PT-PT (e.g. "pesquisa", "arquivo") is left out.
"""

from __future__ import annotations

import re

# PT-BR word or phrase -> its PT-PT equivalent. Matched case-insensitively, as whole words or phrases;
# a hyphenated compound is not split ("pós-graduação" is PT-PT, "graduação" alone is PT-BR).
PTBR_TERMS: dict[str, str] = {
    # vocabulary
    "usuário": "utilizador", "usuária": "utilizadora", "usuários": "utilizadores", "usuárias": "utilizadoras",
    "equipe": "equipa", "equipes": "equipas",
    "gerenciamento": "gestão", "gerenciar": "gerir", "gerenciou": "geriu", "gerenciando": "a gerir",
    "gerenciei": "geri", "gerenciava": "geria",
    "ensino médio": "ensino secundário", "colegial": "ensino secundário",
    "graduação": "licenciatura", "bacharelado": "licenciatura", "bacharel": "licenciado/a",
    "celular": "telemóvel", "tela": "ecrã", "telas": "ecrãs",
    "planilha": "folha de cálculo", "planilhas": "folhas de cálculo",
    "treinamento": "formação", "treinamentos": "formações",
    "aplicativo": "aplicação", "aplicativos": "aplicações",
    "banco de dados": "base de dados", "bancos de dados": "bases de dados",
    "desenvolvedor": "programador", "desenvolvedora": "programadora",
    "desenvolvedores": "programadores", "desenvolvedoras": "programadoras",
    "mídia": "media", "mídias": "media",
    "cadastro": "registo", "cadastros": "registos", "cadastrar": "registar",
    "deletar": "apagar",
    "faturamento": "faturação",
    "contábil": "contabilístico", "contábeis": "contabilísticos",
    "contador": "contabilista", "contadora": "contabilista",
    "imposto de renda": "IRS", "nota fiscal": "fatura", "notas fiscais": "faturas",
    "pronto-socorro": "urgência", "plantão": "turno", "plantões": "turnos",
    "carteira de motorista": "carta de condução", "carteira de trabalho": "contrato de trabalho",
    "ônibus": "autocarro", "trem": "comboio", "banheiro": "casa de banho", "esporte": "desporto",
    "esportes": "desportos", "café da manhã": "pequeno-almoço",
    # spelling (PT-BR ê/ô where PT-PT writes é/ó; consonants PT-PT keeps)
    "registro": "registo", "registros": "registos",
    "contato": "contacto", "contatos": "contactos",
    "seção": "secção", "seções": "secções",
    "econômico": "económico", "econômica": "económica", "econômicos": "económicos", "econômicas": "económicas",
    "eletrônico": "eletrónico", "eletrônica": "eletrónica", "eletrônicos": "eletrónicos",
    "acadêmico": "académico", "acadêmica": "académica", "acadêmicos": "académicos",
    "prêmio": "prémio", "prêmios": "prémios", "gênero": "género", "fenômeno": "fenómeno",
    "anônimo": "anónimo", "bônus": "bónus", "tênis": "ténis", "Antônio": "António",
    "agronômico": "agronómico", "ergonômico": "ergonómico", "sistêmico": "sistémico",
    "polêmico": "polémico", "anatômico": "anatómico", "cômodo": "cómodo",
}

# PT-BR acronyms, matched case-sensitively.
PTBR_ACRONYMS: dict[str, str] = {
    "UTI": "UCI (unidade de cuidados intensivos)", "CPF": "NIF", "CNPJ": "NIPC", "CLT": "Código do Trabalho",
}

# The PT-BR progressive "estar + gerúndio" ("estou trabalhando"); PT-PT says "estar a + infinitivo".
_PROGRESSIVE = re.compile(
    r"(?<![\w-])(?:estou|está|estás|estamos|estão|estava|estavas|estávamos|estavam|estive|esteve|estiveram)"
    r"\s+\w+(?:ando|endo|indo)(?![\w-])",
    re.IGNORECASE,
)

_TERMS = re.compile(
    r"(?<![\w-])(?:" + "|".join(re.escape(t) for t in sorted(PTBR_TERMS, key=len, reverse=True)) + r")(?![\w-])",
    re.IGNORECASE,
)
_ACRONYMS = re.compile(r"(?<![\w-])(?:" + "|".join(PTBR_ACRONYMS) + r")(?![\w-])")


def ptbr_terms(text: str) -> list[str]:
    """The PT-BR terms in a PT-PT text, as written there, in order of first appearance (empty: none found)."""
    found = [m.group(0) for pattern in (_TERMS, _ACRONYMS, _PROGRESSIVE) for m in pattern.finditer(text)]
    return list(dict.fromkeys(sorted(found, key=text.find)))


def describe(terms: list[str]) -> str:
    """'"usuário" (PT-PT: utilizador), ...' for an error message."""
    lookup = {k.casefold(): v for k, v in PTBR_TERMS.items()} | PTBR_ACRONYMS
    def hint(term: str) -> str:
        pt = lookup.get(term.casefold(), lookup.get(term))
        return f'"{term}" (PT-PT: {pt})' if pt else f'"{term}" (PT-PT: estar a + infinitivo)'
    return ", ".join(hint(t) for t in terms)
