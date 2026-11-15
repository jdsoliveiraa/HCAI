"""Is a generated document written in the language it should be? A count of marker words, not a classifier.

The corpus generator once wrote "English" job ads and base CVs in European Portuguese, and the PT-PT
"translation" then translated Portuguese into Portuguese. This check catches that, and the inverse (a
translation left in English), with rules a reader can verify by hand:

- Over the fields of a document, count the tokens that are common function words of English (`the`, `and`,
  `of`, `with`, ...) and of Portuguese (`de`, `em`, `com`, `para`, `uma`, `não`, ...). A lowercase word
  holding a letter English never uses (`ã`, `õ`, `ç`: "reconciliações", "gestão") also counts as Portuguese.
  Words common to both languages or frequent acronyms (`a`, `as`, `no`, `do`, `os`, `se`) are in neither
  list, an all-caps token (`NA`, `IVA`) is never counted, and the `e-` of "e-commerce" or "e-Fatura" and the
  "e.g." abbreviation are not the Portuguese "e".
- A function word between two capitalised words is part of a proper noun ("Ordem dos Contabilistas
  Certificados", "Universidade de Lisboa", "University of Coimbra") and is not counted, so Portuguese
  employer, institution, council and degree names may stay in an English document (and English ones in a
  translation).
- A `prose` field (summary, bullets, requirements) is in the other language when it has at least `MIN_HITS`
  of the other language's words and more of them than of the expected language's. A `name` field (skills,
  certifications, degrees: often proper names) only counts toward the document's totals. The document is in
  the other language when any prose field is, or its totals over all fields are.
- An English `title` field (a headline or job title) is Portuguese when it holds a Portuguese job-title word
  (`contabilista`, `enfermeiro`, `técnico`, ...) and no English one (`accountant`, `nurse`, ...), so
  "Contabilista Certificado" fails while "Contabilista Certificado (Certified Accountant)" passes.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

FUNCTION_WORDS: dict[str, frozenset[str]] = {
    "en": frozenset({
        "the", "and", "of", "to", "in", "with", "for", "on", "at", "by", "from", "is", "are", "was", "were", "be",
        "an", "this", "that", "which", "or", "our", "your", "you", "we", "will", "have", "has", "its", "into",
        "within", "across", "using", "including",
    }),
    "pt": frozenset({
        "de", "da", "das", "dos", "em", "na", "nas", "nos", "num", "numa", "com", "para", "por", "pela", "pelo",
        "pelas", "pelos", "uma", "um", "que", "ao", "aos", "à", "às", "é", "e", "ou", "não", "sua", "seu", "suas",
        "seus", "são", "está", "estão", "também", "mais", "entre", "sobre", "através",
    }),
}
# Letters of Portuguese that English words never use: a lowercase word holding one counts as Portuguese.
PORTUGUESE_LETTERS = re.compile(r"[ãõç]")
# Job-title words, for the title rule (stems: "enfermeir" matches enfermeiro/enfermeira/enfermeiros).
TITLE_WORDS: dict[str, tuple[str, ...]] = {
    "pt": ("contabilist", "contabilidade", "técnic", "tecnic", "enfermeir", "enfermagem", "engenheir", "engenharia",
           "programador", "desenvolvedor", "gestor", "responsável", "licenciad", "fiscalidade",
           "diretor", "chefe", "estagiári", "assistente", "especialista", "analista", "consultor"),
    "en": ("accountant", "accounting", "bookkeep", "nurse", "nursing", "engineer", "developer", "programmer",
           "technician", "auditor", "analyst", "manager", "specialist", "officer", "assistant", "consultant",
           "controller", "director", "lead", "head", "intern", "architect", "administrator", "senior", "junior",
           "professional", "graduate", "certified", "registered", "chartered", "staff", "software", "tax"),
}
LANGUAGE_NAMES = {"en": "English", "pt": "Portuguese"}
MIN_HITS = 3  # a prose field needs at least this many of the other language's words to count against it
KINDS = ("prose", "title", "name")
# "e.g." / "i.e." (also spaced), and the e- prefix of "e-commerce", "e-Fatura": not the Portuguese 'e'.
_NOT_E = re.compile(r"\b(?:e\.\s?g|i\.\s?e)\.?|\be-(?=\w)", re.IGNORECASE)
_WORD = re.compile(r"\w+")


@dataclass(frozen=True)
class Field:
    path: str  # e.g. "experience[2].bullets[1]"
    text: str
    kind: str = "prose"  # one of KINDS


def count(text: str) -> dict[str, int]:
    """The English and Portuguese marker words in `text`, not counting those inside a proper noun."""
    tokens = _WORD.findall(_NOT_E.sub(" ", text))
    hits = dict.fromkeys(FUNCTION_WORDS, 0)
    for i, token in enumerate(tokens):
        if token.isupper():  # an acronym, or a single capital letter
            continue
        if 0 < i < len(tokens) - 1 and tokens[i - 1][0].isupper() and tokens[i + 1][0].isupper():
            continue  # "Ordem dos Contabilistas", "University of Lisbon": part of a name
        word = token.casefold()
        for language, words in FUNCTION_WORDS.items():
            if word in words:
                hits[language] += 1
        if word not in FUNCTION_WORDS["pt"] and token[0].islower() and PORTUGUESE_LETTERS.search(word):
            hits["pt"] += 1
    return hits


def _has_title_word(text: str, language: str) -> bool:
    words = _WORD.findall(text.casefold())
    return any(w.startswith(stem) for w in words for stem in TITLE_WORDS[language])


def language_problem(fields: Iterable[Field], expected: str) -> str:
    """Why the document with these fields is not in `expected` ("en" or "pt"); "" if it is."""
    other = next(lang for lang in FUNCTION_WORDS if lang != expected)
    fields = list(fields)
    counts = [(f, count(f.text)) for f in fields]
    total = {lang: sum(hits[lang] for _, hits in counts) for lang in FUNCTION_WORDS}

    def foreign(hits: dict[str, int]) -> bool:
        return hits[other] >= MIN_HITS and hits[other] > hits[expected]

    offending = [f for f, hits in counts if f.kind == "prose" and foreign(hits)]
    if expected == "en":
        offending += [f for f in fields if f.kind == "title" and _has_title_word(f.text, "pt")
                      and not _has_title_word(f.text, "en")]
    if not offending and not foreign(total):
        return ""
    named = list(dict.fromkeys(offending + [f for f, hits in counts if hits[other] > hits[expected]]))
    shown = []
    for f in named[:6]:
        snippet = f.text if len(f.text) <= 60 else f.text[:60] + "…"
        shown.append(f"{f.path} (\"{snippet}\")")
    more = f" and {len(named) - 6} more" if len(named) > 6 else ""
    return (f"the document must be written in {LANGUAGE_NAMES[expected]}, but parts read as "
            f"{LANGUAGE_NAMES[other]} ({total[other]} {LANGUAGE_NAMES[other]} marker words vs {total[expected]} "
            f"{LANGUAGE_NAMES[expected]}); {LANGUAGE_NAMES[other]} text in: {', '.join(shown)}{more}")
