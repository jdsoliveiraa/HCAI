"""Is a generated document written in the language it should be? A count of function words, not a classifier.

The corpus generator once wrote "English" job ads and base CVs in European Portuguese, and the PT-PT
"translation" then translated Portuguese into Portuguese. This check catches that, and the inverse (a
translation left in English), with a rule a reader can verify by hand:

- Over the prose fields of a document, count the tokens that are common function words of English (`the`,
  `and`, `of`, `with`, ...) and of Portuguese (`de`, `em`, `com`, `para`, `uma`, `não`, ...). Words that are
  common in both languages or are frequent acronyms (`a`, `as`, `no`, `do`, `os`, `se`) are in neither list,
  and an all-caps token (an acronym such as `NA`) is never counted.
- A function word between two capitalised words is part of a proper noun ("Ordem dos Contabilistas
  Certificados", "Universidade de Lisboa", "University of Coimbra") and is not counted, so Portuguese
  employer, institution, council and degree names may stay in an English document (and English ones in a
  translation).
- A field is in the other language when it has at least `MIN_HITS` of the other language's function words
  and more of them than of the expected language's. The document is in the other language when any field is,
  or when its counts over all the fields are.

Short fields with no function word at all (a skill, "Contabilista Certificado") decide nothing on their own;
the document-wide count covers them.
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
LANGUAGE_NAMES = {"en": "English", "pt": "Portuguese"}
MIN_HITS = 3  # a field needs at least this many of the other language's function words to count against it
# English abbreviations whose letters would otherwise read as Portuguese 'e' ("e.g.", "i.e.").
_ABBREVIATIONS = re.compile(r"\b(?:e\.g|i\.e)\.?", re.IGNORECASE)
_WORD = re.compile(r"\w+")


@dataclass(frozen=True)
class FieldCount:
    path: str  # e.g. "experience[2].bullets[1]"
    text: str
    hits: dict[str, int]  # language -> function words counted


def count(text: str) -> dict[str, int]:
    """The English and Portuguese function words in `text`, not counting those inside a proper noun."""
    tokens = _WORD.findall(_ABBREVIATIONS.sub(" ", text))
    hits = dict.fromkeys(FUNCTION_WORDS, 0)
    for i, token in enumerate(tokens):
        if token.isupper():  # an acronym, or a capital at the start of a sentence that is also a single letter
            continue
        if 0 < i < len(tokens) - 1 and tokens[i - 1][0].isupper() and tokens[i + 1][0].isupper():
            continue  # "Ordem dos Contabilistas", "University of Lisbon": part of a name
        word = token.casefold()
        for language, words in FUNCTION_WORDS.items():
            if word in words:
                hits[language] += 1
    return hits


def language_problem(fields: Iterable[tuple[str, str]], expected: str) -> str:
    """Why the document with these (path, text) fields is not in `expected` ("en" or "pt"); "" if it is."""
    other = next(lang for lang in FUNCTION_WORDS if lang != expected)
    counts = [FieldCount(path, text, count(text)) for path, text in fields]
    total = {lang: sum(c.hits[lang] for c in counts) for lang in FUNCTION_WORDS}

    def foreign(hits: dict[str, int]) -> bool:
        return hits[other] >= MIN_HITS and hits[other] > hits[expected]

    if not (foreign(total) or any(foreign(c.hits) for c in counts)):
        return ""
    offending = sorted((c for c in counts if c.hits[other] > c.hits[expected]),
                       key=lambda c: -c.hits[other])
    shown = []
    for c in offending[:6]:
        snippet = c.text if len(c.text) <= 60 else c.text[:60] + "…"
        shown.append(f"{c.path} (\"{snippet}\")")
    more = f" and {len(offending) - 6} more" if len(offending) > 6 else ""
    return (f"the document must be written in {LANGUAGE_NAMES[expected]}, but it reads as "
            f"{LANGUAGE_NAMES[other]} ({total[other]} {LANGUAGE_NAMES[other]} function words vs {total[expected]} "
            f"{LANGUAGE_NAMES[expected]}); {LANGUAGE_NAMES[other]} text in: {', '.join(shown)}{more}")
