"""Name bank: full names per gender x origin, with the public sources they were drawn from."""

from __future__ import annotations

import tomllib
import unicodedata
from dataclasses import dataclass
from pathlib import Path

GENDERS = ("male", "female")


class NameBankError(ValueError):
    """The name bank breaks a rule of the bank (e.g. a repeated first name or surname)."""


@dataclass(frozen=True)
class Source:
    title: str
    url: str


@dataclass(frozen=True)
class Origin:
    key: str  # e.g. "portuguese"
    code: str  # short code used in identity ids, e.g. "pt"
    country: str
    nationality: dict[str, str]  # language -> value written after the nationality label
    names: dict[str, list[str]]  # gender -> full names
    sources: list[Source]


@dataclass(frozen=True)
class NameBank:
    origins: list[Origin]


def load_name_bank(path: str | Path) -> NameBank:
    """Load the bank; a bank that repeats a first name or a surname anywhere is rejected (`check_unique_parts`)."""
    raw = tomllib.loads(Path(path).read_text())
    bank = NameBank(origins=[
        Origin(
            key=key,
            code=o["code"],
            country=o["country"],
            nationality=dict(o["nationality"]),
            names={g: list(o[g]) for g in GENDERS},
            sources=[Source(**s) for s in o["sources"]],
        )
        for key, o in raw["origins"].items()
    ])
    check_unique_parts(bank, path)
    return bank


def name_parts(full_name: str) -> tuple[str, str]:
    """(first name, surname) of a full name: its first and its last word ("Ana Clara Souza" -> Ana, Souza)."""
    words = full_name.split()
    if len(words) < 2:
        raise NameBankError(f"name {full_name!r} needs a first name and a surname")
    return words[0], words[-1]


def check_unique_parts(bank: NameBank, path: str | Path = "name bank") -> None:
    """Every name of the bank has its own first name and its own surname, across every gender and origin
    (compared ignoring case, in Unicode NFC so a pasted decomposed accent is the same letter), so no two identities
    share a name cue. Raises NameBankError naming the duplicates."""
    first: dict[str, list[str]] = {}
    last: dict[str, list[str]] = {}
    for origin in bank.origins:
        for gender in GENDERS:
            for name in origin.names[gender]:
                f, s = name_parts(name)
                first.setdefault(_key(f), []).append(name)
                last.setdefault(_key(s), []).append(name)
    problems = [
        f"{what} {name_parts(names[0])[i]!r} in {', '.join(names)}"
        for what, i, seen in (("surname", 1, last), ("first name", 0, first))
        for names in seen.values() if len(names) > 1
    ]
    if problems:
        raise NameBankError(f"{path}: every name needs its own first name and surname across the bank; repeated: "
                         + "; ".join(problems))


def _key(word: str) -> str:
    return unicodedata.normalize("NFC", word).casefold()
