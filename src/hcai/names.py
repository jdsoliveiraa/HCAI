"""Name bank: full names per gender x origin, with the public sources they were drawn from."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

GENDERS = ("male", "female")


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
    raw = tomllib.loads(Path(path).read_text())
    return NameBank(origins=[
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
