"""Design config: the experimental factors and their levels."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from hcai.names import GENDERS, NameBank, load_name_bank


@dataclass(frozen=True)
class Identity:
    id: str
    gender: str
    origin: str
    names: list[str]
    nationality: str
    # Nationality per language (e.g. {"en": "Brazilian", "pt": "Brasileira"}); falls back to `nationality`.
    nationalities: dict[str, str] = field(default_factory=dict)

    def nationality_in(self, language: str) -> str:
        return self.nationalities.get(language, self.nationality)


@dataclass(frozen=True)
class Design:
    roles: list[str]
    quality_levels: list[str]
    base_cvs_per_role: int
    identities: list[Identity]
    languages: list[str]
    conditions: list[str]
    models: list[str]
    corpus_dir: Path
    temperature: float = 0.0
    seed: int | None = 42


def load_design(path: str | Path) -> Design:
    """Load a design from TOML. `corpus_dir` and `name_bank` are resolved relative to the config file.

    With `name_bank` set, the identities are every gender x origin of the bank; otherwise they come
    from the `[[identities]]` tables.
    """
    path = Path(path)
    raw = tomllib.loads(path.read_text())
    d = raw["design"]
    return Design(
        roles=list(d["roles"]),
        quality_levels=list(d["quality_levels"]),
        base_cvs_per_role=int(d["base_cvs_per_role"]),
        identities=(
            identities_from_name_bank(load_name_bank(path.parent / d["name_bank"]))
            if "name_bank" in d else [Identity(**i) for i in raw["identities"]]
        ),
        languages=list(d["languages"]),
        conditions=list(d["conditions"]),
        models=list(d["models"]),
        corpus_dir=(path.parent / d["corpus_dir"]).resolve(),
        temperature=float(d.get("temperature", 0.0)),
        seed=d.get("seed", 42),
    )


def identities_from_name_bank(bank: NameBank) -> list[Identity]:
    """One identity per origin x gender, origin-major, male first (so the first is <first origin>_m)."""
    return [
        Identity(
            id=f"{origin.code}_{gender[0]}", gender=gender, origin=origin.key, names=origin.names[gender],
            nationality=origin.nationality["en"], nationalities=dict(origin.nationality),
        )
        for origin in bank.origins
        for gender in GENDERS
    ]
