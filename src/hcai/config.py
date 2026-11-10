"""Design config: the experimental factors and their levels."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Identity:
    id: str
    gender: str
    origin: str
    names: list[str]
    nationality: str


@dataclass(frozen=True)
class Pools:
    """Ranker pool composition.

    Equal-quality pools hold every identity of one base CV (one pool per base CV and language).
    Mixed-quality pools hold `mixed_size` distinct base CVs spread over the quality levels, each with
    one identity; `mixed_quality` is how many such pools to build per role and language.
    """

    equal_quality: bool = True
    mixed_quality: int = 0
    mixed_size: int = 8


@dataclass(frozen=True)
class Generator:
    """The CV/job-ad generator: a third model family, configured apart from the audited pipeline models."""

    model: str
    temperature: float = 0.7
    # Borderline base CVs per role, by quality level: they meet every mandatory requirement, but only just.
    borderline: dict[str, int] = field(default_factory=dict)


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
    pools: Pools = field(default_factory=Pools)
    generator: Generator | None = None


def load_design(path: str | Path) -> Design:
    """Load a design from TOML. `corpus_dir` is resolved relative to the config file."""
    path = Path(path)
    raw = tomllib.loads(path.read_text())
    d = raw["design"]
    return Design(
        roles=list(d["roles"]),
        quality_levels=list(d["quality_levels"]),
        base_cvs_per_role=int(d["base_cvs_per_role"]),
        identities=[Identity(**i) for i in raw.get("identities", [])],
        languages=list(d["languages"]),
        conditions=list(d["conditions"]),
        models=list(d["models"]),
        corpus_dir=(path.parent / d["corpus_dir"]).resolve(),
        temperature=float(d.get("temperature", 0.0)),
        seed=d.get("seed", 42),
        pools=Pools(**raw.get("pools", {})),
        generator=Generator(**raw["generator"]) if "generator" in raw else None,
    )
