"""Design config: the experimental factors and their levels."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, replace
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
class RunSettings:
    """How the runner calls the LLMs (the `[run]` config section); none of this changes the design."""

    # A malformed or failed agent call is retried this many times, then its row is recorded as missing.
    retries: int = 2
    # Seconds to wait before retrying a call that raised (e.g. an HTTP error), doubled on each retry.
    retry_backoff: float = 2.0
    # Stability sample: this fraction of the CVs and Ranker pools is re-run `repeats` times.
    repeat_fraction: float = 0.0
    repeats: int = 1


@dataclass(frozen=True)
class PilotSlice:
    """The pilot slice (the `[pilot]` config section): how many of each factor's levels a pilot keeps.

    Each count keeps the first N levels as listed in the design; `None` keeps them all. The default is the
    first role and one base CV per quality level, under every identity, language, condition and model, so
    the pilot exercises every model and condition and its cost can be scaled up to the full run.
    """

    roles: int | None = 1
    quality_levels: int | None = None
    base_cvs_per_role: int | None = 1
    identities: int | None = None
    languages: int | None = None
    conditions: int | None = None
    models: int | None = None
    mixed_quality: int | None = 1


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
    run: RunSettings = field(default_factory=RunSettings)
    pilot: PilotSlice = field(default_factory=PilotSlice)
    pilot_of: Design | None = None  # set on a pilot design only: the full design it is a slice of

    @property
    def size(self) -> int:
        """CV variants x language x condition x model: what the Screener and Evaluator calls scale with."""
        return (len(self.roles) * len(self.quality_levels) * self.base_cvs_per_role * len(self.identities)
                * len(self.languages) * len(self.conditions) * len(self.models))

    @property
    def ranked_slots(self) -> int:
        """CVs shown to the Ranker over all pools (if all are screened in): what Ranker tokens scale with."""
        bases = len(self.quality_levels) * self.base_cvs_per_role
        per_role = ((bases * len(self.identities) if self.pools.equal_quality else 0)
                    + self.pools.mixed_quality * min(self.pools.mixed_size, bases))
        return per_role * len(self.roles) * len(self.languages) * len(self.conditions) * len(self.models)


def pilot_design(design: Design) -> Design:
    """The design restricted to its `[pilot]` slice; the config itself is left as it is."""
    p = design.pilot

    def first[T](levels: list[T], n: int | None) -> list[T]:
        return list(levels if n is None else levels[:n])

    pilot = replace(
        design,
        roles=first(design.roles, p.roles),
        quality_levels=first(design.quality_levels, p.quality_levels),
        base_cvs_per_role=design.base_cvs_per_role if p.base_cvs_per_role is None
        else min(design.base_cvs_per_role, p.base_cvs_per_role),
        identities=first(design.identities, p.identities),
        languages=first(design.languages, p.languages),
        conditions=first(design.conditions, p.conditions),
        models=first(design.models, p.models),
        pools=design.pools if p.mixed_quality is None
        else replace(design.pools, mixed_quality=min(design.pools.mixed_quality, p.mixed_quality)),
    )
    return replace(pilot, pilot_of=design)


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
        run=RunSettings(**raw.get("run", {})),
        pilot=PilotSlice(**raw.get("pilot", {})),
    )
