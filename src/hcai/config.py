"""Design config: the experimental factors and their levels."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

from hcai.llm import DEFAULT_REQUEST_TIMEOUT, http_timeout
from hcai.names import GENDERS, NameBank, load_name_bank
from hcai.providers import BUILTIN_PROVIDERS, DEFAULT_PROVIDER, Provider


@dataclass(frozen=True)
class Identity:
    id: str
    gender: str
    origin: str
    names: list[str]
    nationality: str
    # Nationality per language (e.g. {"en": "Brazilian", "pt": "Brasileira"}); EN falls back to `nationality`.
    nationalities: dict[str, str] = field(default_factory=dict)

    def nationality_in(self, language: str) -> str:
        """The nationality in `language`; `nationality` is the English one, so other languages must be listed."""
        if language in self.nationalities:
            return self.nationalities[language]
        if language == "en":
            return self.nationality
        raise ValueError(f"identity {self.id} has no nationality in {language!r}; add it to `nationalities`")


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
    # New generator calls per document and run before it counts as failed (`[generator] max_attempts`).
    # Answers already cached from earlier runs replay for free and do not use this budget (ADR 0006).
    max_attempts: int = 4

    def __post_init__(self) -> None:
        if not isinstance(self.max_attempts, int) or isinstance(self.max_attempts, bool) or self.max_attempts < 1:
            raise ValueError(f"[generator] max_attempts must be a whole number >= 1, not {self.max_attempts!r}")


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
    # A rate- or usage-limit answer (HTTP 429) is waited out: the provider's Retry-After, else this many
    # seconds doubling up to `rate_limit_max_backoff`. A call that would wait more than `rate_limit_max_wait`
    # seconds in all stops the run cleanly (rerun to resume from the cache). OpenCode Go's budget is over a
    # rolling 5 h window, so the default waits up to 6 h.
    rate_limit_backoff: float = 30.0
    rate_limit_max_backoff: float = 900.0
    rate_limit_max_wait: float = 6 * 3600.0
    # Seconds every provider call waits for the answer (the HTTP read timeout, see `hcai.llm.http_timeout`):
    # above 0, at most a day. A call that times out fails like any failed call (in a run: retried, then its
    # rows are missing). Not part of a request: changing it changes no cache key. The default leaves room for
    # the longest answers of the pilot (the Blinder's).
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT
    # Billing sources a call may be paid from (OpenCode Go's `usage.opencode.billing_source`). The first fresh
    # answer billed elsewhere (e.g. "balance", the pay-as-you-go balance once Extra Usage kicks in) is cached,
    # then the run stops cleanly (exit 3). Answers that name no billing source (e.g. OpenRouter) are never
    # checked; an empty list checks nothing.
    allowed_billing: tuple[str, ...] = ("go",)
    # Failed calls in a row on one model (a call counts once however often it is retried; any provider answer resets
    # it) that stop the run cleanly (exit 3, rerun to resume) instead of recording row after row as missing. 0: never.
    max_consecutive_failures: int = 10
    # Seconds before each repeat of a call that failed transiently (HTTP 5xx, timeout, dropped connection), in
    # place of `retry_backoff`; such a call is repeated up to max(retries, len(transient_backoff)) times.
    transient_backoff: tuple[float, ...] = (30.0, 120.0, 480.0)
    # Share of missing rows above which the run warns (stderr and `missing.above_target` in run_summary.json).
    missing_target: float = 0.02
    # Seconds between heartbeat lines in the run log (rows done, rate, ETA), each also rewriting progress.json.
    heartbeat_s: float = 300.0
    # Agent calls in flight at once (ADR 0005, concurrency amendment): within each role x language x condition x
    # model block, the CV variants (Blinder -> Screener -> Evaluator -> judge) and the Ranker pools (once their CVs
    # are screened) run on up to this many workers. 1 runs everything one after the other. Requests, cache keys and
    # results.csv are the same whatever the value.
    concurrency: int = 1

    def __post_init__(self) -> None:
        try:
            http_timeout(self.request_timeout)
        except ValueError as e:
            raise ValueError(f"[run] {e}") from None
        if isinstance(self.allowed_billing, str) or not all(isinstance(b, str) for b in self.allowed_billing):
            raise ValueError(f"[run] allowed_billing must be a list of billing sources, not {self.allowed_billing!r}")
        object.__setattr__(self, "allowed_billing", tuple(self.allowed_billing))
        n = self.max_consecutive_failures
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError(f"[run] max_consecutive_failures must be a whole number >= 0, not {n!r}")
        b = self.transient_backoff
        if (isinstance(b, (str, bytes)) or not all(
                isinstance(x, (int, float)) and not isinstance(x, bool) and x >= 0 for x in b)):
            raise ValueError(f"[run] transient_backoff must be a list of seconds >= 0, not {b!r}")
        object.__setattr__(self, "transient_backoff", tuple(float(x) for x in b))
        t = self.missing_target
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not 0 <= t <= 1:
            raise ValueError(f"[run] missing_target must be a share between 0 and 1, not {t!r}")
        c = self.concurrency
        if isinstance(c, bool) or not isinstance(c, int) or not 1 <= c <= 64:
            raise ValueError(f"[run] concurrency must be a whole number from 1 to 64, not {c!r}")
        h = self.heartbeat_s
        if isinstance(h, bool) or not isinstance(h, (int, float)) or not 0 < h <= 86400:
            raise ValueError(f"[run] heartbeat_s must be a number of seconds > 0 and at most 86400, not {h!r}")


@dataclass(frozen=True)
class Oversight:
    """The oversight layer of the rubric conditions (C2, C3): the LLM judge and the escalation rule.

    An Evaluator decision is escalated to the human review queue when the judge flags one of its quotes
    (or gives no valid verdict), or when its score is within `margin` points of the shortlist `cutoff`.
    `cutoff` is a fixed score, or "median": the median Evaluator score of the same role x language x
    condition x model (the analysis's primary selection rule). `judge_model` defaults to the generator's
    model, the third model family.
    """

    judge_model: str | None = None
    cutoff: float | str = "median"
    margin: float = 5.0

    def __post_init__(self) -> None:
        if isinstance(self.cutoff, bool) or (isinstance(self.cutoff, str) and self.cutoff != "median"):
            raise ValueError(f'[oversight] cutoff must be a score or "median", not {self.cutoff!r}')
        if isinstance(self.margin, bool) or self.margin < 0:
            raise ValueError(f"[oversight] margin must be a non-negative number of points, not {self.margin!r}")


@dataclass(frozen=True)
class PilotSlice:
    """The pilot slice (the `[pilot]` config section): how many of each factor's levels a pilot keeps.

    Each count keeps the first N levels as listed in the design (N identities are spread over origins and
    genders, see `spread_identities`); `None` keeps them all. The default is the first role and one base
    CV per quality level, under every identity, language, condition and model, so the pilot exercises
    every model and condition and its cost can be scaled up to the full run.
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
    oversight: Oversight = field(default_factory=Oversight)
    pilot_of: Design | None = None  # set on a pilot design only: the full design it is a slice of
    # LLM providers by name (the built-in `opencode-go` and `openrouter`, plus `[providers.<name>]`), and the
    # provider of each model (`[model_providers]`); a model not listed there uses `opencode-go`.
    providers: dict[str, Provider] = field(default_factory=lambda: dict(BUILTIN_PROVIDERS))
    model_providers: dict[str, str] = field(default_factory=dict)

    def provider_for(self, model: str) -> Provider:
        """The provider that serves `model`: its `[model_providers]` entry, else `opencode-go`."""
        return self.providers[self.model_providers.get(model, DEFAULT_PROVIDER)]

    @property
    def all_models(self) -> list[str]:
        """Every model the config names: the pipeline models, the generator and the judge."""
        extra = [self.generator.model] if self.generator else []
        extra += [self.judge_model] if self.judge_model else []
        return list(dict.fromkeys(self.models + extra))

    @property
    def judge_model(self) -> str | None:
        """The oversight judge's model: `[oversight] judge_model`, else the generator's (third-family) model."""
        if self.oversight.judge_model:
            return self.oversight.judge_model
        return self.generator.model if self.generator else None

    @property
    def size(self) -> int:
        """CV variants x language x condition x model: what the Screener and Evaluator calls scale with.

        Each base CV x identity gets one name from the identity's names, so the names don't multiply it.
        """
        return (len(self.roles) * len(self.quality_levels) * self.base_cvs_per_role * len(self.identities)
                * len(self.languages) * len(self.conditions) * len(self.models))

    @property
    def ranked_slots(self) -> int:
        """CVs shown to the Ranker over all pools (if all are screened in): what Ranker tokens scale with."""
        bases = len(self.quality_levels) * self.base_cvs_per_role
        per_role = ((bases * len(self.identities) if self.pools.equal_quality else 0)
                    + self.pools.mixed_quality * min(self.pools.mixed_size, bases))
        return per_role * len(self.roles) * len(self.languages) * len(self.conditions) * len(self.models)


def spread_identities(identities: list[Identity]) -> list[Identity]:
    """The identities reordered so that the first N cover as many origins, then both genders, as they can.

    Rounds over the origins (in order of first appearance), each round shifting which of an origin's
    identities is taken: with 4 origins x 2 genders, pt_m, br_f, cv_m, ua_f, pt_f, br_m, ...
    """
    by_origin: dict[str, list[Identity]] = {}
    for identity in identities:
        by_origin.setdefault(identity.origin, []).append(identity)
    groups = list(by_origin.values())
    out: list[Identity] = []
    for k in range(max((len(g) for g in groups), default=0)):
        for j, group in enumerate(groups):
            left = [i for i in group if i not in out]
            if left:
                preferred = group[(j + k) % len(group)]
                out.append(preferred if preferred in left else left[0])
    return out


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
        identities=design.identities if p.identities is None else spread_identities(design.identities)[: p.identities],
        languages=first(design.languages, p.languages),
        conditions=first(design.conditions, p.conditions),
        models=first(design.models, p.models),
        pools=design.pools if p.mixed_quality is None
        else replace(design.pools, mixed_quality=min(design.pools.mixed_quality, p.mixed_quality)),
    )
    return replace(pilot, pilot_of=design)


def load_design(path: str | Path, base: str | Path | None = None) -> Design:
    """Load a design from TOML. `corpus_dir` and `name_bank` are resolved relative to the config file,
    or to `base` when given (a copy of a config kept elsewhere, e.g. the frozen `data/runs/<run>/core.toml`,
    resolved as if it were in `configs/`).

    With `name_bank` set, the identities are every gender x origin of the bank; otherwise they come
    from the `[[identities]]` tables (setting both is an error).
    """
    path = Path(path)
    raw = tomllib.loads(path.read_text())
    root = Path(base) if base is not None else path.parent
    d = raw["design"]
    if "name_bank" in d and "identities" in raw:
        raise ValueError(f"{path}: set either [design] name_bank or [[identities]], not both")
    design = Design(
        roles=list(d["roles"]),
        quality_levels=list(d["quality_levels"]),
        base_cvs_per_role=int(d["base_cvs_per_role"]),
        identities=(
            identities_from_name_bank(load_name_bank(root / d["name_bank"]))
            if "name_bank" in d else [Identity(**i) for i in raw.get("identities", [])]
        ),
        languages=list(d["languages"]),
        conditions=list(d["conditions"]),
        models=list(d["models"]),
        corpus_dir=(root / d["corpus_dir"]).resolve(),
        temperature=float(d.get("temperature", 0.0)),
        seed=d.get("seed", 42),
        pools=Pools(**raw.get("pools", {})),
        generator=Generator(**raw["generator"]) if "generator" in raw else None,
        run=RunSettings(**raw.get("run", {})),
        pilot=PilotSlice(**raw.get("pilot", {})),
        oversight=Oversight(**raw.get("oversight", {})),
        providers=_providers(raw.get("providers", {}), path),
        model_providers={str(k): str(v) for k, v in raw.get("model_providers", {}).items()},
    )
    _check_model_providers(design, path)
    return design


def _providers(raw: dict, path: Path) -> dict[str, Provider]:
    """The built-in providers with the config's `[providers.<name>]` overrides and additions."""
    providers = dict(BUILTIN_PROVIDERS)
    for name, settings in raw.items():
        if not isinstance(settings, dict):
            raise ValueError(f"{path}: [providers.{name}] must be a table (base_url, key_env, billing)")
        unknown = set(settings) - {"base_url", "key_env", "billing", "usage_include", "headers", "session_header"}
        if unknown:
            raise ValueError(f"{path}: [providers.{name}] has unknown keys {sorted(unknown)}")
        if "headers" in settings:
            headers = settings["headers"]
            if not isinstance(headers, dict) or not all(isinstance(v, str) for v in headers.values()):
                raise ValueError(f"{path}: [providers.{name}] headers must be a table of strings")
            settings = {**settings, "headers": tuple((str(k), v) for k, v in headers.items())}
        if name in providers:
            providers[name] = replace(providers[name], **settings)
        else:
            missing = {"base_url", "key_env"} - set(settings)
            if missing:
                raise ValueError(f"{path}: new provider [providers.{name}] needs {sorted(missing)}")
            providers[name] = Provider(name=name, **settings)
    return providers


def _check_model_providers(design: Design, path: Path) -> None:
    for model, provider in design.model_providers.items():
        if provider not in design.providers:
            raise ValueError(f"{path}: [model_providers] {model!r} = {provider!r}: unknown provider "
                             f"(known: {', '.join(sorted(design.providers))})")
        if model not in design.all_models:
            raise ValueError(f"{path}: [model_providers] lists {model!r}, which is not a model of the config "
                             f"(pipeline models, generator or judge)")




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
