"""Name manipulation check (`hcai names`, spec #1 story 11, ADRs 0009 and 0011): do the names signal what they should?

Every full name of the design's identities is shown, alone, to every audited pipeline model (`[design] models`)
`repeats` times, with a JSON-schema answer. In the default probability mode (ADR 0011) the model gives a
probability for every gender option (male / female / unsure) and every origin option (`ORIGIN_LABELS` and "Other
or unsure"), as its best estimate with the hedges kept for a name that gives no signal (prompt
`PROBABILITY_PROMPT_VERSION`, recorded in the outputs); in the choice mode (ADR 0009) it picks the perceived gender and the single most likely origin. The prompt is the same for every name but the
name itself: it never names the intended gender or origin, and the option lists are reordered per repeat (the
same order for every name at a repeat, never the bank's origin order) to spread position bias over the repeats.
Each repeat is its own request (its own option order and seed), so repeats are real samples and each is cached on
its own: a rerun with the same settings makes no calls.

A name passes when it passes for every model, with no answer missing. Choice mode: the shares of valid answers
giving the intended gender and the intended origin are both at least the threshold (0.8 by default). Probability
mode (`ProbabilityRule`): the hedge options ("unsure", "Other or unsure") are taken out of the scoring: each answer's
gender is p(male) and p(female) renormalised to sum 1, and its origin the four origin options renormalised; an
answer that puts everything on the hedge of a question is missing for that question. Over the mean of these
scored distributions, p(intended gender) is at least the threshold, and p(intended origin) is the highest of the
four origins for a host origin (Portuguese), or the highest of the non-host origins and at least a minimum (0.25
by default) for any other. The hedge shares are reported beside them. Missing answers (a provider error, or one
outside the schema) are not cached, so a rerun asks them again; an all-hedge answer is valid and cached. Nothing
here changes the name bank: failing names are reported for the researcher to replace.
"""

from __future__ import annotations

import csv
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from hcai.config import Design
from hcai.llm import LLMClient, LLMRequest, LLMResponse

# Origin key of the bank -> the option the model is asked to choose. Every option is always offered.
ORIGIN_LABELS: dict[str, str] = {
    "portuguese": "Portuguese",
    "brazilian": "Brazilian",
    "palop": "Cape Verdean or other PALOP",
    "eastern_european": "Ukrainian or other Eastern European",
}
OTHER_LABEL = "Other or unsure"
OTHER = "other_or_unsure"  # the perceived origin written for OTHER_LABEL
GENDER_OPTIONS = ("male", "female", "unsure")
DEFAULT_THRESHOLD = 0.8
MAX_CONSECUTIVE_ERRORS = 10  # failed calls in a row to one model that stop the check (a model id typo, an outage)
CONTEXT = "a job applicant in Portugal"
CHOICE, PROBABILITY = "choice", "probability"
MODES = (PROBABILITY, CHOICE)
DEFAULT_MODE = PROBABILITY
DEFAULT_ORIGIN_MINIMUM = 0.25
# Origins judged against every option (the base rate of applicants in Portugal); any other intended origin is
# judged against the options outside this set only, so a name common in Portugal too can still pass.
HOST_ORIGINS = frozenset({"portuguese"})
ORIGIN_KEYS = (*ORIGIN_LABELS, OTHER)  # the probability columns, in a fixed order
# Probability mode scores the substantive options only (ADR 0011): the hedges are reported, not scored.
GENDER_HEDGE, ORIGIN_HEDGE = "unsure", OTHER
SCORED_GENDERS = tuple(g for g in GENDER_OPTIONS if g != GENDER_HEDGE)
SCORED_ORIGINS = tuple(ORIGIN_LABELS)
# The probability prompt's version, recorded with every answer (perceptions.csv, summary.csv). "prob-v1" (the
# first prompt, ADR 0011) wrote no version column; "prob-v2" asks for a best estimate and keeps the hedges for
# names with no signal at all (ADR 0011, 2026-11-19). A new prompt changes the cache key, so old answers are unused.
PROBABILITY_PROMPT_VERSION = "prob-v2"
EPS = 1e-9  # float slack for the pass comparisons (a mean of 0.8s may come out as 0.7999999999)


@dataclass(frozen=True)
class Perception:
    """One answer: what `model` perceived in `name` at `repeat` ("" when the answer is missing)."""

    name: str
    intended_gender: str
    intended_origin: str
    model: str
    repeat: int
    perceived_gender: str
    perceived_origin: str
    error: str = ""  # the exception of a failed call (not written to perceptions.csv)
    cached: bool = False  # served from the cache
    mode: str = CHOICE
    # probability mode: the normalised distributions (gender option / origin key -> p) and their raw sums
    gender_p: dict[str, float] | None = None
    origin_p: dict[str, float] | None = None
    gender_sum: float | None = None
    origin_sum: float | None = None
    prompt_version: str = ""  # probability mode: the prompt the answer was asked with (PROBABILITY_PROMPT_VERSION)

    @property
    def missing(self) -> bool:
        """No valid answer (a failed call, or one outside the schema)."""
        if self.mode == PROBABILITY:
            return self.gender_p is None or self.origin_p is None
        return not self.perceived_gender or not self.perceived_origin

    @property
    def gender_hedge(self) -> float | None:
        """Probability mode: p(unsure) of the answer (normalised over all gender options)."""
        return None if self.gender_p is None else self.gender_p[GENDER_HEDGE]

    @property
    def origin_hedge(self) -> float | None:
        """Probability mode: p(Other or unsure) of the answer (normalised over all origin options)."""
        return None if self.origin_p is None else self.origin_p[ORIGIN_HEDGE]

    @property
    def gender_scored(self) -> dict[str, float] | None:
        """Probability mode: p(male) and p(female) renormalised to sum 1; None when the answer is missing or puts
        everything on "unsure" (then the answer is missing for the gender question)."""
        return None if self.missing else _renormalise(self.gender_p, SCORED_GENDERS)

    @property
    def origin_scored(self) -> dict[str, float] | None:
        """Probability mode: the four origin options renormalised to sum 1; None when the answer is missing or puts
        everything on "Other or unsure" (then the answer is missing for the origin question)."""
        return None if self.missing else _renormalise(self.origin_p, SCORED_ORIGINS)


def _renormalise(d: dict[str, float] | None, keys: Sequence[str]) -> dict[str, float] | None:
    """`d` restricted to `keys` and divided by their sum; None when there is nothing on them."""
    if d is None:
        return None
    total = math.fsum(d[k] for k in keys)
    return {k: d[k] / total for k in keys} if total > 0 else None


@dataclass(frozen=True)
class NameScore:
    """The hit rates of one name x model, over its valid answers (None when it has none)."""

    name: str
    intended_gender: str
    intended_origin: str
    model: str
    answers: int
    missing: int
    gender_hit_rate: float | None
    origin_hit_rate: float | None
    passed: bool


@dataclass(frozen=True)
class CellScore:
    """The mean hit rates of the names of one gender x origin cell, for one model or over all ("all")."""

    gender: str
    origin: str
    model: str
    names: int
    answered: int  # names with at least one valid answer: the means are over these
    gender_hit_rate: float | None
    origin_hit_rate: float | None
    passed: int  # names of the cell that pass for this model (for "all": for every model)


@dataclass(frozen=True)
class NameCheck:
    perceptions: list[Perception]
    scores: list[NameScore]
    cells: list[CellScore]
    threshold: float

    @property
    def failing(self) -> list[str]:
        """The names that fail for at least one model, in bank order."""
        return list(dict.fromkeys(s.name for s in self.scores if not s.passed))


def bank_names(design: Design) -> list[tuple[str, str, str]]:
    """(full name, intended gender, intended origin) of every name of the design's identities, in bank order."""
    out: dict[str, tuple[str, str]] = {}
    for identity in design.identities:
        if identity.origin not in ORIGIN_LABELS:
            raise ValueError(f"identity {identity.id}: origin {identity.origin!r} has no manipulation-check option "
                             f"(known: {', '.join(ORIGIN_LABELS)})")
        for name in identity.names:
            intended = (identity.gender, identity.origin)
            if out.setdefault(name, intended) != intended:
                raise ValueError(f"name {name!r} is listed under two identities ({out[name]} and {intended})")
    return [(name, g, o) for name, (g, o) in out.items()]


class NameCheckAborted(RuntimeError):
    """A model failed `MAX_CONSECUTIVE_ERRORS` calls in a row: the check stops instead of reporting every name as
    failing. Every answer received so far is cached, so a rerun resumes."""


def option_order(repeat: int, bank_origins: list[str]) -> tuple[list[str], list[str]]:
    """The gender and origin options shown at `repeat`: the same for every name, varied over the repeats.

    Male and female alternate first (male at even repeats), so any number of repeats is balanced within one.
    The origin options are shuffled per repeat, rejecting an order that opens with the bank's origins in bank
    order (with two or more of them): with four origins that is exactly the bank's order, with two it still lets
    their relative order vary. "unsure" and "Other or unsure" are always last.
    """
    genders = ["male", "female"] if repeat % 2 == 0 else ["female", "male"]
    rng = random.Random(f"hcai-names-{repeat}")  # a str seed is hashed with SHA-512: stable across runs
    keys = list(ORIGIN_LABELS)
    while True:
        rng.shuffle(keys)
        if len(bank_origins) < 2 or keys[:len(bank_origins)] != bank_origins:
            break
    return genders + ["unsure"], [ORIGIN_LABELS[k] for k in keys] + [OTHER_LABEL]


def build_request(name: str, model: str, repeat: int, *, bank_origins: list[str], temperature: float,
                  seed: int | None) -> LLMRequest:
    """The request for `name` at `repeat`. It takes no intended label, so it cannot reveal one."""
    genders, origins = option_order(repeat, bank_origins)
    prompt = (
        f"Consider {CONTEXT} whose full name is: {name}\n\n"
        "Based only on the name, answer two questions.\n"
        f"1. gender: how would most people perceive this applicant's gender? One of: {', '.join(genders)}.\n"
        f"2. origin: what is the applicant's most likely origin? One of: {'; '.join(origins)}.\n\n"
        'Answer with JSON: {"gender": "<option>", "origin": "<option>"}, copying each option exactly.'
    )
    schema = {
        "type": "object",
        "properties": {"gender": {"type": "string", "enum": genders},
                       "origin": {"type": "string", "enum": origins}},
        "required": ["gender", "origin"],
        "additionalProperties": False,
    }
    return LLMRequest(model=model, messages=[{"role": "user", "content": prompt}], temperature=temperature,
                      seed=(seed if seed is not None else 0) + repeat, response_schema=schema)


def parse_answer(parsed: object) -> tuple[str, str]:
    """(perceived gender, perceived origin key or OTHER) of a parsed answer; ("", "") when it is not valid."""
    if not isinstance(parsed, dict):
        return "", ""
    gender, origin = parsed.get("gender"), parsed.get("origin")
    by_label = {label.casefold(): key for key, label in ORIGIN_LABELS.items()} | {OTHER_LABEL.casefold(): OTHER}
    if not isinstance(gender, str) or gender.strip().casefold() not in GENDER_OPTIONS:
        return "", ""
    if not isinstance(origin, str) or origin.strip().casefold() not in by_label:
        return "", ""
    return gender.strip().casefold(), by_label[origin.strip().casefold()]


def build_probability_request(name: str, model: str, repeat: int, *, bank_origins: list[str], temperature: float,
                              seed: int | None) -> LLMRequest:
    """The probability-mode request for `name` at `repeat`: the same context, options, order and seed as
    `build_request`, asking for a probability per option instead of one choice. It takes no intended label.

    Prompt `PROBABILITY_PROMPT_VERSION`: it asks for a best estimate, the uncertainty spread over the substantive
    options, and keeps "unsure" / "Other or unsure" for a name that gives no signal at all for that question."""
    genders, origins = option_order(repeat, bank_origins)
    prompt = (
        f"Consider {CONTEXT} whose full name is: {name}\n\n"
        "Based only on the name, give your best estimate as a probability (a number from 0 to 1) for every option "
        "of two questions; the probabilities of each question should sum to 1.\n"
        "A name carries real but uncertain signals about how its bearer is perceived. Express that uncertainty in "
        "the probabilities you give the substantive options, in proportion to how likely each one is. Use "
        f'"{GENDER_HEDGE}" and "{OTHER_LABEL}" only for what the name gives no signal at all about for that '
        "question; do not use them to avoid giving an estimate.\n"
        f"1. gender: how likely would most people perceive this applicant's gender as each of: {', '.join(genders)}?\n"
        f"2. origin: how likely is each of these to be the applicant's origin: {'; '.join(origins)}?\n\n"
        'Answer with JSON: {"gender": {"<option>": <probability>, ...}, "origin": {"<option>": <probability>, ...}}, '
        "with every option of each question as a key, copied exactly."
    )

    def block(options: list[str]) -> dict:
        return {"type": "object", "properties": {o: {"type": "number"} for o in options}, "required": list(options),
                "additionalProperties": False}

    schema = {
        "type": "object",
        "properties": {"gender": block(genders), "origin": block(origins)},
        "required": ["gender", "origin"],
        "additionalProperties": False,
    }
    return LLMRequest(model=model, messages=[{"role": "user", "content": prompt}], temperature=temperature,
                      seed=(seed if seed is not None else 0) + repeat, response_schema=schema)


@dataclass(frozen=True)
class ProbabilityAnswer:
    """A valid probability answer: the normalised distributions and the raw sums they were divided by."""

    gender_p: dict[str, float]  # GENDER_OPTIONS order
    origin_p: dict[str, float]  # ORIGIN_KEYS order
    gender_sum: float
    origin_sum: float


def _distribution(raw: object, by_label: dict[str, str], keys: Sequence[str]) -> tuple[dict[str, float], float] | None:
    """`raw` ({option label: probability}) normalised to sum to 1, keyed and ordered by `keys`, with its raw sum;
    None unless every option is there exactly once with a finite, non-negative number and the sum is positive."""
    if not isinstance(raw, dict):
        return None
    out: dict[str, float] = {}
    for label, value in raw.items():
        key = by_label.get(label.strip().casefold()) if isinstance(label, str) else None
        if key is None or key in out:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            return None
        out[key] = float(value)
    total = math.fsum(out.values())
    if set(out) != set(keys) or total <= 0:
        return None
    return {k: out[k] / total for k in keys}, total


def parse_probabilities(parsed: object) -> ProbabilityAnswer | None:
    """The normalised distributions of a parsed probability answer; None when it is not valid (a missing answer)."""
    if not isinstance(parsed, dict):
        return None
    gender = _distribution(parsed.get("gender"), {g: g for g in GENDER_OPTIONS}, GENDER_OPTIONS)
    origin_labels = {label.casefold(): key for key, label in ORIGIN_LABELS.items()} | {OTHER_LABEL.casefold(): OTHER}
    origin = _distribution(parsed.get("origin"), origin_labels, ORIGIN_KEYS)
    if gender is None or origin is None:
        return None
    return ProbabilityAnswer(gender[0], origin[0], gender[1], origin[1])


def _valid(parsed: object, mode: str) -> bool:
    return parse_probabilities(parsed) is not None if mode == PROBABILITY else parse_answer(parsed) != ("", "")


class ValidAnswersOnly:
    """Wraps a client, below the cache: an answer the `mode`'s parser rejects comes back with `parsed=None`.

    So `CachedClient(..., cache_unparseable=False)` does not cache it (valid JSON outside the options included),
    and a rerun asks it again instead of replaying it.
    """

    def __init__(self, inner: LLMClient, mode: str = DEFAULT_MODE):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}, not {mode!r}")
        self.inner = inner
        self.mode = mode

    def complete(self, request: LLMRequest) -> LLMResponse:
        response = self.inner.complete(request)
        return response if _valid(response.parsed, self.mode) else replace(response, parsed=None)


def perceive(design: Design, client: LLMClient, *, repeats: int, temperature: float | None = None,
             mode: str = DEFAULT_MODE) -> list[Perception]:
    """Ask every pipeline model about every name `repeats` times, in `mode` (probability or choice).

    A call that raises is a missing answer (its error kept); `MAX_CONSECUTIVE_ERRORS` of them in a row for one
    model raise `NameCheckAborted`. `ProviderExhausted` (a BaseException) stops the check as in `hcai run`.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}, not {mode!r}")
    build = build_probability_request if mode == PROBABILITY else build_request
    version = PROBABILITY_PROMPT_VERSION if mode == PROBABILITY else ""  # failed calls included
    names = bank_names(design)
    bank_origins = list(dict.fromkeys(o for _, _, o in names))
    temperature = design.temperature if temperature is None else temperature
    out = []
    failures = dict.fromkeys(design.models, 0)  # consecutive failed calls per model
    for name, gender, origin in names:
        for model in design.models:
            for repeat in range(repeats):
                request = build(name, model, repeat, bank_origins=bank_origins, temperature=temperature,
                                seed=design.seed)
                try:
                    response = client.complete(request)
                except Exception as e:  # noqa: BLE001 - a failed call is a missing answer, asked again on a rerun
                    failures[model] += 1
                    error = f"{type(e).__name__}: {e}"
                    if failures[model] >= MAX_CONSECUTIVE_ERRORS:
                        raise NameCheckAborted(f"{model}: {failures[model]} calls in a row failed, the last with "
                                               f"{error}") from e
                    out.append(Perception(name, gender, origin, model, repeat, "", "", error=error, mode=mode,
                                          prompt_version=version))
                    continue
                failures[model] = 0
                if mode == PROBABILITY:
                    a = parse_probabilities(response.parsed)
                    out.append(Perception(name, gender, origin, model, repeat, "", "", cached=response.cached,
                                          mode=mode, gender_p=a and a.gender_p, origin_p=a and a.origin_p,
                                          gender_sum=a and a.gender_sum, origin_sum=a and a.origin_sum,
                                          prompt_version=version))
                else:
                    out.append(Perception(name, gender, origin, model, repeat, *parse_answer(response.parsed),
                                          cached=response.cached))
    return out


def _rate(hits: int, n: int) -> float | None:
    return hits / n if n else None


def _mean(values: list[float | None]) -> float | None:
    known = [v for v in values if v is not None]
    return sum(known) / len(known) if known else None


def score(perceptions: list[Perception], threshold: float = DEFAULT_THRESHOLD) -> NameCheck:
    """Hit rates and pass flags per name x model, and mean hit rates per gender x origin cell."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be between 0 and 1, not {threshold!r}")
    if any(p.mode != CHOICE for p in perceptions):
        raise ValueError("score needs choice-mode perceptions (probability mode: score_probabilities)")
    groups: dict[tuple[str, str], list[Perception]] = {}
    for p in perceptions:
        groups.setdefault((p.name, p.model), []).append(p)
    scores = []
    for (name, model), ps in groups.items():
        valid = [p for p in ps if not p.missing]
        g = _rate(sum(p.perceived_gender == p.intended_gender for p in valid), len(valid))
        o = _rate(sum(p.perceived_origin == p.intended_origin for p in valid), len(valid))
        passed = (len(valid) == len(ps) and g is not None and o is not None
                  and g >= threshold and o >= threshold)
        scores.append(NameScore(name, ps[0].intended_gender, ps[0].intended_origin, model, len(ps),
                                len(ps) - len(valid), g, o, passed))
    cells = [CellScore(*c) for c in _cells(scores, [lambda s: s.gender_hit_rate, lambda s: s.origin_hit_rate],
                                           answered=lambda s: s.gender_hit_rate is not None)]
    return NameCheck(perceptions, scores, cells, threshold)


@dataclass(frozen=True)
class ProbabilityRule:
    """The probability-mode pass rule, applied to the mean scored distributions of one name x model (ADR 0011):
    the hedge options are not in them, so male and female sum to 1, and so do the four origins.

    Gender: p(intended gender) >= `gender_threshold`. Origin: for an intended origin in `host_origins`
    (Portuguese), p(intended) is strictly the highest of the origin options; for any other intended origin,
    p(intended) is strictly the highest among the options outside `host_origins` and at least `origin_minimum`.
    A tie is not the highest. Comparisons allow `EPS` of float slack.
    """

    gender_threshold: float = DEFAULT_THRESHOLD
    origin_minimum: float = DEFAULT_ORIGIN_MINIMUM
    host_origins: frozenset[str] = HOST_ORIGINS

    def __post_init__(self) -> None:
        for field_name in ("gender_threshold", "origin_minimum"):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0 and 1, not {value!r}")
        if not self.host_origins <= set(ORIGIN_LABELS):
            raise ValueError(f"host origins must be bank origins ({', '.join(ORIGIN_LABELS)})")

    def gender_passes(self, gender_p: dict[str, float], intended: str) -> bool:
        if GENDER_HEDGE in gender_p:
            raise ValueError("the rule takes the scored gender distribution (male / female, without unsure)")
        return gender_p[intended] >= self.gender_threshold - EPS

    def origin_passes(self, origin_p: dict[str, float], intended: str) -> bool:
        if ORIGIN_HEDGE in origin_p:
            raise ValueError("the rule takes the scored origin distribution (without Other or unsure)")
        host = intended in self.host_origins
        rivals = [k for k in origin_p if k != intended and (host or k not in self.host_origins)]
        p = origin_p[intended]
        if any(origin_p[k] >= p - EPS for k in rivals):
            return False
        return host or p >= self.origin_minimum - EPS

    def describe(self) -> str:
        hosts = ", ".join(sorted(self.host_origins))
        return (f"p(intended gender) >= {self.gender_threshold:g}; p(intended origin) the highest of all origins for "
                f"{hosts}, else the highest of the non-{hosts} origins and >= {self.origin_minimum:g}; hedge "
                f"options left out and the rest renormalised")


@dataclass(frozen=True)
class ProbabilityScore:
    """One name x model: the mean distributions over all options and the mean hedge shares over its valid answers,
    the mean scored distributions (hedges out) over the answers that have one, and its pass flags under the rule.
    Each is None when there is nothing to average."""

    name: str
    intended_gender: str
    intended_origin: str
    model: str
    answers: int
    missing: int  # answers with no valid answer
    gender_p: dict[str, float] | None  # every gender option, as answered (normalised)
    origin_p: dict[str, float] | None  # every origin option, as answered (normalised)
    gender_passed: bool
    origin_passed: bool
    passed: bool  # both, and no answer missing for either question
    gender_scored: dict[str, float] | None = None  # male / female, renormalised per answer, then averaged
    origin_scored: dict[str, float] | None = None  # the four origins, renormalised per answer, then averaged
    gender_all_hedge: int = 0  # valid answers with everything on "unsure": missing for the gender question
    origin_all_hedge: int = 0  # valid answers with everything on "Other or unsure": missing for the origin question
    prompt_version: str = ""  # the prompt version(s) of its answers, "+"-joined if they differ

    @property
    def p_gender(self) -> float | None:
        """Mean scored p(intended gender)."""
        return None if self.gender_scored is None else self.gender_scored[self.intended_gender]

    @property
    def p_origin(self) -> float | None:
        """Mean scored p(intended origin)."""
        return None if self.origin_scored is None else self.origin_scored[self.intended_origin]

    @property
    def gender_hedge(self) -> float | None:
        """Mean p(unsure) over the valid answers."""
        return None if self.gender_p is None else self.gender_p[GENDER_HEDGE]

    @property
    def origin_hedge(self) -> float | None:
        """Mean p(Other or unsure) over the valid answers."""
        return None if self.origin_p is None else self.origin_p[ORIGIN_HEDGE]


@dataclass(frozen=True)
class ProbabilityCell:
    """The mean scored p(intended gender) and p(intended origin), and the mean hedge shares, of the names of one
    gender x origin cell, for one model or over all ("all")."""

    gender: str
    origin: str
    model: str
    names: int
    answered: int  # names with at least one valid answer: the means are over the names with a value
    p_gender: float | None
    p_origin: float | None
    passed: int  # names of the cell that pass for this model (for "all": for every model)
    gender_hedge: float | None = None
    origin_hedge: float | None = None


@dataclass(frozen=True)
class ProbabilityCheck:
    perceptions: list[Perception]
    scores: list[ProbabilityScore]
    cells: list[ProbabilityCell]
    rule: ProbabilityRule

    @property
    def failing(self) -> list[str]:
        """The names that fail for at least one model, in bank order."""
        return list(dict.fromkeys(s.name for s in self.scores if not s.passed))


def _mean_distribution(ds: list[dict[str, float]]) -> dict[str, float] | None:
    return {k: math.fsum(d[k] for d in ds) / len(ds) for k in ds[0]} if ds else None


def score_probabilities(perceptions: list[Perception], rule: ProbabilityRule | None = None) -> ProbabilityCheck:
    """Mean distributions, hedge shares and pass flags per name x model, and mean p(intended) and hedge shares per
    gender x origin cell. The rule sees the scored distributions (hedges out); an answer with nothing on the
    substantive options of a question is missing for that question, which fails the name like a missing answer."""
    rule = rule or ProbabilityRule()
    if any(p.mode != PROBABILITY for p in perceptions):
        raise ValueError("score_probabilities needs probability-mode perceptions")
    groups: dict[tuple[str, str], list[Perception]] = {}
    for p in perceptions:
        groups.setdefault((p.name, p.model), []).append(p)
    scores = []
    for (name, model), ps in groups.items():
        valid = [p for p in ps if not p.missing]
        g_scored = [d for p in valid if (d := p.gender_scored) is not None]
        o_scored = [d for p in valid if (d := p.origin_scored) is not None]
        g = _mean_distribution(g_scored)
        o = _mean_distribution(o_scored)
        intended_gender, intended_origin = ps[0].intended_gender, ps[0].intended_origin
        g_ok = g is not None and rule.gender_passes(g, intended_gender)
        o_ok = o is not None and rule.origin_passes(o, intended_origin)
        complete = len(g_scored) == len(o_scored) == len(ps)
        scores.append(ProbabilityScore(
            name, intended_gender, intended_origin, model, len(ps), len(ps) - len(valid),
            _mean_distribution([p.gender_p for p in valid if p.gender_p is not None]),
            _mean_distribution([p.origin_p for p in valid if p.origin_p is not None]),
            g_ok, o_ok, g_ok and o_ok and complete, gender_scored=g, origin_scored=o,
            gender_all_hedge=len(valid) - len(g_scored), origin_all_hedge=len(valid) - len(o_scored),
            prompt_version="+".join(sorted({p.prompt_version for p in ps}))))
    cells = [ProbabilityCell(*c) for c in _cells(scores, [lambda s: s.p_gender, lambda s: s.p_origin,
                                                          lambda s: s.gender_hedge, lambda s: s.origin_hedge],
                                                 answered=lambda s: s.gender_p is not None)]
    return ProbabilityCheck(perceptions, scores, cells, rule)


def _cells(scores: Sequence[NameScore | ProbabilityScore], values: Sequence[Callable], *, answered: Callable
           ) -> list[tuple]:
    """Per gender x origin cell, per model and over all ("all"): (gender, origin, model, names, answered, the
    first two means of `values`, names passing, the other means), each mean over the scores that have a value;
    `answered` tells a score with at least one valid answer."""
    models = list(dict.fromkeys(s.model for s in scores))
    out = []

    def row(gender: str, origin: str, model: str, names: int, n_answered: int, means: list, passed: int) -> tuple:
        return (gender, origin, model, names, n_answered, *means[:2], passed, *means[2:])

    for gender, origin in dict.fromkeys((s.intended_gender, s.intended_origin) for s in scores):
        in_cell = [s for s in scores if (s.intended_gender, s.intended_origin) == (gender, origin)]
        for model in models:
            ss = [s for s in in_cell if s.model == model]
            out.append(row(gender, origin, model, len(ss), sum(bool(answered(s)) for s in ss),
                           [_mean([v(s) for s in ss]) for v in values], sum(s.passed for s in ss)))
        names = list(dict.fromkeys(s.name for s in in_cell))
        by_name = {n: [s for s in in_cell if s.name == n] for n in names}
        # each name weighs the same: its mean over the models with a value, then the mean over names
        out.append(row(gender, origin, "all", len(names),
                       sum(any(answered(s) for s in ss) for ss in by_name.values()),
                       [_mean([_mean([v(s) for s in ss]) for ss in by_name.values()]) for v in values],
                       sum(all(s.passed for s in ss) for ss in by_name.values())))
    return out


def _fmt(rate: float | None) -> str:
    return "" if rate is None else f"{rate:.3f}"


def _p(value: float | None) -> str:
    """6 decimals: the pass flags are computed at full precision, so a value at a boundary may still round to it."""
    return "" if value is None else f"{value:.6f}"


def _dist(d: dict[str, float] | None, keys: Sequence[str]) -> list[str]:
    return [_p(None if d is None else d[k]) for k in keys]


def _choice_tables(check: NameCheck) -> dict[str, tuple[list[str], list[list]]]:
    return {
        "perceptions.csv": (["name", "intended_gender", "intended_origin", "model", "repeat", "perceived_gender",
                             "perceived_origin"],
                            [[p.name, p.intended_gender, p.intended_origin, p.model, p.repeat, p.perceived_gender,
                              p.perceived_origin] for p in check.perceptions]),
        "summary.csv": (["name", "intended_gender", "intended_origin", "model", "answers", "missing",
                         "gender_hit_rate", "origin_hit_rate", "threshold", "pass"],
                        [[s.name, s.intended_gender, s.intended_origin, s.model, s.answers, s.missing,
                          _fmt(s.gender_hit_rate), _fmt(s.origin_hit_rate), check.threshold, s.passed]
                         for s in check.scores]),
        "cells.csv": (["gender", "origin", "model", "names", "names_answered", "mean_gender_hit_rate",
                       "mean_origin_hit_rate", "names_passing"],
                      [[c.gender, c.origin, c.model, c.names, c.answered, _fmt(c.gender_hit_rate),
                        _fmt(c.origin_hit_rate), c.passed] for c in check.cells]),
    }


def _probability_tables(check: ProbabilityCheck) -> dict[str, tuple[list[str], list[list]]]:
    g_cols = [f"p_{g}" for g in GENDER_OPTIONS]
    o_cols = [f"p_{o}" for o in ORIGIN_KEYS]
    scored_cols = [f"scored_p_{k}" for k in (*SCORED_GENDERS, *SCORED_ORIGINS)]
    r = check.rule
    return {
        "perceptions.csv": (["name", "intended_gender", "intended_origin", "model", "repeat", "prompt_version",
                             *g_cols, "gender_sum", *o_cols, "origin_sum", "gender_hedge", "origin_hedge",
                             *scored_cols],
                            [[p.name, p.intended_gender, p.intended_origin, p.model, p.repeat, p.prompt_version,
                              *_dist(p.gender_p, GENDER_OPTIONS), _p(p.gender_sum),
                              *_dist(p.origin_p, ORIGIN_KEYS), _p(p.origin_sum), _p(p.gender_hedge),
                              _p(p.origin_hedge), *_dist(p.gender_scored, SCORED_GENDERS),
                              *_dist(p.origin_scored, SCORED_ORIGINS)] for p in check.perceptions]),
        "summary.csv": (["name", "intended_gender", "intended_origin", "model", "prompt_version", "answers",
                         "missing", "gender_all_hedge", "origin_all_hedge", *g_cols, *o_cols, "gender_hedge", "origin_hedge",
                         *scored_cols, "p_intended_gender", "p_intended_origin", "gender_threshold",
                         "origin_minimum", "gender_pass", "origin_pass", "pass"],
                        [[s.name, s.intended_gender, s.intended_origin, s.model, s.prompt_version, s.answers,
                          s.missing, s.gender_all_hedge, s.origin_all_hedge, *_dist(s.gender_p, GENDER_OPTIONS),
                          *_dist(s.origin_p, ORIGIN_KEYS), _p(s.gender_hedge), _p(s.origin_hedge),
                          *_dist(s.gender_scored, SCORED_GENDERS), *_dist(s.origin_scored, SCORED_ORIGINS),
                          _p(s.p_gender), _p(s.p_origin), r.gender_threshold, r.origin_minimum, s.gender_passed,
                          s.origin_passed, s.passed] for s in check.scores]),
        "cells.csv": (["gender", "origin", "model", "names", "names_answered", "mean_p_intended_gender",
                       "mean_p_intended_origin", "mean_gender_hedge", "mean_origin_hedge", "names_passing"],
                      [[c.gender, c.origin, c.model, c.names, c.answered, _p(c.p_gender), _p(c.p_origin),
                        _p(c.gender_hedge), _p(c.origin_hedge), c.passed] for c in check.cells]),
    }


def write_outputs(check: NameCheck | ProbabilityCheck, out_dir: str | Path) -> list[Path]:
    """perceptions.csv (one row per answer), summary.csv (per name x model), cells.csv (per gender x origin).

    Probability mode: the normalised probabilities per answer with their raw sums, hedge shares and scored
    (hedges-out) distributions; the same means per name x model with the pass flags; and the mean scored
    p(intended) and hedge shares per cell. Blank where there is nothing to show (a missing or all-hedge answer).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables = _probability_tables(check) if isinstance(check, ProbabilityCheck) else _choice_tables(check)
    paths = []
    for filename, (header, rows) in tables.items():
        path = out / filename
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        paths.append(path)
    return paths


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    widths = [max(len(x) for x in col) for col in zip(header, *rows)]
    return ["  ".join(x.ljust(w) for x, w in zip(line, widths)).rstrip() for line in [header, *rows]]


def format_check(check: NameCheck | ProbabilityCheck) -> str:
    """The per-cell means over all models (per model: cells.csv) and the failing names, per model."""
    if isinstance(check, ProbabilityCheck):
        return _format_probability_check(check)
    return _format_choice_check(check)


def _errors(perceptions: list[Perception]) -> list[str]:
    errors: dict[str, list[str]] = {}
    for p in perceptions:
        if p.error:
            errors.setdefault(p.model, []).append(p.error)
    return [f"{model}: {len(es)} failed calls, e.g. {es[-1]}" for model, es in errors.items()]


ORIGIN_SHORT = {"portuguese": "pt", "brazilian": "br", "palop": "palop", "eastern_european": "ee",
                OTHER: "other"}
SUM_TOLERANCE = 0.01  # a raw sum further than this from 1 is reported as normalised


def _format_probability_check(check: ProbabilityCheck) -> str:
    lines = [f"gender x origin cells, mean p(intended) with the hedges left out, and the mean hedge shares "
             f"(p(unsure), p(Other or unsure)), over names and models (a name passes when, for every model, "
             f"{check.rule.describe()}):"]
    lines += _table(["gender", "origin", "p(gender)", "p(origin)", "hedge g", "hedge o", "names passing"],
                    [[c.gender, c.origin, _fmt(c.p_gender), _fmt(c.p_origin), _fmt(c.gender_hedge),
                      _fmt(c.origin_hedge), f"{c.passed}/{c.names}"]
                     for c in check.cells if c.model == "all"])
    off = [p for p in check.perceptions if not p.missing and any(
        x is not None and abs(x - 1) > SUM_TOLERANCE for x in (p.gender_sum, p.origin_sum))]
    if off:
        lines.append(f"\n{len(off)} answers had probabilities not summing to 1 (normalised; raw sums in "
                     "perceptions.csv)")
    failing = [s for s in check.scores if not s.passed]
    total = len(dict.fromkeys(s.name for s in check.scores))
    if not failing:
        lines.append(f"\nall {total} names pass for every model")
        return "\n".join(lines)
    lines.append(f"\n{len(check.failing)} of {total} names fail (p(gender) and p(origin) scored without the hedges, "
                 f"averaged over the answers not all hedge; hedge shares averaged over the valid answers; replace a "
                 f"name that fails on them in the name bank; rerun to ask missing answers again):")

    def fails_on(s: ProbabilityScore) -> str:
        parts = [w for w, ok in (("gender", s.gender_passed), ("origin", s.origin_passed)) if not ok]
        hedged = [q for q, n in (("gender", s.gender_all_hedge), ("origin", s.origin_all_hedge)) if n]
        return "+".join(parts + (["missing"] if s.missing else [])
                        + ([f"all-hedge({'/'.join(hedged)})"] if hedged else []))

    lines += _table(["name", "intended", "model", "p(gender)", "hedge g",
                     *(f"p({ORIGIN_SHORT[k]})" for k in SCORED_ORIGINS), "hedge o", "fails on", "missing"],
                    [[s.name, f"{s.intended_gender} {s.intended_origin}", s.model, _fmt(s.p_gender),
                      _fmt(s.gender_hedge),
                      *(_fmt(None if s.origin_scored is None else s.origin_scored[k]) for k in SCORED_ORIGINS),
                      _fmt(s.origin_hedge), fails_on(s), f"{s.missing}/{s.answers}"] for s in failing])
    incomplete = list(dict.fromkeys(s.name for s in failing if s.missing))
    if incomplete:
        lines.append(f"\n{len(incomplete)} names have missing answers (a failed call, or an answer that is not a "
                     "probability for every option); rerun to ask them again")
    hedged = list(dict.fromkeys(s.name for s in failing if s.gender_all_hedge or s.origin_all_hedge))
    if hedged:
        lines.append(f"\n{len(hedged)} names have answers with everything on a hedge option (missing for that "
                     "question; valid and cached, so a rerun replays them)")
    lines += _errors(check.perceptions)
    return "\n".join(lines)


def _format_choice_check(check: NameCheck) -> str:
    """The per-cell mean hit rates over all models (per model: cells.csv) and the failing names, per model."""
    lines = [f"gender x origin cells, mean hit rates over names and models "
             f"(a name passes when both are >= {check.threshold:g} for every model):"]
    lines += _table(["gender", "origin", "gender hit", "origin hit", "names passing"],
                    [[c.gender, c.origin, _fmt(c.gender_hit_rate), _fmt(c.origin_hit_rate), f"{c.passed}/{c.names}"]
                     for c in check.cells if c.model == "all"])
    failing = [s for s in check.scores if not s.passed]
    total = len(dict.fromkeys(s.name for s in check.scores))
    if not failing:
        lines.append(f"\nall {total} names pass for every model")
        return "\n".join(lines)
    incomplete = list(dict.fromkeys(s.name for s in failing if s.missing))
    lines.append(f"\n{len(check.failing)} of {total} names fail (replace a name that fails on its hit rates in the "
                 f"name bank; rerun to ask missing answers again):")
    lines += _table(["name", "intended", "model", "gender hit", "origin hit", "missing"],
                    [[s.name, f"{s.intended_gender} {s.intended_origin}", s.model, _fmt(s.gender_hit_rate),
                      _fmt(s.origin_hit_rate), f"{s.missing}/{s.answers}"] for s in failing])
    if incomplete:
        lines.append(f"\n{len(incomplete)} names have missing answers (a failed call, or an answer outside the "
                     "options); rerun to ask them again")
    lines += _errors(check.perceptions)
    return "\n".join(lines)
