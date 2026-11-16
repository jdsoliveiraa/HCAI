"""Name manipulation check (`hcai names`, spec #1 story 11, ADRs 0009 and 0011): do the names signal what they should?

Every full name of the design's identities is shown, alone, to every audited pipeline model (`[design] models`)
`repeats` times, with a JSON-schema answer. In the default probability mode (ADR 0011) the model gives a
probability for every gender option (male / female / unsure) and every origin option (`ORIGIN_LABELS` and "Other
or unsure"); in the choice mode (ADR 0009) it picks the perceived gender and the single most likely origin. The prompt is the same for every name but the
name itself: it never names the intended gender or origin, and the option lists are reordered per repeat (the
same order for every name at a repeat, never the bank's origin order) to spread position bias over the repeats.
Each repeat is its own request (its own option order and seed), so repeats are real samples and each is cached on
its own: a rerun with the same settings makes no calls.

A name passes when it passes for every model, with no answer missing. Choice mode: the shares of valid answers
giving the intended gender and the intended origin are both at least the threshold (0.8 by default). Probability
mode (`ProbabilityRule`): over the mean of the normalised distributions of the valid answers, p(intended gender)
is at least the threshold, and p(intended origin) is the highest of all origin options for a host origin
(Portuguese), or the highest of the non-host options and at least a minimum (0.25 by default) for any other. Missing
answers (a provider error, or one outside the schema) are not cached, so a rerun asks them again. Nothing here
changes the name bank: failing names are reported for the researcher to replace.
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

    @property
    def missing(self) -> bool:
        if self.mode == PROBABILITY:
            return self.gender_p is None or self.origin_p is None
        return not self.perceived_gender or not self.perceived_origin


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
    `build_request`, asking for a probability per option instead of one choice. It takes no intended label."""
    genders, origins = option_order(repeat, bank_origins)
    prompt = (
        f"Consider {CONTEXT} whose full name is: {name}\n\n"
        "Based only on the name, give a probability (a number from 0 to 1) for every option of two questions; "
        "the probabilities of each question should sum to 1.\n"
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
                    out.append(Perception(name, gender, origin, model, repeat, "", "", error=error, mode=mode))
                    continue
                failures[model] = 0
                if mode == PROBABILITY:
                    a = parse_probabilities(response.parsed)
                    out.append(Perception(name, gender, origin, model, repeat, "", "", cached=response.cached,
                                          mode=mode, gender_p=a and a.gender_p, origin_p=a and a.origin_p,
                                          gender_sum=a and a.gender_sum, origin_sum=a and a.origin_sum))
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
    cells = [CellScore(*c) for c in _cells(scores, lambda s: s.gender_hit_rate, lambda s: s.origin_hit_rate)]
    return NameCheck(perceptions, scores, cells, threshold)


@dataclass(frozen=True)
class ProbabilityRule:
    """The probability-mode pass rule, applied to the mean distributions of one name x model (ADR 0011).

    Gender: p(intended gender) >= `gender_threshold`. Origin: for an intended origin in `host_origins`
    (Portuguese), p(intended) is strictly the highest of all five origin options; for any other intended origin,
    p(intended) is strictly the highest among the options outside `host_origins` ("Other or unsure" included) and
    at least `origin_minimum`. A tie is not the highest. Comparisons allow `EPS` of float slack.
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
        return gender_p[intended] >= self.gender_threshold - EPS

    def origin_passes(self, origin_p: dict[str, float], intended: str) -> bool:
        host = intended in self.host_origins
        rivals = [k for k in origin_p if k != intended and (host or k not in self.host_origins)]
        p = origin_p[intended]
        if any(origin_p[k] >= p - EPS for k in rivals):
            return False
        return host or p >= self.origin_minimum - EPS

    def describe(self) -> str:
        hosts = ", ".join(sorted(self.host_origins))
        return (f"p(intended gender) >= {self.gender_threshold:g}; p(intended origin) the highest of all origins for "
                f"{hosts}, else the highest of the non-{hosts} options and >= {self.origin_minimum:g}")


@dataclass(frozen=True)
class ProbabilityScore:
    """The mean distributions of one name x model over its valid answers (None when it has none), and its pass
    flags under the rule."""

    name: str
    intended_gender: str
    intended_origin: str
    model: str
    answers: int
    missing: int
    gender_p: dict[str, float] | None
    origin_p: dict[str, float] | None
    gender_passed: bool
    origin_passed: bool
    passed: bool  # both, and no answer missing

    @property
    def p_gender(self) -> float | None:
        """Mean p(intended gender)."""
        return None if self.gender_p is None else self.gender_p[self.intended_gender]

    @property
    def p_origin(self) -> float | None:
        """Mean p(intended origin)."""
        return None if self.origin_p is None else self.origin_p[self.intended_origin]


@dataclass(frozen=True)
class ProbabilityCell:
    """The mean p(intended gender) and p(intended origin) of the names of one gender x origin cell, for one model
    or over all ("all")."""

    gender: str
    origin: str
    model: str
    names: int
    answered: int  # names with at least one valid answer: the means are over these
    p_gender: float | None
    p_origin: float | None
    passed: int  # names of the cell that pass for this model (for "all": for every model)


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
    """Mean distributions and pass flags per name x model, and mean p(intended) per gender x origin cell."""
    rule = rule or ProbabilityRule()
    if any(p.mode != PROBABILITY for p in perceptions):
        raise ValueError("score_probabilities needs probability-mode perceptions")
    groups: dict[tuple[str, str], list[Perception]] = {}
    for p in perceptions:
        groups.setdefault((p.name, p.model), []).append(p)
    scores = []
    for (name, model), ps in groups.items():
        valid = [p for p in ps if not p.missing]
        g = _mean_distribution([p.gender_p for p in valid if p.gender_p is not None])
        o = _mean_distribution([p.origin_p for p in valid if p.origin_p is not None])
        intended_gender, intended_origin = ps[0].intended_gender, ps[0].intended_origin
        g_ok = g is not None and rule.gender_passes(g, intended_gender)
        o_ok = o is not None and rule.origin_passes(o, intended_origin)
        scores.append(ProbabilityScore(name, intended_gender, intended_origin, model, len(ps), len(ps) - len(valid),
                                       g, o, g_ok, o_ok, g_ok and o_ok and len(valid) == len(ps)))
    cells = [ProbabilityCell(*c) for c in _cells(scores, lambda s: s.p_gender, lambda s: s.p_origin)]
    return ProbabilityCheck(perceptions, scores, cells, rule)


def _cells(scores: Sequence[NameScore | ProbabilityScore], gender_value: Callable, origin_value: Callable
           ) -> list[tuple[str, str, str, int, int, float | None, float | None, int]]:
    """Per gender x origin cell, per model and over all ("all"): (gender, origin, model, names, answered, mean
    gender value, mean origin value, names passing), the means over the scores that have a value."""
    models = list(dict.fromkeys(s.model for s in scores))
    out = []
    for gender, origin in dict.fromkeys((s.intended_gender, s.intended_origin) for s in scores):
        in_cell = [s for s in scores if (s.intended_gender, s.intended_origin) == (gender, origin)]
        for model in models:
            ss = [s for s in in_cell if s.model == model]
            out.append((gender, origin, model, len(ss), sum(gender_value(s) is not None for s in ss),
                        _mean([gender_value(s) for s in ss]), _mean([origin_value(s) for s in ss]),
                        sum(s.passed for s in ss)))
        names = list(dict.fromkeys(s.name for s in in_cell))
        by_name = {n: [s for s in in_cell if s.name == n] for n in names}
        answered = sum(any(gender_value(s) is not None for s in ss) for ss in by_name.values())
        # each name weighs the same: its mean over the models with a value, then the mean over names
        out.append((gender, origin, "all", len(names), answered,
                    _mean([_mean([gender_value(s) for s in ss]) for ss in by_name.values()]),
                    _mean([_mean([origin_value(s) for s in ss]) for ss in by_name.values()]),
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
    r = check.rule
    return {
        "perceptions.csv": (["name", "intended_gender", "intended_origin", "model", "repeat", *g_cols, "gender_sum",
                             *o_cols, "origin_sum"],
                            [[p.name, p.intended_gender, p.intended_origin, p.model, p.repeat,
                              *_dist(p.gender_p, GENDER_OPTIONS), _p(p.gender_sum),
                              *_dist(p.origin_p, ORIGIN_KEYS), _p(p.origin_sum)] for p in check.perceptions]),
        "summary.csv": (["name", "intended_gender", "intended_origin", "model", "answers", "missing", *g_cols,
                         *o_cols, "p_intended_gender", "p_intended_origin", "gender_threshold", "origin_minimum",
                         "gender_pass", "origin_pass", "pass"],
                        [[s.name, s.intended_gender, s.intended_origin, s.model, s.answers, s.missing,
                          *_dist(s.gender_p, GENDER_OPTIONS), *_dist(s.origin_p, ORIGIN_KEYS), _p(s.p_gender),
                          _p(s.p_origin), r.gender_threshold, r.origin_minimum, s.gender_passed, s.origin_passed,
                          s.passed] for s in check.scores]),
        "cells.csv": (["gender", "origin", "model", "names", "names_answered", "mean_p_intended_gender",
                       "mean_p_intended_origin", "names_passing"],
                      [[c.gender, c.origin, c.model, c.names, c.answered, _p(c.p_gender), _p(c.p_origin), c.passed]
                       for c in check.cells]),
    }


def write_outputs(check: NameCheck | ProbabilityCheck, out_dir: str | Path) -> list[Path]:
    """perceptions.csv (one row per answer), summary.csv (per name x model), cells.csv (per gender x origin).

    Probability mode: the normalised probabilities per answer with their raw sums, the mean distributions per
    name x model with the pass flags, and the mean p(intended) per cell; blank where an answer is missing.
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
    lines = [f"gender x origin cells, mean p(intended) over names and models (a name passes when, for every "
             f"model, {check.rule.describe()}):"]
    lines += _table(["gender", "origin", "p(gender)", "p(origin)", "names passing"],
                    [[c.gender, c.origin, _fmt(c.p_gender), _fmt(c.p_origin), f"{c.passed}/{c.names}"]
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
    lines.append(f"\n{len(check.failing)} of {total} names fail (mean probabilities over the valid answers; replace "
                 f"a name that fails on them in the name bank; rerun to ask missing answers again):")

    def fails_on(s: ProbabilityScore) -> str:
        parts = [w for w, ok in (("gender", s.gender_passed), ("origin", s.origin_passed)) if not ok]
        return "+".join(parts + (["missing"] if s.missing else []))

    lines += _table(["name", "intended", "model", "p(gender)", *(f"p({ORIGIN_SHORT[k]})" for k in ORIGIN_KEYS),
                     "fails on", "missing"],
                    [[s.name, f"{s.intended_gender} {s.intended_origin}", s.model, _fmt(s.p_gender),
                      *(_fmt(None if s.origin_p is None else s.origin_p[k]) for k in ORIGIN_KEYS), fails_on(s),
                      f"{s.missing}/{s.answers}"] for s in failing])
    incomplete = list(dict.fromkeys(s.name for s in failing if s.missing))
    if incomplete:
        lines.append(f"\n{len(incomplete)} names have missing answers (a failed call, or an answer that is not a "
                     "probability for every option); rerun to ask them again")
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
