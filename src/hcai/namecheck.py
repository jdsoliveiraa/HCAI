"""Name manipulation check (`hcai names`, spec #1 story 11, ADR 0009): do the names signal what they should?

Every full name of the design's identities is shown, alone, to every audited pipeline model (`[design] models`)
`repeats` times, asking for the applicant's perceived gender (male / female / unsure) and most likely origin (one
of `ORIGIN_LABELS` or "Other or unsure"), with a JSON-schema answer. The prompt is the same for every name but the
name itself: it never names the intended gender or origin, and the option lists are reordered per repeat (the
same order for every name at a repeat, never the bank's origin order) to spread position bias over the repeats.
Each repeat is its own request (its own option order and seed), so repeats are real samples and each is cached on
its own: a rerun with the same settings makes no calls.

A name passes when, for every model, the share of its valid answers that give the intended gender and the share
that give the intended origin are both at least the threshold (0.8 by default), with no answer missing. Missing
answers (a provider error, or one outside the schema) are not cached, so a rerun asks them again. Nothing here
changes the name bank: failing names are reported for the researcher to replace.
"""

from __future__ import annotations

import csv
import random
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

    @property
    def missing(self) -> bool:
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


class ValidAnswersOnly:
    """Wraps a client, below the cache: an answer `parse_answer` rejects comes back with `parsed=None`.

    So `CachedClient(..., cache_unparseable=False)` does not cache it (valid JSON outside the options included),
    and a rerun asks it again instead of replaying it.
    """

    def __init__(self, inner: LLMClient):
        self.inner = inner

    def complete(self, request: LLMRequest) -> LLMResponse:
        response = self.inner.complete(request)
        return response if parse_answer(response.parsed) != ("", "") else replace(response, parsed=None)


def perceive(design: Design, client: LLMClient, *, repeats: int, temperature: float | None = None
             ) -> list[Perception]:
    """Ask every pipeline model about every name `repeats` times.

    A call that raises is a missing answer (its error kept); `MAX_CONSECUTIVE_ERRORS` of them in a row for one
    model raise `NameCheckAborted`. `ProviderExhausted` (a BaseException) stops the check as in `hcai run`.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    names = bank_names(design)
    bank_origins = list(dict.fromkeys(o for _, _, o in names))
    temperature = design.temperature if temperature is None else temperature
    out = []
    failures = dict.fromkeys(design.models, 0)  # consecutive failed calls per model
    for name, gender, origin in names:
        for model in design.models:
            for repeat in range(repeats):
                request = build_request(name, model, repeat, bank_origins=bank_origins, temperature=temperature,
                                        seed=design.seed)
                try:
                    response = client.complete(request)
                except Exception as e:  # noqa: BLE001 - a failed call is a missing answer, asked again on a rerun
                    failures[model] += 1
                    error = f"{type(e).__name__}: {e}"
                    if failures[model] >= MAX_CONSECUTIVE_ERRORS:
                        raise NameCheckAborted(f"{model}: {failures[model]} calls in a row failed, the last with "
                                               f"{error}") from e
                    out.append(Perception(name, gender, origin, model, repeat, "", "", error=error))
                    continue
                failures[model] = 0
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
    models = list(dict.fromkeys(s.model for s in scores))
    cells = []
    for gender, origin in dict.fromkeys((s.intended_gender, s.intended_origin) for s in scores):
        in_cell = [s for s in scores if (s.intended_gender, s.intended_origin) == (gender, origin)]
        for model in models:
            ss = [s for s in in_cell if s.model == model]
            cells.append(CellScore(gender, origin, model, len(ss), sum(s.gender_hit_rate is not None for s in ss),
                                   _mean([s.gender_hit_rate for s in ss]),
                                   _mean([s.origin_hit_rate for s in ss]), sum(s.passed for s in ss)))
        names = list(dict.fromkeys(s.name for s in in_cell))
        answered = sum(any(s.gender_hit_rate is not None for s in in_cell if s.name == n) for n in names)
        cells.append(CellScore(gender, origin, "all", len(names), answered, _mean([s.gender_hit_rate for s in in_cell]),
                               _mean([s.origin_hit_rate for s in in_cell]),
                               sum(all(s.passed for s in in_cell if s.name == n) for n in names)))
    return NameCheck(perceptions, scores, cells, threshold)


def _fmt(rate: float | None) -> str:
    return "" if rate is None else f"{rate:.3f}"


def write_outputs(check: NameCheck, out_dir: str | Path) -> list[Path]:
    """perceptions.csv (one row per answer), summary.csv (per name x model), cells.csv (per gender x origin)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "perceptions.csv": (["name", "intended_gender", "intended_origin", "model", "repeat", "perceived_gender",
                             "perceived_origin"],
                            [[p.name, p.intended_gender, p.intended_origin, p.model, p.repeat, p.perceived_gender,
                              p.perceived_origin] for p in check.perceptions]),
        "summary.csv": (["name", "intended_gender", "intended_origin", "model", "answers", "missing",
                         "gender_hit_rate", "origin_hit_rate", "threshold", "pass"],
                        [[s.name, s.intended_gender, s.intended_origin, s.model, s.answers, s.missing,
                          _fmt(s.gender_hit_rate), _fmt(s.origin_hit_rate), check.threshold, s.passed]
                         for s in check.scores]),
        "cells.csv": (["gender", "origin", "model", "names", "names_answered", "mean_gender_hit_rate", "mean_origin_hit_rate",
                       "names_passing"],
                      [[c.gender, c.origin, c.model, c.names, c.answered, _fmt(c.gender_hit_rate), _fmt(c.origin_hit_rate),
                        c.passed] for c in check.cells]),
    }
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


def format_check(check: NameCheck) -> str:
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
    errors: dict[str, list[str]] = {}
    for p in check.perceptions:
        if p.error:
            errors.setdefault(p.model, []).append(p.error)
    if incomplete:
        lines.append(f"\n{len(incomplete)} names have missing answers (a failed call, or an answer outside the "
                     "options); rerun to ask them again")
    lines += [f"{model}: {len(es)} failed calls, e.g. {es[-1]}" for model, es in errors.items()]
    return "\n".join(lines)
