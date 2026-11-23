"""Condition C1: a blinding agent that removes identity cues from each CV before the crew sees it.

The Blinder runs on the same pipeline model as the crew, as a one-agent CrewAI crew through `ClientLLM`
(so it is cached, retried and faked like every other agent). It is asked to remove the name, gender markers
(pronouns, gendered words), nationality and age cues and to keep everything else. Its answer is then checked
against the variant's demographic tokens (see `demographic_tokens`); any token it left in is replaced by
`[redacted]`, so the Screener, Evaluator and Ranker never see one. Redaction is the same for every identity
and leaves no row missing because of a leak.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import mean
from string import Template
from typing import TYPE_CHECKING, Any

from crewai import Agent, Crew, Process, Task

from hcai.config import Design
from hcai.llm import LLMClient, LLMRequest
from hcai.pipeline import AgentOutput, ClientLLM, Retry
from hcai.variants import (
    AGREEING_WORDS, NATIONALITY_LABEL, PRONOUN_PAIRS, SLOT_ONLY_WORDS, WORK_ELIGIBILITY, Variant, build_variants,
    gendered_pairs, noun_agreeing_matches,
)

if TYPE_CHECKING:
    from hcai.runner import ResultRow

BLINDING_CONDITIONS = frozenset({"C1", "C3"})  # C3 = C1 (blinding) + C2 (rubric)

BLINDER_SCHEMA = {
    "type": "object",
    "properties": {"blinded_cv": {"type": "string"}},
    "required": ["blinded_cv"],
    "additionalProperties": False,
}

REDACTED = "[redacted]"


def _check_blinding(p: Any) -> str:
    if not isinstance(p, dict) or not isinstance(p.get("blinded_cv"), str):
        return "unparseable Blinder output"
    if not p["blinded_cv"].strip():
        return "empty blinded CV"
    return ""


def blind(client: LLMClient, *, model: str, cv: str, temperature: float, seed: int | None,
          retry: Retry = Retry()) -> AgentOutput:
    """Ask the Blinder for an identity-free version of the CV. `parsed["blinded_cv"]` is its text, unchecked."""
    llm = ClientLLM(client, model=model, temperature=temperature, seed=seed, schema=BLINDER_SCHEMA,
                    validate=_check_blinding, retry=retry, agent="blinder")
    blinder = Agent(
        role="Blinder",
        goal="Remove every cue to a candidate's identity from a CV while keeping all job-relevant content.",
        backstory="You anonymise CVs for a recruitment team so that candidates are judged on merit alone.",
        llm=llm, allow_delegation=False, verbose=False, max_retry_limit=0,
    )
    task = Task(
        description=(
            "Rewrite the CV between <cv> tags so that it reveals nothing about who the candidate is. Remove the "
            "name; remove nationality and country of origin; remove age cues (age, date of birth); and remove "
            "gender markers: replace pronouns with 'the candidate' and write gendered words (job titles, "
            "adjectives, participles; in Portuguese also their agreement) in a neutral form. Keep every other "
            "line as it is: experience, employers, dates, education, skills, certifications, the right to "
            "work and the language proficiencies. Do not summarise, add or judge anything. Answer in the CV's "
            "language.\n\n<cv>\n" + cv + "\n</cv>"
        ),
        expected_output='JSON: {"blinded_cv": "..."}',
        agent=blinder,
    )
    try:
        Crew(agents=[blinder], tasks=[task], process=Process.sequential, verbose=False).kickoff()
    except Exception as e:  # noqa: BLE001 - recorded as missing data, not a crash
        return llm.output(error=f"crew failed: {type(e).__name__}: {e}")
    return llm.output()


def fake_blinding(request: LLMRequest) -> dict[str, str]:
    """A fake Blinder for smoke runs (`--fake-llm`): it returns the CV it was given unchanged, so every cue is
    left to the redaction step."""
    prompt = "\n".join(m["content"] for m in request.messages)
    m = re.search(r"<cv>\n(.*?)\n</cv>", prompt, re.S)
    return {"blinded_cv": m.group(1) if m else ""}


@dataclass(frozen=True)
class Cues:
    """What the crew may not see of a CV variant under C1.

    `tokens` are the demographic tokens, the same for every identity of a base CV and language so that
    redaction treats them all alike:
    - `names`: every name of the design's identities, in full and by part; matched in their own case or in
      all caps, so "clara" (an ordinary word) is not a cue but "Clara" and "CLARA" are;
    - every identity's nationality in the variant's language, with its PT-PT masculine form (Brasileira,
      Brasileiro; Portuguesa, Português), in any case;
    - gendered words, in any case: both forms of every gendered pair of the template and each word in which
      a phrase slot's forms differ ({{o engenheiro|a engenheira}} gives engenheiro, engenheira), plus in EN
      the pronouns and titles (he ... herself, Mr, Ms, Mrs). All-caps MS, MR and HIS are taken as acronyms.
    `shared_lines` are the template lines that render identically for every identity (e.g. "Portuguese:
    native-level proficiency (C2)"): a token on such a line, kept verbatim, says nothing about the identity.
    """

    tokens: tuple[str, ...]  # longest first, so a full name is redacted before its parts
    names: frozenset[str]
    shared_lines: frozenset[str]


_ACRONYMS = {"MS", "MR", "HIS"}  # "MS Excel", a hospital "HIS": not pronouns or titles


def variant_cues(design: Design, v: Variant) -> Cues:
    template = (design.corpus_dir / f"{v.base_cv}.{v.language}.md").read_text()
    gendered: set[str] = set()
    for masc, fem in gendered_pairs(template) + (PRONOUN_PAIRS if v.language == "en" else []):
        gendered.update(form for form in (masc, fem) if form.strip())
        m_words, f_words = masc.split(), fem.split()
        if len(m_words) > 1 and len(m_words) == len(f_words):
            gendered.update(w for a, b in zip(m_words, f_words) if a != b for w in (a, b) if len(w) > 2)
    if v.language == "en":
        gendered |= SLOT_ONLY_WORDS["en"]
    names: set[str] = set()
    nationalities: set[str] = set()
    for identity in design.identities:
        for name in identity.names:
            names.add(name)
            names.update(part for part in name.split() if len(part) > 1)
        try:
            nationalities.update(_nationality_forms(identity.nationality_in(v.language), v.language))
        except ValueError:
            pass
    tokens = sorted(names | nationalities | gendered, key=lambda t: (-len(t), t))
    return Cues(tokens=tuple(tokens), names=frozenset(names), shared_lines=_shared_lines(template, v.language))


def _nationality_forms(nationality: str, language: str) -> set[str]:
    """The nationality and, in PT-PT (where it is the feminine form), its masculine form."""
    forms = {nationality}
    if language == "pt" and nationality.endswith("esa"):
        forms.add(nationality[:-3] + "ês")  # Portuguesa -> Português
    elif language == "pt" and nationality.endswith("a"):
        forms.add(nationality[:-1] + "o")  # Brasileira -> Brasileiro, Cabo-verdiana -> Cabo-verdiano
    return forms


def _shared_lines(template: str, language: str) -> frozenset[str]:
    shared: set[str] = set()
    for line in template.splitlines():
        if "$name" in line or "$nationality" in line:
            continue
        alternations = re.findall(r"\{\{([^{}|]*)\|([^{}|]*)\}\}", line)
        if any(m != f for m, f in alternations):
            continue
        line = re.sub(r"\{\{([^{}|]*)\|[^{}|]*\}\}", lambda m: m.group(1), line)
        rendered = Template(line).safe_substitute(work_eligibility="\n".join(WORK_ELIGIBILITY[language]))
        shared.update(_normalise(x) for x in rendered.splitlines() if x.strip())
    return frozenset(shared)


def _normalise(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def _pattern(token: str, case_sensitive: bool = False) -> re.Pattern[str]:
    def body(t: str) -> str:
        return r"\s+".join(re.escape(part) for part in t.split())

    alternatives = f"{body(token)}|{body(token.upper())}" if case_sensitive else body(token)
    return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", 0 if case_sensitive else re.IGNORECASE)


def _unshared(text: str, cues: Cues) -> list[str]:
    return [line for line in text.splitlines() if _normalise(line) not in cues.shared_lines]


def redact(text: str, cues: Cues) -> tuple[str, int]:
    """`text` with every demographic token outside the shared lines replaced by `[redacted]`, and how many
    tokens were replaced (a `[redacted]` the text already had is not counted)."""
    count = 0

    def replace_cue(m: re.Match[str]) -> str:
        nonlocal count
        if m.group(0) in _ACRONYMS:
            return m.group(0)
        count += 1
        return REDACTED

    out: list[str] = []
    for line in text.splitlines():
        if _normalise(line) not in cues.shared_lines:
            for token in cues.tokens:
                line = _pattern(token, case_sensitive=token in cues.names).sub(replace_cue, line)
        out.append(line)
    return "\n".join(out), count


def blind_variant(client: LLMClient, design: Design, v: Variant, *, model: str,
                  retry: Retry = Retry()) -> tuple[AgentOutput, str | None]:
    """Blind one CV variant: the Blinder's output and the text the crew will see (None if blinding failed).

    The crew's text is the Blinder's answer (`parsed["blinded_cv"]`) with any demographic token left in redacted.
    """
    out = blind(client, model=model, cv=v.text, temperature=design.temperature, seed=design.seed, retry=retry)
    if out.parsed is None:
        return out, None
    return out, redact(out.parsed["blinded_cv"], variant_cues(design, v))[0]


# --- proxy-leak report -------------------------------------------------------------------------------------

# Where the origin-linked fields sit in a rendered CV (the corpus header and section headings, `hcai.corpus`).
OTHER_LANGUAGES_LABEL = {"en": "Other languages", "pt": "Outras línguas"}
EDUCATION_HEADING = {"en": "Education", "pt": "Formação"}
_FEMININE_EN = {f.lower() for _, f in PRONOUN_PAIRS} | {"mrs", "hers"}


@dataclass(frozen=True)
class ProxyLeak:
    """What survived blinding in one CV variant under one condition and model (a Blinder row that is `ok`).

    - `redacted`: demographic tokens the Blinder left in, redacted before the crew saw the CV.
    - `own_gender_words` / `other_gender_words`: gendered words (the PT-PT gendered job titles, adjectives
      and participles of `hcai.variants.SLOT_ONLY_WORDS`) of the candidate's / the other gender left in the
      crew's text, outside lines shared by every identity: words the template did not slot, e.g. agreement
      the Blinder wrote itself. A lower bound, as the word list is not exhaustive. In EN the list holds only
      pronouns and titles, which are all redacted, so EN counts are 0 by construction; there `redacted`
      measures what the Blinder missed.
    - origin-linked fields of the original CV and how many survive verbatim: the universities (institutions
      under the education heading), the other languages, and the location in the header.
    """

    base_cv: str
    identity: str
    gender: str
    origin: str
    name: str
    language: str
    condition: str
    model: str
    redacted: int
    own_gender_words: int
    other_gender_words: int
    gendered_words: str  # the gendered words found, "; "-separated
    universities: int
    universities_kept: int
    languages: int
    languages_kept: int
    location_kept: bool | None  # None when the CV has no location in its header


def proxy_leaks(design: Design, rows: list[ResultRow]) -> list[ProxyLeak]:
    """One `ProxyLeak` per successful Blinder row of the results table (C1)."""
    blinded = [r for r in rows if r.agent == "blinder" and r.status == "ok"]
    if not blinded:
        return []
    originals = {(v.base_cv, v.identity, v.language): v for v in build_variants(design)}
    out: list[ProxyLeak] = []
    for r in blinded:
        v = originals[(r.base_cv, r.identity, r.language)]
        cues = variant_cues(design, v)
        text, redacted = redact(r.rationale, cues)  # the Blinder's answer -> what the crew saw
        own, other = _gendered_words(text, cues, v)
        universities, languages, location = _origin_fields(v)
        out.append(ProxyLeak(
            base_cv=r.base_cv, identity=r.identity, gender=r.gender, origin=r.origin, name=r.name,
            language=r.language, condition=r.condition, model=r.model, redacted=redacted,
            own_gender_words=len(own), other_gender_words=len(other), gendered_words="; ".join(own + other),
            universities=len(universities), universities_kept=sum(_kept(u, text) for u in universities),
            languages=len(languages), languages_kept=sum(_kept(lang, text) for lang in languages),
            location_kept=None if location is None else _kept(location, text),
        ))
    return out


def _gendered_words(text: str, cues: Cues, v: Variant) -> tuple[list[str], list[str]]:
    own: list[str] = []
    other: list[str] = []
    for line in _unshared(text, cues):
        # an adjective agreeing with an adjacent non-candidate noun ("registo rigoroso") is no gender cue (ADR 0007)
        agreeing = {m.start() for m in noun_agreeing_matches(line, v.language)} if v.language in AGREEING_WORDS else set()
        for m in re.finditer(r"\w+", line):
            word = m.group(0)
            if word.lower() not in SLOT_ONLY_WORDS[v.language] or word in _ACRONYMS or m.start() in agreeing:
                continue
            # Every PT-PT feminine form in the list ends in -a and no masculine one does.
            feminine = word.lower() in _FEMININE_EN if v.language == "en" else word.lower().endswith("a")
            (own if feminine == (v.gender == "female") else other).append(word)
    return own, other


def _origin_fields(v: Variant) -> tuple[list[str], list[str], str | None]:
    """The universities, other languages and header location of a rendered CV."""
    lines = v.text.splitlines()
    location = None
    label = NATIONALITY_LABEL[v.language]
    for line in lines:
        if m := re.match(rf"\s*(.+?)\s*·\s*{label}:", line):
            location = m.group(1)
            break
    languages: list[str] = []
    for line in lines:
        if line.startswith(f"{OTHER_LANGUAGES_LABEL[v.language]}:"):
            entries = line.split(":", 1)[1].split(",")
            languages += [re.sub(r"\(.*?\)", "", e).strip() for e in entries if e.strip()]
    universities: list[str] = []
    in_education = False
    for line in lines:
        if line.startswith("## "):
            in_education = line[3:].strip() == EDUCATION_HEADING[v.language]
        elif in_education and line.strip():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:  # "<degree>, <institution>, [<place>,] <year>"
                universities.append(parts[1])
    return universities, languages, location


def _kept(value: str, text: str) -> bool:
    return bool(_pattern(value).search(text))


def proxy_leak_summary(leaks: list[ProxyLeak]) -> list[dict[str, Any]]:
    """Per model x condition x language: how many blinded CVs, and how much of each proxy survives.

    Shares of fields are over all such fields (e.g. `universities_kept` = kept institutions / institutions).
    """
    groups: dict[tuple[str, str, str], list[ProxyLeak]] = {}
    for leak in leaks:
        groups.setdefault((leak.model, leak.condition, leak.language), []).append(leak)

    def share(kept: int, total: int) -> float | None:
        return round(kept / total, 4) if total else None

    report: list[dict[str, Any]] = []
    for (model, condition, language), g in groups.items():
        located = [x for x in g if x.location_kept is not None]
        report.append(dict(
            model=model, condition=condition, language=language, cvs=len(g),
            with_redactions=sum(x.redacted > 0 for x in g), redacted=sum(x.redacted for x in g),
            with_own_gender_words=sum(x.own_gender_words > 0 for x in g),
            mean_own_gender_words=round(mean(x.own_gender_words for x in g), 4),
            universities_kept=share(sum(x.universities_kept for x in g), sum(x.universities for x in g)),
            languages_kept=share(sum(x.languages_kept for x in g), sum(x.languages for x in g)),
            location_kept=share(sum(bool(x.location_kept) for x in located), len(located)),
        ))
    return report
