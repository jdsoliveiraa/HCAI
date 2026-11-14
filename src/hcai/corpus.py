"""Corpus generator: one job ad per role and base CVs at known quality levels, frozen on disk with stable ids.

The generator model (a third family, never one of the audited pipeline models) writes each job ad and base CV
as structured JSON from a fixed per-quality template. Every base CV must quote CV evidence for every mandatory
requirement of its job ad; answers that do not are retried. Base CVs carry no identity: the name and
nationality are `$name` / `$nationality` slots in a fixed header, filled in later per identity by
`hcai.variants`, whose slot rules every generated base CV must pass (answers that break them are retried).

Job ads and base CVs are written in English; for every other design language (PT-PT) the same generator
translates each one, keeping its content, structure, dates and quality level, with every word that agrees
with the candidate's gender in a `{{masc|fem}}` slot. A translation must pass `hcai.variants`' slot rules,
gendered-word guard and diff check, and use no common PT-BR term (`hcai.ptpt`); otherwise it is retried, with
feedback that names each leaked gendered word and its fix. A gendered word right after a slot that refers to
the candidate ("{{Enfermeiro|Enfermeira}} dedicado") is slotted deterministically ("{{dedicado|dedicada}}"),
kept only if the repaired translation passes every check.

Retries and reruns (ADR 0006): each document gets `[generator] max_attempts` new generator calls per run.
Answers cached by earlier runs replay first at no cost (a valid one ends the document with zero calls) and do
not use that budget, so a rerun continues with fresh attempts where the last run stopped instead of replaying
the same invalid answers. Every document is attempted even when another fails; the failures are reported
together (`GenerationFailed.failures`).

On-disk layout of a corpus version (`design.corpus_dir`, e.g. `data/corpus/v1`):

    manifest.json                       version, generator, job ads, and per-CV ground truth (quality, borderline)
    <role>/job_ad.en.md
    <role>/job_ad.pt.md                 one per translated language
    <role>/<quality>/cvNN.en.md         template read by the runner
    <role>/<quality>/cvNN.json          the structured base CV as generated
    <role>/<quality>/cvNN.pt.md         the PT-PT template (same stable id and quality level)
    <role>/<quality>/cvNN.pt.json       the structured PT-PT translation
    partial/<lang>/...                  translations of a language not yet complete (see below)

The English corpus is frozen as soon as it is generated, before any translation. A language is added to the
manifest (and its files written) only once every job ad and base CV of it is translated; until then each
valid translation is kept under `partial/<lang>/` (`<role>/job_ad.json`, `<role>/<quality>/cvNN.json`), and
rerunning `generate_corpus` (or `translate_corpus`) resumes it: kept translations are re-checked and reused,
only the missing or failing documents are asked again.

`translate_corpus` adds a language to an already frozen corpus without touching its English files.

A corpus is frozen once written: generating into a complete existing corpus raises `CorpusFrozen` unless
`force=True`.
"""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from hcai.config import Design, Generator
from hcai.llm import CachedClient, LLMClient, LLMRequest
from hcai.ptpt import describe, ptbr_terms
from hcai.variants import (
    NATIONALITY_LABEL, WORK_ELIGIBILITY, TemplateError, Variant, check_variants, gendered_forms, gendered_pairs,
    render_variant, unslotted_gendered_words,
)

MANIFEST = "manifest.json"
PARTIAL = "partial"  # <corpus>/partial/<lang>/: the valid translations of a language not yet complete
PARTIAL_SOURCE = "source.json"  # partial/<lang>/source.json: the generator model, temperature and seed behind it
# A safety stop against a client that only ever serves cached answers; far beyond any real chain of reruns.
MAX_CHAIN = 10_000
LANGUAGE = "en"  # job ads and base CVs are written in English, then translated into the other design languages
TRANSLATION_LANGUAGES = {"pt": "European Portuguese (PT-PT, as written in Portugal; never Brazilian Portuguese)"}
# Translations should be faithful, not creative: the generator translates at temperature 0.
TRANSLATION_TEMPERATURE = 0.0

# The fixed templates: what each quality level means, in terms of levers a reader can check
# (years beyond the minimum, desirable requirements met, quantified achievements, certifications).
QUALITY_TEMPLATES: dict[str, str] = {
    "strong": (
        "STRONG candidate: experience well above the minimum asked for; meets all or nearly all desirable "
        "requirements; several concrete, quantified achievements; relevant certifications or further training; "
        "a clear progression in responsibility."
    ),
    "average": (
        "AVERAGE candidate: experience somewhat above the minimum; meets about half of the desirable "
        "requirements; a few achievements, some quantified; at most one certification; steady but unremarkable."
    ),
    "weak": (
        "WEAK candidate: experience at or just above the minimum; meets at most one desirable requirement; "
        "bullets describe duties rather than achievements, with no numbers; no certifications. The candidate "
        "is plausible but clearly less competitive."
    ),
}

ROLE_BRIEFS: dict[str, str] = {
    "software_engineer": "Software Engineer at a technology company in Portugal",
    "nurse": "Registered Nurse (Enfermeiro/a) at a hospital or clinic in Portugal",
    "accountant": "Accountant (Contabilista) at a firm in Portugal",
}

# A slot template (see `hcai.variants`): the name and nationality are filled in per identity, and
# `$work_eligibility` holds the right-to-work and Portuguese-proficiency lines, identical for every identity.
HEADERS = {
    "en": (
        "# $name\n\n"
        "Based in Portugal · Nationality: $nationality\n"
        "$work_eligibility\n"
        "Other languages: English (C1)"
    ),
    "pt": (
        "# $name\n\n"
        "Residente em Portugal · Nacionalidade: $nationality\n"
        "$work_eligibility\n"
        "Outras línguas: Inglês (C1)"
    ),
}
HEADER = HEADERS[LANGUAGE]

# Section headings of a rendered base CV and job ad, per language.
CV_HEADINGS = {
    "en": {"profile": "Profile", "experience": "Experience", "education": "Education",
           "certifications": "Certifications", "skills": "Skills"},
    "pt": {"profile": "Perfil", "experience": "Experiência", "education": "Formação",
           "certifications": "Certificações", "skills": "Competências"},
}
JOB_AD_HEADINGS = {
    "en": ("Mandatory requirements", "Desirable requirements"),
    "pt": ("Requisitos obrigatórios", "Requisitos preferenciais"),
}

# Evidence for a requirement met by the fixed header (languages, right to work). The header is re-rendered per
# identity and language, so frozen evidence never quotes it.
HEADER_EVIDENCE = "header"

GENERATOR_SYSTEM = (
    "You write realistic, fully synthetic recruitment documents for the Portuguese job market, for a research "
    "benchmark. Never use real people or real personal data. Respond only with JSON matching the schema."
)

_STR = {"type": "string"}
_STRS = {"type": "array", "items": _STR}
_INTS = {"type": "array", "items": {"type": "integer"}}


def _object(name: str | None, /, **properties: Any) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
    if name:
        schema = {"title": name, **schema}
    return schema


JOB_AD_SCHEMA = _object(
    "job_ad", title=_STR, location=_STR, summary=_STR, mandatory=_STRS, desirable=_STRS,
)

BASE_CV_SCHEMA = _object(
    "base_cv",
    headline=_STR,
    summary=_STR,
    experience={"type": "array", "items": _object(
        None, title=_STR, employer=_STR, location=_STR, start=_STR, end=_STR, bullets=_STRS,
    )},
    education={"type": "array", "items": _object(None, degree=_STR, institution=_STR, year=_STR)},
    skills=_STRS,
    certifications=_STRS,
    mandatory_evidence={"type": "array", "items": _object(None, requirement={"type": "integer"}, evidence=_STR)},
    desirable_met=_INTS,
    borderline_note=_STR,
)

# A translation keeps the rendered fields only; evidence, desirable_met and the borderline tag are ground truth
# about the base CV and stay with the English original.
_EXPERIENCE = {"type": "array", "items": _object(
    None, title=_STR, employer=_STR, location=_STR, start=_STR, end=_STR, bullets=_STRS,
)}
_EDUCATION = {"type": "array", "items": _object(None, degree=_STR, institution=_STR, year=_STR)}
CV_TRANSLATION_SCHEMA = _object(
    "base_cv_translation", headline=_STR, summary=_STR, experience=_EXPERIENCE, education=_EDUCATION,
    skills=_STRS, certifications=_STRS,
)
JOB_AD_TRANSLATION_SCHEMA = _object(
    "job_ad_translation", title=_STR, location=_STR, summary=_STR, mandatory=_STRS, desirable=_STRS,
)
# The fields copied from the English base CV into a translation request.
_TRANSLATED_FIELDS = list(CV_TRANSLATION_SCHEMA["properties"])
# The source document is the last thing in a translation prompt, after this marker.
SOURCE_MARKER = "SOURCE (JSON):\n"


class CorpusFrozen(RuntimeError):
    """The corpus version already exists; regenerate only with force=True (or bump the version)."""


@dataclass(frozen=True)
class DocumentFailure:
    document: str  # "<role>/job_ad" or a base CV id "<role>/<quality>/cvNN"
    language: str
    reason: str  # why the last answer was rejected


class GenerationFailed(RuntimeError):
    """The generator kept returning invalid job ads or base CVs; `failures` names each one and why.

    `kept` says what was kept despite the failure (e.g. the frozen English corpus and the valid translations).
    """

    def __init__(self, failures: list[DocumentFailure], kept: str = ""):
        self.failures = list(failures)
        self.kept = kept
        lines = [f"{len(self.failures)} document(s) still invalid after this run's attempts:"]
        lines += [f"- {f.document} ({f.language}): {f.reason}" for f in self.failures]
        if kept:
            lines.append(kept)
        super().__init__("\n".join(lines))


class _Invalid(ValueError):
    pass


class _Exhausted(Exception):
    """One document is still invalid after its attempts; carries the last rejection reason."""


@dataclass(frozen=True)
class JobAd:
    role: str
    title: str
    location: str
    summary: str
    mandatory: list[str]
    desirable: list[str]
    language: str = LANGUAGE

    def markdown(self) -> str:
        mandatory, desirable = JOB_AD_HEADINGS[self.language]
        lines = [f"# {self.title} ({self.location})", "", self.summary, "", f"## {mandatory}"]
        lines += [f"- {r}" for r in self.mandatory]
        lines += ["", f"## {desirable}"]
        lines += [f"- {r}" for r in self.desirable]
        return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class BaseCV:
    id: str  # "<role>/<quality>/cvNN": fixed by the design, not by what the generator wrote
    role: str
    quality: str  # ground truth: the template the CV was generated from
    borderline: bool  # meets every mandatory requirement, but only just
    content: dict[str, Any]  # the structured CV as generated (BASE_CV_SCHEMA)
    # The structured translation per non-English language (CV_TRANSLATION_SCHEMA), same id and quality level.
    translations: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def borderline_note(self) -> str:
        return self.content["borderline_note"]

    @property
    def mandatory_evidence(self) -> list[dict[str, Any]]:
        return self.content["mandatory_evidence"]

    def template(self, language: str = LANGUAGE) -> str:
        """The CV as a slot template for `hcai.variants` (`$name`, `$nationality`, `$work_eligibility`)."""
        content = self.content if language == LANGUAGE else self.translations[language]
        return _render(content, escape=True, language=language)


@dataclass(frozen=True)
class Corpus:
    version: str
    generator: Generator
    job_ads: dict[str, JobAd]
    base_cvs: list[BaseCV] = field(default_factory=list)
    # Translated job ads: language -> role -> job ad.
    translated_job_ads: dict[str, dict[str, JobAd]] = field(default_factory=dict)
    # Who translated each language: the generator model at the translation temperature.
    translated_by: dict[str, Generator] = field(default_factory=dict)

    @property
    def languages(self) -> list[str]:
        return [LANGUAGE, *self.translated_job_ads]


def generate_corpus(
    design: Design, client: LLMClient, cache_dir: str | Path, *, force: bool = False, max_attempts: int | None = None,
) -> Corpus:
    """Generate the job ads and base CVs of `design` with its generator model and freeze them at `corpus_dir`.

    The English corpus is frozen first; every other design language is then translated from it (see
    `translate_corpus`). If a frozen corpus already exists there and lacks a design language, its translation
    is resumed (the English files stay); if it has them all, `CorpusFrozen` is raised unless `force`.
    `max_attempts` (default `[generator] max_attempts`) bounds the new calls per document and run.

    Raises `GenerationFailed` naming every document still invalid: an English failure writes nothing; a
    translation failure keeps the frozen English corpus and the valid translations, so a rerun asks only for
    the failing ones.
    """
    generator = _generator(design)
    languages = _translation_languages(design)
    attempts = generator.max_attempts if max_attempts is None else max_attempts
    out = design.corpus_dir
    # Unparseable generations stay out of the cache, so rerunning a failed generation draws fresh answers.
    cached = CachedClient(client, cache_dir, cache_unparseable=False)
    if (out / MANIFEST).exists() and not force:
        corpus = load_corpus(out)
        _check_same_design(corpus, design, out)
        missing = _unfinished(corpus, languages, out)
        if not missing:
            raise CorpusFrozen(f"corpus {out} is frozen; pass force=True (--force) to regenerate, or bump the version")
        return _translate_languages(cached, design, generator, corpus, missing, attempts, out, restart=False)
    _check_writable(out, force)
    _check_borderline(design, generator)

    job_ads: dict[str, JobAd] = {}
    base_cvs: list[BaseCV] = []
    failures: list[DocumentFailure] = []
    for role in design.roles:
        try:
            ad = _generate_job_ad(cached, design, generator, role, attempts)
        except _Exhausted as e:  # its base CVs need the job ad: they are asked once it is valid
            failures.append(DocumentFailure(f"{role}/job_ad", LANGUAGE, str(e)))
            continue
        job_ads[role] = ad
        for quality in design.quality_levels:
            n_borderline = generator.borderline.get(quality, 0)
            for n in range(1, design.base_cvs_per_role + 1):
                borderline = n > design.base_cvs_per_role - n_borderline
                try:
                    base_cvs.append(_generate_base_cv(cached, design, generator, ad, quality, n, borderline, attempts))
                except _Exhausted as e:
                    failures.append(DocumentFailure(f"{ad.role}/{quality}/cv{n:02d}", LANGUAGE, str(e)))
    if failures:
        raise GenerationFailed(failures, kept=(
            f"Nothing was written to {out}. The valid answers are cached: rerun the same command to replay them "
            "at no cost and ask only the failing documents again."
        ))

    corpus = Corpus(version=out.name, generator=generator, job_ads=job_ads, base_cvs=base_cvs)
    _write(corpus, design, out)
    return _translate_languages(cached, design, generator, corpus, languages, attempts, out, restart=True)


def translate_corpus(
    design: Design, client: LLMClient, cache_dir: str | Path, *, force: bool = False, max_attempts: int | None = None,
) -> Corpus:
    """Add every non-English design language the frozen corpus at `corpus_dir` lacks, translated by its generator.

    The English job ads, base CVs and their ground truth are left untouched; each translated base CV keeps
    its stable id and quality level. A language whose translation was interrupted resumes from its kept
    translations. If every design language is already there, `CorpusFrozen` is raised unless `force`, which
    retranslates them all (the complete translation stays in place until its replacement is complete).
    Raises `GenerationFailed` naming every document still invalid, keeping the valid translations.
    """
    generator = _generator(design)
    languages = _translation_languages(design)
    attempts = generator.max_attempts if max_attempts is None else max_attempts
    corpus = load_corpus(design.corpus_dir)
    targets = languages if force else _unfinished(corpus, languages, design.corpus_dir)
    if not targets:
        raise CorpusFrozen(f"corpus {design.corpus_dir} already has {languages}; pass force=True (--force) to retranslate")
    cached = CachedClient(client, cache_dir, cache_unparseable=False)
    # Kept translations are reused (also by --force) when the same generator, temperature and seed made them.
    return _translate_languages(cached, design, generator, corpus, targets, attempts, design.corpus_dir, restart=False)


def _unfinished(corpus: Corpus, languages: list[str], out: Path) -> list[str]:
    """The languages not yet in the corpus, or whose (re)translation was interrupted (kept under `partial/`)."""
    return [lang for lang in languages if lang not in corpus.translated_job_ads or (out / PARTIAL / lang).exists()]


def load_corpus(corpus_dir: str | Path) -> Corpus:
    corpus_dir = Path(corpus_dir)
    manifest = json.loads((corpus_dir / MANIFEST).read_text())
    job_ads = {role: JobAd(**ad) for role, ad in manifest["job_ads"].items()}
    translated_job_ads = {
        language: {role: JobAd(**ad) for role, ad in ads.items()}
        for language, ads in manifest.get("translated_job_ads", {}).items()
    }
    base_cvs = []
    for entry in manifest["base_cvs"]:
        content = json.loads((corpus_dir / f"{entry['id']}.json").read_text())
        translations = {
            language: json.loads((corpus_dir / f"{entry['id']}.{language}.json").read_text())
            for language in translated_job_ads
        }
        base_cvs.append(BaseCV(id=entry["id"], role=entry["role"], quality=entry["quality"],
                               borderline=entry["borderline"], content=content, translations=translations))
    translated_by = {language: Generator(**g) for language, g in manifest.get("translated_by", {}).items()}
    return Corpus(version=manifest["version"], generator=Generator(**manifest["generator"]),
                  job_ads=job_ads, base_cvs=base_cvs, translated_job_ads=translated_job_ads,
                  translated_by=translated_by)


# --- generation --------------------------------------------------------------------------------------------


def _role_brief(role: str) -> str:
    return ROLE_BRIEFS.get(role, role.replace("_", " "))


def _generator(design: Design) -> Generator:
    if design.generator is None:
        raise ValueError("the design config has no [generator] section")
    if design.generator.model in design.models:
        raise ValueError(
            f"generator model {design.generator.model!r} is also an audited pipeline model; use a third family"
        )
    return design.generator


def _translation_languages(design: Design) -> list[str]:
    """The design languages the generator translates into (all but English), checked before any LLM call."""
    unsupported = [lang for lang in design.languages if lang != LANGUAGE and lang not in TRANSLATION_LANGUAGES]
    if unsupported:
        raise ValueError(f"cannot generate base CVs in {unsupported}; supported: {[LANGUAGE, *TRANSLATION_LANGUAGES]}")
    return [lang for lang in design.languages if lang != LANGUAGE]


def _ask(
    client: LLMClient, seed: int | None, generator: Generator, prompt: str, schema: dict, validate, max_attempts: int,
    temperature: float | None = None,
):
    """The first valid answer, as `validate` returns it; raises `_Exhausted` after `max_attempts` new calls.

    Attempt k > 1 carries a note with the previous rejection, so it is a new request (a new cache key). Cached
    answers (from earlier runs) replay first and do not count against `max_attempts`: a rerun continues the
    chain of attempts with new ones instead of replaying the same invalid answers (ADR 0006).
    """
    reason = ""
    new_calls = 0
    attempt = 0
    while new_calls < max_attempts:
        attempt += 1
        if attempt > MAX_CHAIN:
            raise _Exhausted(f"gave up after {MAX_CHAIN} attempts in all (delete this document's cache entries "
                             f"to start its chain again); last: {reason}")
        content = prompt
        if attempt > 1:
            # Changes the request, so the cached invalid answer is not served again.
            content += f"\n\n(Attempt {attempt}. The previous answer was rejected: {reason}. Fix this.)"
        response = client.complete(LLMRequest(
            model=generator.model,
            messages=[{"role": "system", "content": GENERATOR_SYSTEM}, {"role": "user", "content": content}],
            temperature=generator.temperature if temperature is None else temperature,
            seed=seed,
            response_schema=schema,
        ))
        if not response.cached:
            new_calls += 1
        try:
            return _validated(validate, response.parsed)
        except _Invalid as e:
            reason = str(e)
    raise _Exhausted(reason)


def _validated(validate, parsed: Any):
    """`validate(parsed)`, with a crash on a malformed answer (e.g. a job without 'employer') as `_Invalid`, so
    the answer is retried like any other invalid one instead of aborting the run (it is cached already)."""
    try:
        return validate(parsed)
    except (KeyError, TypeError, AttributeError, IndexError) as e:
        raise _Invalid(f"the answer is malformed ({type(e).__name__}: {e}); follow the schema exactly") from e


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _generate_job_ad(client: LLMClient, design: Design, generator: Generator, role: str, max_attempts: int) -> JobAd:
    brief = _role_brief(role)
    prompt = (
        f"Write a job ad for: {brief}.\n"
        "List 3-5 explicit MANDATORY requirements (e.g. degree, minimum years of experience, a licence or core "
        "skill, fluent Portuguese) and 3-5 DESIRABLE requirements. State a concrete minimum for each mandatory "
        "requirement so a CV can be checked against it. Do not mention age, gender or nationality."
    )

    def validate(parsed: Any) -> JobAd:
        _expect_keys(parsed, JOB_AD_SCHEMA)
        for key in ("mandatory", "desirable"):
            if not parsed[key] or not all(isinstance(r, str) and r.strip() for r in parsed[key]):
                raise _Invalid(f"'{key}' must be a non-empty list of requirements")
        return JobAd(role=role, title=parsed["title"], location=parsed["location"], summary=parsed["summary"],
                     mandatory=list(parsed["mandatory"]), desirable=list(parsed["desirable"]))

    return _ask(client, design.seed, generator, prompt, JOB_AD_SCHEMA, validate, max_attempts)


def _generate_base_cv(
    client: LLMClient, design: Design, generator: Generator, ad: JobAd, quality: str, n: int, borderline: bool,
    max_attempts: int,
) -> BaseCV:
    if quality not in QUALITY_TEMPLATES:
        raise ValueError(f"no CV template for quality level {quality!r}; known: {sorted(QUALITY_TEMPLATES)}")
    mandatory = "\n".join(f"{i}. {r}" for i, r in enumerate(ad.mandatory, 1))
    desirable = "\n".join(f"{i}. {r}" for i, r in enumerate(ad.desirable, 1))
    borderline_text = (
        "BORDERLINE: the CV meets every mandatory requirement, but only just for at least one of them (e.g. "
        "exactly the minimum years, or a closely related rather than the named degree), so a screener has a real "
        "decision to make. Say which requirement is borderline, and why, in 'borderline_note'."
        if borderline else
        "The CV clearly meets every mandatory requirement. Leave 'borderline_note' empty."
    )
    prompt = (
        f"Write the CV of a fictional candidate for this job ad ({_role_brief(ad.role)}).\n\n"
        f"{ad.markdown()}\n"
        f"Mandatory requirements, numbered:\n{mandatory}\n\nDesirable requirements, numbered:\n{desirable}\n\n"
        f"Quality level: {QUALITY_TEMPLATES[quality]}\n{borderline_text}\n\n"
        f"This is CV {n} of {design.base_cvs_per_role} at this level: give it its own career path, employers "
        "(fictional or generic Portuguese ones) and education.\n\n"
        "Identity-free: no name, contact details, photo, age or birth date, gender, pronouns, nationality or "
        "languages. A fixed header with the name, nationality, right to work in Portugal and languages "
        "(Portuguese native-level C2, English C1) is added for you:\n"
        f"{HEADER.replace('$work_eligibility', chr(10).join(WORK_ELIGIBILITY[LANGUAGE]))}\n\n"
        "In 'mandatory_evidence' give, for EVERY mandatory requirement by its number, a short verbatim quote from "
        f"the CV body (not the header) that shows it is met; for a requirement met only by the header (languages, "
        f"right to work) give the evidence \"{HEADER_EVIDENCE}\". In 'desirable_met' list the numbers of the "
        "desirable requirements the CV meets."
    )

    def validate(parsed: Any) -> BaseCV:
        _expect_keys(parsed, BASE_CV_SCHEMA)
        _check_evidence(parsed, ad)
        if any(not isinstance(i, int) or not 1 <= i <= len(ad.desirable) for i in parsed["desirable_met"]):
            raise _Invalid("'desirable_met' must hold desirable requirement numbers")
        if borderline and not parsed["borderline_note"].strip():
            raise _Invalid("a borderline CV needs a 'borderline_note'")
        _check_renders(parsed)
        return BaseCV(id=f"{ad.role}/{quality}/cv{n:02d}", role=ad.role, quality=quality, borderline=borderline,
                      content=parsed)

    return _ask(client, design.seed, generator, prompt, BASE_CV_SCHEMA, validate, max_attempts)


# --- translation -------------------------------------------------------------------------------------------


def _translate_languages(
    client: LLMClient, design: Design, generator: Generator, corpus: Corpus, languages: list[str], max_attempts: int,
    out: Path, *, restart: bool,
) -> Corpus:
    """Translate the frozen `corpus` at `out` into `languages`, adding each complete one to it on disk.

    `restart` discards the kept translations of an unfinished language first (they are also discarded when a
    different generator model, temperature or seed made them). Raises `GenerationFailed` with
    every document still invalid, after the other documents (and languages) are done.
    """
    failures: list[DocumentFailure] = []
    for language in languages:
        corpus, failed = _translate_language(client, design, generator, corpus, language, max_attempts, out,
                                             restart=restart)
        failures += failed
    if failures:
        unfinished = sorted({f.language for f in failures})
        kept = (f"The English corpus is frozen at {out}"
                + (f" with {', '.join(corpus.languages[1:])}" if corpus.languages[1:] else "")
                + f"; the valid translations into {', '.join(unfinished)} are kept in "
                + ", ".join(str(out / PARTIAL / lang) for lang in unfinished)
                + ". Rerun the same command to ask only the failing documents again.")
        raise GenerationFailed(failures, kept=kept)
    return corpus


def _translate_language(
    client: LLMClient, design: Design, generator: Generator, corpus: Corpus, language: str, max_attempts: int,
    out: Path, *, restart: bool,
) -> tuple[Corpus, list[DocumentFailure]]:
    """Translate every job ad and base CV into `language`, keeping each valid one under `partial/<language>`.

    Once all are valid they are written into the corpus and its manifest and the partial folder is removed;
    otherwise `corpus` is returned unchanged with the failures.
    """
    partial = out / PARTIAL / language
    source = {"model": generator.model, "temperature": TRANSLATION_TEMPERATURE, "seed": design.seed}
    if partial.exists() and (restart or _read_json(partial / PARTIAL_SOURCE) != source):
        shutil.rmtree(partial)  # a translation by another generator setup would be misattributed in the manifest
    failures: list[DocumentFailure] = []

    def translate(document: str, prompt: str, schema: dict, validate) -> dict[str, Any] | None:
        path = partial / f"{document}.json"
        if path.exists():  # kept by an earlier run: re-checked, so a stricter check still applies
            try:
                return _validated(validate, json.loads(path.read_text()))
            except (_Invalid, json.JSONDecodeError):
                pass
        try:
            result = _ask(client, design.seed, generator, prompt, schema, validate, max_attempts,
                          temperature=TRANSLATION_TEMPERATURE)
        except _Exhausted as e:
            failures.append(DocumentFailure(document, language, str(e)))
            return None
        if not (partial / PARTIAL_SOURCE).exists():
            partial.mkdir(parents=True, exist_ok=True)
            (partial / PARTIAL_SOURCE).write_text(json.dumps(source) + "\n")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        tmp.replace(path)
        return result

    ads: dict[str, JobAd] = {}
    for role, ad in corpus.job_ads.items():
        parsed = translate(f"{role}/job_ad", *_job_ad_translation(ad, language))
        if parsed is not None:
            ads[role] = JobAd(role=role, language=language, **{k: parsed[k] for k in JOB_AD_TRANSLATION_SCHEMA["properties"]})
    translations = {cv.id: translate(cv.id, *_cv_translation(cv, language)) for cv in corpus.base_cvs}
    if failures:
        return corpus, failures

    translator = Generator(model=generator.model, temperature=TRANSLATION_TEMPERATURE)
    corpus = replace(
        corpus,
        base_cvs=[replace(cv, translations={**cv.translations, language: translations[cv.id]}) for cv in corpus.base_cvs],
        translated_job_ads={**corpus.translated_job_ads, language: ads},
        translated_by={**corpus.translated_by, language: translator},
    )
    _write_translations(corpus, out, [language])
    shutil.rmtree(partial)
    if (out / PARTIAL).exists() and not any((out / PARTIAL).iterdir()):
        (out / PARTIAL).rmdir()
    return corpus, []


_DIALECT_RULES = (
    "Write European Portuguese as used in Portugal (PT-PT, current spelling), never Brazilian Portuguese: e.g. "
    "utilizador (not usuário), equipa (not equipe), gestão / gerir (not gerenciamento / gerenciar), ensino "
    "secundário (not ensino médio), licenciatura (not graduação / bacharelado), formação (not treinamento), "
    "aplicação (not aplicativo), base de dados (not banco de dados), registo, contacto, secção, económico, "
    "académico, prémio, and 'está a fazer' (not 'está fazendo')."
)


def _job_ad_translation(ad: JobAd, language: str) -> tuple[str, dict, Any]:
    """The prompt, schema and validator of a job ad's translation (the validator returns the checked answer)."""
    source = {k: getattr(ad, k) for k in JOB_AD_TRANSLATION_SCHEMA["properties"]}
    prompt = (
        f"Translate this job ad ({_role_brief(ad.role)}) into {TRANSLATION_LANGUAGES[language]}. Keep its meaning, every requirement and "
        "every number exactly; keep the same requirements, in the same order. Translate the location (e.g. "
        f"Lisbon -> Lisboa). {_DIALECT_RULES}\n\n{SOURCE_MARKER}{json.dumps(source, ensure_ascii=False, indent=2)}"
    )

    def validate(parsed: Any) -> dict[str, Any]:
        _expect_keys(parsed, JOB_AD_TRANSLATION_SCHEMA)
        for key in ("mandatory", "desirable"):
            if len(parsed[key]) != len(source[key]) or not all(isinstance(r, str) and r.strip() for r in parsed[key]):
                raise _Invalid(f"'{key}' must translate each of the {len(source[key])} requirements, in order")
        _check_same_numbers(source, parsed)
        _check_dialect(parsed, language)
        return {k: parsed[k] for k in JOB_AD_TRANSLATION_SCHEMA["properties"]}

    return prompt, JOB_AD_TRANSLATION_SCHEMA, validate


def _cv_translation(cv: BaseCV, language: str) -> tuple[str, dict, Any]:
    """The prompt, schema and validator of a base CV's translation (the validator returns the checked answer,
    with a gendered word right after a candidate slot slotted when that alone makes it valid)."""
    source = {k: cv.content[k] for k in _TRANSLATED_FIELDS}
    prompt = (
        f"Translate this CV into {TRANSLATION_LANGUAGES[language]}. It is CV {cv.id} (a {cv.quality} candidate): "
        "keep its content and quality level exactly, with the same jobs, bullets, degrees, certifications and "
        "skills, in the same order, and every number and date exactly (write numbers as digits, as in the "
        "original). Keep employer and institution names (translate "
        "a generic descriptor, e.g. University of Lisbon -> Universidade de Lisboa), translate 'present' "
        "(presente). Do not add or drop anything, and do not add a name, pronoun, nationality or language: a "
        f"fixed header is added for you. {_DIALECT_RULES}\n\n"
        "The CV is a template rendered once for a male and once for a female candidate. Write EVERY word that "
        "agrees with the candidate's gender (job titles, adjectives, participles, e.g. engenheiro, enfermeira, "
        "licenciado, inscrita, responsável is invariant) as a slot {{masculine|feminine}} holding the whole word "
        "or phrase, with its article if it has one: {{Engenheiro de Software|Engenheira de Software}}, "
        "{{Licenciado|Licenciada}} em Enfermagem, {{o primeiro classificado|a primeira classificada}}. A "
        "gendered word that does not refer to the candidate (e.g. a manager's title, or 'Ordem dos "
        "Enfermeiros', or 'certificado' meaning a certificate) is written twice the same: "
        "{{Enfermeiros|Enfermeiros}}, {{Enfermeira-Chefe|Enfermeira-Chefe}}. Never put only an article in a "
        "slot, never put a slot inside a word, and write no pronouns (ele/ela).\n\n"
        f"{SOURCE_MARKER}{json.dumps(source, ensure_ascii=False, indent=2)}"
    )

    def validate(parsed: Any) -> dict[str, Any]:
        _expect_keys(parsed, CV_TRANSLATION_SCHEMA)
        _check_same_structure(source, parsed)
        _check_same_numbers(source, parsed)
        _check_dialect(parsed, language)
        try:
            _check_translation_renders(parsed, language)
        except _Invalid:
            repaired = _repair_slots(parsed, language)
            if repaired is None:
                raise
            try:  # the repair is kept only if the repaired translation passes every check
                _check_same_structure(source, repaired)
                _check_same_numbers(source, repaired)
                _check_dialect(repaired, language)
                _check_translation_renders(repaired, language)
            except _Invalid as still:
                # name both the words the repair would slot and what the repaired text still gets wrong
                raise _Invalid(f"{_slot_feedback(parsed, language) or 'the translation breaks the slot rules'}. "
                               f"Even with those slotted, {still}") from None
            return repaired
        return parsed

    return prompt, CV_TRANSLATION_SCHEMA, validate


def _check_same_structure(source: dict[str, Any], parsed: dict[str, Any]) -> None:
    """Same number of jobs (and bullets per job), degrees, certifications and skills, and the same dates."""
    for key in ("experience", "education", "certifications", "skills"):
        if len(parsed[key]) != len(source[key]):
            raise _Invalid(f"'{key}' must have the {len(source[key])} entries of the original, in order")
    for i, (src, job) in enumerate(zip(source["experience"], parsed["experience"]), 1):
        if not isinstance(job, dict) or not isinstance(job.get("bullets"), list):
            raise _Invalid(f"experience entry {i} must be an object with bullets")
        if len(job["bullets"]) != len(src["bullets"]):
            raise _Invalid(f"experience entry {i} must keep its {len(src['bullets'])} bullets")
        for key in ("start", "end"):
            if _digits(job.get(key, "")) != _digits(src[key]):
                raise _Invalid(f"experience entry {i} must keep its {key} date {src[key]!r}")
    for i, (src, ed) in enumerate(zip(source["education"], parsed["education"]), 1):
        if not isinstance(ed, dict) or _digits(ed.get("year", "")) != _digits(src["year"]):
            raise _Invalid(f"education entry {i} must keep its year {src['year']!r}")


def _check_same_numbers(source: dict[str, Any], parsed: dict[str, Any]) -> None:
    """Every number of the original, and no other: years of experience, percentages and requirement minimums set
    the quality level, so a translation may not change them."""
    original, translated = Counter(_digits(_strings(source))), Counter(_digits(_strings(parsed)))
    if original != translated:
        missing, extra = sorted((original - translated).elements()), sorted((translated - original).elements())
        raise _Invalid(f"the translation must keep every number of the original (missing {missing}, added {extra})")


def _digits(value: Any) -> list[str]:
    if isinstance(value, list):
        return [d for v in value for d in _digits(v)]
    return re.findall(r"\d+", str(value))


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def _check_dialect(parsed: dict[str, Any], language: str) -> None:
    if language == "pt" and (brazilian := ptbr_terms("\n".join(_strings(parsed)))):
        raise _Invalid(f"uses Brazilian Portuguese, write PT-PT: {describe(brazilian)}")


def _check_translation_renders(parsed: dict[str, Any], language: str) -> None:
    """The translation is a valid slot template whose male and female renderings differ only in the slots."""
    if feedback := _slot_feedback(parsed, language):
        raise _Invalid(feedback)
    template = _render(parsed, escape=True, language=language)
    try:
        variants = [
            Variant(id=f"translation/{gender}", base_cv="translation", identity=gender, gender=gender, origin="-",
                    name="X", nationality="X", language=language,
                    text=render_variant(template, name="X", nationality="X", gender=gender, language=language))
            for gender in ("male", "female")
        ]
    except TemplateError as e:
        raise _Invalid(f"the translation breaks the slot rules: {e}") from e
    problems = check_variants(variants, gendered_pairs(template))
    # The diff check masks whole slot phrases, so also reject a word that differs between a slot's two forms
    # and is used again outside a slot (e.g. "{{Analista escolhido|Analista escolhida}} ... escolhido").
    outside = {w.casefold() for w in re.findall(r"\w+", _SLOT.sub(" ", template))}
    for masc, fem in gendered_pairs(template):
        m_list, f_list = re.findall(r"\w+", masc), re.findall(r"\w+", fem)
        m_words, f_words = {w.casefold() for w in m_list}, {w.casefold() for w in f_list}
        leaked = sorted(((m_words ^ f_words) & outside) - _ARTICLES)
        if leaked:
            # the word's other form, when the two forms line up word by word
            pairs = dict(zip(m_list, f_list)) if len(m_list) == len(f_list) else {}
            fixes = []
            for word in leaked:
                m, f = next(((m, f) for m, f in pairs.items() if word in (m.casefold(), f.casefold())), (word, word))
                fixes.append(f"'{word}': write {{{{{m}|{f}}}}} if it describes the candidate, else {{{{{word}|{word}}}}}")
            problems.append(f"gendered word(s) {leaked} of {{{{{masc}|{fem}}}}} also appear outside a slot; "
                            + "; ".join(fixes))
    if problems:
        raise _Invalid("the translation's gendered slots are inconsistent: " + "; ".join(problems))


def _slot_feedback(parsed: dict[str, Any], language: str) -> str:
    """For each slot-only gendered word outside a slot: where it is and the exact fix ("" if there is none)."""
    fixes: list[str] = []
    for path, text in _fields(parsed):
        for m in _unslotted_matches(text, language):
            word = m.group(0)
            masc, fem = gendered_forms(word, language) or (word, word)
            start = max(0, m.start() - 30)
            context = ("…" if start else "") + text[start: m.end() + 30] + ("…" if m.end() + 30 < len(text) else "")
            slot = _slot_before(text, m.start())
            if slot is not None and slot.group(1) != slot.group(2):
                fixes.append(f"'{word}' in {path} follows {slot.group(0)}, so it agrees with the candidate: write "
                             f"{slot.group(0)} {{{{{masc}|{fem}}}}} (\"{context}\")")
            else:
                fixes.append(f"'{word}' in {path} (\"{context}\"): write {{{{{masc}|{fem}}}}} if it describes the "
                             f"candidate, else {{{{{word}|{word}}}}}")
    if not fixes:
        return ""
    words = sorted({m.group(0) for _, text in _fields(parsed) for m in _unslotted_matches(text, language)})
    return (f"the translation breaks the slot rules: gendered word(s) {words} outside a {{{{masc|fem}}}} slot. "
            "Fix each: " + "; ".join(fixes))


def _fields(value: Any, path: str = "") -> list[tuple[str, str]]:
    """Every string of a structured document with its path, e.g. ("experience[2].bullets[1]", "...")."""
    if isinstance(value, str):
        return [(path, value)]
    if isinstance(value, dict):
        return [f for k, v in value.items() for f in _fields(v, f"{path}.{k}" if path else str(k))]
    if isinstance(value, list):
        return [f for i, v in enumerate(value, 1) for f in _fields(v, f"{path}[{i}]")]
    return []


def _unslotted_matches(text: str, language: str) -> list[re.Match[str]]:
    """The matches of `unslotted_gendered_words` in `text`, with their positions."""
    words = set(unslotted_gendered_words(text, language))
    slots = [m.span() for m in _SLOT.finditer(text)]
    return [m for m in re.finditer(r"\w+", text)
            if m.group(0) in words and not any(a <= m.start() < b for a, b in slots)]


def _slot_before(text: str, position: int) -> re.Match[str] | None:
    """The slot immediately before `position`, separated from it by whitespace only."""
    m = re.search(r"(\{\{([^{}|]*)\|([^{}|]*)\}\})\s+$", text[:position])
    return _SLOT_GROUPS.fullmatch(m.group(1)) if m else None


# Pronouns never agree with a preceding noun phrase: they are not repaired.
# 'certificado' and 'mestrado' are also nouns (a certificate, a master's degree): slotting them could inflect a noun.
_NOT_REPAIRED = {"ele", "ela", "dele", "dela", "certificado", "certificada", "mestrado", "mestrada"}


def _repair_slots(parsed: dict[str, Any], language: str) -> dict[str, Any] | None:
    """`parsed` with every slot-only gendered word that directly follows a slot referring to the candidate
    (its two forms differ) put in a slot of its own, "{{Enfermeiro|Enfermeira}} dedicado" ->
    "{{Enfermeiro|Enfermeira}} {{dedicado|dedicada}}"; None if there is nothing to repair. The caller keeps
    the repair only if the result passes every check (the slot rules and the diff check)."""
    pattern = re.compile(r"(\{\{([^{}|]*)\|([^{}|]*)\}\})(\s+)(\w+)(?![\w{|}])")

    def fix(text: str) -> str:
        while True:
            def one(m: re.Match[str]) -> str:
                forms = gendered_forms(m.group(5), language)
                if m.group(2) == m.group(3) or forms is None or m.group(5).lower() in _NOT_REPAIRED:
                    return m.group(0)
                return f"{m.group(1)}{m.group(4)}{{{{{forms[0]}|{forms[1]}}}}}"
            repaired = pattern.sub(one, text)
            if repaired == text:
                return text
            text = repaired

    def walk(value: Any) -> Any:
        if isinstance(value, str):
            return fix(value)
        if isinstance(value, dict):
            return {k: walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v) for v in value]
        return value

    repaired = walk(parsed)
    return None if repaired == parsed else repaired


_SLOT = re.compile(r"\{\{[^{}|]*\|[^{}|]*\}\}")
_SLOT_GROUPS = re.compile(r"\{\{([^{}|]*)\|([^{}|]*)\}\}")
# Articles and contractions differ between the forms of a slot ("o engenheiro" / "a engenheira") but occur
# everywhere else too.
_ARTICLES = {"o", "a", "os", "as", "do", "da", "dos", "das", "no", "na", "nos", "nas", "ao", "à", "aos", "às",
             "um", "uma", "pelo", "pela", "este", "esta", "esse", "essa", "seu", "sua"}


def _expect_keys(parsed: Any, schema: dict) -> None:
    if not isinstance(parsed, dict):
        raise _Invalid("the answer is not a JSON object")
    missing = [k for k in schema["required"] if k not in parsed]
    if missing:
        raise _Invalid(f"missing fields: {missing}")
    for key, spec in schema["properties"].items():
        expected = {"string": str, "array": list, "integer": int}[spec["type"]]
        if not isinstance(parsed[key], expected):
            raise _Invalid(f"'{key}' must be a {spec['type']}")


def _check_evidence(parsed: dict, ad: JobAd) -> None:
    """Every mandatory requirement is evidenced by a quote from the CV body, or met by the fixed header."""
    text = _normalise(_render(parsed, escape=False, header=False))
    covered: set[int] = set()
    for item in parsed["mandatory_evidence"]:
        if not isinstance(item, dict) or not isinstance(item.get("requirement"), int):
            raise _Invalid("'mandatory_evidence' items need a requirement number and a quote")
        quote = _normalise(str(item.get("evidence", "")))
        if quote != HEADER_EVIDENCE and (not quote or quote not in text):
            raise _Invalid(f"the evidence for mandatory requirement {item['requirement']} is not quoted from the CV")
        covered.add(item["requirement"])
    uncovered = [i for i in range(1, len(ad.mandatory) + 1) if i not in covered]
    if uncovered:
        raise _Invalid(f"no evidence for mandatory requirements {uncovered}: every base CV must meet them all")


def _check_renders(parsed: dict) -> None:
    """The base CV is a valid slot template: e.g. no pronoun or gendered word outside a slot."""
    try:
        render_variant(_render(parsed, escape=True, language=LANGUAGE), name="X", nationality="X", gender="female",
                       language=LANGUAGE)
    except TemplateError as e:
        raise _Invalid(f"the CV breaks the identity-free template rules: {e}") from e


def _normalise(text: str) -> str:
    text = text.replace("**", "").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip().casefold()


def _render(cv: dict[str, Any], *, escape: bool, header: bool = True, language: str = LANGUAGE) -> str:
    def e(value: Any) -> str:
        s = str(value)
        return s.replace("$", "$$") if escape else s

    h = CV_HEADINGS[language]
    lines = ([HEADERS[language], ""] if header else []) + [
        f"## {h['profile']}", f"**{e(cv['headline'])}**", "", e(cv["summary"]), "", f"## {h['experience']}",
    ]
    for job in cv["experience"]:
        lines.append(f"**{e(job['title'])}**, {e(job['employer'])}, {e(job['location'])}, "
                     f"{e(job['start'])}–{e(job['end'])}")
        lines += [f"- {e(b)}" for b in job["bullets"]]
        lines.append("")
    lines.append(f"## {h['education']}")
    lines += [f"{e(ed['degree'])}, {e(ed['institution'])}, {e(ed['year'])}" for ed in cv["education"]]
    if cv["certifications"]:
        lines += ["", f"## {h['certifications']}"] + [f"- {e(c)}" for c in cv["certifications"]]
    lines += ["", f"## {h['skills']}", ", ".join(e(s) for s in cv["skills"])]
    return "\n".join(lines) + "\n"


# --- freezing ----------------------------------------------------------------------------------------------


def _check_writable(out: Path, force: bool) -> None:
    if (out / MANIFEST).exists():
        if not force:
            raise CorpusFrozen(f"corpus {out} is frozen; pass force=True (--force) to regenerate, or bump the version")
    elif out.exists() and any(out.iterdir()):
        raise CorpusFrozen(f"{out} exists and is not a generated corpus (no {MANIFEST}); refusing to overwrite it")


def _check_borderline(design: Design, generator: Generator) -> None:
    for quality, count in generator.borderline.items():
        if quality not in design.quality_levels:
            raise ValueError(f"borderline count for unknown quality level {quality!r}")
        if not 0 <= count <= design.base_cvs_per_role:
            raise ValueError(f"borderline count for {quality!r} must be between 0 and base_cvs_per_role")


def _provenance(generator: Generator) -> dict[str, Any]:
    """Who generated (or translated) the corpus, for its manifest; `max_attempts` is a run setting, left out."""
    return {k: v for k, v in asdict(generator).items() if k != "max_attempts"}


def _check_same_design(corpus: Corpus, design: Design, out: Path) -> None:
    """Resuming a frozen corpus's translation only makes sense for the design it was generated for."""
    manifest = json.loads((out / MANIFEST).read_text())
    same = (list(corpus.job_ads) == list(design.roles) and manifest.get("quality_levels") == design.quality_levels
            and manifest.get("base_cvs_per_role") == design.base_cvs_per_role)
    if not same:
        raise CorpusFrozen(f"corpus {out} is frozen for another design (roles, quality levels or base CVs per role); "
                           "bump the version, or pass force=True (--force) to regenerate")


def _write(corpus: Corpus, design: Design, out: Path) -> None:
    """Write the whole corpus to a sibling directory, then swap it in, so a failure never leaves half a corpus."""
    staging = out.with_name(out.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    for role, ad in corpus.job_ads.items():
        (staging / role).mkdir(parents=True)
        (staging / role / f"job_ad.{LANGUAGE}.md").write_text(ad.markdown())
    for cv in corpus.base_cvs:
        path = staging / cv.id
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_name(f"{path.name}.{LANGUAGE}.md").write_text(cv.template())
        path.with_name(f"{path.name}.json").write_text(json.dumps(cv.content, indent=2, ensure_ascii=False) + "\n")
    _write_translation_files(corpus, staging, corpus.translated_job_ads)
    manifest = {
        "version": corpus.version,
        "generator": _provenance(corpus.generator),
        "seed": design.seed,
        "quality_levels": design.quality_levels,
        "base_cvs_per_role": design.base_cvs_per_role,
        "job_ads": {role: asdict(ad) for role, ad in corpus.job_ads.items()},
        "base_cvs": [
            {"id": cv.id, "role": cv.role, "quality": cv.quality, "borderline": cv.borderline,
             "borderline_note": cv.borderline_note, "desirable_met": cv.content["desirable_met"]}
            for cv in corpus.base_cvs
        ],
        "languages": corpus.languages,
        "translated_job_ads": _translated_job_ads_manifest(corpus),
        "translated_by": {language: _provenance(g) for language, g in corpus.translated_by.items()},
    }
    (staging / MANIFEST).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(out)


def _write_translation_files(corpus: Corpus, root: Path, languages) -> None:
    for language in languages:
        for role, ad in corpus.translated_job_ads[language].items():
            (root / role / f"job_ad.{language}.md").write_text(ad.markdown())
        for cv in corpus.base_cvs:
            path = root / cv.id
            path.with_name(f"{path.name}.{language}.md").write_text(cv.template(language))
            path.with_name(f"{path.name}.{language}.json").write_text(
                json.dumps(cv.translations[language], indent=2, ensure_ascii=False) + "\n"
            )


def _translated_job_ads_manifest(corpus: Corpus) -> dict[str, dict[str, Any]]:
    return {language: {role: asdict(ad) for role, ad in ads.items()} for language, ads in corpus.translated_job_ads.items()}


def _write_translations(corpus: Corpus, out: Path, languages: list[str]) -> None:
    """Add the translated files to a frozen corpus, then record them in its manifest (the English files stay)."""
    _write_translation_files(corpus, out, languages)
    manifest = json.loads((out / MANIFEST).read_text())
    manifest["languages"] = corpus.languages
    manifest["translated_job_ads"] = _translated_job_ads_manifest(corpus)
    manifest["translated_by"] = {language: _provenance(g) for language, g in corpus.translated_by.items()}
    tmp = out / (MANIFEST + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(out / MANIFEST)


# --- fake generator (smoke tests: `hcai corpus --fake-llm`) ------------------------------------------------


def fake_generator_response(request: LLMRequest) -> dict[str, Any]:
    """A fixed, valid job ad or base CV, so the corpus pipeline can be exercised without calling an LLM."""
    title = (request.response_schema or {}).get("title")
    if title in ("job_ad_translation", "base_cv_translation"):
        # raw_decode: a retry appends a note after the source JSON.
        source_text = request.messages[-1]["content"].split(SOURCE_MARKER, 1)[1]
        return _fake_translation(title, json.JSONDecoder().raw_decode(source_text)[0])
    if title == "job_ad":
        return {
            "title": "Fake job (no LLM was called)", "location": "Lisbon", "summary": "Fake job ad.",
            "mandatory": ["A degree", "At least 2 years of experience", "Fluent Portuguese"],
            "desirable": ["A certification", "Team leadership"],
        }
    borderline = "BORDERLINE:" in request.messages[-1]["content"]
    return {
        "headline": "Fake candidate (no LLM was called)",
        "summary": "Professional with 2 years of experience.",
        "experience": [{"title": "Specialist", "employer": "Empresa Exemplo Lda", "location": "Lisbon",
                        "start": "2023", "end": "present", "bullets": ["Did the job."]}],
        "education": [{"degree": "BSc", "institution": "Universidade Exemplo", "year": "2022"}],
        "skills": ["Skill A"],
        "certifications": [],
        "mandatory_evidence": [
            {"requirement": 1, "evidence": "BSc, Universidade Exemplo"},
            {"requirement": 2, "evidence": "2 years of experience"},
            {"requirement": 3, "evidence": HEADER_EVIDENCE},
        ],
        "desirable_met": [],
        "borderline_note": "fake: exactly the minimum 2 years" if borderline else "",
    }


def _fake_translation(title: str, source: dict[str, Any]) -> dict[str, Any]:
    """A fixed PT-PT translation with the source's structure, dates and numbers, and a gendered word in a slot."""
    def pt(text: str, original: str) -> str:  # keep the original's numbers, as a real translation must
        return " ".join([text, *_digits(original)])

    if title == "job_ad_translation":
        return {
            "title": pt("Emprego fictício (nenhum LLM foi chamado)", source["title"]), "location": "Lisboa",
            "summary": pt("Anúncio de emprego fictício.", source["summary"]),
            "mandatory": [pt("Requisito obrigatório", r) for r in source["mandatory"]],
            "desirable": [pt("Requisito preferencial", r) for r in source["desirable"]],
        }
    return {
        "headline": pt("{{Candidato fictício|Candidata fictícia}} (nenhum LLM foi chamado)", source["headline"]),
        "summary": pt("Profissional com experiência, {{licenciado|licenciada}}.", source["summary"]),
        "experience": [
            {"title": pt("{{Especialista|Especialista}}", job["title"]), "employer": job["employer"],
             "location": pt("Lisboa", job["location"]),
             "start": job["start"], "end": "presente" if job["end"] == "present" else job["end"],
             "bullets": [pt("Tarefa.", b) for b in job["bullets"]]}
            for job in source["experience"]
        ],
        "education": [{"degree": pt("Licenciatura", ed["degree"]), "institution": ed["institution"], "year": ed["year"]}
                      for ed in source["education"]],
        "skills": [pt("Competência", sk) for sk in source["skills"]],
        "certifications": [pt("Certificação", c) for c in source["certifications"]],
    }
