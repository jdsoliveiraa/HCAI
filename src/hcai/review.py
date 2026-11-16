"""The realism review (spec #1 story 12, ADR 0010): the review sample and the review record's two formats.

The researcher hand-reviews a small sample of the frozen corpus for realism on the project site's review page
(`hcai site`, `hcai.site`). The browser keeps the answers as a draft; the record is the exported Markdown and JSON,
committed to `docs/pilot/`. This module fixes what is reviewed and what the record looks like:

- `review_sample` draws the review sample by a documented seeded rule (below), so it is reproducible.
- `review_markdown` is the Markdown record of a review, in the layout of the pilot's first
  `docs/pilot/cv-realism-review.md`. The review page's JavaScript (`assets/review.js`) writes the same text; a test
  runs both on the same answers and compares them.
- `blank_review` is the JSON record of an empty review; `check_review` says why a JSON record does not belong to a
  sample (another corpus version, seed or set of items), or returns None.

**Sampling rule.** With `rng = random.Random(seed)`:

1. Job ads: every role of the design in every design language, in design order.
2. Origins: the design identities' origins, in order of first appearance, shuffled once by `rng`.
3. CV variants, per role in design order (the k-th role, from 0): one base CV that is not borderline, of quality level
   `quality_levels[k % len(quality_levels)]`, in the first design language, then one borderline base CV, of any
   quality level, in the last design language. Each is `rng.choice` over the candidates, sorted by id. A role with no
   such non-borderline CV takes any non-borderline one (else any); a role without borderline CVs takes another base CV
   instead. With a single language both are in it. So 3 roles cover every quality level and every role a borderline.
4. Identities: the k-th CV variant (counting over all roles, from 0) takes origin `origins[k % len(origins)]`; its
   gender alternates per role and flips from one role to the next: female, male (first role), male, female (second),
   female, male (third), ... (female when `k // 2 + k % 2` is even). So 4 or more CVs cover every origin, the genders
   balance, and with 2 or more roles each language gets both genders (PT-PT gender agreement is reviewed both ways).
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

FORMAT = "hcai-realism-review"
FORMAT_VERSION = 1

# Checklist: key -> the label written in the Markdown record.
CHECKS: dict[str, str] = {
    "realistic": "Realistic",
    "quality": "Quality matches label",
    "mandatory": "Meets mandatory",
    "language": "Language OK",
    "no_stray_cues": "No stray cues",
}
JOB_AD_CHECKS = ("realistic", "language")
CV_CHECKS = tuple(CHECKS)

# What each check means (shown on the review page and written into the Markdown record).
CHECK_GUIDANCE = [
    ("Realistic", "reads like a real CV/ad in the Portuguese market (employers, degrees, career path, dates)."),
    ("Quality matches label", "a strong CV is clearly stronger than a weak one; borderline only just meets the "
                              "mandatory requirements."),
    ("Meets mandatory requirements", "of its job ad (all CVs should)."),
    ("Language", "EN is natural English; PT is natural PT-PT (not PT-BR), gender agreement correct."),
    ("No stray identity cues", "beyond name, nationality, pronouns and gendered words."),
]
VERDICT_LABEL = "Corpus is realistic enough to freeze"


@dataclass(frozen=True)
class SampleCV:
    """A base CV the sample can draw from."""

    id: str  # <role>/<quality>/<cv id>
    role: str
    quality: str
    borderline: bool


@dataclass(frozen=True)
class SampleIdentity:
    id: str
    gender: str
    origin: str


@dataclass(frozen=True)
class ReviewItem:
    """One job ad in one language, or one CV variant, of the review sample."""

    id: str  # "job_ad/<role>/<language>", or the variant id "<base cv>/<identity>/<language>"
    kind: str  # "job_ad" or "cv"
    role: str
    language: str
    base_cv: str = ""
    quality: str = ""
    borderline: bool = False
    identity: str = ""
    gender: str = ""
    origin: str = ""
    name: str = ""  # filled in by the caller, who renders the variant (the sample does not pick names)

    @property
    def checks(self) -> tuple[str, ...]:
        return JOB_AD_CHECKS if self.kind == "job_ad" else CV_CHECKS

    def title(self) -> str:
        if self.kind == "job_ad":
            return f"Job ad: {self.role} ({self.language})"
        label = self.quality + (", borderline" if self.borderline else "")
        return f"CV: {self.base_cv} ({label}) as {self.name} ({self.gender}, {self.origin}), {self.language}"


@dataclass(frozen=True)
class ReviewSample:
    seed: int
    items: list[ReviewItem] = field(default_factory=list)

    @property
    def cvs(self) -> list[ReviewItem]:
        return [i for i in self.items if i.kind == "cv"]


def review_sample(
    roles: Sequence[str], quality_levels: Sequence[str], languages: Sequence[str], job_ad_roles: Sequence[str],
    base_cvs: Sequence[SampleCV], identities: Sequence[SampleIdentity], seed: int,
) -> ReviewSample:
    """The review sample, by the seeded rule of the module docstring.

    `job_ad_roles` are the roles that have a job ad; `base_cvs` and `identities` are what the corpus and the design
    offer. A role without base CVs gets no CV variants; with no identities the sample has job ads only.
    """
    rng = random.Random(seed)
    items = [ReviewItem(id=f"job_ad/{role}/{language}", kind="job_ad", role=role, language=language)
             for role in roles if role in job_ad_roles for language in languages]
    origins = list(dict.fromkeys(i.origin for i in identities))
    rng.shuffle(origins)
    picks: list[tuple[SampleCV, str]] = []
    for k, role in enumerate(roles):
        cvs = sorted((cv for cv in base_cvs if cv.role == role), key=lambda cv: cv.id)
        if not cvs:
            continue
        plain = [cv for cv in cvs if not cv.borderline] or cvs
        level = quality_levels[k % len(quality_levels)] if quality_levels else ""
        first = rng.choice([cv for cv in plain if cv.quality == level] or plain)
        picks.append((first, languages[0]))
        borderline = [cv for cv in cvs if cv.borderline and cv != first] or [cv for cv in cvs if cv != first]
        if borderline:
            picks.append((rng.choice(borderline), languages[-1]))
    if origins:
        for k, (cv, language) in enumerate(picks):
            female = (k // 2 + k % 2) % 2 == 0
            identity = _identity(identities, origins[k % len(origins)], "female" if female else "male")
            items.append(ReviewItem(
                id=f"{cv.id}/{identity.id}/{language}", kind="cv", role=cv.role, language=language, base_cv=cv.id,
                quality=cv.quality, borderline=cv.borderline, identity=identity.id, gender=identity.gender,
                origin=identity.origin,
            ))
    return ReviewSample(seed=seed, items=items)


def _identity(identities: Sequence[SampleIdentity], origin: str, gender: str) -> SampleIdentity:
    of_origin = [i for i in identities if i.origin == origin]
    return next((i for i in of_origin if i.gender == gender), of_origin[0])


# --- the review record ---------------------------------------------------------------------------------------------


def blank_review(meta: dict[str, Any], items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The JSON record of an empty review of `items` (dicts with `id`, `title`, `checks`)."""
    return {
        "format": FORMAT, "format_version": FORMAT_VERSION,
        "corpus_version": meta["corpus_version"], "generator": meta["generator"], "seed": meta["seed"],
        "reviewer": "", "date": "",
        "verdict": {"freeze": False, "notes": ""},
        "items": [{"id": i["id"], "title": i["title"], "checks": {c: False for c in i["checks"]}, "notes": ""}
                  for i in items],
    }


def check_review(review: Any, meta: dict[str, Any], items: Sequence[dict[str, Any]]) -> str | None:
    """Why `review` (a parsed JSON record) is not a review of this sample, or None if it is."""
    if not isinstance(review, dict) or review.get("format") != FORMAT:
        return "not a realism review export"
    if review.get("format_version") != FORMAT_VERSION:
        return f"unsupported format version {review.get('format_version')!r}"
    for key in ("corpus_version", "seed"):
        if review.get(key) != meta[key]:
            return f"it reviews {key.replace('_', ' ')} {review.get(key)!r}, this sample is {meta[key]!r}"
    if not isinstance(review.get("items"), list):
        return "its items differ from this sample's"
    got = [i.get("id") for i in review["items"] if isinstance(i, dict)]
    want = [i["id"] for i in items]
    if sorted(map(str, got)) != sorted(want):
        return "its items differ from this sample's"
    return None


def review_markdown(meta: dict[str, Any], items: Sequence[dict[str, Any]], review: dict[str, Any]) -> str:
    """The Markdown record of `review` (a JSON record) over `items` (dicts with `id`, `kind`, `title`, `checks`,
    `text`). Mirrored line by line by `reviewMarkdown` in the review page's JavaScript."""
    answers = {i["id"]: i for i in review["items"]}
    langs = " + ".join(lang.upper() for lang in meta["languages"])
    cvs = [i for i in items if i["kind"] == "cv"]
    roles = len({i["role"] for i in items if i["kind"] == "job_ad"})
    out = [
        "# CV realism review (pilot, #11 Part B)", "",
        f"Hand review of a stratified sample of the frozen corpus {meta['corpus_version']} (generator "
        f"{meta['generator']}): {roles} job ads ({langs}) and {len(cvs)} CV variants, one per role x language, "
        f"drawn by hcai.review.review_sample with seed {meta['seed']}.", "",
        "For each document, tick or note:", "",
        *[f"- **{label}**: {text}" for label, text in CHECK_GUIDANCE], "",
        f"Reviewer: {review['reviewer'] or '______'}  Date: {review['date'] or '______'}", "",
        "## Overall verdict", "",
        f"- [{_tick(review['verdict']['freeze'])}] {VERDICT_LABEL}",
        _notes("- Notes:", review["verdict"]["notes"]), "", "",
    ]
    for item in items:
        answer = answers[item["id"]]
        boxes = "  ".join(f"- [{_tick(answer['checks'].get(c, False))}] {CHECKS[c]}" for c in item["checks"])
        out += [f"## {item['title']}", ""]
        if item["kind"] == "job_ad":
            out += [_notes(f"{boxes}  - Notes:", answer["notes"])]
        else:
            out += [boxes, _notes("- Notes:", answer["notes"])]
        out += ["", "<details><summary>Text</summary>", "", item["text"], "", "</details>", ""]
    return "\n".join(out)


def _tick(value: Any) -> str:
    return "x" if value is True else " "


def _notes(prefix: str, notes: str) -> str:
    """`prefix` and the notes, continuation lines indented so they stay in the list item."""
    lines = notes.strip().replace("\r\n", "\n").split("\n")
    text = "\n  ".join(line.rstrip() for line in lines)
    return f"{prefix} {text}" if text else prefix
