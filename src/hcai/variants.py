"""Identity and language variants of the base CVs, and the demographic-only diff check.

A base CV is a slot template, one file per language (`<corpus>/<role>/<quality>/<id>.<lang>.md`).
Only these slots change between identities:

- `$name`: the candidate's full name, drawn from the identity's names in the name bank.
- `$nationality`: the nationality value. It must follow the language's nationality label
  ("Nationality: $nationality" in EN, "Nacionalidade: $nationality" in PT-PT).
- `{{masc|fem}}`: a gendered word or phrase (pronouns, PT-PT job titles and adjectives),
  e.g. `{{He|She}}`, `{{Engenheiro|Engenheira}}`, `{{licenciado|licenciada}}`. Each must be a
  whole word or phrase, not part of a word, so the diff check can mask it.
- `$work_eligibility`: the standard right-to-work and Portuguese-proficiency lines, identical for
  every identity, so origin is not confounded with legal or language requirements.

Everything else is the CV body, which must be byte-identical across the identities of a base CV
and language. `check_variants` verifies that on the rendered texts.
"""

from __future__ import annotations

import csv
import difflib
import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from string import Template

from hcai.config import Design, Identity
from hcai.ptpt import describe, ptbr_terms

NATIONALITY_LABEL = {"en": "Nationality", "pt": "Nacionalidade"}

WORK_ELIGIBILITY = {
    "en": (
        "Right to work in Portugal: yes (no visa sponsorship required)",
        "Portuguese: native-level proficiency (C2)",
    ),
    "pt": (
        "Autorização para trabalhar em Portugal: sim (não necessita de visto)",
        "Português: proficiência de nível nativo (C2)",
    ),
}

# English pronouns are always masked by the diff check, also when a template forgot the alternation.
PRONOUN_PAIRS = [
    ("he", "she"), ("He", "She"), ("him", "her"), ("his", "her"), ("His", "Her"),
    ("himself", "herself"), ("Mr", "Ms"),
]

# Gendered words that a template may only use inside a `{{masc|fem}}` slot (matched case-insensitively,
# as whole words; all-caps tokens such as "MS Excel" or a hospital "HIS" are acronyms and exempt), so no
# masculine or feminine form leaks into the other gender's variants. A word that does not refer to the
# candidate (e.g. their manager's title) is marked invariant as `{{X|X}}`. The PT-PT list is not
# exhaustive: it catches the common job titles and adjectives, template authors still slot the rest.
# PT-PT templates must also be free of common PT-BR terms (`hcai.ptpt`): no dialect cue in the body.
# The PT-PT (masculine, feminine) pairs whose forms may only appear inside a slot.
GENDERED_WORD_PAIRS: dict[str, list[tuple[str, str]]] = {
    "pt": [
        ("engenheiro", "engenheira"), ("enfermeiro", "enfermeira"), ("técnico", "técnica"),
        ("diretor", "diretora"), ("coordenador", "coordenadora"), ("gestor", "gestora"),
        ("programador", "programadora"), ("auditor", "auditora"), ("consultor", "consultora"),
        ("supervisor", "supervisora"), ("colaborador", "colaboradora"),
        ("licenciado", "licenciada"), ("mestrado", "mestrada"), ("doutorado", "doutorada"),
        ("formado", "formada"), ("certificado", "certificada"), ("dedicado", "dedicada"),
        ("motivado", "motivada"), ("organizado", "organizada"), ("proativo", "proativa"),
        ("empenhado", "empenhada"), ("ele", "ela"), ("dele", "dela"),
        ("estagiário", "estagiária"), ("voluntário", "voluntária"), ("bolseiro", "bolseira"),
        ("investigador", "investigadora"), ("administrador", "administradora"), ("formador", "formadora"),
        ("professor", "professora"), ("mestre", "mestra"), ("director", "directora"),
        ("inscrito", "inscrita"), ("especializado", "especializada"),
        ("habilitado", "habilitada"), ("promovido", "promovida"),
        ("contratado", "contratada"), ("nomeado", "nomeada"), ("eleito", "eleita"),
        ("distinguido", "distinguida"), ("premiado", "premiada"), ("focado", "focada"),
        ("rigoroso", "rigorosa"), ("meticuloso", "meticulosa"),
        ("revisor", "revisora"), ("tesoureiro", "tesoureira"), ("fundador", "fundadora"),
        ("orientador", "orientadora"), ("candidato", "candidata"),
    ],
}
SLOT_ONLY_WORDS = {
    "en": {"he", "she", "him", "her", "his", "hers", "himself", "herself", "mr", "ms", "mrs"},
    "pt": {form for pair in GENDERED_WORD_PAIRS["pt"] for form in pair},
}

# Noun agreement (docs/adr/0007). A slot-only adjective or participle may stay outside a slot when it plainly
# agrees with an adjacent common noun that is not the candidate: "Registo rigoroso de avaliações",
# "rigoroso controlo de stocks". It is exempt only if all of these hold:
#   - the word is one of AGREEING_WORDS (adjectives/participles that also qualify things; never job titles,
#     degrees or pronouns);
#   - the noun right before or right after it, separated by one space only (no punctuation, no line break,
#     no slot), is in AGREEMENT_NOUNS with the word's gender (all forms on the list are singular);
#   - its clause (since the last , ; : . ! ? or line break) has no candidate slot (`{{masc|fem}}` whose forms
#     differ) and no word that refers to a person (CANDIDATE_WORDS, or any job title of the list) before it.
# Anything else ("Profissional rigoroso", "{{Enfermeiro|Enfermeira}} dedicado", "Rigoroso e dedicado, ...")
# still needs a slot. Coordinated adjectives ("controlo rigoroso e organizado") are not exempt for the second one.
AGREEING_WORDS: dict[str, set[str]] = {
    "pt": {form for pair in [
        ("rigoroso", "rigorosa"), ("meticuloso", "meticulosa"), ("dedicado", "dedicada"),
        ("organizado", "organizada"), ("especializado", "especializada"), ("focado", "focada"),
        ("proativo", "proativa"), ("certificado", "certificada"), ("premiado", "premiada"),
        ("distinguido", "distinguida"), ("promovido", "promovida"), ("contratado", "contratada"),
    ] for form in pair},
}
# Singular common nouns of CV text that never denote a person, by gender ("m"/"f").
AGREEMENT_NOUNS: dict[str, dict[str, str]] = {
    "pt": {
        **dict.fromkeys([
            "registo", "controlo", "acompanhamento", "cumprimento", "planeamento", "atendimento", "cuidado",
            "tratamento", "procedimento", "protocolo", "processo", "projeto", "trabalho", "serviço", "sistema",
            "plano", "programa", "relatório", "estudo", "ensino", "apoio", "curso", "desenvolvimento",
            "levantamento", "inventário", "código", "produto", "documento", "ensaio", "teste", "modelo",
            "método", "seguimento", "rastreio", "diagnóstico", "orçamento", "suporte", "fecho", "reporte",
            "arquivo", "circuito", "ciclo", "fluxo", "lançamento", "pagamento", "balanço", "cálculo",
        ], "m"),
        **dict.fromkeys([
            "gestão", "avaliação", "análise", "monitorização", "documentação", "formação", "abordagem",
            "revisão", "auditoria", "verificação", "prática", "triagem", "administração", "preparação",
            "vigilância", "atenção", "comunicação", "organização", "execução", "implementação", "manutenção",
            "aplicação", "solução", "plataforma", "ferramenta", "prestação", "higiene", "investigação",
            "intervenção", "campanha", "iniciativa", "consulta", "medicação", "prescrição", "estratégia",
            "política", "metodologia", "reconciliação", "conciliação", "contabilidade", "faturação",
            "previsão", "otimização", "migração", "integração",
        ], "f"),
    },
}
# Gender-invariant words that refer to a person: after one, an adjective may describe the candidate.
CANDIDATE_WORDS: dict[str, set[str]] = {
    "pt": {"profissional", "pessoa", "membro", "elemento", "especialista", "responsável", "líder", "perfil",
           "colega", "analista", "assistente", "gerente", "chefe", "jovem", "auxiliar", "farmacêutico",
           "farmacêutica", "contabilista", "economista", "sou", "estou", "fui"},
}
_AGREEMENT_FORMS = {"pt": {masc: "m" for masc, _ in GENDERED_WORD_PAIRS["pt"]}
                    | {fem: "f" for _, fem in GENDERED_WORD_PAIRS["pt"]}}
_CLAUSE_BREAK = re.compile(r"[,;:.!?\n]")

# Articles and contractions are too short to mask or leak-check reliably ("a" and "o" occur everywhere):
# a slot must hold the whole gendered phrase, e.g. `{{o engenheiro|a engenheira}}`.
_TOO_SHORT_FORMS = {"o", "a", "os", "as", "do", "da", "dos", "das", "no", "na", "nos", "nas", "ao", "à", "aos", "às",
                    "um", "uma", "pelo", "pela"}

_ALTERNATION = re.compile(r"\{\{([^{}|]*)\|([^{}|]*)\}\}")
_GENDER_INDEX = {"male": 0, "female": 1}


class TemplateError(ValueError):
    """A base-CV template breaks the slot rules (see the module docstring)."""


class IdentityError(ValueError):
    """An identity cannot be rendered: unknown gender, or no nationality in a design language."""


class VariantCheckError(Exception):
    """Variants of a base CV differ beyond demographic cues (see `check_variants`)."""

    def __init__(self, problems: list[str]):
        super().__init__(f"{len(problems)} variant check problem(s):\n" + "\n".join(problems))
        self.problems = problems


@dataclass(frozen=True)
class Variant:
    id: str  # <base_cv>/<identity>/<language>, e.g. "nurse/high/cv01/br_f/pt"
    base_cv: str  # <role>/<quality>/<cv id>
    identity: str
    gender: str
    origin: str
    name: str
    nationality: str
    language: str
    text: str

    @property
    def role(self) -> str:
        return self.base_cv.split("/")[0]

    @property
    def quality(self) -> str:
        return self.base_cv.split("/")[1]


def pick_name(identity: Identity, base_cv: str) -> str:
    """The identity's name for this base CV: stable across runs and languages, spread over the names."""
    digest = hashlib.sha256(f"{base_cv}|{identity.id}".encode()).digest()
    return identity.names[int.from_bytes(digest[:8], "big") % len(identity.names)]


def gendered_pairs(template: str) -> list[tuple[str, str]]:
    """The `{{masc|fem}}` alternations of a template, in order of first appearance."""
    return list(dict.fromkeys(_ALTERNATION.findall(template)))


def unslotted_gendered_words(text: str, language: str) -> list[str]:
    """The slot-only gendered words of `text` written outside a `{{masc|fem}}` slot, as written, in order
    (all-caps tokens such as "MS Excel" are acronyms and exempt, and so is an adjective that agrees with an
    adjacent non-candidate noun, see AGREEING_WORDS)."""
    return [m.group(0) for m in unslotted_gendered_matches(text, language)]


def unslotted_gendered_matches(text: str, language: str) -> list[re.Match[str]]:
    """The matches (with positions in `text`) of `unslotted_gendered_words`."""
    tokens = _unslotted_tokens(text)
    return [
        m for i, m in enumerate(tokens)
        if m.group(0).lower() in SLOT_ONLY_WORDS[language] and not (len(m.group(0)) > 1 and m.group(0).isupper())
        and not _agrees_with_a_noun(text, tokens, i, language)
    ]


def noun_agreeing_matches(text: str, language: str) -> list[re.Match[str]]:
    """The slot-only words of `text` outside a slot that are exempt because they agree with an adjacent
    non-candidate noun ("Registo rigoroso"), with their positions."""
    tokens = _unslotted_tokens(text)
    return [m for i, m in enumerate(tokens) if _agrees_with_a_noun(text, tokens, i, language)]


def _unslotted_tokens(text: str) -> list[re.Match[str]]:
    slots = [m.span() for m in _ALTERNATION.finditer(text)]
    return [m for m in re.finditer(r"\w+", text) if not any(a <= m.start() < b for a, b in slots)]


def _agrees_with_a_noun(text: str, tokens: list[re.Match[str]], i: int, language: str) -> bool:
    """Whether `tokens[i]` is an AGREEING_WORDS adjective that agrees with an adjacent non-candidate noun."""
    word = tokens[i].group(0).lower()
    if word not in AGREEING_WORDS.get(language, set()) or tokens[i].group(0).isupper():
        return False
    gender = _AGREEMENT_FORMS[language][word]
    nouns = AGREEMENT_NOUNS[language]
    neighbours = []
    if i > 0 and text[tokens[i - 1].end(): tokens[i].start()] == " ":
        neighbours.append(tokens[i - 1].group(0).lower())
    if i + 1 < len(tokens) and text[tokens[i].end(): tokens[i + 1].start()] == " ":
        neighbours.append(tokens[i + 1].group(0).lower())
    if not any(nouns.get(n) == gender for n in neighbours):
        return False
    # The clause before the word must not name the candidate: no candidate slot, no person word.
    clause_start = max((m.end() for m in _CLAUSE_BREAK.finditer(text, 0, tokens[i].start())), default=0)
    clause = text[clause_start: tokens[i].start()]
    if any(m.group(1) != m.group(2) for m in _ALTERNATION.finditer(clause)):
        return False
    persons = CANDIDATE_WORDS[language] | {f for pair in GENDERED_WORD_PAIRS[language] for f in pair} - AGREEING_WORDS[language]
    return not any(w.lower() in persons for w in re.findall(r"\w+", _ALTERNATION.sub(" ", clause)))


def gendered_forms(word: str, language: str) -> tuple[str, str] | None:
    """The (masculine, feminine) forms of a slot-only `word` (either form), in its capitalisation, else None."""
    for masc, fem in GENDERED_WORD_PAIRS.get(language, []):
        if word.lower() in (masc, fem):
            if word.isupper():
                return masc.upper(), fem.upper()
            if word[:1].isupper():
                return masc.capitalize(), fem.capitalize()
            return masc, fem
    return None


def render_variant(template: str, *, name: str, nationality: str, gender: str, language: str) -> str:
    """Fill a base-CV template's demographic slots for one identity in one language."""
    _validate_template(template, language)
    text = _ALTERNATION.sub(lambda m: m.group(1 + _GENDER_INDEX[gender]), template)
    return Template(text).substitute(
        name=name, nationality=nationality, work_eligibility="\n".join(WORK_ELIGIBILITY[language]),
    )


def _validate_template(template: str, language: str) -> None:
    if language not in WORK_ELIGIBILITY:
        raise TemplateError(f"unsupported language {language!r}; expected one of {sorted(WORK_ELIGIBILITY)}")
    label = NATIONALITY_LABEL[language]
    if template.count("$nationality") != template.count(f"{label}: $nationality"):
        raise TemplateError(f"$nationality must follow the label '{label}: ' in {language} templates")
    for m in _ALTERNATION.finditer(template):
        before, after = template[m.start() - 1: m.start()], template[m.end(): m.end() + 1]
        if before.isalnum() or after.isalnum():
            raise TemplateError(f"gendered alternation {m.group(0)!r} must be a whole word, not part of one")
        if any(form.strip().lower() in _TOO_SHORT_FORMS for form in m.groups()):
            raise TemplateError(
                f"gendered alternation {m.group(0)!r} is only an article; slot the whole phrase, e.g. {{{{o engenheiro|a engenheira}}}}"
            )
    unslotted = sorted(set(unslotted_gendered_words(template, language)))
    if unslotted:
        raise TemplateError(
            f"gendered word(s) {unslotted} outside a {{{{masc|fem}}}} slot; write {{{{X|X}}}} if one does not refer to the candidate"
        )
    if language == "pt" and (brazilian := ptbr_terms(template)):
        raise TemplateError(f"PT-PT template uses PT-BR term(s): {describe(brazilian)}")


def base_cv_templates(design: Design, role: str, quality: str, language: str) -> list[tuple[str, str]]:
    """(base CV id, template) for the first `base_cvs_per_role` base CVs of a role x quality in a language."""
    folder = design.corpus_dir / role / quality
    files = sorted(folder.glob(f"*.{language}.md"))[: design.base_cvs_per_role]
    return [(f"{role}/{quality}/{f.name.removesuffix(f'.{language}.md')}", f.read_text()) for f in files]


def build_variants(design: Design) -> list[Variant]:
    """Render every base CV x identity x language of the design, then check the variants.

    Every base CV must exist in every language (else `FileNotFoundError`), every identity must be male or
    female and have a nationality in every language (else `IdentityError`). Raises `VariantCheckError` if
    variants of a base CV differ in anything but demographic cues.
    """
    _check_identities(design)
    variants: list[Variant] = []
    problems: list[str] = []
    for role in design.roles:
        for quality in design.quality_levels:
            per_language = {
                language: [b for b, _ in base_cv_templates(design, role, quality, language)]
                for language in design.languages
            }
            for language, base_cvs in per_language.items():
                extra = sorted(set(base_cvs) - set(per_language[design.languages[0]]))
                if extra:
                    raise FileNotFoundError(f"base CV(s) {extra} have a {language} version but no {design.languages[0]} one")
            for base_cv in per_language[design.languages[0]]:
                for language in design.languages:
                    template = _read_template(design, base_cv, language)
                    group = [_render(template, base_cv, identity, language) for identity in design.identities]
                    problems += check_variants(group, gendered_pairs(template))
                    variants += group
    if problems:
        raise VariantCheckError(problems)
    return variants


def _check_identities(design: Design) -> None:
    for identity in design.identities:
        if identity.gender not in _GENDER_INDEX:
            raise IdentityError(f"identity {identity.id}: gender must be one of {sorted(_GENDER_INDEX)}, not {identity.gender!r}")
        for language in design.languages:
            try:
                identity.nationality_in(language)
            except ValueError as e:
                raise IdentityError(str(e)) from e


def _read_template(design: Design, base_cv: str, language: str) -> str:
    path = design.corpus_dir / f"{base_cv}.{language}.md"
    if not path.exists():
        raise FileNotFoundError(f"base CV {base_cv} has no {language} version at {path}")
    return path.read_text()


def _render(template: str, base_cv: str, identity: Identity, language: str) -> Variant:
    name, nationality = pick_name(identity, base_cv), identity.nationality_in(language)
    try:
        text = render_variant(template, name=name, nationality=nationality, gender=identity.gender, language=language)
    except TemplateError as e:
        raise TemplateError(f"{base_cv}.{language}: {e}") from e
    return Variant(
        id=f"{base_cv}/{identity.id}/{language}", base_cv=base_cv, identity=identity.id,
        gender=identity.gender, origin=identity.origin, name=name, nationality=nationality, language=language,
        text=text,
    )


def check_variants(variants: Sequence[Variant], gendered: Iterable[tuple[str, str]] = ()) -> list[str]:
    """Problems with variants of one base CV: missing work-eligibility lines, or non-demographic differences.

    Variants are compared within each (base CV, language) group after masking the demographic cues:
    the name, the nationality value after its label, and both forms of every gendered pair (the
    template's `gendered` alternations plus English pronouns). Each variant must also not contain the
    other gender's form of a `gendered` pair, or in English of a pronoun. An empty list means the check passed.
    """
    gendered = list(gendered)
    classes = _form_classes([*gendered, *PRONOUN_PAIRS])
    invariant = sorted({m for m, f in gendered if m == f and m}, key=len, reverse=True)
    other_forms = {"en": _other_gender_forms([*gendered, *PRONOUN_PAIRS]), "pt": _other_gender_forms(gendered)}
    problems: list[str] = []
    groups: dict[tuple[str, str], list[Variant]] = {}
    for v in variants:
        groups.setdefault((v.base_cv, v.language), []).append(v)
        missing = [line for line in WORK_ELIGIBILITY[v.language] if line not in v.text.splitlines()]
        if missing:
            problems.append(f"{v.id}: missing work-eligibility line(s): {missing}")
        # An invariant `{{X|X}}` phrase may contain the other gender's form ("Enfermeira-Chefe" next to
        # {{Enfermeiro|Enfermeira}}): it is the same in every variant, so it is not a leak.
        text = v.text
        for form in invariant:
            text = re.sub(rf"(?<!\w){re.escape(form)}(?!\w)", "\0", text)
        leaked = [f for f in other_forms[v.language][v.gender] if re.search(rf"(?<!\w){re.escape(f)}(?!\w)", text)]
        if leaked:
            problems.append(f"{v.id}: {v.gender} variant has the other gender's form(s): {leaked}")
    for group in groups.values():
        reference, ref_masked = group[0], _mask(group[0], classes)
        for v in group[1:]:
            diff = list(difflib.unified_diff(ref_masked, _mask(v, classes), reference.id, v.id, lineterm="", n=0))
            if diff:
                problems.append(f"{v.id}: differs from {reference.id} beyond demographic cues:\n" + "\n".join(diff))
    return problems


def _mask(v: Variant, classes: list[list[str]]) -> list[str]:
    label = NATIONALITY_LABEL[v.language]
    text = v.text.replace(v.name, "\0NAME\0").replace(f"{label}: {v.nationality}", f"{label}: \0NATIONALITY\0")
    for i, forms in enumerate(classes):
        alternatives = "|".join(re.escape(f) for f in sorted(forms, key=len, reverse=True))
        text = re.sub(rf"(?<!\w)(?:{alternatives})(?!\w)", f"\0G{i}\0", text)
    return [line.rstrip() for line in text.strip().splitlines()]


def _other_gender_forms(pairs: list[tuple[str, str]]) -> dict[str, list[str]]:
    """Per gender, the forms only the other gender uses (forms also used by that gender, as in an
    invariant `{{X|X}}` pair or the shared "her" of him/her and his/her, are excluded)."""
    masc, fem = {m for m, _ in pairs}, {f for _, f in pairs}
    return {"male": sorted(fem - masc), "female": sorted(masc - fem)}


def _form_classes(pairs: Iterable[tuple[str, str]]) -> list[list[str]]:
    """Merge gendered pairs that share a form (him/her, his/her) so all their forms mask to one token."""
    classes: list[set[str]] = []
    for pair in pairs:
        forms = {f for f in pair if f}
        overlapping = [c for c in classes if c & forms]
        for c in overlapping:
            forms |= c
            classes.remove(c)
        classes.append(forms)
    return [sorted(c) for c in classes]


def write_variants(variants: Sequence[Variant], out_dir: str | Path) -> Path:
    """Write each variant to `<out>/<base_cv>/<identity>.<language>.md` and a `variants.csv` manifest.

    Variant files listed in a previous manifest at `out_dir` are removed first, so no stale variant
    outlives a smaller design.
    """
    out = Path(out_dir)
    previous = out / "variants.csv"
    if previous.exists():
        with previous.open(newline="") as f:
            for row in csv.DictReader(f):
                (out / row["path"]).unlink(missing_ok=True)
    for v in variants:
        path = out / v.base_cv / f"{v.identity}.{v.language}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(v.text)
    manifest = out / "variants.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="") as f:
        columns = [f.name for f in fields(Variant) if f.name != "text"]
        writer = csv.DictWriter(f, fieldnames=[*columns, "path"])
        writer.writeheader()
        for v in variants:
            writer.writerow({c: getattr(v, c) for c in columns} | {"path": f"{v.base_cv}/{v.identity}.{v.language}.md"})
    return manifest
