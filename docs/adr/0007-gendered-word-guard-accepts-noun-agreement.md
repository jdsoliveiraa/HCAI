# 7. The gendered-word guard accepts an adjective that agrees with an adjacent non-candidate noun

Date: 2026-11-15 · Status: accepted (#11 Part B, #14); refines the slot rules of `hcai.variants` and ADR 0006

## Context

The PT-PT guard (`GENDERED_WORD_PAIRS`) rejects every listed gendered word outside a `{{masc|fem}}` slot. Some of
those words are adjectives or participles that also qualify things, and then they agree with that noun, not with
the candidate. The generator wrote, for nurse/strong/cv01, "Registo rigoroso de avaliações iniciais de
enfermagem": `rigoroso` agrees with `Registo` (masculine) and is correct in both variants. The guard demanded
`{{rigoroso|rigoroso}}`, the model never wrote that, and the corpus could never complete.

## Decision

A slot-only word outside a slot is accepted as written when all of these hold (`hcai.variants`,
`AGREEING_WORDS`, `AGREEMENT_NOUNS`, `CANDIDATE_WORDS`):

1. It is one of a small list of adjectives/participles that also qualify things (`rigoroso`, `meticuloso`,
   `dedicado`, `organizado`, `especializado`, `focado`, `proativo`, `certificado`, `premiado`, `distinguido`,
   `promovido`, `contratado`). Job titles, degrees (`licenciado`, `mestrado`, ...), `inscrito`, `empenhado`,
   `motivado` and pronouns never qualify.
2. The word right before or right after it, separated by exactly one space (no punctuation, no line break, no
   slot), is a curated singular common noun that never denotes a person (`registo`, `controlo`,
   `acompanhamento`, `gestão`, `avaliação`, ...) of the same gender. The listed forms are all singular, so number
   agrees too.
3. Its clause (since the last `, ; : . ! ?` or line break) has, before it, no slot whose two forms differ and no
   word that refers to a person: a job title of the guard list or a gender-invariant person word (`profissional`,
   `especialista`, `responsável`, `perfil`, ...).

The template validator, the translation's retry feedback and the "slot word also used outside a slot" check all use
the same rule, so a template the corpus accepts also renders.

## Consequences

- "Registo rigoroso de avaliações", "Rigoroso controlo de stocks", "gestão rigorosa" pass. "Profissional
  rigoroso", "{{Enfermeiro|Enfermeira}} dedicado", "Enfermeira dedicado", "Rigoroso e dedicado, ..." and
  "Registo rigorosa" still fail.
- Conservative by design, so some correct text still needs a `{{X|X}}` slot: a noun outside the list, a second
  coordinated adjective ("controlo rigoroso e organizado"), or a clause that also names a person ("Profissional
  com registo rigoroso"). The retry note still gives the exact fix for these.
- Known limit: the rule reads only the clause up to the word. A self-description split across clauses whose
  adjective happens to sit next to a listed noun of the same gender would pass; with only non-person nouns on the
  list such text agrees grammatically with that noun in both variants, so it is not a gender leak in the rendered
  CVs. Extending the noun list is a code change reviewed against this ADR.
