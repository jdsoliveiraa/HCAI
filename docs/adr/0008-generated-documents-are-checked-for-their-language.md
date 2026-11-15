# 8. Generated documents are checked for their language; a frozen corpus is repaired with --recheck

Date: 2026-11-15 · Status: accepted (#11 Part B); extends ADR 0006

## Context

In the real corpus (gemini-4-argon) the English job ads and base CVs of software_engineer and accountant were
written in European Portuguese ("A Lusitânia Tech Solutions está a recrutar...", "Contabilista Certificado"),
while nurse came out in English. Nothing checked the language, so they were frozen, and the PT-PT
"translation" translated Portuguese into Portuguese: the EN arm (RQ1) was invalid for 2 of 3 roles.

## Decision

- The job ad and base CV prompts say every field is in English; Portuguese proper nouns (employers,
  institutions, cities, the Ordem / OCC names, degree names such as Licenciatura) may stay, glossed in English
  where natural.
- `hcai.language` counts English and Portuguese function words over a document's prose fields (not employer,
  institution, location, dates or evidence quotes). Words common to both languages (`a`, `as`, `no`, `do`,
  `os`, `se`), all-caps tokens and a function word between two capitalised words (a name such as "Ordem dos
  Contabilistas Certificados") are not counted. A field is in the wrong language with at least 3 function
  words of the other language and more of them than of its own; a document is when any field is, or its totals
  are. An English source must not read as Portuguese, a PT-PT translation must not read as English. A failure
  is an ordinary invalid answer (ADR 0002/0006): the retry note names the offending fields.
- A kept translation under `partial/<lang>/` records the hash of the English text it was made from and is
  reused only for that text, so a regenerated source is always translated again.
- `hcai corpus <config> --recheck` re-validates a frozen corpus with today's checks. Valid English documents
  are kept as frozen (no call, no cache needed); a failing job ad is regenerated with all its base CVs (they
  quote its requirements), a failing base CV alone. Translations of unchanged sources are re-checked and kept;
  the regenerated and failing ones are translated again. The corpus is rewritten without the retranslated
  languages first, so a language is in the manifest only when all of it is valid; an interrupted recheck
  resumes with a plain `hcai corpus <config>`. It needs the generator model, temperature and seed the corpus
  was made with.

## Consequences

- The function-word count is a heuristic, not a language classifier: an English document that quotes long
  lowercase Portuguese phrases could be rejected, and a document with almost no function words (a list of
  skills) decides nothing on its own. The thresholds are in `hcai.language`, and the retry note says what
  was counted.
- The new prompt text changes the generation cache keys, so `--force` on an existing corpus regenerates every
  English document with new calls; `--recheck` is the repair path for a corpus frozen before this check.
