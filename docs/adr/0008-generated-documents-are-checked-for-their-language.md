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
- `hcai.language` counts English and Portuguese function words over a document's fields (not employer,
  institution, location, dates or evidence quotes); a lowercase word with `ã`, `õ` or `ç` also counts as
  Portuguese. Words common to both languages (`a`, `as`, `no`, `do`, `os`, `se`), all-caps tokens, the `e-` of
  "e-commerce"/"e-Fatura", "e.g." and a function word between two capitalised words (a name such as "Ordem dos
  Contabilistas Certificados") are not counted. A prose field (summary, bullets, requirements) is in the wrong
  language with at least 3 words of the other language and more of them than of its own; skills,
  certifications and degrees (often proper names) count only toward the document's totals, which must not
  lean to the other language either. An English headline or job title is Portuguese when it holds a
  Portuguese job-title word and no English one ("Contabilista Certificado" fails, "Contabilista Certificado
  (Certified Accountant)" passes). A translation also may not copy a prose field or title of 3+ words verbatim
  from the original (a terse English bullet has too few function words to count). A failure is an ordinary
  invalid answer (ADR 0002/0006): the retry note names the offending fields.
- A kept translation under `partial/<lang>/` records the hash of the English text it was made from and is
  reused only for that text, so a regenerated source is always translated again.
- `hcai corpus <config> --recheck` re-validates a frozen corpus with today's checks. Valid English documents
  are kept as frozen (no call, no cache needed); a failing job ad is regenerated with all its base CVs (they
  quote its requirements), a failing base CV alone. Translations of unchanged sources are re-checked and kept;
  the regenerated and failing ones are translated again. Those kept translations are written under
  `partial/<lang>/` first, then the corpus is rewritten without the languages to retranslate, so a language is
  in the manifest only when all of it is valid; an interrupted recheck resumes with a plain `hcai corpus
  <config>`. A complete, valid language with only a leftover `partial/<lang>` (an interrupted `--translate
  --force`) stays in place while it resumes. A rewrite swaps directories with two renames and puts the
  previous corpus back if interrupted between them. It needs the generator model, temperature and seed the corpus
  was made with.

## Consequences

- The function-word count is a heuristic, not a language classifier: an English document that quotes long
  lowercase Portuguese phrases could be rejected, and a document with almost no function words (a list of
  skills) decides nothing on its own. The thresholds are in `hcai.language`, and the retry note says what
  was counted.
- Partial translations kept before this change have no source hash and are asked again (free while the cache
  that made them exists).
- A rewrite re-renders every English template from its JSON: if the renderer (header, headings) changed since
  the corpus was frozen, unchanged documents' `.md` files change with it.
- The new prompt text changes the generation cache keys, so `--force` on an existing corpus regenerates every
  English document with new calls; `--recheck` is the repair path for a corpus frozen before this check.
