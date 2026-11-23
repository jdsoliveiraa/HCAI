# Corpus v1 corrections after the realism review (#11)

The realism review (`cv-realism-review.md`) found text errors in the nurse and software engineer documents. We
corrected them by hand-chosen text edits, not by regenerating the corpus. The full list of 74 edits, with the file,
the field, the old text, the new text and the reason, is in [`corpus-corrections.csv`](corpus-corrections.csv).

## Why edits, not regeneration

- The edits are made in the base CVs and job ads, before `hcai variants` renders the identities. Each edit is the
  same for all 8 identities and both languages, so it cannot add a demographic difference.
- The documents keep their reviewed content. A regeneration would make new documents that need a new review.
- The edits apply to this corpus version only. A new corpus (other model or prompt) gets its own review.

## What changed

| Edit | Where | Count | Reason |
|---|---|---|---|
| Remove "(fictitious unit)", "(Fictitious)", "(Unidade Fictícia)", "(hospital fictício)" and similar | nurse job ad and nurse CVs, EN and PT | 58 | the generator's "fictional employer" instruction leaked into the text |
| "Hospital Lusíadas Alvalade" → "Hospital Central de Alvalade" | nurse job ad, EN and PT | 4 | a real private hospital brand; the ad also names SClínico, the public SNS record system, which fits an invented public hospital |
| "frameworks such as React or TypeScript" → "using React and TypeScript" | software engineer job ad, EN and PT | 4 | TypeScript is a language, not a framework |
| "Lisbon, Portugal (Hybrid)" → "Lisbon, Portugal, hybrid" | software engineer job ad, EN and PT | 4 | nested parentheses in the title |
| "Engenheiro de Software (Software Engineer)" → "Engenheiro de Software" | software engineer job ad, PT | 2 | untranslated title in parentheses |
| "estritamente relacionada" → "estreitamente relacionada" | software engineer job ad, PT | 2 | wrong word for "closely related" |

Counts include each copy of the text: the JSON record (`manifest.json` for job ads, `<cv>.json` / `<cv>.pt.json`
for CVs) and its rendered `.md` file.

## Checks after the edits

- No "fictitious", "fictional" or "fictício/a" and no "Lusíadas" is left in `data/corpus/v1`.
- Every JSON file keeps its format, and every `.md` file equals a fresh render of its JSON record (6 job ads,
  54 CV files).
- `recheck_corpus` (the `hcai corpus --recheck` checks: requirements, language, gendered words) passes on the
  edited corpus with a client that refuses every call: nothing to regenerate.
- `hcai variants configs/core.toml`: 432 variants, demographic-only diff check passed.

## Not changed

The review also noted smaller points that we left as they are, because they are the same for every identity and
do not affect realism much: the English ad's mix of Portuguese degree names with translations, "equipa da
plataforma principal" style translationese, "durante exatamente 24 meses" in one nurse CV, and ISO dates
(2022-09) in one accountant CV.

The edits change the job ad and CV text, so the pilot answers made before them (`results/pilot/`) are from the
uncorrected text. The full run uses the corrected corpus.
