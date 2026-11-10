# 3. Blinding (C1) redacts what the Blinder leaves in, the same way for every identity

Date: 2026-11-10 · Status: accepted (ticket #7)

## Context

Under C1 a Blinder agent rewrites each CV before the Screener, Evaluator and Ranker see it. The crew must not see the variant's demographic tokens (#7), but an LLM blinder will sometimes leave one in. Retrying until the answer is clean costs calls and can still fail, and a CV recorded as missing because of a leak would make missingness depend on identity (a PT-PT feminine CV has more gendered words to miss than an EN one).

## Decision

- The Blinder runs on the pipeline model under test, as one more agent of the crew, through `ClientLLM` (cached, validated for shape, retried per ADR 0002). Only an unparseable or empty answer is retried; after the retries the CV's rows are `missing` and it is never sent unblinded.
- Its answer is then redacted deterministically: every demographic token outside a shared line becomes `[redacted]`.
- The tokens are the same for every identity of a base CV and language: all names of the design's identities (full names and parts), all identities' nationalities in that language, and both forms of every gendered pair of the template (plus the EN pronoun pairs). Redacting only the variant's own tokens would leave the other gender's or other origins' words in, and a `{{her|her}}` invariant word would be redacted for one gender only, which is itself a cue.
- A shared line is a template line that renders identically for every identity (no `$name`, `$nationality` or differing `{{masc|fem}}` slot; `$work_eligibility` expands to its lines). Tokens on such a line, kept verbatim, say nothing about the identity, so "Portuguese: native-level proficiency (C2)" survives while "Nationality: Portuguese" does not.
- The Blinder row's `rationale` holds the redacted text (what the crew saw); the count of `[redacted]` markers is how many tokens the Blinder missed. The proxy-leak report is computed from those rows and the design, so it can be recomputed from `results.csv`.

## Consequences

The crew's input is clean by construction, and C1 has no leak-driven missing data. Redaction can leave odd text ("[redacted] is a data analyst"), and a token that is also an ordinary word on a non-shared line is redacted too (e.g. a rephrased "Fluent in Portuguese"), equally for every identity. Matching is exact and case-insensitive: a misspelt or transliterated name ("Joao") is not caught. Proxies that are not tokens (gendered agreement outside the word list, universities, languages, location) are measured by the proxy-leak report, not removed.
