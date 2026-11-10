# 3. Blinding (C1) redacts what the Blinder leaves in, the same way for every identity

Date: 2026-11-10 · Status: accepted (ticket #7)

## Context

Under C1 a Blinder agent rewrites each CV before the Screener, Evaluator and Ranker see it. The crew must not see the variant's demographic tokens (#7), but an LLM blinder will sometimes leave one in. Retrying until the answer is clean costs calls and can still fail, and a CV recorded as missing because of a leak would make missingness depend on identity (a PT-PT feminine CV has more gendered words to miss than an EN one).

## Decision

- The Blinder runs on the pipeline model under test, as one more agent of the crew, through `ClientLLM` (cached, validated for shape, retried per ADR 0002). Only an unparseable or empty answer is retried; after the retries the CV's rows are `missing` and it is never sent unblinded.
- Its answer is then redacted deterministically: every demographic token outside a shared line becomes `[redacted]`.
- The tokens are the same for every identity of a base CV and language: all names of the design's identities (full names and parts, matched in their own case or all caps, so the word "clara" is not the name Clara), all identities' nationalities in that language (in PT-PT with the masculine form, Brasileira/Brasileiro), and both forms of every gendered pair of the template, each word in which a phrase slot's forms differ, plus EN pronouns and titles (all-caps MS, MR, HIS are acronyms). Redacting only the variant's own tokens would leave the other gender's or other origins' words in, and a `{{her|her}}` invariant word would be redacted for one gender only, which is itself a cue.
- A shared line is a template line that renders identically for every identity (no `$name`, `$nationality` or differing `{{masc|fem}}` slot; `$work_eligibility` expands to its lines). Tokens on such a line, kept verbatim, say nothing about the identity, so "Portuguese: native-level proficiency (C2)" survives while "Nationality: Portuguese" does not.
- The Blinder row's `rationale` holds the Blinder's own answer, before redaction. Redaction is deterministic, so the crew's text, the number of tokens the Blinder missed and the proxy-leak report are all recomputed from that answer and the design (`hcai.blinding.proxy_leaks`); a `[redacted]` the Blinder wrote itself is not counted as a leak.
- The stability sample re-runs the Screener, Evaluator and Ranker on the same blinded text; the Blinder's own run-to-run variation is not sampled.

## Consequences

The crew's input is clean by construction, and C1 has no leak-driven missing data. Redaction can leave odd text ("[redacted] is a data analyst"), and a token that is also an ordinary word on a non-shared line is redacted too (e.g. a rephrased "Fluent in Portuguese"), equally for every identity. Matching is exact: a misspelt or transliterated name ("Joao"), an inflection beyond the ones listed (plurals) or a country name ("Brasil") is not caught. The origin-field parsers of the proxy-leak report assume the corpus layout (`<degree>, <institution>, ..., <year>` education lines, a `<location> · Nationality:` header). Proxies that are not tokens (gendered agreement outside the word list, universities, languages, location) are measured by the proxy-leak report, not removed.
