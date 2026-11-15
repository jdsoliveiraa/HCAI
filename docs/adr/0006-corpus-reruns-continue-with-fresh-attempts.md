# 6. Corpus reruns continue with fresh attempts; English freezes before translation

Date: 2026-11-14 · Status: accepted (#11 Part B, #14); refines ADR 0002 for the corpus generator; the guard is refined by ADR 0007

## Context

With the real generator, PT-PT translations of base CVs kept failing the slot rules (a gendered adjective such as
`rigoroso` or `dedicado` left outside a `{{masc|fem}}` slot). After `max_attempts` the whole `hcai corpus` failed
and froze nothing, not even the valid English corpus. Rerunning made no progress: translations run at temperature
0, parseable invalid answers are cached, and attempt k's retry note depends only on attempt k-1's answer, so every
rerun replayed the same chain of invalid answers from the cache without a single new call.

## Decision

- Each document (job ad, base CV, translation) gets `[generator] max_attempts` (default 4) **new** calls per run.
  Attempts are still numbered and carry the previous rejection (ADR 0002), so each is its own cache key. Answers
  served from the cache do not use the budget: a rerun replays the earlier attempts for free and continues the
  chain at the next attempt number with a new request. A valid cached answer ends the document with zero calls.
  An answer that crashes a check (e.g. a job without `employer`) counts as invalid, so it cannot trap a rerun
  either. (A safety stop ends a chain at 10,000 attempts in all.)
- Every document is attempted even when another fails; `GenerationFailed` lists each failing document, its
  language and the last rejection reason, and `hcai corpus` exits 1.
- The English corpus is frozen as soon as all of it is valid; an English failure writes nothing (the valid answers
  are in the cache). Translations are kept per document under `<corpus>/partial/<lang>/` with the generator model, temperature and seed that
  made them (`source.json`; a different setup discards them), and re-checked on resume;
  a language enters the manifest only when complete. Rerunning `hcai corpus` on a frozen corpus that lacks a design
  language, or whose `--force` retranslation was interrupted, resumes it.
- A slot-only gendered word directly after a slot whose forms differ (`{{Enfermeiro|Enfermeira}} dedicado`) is
  slotted deterministically (`{{dedicado|dedicada}}`; not pronouns, nor `certificado`/`mestrado`, which may be
  nouns); the repair is kept only if the result passes the slot rules,
  gendered-word guard and diff check. Otherwise the retry note names each leaked word, where it is, and the exact
  fix (`{{rigoroso|rigorosa}}` if it describes the candidate, else `{{rigoroso|rigoroso}}`).

## Consequences

A rerun of a failed corpus always makes progress and spends calls only on failing documents. The attempt number in
a translation's prompt depends on how many runs it took, so two people regenerating from empty caches may see
different attempt chains; the frozen corpus, not the cache, is the reference. `max_attempts` is a run setting and
is not recorded in the manifest.
