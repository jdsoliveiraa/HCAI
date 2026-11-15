# 9. The name manipulation check asks each pipeline model about each name, blind to the intended answer

Date: 2026-11-15 · Status: accepted (#11 Part B; spec #1 story 11)

## Context

The audit attributes score gaps to the gender and origin a name signals. That claim needs evidence that the
audited models themselves read each name of `data/names/name_bank.toml` as intended, and it must be gathered
before the full run so that weak names can be replaced.

## Decision

`hcai names <config>` (`hcai.namecheck`) shows every full name of the design's identities to every pipeline model
(`[design] models` only: the generator and the judge are not audited) `--repeats` times (default 5), through the
same provider client and request cache as `hcai run`, and asks for a JSON-schema answer: perceived gender (male,
female, unsure) and most likely origin among a fixed set (Portuguese; Brazilian; Cape Verdean or other PALOP;
Ukrainian or other Eastern European; Other or unsure), with the context "a job applicant in Portugal".

- **Blind prompt.** The request is built from the name and the repeat index only, never from the identity, so it
  cannot carry the intended labels; at a given repeat it is identical for every name but the name.
- **Option order.** The gender and origin options are shuffled per repeat with a fixed seed (the same order for
  every name at that repeat), rejecting any order that lists the bank's origins in bank order; "unsure" and
  "Other or unsure" stay last. Position bias is thus spread over the repeats instead of favouring one origin.
- **Repeats are samples.** Each repeat is a distinct request (its own option order and seed `[design] seed +
  repeat`), cached on its own, at the design's temperature (overridable). A rerun replays the cache at no cost.
- **Scoring.** Per name x model, the gender and origin hit rates are the shares of valid answers equal to the
  intended label ("unsure" and "Other or unsure" are misses). A name passes when both rates are at least the
  threshold (default 0.8) for every model and no answer is missing. A missing answer (a failed call, invalid JSON
  or a value outside the options) is left out of the rates, is not cached, and is asked again on a rerun. Cells
  report the unweighted mean of their names' rates, per model and over all models.
- **No automatic edits.** The command only reports (exit 2 when a name fails); replacing a name in the bank is a
  reviewed hand edit, after which the check is rerun.

## Consequences

- At the pipeline's temperature 0 the repeats differ mainly through the option order and seed, so a high hit rate
  says the name is read as intended robustly across orders, not that sampling noise was averaged out; pass
  `--temperature` for a sampling-based check.
- With 5 repeats the default threshold needs at least 4 of 5 hits per model; rates on so few answers are coarse,
  so borderline names deserve more repeats.
- The origin options are tied to the four bank origins (`ORIGIN_LABELS`); a config with another origin fails
  before any call until an option is added for it.
