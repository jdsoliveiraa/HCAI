# 9. The name manipulation check asks each pipeline model about each name, blind to the intended answer

Date: 2026-11-15 · Status: accepted (#11 Part B; spec #1 story 11); amended by ADR 0011, which makes a probability
mode the default. What follows is the choice mode, still available as `hcai names --mode choice`.

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
- **Option order.** The options are the same for every name at a repeat and vary over the repeats: male and
  female alternate first (balanced within one for any number of repeats); the origins are shuffled with a fixed
  seed, rejecting an order that opens with the bank's origins in bank order (for the four-origin bank, exactly
  its order; for a two-origin bank the pair's relative order still varies, where rejecting every bank-order pair
  would fix it). "unsure" and "Other or unsure" stay last. Position bias is thus spread over the repeats.
- **Repeats are samples.** Each repeat is a distinct request (its own option order and seed `[design] seed +
  repeat`, sent even when the design sets no seed, so repeats never share a request), cached on its own, at the
  design's temperature (overridable). A rerun replays the cache at no cost.
- **Scoring.** Per name x model, the gender and origin hit rates are the shares of valid answers equal to the
  intended label ("unsure" and "Other or unsure" are misses). A name passes when both rates are at least the
  threshold (default 0.8) for every model and no answer is missing. A missing answer (a failed call, invalid JSON
  or a value outside the options) is left out of the rates, is not cached, and is asked again on a rerun. Cells
  report the unweighted mean of the rates of their names that have a valid answer (and how many those are), per
  model and over all models.
- **Failures are not perception.** Failed calls are reported with their error, not only as failing names; 10 in a
  row for one model stop the check (a wrong model id or an outage would otherwise fail every name), and a provider
  limit stops it as in `hcai run` (ADR 0005). Either way the answers so far are cached.
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
