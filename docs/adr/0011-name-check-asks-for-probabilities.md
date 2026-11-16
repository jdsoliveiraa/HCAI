# 11. The name manipulation check asks for probabilities, judging non-Portuguese origins among themselves

Date: 2026-11-16 · Status: accepted (#11 Part B); amends ADR 0009 (the choice mode stays available)

## Context

The first real check (ADR 0009, choice mode) asked each pipeline model for the single most likely origin of "a job
applicant in Portugal". Names common in Portugal as well as in Brazil or Cape Verde (Leandro Lima, Fernanda Lima,
Jéssica Oliveira, Odair Monteiro, Neusa Monteiro) were mostly answered "Portuguese". In Portugal that is the right
base-rate answer, so the choice mode cannot tell a name that leans strongly Brazilian from one with no Brazilian
signal at all: both lose to "Portuguese". The researcher decided to ask for probabilities instead.

## Decision

`hcai names` runs in a probability mode by default (`--mode probability`); `--mode choice` keeps the ADR 0009
check unchanged.

- **Same prompt otherwise.** The same context ("a job applicant in Portugal", "based only on the name"), the same
  options (male, female, unsure; the four origins and "Other or unsure"), the same per-repeat option order and seed,
  and the same blindness to the intended labels. The model returns, by JSON schema, an object per question with
  every option as a key and a number as its value. The prompt differs from the choice prompt, so its answers get
  their own cache keys and the choice answers stay valid.
- **Parsing.** An answer is valid when each question has every option exactly once (labels matched ignoring case
  and padding) with a finite, non-negative number (not a string or a bool) and a positive sum. Each distribution is
  divided by its sum, which is recorded (`gender_sum`, `origin_sum`), so percentages or loose sums still count. Any
  other answer is missing: not cached, asked again on a rerun, and failing the name, as in ADR 0009.
- **Aggregation.** Per name x model, the normalised distributions of the valid answers are averaged (an unweighted
  mean over repeats).
- **Pass rule** (`ProbabilityRule`; comparisons with 1e-9 float slack):
  - gender: mean p(intended gender) >= `--threshold` (default 0.8);
  - origin, intended Portuguese (`HOST_ORIGINS`): mean p(Portuguese) is strictly the highest of all five origin
    options;
  - origin, any other intended origin: mean p(intended) is strictly the highest among the options other than
    Portuguese ("Other or unsure" included) and at least `--origin-min` (default 0.25).
  A tie is not "the highest". A name passes when it passes for every model with no answer missing.
- **Outputs.** `perceptions.csv` has the normalised probabilities and raw sums per answer; `summary.csv` the mean
  distribution, mean p(intended gender) and p(intended origin), the rule's parameters and the gender, origin and
  overall pass flags per name x model; `cells.csv` the mean p(intended) per gender x origin cell, per model and over
  all (each name weighs the same: its mean over the models with a value, then the mean over names). Probabilities
  are written with 6 decimals; the pass flags are computed at full precision, so a value printed at a boundary may
  still fail it. The printed report lists the failing names with p(intended gender), the whole origin
  distribution and what they fail on, and counts answers whose raw sums were off 1 by more than 0.01.

## Consequences

- A Brazilian or Cape Verdean name may now pass while "Portuguese" is its most likely origin, as long as its own
  origin clearly beats the other non-Portuguese readings and reaches the minimum. The audit's claim is accordingly
  "the name leans towards its intended origin more than towards any other minority origin", not "the name is read
  as foreign"; results should be reported that way.
- "Other or unsure" competes with a non-Portuguese intended origin, so a name the model cannot place does not pass
  on a low bar.
- Self-reported probabilities from an LLM are not calibrated likelihoods; they are a graded perception score. At
  temperature 0 the repeats still mainly vary the option order, as in ADR 0009.
- The schema uses the option labels (with spaces) as property keys. OpenAI-style strict structured outputs accept
  that; a backend that restricts property names or reorders keys would fail the calls (and the check would stop
  after 10 in a row) or undo the option shuffle in what the model writes. Switch to slug keys if a pipeline model
  moves to such a backend.
- The minimum (0.25) and the threshold are researcher choices, recorded in `summary.csv` with every result.
