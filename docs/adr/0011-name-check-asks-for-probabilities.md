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
- **Hedges are not scored** (amended 2026-11-16, see below). Each answer is scored on its substantive options only:
  gender on p(male) and p(female) renormalised to sum 1 ("unsure" left out), origin on the four origin options
  renormalised ("Other or unsure" left out). The hedge share of each question, p(unsure) and p(Other or unsure)
  of the answer's normalised distribution, is reported beside the scores. An answer with nothing on the
  substantive options of a question (all hedge) is missing for that question: it is left out of that question's
  mean and fails the name like a missing answer, but it is a valid answer, so it is cached and a rerun replays it.
  (Amended 2026-11-22, see below: it is first asked again, up to `--reasks` times.)
- **Aggregation.** Per name x model, the scored distributions are averaged over the answers that have one, and the
  full normalised distributions and hedge shares over the valid answers (unweighted means over repeats). (Since
  2026-11-22 the answers are the answers on record, one per repeat: an all-hedge answer that was asked again is
  in none of these means, see below.)
- **Pass rule** (`ProbabilityRule`, on the mean scored distributions; comparisons with 1e-9 float slack):
  - gender: mean p(intended gender) >= `--threshold` (default 0.8);
  - origin, intended Portuguese (`HOST_ORIGINS`): mean p(Portuguese) is strictly the highest of the four origins;
  - origin, any other intended origin: mean p(intended) is strictly the highest among the origins other than
    Portuguese and at least `--origin-min` (default 0.25).
  A tie is not "the highest". A name passes when it passes for every model with no answer missing for either
  question.
- **Outputs.** `perceptions.csv` has the normalised probabilities over all options and raw sums per answer, its
  hedge shares (`gender_hedge`, `origin_hedge`) and its scored distributions (`scored_p_<option>`), and (since
  2026-11-22) its `attempt`; `reasked.csv` (since 2026-11-22) the all-hedge answers that were asked again;
  `summary.csv` per name x model the re-asks made and their limit (`reasks`, `max_reasks`, since 2026-11-22), the
  counts of all-hedge answers per question, the mean full distribution, mean hedge shares,
  mean scored distribution, scored p(intended gender) and p(intended origin), the rule's parameters and the gender,
  origin and overall pass flags; `cells.csv` the mean scored p(intended) and the mean hedge shares per gender x
  origin cell, per model and over all (each name weighs the same: its mean over the models with a value, then the
  mean over names). Probabilities
  are written with 6 decimals; the pass flags are computed at full precision, so a value printed at a boundary may
  still fail it. The printed report shows the cells with their hedge shares, lists the failing names with
  p(intended gender), the scored origin distribution, both hedge shares and what they fail on, and counts answers
  whose raw sums were off 1 by more than 0.01.

### Amendment (2026-11-16): the hedge options are left out of the scoring

The first probability run scored the distributions over every option, hedges included. gpt-5.6-luna put about
41% of its probability on the hedge options ("unsure", "Other or unsure") on average, against about 9% for
qwen3.8-flash. A name that luna read as clearly male but hedged on then fell below the gender threshold, and
"Other or unsure" outweighed the intended minority origin, so the strict scoring measured how much a model hedges,
not what it perceives in the name. The researcher decided to score the substantive options renormalised and to
report the hedge share separately, so hedging stays visible without deciding the pass.

### Amendment (2026-11-19): the probability prompt asks for a best estimate (`prompt_version` prob-v2)

With the hedges left out of the scoring, the rerun showed gpt-5.6-luna putting 100% on "unsure" and/or "Other or
unsure" in 25 of its 160 answers (16%), spread over every repeat (0-4) and every intended origin, while its other
answers on the same names were clear (e.g. 0.99 on the Ukrainian option). These all-hedge answers are missing for
that question and fail the name, so the check failed names on an answering habit of one model rather than on
what the name signals. The researcher kept the strict rule (an all-hedge answer is missing and fails the name)
and changed the probability prompt instead:

- It asks for the model's best estimate, says that a name carries real but uncertain signals, asks it to express
  that uncertainty in the probabilities of the substantive options in proportion to how likely each one is, and
  keeps "unsure" / "Other or unsure" for a name that gives no signal at all for that question.
- Unchanged: the context ("a job applicant in Portugal", "based only on the name"), the options, their per-repeat
  order, the seed and the JSON schema. The new text names no gender or origin option outside the option lists
  (only the two hedge labels it explains), no intended label, nothing from the name bank and no example name.
- The choice-mode prompt is unchanged.
- `PROBABILITY_PROMPT_VERSION` ("prob-v2") is recorded per answer in `perceptions.csv` and per name x model in
  `summary.csv` (`prompt_version`). The first prompt is "prob-v1"; its outputs have no `prompt_version` column.
- The new prompt has new cache keys: the prob-v1 answers stay in the cache, unused, and the check asks again.

A model can still answer all-hedge under prob-v2; that answer is still missing and fails the name (since
2026-11-22 it is asked again first, see the next amendment). Asking for a
best estimate pushes a model towards committing to a reading, so a pass under prob-v2 is not directly comparable
with one under prob-v1: compare results of the same `prompt_version` only.
"Other or unsure" also means an origin outside the four listed; by keeping it for names with no signal, prob-v2
may move the probability of a name that points elsewhere (an Italian- or Spanish-sounding surname, say) onto
the listed origins, which can help such a name pass. Read origin passes under prob-v2 with that in mind. Failed
calls carry the prompt version too (the version of the request that failed).

### Amendment (2026-11-22): an all-hedge answer is asked again (re-asks, `--reasks`)

Under prob-v2, gpt-5.6-luna still put 100% on "unsure" and/or "Other or unsure" in 10 of its 160 answers
(qwen3.8-flash: in none of 160): 8 on both questions and 2 on origin only, with a clear gender. They are spread
over every repeat (0-4) and every intended origin, on 9 of the 32 names (8 names once, one name twice). Each of
the 9 passes gender and origin on its other answers (p(intended gender) at least 0.99, p(intended origin) 0.61
to 0.99) and fails only because an answer is missing for a question; no other name fails. The same name is
answered clearly at its other repeats and the same option order is answered clearly for other names, and neither
pipeline model is deterministic at temperature 0 (`configs/core.toml`). So the check was failing names on
answers in which the model gave no estimate at all, not on what it perceives in the name.

The researcher decided to treat an all-hedge answer like a failed call and ask it again as a fresh attempt:

- **What is asked again.** In probability mode, a valid answer with nothing on the substantive options of gender
  or of origin: exactly the answers the scoring already counts as missing for a question. There is no threshold
  on the hedge share: an answer with any probability on a substantive option is an estimate and is scored as
  before, however much it hedges.
- **The re-ask.** The same name, model and repeat, with the same prompt (prob-v2: the text is unchanged, so the
  version is too), options, option order of that repeat and JSON schema. Only the seed changes: the design seed +
  the repeat + 1000 x the attempt (`REASK_SEED_STEP`), which gives every attempt its own cache key, as the
  attempt number in the prompt does in ADR 0006. The pipeline models do not honour the seed, so it is not what
  makes the new answer differ: a re-ask is a new draw of the same question. Attempt 0 adds nothing to the seed,
  so its request and cache key are what they were, and the answers already cached replay (a test pins the keys).
- **The chain.** Up to `--reasks` re-asks per answer (default 3; 0 turns them off). The first attempt that is not
  all hedge is the answer, whole and whatever it says, and the chain stops there. If every attempt is all hedge,
  the last one is the answer on record: still missing for that question, so the name still fails. The strict
  rule is unchanged.
- **The whole answer is replaced.** An answer that is all hedge on one question only is asked again for both, and
  its clear answer to the other question is not scored (it is kept in `reasked.csv`). One answer is one request:
  the two questions are never put together from different attempts.
- **Failures.** A re-ask whose call fails, or whose answer is not valid, is a missing answer at that attempt, as
  before: not cached, so a rerun replays the attempts before it from the cache and asks that attempt again. Only
  a valid all-hedge answer moves the chain to the next attempt.
- **Cache.** Every attempt is a valid answer and is cached on its own, so a rerun with the same settings makes no
  call and gives the same tables. A higher `--reasks` continues a chain where it stopped; a lower one replays
  the start of it.
- **Record.** `perceptions.csv` keeps one row per answer and gains `attempt` (0, or k for the k-th re-ask).
  `reasked.csv` has one row, with the same columns, per all-hedge answer that was asked again, so the two files
  hold every valid answer received. `summary.csv` gains `reasks` (the re-asks made, per name x model) and
  `max_reasks` (the `--reasks` limit of the run, beside the rule's other parameters). The printed report gives per
  model the re-asks made, for how many answers, and how those answers ended: answered, still all hedge, or
  missing (the re-ask failed or was not valid). Its last line counts the answers, the re-asks, the LLM calls and
  what came from the cache (answers + re-asks = calls + from cache). A choice-mode run writes no `reasked.csv`
  and removes one left in its output directory by a probability run, so no re-asks of another run sit beside it.
- **Choice mode** is unchanged: "unsure" and "Other or unsure" are answers there and are never asked again.

This is a retry of non-answers, with a record, not a filter on content. What decides a re-ask is that an answer
gives no estimate for a question; the intended labels, and which gender or origin an answer favours, play no part
in it. An answer that gives an estimate is never asked again, whether or not it suits the name, and the first one
that does is kept: nothing chooses between two estimates. What it does change is stated here so that it is read
with the results:

- The scores describe what a model perceives when it gives an estimate. How often it gives none is a property of
  the model (10 of 160 answers for gpt-5.6-luna, none for qwen3.8-flash) and stays in the outputs: report the
  re-asks per model beside the pass rates.
- The prompt keeps the hedges for a name that gives no signal at all, so an all-hedge answer can be the model's
  reading of the name, not a refusal. Re-asks do not hide that case as long as it is the model's usual answer:
  the name then stays all hedge and fails. But one estimate in up to four attempts now replaces the all-hedge
  answers before it, so a name the model declines to place fairly often can pass. `attempt` and `reasks` show how
  many re-asks a pass needed; a name that needed several, or on several repeats, deserves a look in `reasked.csv`.
- The hedge shares (`gender_hedge`, `origin_hedge`: per answer, per name x model and per cell) are over the
  answers on record. An all-hedge answer that was asked again had a hedge share of 1 and is in none of them, so
  with re-asks the hedge columns say how much the model hedges when it gives an estimate and understate its
  hedging overall: read them with `reasks` and `reasked.csv`.
- Results with and without re-asks differ on the names that needed one. `--reasks 0` gives the earlier behaviour
  from the same cache; compare runs of the same `--reasks` only (`max_reasks` in `summary.csv`).
- As in ADR 0006, a chain depends on what the model answered when it was asked: two people starting from empty
  caches may see different chains. The outputs and the cache are the record.
- A re-ask costs one call, at most `--reasks` per all-hedge answer. With re-asks on, `--repeats` is at most 1000,
  so that no attempt has the seed of another repeat. The attempt is carried by the seed alone, so in one cache a
  design whose seed is a multiple of 1000 away from another's would take that design's re-asks for its first
  answers (and the reverse), as would a `--reasks 0` run of more than 1000 repeats; neither applies to the
  design (seed 42, 5 repeats). Use a separate `--cache` for such a run.

## Consequences

- A Brazilian or Cape Verdean name may now pass while "Portuguese" is its most likely origin, as long as its own
  origin clearly beats the other non-Portuguese readings and reaches the minimum. The audit's claim is accordingly
  "the name leans towards its intended origin more than towards any other minority origin", not "the name is read
  as foreign"; results should be reported that way.
- "Other or unsure" no longer competes with a non-Portuguese intended origin. A name the model can hardly place
  still has to lead the other minority origins and reach the minimum among what is left; how much the model
  hedged on it is in the hedge columns and should be read alongside a pass. A model that hedges heavily is scored
  on the small remainder of its probability: only an all-hedge question counts as missing, so an answer with 99%
  on the hedges is scored on the remaining 1% at full weight and can pass outright. Read every pass together with
  its hedge shares (and, since 2026-11-22, with its re-asks: the hedge shares leave out the all-hedge answers
  that were asked again).
- Self-reported probabilities from an LLM are not calibrated likelihoods; they are a graded perception score. At
  temperature 0 the repeats still mainly vary the option order, as in ADR 0009.
- The schema uses the option labels (with spaces) as property keys. OpenAI-style strict structured outputs accept
  that; a backend that restricts property names or reorders keys would fail the calls (and the check would stop
  after 10 in a row) or undo the option shuffle in what the model writes. Switch to slug keys if a pipeline model
  moves to such a backend.
- The minimum (0.25) and the threshold are researcher choices, recorded in `summary.csv` with every result.
