# Validation report: HCAI 2026/27 project proposal

**Proposal:** "Who Gets Shortlisted? Auditing and Mitigating Demographic Bias in Multi-Agent LLM Hiring Pipelines for the Portuguese Job Market" (`ijcai27.tex`, 3 pages)
**Validated against:** `Project/Project_HcAI_26_27.pdf` (brief), Lectures #1–#10, the student's answers (2026-11-09)
**Validator date:** 2026-11-09

## Verdict: **Submit after fixes, and cut the scope.**

The proposal answers every question in the brief, fits the course well and is methodologically careful in places (shuffled positions, several names per identity, paired variants, Holm correction). It needs fixes for three reasons:

1. **The scope is not feasible.** You said you have **under 10 h/week, and that covers code, runs, analysis and the 15-page paper**. That is at most ~25 h until 27 Nov. The design as written needs ~35–40k LLM calls over 2 models and 4 conditions, plus a 3-agent framework build, an LLM judge, a 2-person audit, three kinds of statistical test and the paper. A realistic estimate is 60–90 h. The proposal should commit to a smaller core and move the rest to stretch goals (section 3.4).
2. **The PDF is not in Times.** The Tectonic/XeTeX build substitutes Latin Modern for Times (`pdffonts` shows only LMRoman/CM fonts, and the log says "Font shape `TU/ptm/m/n' undefined"). The IJCAI template requires Times. This is a one-line fix.
3. **There are a few methodological gaps.** None is fatal, but a careful reader (the teacher, at the defence) will spot them. The main ones: the mixed-model formula cannot answer RQ3, the Ranker's quality cannot be measured with equal-quality pools, the claim "one of the first audits" overreaches, and work authorisation is a confound for the origin groups (section 4).

Two things only you can check (not errors in the document):
- The brief and Lecture #1 give the proposal deadline as **9 Oct** and say projects are for **groups of three**. You said the teacher extended your deadline. Also make sure working alone is agreed. Getting both in writing (Discord or email) costs nothing.
- The teacher asked for the HCAI topic to be explicit. The proposal states it in the abstract and §1, but **not in the title**. See edit E3.

---

## 1. Compliance checklist (brief = `Project_HcAI_26_27.pdf`)

| # | Requirement (brief §) | Status | Evidence / note |
|---|---|---|---|
| 1 | ≤ 3 pages, 2-column (§2) | **Met** | PDF is exactly 3 pages including references. There is no spare room, so every edit below must be offset by cuts. |
| 2 | IJCAI formatting guidelines (§2, fn. 1) | **Partial** | The template and `named.bst` are used, but **the body font is Latin Modern, not Times** (`ijcai27.log`: "Font shape `TU/ptm/m/n' undefined … defaults substituted"; `pdffonts` lists only LMRoman/CM). See E1. |
| 3 | English or Portuguese (§2) | **Met** | English. |
| 4 | Human-centred AI problem: trustworthy, cooperative, respects human values and control (§2) | **Met** | Fairness in high-risk hiring, with human oversight (Abstract; §1 para 2). |
| 5 | Related to ≥1 course topic, made explicit in the title **or** the documentation (§2) | **Met (could be stronger)** | §1 last sentence: "connects to the course topics of trustworthy AI and AI risks, agents and multi-agent systems, human oversight (HitL/HotL) and transparency". The title has no HCAI term, and the teacher asked for explicitness (E3). |
| 6 | Well supported, with bibliographic references (§2) | **Met** | 16 references, all cited. Some metadata gaps and one missing citation (four-fifths rule) are covered in §4.3. |
| 7 | Problem & Motivation: what, why interesting, why important | **Met** | §1 "Problem and Motivation". "Important" = EU AI Act Annex III / Art. 14, UNESCO, SDGs, Amazon; "interesting" = multi-agent, non-English, PT groups. |
| 8 | Background: literature giving context | **Met** | §2: correspondence studies, LLM hiring audits, fairness and oversight. Recent (2025–26) multi-agent hiring work is missing, which makes the novelty claim in §3 overreach (E6). |
| 9 | Goals: aim, success criteria, expected contributions | **Met** | §3: aim, RQ1–4, "Success criteria" (core and stretch), "Expected contributions". The success criteria are tied to the full scope (E4). |
| 10 | Approach: methods, existing implementations, how they are modified | **Partial** | §4 describes the pipeline, the counterfactual design and the mitigations. Being "modelled on public recruitment agents" (LangGraph/CrewAI examples) is mentioned, but it is not said **which** example or **what** is modified. One sentence fixes this. |
| 11 | Materials: data, how it is collected, frameworks/tools | **Met** | §5 Data (synthetic CVs, 3 roles × 3 quality levels × ~30 base × 8 × 2 = ~1,440), Tools (Python, LangGraph/CrewAI, OpenRouter, pandas/SciPy/statsmodels). Which models are used is open ("chosen after a pilot"), which is acceptable at proposal stage. |
| 12 | Evaluation: qualitative/quantitative, metrics, statistical tests, expected plots | **Met** | §6: Wilcoxon, mixed model, impact ratio vs 0.8, permutation test, Holm, Spearman, pairwise accuracy, Cohen's κ, and the expected figures. Internal-consistency problems are in §4. |
| 13 | Final deliverables (paper ≤15 p English, GitHub, master script/README, LaTeX sources) (§3, §4.2) | **Met (stated)** | §5 Tools: "released on GitHub with a master script that reproduces all results". |
| 14 | Intermediate stages / plan (§4.2, suggested) | **Partial** | Milestones exist (10/17/21/27 Nov), but **10 Nov is tomorrow** and comes before the proposal itself is submitted. They need re-dating (E5). |
| 15 | Grading criteria: relevance, significance, novelty, soundness, clarity (§3.3) | **Mostly met** | Relevance and novelty (PT-PT, PT-market groups, per-agent measurement) are strong. Soundness has fixable gaps (§4). Clarity is good. |

---

## 2. Course fit

**Strong points**
- The proposal uses the course's own vocabulary. "Human-in-the-loop, human-on-the-loop" appears verbatim in *Trustworthy AI*, Requirement 1 "Human Agency and Oversight" (Lecture #2) and in *Agentic AI*, "Design Operational Mitigations" (Lecture #4: "HITL or on-the-loop: require human approvals for high-risk actions"; "Explainability and logs"; "Continuous monitoring").
- It covers the EU AI Act, UNESCO and the SDGs, which are listed as course learning outcomes and bibliography (Lecture #1), and it cites Shneiderman 2022, Dignum and Russell & Norvig, all on the course bibliography list.
- It touches several syllabus items: §2 agents/MAS, §3 human-AI cooperation, §4 oversight, §5 transparency, §6 safety/risks.

**Gaps**
1. **The core trustworthy-AI requirement is never named.** HLEG Requirement 5, *"Diversity, Non-discrimination, and Fairness"* (Lecture #2, Trustworthy AI slide 7), is exactly what this project measures. The proposal names oversight and transparency but not this requirement. Naming it, together with Req. 1 (oversight) and Req. 4 (transparency: quoted evidence), is the cheapest way to meet the teacher's "make the HCAI topic explicit".
2. **HitL is used loosely.** In the HLEG/course sense, HitL means a human intervenes in *every* decision cycle and HotL means the human monitors and can intervene. The proposal's HitL only escalates flagged or borderline cases. That is selective HitL (deferral), not full HitL. The HotL description (sample audit plus monitoring of metrics) is correct. One clause fixes it (E8). Lecture #5 (HitL vs MitL) frames the judge→human escalation as AI assisting a human decision, which you could cite.
3. **Annotation agreement is a course topic.** Lecture #7 (*Agreement*: Cohen's κ, Fleiss, Krippendorff's α; *Comparison with Ground Truth*) underpins RQ4. Since a colleague is available, measure **human–human κ** (you and the colleague) as the ceiling, and judge–human κ against it. This is a direct use of course material and costs ~1 h of the colleague's time.
4. **"Multi-agent system" is a stretch in Russell & Norvig terms.** The three "agents" are prompted LLM calls in a fixed workflow, with no autonomy, tools or interaction loop. The *Agentic AI* lecture distinguishes workflows from agents. Calling it an "agentic LLM pipeline (three LLM-based agents in a fixed workflow)" is accurate and still on topic. The sentence "Agents follow the standard definition of Russell and Norvig" invites a question at the defence. It can be dropped to save space.

---

## 3. Feasibility

### 3.1 Inputs from the student
| Factor | Answer | Effect |
|---|---|---|
| Time | **< 10 h/week, covering everything including the paper** | ~2.6 weeks → **≤ 25 h total**. The binding constraint. |
| Experience | Strong in Python, LangGraph/CrewAI and statistics | Lowers build and analysis risk. Does not create hours. |
| LLM access | Prefers Claude Code / OpenCode Go subscription models, OpenRouter "when no alternative"; wiring undecided | Significant risk; see 3.3. |
| Human audit | Student plus 1 colleague | Enough for a human–human κ on ~50 items. |
| Teacher | Wants the HCAI topic explicit | E2, E3. |

### 3.2 Call count and cost (recomputed)

**Full design as written** (3 roles × 30 base CVs × 8 identities × 2 languages = **1,440 CVs**; ranking pools = one per base CV per language = **180 pools**):

| Step | C0 | C1 Blind | C2 Rubric | C3 Both | Total |
|---|---|---|---|---|---|
| Blinding agent | – | 1,440 | – | 0 (reuse C1) or 1,440 | 1,440–2,880 |
| Screener | 1,440 | 1,440 | 1,440 | 1,440 | 5,760 |
| Evaluator | 1,440 | 1,440 | 1,440 | 1,440 | 5,760 (fewer if some are screened out) |
| Ranker | 180 | 180 | 180 | 180 | 720 |
| LLM judge | – | – | 1,440 | 1,440 | 2,880 |
| **Per model** | 3,060 | 4,500 | 4,500 | 4,500–5,940 | **16,560–18,000** |

- With **2 models** (commercial + open-weight): **~33–36k calls**. Adding the stability repeats, the pilot and CV generation (~180 calls if variants are templated, ~1,440 more if each variant is LLM-written) brings it to **~35–40k calls**.
- Token estimate per model, assuming a CV ≈ 700 tokens, a job ad plus instructions ≈ 600, a Ranker prompt ≈ 6.2k, and rubric output ≈ 600: **~27M input and ~5M output tokens**.
- Cost per model on OpenRouter, using illustrative price bands rather than quotes (check current prices before choosing):
  - cheap tier (~$0.15 in / $0.60 out per M tokens) → **~$7**
  - mid tier (~$0.50 / $2.00) → **~$25**
  - Two models: **~$15–50**. Money is not the problem.
- **Wall-clock time and the engineering around it are the problem.** At ~3–5 s per call with modest concurrency, one model takes ~5–10 h of runtime. Then come retries, JSON-parsing failures and reruns after prompt fixes, which typically double it.

**Reduced core (recommended, section 3.4):** 2 roles × 15 base × 8 × 2 = **480 CVs**, 60 pools, conditions C0/C1/C2, 1 model:
- Screener 1,440 + Evaluator 1,440 + Ranker 180 + Blinding 480 + Judge 480 = **~4,000 calls**, or ~2,600 without the Screener
- **< $5**, ~1–2 h of runtime.

### 3.3 LLM access via Claude Code / OpenCode CLIs: risk
Driving the agents with `claude -p` / `opencode run` (or any coding-agent CLI) is **not recommended for the measured pipeline**:
- **Confound.** The CLI wraps every request in its own long coding-agent system prompt and tools, so you would be auditing "Claude Code acting as a recruiter", not a recruitment agent. This undermines RQ1–2.
- **No control.** There is no temperature or seed control and model versions are not pinned. That contradicts the proposal's "temperature 0 with a fixed seed" (§5 Tools) and hurts reproducibility, which the brief explicitly values.
- **Throughput.** Per-call start-up overhead plus subscription rate windows make thousands of calls slow and unpredictable. Also check the plan's terms on automated batch use.
- **Where it is fine:** using these subscriptions **to help you write the code**, or to generate the base CVs (one-off, not measured). If OpenCode Go exposes a raw OpenAI-compatible endpoint with temperature control, it could serve as the "open-weight" model. Verify that in the pilot.
- **Recommendation:** run the measured pipeline through one cheap OpenRouter model (or a direct API). At the reduced scope it costs a few euros. Keep the proposal text as it is ("accessed through OpenRouter"). It is the right choice.

### 3.4 Concrete cuts, ordered by impact
| # | Cut | Saves | Keeps |
|---|---|---|---|
| 1 | **One model** in core; second model (open-weight) → stretch | ½ of all calls and runs; one fewer API to debug | All RQs |
| 2 | **Drop C3** (C1+C2 combined) → stretch | ~25% of calls; one less analysis cell | RQ3 still compares blinding vs rubric |
| 3 | **2 roles (software engineer vs nurse), ~15 base CVs each** | 1,440 → 480 CVs; less generation and hand-checking | Stereotype contrast (the interesting part); both languages; all 8 identities |
| 4 | **LLM judge only on C2, audit 50 items stratified** (~25 flagged + ~25 unflagged), rated by both you and your colleague | Judge calls ÷2; κ is not degenerate | RQ4; course link to Lecture #7 |
| 5 | **Plain Python orchestration** (or the thinnest LangGraph graph) instead of adapting a CrewAI/LangGraph example | 3–6 h of framework friction | Same 3-agent structure; easier logging and replay |
| 6 | **One primary test per outcome**: a mixed model (with bootstrap CIs) for scores, a logistic GLMM or bootstrap for selection. Drop the duplicate Wilcoxon and permutation analyses (or put them in the appendix) | 2–4 h of analysis and writing | Rigour; fewer multiple-comparison families |
| 7 | **Screener optional.** With "all CVs meet mandatory requirements" it will pass ~100%, which gives no variance for RQ2. Either keep it only for the "borderline" CVs or make it rule-based | ~1/3 of pipeline calls | RQ2 then compares Evaluator → Ranker (still per agent) |
| 8 | Age, career gap and parenthood stay stretch (as now). Correll 2007 can then be dropped to free space | — | — |

**Realistic hour budget for the reduced core (~25 h):**

| Task | Hours |
|---|---|
| Base CVs, templating and name lists, PT gender agreement | 4 |
| Pipeline, logging and pilot | 5 |
| Full runs, fixes and reruns | 3 |
| Analysis and figures | 4 |
| Judge plus 50-item audit (your share) | 2 |
| Paper: §1–2 reused from the proposal; about 8–10 pages, not 15 | 7 |

That is tight, but feasible with heavy coding-agent help. The full design is not feasible.

### 3.5 Milestones
The current milestones are "pilot and baseline pipeline by 10 Nov, core experiments by 17 Nov, analysis and human audit by 21 Nov, final 27 Nov". 10 Nov is **tomorrow** and falls before the proposal is submitted. Suggested dates:
- **15 Nov:** CVs generated and pipeline piloted (C0, ~40 CVs)
- **19 Nov:** all core runs done
- **22 Nov:** analysis and human audit done
- **24 Nov:** paper draft
- **27 Nov:** final submission

Start writing the paper's §1–2 from the proposal on 20 Nov, while the runs finish.

### 3.6 Risk list
| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Time overrun (≤25 h for everything) | **High** | High | Cuts 1–7; freeze scope at the 15 Nov pilot |
| Baseline shows no measurable bias (common with recent models on name swaps), so the "mitigation reduces bias" criterion is moot | Medium–High | Medium | Pre-register a null result as a valid outcome. Report equivalence bounds (e.g. a 90% CI on the gap within ±2 points). Keep RQ4 and the PT-PT comparison as contributions that do not depend on bias being found |
| Synthetic CVs do not separate quality levels (strong vs weak), so the quality metric is noisy | Medium | Medium | Use deterministic quality levers (years, certifications, matched skills); check Evaluator Spearman on the pilot |
| PT grammatical-gender templating leaks or breaks text | Medium | Medium | Slot templates, not free LLM rewriting of variants; hand-check 1 CV per identity per language |
| Blinding leaks identity (university, languages, grammatical gender) | High in PT | Low (it is a finding) | Measure residual leakage: ask an LLM to guess gender/origin from blinded CVs |
| JSON/structured-output failures, rate limits | Medium | Medium | Pydantic validation plus retry; cache every response keyed by a prompt hash |
| Too few positive cases for κ (judge flags ~5%) | Medium | Medium | Stratified sample (cut 4) |
| LLM access via CLI subscriptions | Medium | High | §3.3: API for measured runs |
| Same model generates CVs and judges them (self-preference) | Medium | Low–Med | Generate CVs with a different model family than the pipeline/judge |
| Administrative (late proposal, solo vs group of 3) | Low | High | Written confirmation from the teacher |

---

## 4. Correctness

### 4.1 Methodological issues
1. **The mixed model cannot answer RQ3.** The model `score ~ gender × origin + role + lang + cond + (1|CV)` estimates *one* bias pattern shared by all conditions; `cond` only shifts the mean score. To test whether a mitigation reduces bias you need the **identity × condition interaction**, i.e. `gender × origin × cond`, or fit one model per condition and compare the gaps. RQ1 also asks about role and language dependence, so `identity × role` and `identity × lang` terms are needed too. Add a random intercept for **name** `(1|name)`, since several names per identity is the design's point. Screener pass/fail and top-3 are binary, so they need a logistic GLMM, not the linear model.
2. **The Ranker's quality cannot be measured.** Ranking pools are "eight CVs of *equal quality* that differ only in identity". "Pairwise accuracy (strong ranked above weak)" therefore never applies to the Ranker, and Spearman vs quality level only measures the **Evaluator's** scores. Either add a small set of **mixed-quality pools** (e.g. 1 strong, 3 average, 4 weak across identities) or state that ranking quality is measured on Evaluator scores only.
3. **The success criterion uses unmatched units.** It says "losing at most ~5 percentage points of ranking quality", but Spearman ρ is not in percentage points. Tie the 5 pp to pairwise accuracy (or top-3 precision) and give a separate Δρ threshold.
4. **"Every selection-rate ratio ≥ 0.8" is noisy at this n.** With top-3 of 8 the base rate is 0.375. Per identity × role × language there are only ~30 pools (SE ≈ 0.09), so ratios swing past 0.8 by chance. The ratio relative to the *most-selected* of 8 groups is also biased downward by the max. Report the impact ratio with bootstrap CIs, pooled over roles and languages, and make the primary target the effect-size criterion ("largest score gap halved").
5. **The blinding effect is near-tautological.** If C1 removes names and gender markers, the 8 variants become (almost) the same text, so a smaller paired gap is expected by construction. The informative quantities are **residual leakage** (can identity still be inferred?) and **quality cost**. Frame RQ3 that way so the result does not look trivial at the defence.
6. **Work authorisation confounds origin.** In Portugal, Brazilian, PALOP and some Eastern European (non-EU, e.g. Ukrainian) candidates may need a residence or work permit. A model that penalises them could be applying a legal requirement, not bias. State in every CV that the candidate holds the right to work in Portugal, or hold residence status constant. Also hold **language proficiency** constant (e.g. Portuguese C2/native for everyone) and say so.
7. **Name validity.** "Names only signal perceived gender and origin" is acknowledged. Add a quick **manipulation check** (ask a model, or your colleague, to classify a sample of names) so that you show the cues are perceivable at all.
8. **The quality ground truth is by construction.** The quality level comes from the generator, so it is only valid if the CVs actually differ in the levers used. A hand check of a sample is already planned. Say which levers (years, matched requirements, certifications) define the levels.
9. **HitL "before they count" is simulated.** Nobody will review the hundreds of escalated decisions. Say the escalation is **counted and simulated** (cost of oversight = number of escalations), and that the human review is the 50-item audit. Otherwise "decisions … go to a human reviewer before they count" over-claims what is done.
10. **Determinism.** "Temperature 0 with a fixed seed where supported" is fine, but temperature 0 is not deterministic on most hosted models. The planned stability repeat is the right fix; report its agreement.

### 4.2 Claims that overreach
- §3 Goals, "Expected contributions": *"one of the first audits of a **multi-agent** hiring pipeline"*. Work on agent-mediated and multi-stage LLM hiring has been appearing since 2025. A quick search found, for example, arXiv 2609.19530 ("When Hiring Becomes Agent-Mediated: … Two-Agent Résumé Screening"), arXiv 2608.26899 ("Counterfactual Bias Testing for Application Tracking System"), arXiv 2601.11379 ("Evaluating LLM Behavior in Hiring …") and arXiv 2507.02087 ("Evaluating the Promise and Pitfalls of LLMs in Hiring Decisions"). Those titles come from search snippets; open the papers before citing them. The **defensible novelty** is the combination of per-agent measurement + European Portuguese + Portuguese-market origin groups. Rephrase (E6), and add one sentence to Background acknowledging this line of work.
- §3 Goals, "Expected contributions": *"release an open, reproducible benchmark"*. Under the reduced core, "dataset and code" is more accurate than "benchmark".
- §1: *"Frameworks such as CrewAI and LangGraph already ship example agents that match CVs…"*. True for community examples, but give one concrete reference or URL, or soften to "public examples built with…".
- §2: *"Strong LLMs can serve as judges of other models' outputs"*. Zheng et al. validate judges on chat-quality preferences, not on evidence-faithfulness checks. "Have been used as judges" is safer.

### 4.3 Citations (checked against `ijcai27.bib`; every key cited and every entry used)
| Key | Status | Issue / fix |
|---|---|---|
| an2024discriminate | OK | ACL 2024 short paper, pages and DOI present. "Sensitivity depends on the prompt" is a fair paraphrase (templates matter). |
| armstrong2024silicon | OK | EAAMO '24; arXiv 2405.04412 confirmed to exist. Consider adding a DOI. |
| bertrand2004emily, goldin2000orchestrating | OK | Classic, metadata correct. |
| correll2007motherhood | OK | Only supports a stretch goal, so it is a candidate to drop for space. |
| dastin2018amazon | Minor | Add the URL (Reuters, 10 Oct 2018). |
| dignum2019responsible | OK | Springer 2019 is correct (the course slide says 2021; 2019 is the publication year). |
| euaiact2024 | Minor | Add the date (13 June 2024; OJ L, 12.7.2024) and the ELI/URL. |
| kusner2017counterfactual | Imprecise use | Kusner et al. define counterfactual fairness over a **causal model**, which a name swap is not. A name swap is closer to *counterfactual token fairness* (Garg et al., AIES 2019). Write "inspired by counterfactual fairness", or cite Garg et al. |
| raghavan2020mitigating | Minor | Pages (469–481 per my recollection; I could not confirm them via search) and DOI are missing. Check in the ACM DL. |
| russell2020aima, shneiderman2022hcai | OK | On the course bibliography. Shneiderman's "high automation and high human control" is correctly paraphrased. |
| unesco2021ethics | OK | — |
| wilson2024resume | OK | AIES vol. 7, pp. 1578–1590, DOI present. "Retrieval-based" is accurate (MTE retrieval). |
| yin2024bloomberg | Minor | Add the URL (Bloomberg, 8 Mar 2024). |
| zheng2023judging | OK | NeurIPS 2023 D&B. |
| **Missing** | Add | The **four-fifths rule** is used as a threshold without a source (US EEOC Uniform Guidelines, 29 CFR §1607.4(D), 1978). Note that it is a US heuristic, not EU law. **HLEG Ethics Guidelines for Trustworthy AI (2019)** supports the oversight terminology the course uses. |

---

## 5. Prioritised edits (exact text → replacement)

The page is full, so each addition should be paired with a cut. Suggested space savers: drop Correll (stretch only), drop the Russell & Norvig sentence, and shorten the Wilcoxon/permutation duplication.

**E1 — Font (format compliance, must fix).** In the preamble, after `\usepackage{ijcai27}`, change
```latex
\usepackage{times}
```
to
```latex
\usepackage[T1]{fontenc}
\usepackage{times}
```
and delete `\usepackage[utf8]{inputenc}` (ignored under XeTeX anyway). If Tectonic still substitutes, use `\usepackage{newtxtext,newtxmath}` instead of `times`. Check with `pdffonts ijcai27.pdf`: you should see NimbusRom/Times or TeXGyreTermes, not LMRoman.

**E2 — Name the trustworthy-AI requirement (teacher's explicitness request).** §1 last sentence:
> "The project connects to the course topics of \emph{trustworthy AI and AI risks}, \emph{agents and multi-agent systems}, \emph{human oversight (HitL/HotL)} and \emph{transparency}."

→
> "The project connects to the course topics of \emph{trustworthy AI}, specifically the HLEG requirements of \emph{diversity, non-discrimination and fairness}, \emph{human agency and oversight (HitL/HotL)} and \emph{transparency}, as well as \emph{agentic AI} and \emph{AI risks}."

Add a `hleg2019ethics` bib entry and cite it here.

**E3 — HCAI term in the title.**
> "Who Gets Shortlisted? Auditing and Mitigating Demographic Bias\\ in Multi-Agent LLM Hiring Pipelines for the Portuguese Job Market"

→ for example
> "Who Gets Shortlisted? Fairness Auditing and Human-Overseen Mitigation of Demographic Bias in Agentic LLM Hiring Pipelines for the Portuguese Job Market"

**E4 — Commit to a feasible core (scope).**
- §3, Success criteria:
  > "\emph{Core:} (i)~a reproducible audit of gender~$\times$~origin over three roles in two languages"

  → "\emph{Core:} (i)~a reproducible audit of gender~$\times$~origin over two roles (software engineer, nurse) in two languages, with one LLM"
- Same paragraph:
  > "\emph{Stretch:} add age and career gap / parenthood, more roles and a second LLM."

  → "\emph{Stretch:} a second (open-weight) LLM, the combined condition C3, a third role (accountant), and age and career gap / parenthood."
- §5 Data:
  > "The core design uses three roles: software engineer (male-stereotyped), nurse (female-stereotyped) and accountant (neutral)."

  → "The core design uses two roles: software engineer (male-stereotyped) and nurse (female-stereotyped); a neutral role (accountant) is a stretch goal."
- §5 Data:
  > "About 30 base CVs per role, each with 8 identity variants in 2 languages, give about 1,440 CV texts."

  → "About 15 base CVs per role, each with 8 identity variants in 2 languages, give about 480 CV texts (about 4,000 LLM calls per model)."
- §5 Tools:
  > "at least one commercial and one open-weight model, chosen after a pilot"

  → "one low-cost model chosen after a pilot (an open-weight model is a stretch goal)"
- Table 1: mark C3 as "(stretch)".

**E5 — Milestones.**
> "Milestones: a pilot and the baseline pipeline by 10~Nov, core experiments by 17~Nov, analysis and human audit by 21~Nov, and the final paper and code by 27~Nov."

→
> "Milestones: CV generation and a pilot of the baseline pipeline by 15~Nov, core experiments by 19~Nov, analysis and human audit by 22~Nov, a paper draft by 24~Nov, and the final paper and code by 27~Nov."

**E6 — Novelty claim.**
> "The project will provide one of the first audits of a \emph{multi-agent} hiring pipeline, measured per agent and in European Portuguese."

→
> "The project will provide a per-agent bias audit of an agentic hiring pipeline in European Portuguese, with origin groups relevant to the Portuguese job market, which, to our knowledge, have not been studied."

Also add one Background sentence on recent agent-mediated hiring audits, citing 1–2 papers after you have checked them.

**E7 — Fix the statistical model (soundness).**
> "$\mathit{score} \sim \mathit{gender} \times \mathit{origin} + \mathit{role} + \mathit{lang} + \mathit{cond} + (1 \mid \mathit{CV})$"

→
> "$\mathit{score} \sim \mathit{gender} \times \mathit{origin} \times \mathit{cond} + (\mathit{gender} + \mathit{origin}) \times (\mathit{role} + \mathit{lang}) + (1 \mid \mathit{CV}) + (1 \mid \mathit{name})$, with a logistic counterpart for screening and top-3 selection"

Then shorten "tested with the Wilcoxon signed-rank test and a mixed-effects model" to "tested with a mixed-effects model with bootstrap confidence intervals".

**E8 — Oversight terminology and honesty about the simulation.**
> "Under \emph{human-in-the-loop} oversight, decisions with a flagged quote or a score near the shortlist cutoff go to a human reviewer before they count."

→
> "Under selective \emph{human-in-the-loop} oversight, decisions with a flagged quote or a score near the shortlist cutoff are escalated to a human reviewer; we count escalations as the cost of oversight and review a sample by hand."

**E9 — Close the confounds (soundness).** §5 Data, after "No real personal data is used.", add:
> "All candidates state the right to work in Portugal and the same Portuguese proficiency, so origin is not confounded with legal or language requirements; a manipulation check confirms that names signal the intended gender and origin."

**E10 — Ranking quality.**
> "Quality is measured as the Spearman correlation between scores and the true quality level, and as pairwise accuracy (strong ranked above weak), comparing C0 with C1--C3."

→
> "Quality is measured as the Spearman correlation between Evaluator scores and the true quality level, and as the Ranker's pairwise accuracy (strong above weak) on a small set of mixed-quality pools, comparing C0 with C1--C2."

Also, in Success criteria, change "at most ${\sim}5$ percentage points of ranking quality" → "at most ${\sim}5$ percentage points of pairwise accuracy".

**E11 — Human audit design (RQ4, course link to Lecture #7).**
> "and Cohen's $\kappa$ between the judge and a human on about 50 audited justifications."

→
> "and Cohen's $\kappa$ between the judge and two human raters on 50 justifications stratified by the judge's flag, with human--human $\kappa$ as the reference."

**E12 — Bibliography hygiene (low priority).**
- Add URLs to `dastin2018amazon` and `yin2024bloomberg`.
- Add pages and DOI to `raghavan2020mitigating`.
- Add the date and ELI to `euaiact2024`.
- Add `hleg2019ethics` and an EEOC four-fifths source.
- Replace or qualify the Kusner citation (§4.3).

**E13 — Optional space savers.**
- Delete "Agents follow the standard definition of Russell and Norvig~\shortcite{russell2020aima}." (also removes a defence question). Keep R&N in the bib only if it is cited elsewhere.
- Delete "They have also documented a motherhood penalty~\cite{correll2007motherhood}." while parenthood remains a stretch goal.

---

### Sources consulted for verification
- [arXiv 2405.04412 — Armstrong et al., The Silicon Ceiling](https://arxiv.org/pdf/2405.04412)
- [arXiv 2407.20371 — Wilson & Caliskan](https://arxiv.org/html/2407.20371v1)
- [arXiv 1906.09208 — Raghavan et al.](https://arxiv.org/pdf/1906.09208)
- [arXiv 2609.19530 — Agent-mediated résumé screening](https://arxiv.org/pdf/2609.19530)
- [arXiv 2608.26899 — Counterfactual bias testing for ATS](https://arxiv.org/pdf/2608.26899)
- [arXiv 2601.11379 — Evaluating LLM behavior in hiring](https://arxiv.org/abs/2601.11379)
- [arXiv 2507.02087 — Promise and pitfalls of LLMs in hiring](https://arxiv.org/abs/2507.02087)

---

## 6. Applied after the grilling session (2026-11-09)
Agreed with the student and applied to `ijcai27.tex` and `ijcai27.bib`. The PDF was rebuilt and is still 3 pages, now in Times (NimbusRomNo9L).
- **E1 font:** fixed.
- **E2 HCAI requirements:** named in §1, with an HLEG 2019 citation added.
- **E3 title:** kept as is (student's choice).
- **E4 scope:** 2 models (commercial + open-weight) kept in the core; C3 moved to stretch; **3 roles × 10 base CVs** = 480 CVs; Screener kept as written.
- **Null-result clause:** added.
- **E5 milestones:** 15/19/22/24/27 Nov.
- **E6–E11:** applied, with mixed-quality pools for Ranker quality and a third-family model for CV generation and judging.
- **E12 bibliography:**
  - URLs added to the Reuters and Bloomberg entries.
  - EU AI Act date and ELI added.
  - Raghavan pages 469–481 and DOI added (checked via Crossref).
  - EEOC and HLEG entries added.
  - Kusner citation reworded as "inspired by".
- **E13:** Correll and Russell & Norvig removed.
- **Still open:**
  - Orchestration (LangGraph vs plain Python): deferred.
  - No 2025–26 multi-agent hiring paper added to Background, because none has been checked yet.
