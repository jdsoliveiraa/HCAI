# HCAI

Auditing demographic bias in a multi-agent LLM hiring pipeline: synthetic CVs that differ only in identity go through an LLM crew, and the gaps are measured and mitigated.

## Corpus

**Base CV**:
One generated CV of a role at a quality level, written as an identity-free template.
_Avoid_: Template CV, source CV

**Quality level**:
The ground-truth strength of a base CV: strong, average or weak.
_Avoid_: Grade, tier

**Borderline**:
A base CV that meets every mandatory requirement of its job ad, but only just.
_Avoid_: Edge case, marginal CV

**Identity**:
One gender × origin combination, carried by a name, a nationality, pronouns and grammatical gender.
_Avoid_: Persona, profile, demographic

**Variant**:
A base CV rendered for one identity in one language.
_Avoid_: Counterfactual CV, copy

## Realism review

**Realism review**:
The researcher's hand review of whether the corpus reads like real job ads and CVs in the Portuguese market, done once on the pilot corpus before it is frozen.
_Avoid_: CV review, hand check

**Review sample**:
The job ads (in every language) and the CV variants chosen, by a fixed seeded rule, for the realism review.

**Review item**:
One job ad in one language, or one CV variant, in the review sample.

**Checklist**:
The five checks of a review item: realistic, quality matches label, meets mandatory requirements, language OK, no stray identity cues (job ads use only the first and fourth).

**Verdict**:
The reviewer's overall decision on whether the corpus is realistic enough to freeze.

## Project site

**Project site**:
The static site generated from the repository: the place to do the realism review and the navigation hub for the project, next to the notebook and the final report.
_Avoid_: Dashboard, website
