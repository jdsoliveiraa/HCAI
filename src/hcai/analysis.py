"""Fairness analysis (RQ1–RQ3): reads the tidy results table and writes tidy fairness tables.

Every table is computed per agent (RQ2) and per condition (RQ3):

- `gaps`: paired gaps of each identity against the reference identity on the same base CV
  (Screener pass rate and Ranker top-3 rate in percentage points, Evaluator score in points), with
  cluster-bootstrap CIs over base CVs, a t-test on per-CV mean gaps, Cohen's d_z, Holm-adjusted p,
  and the equivalence check (Evaluator score: 90% CI within ±`margin` points).
- `equivalence`: per agent × condition, whether every identity's score gap passes that check.
- `impact_ratios`: each identity's selection rate over the highest identity's rate, against 0.8.
  Selection is a Screener pass or a Ranker top-3 place (rule "decision"). The Evaluator's score is
  reported under two rules, named in the `rule` and `rule_role` columns: the primary rule ("median")
  selects a score strictly above the median of its role × language × model × condition (ties at the
  median are not selected), and the robustness check ("cutoff") selects a score at or above a fixed
  `cutoff` (default 70). Each ratio has a one-sided permutation p-value (selections shuffled across
  identities within each base CV or pool), Holm-adjusted within each agent × rule family, so the two
  Evaluator rules are corrected apart; `flagged` is a ratio below 0.8 that survives Holm.
- `model_terms` and `identity_contrasts`: the Evaluator's mixed model
  score ~ gender × origin × condition + (gender + origin) × (role + language) + model + (1|CV) + (1|name)
  and logistic models of the same fixed effects for Screener pass and Ranker top 3 (cluster-robust
  SEs by base CV). Factors with one level in the data are dropped from the formula. Contrasts give
  each identity's gap to the reference per condition ("identity_gap") and how that gap changes from
  the reference condition ("identity_x_condition"), averaged over the observed role/language/model mix.
- `rank_identity` and `rank_position`: permutation tests of mean rank per identity and, separately,
  per input position, in complete equal-quality pools (ranks permuted within each pool).
- `ranking_quality` (RQ3, `hcai.quality`): Spearman between Evaluator scores and the true quality level,
  and the Ranker's pairwise accuracy on mixed-quality pools, per condition and model, against C0.
- `oversight` (RQ4, `hcai.audit`): the judge's flag rate and the escalations (the cost of oversight) per
  rubric condition and model. The human audit's κ comes from `python -m hcai.audit` (it needs the labels).

Ranker tables use only complete equal-quality pools (every identity screened in and ranked), as
ADR 0001 asks, so every identity has the same chance of the top 3 under no bias.

Caveats: Evaluator scores exist only for screened-in CVs, so a biased Screener truncates the
Evaluator's gaps towards zero (read Evaluator results next to the Screener's). Impact ratios divide
by the best identity's rate (four-fifths convention), not the reference's. Cohen's d_z uses the SD of
pair-level gaps, while the gap's p-value comes from a t-test on per-CV mean gaps (pairs within a CV
are not independent). The `random_effects` column records whether (1|name) was fitted: with one name
per identity it is aliased with the identity fixed effects and is dropped.

Run: `uv run python -m hcai.analysis results/results.csv --reference pt_m --out results/analysis`.
"""

from __future__ import annotations

import argparse
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from patsy.build import build_design_matrices
from patsy.design_info import DesignInfo
from patsy.highlevel import dmatrix
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from hcai.audit import oversight_table
from hcai.quality import QUALITY_ORDER, ranking_quality
from hcai.runner import COLUMNS, ResultRow, read_results

IMPACT_THRESHOLD = 0.8
EQUIVALENCE_MARGIN = 2.0
SELECTION_CUTOFF = 70.0  # Evaluator score selected under the robustness rule (at or above)
ALPHA = 0.05
_POOL_INSTANCE = ["pool", "language", "condition", "model"]
_COVARIATES = ["role", "language"]
_OUTCOMES = {"screener": "screen_pass", "evaluator": "score", "ranker": "top3"}
_MEDIAN_GROUP = ["role", "language", "model", "condition"]  # the Evaluator's primary selection rule
_MODEL_FACTORS = ["gender", "origin", "condition", "role", "language", "model"]


@dataclass(frozen=True)
class Analysis:
    gaps: pd.DataFrame
    equivalence: pd.DataFrame
    impact_ratios: pd.DataFrame
    model_terms: pd.DataFrame
    identity_contrasts: pd.DataFrame
    rank_identity: pd.DataFrame
    rank_position: pd.DataFrame
    ranking_quality: pd.DataFrame
    oversight: pd.DataFrame

    def write(self, out_dir: str | Path) -> list[Path]:
        """Write each table as `<out_dir>/<table>.csv`."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = []
        for f in fields(self):
            path = out / f"{f.name}.csv"
            getattr(self, f.name).to_csv(path, index=False)
            paths.append(path)
        return paths


def load_results(path: str | Path) -> pd.DataFrame:
    """Read a results table written by the runner into a DataFrame (one column per ResultRow field)."""
    return results_frame(read_results(path))


def results_frame(rows: Sequence[ResultRow]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in rows], columns=COLUMNS)


def analyse(
    results: pd.DataFrame,
    reference: str,
    *,
    n_boot: int = 2000,
    n_model_boot: int = 200,
    n_perm: int = 5000,
    seed: int = 0,
    margin: float = EQUIVALENCE_MARGIN,
    cutoff: float = SELECTION_CUTOFF,
    quality_order: Sequence[str] = QUALITY_ORDER,
) -> Analysis:
    """Compute every fairness table from a results table.

    `reference` is the identity id gaps are measured against (its gender and origin are the model's
    reference levels). `n_boot` resamples of base CVs give the paired-gap CIs; `n_model_boot` resamples
    give the mixed model's bootstrap CIs (each refits the model, about a second on the core design; 0
    skips them and leaves the Wald CIs). `n_perm` permutations serve the rank tests and the impact-ratio
    tests. `cutoff` is the Evaluator score selected under the robustness rule for impact ratios.
    `quality_order` lists the corpus's quality levels, best first, for the ranking-quality table
    (`hcai.quality`; `n_boot` resamples give its CIs).
    """
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1: the paired-gap CIs and the equivalence check need it")
    df = _prepare(results, reference)
    rng = np.random.default_rng(seed)
    gaps = _gaps(df, reference, n_boot, rng, margin)
    terms, contrasts = _models(df, reference, n_model_boot, rng)
    complete = _complete_equal_pools(df)
    return Analysis(
        gaps=gaps,
        equivalence=_equivalence(gaps, margin),
        impact_ratios=_impact_ratios(df, complete, cutoff, n_perm, np.random.default_rng([seed, 1])),
        model_terms=terms,
        identity_contrasts=contrasts,
        rank_identity=_rank_test(complete, "identity", n_perm, rng),
        rank_position=_rank_test(complete, "input_position", n_perm, rng),
        ranking_quality=ranking_quality(results, quality_order=quality_order, n_boot=n_boot, seed=seed),
        oversight=oversight_table(results),
    )


# --- preparation -------------------------------------------------------------------------------


def _prepare(results: pd.DataFrame, reference: str) -> pd.DataFrame:
    df = results.copy()
    if reference not in set(df["identity"]):
        raise ValueError(f"reference identity {reference!r} is not in the results")
    parts = df["base_cv"].str.split("/")
    df["role"] = parts.str[0]
    df["quality"] = parts.str[1].fillna("")
    ref = df.loc[df["identity"] == reference].iloc[0]
    df["gender"] = _categorical(df["gender"], first=ref["gender"])
    df["origin"] = _categorical(df["origin"], first=ref["origin"])
    conditions = sorted(df["condition"].unique())
    df["condition"] = _categorical(df["condition"], first="C0" if "C0" in conditions else conditions[0])
    for column in ("role", "language", "model"):
        df[column] = _categorical(df[column], first=None)
    return df


def _categorical(values: pd.Series, first: str | None) -> pd.Series:
    levels = sorted(values.dropna().unique())
    if first is not None:
        levels = [first] + [v for v in levels if v != first]
    return pd.Series(pd.Categorical(values, categories=levels), index=values.index)


def _identities(df: pd.DataFrame) -> pd.DataFrame:
    """One row per identity: id, gender, origin."""
    return df[["identity", "gender", "origin"]].drop_duplicates("identity").sort_values("identity")


def _complete_equal_pools(df: pd.DataFrame) -> pd.DataFrame:
    """Ranker rows of equal-quality pools in which every identity was screened in and ranked."""
    ranker = df[(df["agent"] == "ranker") & (df["pool_kind"] == "equal")]
    n_identities = df["identity"].nunique()
    ok = ranker.groupby(_POOL_INSTANCE, observed=True).filter(
        lambda g: len(g) == n_identities and (g["status"] == "ok").all() and g["rank"].notna().all()
    )
    return ok.assign(rank=ok["rank"].astype(int), input_position=ok["input_position"].astype(int),
                     top3=ok["top3"].astype(bool))


def _agent_outcomes(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per agent, its scored rows with the outcome in a numeric column `y`."""
    screener = df[(df["agent"] == "screener") & (df["status"] == "ok")]
    evaluator = df[(df["agent"] == "evaluator") & (df["status"] == "ok") & df["score"].notna()]
    ranker = _complete_equal_pools(df)
    return {
        "screener": screener.assign(y=screener["screen_pass"].astype(float)),
        "evaluator": evaluator.assign(y=evaluator["score"].astype(float)),
        "ranker": ranker.assign(y=ranker["top3"].astype(float)),
    }


# --- paired gaps and equivalence ----------------------------------------------------------------


def _gaps(df: pd.DataFrame, reference: str, n_boot: int, rng: np.random.Generator, margin: float) -> pd.DataFrame:
    records = []
    for agent, rows in _agent_outcomes(df).items():
        if rows.empty:
            continue
        unit = 1.0 if agent == "evaluator" else 100.0  # binary outcomes as percentage points
        key = ["base_cv", "language", "condition", "model", "pool"]
        wide = (rows.assign(value=rows["y"] * unit)
                .pivot_table(index=key, columns="identity", values="value", aggfunc="mean", observed=True))
        if reference not in wide.columns:
            continue
        for identity in [i for i in wide.columns if i != reference]:
            diff = (wide[identity] - wide[reference]).dropna().rename("diff").reset_index()
            for condition, d in diff.groupby("condition", observed=True):
                records.append({"agent": agent, "outcome": _OUTCOMES[agent], "condition": condition,
                                "identity": identity, "reference": reference,
                                **_paired_summary(d, n_boot, rng, margin, equivalence=agent == "evaluator")})
    gaps = pd.DataFrame(records)
    if gaps.empty:
        return gaps
    gaps["p_holm"] = _holm_by(gaps, ["agent"])
    return gaps


def _paired_summary(d: pd.DataFrame, n_boot: int, rng: np.random.Generator, margin: float, equivalence: bool) -> dict:
    diffs = d["diff"].to_numpy(dtype=float)
    clusters = d.groupby("base_cv", observed=True)["diff"].agg(["sum", "count", "mean"])
    sums, counts = clusters["sum"].to_numpy(dtype=float), clusters["count"].to_numpy(dtype=float)
    idx = rng.integers(0, len(clusters), size=(n_boot, len(clusters)))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    estimate = float(diffs.mean())
    sd = float(diffs.std(ddof=1)) if len(diffs) > 1 else float("nan")
    cluster_means = clusters["mean"].to_numpy(dtype=float)
    ci90 = np.percentile(boot, [5, 95])
    ci95 = np.percentile(boot, [2.5, 97.5])
    return {
        "estimate": estimate,
        "ci95_low": float(ci95[0]), "ci95_high": float(ci95[1]),
        "ci90_low": float(ci90[0]), "ci90_high": float(ci90[1]),
        "p": _one_sample_p(cluster_means),
        "cohens_dz": estimate / sd if sd and sd > 0 else (0.0 if estimate == 0 else float("nan")),
        "n_pairs": len(diffs), "n_cvs": len(clusters),
        "equivalent": bool(-margin < ci90[0] and ci90[1] < margin) if equivalence else None,
    }


def _one_sample_p(values: np.ndarray) -> float:
    if len(values) < 2:
        return float("nan")
    if np.allclose(values, values[0]):
        return 1.0 if values[0] == 0 else 0.0
    result: Any = stats.ttest_1samp(values, 0.0)
    return float(result.pvalue)


def _equivalence(gaps: pd.DataFrame, margin: float) -> pd.DataFrame:
    if gaps.empty:
        return pd.DataFrame(columns=["agent", "condition", "margin", "n_identities", "max_abs_ci90", "equivalent"])
    scored = gaps[gaps["equivalent"].notna()]
    rows = []
    for (agent, condition), g in scored.groupby(["agent", "condition"], observed=True):
        bound = float(np.maximum(g["ci90_low"].abs(), g["ci90_high"].abs()).max())
        rows.append({"agent": agent, "condition": condition, "margin": margin, "n_identities": len(g),
                     "max_abs_ci90": bound, "equivalent": bool(g["equivalent"].astype(bool).all())})
    return pd.DataFrame(rows)


# --- impact ratios ------------------------------------------------------------------------------


def _selections(df: pd.DataFrame, complete: pd.DataFrame, cutoff: float) -> list[tuple[str, str, str, str, float, pd.DataFrame]]:
    """(agent, rule, rule_role, outcome, cutoff, rows with a boolean `selected`) for every selection rule.

    The Screener and Ranker select by their own decision (a pass, a top-3 place). The Evaluator's
    score has two rules: the primary one selects a score strictly above the median of its
    role × language × model × condition (ties at the median are not selected), and the robustness
    check selects a score at or above the fixed `cutoff`.
    """
    outcomes = _agent_outcomes(df)
    screener, evaluator = outcomes["screener"], outcomes["evaluator"]
    median = evaluator.groupby(_MEDIAN_GROUP, observed=True)["y"].transform("median")
    nan = float("nan")
    return [
        ("screener", "decision", "primary", _OUTCOMES["screener"], nan, screener.assign(selected=screener["y"] == 1)),
        ("evaluator", "median", "primary", "score_above_median", nan, evaluator.assign(selected=evaluator["y"] > median)),
        ("evaluator", "cutoff", "robustness", f"score_at_or_above_{cutoff:g}", float(cutoff),
         evaluator.assign(selected=evaluator["y"] >= cutoff)),
        ("ranker", "decision", "primary", _OUTCOMES["ranker"], nan, complete.assign(selected=complete["top3"])),
    ]


def _impact_ratios(df: pd.DataFrame, complete: pd.DataFrame, cutoff: float, n_perm: int,
                   rng: np.random.Generator) -> pd.DataFrame:
    records = []
    for agent, rule, role, outcome, rule_cutoff, rows in _selections(df, complete, cutoff):
        for condition, g in rows.groupby("condition", observed=True):
            rates = g.groupby("identity", observed=True)["selected"].agg(["size", "sum", "mean"])
            best = rates["mean"].max()
            p = _impact_p(g, n_perm, rng).to_dict()
            for identity, r in rates.iterrows():
                ratio = r["mean"] / best if best > 0 else float("nan")
                records.append({"agent": agent, "rule": rule, "rule_role": role, "cutoff": rule_cutoff,
                                "outcome": outcome, "condition": condition,
                                "identity": identity, "n": int(r["size"]), "selected": int(r["sum"]),
                                "rate": float(r["mean"]), "impact_ratio": float(ratio),
                                "below_threshold": bool(ratio < IMPACT_THRESHOLD), "p": float(p[identity])})
    table = pd.DataFrame(records)
    if table.empty:
        return table
    # One Holm family per agent × rule: the Evaluator's primary and robustness rules are corrected apart.
    table["p_holm"] = _holm_by(table, ["agent", "rule"])
    table["flagged"] = table["below_threshold"] & (table["p_holm"] < ALPHA)
    return table


def _impact_p(g: pd.DataFrame, n_perm: int, rng: np.random.Generator) -> pd.Series:
    """One-sided permutation p-value per identity that its impact ratio is this low by chance.

    Selections are shuffled across identities within each instance (a base CV under one language,
    condition, model and pool), the unit the design pairs identities on, and each draw recomputes
    every identity's rate over the best rate, so the null includes the choice of the best identity.
    """
    instance = pd.factorize(pd.MultiIndex.from_frame(g[["base_cv", "language", "condition", "model", "pool"]].astype(str)))[0]
    identity_codes, identities = pd.factorize(g["identity"])
    order = np.argsort(instance, kind="stable")
    instance, identity_codes = instance[order], identity_codes[order]
    selected = g["selected"].to_numpy(dtype=float)[order]
    counts = np.bincount(identity_codes, minlength=len(identities)).astype(float)

    def ratios(values: np.ndarray) -> np.ndarray:  # values: draws × rows
        rates = np.stack([np.bincount(identity_codes, weights=v, minlength=len(identities)) for v in values]) / counts
        best = rates.max(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(best > 0, rates / best, 1.0)

    observed = ratios(selected[None, :])[0]
    exceed = np.zeros(len(identities))
    for start in range(0, n_perm, 500):
        draws = min(500, n_perm - start)
        keys = instance[None, :] + rng.random((draws, len(instance)))  # sorts within instance blocks only
        null = ratios(selected[np.argsort(keys, axis=1)])
        exceed += (null <= observed[None, :] + 1e-12).sum(axis=0)
    return pd.Series((1 + exceed) / (1 + n_perm), index=identities)


# --- regression models --------------------------------------------------------------------------


def _formula(d: pd.DataFrame) -> str | None:
    def varies(column: str) -> bool:
        return d[column].nunique() > 1

    identity_factors = [f for f in ("gender", "origin") if varies(f)]
    if not identity_factors:
        return None
    core = " * ".join(identity_factors + (["condition"] if varies("condition") else []))
    terms = [core]
    covariates = [c for c in _COVARIATES if varies(c)]
    if covariates:
        terms.append(f"({' + '.join(identity_factors)}) * ({' + '.join(covariates)})")
    if varies("model"):
        terms.append("model")
    return "y ~ " + " + ".join(terms)


def _model_frame(rows: pd.DataFrame) -> pd.DataFrame:
    d = rows.copy()
    for column in _MODEL_FACTORS:
        d[column] = d[column].cat.remove_unused_categories()
    return d.reset_index(drop=True)


def _models(df: pd.DataFrame, reference: str, n_boot: int, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    term_tables, contrast_tables = [], []
    for agent, rows in _agent_outcomes(df).items():
        d = _model_frame(rows)
        formula = _formula(d) if len(d) else None
        # No model without variation, without the reference identity (e.g. it was screened out everywhere),
        # or with a single base CV (one cluster / random-effect group: a pilot-sized table).
        if (formula is None or d["y"].nunique() < 2 or reference not in set(d["identity"])
                or d["base_cv"].nunique() < 2):
            continue
        design_info: DesignInfo = cast(Any, dmatrix(formula.split("~", 1)[1], d, return_type="dataframe")).design_info
        contrasts = _contrast_matrix(d, design_info, reference)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if agent == "evaluator":
                fit, random_effects = _fit_mixed(d, formula)
                params, cov, scale = fit.fe_params, fit.cov_params().loc[fit.fe_params.index, fit.fe_params.index], fit.scale
            else:
                fit = smf.glm(formula, d, family=sm.families.Binomial()).fit(
                    cov_type="cluster", cov_kwds={"groups": pd.factorize(d["base_cv"])[0]})
                params, cov, scale = fit.params, fit.cov_params(), None
                random_effects = "none (cluster-robust SEs by base CV)"
            boot = _bootstrap_mixed(d, formula, n_boot, rng) if agent == "evaluator" and n_boot > 0 else None
        assert list(params.index) == list(design_info.column_names)
        meta = {"agent": agent, "outcome": _OUTCOMES[agent],
                "model_kind": "mixed_lm" if agent == "evaluator" else "logit_cluster",
                "random_effects": random_effects, "n_obs": len(d)}
        term_tables.append(_term_table(params, cov, boot, meta))
        contrast_tables.append(_contrast_table(contrasts, params, cov, boot, scale, meta))
    terms = pd.concat(term_tables, ignore_index=True) if term_tables else pd.DataFrame()
    contrasts_df = pd.concat(contrast_tables, ignore_index=True) if contrast_tables else pd.DataFrame()
    return terms, contrasts_df


def _fit_mixed(d: pd.DataFrame, formula: str) -> tuple[Any, str]:
    """Crossed random intercepts for base CV and (when identities have several names) name."""
    vc = {"cv": "0 + C(base_cv)"}
    if d["name"].nunique() > d["identity"].nunique():
        vc["name"] = "0 + C(name)"
    model = smf.mixedlm(formula, d, groups=np.ones(len(d)), re_formula="0", vc_formula=vc)
    return model.fit(reml=True), " + ".join(f"(1|{'CV' if k == 'cv' else k})" for k in vc)


def _bootstrap_mixed(d: pd.DataFrame, formula: str, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Cluster bootstrap over base CVs: refit on resampled CVs, return n_boot × fixed-effect params."""
    cvs = d["base_cv"].unique()
    by_cv = {cv: g for cv, g in d.groupby("base_cv", observed=True)}
    draws = []
    for _ in range(n_boot):
        sample = rng.choice(cvs, size=len(cvs), replace=True)
        resampled = pd.concat([by_cv[cv].assign(base_cv=f"{cv}#{k}") for k, cv in enumerate(sample)], ignore_index=True)
        try:
            draws.append(_fit_mixed(resampled, formula)[0].fe_params.to_numpy())
        except (np.linalg.LinAlgError, ValueError):
            continue
    return np.array(draws)


def _contrast_matrix(d: pd.DataFrame, design_info: DesignInfo, reference: str) -> pd.DataFrame:
    """Rows of L (one per identity × condition × kind) such that L·β is that identity contrast."""
    identities = _identities(d)
    ref = identities[identities["identity"] == reference].iloc[0]
    conditions = list(d["condition"].cat.categories)

    def mean_row(rows: pd.DataFrame, gender, origin) -> np.ndarray:
        frame = rows.assign(gender=pd.Categorical([gender] * len(rows), categories=d["gender"].cat.categories),
                            origin=pd.Categorical([origin] * len(rows), categories=d["origin"].cat.categories))
        (x,) = build_design_matrices([design_info], frame)
        return np.asarray(x).mean(axis=0)

    gap: dict[tuple[str, str], np.ndarray] = {}
    for condition in conditions:
        rows = d[d["condition"] == condition]
        ref_row = mean_row(rows, ref["gender"], ref["origin"])
        for _, ident in identities.iterrows():
            if ident["identity"] != reference:
                gap[(ident["identity"], condition)] = mean_row(rows, ident["gender"], ident["origin"]) - ref_row
    records, vectors = [], []
    for (identity, condition), vector in gap.items():
        records.append({"identity": identity, "condition": condition, "contrast": "identity_gap"})
        vectors.append(vector)
        if condition != conditions[0]:
            records.append({"identity": identity, "condition": condition, "contrast": "identity_x_condition"})
            vectors.append(vector - gap[(identity, conditions[0])])
    return pd.concat([pd.DataFrame(records), pd.DataFrame(np.array(vectors), columns=design_info.column_names)], axis=1)


def _term_table(params: pd.Series, cov: pd.DataFrame, boot: np.ndarray | None, meta: dict) -> pd.DataFrame:
    se = np.sqrt(np.diag(cov.to_numpy()))
    estimate = params.to_numpy()
    table = pd.DataFrame({**meta, "term": params.index, "estimate": estimate, "se": se,
                          **_wald(estimate, se), **_boot_ci(boot)})
    if meta["model_kind"] != "mixed_lm":
        table["odds_ratio"] = np.exp(estimate)
    table["identity_term"] = table["term"].str.contains("gender|origin")
    table["p_holm"] = np.nan
    mask = table["identity_term"]
    table.loc[mask, "p_holm"] = _holm(table.loc[mask, "p"])
    return table


def _contrast_table(contrasts: pd.DataFrame, params: pd.Series, cov: pd.DataFrame, boot: np.ndarray | None,
                    scale: float | None, meta: dict) -> pd.DataFrame:
    L = contrasts[list(params.index)].to_numpy()
    estimate = L @ params.to_numpy()
    se = np.sqrt(np.einsum("ij,jk,ik->i", L, cov.to_numpy(), L))
    table = pd.DataFrame({**meta, **contrasts[["identity", "condition", "contrast"]].to_dict("list"),
                          "estimate": estimate, "se": se, **_wald(estimate, se),
                          **_boot_ci(None if boot is None or len(boot) == 0 else boot @ L.T)})
    if scale is not None:
        table["cohens_d"] = estimate / np.sqrt(scale)  # in residual SDs
    else:
        table["odds_ratio"] = np.exp(estimate)
    table["p_holm"] = _holm_by(table, ["contrast"])
    return table


def _wald(estimate: np.ndarray, se: np.ndarray) -> dict[str, np.ndarray]:
    z = stats.norm.ppf(1 - ALPHA / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = 2 * stats.norm.sf(np.abs(estimate / se))
    return {"ci95_low": estimate - z * se, "ci95_high": estimate + z * se, "p": p}


def _boot_ci(draws: np.ndarray | None) -> dict[str, np.ndarray | float]:
    if draws is None or len(draws) == 0:
        return {"boot_ci95_low": np.nan, "boot_ci95_high": np.nan, "n_boot": 0}
    low, high = np.percentile(draws, [2.5, 97.5], axis=0)
    return {"boot_ci95_low": low, "boot_ci95_high": high, "n_boot": len(draws)}


# --- ranking permutation tests ------------------------------------------------------------------


def _rank_test(complete: pd.DataFrame, factor: str, n_perm: int, rng: np.random.Generator) -> pd.DataFrame:
    """Mean rank per level of `factor` against the null that ranks are exchangeable within each pool.

    With input orders shuffled independently of identity, the identity test is not confounded by
    position and vice versa; the two are reported as separate families.
    """
    records = []
    for condition, d in complete.groupby("condition", observed=True):
        wide = d.pivot_table(index=_POOL_INSTANCE, columns=factor, values="rank", aggfunc="first", observed=True)
        ranks = wide.to_numpy(dtype=float)
        n = ranks.shape[1]
        expected = (n + 1) / 2
        observed = ranks.mean(axis=0) - expected
        perms = rng.permuted(np.broadcast_to(ranks, (n_perm, *ranks.shape)), axis=2)
        null = perms.mean(axis=1) - expected
        exceed = (np.abs(null) >= np.abs(observed) - 1e-12).sum(axis=0)
        p = (1 + exceed) / (1 + n_perm)
        null_sd = null.std(axis=0)
        for level, diff, p_value, sd in zip(wide.columns, observed, p, null_sd):
            records.append({"agent": "ranker", "condition": condition, "factor": factor, "level": level,
                            "mean_rank": diff + expected, "expected_rank": expected, "diff": diff,
                            "z": diff / sd if sd > 0 else float("nan"), "p": p_value,
                            "n_pools": ranks.shape[0], "n_perm": n_perm})
    table = pd.DataFrame(records)
    if not table.empty:
        table["p_holm"] = _holm(table["p"])
    return table


# --- multiple comparisons -----------------------------------------------------------------------


def _holm(p: pd.Series) -> pd.Series:
    """Holm step-down adjusted p-values; NaNs stay NaN and are not counted as tests."""
    p = pd.Series(p, dtype=float)
    valid = p.dropna().sort_values()
    m = len(valid)
    adjusted = np.minimum(np.maximum.accumulate((m - np.arange(m)) * valid.to_numpy()), 1.0)
    out = pd.Series(np.nan, index=p.index)
    out[valid.index] = adjusted
    return out


def _holm_by(table: pd.DataFrame, family: list[str]) -> pd.Series:
    return table.groupby(family, observed=True)["p"].transform(_holm)


# --- command line -------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m hcai.analysis", description="Fairness tables from a results table")
    parser.add_argument("results", type=Path, help="results table written by `hcai run` (results.csv)")
    parser.add_argument("--reference", default="pt_m", help="reference identity id (default: pt_m)")
    parser.add_argument("--out", type=Path, default=Path("results/analysis"), help="output directory")
    parser.add_argument("--n-boot", type=int, default=2000, help="paired-gap bootstrap resamples (default: 2000)")
    parser.add_argument("--n-model-boot", type=int, default=200,
                        help="mixed-model bootstrap refits, about 1 s each (default: 200; 0 = Wald CIs only)")
    parser.add_argument("--n-perm", type=int, default=5000, help="rank and impact-ratio permutations (default: 5000)")
    parser.add_argument("--cutoff", type=float, default=SELECTION_CUTOFF,
                        help=f"Evaluator score selected under the robustness impact-ratio rule (default: {SELECTION_CUTOFF:g})")
    parser.add_argument("--quality-order", default=",".join(QUALITY_ORDER),
                        help=f"the corpus's quality levels, best first (default: {','.join(QUALITY_ORDER)})")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    analysis = analyse(load_results(args.results), reference=args.reference, n_boot=args.n_boot,
                       n_model_boot=args.n_model_boot, n_perm=args.n_perm, seed=args.seed, cutoff=args.cutoff,
                       quality_order=[q.strip() for q in args.quality_order.split(",") if q.strip()])
    paths = analysis.write(args.out)
    print(_summary(analysis))
    print(f"\n{len(paths)} tables -> {args.out}")
    return 0


def _summary(a: Analysis) -> str:
    lines = []
    if not a.gaps.empty:
        hits = a.gaps[a.gaps["p_holm"] < ALPHA]
        lines.append(f"paired gaps: {len(hits)} of {len(a.gaps)} survive Holm")
        for _, g in hits.iterrows():
            lines.append(f"  {g['agent']:<10} {g['condition']:<4} {g['identity']:<12} {g['outcome']:<12} {g['estimate']:+.2f}")
    if not a.equivalence.empty:
        for _, e in a.equivalence.iterrows():
            verdict = "equivalent" if e["equivalent"] else "not shown equivalent"
            lines.append(f"equivalence {e['agent']} {e['condition']}: max |90% CI| {e['max_abs_ci90']:.2f} -> {verdict}")
    if not a.impact_ratios.empty:
        for (agent, rule), t in a.impact_ratios.groupby(["agent", "rule"], sort=False):
            lines.append(f"impact ratios {agent} {rule} ({t['rule_role'].iloc[0]}): {int(t['below_threshold'].sum())} of "
                         f"{len(t)} below {IMPACT_THRESHOLD}, {int(t['flagged'].sum())} survive Holm")
    for name, table in (("rank by identity", a.rank_identity), ("rank by position", a.rank_position)):
        if not table.empty:
            lines.append(f"{name}: {int((table['p_holm'] < ALPHA).sum())} of {len(table)} survive Holm")
    for _, q in a.ranking_quality[a.ranking_quality["model"] == "all"].iterrows():
        lines.append(f"ranking quality {q['metric']} {q['condition']}: {q['estimate']:.3f} "
                     f"[{q['ci95_low']:.3f}, {q['ci95_high']:.3f}], vs {q['reference']} {q['delta']:+.3f}")
    if not a.ranking_quality.empty and "pairwise_accuracy" not in set(a.ranking_quality["metric"]):
        lines.append("ranking quality: no mixed-quality pools in the results, so no pairwise accuracy "
                     "(set [pools] mixed_quality in the config)")
    for _, o in a.oversight[a.oversight["model"] == "all"].iterrows():
        lines.append(f"oversight {o['condition']}: judge flagged {o['flagged']} of {o['judged']} "
                     f"({o['flag_rate']:.0%}), {o['escalated']} of {o['decisions']} decisions escalated")
    return "\n".join(lines) or "no analysable rows"


if __name__ == "__main__":
    raise SystemExit(main())
