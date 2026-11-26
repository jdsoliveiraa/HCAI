"""Extra analyses of a frozen run (post hoc, NOT pre-registered; not in the proposal).

They were added after the first findings (#13) to answer four questions the pre-registered tables leave open.
Each one reads only the results table (and, for 2 and 4, the corpus manifest and the repeat sample). Failed
rubric answers count as missing, as in `hcai.analysis` (ADR 0012). Each table has its own seeded generator, so
it does not change the pre-registered tables.

1. `reference_free_gaps`: each identity against the mean of all identities in the same pairing unit (a base CV
   under one language, condition, model and pool), instead of against one reference identity. Only complete
   units (every identity has an outcome) count, so the mean is over the same identities everywhere. The
   deviations of the identities in a unit sum to 0, so the tests are not independent; Holm holds under any
   dependence. Same bootstrap over base CVs, t-test on per-CV means and Holm family (per agent) as `gaps`.
   Equivalence: the 90% CI of each deviation within ±margin. A deviation of ±m allows a gap of up to 2m
   between two identities.
2. `subset_gaps`: the paired gaps (against the reference and against the mean) in two subsets where a small
   bias would matter most: the borderline base CVs of the corpus (`borderline` in its manifest; all agents),
   and the Evaluator's pairing units whose mean score is within `near_margin` points of the median of their
   role × language × model × condition (the escalation rule's cutoff; the unit mean treats every identity the
   same, so the selection does not favour one). Holm per subset × contrast × agent.
3. `rank_adjusted`: per condition, OLS of the Ranker's rank (and top 3, in pp) on input position and identity
   (both categorical) in complete equal-quality pools, with SEs clustered by pool (t and F tests with G − 1
   denominator degrees of freedom, G = pools, as usual for cluster-robust Wald tests). A pool's mean rank is fixed
   ((n+1)/2), so pool, model and language terms would be 0 and are not fitted. Identity effects are given
   against the mean of the identities (deviation contrasts); Holm per outcome over conditions × identities,
   plus a joint Wald test of all identity terms per condition (Holm per outcome over conditions).
4. `repeat_noise` and `noise_vs_gap`: the run-to-run SD of one Evaluator score, σ, pooled over the repeat
   sample (within-item variance of the original and its repeats), with a bootstrap CI over items. A paired gap
   of one pair has noise SD σ√2 (against the reference) or σ√((k−1)/k) (against the mean of k). Per gap:
   the gap in units of that noise SD, the noise-only SE of the mean gap, the share of the pair-to-pair variance
   that noise explains, and `inside_noise`: the gap's 90% CI lies within ± the lower 95% bound of the noise SD
   of one paired comparison (a bias smaller than what one repeat of the same call can change).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from hcai.analysis import (
    _MEDIAN_GROUP,
    _OUTCOMES,
    _POOL_INSTANCE,
    ALPHA,
    EQUIVALENCE_MARGIN,
    _agent_outcomes,
    _complete_equal_pools,
    _equivalence,
    _gaps,
    _holm_by,
    _paired_summary,
    _prepare,
    failed_answers,
)

UNIT = ["base_cv", "language", "condition", "model", "pool"]  # the pairing unit of the paired gaps
NEAR_MARGIN = 5.0  # the core run's [oversight] margin around the median cutoff
MEAN_REFERENCE = "mean of identities"
_ITEM = ["base_cv", "identity", "name", "language", "condition", "model", "agent", "pool"]


def prepare(results: pd.DataFrame, reference: str, exclude_failed: bool = True) -> pd.DataFrame:
    """The analysis frame of `hcai.analysis.analyse`: failed rubric answers counted as missing (ADR 0012)."""
    failed = failed_answers(results) if exclude_failed else pd.Series(False, index=results.index)
    results = results.assign(status=results["status"].where(~failed, "missing"), score=results["score"].where(~failed))
    return _prepare(results, reference)


# --- 1. reference-free contrast -------------------------------------------------------------------


def _deviations(rows: pd.DataFrame, unit: float) -> pd.DataFrame:
    """Per complete pairing unit, each identity's outcome minus the unit mean (wide: one column per identity)."""
    wide = (rows.assign(value=rows["y"] * unit)
            .pivot_table(index=UNIT, columns="identity", values="value", aggfunc="mean", observed=True))
    wide = wide.dropna()
    return wide.sub(wide.mean(axis=1), axis=0)


def reference_free_gaps(df: pd.DataFrame, n_boot: int, rng: np.random.Generator,
                        margin: float = EQUIVALENCE_MARGIN, holm: bool = True) -> pd.DataFrame:
    """Each identity's paired deviation from the mean of all identities, per agent and condition."""
    records = []
    for agent, rows in _agent_outcomes(df).items():
        if rows.empty:
            continue
        dev = _deviations(rows, 1.0 if agent == "evaluator" else 100.0)
        for identity in dev.columns:
            diff = dev[identity].rename("diff").reset_index()
            for condition, d in diff.groupby("condition", observed=True):
                records.append({"agent": agent, "outcome": _OUTCOMES[agent], "condition": condition,
                                "identity": identity, "reference": MEAN_REFERENCE, "n_identities": dev.shape[1],
                                **_paired_summary(d, n_boot, rng, margin, equivalence=agent == "evaluator")})
    table = pd.DataFrame(records)
    if not table.empty and holm:
        table["p_holm"] = _holm_by(table, ["agent"])
    return table


# --- 2. borderline and near-cutoff subsets -------------------------------------------------------


def borderline_cvs(manifest: str | Path) -> list[str]:
    """The base CVs the corpus manifest marks borderline (they meet every mandatory requirement, only just)."""
    path = Path(manifest)
    if not path.exists():
        return []
    return sorted(cv["id"] for cv in json.loads(path.read_text())["base_cvs"] if cv.get("borderline"))


def near_cutoff_units(df: pd.DataFrame, margin: float = NEAR_MARGIN) -> pd.DataFrame:
    """Evaluator pairing units whose mean score is within `margin` of the median of their role × language ×
    model × condition (the escalation rule's cutoff). One row per unit, with its mean and the median."""
    ev = _agent_outcomes(df)["evaluator"]
    ev = ev.assign(median=ev.groupby(_MEDIAN_GROUP, observed=True)["y"].transform("median"))
    units = ev.groupby(UNIT, observed=True).agg(unit_mean=("y", "mean"), median=("median", "first")).reset_index()
    return units[(units["unit_mean"] - units["median"]).abs() <= margin].reset_index(drop=True)


def subset_gaps(df: pd.DataFrame, reference: str, borderline: list[str], n_boot: int, rng: np.random.Generator,
                margin: float = EQUIVALENCE_MARGIN, near_margin: float = NEAR_MARGIN) -> pd.DataFrame:
    """Paired gaps (`contrast` "vs_reference" and "vs_mean") in the borderline and the near-cutoff subsets."""
    subsets: list[tuple[str, pd.DataFrame]] = []
    if borderline:
        subsets.append(("borderline", df[df["base_cv"].isin(borderline)]))
    near = near_cutoff_units(df, near_margin)
    keys = pd.MultiIndex.from_frame(near[UNIT].astype(str))
    ev = df[df["agent"] == "evaluator"]
    in_near = pd.MultiIndex.from_frame(ev[UNIT].astype(str)).isin(keys)
    subsets.append(("near_cutoff", ev[in_near]))
    tables = []
    for subset, rows in subsets:
        if rows.empty:
            continue
        for contrast, table in (("vs_reference", _gaps(rows, reference, n_boot, rng, margin, holm=False)),
                                ("vs_mean", reference_free_gaps(rows, n_boot, rng, margin, holm=False))):
            if not table.empty:
                tables.append(table.assign(subset=subset, contrast=contrast))
    if not tables:
        return pd.DataFrame()
    table = pd.concat(tables, ignore_index=True)
    table["p_holm"] = _holm_by(table, ["subset", "contrast", "agent"])
    first = ["subset", "contrast"]
    return table[first + [c for c in table.columns if c not in first]]


def subset_equivalence(table: pd.DataFrame, margin: float = EQUIVALENCE_MARGIN) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame()
    parts = [_equivalence(g, margin).assign(subset=str(s), contrast=str(c)) for (s, c), g in table.groupby(["subset", "contrast"])]
    out = pd.concat(parts, ignore_index=True)
    n = table.groupby(["subset", "contrast", "condition"], observed=True)["n_pairs"].agg(["min", "max"])
    out = out.join(n.rename(columns={"min": "n_pairs_min", "max": "n_pairs_max"}), on=["subset", "contrast", "condition"])
    first = ["subset", "contrast"]
    return out[first + [c for c in out.columns if c not in first]]


# --- 3. position-adjusted Ranker ----------------------------------------------------------------


def rank_adjusted(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Identity effects on rank and top 3 after input position, per condition (deviation from the identity mean).

    Returns (per identity, joint Wald test per condition × outcome)."""
    complete = _complete_equal_pools(df)
    rows, joint = [], []
    if complete.empty:
        return pd.DataFrame(), pd.DataFrame()
    for condition, d in complete.groupby("condition", observed=True):
        d = d.assign(identity=d["identity"].astype(str), top3_pp=d["top3"].astype(float) * 100.0,
                     cluster=pd.factorize(pd.MultiIndex.from_frame(d[_POOL_INSTANCE].astype(str)))[0])
        identities = sorted(d["identity"].unique())
        if d["cluster"].nunique() < 2 * len(identities):  # too few pools for cluster-robust SEs
            continue
        base = identities[0]
        k = len(identities)
        names = [f"C(identity, Treatment('{base}'))[T.{i}]" for i in identities[1:]]
        for outcome in ("rank", "top3_pp"):
            fit = smf.ols(f"{outcome} ~ C(input_position) + C(identity, Treatment('{base}'))", d).fit(
                cov_type="cluster", cov_kwds={"groups": d["cluster"].to_numpy()})
            b = np.concatenate([[0.0], fit.params[names].to_numpy()])
            cov = np.zeros((k, k))
            cov[1:, 1:] = fit.cov_params().loc[names, names].to_numpy()
            centre = np.eye(k) - 1.0 / k
            est, vcov = centre @ b, centre @ cov @ centre.T
            se = np.sqrt(np.diag(vcov))
            groups = int(d["cluster"].nunique())
            t_crit = float(stats.t.ppf(1 - ALPHA / 2, groups - 1))
            p = 2 * stats.t.sf(np.abs(est / se), groups - 1)  # small-sample reference: t with G − 1 df
            raw = d.groupby("identity")[outcome].mean() - d[outcome].mean()
            for i, identity in enumerate(identities):
                rows.append({"agent": "ranker", "condition": condition, "outcome": outcome, "identity": identity,
                             "reference": MEAN_REFERENCE, "unadjusted": float(raw[identity]),
                             "estimate": float(est[i]), "se": float(se[i]),
                             "ci95_low": float(est[i] - t_crit * se[i]), "ci95_high": float(est[i] + t_crit * se[i]),
                             "p": float(p[i]), "n_pools": groups, "n_obs": len(d)})
            bi = b[1:]
            vi = cov[1:, 1:]
            f = float(bi @ np.linalg.solve(vi, bi)) / (k - 1)
            joint.append({"agent": "ranker", "condition": condition, "outcome": outcome, "test": "all identity terms = 0",
                          "f": f, "df1": k - 1, "df2": groups - 1, "p": float(stats.f.sf(f, k - 1, groups - 1)),
                          "position_wald_chi2": _joint_position(fit), "n_pools": groups})
    table, joint_table = pd.DataFrame(rows), pd.DataFrame(joint)
    if table.empty:
        return table, joint_table
    table["p_holm"] = _holm_by(table, ["outcome"])
    joint_table["p_holm"] = _holm_by(joint_table, ["outcome"])
    return table, joint_table


def _joint_position(fit: Any) -> float:
    names = [n for n in fit.params.index if n.startswith("C(input_position)")]
    b = fit.params[names].to_numpy()
    return float(b @ np.linalg.solve(fit.cov_params().loc[names, names].to_numpy(), b))


# --- 4. run-to-run noise against the gaps --------------------------------------------------------


def repeat_noise(results: pd.DataFrame, repeats: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> pd.DataFrame:
    """The run-to-run SD of one Evaluator score per condition (model "all") and per condition × model.

    σ² is the pooled within-item variance of an original score and its repeats (both ok, neither a failed
    answer); for model "all" it is the mean of the models' σ² (the models weighted equally). CI: percentile
    bootstrap over items, within each model. Only the call that is repeated varies: under C1 the Blinder's text
    is the same in the original and the repeat, so σ leaves out the Blinder's own run-to-run noise."""
    def ok(frame: pd.DataFrame) -> pd.DataFrame:
        keep = (frame["agent"] == "evaluator") & (frame["status"] == "ok") & frame["score"].notna() & ~failed_answers(frame)
        return frame[keep]

    original = ok(results)[_ITEM + ["score"]]
    again = ok(repeats)[_ITEM + ["score"]]
    keys_o, keys_a = original.set_index(_ITEM).index, again.set_index(_ITEM).index
    both = pd.concat([original[keys_o.isin(keys_a)], again[keys_a.isin(keys_o)]])
    items = both.groupby(_ITEM).agg(ss=("score", lambda s: float(((s - s.mean()) ** 2).sum())),
                                    n=("score", "size"), range=("score", lambda s: float(s.max() - s.min())))
    items = items.reset_index()
    rows = []

    def variance(g: pd.DataFrame, draws: int) -> tuple[float, np.ndarray]:
        """Pooled within-item variance and `draws` bootstrap values of it (items resampled)."""
        ss, dof = g["ss"].to_numpy(), (g["n"] - 1).to_numpy(dtype=float)
        idx = rng.integers(0, len(g), size=(draws, len(g)))
        return float(ss.sum() / dof.sum()), ss[idx].sum(axis=1) / dof[idx].sum(axis=1)

    for condition, c in items.groupby("condition"):
        per_model = {model: (g, *variance(g, n_boot)) for model, g in c.groupby("model")}
        # "all": the models weighted equally (the gaps are balanced over models; the repeat sample is not)
        var_all = float(np.mean([v for _, v, _ in per_model.values()]))
        boot_all = np.mean([b for _, _, b in per_model.values()], axis=0)
        for model, (g, var, boot) in [("all", (c, var_all, boot_all)), *per_model.items()]:
            low, high = np.sqrt(np.percentile(boot, [2.5, 97.5]))
            rows.append({"condition": condition, "model": model, "items": len(g), "pairs": int((g["n"] - 1).sum()),
                         "same_score": float((g["range"] == 0).mean()), "sigma_score": float(np.sqrt(var)),
                         "sigma_ci95_low": float(low), "sigma_ci95_high": float(high)})
    return pd.DataFrame(rows)


def _pair_sd(df: pd.DataFrame, reference: str) -> pd.DataFrame:
    """SD of the Evaluator's pair-level differences per contrast × condition × identity."""
    rows = _agent_outcomes(df)["evaluator"]
    wide = rows.pivot_table(index=UNIT, columns="identity", values="y", aggfunc="mean", observed=True)
    out = []
    dev = wide.dropna().pipe(lambda w: w.sub(w.mean(axis=1), axis=0))
    for contrast, frame in (("vs_reference", wide.drop(columns=reference).sub(wide[reference], axis=0)),
                            ("vs_mean", dev)):
        for identity in frame.columns:
            for condition, s in frame[identity].dropna().groupby(level="condition", observed=True):
                out.append({"contrast": contrast, "condition": condition, "identity": identity,
                            "observed_sd_pair": float(s.std(ddof=1))})
    return pd.DataFrame(out)


def noise_vs_gap(gaps: pd.DataFrame, free: pd.DataFrame, noise: pd.DataFrame, df: pd.DataFrame,
                 reference: str) -> pd.DataFrame:
    """Each Evaluator gap (against the reference and against the mean) next to the run-to-run noise."""
    columns = ["condition", "identity", "estimate", "ci90_low", "ci90_high", "p_holm", "n_pairs"]
    ev = [g[g["agent"] == "evaluator"][columns].assign(contrast=c) for c, g in (("vs_reference", gaps), ("vs_mean", free))
          if not g.empty]
    if not ev or noise.empty:
        return pd.DataFrame()
    table = pd.concat(ev, ignore_index=True)
    table["condition"] = table["condition"].astype(str)
    k = int(free["n_identities"].iloc[0]) if not free.empty else df["identity"].nunique()
    factor = table["contrast"].map({"vs_reference": np.sqrt(2.0), "vs_mean": np.sqrt((k - 1) / k)})
    n = noise[noise["model"] == "all"].assign(condition=lambda x: x["condition"].astype(str))
    table = table.merge(n[["condition", "items", "sigma_score", "sigma_ci95_low", "sigma_ci95_high"]], on="condition")
    table["noise_sd_pair"] = table["sigma_score"] * factor
    table["noise_sd_pair_ci95_low"] = table["sigma_ci95_low"] * factor
    table["noise_sd_pair_ci95_high"] = table["sigma_ci95_high"] * factor
    table["gap_over_noise_sd"] = table["estimate"] / table["noise_sd_pair"]
    table["noise_se_mean"] = table["noise_sd_pair"] / np.sqrt(table["n_pairs"])
    sd = _pair_sd(df, reference).assign(condition=lambda x: x["condition"].astype(str))
    table = table.merge(sd, on=["contrast", "condition", "identity"], how="left")
    table["noise_share_var"] = (table["noise_sd_pair"] / table["observed_sd_pair"]) ** 2
    table["noise_share_ci95_low"] = (table["noise_sd_pair_ci95_low"] / table["observed_sd_pair"]) ** 2
    table["noise_share_ci95_high"] = (table["noise_sd_pair_ci95_high"] / table["observed_sd_pair"]) ** 2
    bound = np.maximum(table["ci90_low"].abs(), table["ci90_high"].abs())
    table["inside_noise"] = bound < table["noise_sd_pair_ci95_low"]
    first = ["contrast", "condition", "identity"]
    return table[first + [c for c in table.columns if c not in first]]


# --- all four -----------------------------------------------------------------------------------


def extra_tables(results: pd.DataFrame, repeats: pd.DataFrame | None, gaps: pd.DataFrame, reference: str,
                 borderline: list[str], *, n_boot: int = 2000, seed: int = 0, margin: float = EQUIVALENCE_MARGIN,
                 near_margin: float = NEAR_MARGIN) -> dict[str, pd.DataFrame]:
    """Every extra table, by file name. `gaps` is the pre-registered paired-gap table (against `reference`)."""
    df = prepare(results, reference)
    free = reference_free_gaps(df, n_boot, np.random.default_rng([seed, 6]), margin)
    subsets = subset_gaps(df, reference, borderline, n_boot, np.random.default_rng([seed, 7]), margin, near_margin)
    rank, rank_joint = rank_adjusted(df)
    noise = (repeat_noise(results, repeats, n_boot, np.random.default_rng([seed, 8]))
             if repeats is not None and len(repeats) else pd.DataFrame())
    return {
        "extra_gaps_reference_free": free,
        "extra_equivalence_reference_free": _equivalence(free, margin) if not free.empty else pd.DataFrame(),
        "extra_gaps_subsets": subsets,
        "extra_equivalence_subsets": subset_equivalence(subsets, margin),
        "extra_rank_adjusted": rank,
        "extra_rank_adjusted_joint": rank_joint,
        "extra_repeat_noise": noise,
        "extra_noise_vs_gap": noise_vs_gap(gaps, free, noise, df, reference),
    }


def extra_headline(t: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """The numbers FINDINGS.md cites for the extra analyses."""
    def surviving(table: pd.DataFrame, columns: list[str]) -> list[dict]:
        return table.loc[table["p_holm"] < ALPHA, columns].to_dict("records") if not table.empty else []

    free = t["extra_gaps_reference_free"]
    ev = free[free["agent"] == "evaluator"] if not free.empty else free
    subsets = t["extra_gaps_subsets"]
    noise = t["extra_noise_vs_gap"]
    out: dict[str, Any] = {
        "label": "post hoc, not pre-registered",
        "reference_free": {
            "evaluator_range_per_condition": (ev.groupby("condition", observed=True)["estimate"].agg(["min", "max"])
                                              .reset_index().to_dict("records") if not ev.empty else []),
            "surviving_holm": surviving(free, ["agent", "condition", "identity", "estimate", "ci95_low", "ci95_high",
                                               "p_holm"]),
            "n_tests": len(free),
            "equivalence": t["extra_equivalence_reference_free"].to_dict("records"),
        },
        "subsets": {
            "pairs": (subsets.groupby(["subset", "contrast", "agent", "condition"], observed=True)["n_pairs"].max()
                      .reset_index().to_dict("records") if not subsets.empty else []),
            "surviving_holm": surviving(subsets, ["subset", "contrast", "agent", "condition", "identity", "estimate",
                                                  "n_pairs", "p_holm"]),
            "n_tests": len(subsets),
            "equivalence": t["extra_equivalence_subsets"].to_dict("records"),
        },
        "rank_adjusted": {
            "surviving_holm": surviving(t["extra_rank_adjusted"], ["condition", "outcome", "identity", "estimate", "p_holm"]),
            "n_tests": len(t["extra_rank_adjusted"]),
            "joint": t["extra_rank_adjusted_joint"].to_dict("records"),
        },
        "noise": {"repeat_noise": t["extra_repeat_noise"].to_dict("records")},
    }
    if not noise.empty:
        out["noise"]["per_contrast_condition"] = (
            noise.assign(abs_gap=noise["estimate"].abs())
            .groupby(["contrast", "condition"])
            .agg(max_abs_gap=("abs_gap", "max"), noise_sd_pair=("noise_sd_pair", "first"),
                 max_gap_over_noise_sd=("gap_over_noise_sd", lambda s: float(s.abs().max())),
                 inside_noise=("inside_noise", "sum"), n=("inside_noise", "size"),
                 noise_share_min=("noise_share_var", "min"), noise_share_max=("noise_share_var", "max"))
            .reset_index().to_dict("records"))
    return out

