"""Ranking quality (RQ3): how well each condition's agents order CVs by their true quality level.

- `spearman`: the Spearman correlation between the Evaluator's scores and the true quality level of the
  base CV (strong > average > weak, the ground truth of the corpus), over every scored decision.
- `pairwise_accuracy`: on mixed-quality pools, the share of pairs of ranked CVs of different quality
  levels that the Ranker puts in the right order (the better CV ranked higher).

Both are reported per condition and model and pooled over models (`model = "all"`), with 95% cluster-
bootstrap CIs (base CVs for the Evaluator, pools for the Ranker), and against the reference condition
(C0): `delta` is the condition's value minus the reference's, its CI from the same paired resamples (the
same base CVs and pools appear under every condition). `within_tolerance` checks the pre-registered success
criterion on pairwise accuracy: the loss from the reference is at most `tolerance` (5 percentage points),
on the point estimate.

Caveats: screened-out CVs have no score and are not ranked, so both metrics cover screened-in CVs only (a
Screener that rejects weak CVs leaves fewer, closer levels to order). Equal-quality pools have no pairs of
different quality and are not used.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

QUALITY_ORDER = ("strong", "average", "weak")  # best first, as in configs/core.toml
PAIRWISE_TOLERANCE = 0.05  # pre-registered: at most ~5 pp of pairwise accuracy lost
ALL_MODELS = "all"
COLUMNS = ["metric", "agent", "condition", "model", "estimate", "ci95_low", "ci95_high", "n", "n_clusters",
           "reference", "delta", "delta_ci95_low", "delta_ci95_high", "within_tolerance", "n_boot"]

Stat = Callable[[np.ndarray], float]


def ranking_quality(results: pd.DataFrame, *, quality_order: Sequence[str] = QUALITY_ORDER, n_boot: int = 2000,
                    seed: int = 0, tolerance: float = PAIRWISE_TOLERANCE) -> pd.DataFrame:
    """The ranking-quality table of a results table (one column per `ResultRow` field).

    `quality_order` lists the corpus's quality levels, best first; rows of other levels are left out.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    level = {q: len(quality_order) - i for i, q in enumerate(quality_order)}
    df = results.assign(quality=results["base_cv"].str.split("/").str[1].map(level))
    df = df[df["quality"].notna()]
    conditions = sorted(df["condition"].unique())
    if not conditions:
        return pd.DataFrame(columns=COLUMNS)
    reference = "C0" if "C0" in conditions else conditions[0]
    rng = np.random.default_rng(seed)
    tables = [
        # Base CVs are resampled within their quality level, keeping the design's fixed level proportions
        # (Spearman against a tied level depends on them: an unstratified resample biases the CI low).
        _metric("spearman", "evaluator", _evaluator_obs(df), ["base_cv"], _spearman, reference, n_boot, rng, None,
                strata="x"),
        _metric("pairwise_accuracy", "ranker", _pair_obs(df), ["pool", "language"], _pair_share, reference,
                n_boot, rng, tolerance),
    ]
    records = [r for t in tables for r in t]
    return pd.DataFrame(records, columns=COLUMNS)


# --- observations ---------------------------------------------------------------------------------


def _evaluator_obs(df: pd.DataFrame) -> pd.DataFrame:
    """One row per scored Evaluator decision: x = true quality level, y = score, n = 1."""
    ev = df[(df["agent"] == "evaluator") & (df["status"] == "ok") & df["score"].notna()]
    return pd.DataFrame({"condition": ev["condition"], "model": ev["model"], "base_cv": ev["base_cv"],
                         "x": ev["quality"].astype(float), "y": ev["score"].astype(float), "n": 1.0})


def _pair_obs(df: pd.DataFrame) -> pd.DataFrame:
    """One row per ranked mixed-quality pool: x = pairs ordered right, y = 0, n = pairs of different quality."""
    ranker = df[(df["agent"] == "ranker") & (df["pool_kind"] == "mixed") & (df["status"] == "ok") & df["rank"].notna()]
    records = []
    for (pool, language, condition, model), g in ranker.groupby(["pool", "language", "condition", "model"]):
        q = g["quality"].to_numpy(dtype=float)
        r = g["rank"].to_numpy(dtype=float)
        better = q[:, None] > q[None, :]  # i is of higher quality than j
        pairs = int(better.sum())
        if pairs:
            records.append({"condition": condition, "model": model, "pool": pool, "language": language,
                            "x": float((better & (r[:, None] < r[None, :])).sum()), "y": 0.0, "n": float(pairs)})
    return pd.DataFrame(records, columns=["condition", "model", "pool", "language", "x", "y", "n"])


def _spearman(obs: np.ndarray) -> float:
    x, y = obs[:, 0], obs[:, 1]
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(stats.rankdata(x), stats.rankdata(y))[0, 1])


def _pair_share(obs: np.ndarray) -> float:
    pairs = obs[:, 2].sum()
    return float(obs[:, 0].sum() / pairs) if pairs else float("nan")


# --- estimates and paired bootstrap -----------------------------------------------------------------


def _metric(metric: str, agent: str, obs: pd.DataFrame, cluster: list[str], stat: Stat, reference: str,
            n_boot: int, rng: np.random.Generator, tolerance: float | None, strata: str | None = None) -> list[dict]:
    if obs.empty:
        return []
    records = []
    models = [ALL_MODELS] + sorted(obs["model"].unique())
    for model in models:
        d = obs if model == ALL_MODELS else obs[obs["model"] == model]
        records += _by_condition(d, cluster, stat, reference, n_boot, rng, tolerance,
                                 dict(metric=metric, agent=agent, model=model), strata)
    return records


def _by_condition(d: pd.DataFrame, cluster: list[str], stat: Stat, reference: str, n_boot: int,
                  rng: np.random.Generator, tolerance: float | None, meta: dict, strata: str | None) -> list[dict]:
    codes, _ = pd.factorize(pd.MultiIndex.from_frame(d[cluster].astype(str)))
    conditions = sorted(d["condition"].unique())
    cond = d["condition"].to_numpy()
    values = d[["x", "y", "n"]].to_numpy(dtype=float)
    members = [np.flatnonzero(codes == c) for c in range(codes.max() + 1)]
    # Clusters grouped by stratum (one stratum unless `strata` names a column constant within each cluster).
    stratum_of = d[strata].to_numpy() if strata else np.zeros(len(d))
    groups = [np.array([k for k in range(len(members)) if stratum_of[members[k][0]] == s])
              for s in pd.unique(stratum_of)]
    estimate = {c: stat(values[cond == c]) for c in conditions}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        draws = {c: np.full(n_boot, np.nan) for c in conditions}
        for b in range(n_boot):
            drawn = np.concatenate([g[rng.integers(0, len(g), size=len(g))] for g in groups])
            idx = np.concatenate([members[k] for k in drawn])
            for c in conditions:
                draws[c][b] = stat(values[idx][cond[idx] == c])
        records = []
        for c in conditions:
            in_c = cond == c
            low, high = _ci(draws[c])
            has_ref = reference in estimate
            delta = estimate[c] - estimate[reference] if has_ref else float("nan")
            d_low, d_high = _ci(draws[c] - draws[reference]) if has_ref else (float("nan"), float("nan"))
            records.append({
                **meta, "condition": c, "estimate": estimate[c], "ci95_low": low, "ci95_high": high,
                "n": int(values[in_c, 2].sum()), "n_clusters": len(np.unique(codes[in_c])),
                "reference": reference, "delta": delta, "delta_ci95_low": d_low, "delta_ci95_high": d_high,
                "within_tolerance": None if tolerance is None or np.isnan(delta) else bool(delta >= -tolerance - 1e-12),
                "n_boot": n_boot,
            })
    return records


def _ci(draws: np.ndarray) -> tuple[float, float]:
    finite = draws[np.isfinite(draws)]
    if len(finite) == 0:
        return float("nan"), float("nan")
    low, high = np.percentile(finite, [2.5, 97.5])
    return float(low), float(high)
