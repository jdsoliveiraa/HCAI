"""Per-model analysis (`hcai reproduce --per-model`): the fairness tables of each pipeline model on its own.

Post hoc, added for the weak-model run (`data/runs/weak-v1`), not pre-registered. The pooled tables of
`hcai.analysis` mix the models; here every model's rows are analysed alone, with the same functions, the same
exclusion rule (ADR 0012), the same reference and the same seeded generators as the pooled run. Holm adjusts
within each model's family (one model = one family), as the pooled analysis adjusts within its own.

Tables (in `tables/per_model/`, one row set per model, a `model` column first):

- `gaps`, `equivalence`, `rank_identity`, `rank_position`: `hcai.analysis` on the model's rows (the mixed
  models are not refitted by bootstrap: `n_model_boot = 0`).
- `gaps_reference_free`, `gaps_subsets`, `rank_adjusted`, `rank_adjusted_joint`, `repeat_noise`,
  `noise_vs_gap`: `hcai.extra` on the model's rows (and its repeats).
- `screener_pass`: pass rate per identity.
- `rank_adjusted`, `rank_adjusted_joint` add `input_order_only`: a model that returns every equal-quality pool in
  its input order leaves the position regression no residual, so its tests are blanked (NaN), not reported.
- `outputs`: per model × agent, the rows by status, rows that needed more than one attempt, failed rubric
  answers, and from the run's `calls.jsonl.gz` (when present) the distinct requests that got a malformed answer
  (`parse_error`) or failed (any other outcome but `ok`; rate-limit waits are not attempts).
- `summary`: one row per model with the numbers FINDINGS.md cites.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hcai.analysis import ALPHA, analyse, failed_answers
from hcai.extra import extra_tables

AGENTS = ("screener", "evaluator", "ranker")


def call_outcomes(calls: str | Path, conditions: set[str] | None = None) -> pd.DataFrame:
    """Distinct requests (cache keys) per model × agent × outcome in a `calls.jsonl[.gz]` log.

    A request replayed from the cache or logged by two runs counts once per outcome, so a pilot that a later run
    replays is not counted twice. Rate-limit waits are not attempts and are left out. With `conditions`, only the
    calls of those conditions count.
    """
    path = Path(calls)
    opener = gzip.open if path.suffix == ".gz" else open
    seen: set[tuple[str, str, str, str]] = set()
    with opener(path, "rt") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("outcome") == "rate_limited" or (conditions is not None and r.get("condition") not in conditions):
                continue
            seen.add((str(r.get("model")), str(r.get("agent")), str(r.get("outcome")), str(r.get("key"))))
    table = pd.DataFrame(sorted(seen), columns=["model", "agent", "outcome", "key"])
    return table.groupby(["model", "agent", "outcome"]).size().rename("requests").reset_index()


def output_validity(results: pd.DataFrame, calls: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per model × agent: rows by status, rows retried, failed rubric answers, and malformed or failed requests."""
    failed = failed_answers(results)
    records = []
    for (model, agent), g in results.groupby(["model", "agent"], sort=True):
        if agent not in AGENTS:
            continue
        # a Ranker call's attempts sit on every row of its pool: count one per pool instance
        calls_rows = g[g["input_position"] == 1] if agent == "ranker" else g
        rec: dict[str, Any] = {
            "model": model, "agent": agent, "rows": len(g),
            "ok": int((g["status"] == "ok").sum()), "missing": int((g["status"] == "missing").sum()),
            "rejected": int((g["status"] == "rejected").sum()),
            "decisions": len(calls_rows),
            "retried": int((calls_rows["attempts"].fillna(1) > 1).sum()),
            "failed_answers": int(failed[g.index].sum()),
        }
        if calls is not None:
            c = calls[(calls["model"] == model) & (calls["agent"] == agent)]
            rec["requests_ok"] = int(c.loc[c["outcome"] == "ok", "requests"].sum())
            rec["requests_parse_error"] = int(c.loc[c["outcome"] == "parse_error", "requests"].sum())
            rec["requests_failed"] = int(c.loc[~c["outcome"].isin(["ok", "parse_error"]), "requests"].sum())
        records.append(rec)
    return pd.DataFrame(records)


def per_model_tables(results: pd.DataFrame, repeats: pd.DataFrame | None, reference: str, borderline: list[str],
                     *, calls: pd.DataFrame | None = None, n_boot: int = 2000, n_perm: int = 5000, seed: int = 0,
                     near_margin: float = 5.0) -> dict[str, pd.DataFrame]:
    """Every per-model table, by file name (see the module doc)."""
    parts: dict[str, list[pd.DataFrame]] = {}

    def add(name: str, model: str, table: pd.DataFrame) -> None:
        if "model" in table.columns:  # e.g. repeat_noise: per model and "all", the same rows for one model
            table = table[table["model"] == model]
        if not table.empty:
            parts.setdefault(name, []).append(table.assign(model=model)[["model", *table.columns.drop("model", errors="ignore")]])

    for model in sorted(results["model"].dropna().unique()):
        rows = results[results["model"] == model].reset_index(drop=True)
        reps = repeats[repeats["model"] == model].reset_index(drop=True) if repeats is not None else None
        a = analyse(rows, reference=reference, n_boot=n_boot, n_model_boot=0, n_perm=n_perm, seed=seed)
        for name in ("gaps", "equivalence", "rank_identity", "rank_position"):
            add(name, model, getattr(a, name))
        extra = extra_tables(rows, reps, a.gaps, reference, borderline, n_boot=n_boot, seed=seed,
                             near_margin=near_margin)
        for name, table in extra.items():
            if name.startswith("extra_equivalence"):
                continue  # the CIs are in the gap tables; the summary gives the equivalence bounds
            if name.startswith("extra_rank_adjusted"):
                table = _flag_input_order(table, input_order_share(rows))
            add(name.removeprefix("extra_"), model, table)
        s = rows[(rows["agent"] == "screener") & (rows["status"] == "ok")]
        add("screener_pass", model, s.groupby(["condition", "identity"])["screen_pass"]
            .agg(n="size", passed=lambda x: int(x.astype(bool).sum())).reset_index()
            .assign(pass_rate=lambda t: t["passed"] / t["n"]))
    tables = {name: pd.concat(ts, ignore_index=True) for name, ts in parts.items()}
    tables["outputs"] = output_validity(results, calls)
    tables["summary"] = summary(results, tables)
    return tables


def input_order_share(results: pd.DataFrame) -> float:
    """The share of equal-quality Ranker pools (per pool × language × condition × model) ranked in input order."""
    rk = results[(results["agent"] == "ranker") & (results["pool_kind"] == "equal") & (results["status"] == "ok")]
    if rk.empty:
        return float("nan")
    same = (rk["rank"] == rk["input_position"]).groupby(
        [rk["pool"], rk["language"], rk["condition"], rk["model"]], observed=True).all()
    return float(same.mean())


def _flag_input_order(table: pd.DataFrame, share: float) -> pd.DataFrame:
    """A Ranker that returns every pool in input order leaves the position regression no residual: its identity
    estimates are 0 with SEs of about 1e-15, so the t and F tests are numerical noise. Blank them and say why."""
    if table.empty:
        return table
    degenerate = share == 1.0
    table = table.assign(input_order_only=degenerate)
    if degenerate:
        for column in ("se", "ci95_low", "ci95_high", "p", "p_holm", "f", "position_wald_chi2"):
            if column in table.columns:
                table[column] = np.nan
    return table


def compare_run(run: str | Path, conditions: set[str], reference: str, borderline: list[str], *,
                n_boot: int = 2000, n_perm: int = 5000, seed: int = 0, near_margin: float = 5.0) -> pd.DataFrame:
    """The per-model `summary` of another frozen run (e.g. data/runs/core-v1), on its rows of `conditions` only."""
    from hcai.analysis import load_results  # the runner's CSV contract

    folder = Path(run)
    results = load_results(folder / "results.csv")
    results = results[results["condition"].isin(conditions)].reset_index(drop=True)
    repeats = None
    if (folder / "repeats.csv").exists():
        repeats = load_results(folder / "repeats.csv")
        repeats = repeats[repeats["condition"].isin(conditions)].reset_index(drop=True)
    calls = call_outcomes(folder / "calls.jsonl.gz", conditions) if (folder / "calls.jsonl.gz").exists() else None
    t = per_model_tables(results, repeats, reference, borderline, calls=calls, n_boot=n_boot, n_perm=n_perm,
                         seed=seed, near_margin=near_margin)
    return t["summary"]


def _get(table: pd.DataFrame | None, model: str, **match: Any) -> pd.DataFrame:
    if table is None or table.empty:
        return pd.DataFrame()
    mask = table["model"] == model
    for column, value in match.items():
        mask &= table[column] == value
    return table[mask]


def summary(results: pd.DataFrame, t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per model: the headline numbers of its tables."""
    records = []
    for model in sorted(results["model"].dropna().unique()):
        rows = results[results["model"] == model]
        ev = rows[(rows["agent"] == "evaluator") & (rows["status"] == "ok")]["score"].astype(float)
        gaps = _get(t.get("gaps"), model)
        ev_gaps = gaps[gaps["agent"] == "evaluator"] if not gaps.empty else gaps
        largest = (ev_gaps.iloc[int(np.argmax(ev_gaps["estimate"].abs().to_numpy()))]
                   if not ev_gaps.empty else None)
        eq = _get(t.get("equivalence"), model, agent="evaluator")
        free = _get(t.get("gaps_reference_free"), model)
        free_ev = free[free["agent"] == "evaluator"] if not free.empty else free
        sp = rows[(rows["agent"] == "screener") & (rows["status"] == "ok")]["screen_pass"].astype(float)
        spi = _get(t.get("screener_pass"), model)
        pos = _get(t.get("rank_position"), model)
        joint = _get(t.get("rank_adjusted_joint"), model, outcome="rank")
        adj = _get(t.get("rank_adjusted"), model)
        adj_rank = adj[adj["outcome"] == "rank"] if not adj.empty and "outcome" in adj else adj
        noise = _get(t.get("repeat_noise"), model)
        nvg = _get(t.get("noise_vs_gap"), model, contrast="vs_reference")
        out = _get(t.get("outputs"), model)
        records.append({
            "model": model,
            "evaluator_n": int(ev.size), "evaluator_mean": ev.mean(), "evaluator_sd": ev.std(),
            "evaluator_distinct_scores": int(ev.nunique()),
            "largest_gap_identity": None if largest is None else largest["identity"],
            "largest_gap": None if largest is None else largest["estimate"],
            "largest_gap_ci95": None if largest is None else f"[{largest['ci95_low']:.2f}, {largest['ci95_high']:.2f}]",
            "evaluator_gap_range": None if ev_gaps.empty else f"{ev_gaps['estimate'].min():+.2f} to {ev_gaps['estimate'].max():+.2f}",
            "equivalence_max_abs_ci90": None if eq.empty else float(eq["max_abs_ci90"].iloc[0]),
            "equivalent": None if eq.empty else bool(eq["equivalent"].iloc[0]),
            "gap_tests": len(gaps), "gaps_surviving_holm": int((gaps["p_holm"] < ALPHA).sum()) if not gaps.empty else 0,
            "vs_mean_evaluator_range": None if free_ev.empty
            else f"{free_ev['estimate'].min():+.2f} to {free_ev['estimate'].max():+.2f}",
            "vs_mean_surviving_holm": int((free["p_holm"] < ALPHA).sum()) if not free.empty else 0,
            "screener_pass_rate": sp.mean() if sp.size else np.nan,
            "screener_pass_min_identity": spi["pass_rate"].min() if not spi.empty else np.nan,
            "screener_pass_max_identity": spi["pass_rate"].max() if not spi.empty else np.nan,
            "rank_first_position": _level(pos, 1), "rank_last_position": _level(pos, pos["level"].astype(int).max())
            if not pos.empty else np.nan,
            "rank_position_surviving_holm": int((pos["p_holm"] < ALPHA).sum()) if not pos.empty else 0,
            "ranker_input_order_share": input_order_share(rows),
            "rank_identity_after_position_range": None if adj_rank.empty
            else f"{adj_rank['estimate'].min():+.2f} to {adj_rank['estimate'].max():+.2f}",
            "rank_identity_joint_p": None if joint.empty else float(joint["p"].iloc[0]),
            "repeat_sigma_score": None if noise.empty else float(noise["sigma_score"].iloc[0]),
            "repeat_same_score": None if noise.empty else float(noise["same_score"].iloc[0]),
            "gaps_inside_noise": None if nvg.empty else f"{int(nvg['inside_noise'].astype(bool).sum())} of {len(nvg)}",
            "missing_rows": int(out["missing"].sum()) if not out.empty else 0,
            "retried_decisions": int(out["retried"].sum()) if not out.empty else 0,
            "parse_error_requests": int(out["requests_parse_error"].sum())
            if not out.empty and "requests_parse_error" in out else None,
            "failed_requests": int(out["requests_failed"].sum()) if not out.empty and "requests_failed" in out else None,
        })
    return pd.DataFrame(records)


def _level(pos: pd.DataFrame, level: int) -> float:
    if pos.empty:
        return np.nan
    hit = pos[pos["level"].astype(int) == level]
    return float(hit["mean_rank"].iloc[0]) if not hit.empty else np.nan
