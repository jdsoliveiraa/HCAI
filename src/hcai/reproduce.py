"""The master command (#13): every table, figure and the human-audit export from one frozen run, offline.

`hcai reproduce data/runs/core-v1` reads only the frozen run folder (results.csv, stability.json,
proxy_leaks.json, run_summary.json, core.toml), the frozen corpus and name bank the config names (resolved
from `configs/`, where the config was used), and the raters' label sheets. It never reads `results/` or the
LLM cache, and it makes no network call: a guard refuses every socket connection and name lookup while it
runs, so a step that tried to reach a provider would fail, not spend.

It writes to `<run>/analysis/` (or `--out`):

- `tables/`: the fairness tables of `hcai.analysis` (gaps, gaps_by, equivalence, impact_ratios, model_terms,
  identity_contrasts, model_notes, exclusions, rank_identity, rank_position, ranking_quality, oversight), with
  failed rubric answers counted as missing (`hcai.analysis.failed_answers`); `sensitivity_as_run/` has the
  gaps, equivalence check, impact ratios and contrasts with those rows kept as run; plus
  `data_accounting` (rows per agent × condition × model × status, and complete equal-quality pools),
  `screener_pass` (pass counts per identity), `stability` and `proxy_leaks` (the run's own summaries, as CSV).
- `figures/`: the paper figures (`hcai.figures`), PDF and PNG.
- `audit/`: the human-audit export (50 justifications, half flagged; `audit_items.csv`, `audit_items.json`
  for the site's audit page, `audit_key.csv`, `INSTRUCTIONS.md`), and once label sheets exist
  (`docs/audit/labels/labels_*.csv`), `agreement.csv` (κ per justification) and `agreement_criteria.csv`
  (κ per quote).
- `tables/extra_*.csv`: the post-hoc analyses of `hcai.extra` (not pre-registered): gaps against the mean of the
  identities, gaps in the borderline and near-cutoff subsets, the position-adjusted Ranker, and the gaps against
  the run-to-run noise of the repeat sample (`repeats.csv`). They use their own seeded generators, so the
  pre-registered tables do not change.
- `headline.json`: the numbers FINDINGS.md cites, per research question (`extra_not_preregistered` for the above).
- `MANIFEST.txt`: the sha256 of every file written, so a rerun can be compared byte for byte.

FINDINGS.md in the same folder is written by hand and is never touched by the command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import socket
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hcai.analysis import ALPHA, Analysis, analyse, load_results
from hcai.audit import AUDIT_SIZE, AuditLabelError, export_audit, rater_agreement, read_key
from hcai.config import load_design
from hcai.extra import NEAR_MARGIN, borderline_cvs, extra_headline, extra_tables
from hcai.figures import write_figures

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABELS = ROOT / "docs" / "audit" / "labels"
DEFAULT_CONFIG_BASE = ROOT / "configs"


class NetworkBlocked(RuntimeError):
    """A network connection was attempted while `hcai reproduce` runs offline."""


@contextmanager
def no_network() -> Iterator[None]:
    """Refuse every socket connection and DNS lookup inside the block (local Unix sockets too)."""
    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise NetworkBlocked("hcai reproduce makes no network calls")

    saved = (socket.socket.connect, socket.socket.connect_ex, socket.create_connection, socket.getaddrinfo)
    socket.socket.connect = refuse  # type: ignore[method-assign]
    socket.socket.connect_ex = refuse  # type: ignore[method-assign]
    socket.create_connection = refuse  # type: ignore[assignment]
    socket.getaddrinfo = refuse  # type: ignore[assignment]
    try:
        yield
    finally:
        (socket.socket.connect, socket.socket.connect_ex, socket.create_connection,  # type: ignore[method-assign]
         socket.getaddrinfo) = saved  # type: ignore[assignment]


def reproduce(run_dir: str | Path, out: str | Path | None = None, *, labels: str | Path = DEFAULT_LABELS,
              config_base: str | Path = DEFAULT_CONFIG_BASE, reference: str = "pt_m", n_boot: int = 2000,
              n_model_boot: int = 200, n_perm: int = 5000, seed: int = 0,
              audit_n: int = AUDIT_SIZE) -> dict[str, Any]:
    """Run every step on the frozen run in `run_dir`; return the headline numbers (also `headline.json`)."""
    run = Path(run_dir)
    target = Path(out) if out is not None else run / "analysis"
    for stale in ("tables", "figures", "audit"):  # outputs of an earlier run that this one may no longer write
        if (target / stale).is_dir():
            shutil.rmtree(target / stale)
    with no_network():
        results = load_results(run / "results.csv")
        a = analyse(results, reference=reference, n_boot=n_boot, n_model_boot=n_model_boot, n_perm=n_perm, seed=seed)
        tables = target / "tables"
        written = a.write(tables)
        extra = {"data_accounting": data_accounting(results, a), "screener_pass": screener_pass(results),
                 "stability": _json_table(run / "stability.json"), "proxy_leaks": _json_table(run / "proxy_leaks.json")}
        for name, table in extra.items():
            path = tables / f"{name}.csv"
            table.to_csv(path, index=False)
            written.append(path)
        # Sensitivity: the same paired gaps, equivalence check and contrasts with the failed answers kept as run.
        as_run = analyse(results, reference=reference, n_boot=n_boot, n_model_boot=0, n_perm=n_perm, seed=seed,
                         exclude_failed=False)
        sensitivity = tables / "sensitivity_as_run"
        sensitivity.mkdir(parents=True, exist_ok=True)
        for name in ("gaps", "equivalence", "impact_ratios", "identity_contrasts"):
            path = sensitivity / f"{name}.csv"
            getattr(as_run, name).to_csv(path, index=False)
            written.append(path)
        repeats = load_results(run / "repeats.csv") if (run / "repeats.csv").exists() else None
        borderline, near_margin = _corpus_facts(run, Path(config_base))
        post_hoc = extra_tables(results, repeats, a.gaps, reference, borderline, n_boot=n_boot, seed=seed,
                                near_margin=near_margin)
        for name, table in post_hoc.items():
            path = tables / f"{name}.csv"
            table.to_csv(path, index=False)
            written.append(path)
        written += write_figures(a, target / "figures")
        audit = _audit(results, run, target / "audit", Path(labels), Path(config_base), audit_n, seed)
        written += audit.pop("paths")
        headline = headline_numbers(a, results, run, audit, extra["stability"])
        headline["sensitivity_as_run"] = {
            "excluded_rows": a.exclusions.drop(columns=["rationale"]).to_dict("records"),
            "equivalence": as_run.equivalence[["agent", "condition", "max_abs_ci90", "equivalent"]].to_dict("records"),
            "gaps_surviving_holm": as_run.gaps.loc[as_run.gaps["p_holm"] < ALPHA,
                                                   ["agent", "condition", "identity", "estimate", "p_holm"]].to_dict("records"),
        }
        headline["extra_not_preregistered"] = {"borderline_cvs": borderline, "near_margin": near_margin,
                                               **extra_headline(post_hoc)}
        path = target / "headline.json"
        path.write_text(json.dumps(headline, indent=1, default=_jsonable) + "\n")
        written.append(path)
        manifest = target / "MANIFEST.txt"
        manifest.write_text("".join(f"{_sha256(p)}  {p.relative_to(target).as_posix()}\n"
                                    for p in sorted(written, key=lambda p: p.as_posix())))
    return headline


# --- extra tables -------------------------------------------------------------------------------


def data_accounting(results: pd.DataFrame, a: Analysis) -> pd.DataFrame:
    """Rows per agent × condition × model × status, and the equal-quality pools the Ranker tests keep."""
    counts = (results.groupby(["agent", "condition", "model", "status"], dropna=False).size()
              .rename("rows").reset_index())
    equal = results[(results["agent"] == "ranker") & (results["pool_kind"] == "equal")]
    pools = equal.groupby(["pool", "language", "condition", "model"]).size().reset_index()
    kept = a.rank_identity.groupby("condition")["n_pools"].first() if not a.rank_identity.empty else pd.Series(dtype=int)
    pool_rows = []
    for condition, g in pools.groupby("condition"):
        pool_rows.append({"agent": "ranker", "condition": condition, "model": "all",
                          "status": "equal-quality pools, complete (used) / all",
                          "rows": f"{int(kept.get(condition, 0))} / {len(g)}"})
    return pd.concat([counts.astype({"rows": object}), pd.DataFrame(pool_rows)], ignore_index=True)


def screener_pass(results: pd.DataFrame) -> pd.DataFrame:
    s = results[(results["agent"] == "screener") & (results["status"] == "ok")]
    table = s.groupby(["condition", "model", "identity"])["screen_pass"].agg(n="size", passed="sum").reset_index()
    table["passed"] = table["passed"].astype(int)
    table["rejected"] = table["n"] - table["passed"]
    return table


def _corpus_facts(run: Path, config_base: Path) -> tuple[list[str], float]:
    """The corpus's borderline base CVs and the escalation margin of the run's config (empty, default if absent)."""
    config = run / "core.toml"
    if not config.exists():
        return [], NEAR_MARGIN
    try:
        design = load_design(config, base=config_base)
    except (OSError, ValueError, KeyError):
        return [], NEAR_MARGIN
    return borderline_cvs(design.corpus_dir / "manifest.json"), float(design.oversight.margin)


def _json_table(path: Path) -> pd.DataFrame:
    return pd.DataFrame(json.loads(path.read_text())) if path.exists() else pd.DataFrame()


# --- human audit --------------------------------------------------------------------------------


def _audit(results: pd.DataFrame, run: Path, out: Path, labels: Path, config_base: Path, n: int,
           seed: int) -> dict[str, Any]:
    """Export the audit sample and, when label sheets exist, compute κ. Never fails the whole command."""
    info: dict[str, Any] = {"paths": [], "exported": False, "raters": [], "status": ""}
    config = run / "core.toml"
    if not config.exists():
        info["status"] = f"not exported: {config} is missing"
        return info
    try:
        paths = export_audit(results, load_design(config, base=config_base), out, n=n, seed=seed)
    except ValueError as e:
        info["status"] = f"not exported: {e}"
        return info
    info["exported"] = True
    info["paths"] += [paths.items, paths.key, paths.instructions, out / "audit_items.json"]
    sheets = sorted(labels.glob("labels_*.csv")) if labels.is_dir() else []
    info["raters"] = [p.stem for p in sheets]
    for stale in ("agreement.csv", "agreement_criteria.csv"):
        (out / stale).unlink(missing_ok=True)
    if len(sheets) < 2:
        info["status"] = f"pending: {len(sheets)} of 2 label sheets in {labels}"
        if not sheets:
            return info
    try:
        items, quotes = rater_agreement(read_key(paths.key), sheets, seed=seed)
    except AuditLabelError as e:
        info["status"] = f"label sheets refused: {e}"
        return info
    items.to_csv(out / "agreement.csv", index=False)
    info["paths"].append(out / "agreement.csv")
    if quotes is not None:
        quotes.to_csv(out / "agreement_criteria.csv", index=False)
        info["paths"].append(out / "agreement_criteria.csv")
    info["kappa"] = items.to_dict("records")
    info["kappa_per_quote"] = quotes.to_dict("records") if quotes is not None else None
    info["status"] = info["status"] or "computed"
    return info


# --- headline numbers ---------------------------------------------------------------------------


def headline_numbers(a: Analysis, results: pd.DataFrame, run: Path, audit: dict[str, Any],
                     stability: pd.DataFrame) -> dict[str, Any]:
    """The numbers per research question that FINDINGS.md cites."""
    def records(table: pd.DataFrame, columns: list[str]) -> list[dict]:
        return table[columns].to_dict("records") if not table.empty else []

    gaps = a.gaps
    evaluator = gaps[gaps["agent"] == "evaluator"] if not gaps.empty else gaps
    surviving = gaps[gaps["p_holm"] < ALPHA] if not gaps.empty else gaps
    largest = (evaluator.assign(abs=evaluator["estimate"].abs()).sort_values("abs", ascending=False)
               .groupby("condition").head(1) if not evaluator.empty else evaluator)
    contrasts = a.identity_contrasts
    primary = contrasts[(contrasts["agent"] == "evaluator") & (contrasts["spec"] == "pairing")] if len(contrasts) else contrasts
    summary = json.loads((run / "run_summary.json").read_text()) if (run / "run_summary.json").exists() else {}
    status = results.groupby("status").size().to_dict()
    ir = a.impact_ratios
    return {
        "data": {"rows": len(results), "by_status": status,
                 "missing_share": summary.get("missing", {}).get("share"),
                 "screened_out_cvs": int(((results["agent"] == "screener") & (results["screen_pass"] == False)).sum()),  # noqa: E712
                 "model_notes": a.model_notes.to_dict("records")},
        "rq1_fairness": {
            "gaps_surviving_holm": records(surviving, ["agent", "condition", "identity", "estimate", "ci95_low",
                                                       "ci95_high", "p_holm"]),
            "n_gap_tests": len(gaps),
            "largest_evaluator_gap_per_condition": records(largest, ["condition", "identity", "estimate", "ci90_low",
                                                                     "ci90_high", "p_holm"]),
            "equivalence": records(a.equivalence, ["agent", "condition", "max_abs_ci90", "equivalent"]),
            "impact_ratios_below_0_8": records(ir[ir["below_threshold"]] if not ir.empty else ir,
                                               ["agent", "rule", "condition", "identity", "impact_ratio", "p_holm",
                                                "flagged"]),
            "impact_ratios_flagged": int(ir["flagged"].sum()) if not ir.empty else 0,
            "rank_identity_surviving_holm": int((a.rank_identity["p_holm"] < ALPHA).sum()) if not a.rank_identity.empty else 0,
            "gaps_by_surviving_holm": records(a.gaps_by[a.gaps_by["p_holm"] < ALPHA] if not a.gaps_by.empty else a.gaps_by,
                                              ["agent", "scope", "stratum", "condition", "identity", "estimate", "p_holm"]),
        },
        "rq2_per_agent": {
            "rank_position": records(a.rank_position, ["condition", "level", "mean_rank", "p_holm"]),
            "screener_pass_rate": float(results.loc[(results["agent"] == "screener") & (results["status"] == "ok"),
                                                    "screen_pass"].astype(float).mean()),
        },
        "rq3_mitigation": {
            "identity_x_condition_primary": records(primary[primary["contrast"] == "identity_x_condition"]
                                                    if not primary.empty else primary,
                                                    ["condition", "identity", "estimate", "ci95_low", "ci95_high",
                                                     "boot_ci95_low", "boot_ci95_high", "p_holm"]),
            "ranking_quality": records(a.ranking_quality, ["metric", "condition", "model", "estimate", "ci95_low",
                                                           "ci95_high", "delta", "delta_ci95_low", "delta_ci95_high",
                                                           "within_tolerance"]),
        },
        "rq4_oversight": {"oversight": a.oversight.to_dict("records"),
                          "audit": {k: v for k, v in audit.items() if k != "paths"}},
        "stability": records(stability, [c for c in ("model", "condition", "language", "agent", "agreement",
                                                      "mean_abs_diff") if c in stability.columns]),
        "cost": {"allowance_total_usd": summary.get("allowance_total")},
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return str(value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- command line -------------------------------------------------------------------------------


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("run", type=Path, help="a frozen run folder, e.g. data/runs/core-v1")
    parser.add_argument("--out", type=Path, default=None, help="output folder (default: <run>/analysis)")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS,
                        help="folder of the raters' labels_*.csv sheets (default: docs/audit/labels)")
    parser.add_argument("--config-base", type=Path, default=DEFAULT_CONFIG_BASE,
                        help="folder the run's core.toml paths are resolved from (default: configs/)")
    parser.add_argument("--reference", default="pt_m")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-model-boot", type=int, default=200,
                        help="bootstrap refits of the primary mixed model (about 10 s each on the core run)")
    parser.add_argument("--n-perm", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--audit-n", type=int, default=AUDIT_SIZE)


def run_command(args: argparse.Namespace) -> int:
    if not (args.run / "results.csv").exists():
        print(f"reproduce: {args.run / 'results.csv'} not found (give a frozen run folder)", file=sys.stderr)
        return 1
    h = reproduce(args.run, args.out, labels=args.labels, config_base=args.config_base, reference=args.reference,
                  n_boot=args.n_boot, n_model_boot=args.n_model_boot, n_perm=args.n_perm, seed=args.seed,
                  audit_n=args.audit_n)
    out = args.out or args.run / "analysis"
    print(f"{h['data']['rows']} rows; {len(h['rq1_fairness']['gaps_surviving_holm'])} of "
          f"{h['rq1_fairness']['n_gap_tests']} paired gaps survive Holm")
    for e in h["rq1_fairness"]["equivalence"]:
        print(f"  equivalence {e['agent']} {e['condition']}: max |90% CI| {e['max_abs_ci90']:.2f} -> "
              f"{'equivalent' if e['equivalent'] else 'not shown equivalent'}")
    print(f"audit: {h['rq4_oversight']['audit']['status'] or 'exported'}")
    print(f"-> {out} (tables/, figures/, audit/, headline.json, MANIFEST.txt)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hcai reproduce", description=(__doc__ or "").split("\n", 1)[0])
    add_arguments(parser)
    return run_command(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
