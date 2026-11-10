"""Output stability: how often a re-run of the same LLM call gives the same answer (the repeat sample)."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, stdev
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hcai.runner import ResultRow

# The outcome each agent's stability is measured on.
_OUTCOME = {"screener": "screen_pass", "evaluator": "score", "ranker": "rank"}


def _item(r: ResultRow) -> tuple:
    return (r.base_cv, r.identity, r.name, r.language, r.condition, r.model, r.agent, r.pool)


def stability_report(rows: list[ResultRow], repeats: list[ResultRow]) -> list[dict[str, Any]]:
    """Agreement of each repeat with the original answer, per model x condition x language x agent.

    Only pairs where both the original and the repeat are `ok` count. Per group:
    - `items`: sampled rows with at least one such pair; `pairs`: (original, repeat) pairs
    - `agreement`: share of pairs with the same outcome (pass/reject, score, or rank)
    - `mean_abs_diff`: mean |repeat - original| of the score or rank (None for the Screener)
    - `mean_sd`: mean over items of the SD of the original and repeated scores or ranks (None for the Screener)
    - `top3_agreement` (Ranker): share of pairs agreeing on shortlist membership
    """
    originals = {_item(r): r for r in rows if r.status == "ok"}
    by_item: dict[tuple, list[ResultRow]] = defaultdict(list)
    for r in repeats:
        if r.status == "ok" and _item(r) in originals:
            by_item[_item(r)].append(r)

    groups: dict[tuple, list[tuple[ResultRow, list[ResultRow]]]] = defaultdict(list)
    for item, again in by_item.items():
        o = originals[item]
        groups[(o.model, o.condition, o.language, o.agent)].append((o, again))

    report: list[dict[str, Any]] = []
    for (model, condition, language, agent), items in groups.items():
        field = _OUTCOME[agent]
        pairs = [(getattr(o, field), getattr(r, field)) for o, again in items for r in again]
        entry: dict[str, Any] = dict(
            model=model, condition=condition, language=language, agent=agent, items=len(items), pairs=len(pairs),
            agreement=round(mean(a == b for a, b in pairs), 4), mean_abs_diff=None, mean_sd=None,
        )
        if agent != "screener":
            entry["mean_abs_diff"] = round(mean(abs(a - b) for a, b in pairs), 4)
            entry["mean_sd"] = round(mean(stdev([getattr(o, field)] + [getattr(r, field) for r in again])
                                          for o, again in items), 4)
        if agent == "ranker":
            entry["top3_agreement"] = round(mean(o.top3 == r.top3 for o, again in items for r in again), 4)
        report.append(entry)
    return report
