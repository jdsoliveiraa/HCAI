"""Paired score-gap report: each identity's score minus the reference identity's, on the same base CV."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean

from hcai.runner import ResultRow


@dataclass(frozen=True)
class Gap:
    identity: str
    language: str
    condition: str
    model: str
    agent: str
    mean_gap: float
    n_pairs: int


def score_gaps(rows: list[ResultRow], reference: str) -> list[Gap]:
    """Mean paired gap (identity − reference) per identity × language × condition × model × agent.

    A pair is the same base CV under the same language, condition, model and agent. Pairs where
    either score is missing are skipped.
    """
    scores: dict[tuple, dict[str, int]] = defaultdict(dict)
    identities: list[str] = []
    for r in rows:
        if r.identity not in identities:
            identities.append(r.identity)
        if r.score is not None:
            scores[(r.base_cv, r.language, r.condition, r.model, r.agent)][r.identity] = r.score

    diffs: dict[tuple, list[int]] = defaultdict(list)
    for (_, language, condition, model, agent), by_identity in scores.items():
        if reference not in by_identity:
            continue
        for identity, score in by_identity.items():
            if identity != reference:
                diffs[(identity, language, condition, model, agent)].append(score - by_identity[reference])

    order = {i: n for n, i in enumerate(identities)}
    keys = sorted(diffs, key=lambda k: (k[3], k[2], k[1], k[4], order[k[0]]))
    return [Gap(*k, mean_gap=float(mean(diffs[k])), n_pairs=len(diffs[k])) for k in keys]


def format_report(gaps: list[Gap], reference: str) -> str:
    lines = [f"Paired score gap vs. reference identity '{reference}' (negative = scored lower)", ""]
    header = f"{'model':<28} {'cond':<5} {'lang':<5} {'agent':<10} {'identity':<12} {'gap':>7} {'pairs':>6}"
    lines += [header, "-" * len(header)]
    for g in gaps:
        lines.append(
            f"{g.model:<28} {g.condition:<5} {g.language:<5} {g.agent:<10} {g.identity:<12} {g.mean_gap:>+7.1f} {g.n_pairs:>6}"
        )
    if not gaps:
        lines.append("(no complete pairs)")
    return "\n".join(lines)
