"""The paper's figures, drawn from the analysis tables (`hcai.analysis.Analysis`). No LLM call, no network.

Every figure is written as PDF (vector, for LaTeX) and PNG (300 dpi), at IJCAI sizes: one column is 3.3 in
wide, the full text width 7 in. Fonts are serif at 7-8 pt, to match the template's Times at print size.

- `forest_models`: the Evaluator's paired score gaps per identity against the reference, per condition,
  pooled over models and within each model, with 95% (thin) and 90% (thick) cluster-bootstrap CIs and the
  ±2-point equivalence band (RQ1, RQ3).
- `forest_roles`, `forest_languages`: the same gaps within each role and each language (RQ1).
- `impact_heatmap`: impact ratios per identity × agent, per condition (RQ1, RQ2); a cell below 0.8 has a
  black frame, and a `*` when it survives Holm.
- `tradeoff`: fairness vs. quality per condition × model: the largest absolute Evaluator gap (marker; the
  vertical line goes up to the largest absolute bound of any identity's 90% CI, the quantity the
  equivalence check compares with 2 points) against the Spearman correlation of scores with the true
  quality level (horizontal line: its 95% CI) (RQ3). Pairwise accuracy is not drawn: see the table.
- `rank_position`, `rank_identity`: the Ranker's mean rank in complete equal-quality pools by input position
  and by identity, with the 95% band of the permutation null (RQ1, RQ2).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from typing import Any, cast  # noqa: E402

from matplotlib.artist import Artist  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from hcai.analysis import EQUIVALENCE_MARGIN, IMPACT_THRESHOLD, Analysis  # noqa: E402

COLUMN = 3.3  # in, IJCAI column width
FULL = 7.0  # in, IJCAI text width
# Fixed categorical order (one hue per condition, never cycled); models by marker shape.
CONDITION_COLOURS = {"C0": "#2a78d6", "C1": "#eb6834", "C2": "#1baf7a", "C3": "#6250d6"}
CONDITION_NAMES = {"C0": "C0 baseline", "C1": "C1 blinding", "C2": "C2 rubric", "C3": "C3 both"}
MODEL_MARKERS = ["o", "s", "^", "D"]
INK, MUTED, BAND = "#0b0b0b", "#52514e", "#e4e3df"
ORIGIN_LABELS = {"pt": "PT", "br": "BR", "palop": "AO", "ee": "UA"}
AGENT_LABELS = {("screener", "decision"): "Scr", ("evaluator", "median"): "Eval", ("ranker", "decision"): "Rank"}

STYLE: dict[str, Any] = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Nimbus Roman No9 L", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": MUTED,
    "ytick.color": MUTED, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.labelcolor": INK,
    "text.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 100,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "svg.hashsalt": "hcai", "pdf.compression": 6,
}


def identity_label(identity: str) -> str:
    """`br_f` -> `BR F`: origin (PT, BR, AO = Angola for PALOP, UA = Ukraine for Eastern Europe) and gender."""
    origin, _, gender = identity.rpartition("_")
    return f"{ORIGIN_LABELS.get(origin, origin.upper())} {gender.upper()}"


def write_figures(a: Analysis, out_dir: str | Path) -> list[Path]:
    """Draw every figure the tables allow into `out_dir` (`<name>.pdf` and `<name>.png`); return the paths.

    A figure whose table is empty (e.g. no complete equal-quality pools in a tiny run) is left out.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    makers: dict[str, Callable[[Analysis], Figure | None]] = {
        "forest_models": forest_models, "forest_roles": lambda x: forest_scope(x, "role"),
        "forest_languages": lambda x: forest_scope(x, "language"), "impact_heatmap": impact_heatmap,
        "tradeoff": tradeoff, "rank_position": lambda x: rank_plot(x.rank_position, "input position"),
        "rank_identity": lambda x: rank_plot(x.rank_identity, "identity"),
    }
    paths = []
    with plt.rc_context(cast(Any, STYLE)):
        for name, make in makers.items():
            fig = make(a)
            if fig is None:
                continue
            for ext in ("pdf", "png"):
                path = out / f"{name}.{ext}"
                # No creation date in the PDF, so a rerun writes the same bytes.
                fig.savefig(path, bbox_inches="tight", pad_inches=0.02,
                            metadata={"CreationDate": None} if ext == "pdf" else {"Software": None})
                paths.append(path)
            plt.close(fig)
    return paths


# --- forest plots -------------------------------------------------------------------------------


def _evaluator(table: pd.DataFrame) -> pd.DataFrame:
    return table[table["agent"] == "evaluator"] if not table.empty else table


def forest_models(a: Analysis) -> Figure | None:
    pooled = _evaluator(a.gaps)
    if pooled.empty:
        return None
    by = _evaluator(a.gaps_by)
    panels = [("Both models", pooled)]
    if not by.empty:
        panels += [(str(m), by[(by["scope"] == "model") & (by["stratum"] == m)])
                   for m in sorted(by.loc[by["scope"] == "model", "stratum"].unique())]
    return _forest(panels, FULL if len(panels) > 2 else COLUMN)


def forest_scope(a: Analysis, scope: str) -> Figure | None:
    by = _evaluator(a.gaps_by)
    if by.empty or scope not in set(by["scope"]):
        return None
    by = by[by["scope"] == scope]
    panels = [(_stratum_label(str(s)), by[by["stratum"] == s]) for s in sorted(by["stratum"].unique())]
    return _forest(panels, FULL)


def _stratum_label(stratum: str) -> str:
    """`software_engineer` -> `Software engineer`; languages `en`/`pt` -> `EN`/`PT-PT`."""
    return {"en": "EN", "pt": "PT-PT"}.get(stratum, stratum.replace("_", " ").capitalize())


def _forest(panels: Sequence[tuple[str, pd.DataFrame]], width: float) -> Figure:
    identities = sorted({i for _, t in panels for i in t["identity"]})
    conditions = sorted({c for _, t in panels for c in t["condition"]})
    reference = next(iter({r for _, t in panels for r in t["reference"]}), "")
    height = 0.75 + 0.2 * len(identities) * max(1.0, len(conditions) / 3)
    fig, axes = plt.subplots(1, len(panels), figsize=(width, height), sharey=True, squeeze=False)
    lim = max(EQUIVALENCE_MARGIN + 0.5, *(np.nanmax(np.abs(t[["ci95_low", "ci95_high"]].to_numpy()))
                                         for _, t in panels if not t.empty))
    step = 0.22
    for ax, (title, t) in zip(axes[0], panels):
        ax.axvspan(-EQUIVALENCE_MARGIN, EQUIVALENCE_MARGIN, color=BAND, lw=0, zorder=0)
        ax.axvline(0, color=MUTED, lw=0.6, zorder=1)
        for k, condition in enumerate(conditions):
            rows = t[t["condition"] == condition].set_index("identity")
            for y, identity in enumerate(identities):
                if identity not in rows.index:
                    continue
                r = rows.loc[identity]
                yy = y + (k - (len(conditions) - 1) / 2) * step
                colour = CONDITION_COLOURS.get(condition, INK)
                ax.plot([r["ci95_low"], r["ci95_high"]], [yy, yy], color=colour, lw=0.7, solid_capstyle="round")
                ax.plot([r["ci90_low"], r["ci90_high"]], [yy, yy], color=colour, lw=1.8, solid_capstyle="round")
                significant = bool(r.get("p_holm", 1.0) < 0.05)
                ax.plot(r["estimate"], yy, "o", ms=3.2, color=colour if significant else "white",
                        mec=colour, mew=0.9, zorder=3)
        ax.set_title(title)
        ax.set_xlim(-lim * 1.05, lim * 1.05)
        ax.set_yticks(range(len(identities)), [identity_label(i) for i in identities])
        ax.set_ylim(len(identities) - 0.5, -0.5)
        ax.set_xlabel(f"Score gap vs {identity_label(reference)} (points)")
        ax.grid(axis="x", color=BAND, lw=0.4)
    handles: list[Artist] = [Line2D([], [], color=CONDITION_COLOURS.get(c, INK), marker="o", ms=3.2, lw=1.8,
                          label=CONDITION_NAMES.get(c, c)) for c in conditions]
    handles.append(Rectangle((0, 0), 1, 1, color=BAND, label=f"±{EQUIVALENCE_MARGIN:g} equivalence"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(w_pad=0.6)
    return fig


# --- impact ratios ------------------------------------------------------------------------------


def impact_heatmap(a: Analysis) -> Figure | None:
    t = a.impact_ratios
    if t.empty:
        return None
    t = t[t["rule_role"] == "primary"]
    columns = [(c, agent, rule) for c in sorted(t["condition"].unique())
               for agent, rule in AGENT_LABELS if ((t["agent"] == agent) & (t["rule"] == rule)).any()]
    identities = sorted(t["identity"].unique())
    values = np.full((len(identities), len(columns)), np.nan)
    flagged = np.zeros_like(values, dtype=bool)
    for j, (c, agent, rule) in enumerate(columns):
        rows = t[(t["condition"] == c) & (t["agent"] == agent) & (t["rule"] == rule)].set_index("identity")
        for i, identity in enumerate(identities):
            if identity in rows.index:
                values[i, j] = rows.loc[identity, "impact_ratio"]
                flagged[i, j] = bool(rows.loc[identity, "flagged"])
    fig, ax = plt.subplots(figsize=(COLUMN, 0.9 + 0.2 * len(identities)))
    low = min(0.5, float(np.nanmin(values))) if np.isfinite(values).any() else 0.5
    image = ax.imshow(values, cmap="Blues", vmin=low, vmax=1.0, aspect="auto")
    for i in range(len(identities)):
        for j in range(len(columns)):
            v = values[i, j]
            if not np.isfinite(v):
                continue
            dark = v > low + 0.6 * (1 - low)
            ax.text(j, i, f"{v:.2f}".lstrip("0") + ("*" if flagged[i, j] else ""), ha="center", va="center",
                    fontsize=5.5, color="white" if dark else INK)
            if v < IMPACT_THRESHOLD:
                ax.add_patch(Rectangle((j - 0.47, i - 0.47), 0.94, 0.94, fill=False, ec=INK, lw=0.9))
    ax.set_xticks(range(len(columns)), [AGENT_LABELS[(agent, rule)] for _, agent, rule in columns],
                  rotation=90, fontsize=6)
    ax.set_yticks(range(len(identities)), [identity_label(i) for i in identities])
    conditions = sorted({c for c, _, _ in columns})
    per = len(columns) // max(1, len(conditions))
    for k, c in enumerate(conditions):
        ax.text(k * per + (per - 1) / 2, -0.8, c, ha="center", va="bottom", fontsize=7)
        if k:
            ax.axvline(k * per - 0.5, color="white", lw=2)
    ax.tick_params(length=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.03)
    bar.ax.axhline(IMPACT_THRESHOLD, color=INK, lw=0.8)
    bar.set_label("Impact ratio (framed: < 0.8)", fontsize=6.5)
    bar.ax.tick_params(labelsize=6)
    bar.outline.set_visible(False)
    fig.tight_layout()
    return fig


# --- fairness vs. quality -----------------------------------------------------------------------


def tradeoff(a: Analysis) -> Figure | None:
    quality = a.ranking_quality
    by = _evaluator(a.gaps_by)
    if quality.empty or by.empty:
        return None
    spearman = quality[quality["metric"] == "spearman"]
    by = by[by["scope"] == "model"]
    models = sorted(by["stratum"].unique())
    fig, ax = plt.subplots(figsize=(COLUMN, 2.3))
    ax.axhline(EQUIVALENCE_MARGIN, color=MUTED, lw=0.6, ls="--")
    ax.text(0.01, EQUIVALENCE_MARGIN, f"{EQUIVALENCE_MARGIN:g}-point margin", transform=ax.get_yaxis_transform(),
            ha="left", va="bottom", fontsize=6, color=MUTED)
    for m, model in enumerate(models):
        for condition, g in by[by["stratum"] == model].groupby("condition"):
            q = spearman[(spearman["model"] == model) & (spearman["condition"] == condition)]
            if q.empty:
                continue
            q = q.iloc[0]
            worst = g.loc[g["estimate"].abs().idxmax()]
            bound = float(np.maximum(g["ci90_low"].abs(), g["ci90_high"].abs()).max())
            colour = CONDITION_COLOURS.get(str(condition), INK)
            ax.plot([q["estimate"], q["estimate"]], [abs(worst["estimate"]), bound], color=colour, lw=0.7)
            ax.plot([q["ci95_low"], q["ci95_high"]], [abs(worst["estimate"])] * 2, color=colour, lw=0.7)
            ax.plot(q["estimate"], abs(worst["estimate"]), MODEL_MARKERS[m % len(MODEL_MARKERS)], ms=5,
                    color=colour, mec="white", mew=0.6, zorder=3)
    ax.set_xlabel("Quality: Spearman ρ of Evaluator score vs. true level")
    ax.set_ylabel("Largest |score gap| (points)")
    ax.set_ylim(bottom=0)
    ax.grid(color=BAND, lw=0.4)
    handles: list[Artist] = [Line2D([], [], color=CONDITION_COLOURS.get(c, INK), marker="o", lw=0, ms=4,
                          label=CONDITION_NAMES.get(c, c)) for c in sorted(by["condition"].unique())]
    handles += [Line2D([], [], color=MUTED, marker=MODEL_MARKERS[m % len(MODEL_MARKERS)], lw=0, ms=4,
                           label=model) for m, model in enumerate(models)]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.97), ncol=2, fontsize=6, handletextpad=0.2,
              columnspacing=0.8)
    fig.tight_layout()
    return fig


# --- Ranker -------------------------------------------------------------------------------------


def rank_plot(table: pd.DataFrame, factor: str) -> Figure | None:
    if table.empty:
        return None
    conditions = sorted(table["condition"].unique())
    by_identity = factor == "identity"
    levels = sorted(table["level"].unique(), key=lambda v: (str(v) if by_identity else int(v)))
    fig, ax = plt.subplots(figsize=(COLUMN, 2.2))
    expected = float(table["expected_rank"].iloc[0])
    sd = float(table["null_sd"].max()) if "null_sd" in table else 0.0
    ax.axhspan(expected - 1.96 * sd, expected + 1.96 * sd, color=BAND, lw=0, zorder=0)
    ax.axhline(expected, color=MUTED, lw=0.6)
    x = np.arange(len(levels))
    if not by_identity:
        ax.plot(x, [int(v) for v in levels], color=MUTED, lw=0.6, ls=":", label="rank = position")
    step = 0.18 if by_identity else 0.0
    for k, condition in enumerate(conditions):
        rows = table[table["condition"] == condition].set_index("level")
        y = [rows.loc[v, "mean_rank"] if v in rows.index else np.nan for v in levels]
        colour = CONDITION_COLOURS.get(condition, INK)
        xx = x + (k - (len(conditions) - 1) / 2) * step
        ax.plot(xx, y, lw=0 if by_identity else 1.2, color=colour, marker="o", ms=3.5, mec="white", mew=0.5,
                label=CONDITION_NAMES.get(condition, condition))
    ax.set_xticks(x, [identity_label(str(v)) for v in levels] if by_identity else [str(v) for v in levels])
    if by_identity:
        ax.tick_params(axis="x", labelsize=6)
    ax.set_xlabel("Identity" if by_identity else "Input position in the Ranker prompt")
    ax.set_ylabel("Mean rank (1 = best)")
    n = int(table["n_pools"].max())
    ax.set_ylim(len(levels) + 0.5, 0.5) if not by_identity else ax.set_ylim(expected + 1.5, expected - 1.5)
    ax.grid(axis="y", color=BAND, lw=0.4)
    ax.legend(loc="lower left" if not by_identity else "upper left", ncol=2, fontsize=6,
              title=f"{n} complete equal-quality pools per condition; band: 95% null", title_fontsize=5.5)
    fig.tight_layout()
    return fig
