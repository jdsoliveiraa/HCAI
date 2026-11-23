"""The name manipulation check as the project site shows it (ADR 0010; the check itself is ADRs 0009 and 0011).

`load_record` reads the tables `hcai names` wrote (cells.csv, summary.csv, perceptions.csv, reasked.csv) from one
directory, as text. Nothing is recomputed: every probability and every pass mark on the page is the CSV's own, a
number only rounded for display (`shown`, 3 decimals, half up). `cell_chart` draws one gender x origin cell as
inline SVG, and `reask_chains` tells how each answer that was asked again ended. `hcai.site` builds the page from
these and passes its `ui` function in, so the interface strings stay in its one EN / PT-PT table.

The chart, one per cell: a row per name x model with a grey bar for the scored p(intended gender), a stacked bar
of the scored origin shares (the hedge options are not in them, ADR 0011) and a pass or fail mark. The intended
origin is always the first segment, drawn from 0 to its exact share, so the origin minimum of the pass rule can
be read against its end; that line is drawn only where the rule applies it (not for a host origin,
`HOST_ORIGINS`). A bar is about 130 units long, so it cannot show a difference under about 0.01: at a threshold
the mark and the table say whether a row passes, not the bar. A question with no estimate has no bar and no
line, only the words. Colours are classes styled in site_assets/style.css (`nc-*`), one per origin in the order
of the CSV's columns: the four were checked pairwise for colour-vision deficiencies in both themes, since any two
origins can touch. A fifth origin would get the neutral class, never a new hue (`hcai names` writes four).
Names and models come from the CSV and are always escaped; a row's <title> reads it in the CSV's own words.
"""

from __future__ import annotations

import csv
import html
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path

from hcai.namecheck import HOST_ORIGINS, ORIGIN_LABELS, REASKED_FILE, SCORED_GENDERS

CELLS, SUMMARY, PERCEPTIONS = "cells.csv", "summary.csv", "perceptions.csv"
FILES = (CELLS, SUMMARY, PERCEPTIONS, REASKED_FILE)
# What summary.csv must have for the probability view (ADR 0011); a record without them shows its tables only.
PROBABILITY_COLUMNS = frozenset({"name", "model", "intended_gender", "intended_origin", "p_intended_gender",
                                 "p_intended_origin", "pass"})
SCORED = "scored_p_"
MISSING = "–"  # shown for a blank value (nothing to average, or a column the record does not have)
ORIGIN_CLASSES = 4  # origin colours in style.css (.nc-o1 to .nc-o4); a further origin is drawn neutral (.nc-o0)

UiFn = Callable[..., str]  # hcai.site.ui: (key, tag="span", **attributes) -> an element with the interface string


@dataclass(frozen=True)
class Table:
    header: list[str]
    rows: list[dict[str, str]]


@dataclass(frozen=True)
class NamesRecord:
    """The name check tables of one directory: the files that exist and could be read, by file name."""

    path: Path
    tables: dict[str, Table]
    problems: tuple[str, ...] = ()  # files that could not be read, to show on the page

    def rows(self, filename: str) -> list[dict[str, str]]:
        return self.tables[filename].rows if filename in self.tables else []

    @property
    def summary(self) -> list[dict[str, str]]:
        return self.rows(SUMMARY)

    @property
    def probability(self) -> bool:
        """Whether summary.csv is a probability-mode table with at least one row."""
        table = self.tables.get(SUMMARY)
        return table is not None and bool(table.rows) and PROBABILITY_COLUMNS <= set(table.header)

    @property
    def origins(self) -> list[str]:
        """The scored origin keys, in the order of summary.csv's columns."""
        return scored_origins(self.tables[SUMMARY].header) if SUMMARY in self.tables else []

    @property
    def names(self) -> list[str]:
        return list(dict.fromkeys(r.get("name", "") for r in self.summary))

    @property
    def failing(self) -> list[str]:
        """The names that do not pass for at least one model, in the file's order."""
        return list(dict.fromkeys(r.get("name", "") for r in self.summary if not is_true(r.get("pass"))))


def read_table(path: Path) -> Table:
    """A CSV file as text. A byte order mark is dropped. A file that is not comma-separated (a spreadsheet's
    semicolon or tab export) is refused: read as one column it would pass for a record with nothing in it."""
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, restval="")
        rows = [{k: v for k, v in row.items() if k is not None} for row in reader]  # None: cells past the header
        header = list(reader.fieldnames or [])
    if len(header) == 1 and (";" in header[0] or "\t" in header[0]):
        raise ValueError("not comma-separated; keep the file as hcai names wrote it (UTF-8, commas, decimal points)")
    return Table(header, rows)


def load_record(path: Path) -> NamesRecord | None:
    """The name check tables in `path`, or None when it holds no record: no summary.csv or cells.csv, or none
    with a row (an empty file is no record, so it never stands in front of another directory's results). A file
    that cannot be read is a problem of the record, to be shown, not a missing record."""
    if not (path / SUMMARY).is_file() and not (path / CELLS).is_file():
        return None
    tables, problems = {}, []
    for filename in FILES:
        if (path / filename).is_file():
            try:
                tables[filename] = read_table(path / filename)
            except (csv.Error, ValueError, OSError) as e:  # ValueError: not UTF-8, or not comma-separated
                problems.append(f"{filename}: {e}")
    record = NamesRecord(path, tables, tuple(problems))
    return record if problems or record.summary or record.rows(CELLS) else None


def scored_origins(header: Sequence[str]) -> list[str]:
    return [c.removeprefix(SCORED) for c in header if c.startswith(SCORED) and c.removeprefix(SCORED) not in SCORED_GENDERS]


def is_true(value: str | None) -> bool:
    return (value or "").strip().lower() in ("true", "1", "yes")


def number(value: str | None) -> float | None:
    """A finite number of the CSV, or None when the cell is blank or not one."""
    try:
        x = float(value or "")
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def shown(value: str | None, digits: int = 3) -> str:
    """A CSV number rounded half up to `digits` decimals, from its decimal text (no float in between)."""
    try:
        d = Decimal((value or "").strip())
        return str(d.quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)) if d.is_finite() else MISSING
    except InvalidOperation:  # blank, not a number, or too large to round
        return MISSING


def count(row: dict[str, str], key: str) -> int:
    x = number(row.get(key))
    return int(x) if x is not None else 0


def by_cell(summary: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, list[dict[str, str]]]]:
    """(intended gender, intended origin) -> name -> its rows (one per model), all in the file's order."""
    cells: dict[tuple[str, str], dict[str, list[dict[str, str]]]] = {}
    for row in summary:
        cell = cells.setdefault((row.get("intended_gender", ""), row.get("intended_origin", "")), {})
        cell.setdefault(row.get("name", ""), []).append(row)
    return cells


def fails_on(row: dict[str, str]) -> list[str]:
    """What a failing name x model fails on, in the CSV's words (as the report of `hcai names` puts it)."""
    out = [q for q in ("gender", "origin") if f"{q}_pass" in row and not is_true(row[f"{q}_pass"])]
    if count(row, "missing"):
        out.append("missing answers")
    if count(row, "gender_all_hedge") or count(row, "origin_all_hedge"):
        out.append("all-hedge answers")
    return out


def row_reading(row: dict[str, str], origins: Sequence[str]) -> str:
    """One name x model in the CSV's words: 'Rui Antunes · model-a: p(male) 0.996; portuguese 0.865, …; pass'."""
    shares = ", ".join(f"{o} {shown(row.get(SCORED + o))}" for o in origins)
    verdict = "pass" if is_true(row.get("pass")) else "fail"
    if verdict == "fail" and fails_on(row):
        verdict += f" ({', '.join(fails_on(row))})"
    return (f"{row.get('name', '')} · {row.get('model', '')}: p({row.get('intended_gender', '')}) "
            f"{shown(row.get('p_intended_gender'))}; {shares}; {verdict}")


# --- labels ----------------------------------------------------------------------------------------------------


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def origin_label(origin: str, ui: UiFn, *, short: bool = False) -> str:
    """The origin as an interface string, or its key for an origin the check has no option for: the option the
    models were asked about (`ORIGIN_LABELS`), or with `short` a one- or two-word name for headings and columns."""
    if origin not in ORIGIN_LABELS:
        return esc(origin)
    return ui(f"names.origin.short.{origin}" if short else f"names.origin.{origin}")


def gender_label(gender: str, ui: UiFn) -> str:
    return ui(f"names.gender.{gender}") if gender in SCORED_GENDERS else esc(gender)


def origin_classes(origins: Sequence[str]) -> dict[str, str]:
    """origin -> its colour class: by position among the CSV's origin columns, so it never depends on the cell."""
    return {o: f"nc-o{k + 1}" if k < ORIGIN_CLASSES else "nc-o0" for k, o in enumerate(origins)}


# --- the chart -------------------------------------------------------------------------------------------------

# Geometry in SVG user units. The chart is one column wide (W) and scales with its container: a 320px phone shows
# it at 0.9 of its size (its smallest text at about 10px), a 360px one at full size, and wider screens at up to
# 1.2; hcai.site puts the cells side by side when there is room. The text sizes are the font-size of the nc-*
# classes in style.css: change them together.
W = 320
NAME_PX, MODEL_PX, HEAD_PX = 13, 12, 11  # .nc-name, .nc-model, .nc-head / .nc-tick / .nc-none
HEAD = 32  # column titles (baseline 11) and threshold labels (baseline 26)
NAME_H, ROW_H, BAR_H, GAP = 19, 16, 10, 8  # name line, model row, bar thickness, space after a name
GENDER_W = 56
MARK_W = 20
LABEL_MIN, LABEL_MAX = 44, 108  # the model label column
SEG_GAP = 1.5  # gap in the page colour between two origin segments
CHECK, CROSS = "l3.2 3.2 6-7", "l7 7m0-7l-7 7"  # the pass and fail marks, in a 12 x 12 box
ELLIPSIS = "…"


def _n(x: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".")


def text_width(text: str, px: float, *, bold: bool = False) -> float:
    """An upper estimate of a label's width: there are no font metrics at build time, so each character counts
    as its class in a wide sans-serif (the widest a browser is likely to pick for system-ui). A label that fits
    by this estimate fits; one that does not is cut, with its full text in the row's <title> and in the table."""
    def em(ch: str) -> float:
        if ch in " .,:;'|!ijlI/-()[]":
            return 0.37
        if ch in "tfr":
            return 0.43
        if ch in "mwMW@%":
            return 1.0
        return 0.77 if ch.isupper() else 0.65
    return sum(em(ch) for ch in text) * px * (1.12 if bold else 1.0)


def fit(text: str, px: float, width: float, *, bold: bool = False, keep_end: bool = False) -> str:
    """`text` if it fits `width`, else cut with an ellipsis: at its end, or with `keep_end` in the middle, since
    the ids of one vendor's models differ at their end."""
    if text_width(text, px, bold=bold) <= width:
        return text
    room = width - text_width(ELLIPSIS, px, bold=bold)
    head_room = room / 2 if keep_end else room
    head = ""
    for ch in text:
        if text_width(head + ch, px, bold=bold) > head_room:
            break
        head += ch
    tail = ""
    if keep_end:
        for ch in reversed(text[len(head):]):
            if text_width(head + ch + tail, px, bold=bold) > room:
                break
            tail = ch + tail
    return (head.rstrip() or text[:1]) + ELLIPSIS + tail


def model_labels(models: Sequence[str], width: float) -> dict[str, str]:
    """model id -> the label drawn beside its rows: at most `width` wide, and never the same for two models (ids
    that are still alike once cut are numbered in their order), so a mark can always be told to its model."""
    labels = {m: fit(m, MODEL_PX, width, keep_end=True) for m in models}
    if len(set(labels.values())) < len(labels):
        labels = {m: fit(f"{k}: {m}", MODEL_PX, width, keep_end=True) for k, m in enumerate(models, start=1)}
    return labels


def _first(rows: list[dict[str, str]], key: str) -> float | None:
    """The first value of a rule parameter in `rows`, if it is a probability."""
    for row in rows:
        x = number(row.get(key))
        if x is not None:
            return x if 0.0 <= x <= 1.0 else None
    return None


def mark(passed: bool, x: float, y: float) -> str:
    """The pass (check) or fail (cross) mark with its top left corner at (x, y): told apart by shape, not colour."""
    if passed:
        return f'<path class="nc-mark nc-pass" d="M{_n(x + 1.5)} {_n(y + 6.5)}{CHECK}"/>'
    return f'<path class="nc-mark nc-fail" d="M{_n(x + 2.5)} {_n(y + 2.5)}{CROSS}"/>'


def cell_chart(names: dict[str, list[dict[str, str]]], *, gender: str, origin: str, origins: Sequence[str],
               uid: str, ui: UiFn) -> str:
    """One gender x origin cell as inline SVG: `names` is name -> its summary.csv rows (one per model).

    The SVG is one image for assistive technology, named by its caption (`<uid>-cap`, written by the caller) and
    its <title>, and described by its <desc>; its values are in the table the page shows after the charts.
    """
    rows = [r for name_rows in names.values() for r in name_rows]
    models = list(dict.fromkeys(r.get("model", "") for r in rows))
    widest = max((text_width(m, MODEL_PX) for m in models), default=0.0)
    label_w = min(LABEL_MAX, max(LABEL_MIN, math.ceil(widest) + 2))
    labels = model_labels(models, label_w)
    gx = label_w + 6
    ox = gx + GENDER_W + 12
    ow = W - ox - MARK_W
    order = [o for o in origins if o == origin] + [o for o in origins if o != origin]  # the intended origin first
    classes = origin_classes(origins)
    threshold = _first(rows, "gender_threshold")
    minimum = None if origin in HOST_ORIGINS else _first(rows, "origin_minimum")

    gender_title = (ui(f"names.chart.p.{gender}", "text", class_="nc-head", x=_n(gx), y="11")
                    if gender in SCORED_GENDERS else f'<text class="nc-head" x="{_n(gx)}" y="11">p({esc(gender)})</text>')
    parts = [gender_title, ui("names.chart.origins", "text", class_="nc-head", x=_n(ox), y="11")]
    lines = [(value, x0, width) for value, x0, width in ((threshold, gx, GENDER_W), (minimum, ox, ow))
             if value is not None]
    parts += [f'<text class="nc-tick" x="{_n(x0 + value * width)}" y="26">{value:g}</text>' for value, x0, width in lines]

    y = float(HEAD)
    for name, name_rows in names.items():
        parts.append(f'<text class="nc-name" x="0" y="{_n(y + 14)}">{esc(fit(name, NAME_PX, W, bold=True))}</text>')
        top = y + NAME_H
        for k, row in enumerate(name_rows):
            parts.append(_row(row, top + k * ROW_H, gx=gx, ox=ox, ow=ow, order=order, classes=classes,
                              origins=origins, label=labels[row.get("model", "")], threshold=threshold,
                              minimum=minimum, ui=ui))
        y = top + len(name_rows) * ROW_H + GAP
    height = _n(y - GAP + 2)
    return (f'<svg class="names-chart" role="img" viewBox="0 0 {W} {height}" width="{W}" height="{height}" '
            f'aria-labelledby="{esc(uid)}-cap {esc(uid)}-t" aria-describedby="{esc(uid)}-d">'
            + ui("names.chart.title", "title", id=f"{uid}-t") + ui("names.chart.desc", "desc", id=f"{uid}-d")
            + "".join(parts) + "</svg>")


def _threshold(x: float, y: float) -> str:
    """A threshold of the pass rule across one row: a hairline over the row's whole height, so it shows above
    and below the bar whatever the fill, and joins the line of the next row. It is drawn only on a row that has
    that bar: never across a name, nor across a question with no estimate."""
    return f'<line class="nc-thr" x1="{_n(x)}" x2="{_n(x)}" y1="{_n(y)}" y2="{_n(y + ROW_H)}"/>'


def _row(row: dict[str, str], y: float, *, gx: float, ox: float, ow: float, order: Sequence[str],
         classes: dict[str, str], origins: Sequence[str], label: str, threshold: float | None,
         minimum: float | None, ui: UiFn) -> str:
    bar_y, text_y = _n(y + (ROW_H - BAR_H) / 2), _n(y + 12)
    out = [f'<g class="nc-row" data-name="{esc(row.get("name", ""))}" data-model="{esc(row.get("model", ""))}">',
           f"<title>{esc(row_reading(row, origins))}</title>",
           f'<rect class="nc-hit" x="0" y="{_n(y)}" width="{W}" height="{ROW_H}"/>',
           f'<text class="nc-model" x="0" y="{text_y}">{esc(label)}</text>']
    p = number(row.get("p_intended_gender"))
    if p is not None:
        out.append(f'<rect class="nc-track" x="{_n(gx)}" y="{bar_y}" width="{GENDER_W}" height="{BAR_H}" rx="2"/>'
                   f'<rect class="nc-gender" x="{_n(gx)}" y="{bar_y}" width="{_n(GENDER_W * min(1.0, max(0.0, p)))}" '
                   f'height="{BAR_H}" rx="2"/>')
        if threshold is not None:
            out.append(_threshold(gx + threshold * GENDER_W, y))
    else:  # no scored gender: every answer missing or all hedge. An empty bar would read as a probability of 0.
        out.append(ui("names.chart.none.short", "text", class_="nc-none", x=_n(gx), y=text_y))
    shares = [number(row.get(SCORED + o)) for o in order]
    if shares and all(s is not None for s in shares):
        x = ox
        for k, (o, share) in enumerate(zip(order, shares)):
            width = min(ow * max(0.0, share or 0.0), ox + ow - x)  # never past the end of the bar
            # The first segment (the intended origin) is drawn to its exact end, where the minimum is read. Each
            # later one gives up SEG_GAP at its start for the gap in the page colour; a sliver keeps 0.75 of itself.
            drawn = width if k == 0 else max(width - SEG_GAP, min(width, 0.75))
            if drawn > 0:
                out.append(f'<rect class="nc-o {classes[o]}" x="{_n(x + width - drawn)}" y="{bar_y}" '
                           f'width="{_n(drawn)}" height="{BAR_H}"/>')
            x += width
        if minimum is not None:
            out.append(_threshold(ox + minimum * ow, y))
    else:  # no scored origin distribution: every answer missing or all hedge
        out.append(ui("names.chart.none", "text", class_="nc-none", x=_n(ox), y=text_y))
    out.append(mark(is_true(row.get("pass")), W - 13, y + 2))
    return "".join(out) + "</g>"


def _swatch(inner: str, width: int = 18) -> str:
    return f'<svg class="names-key" width="{width}" height="12" viewBox="0 0 {width} 12" aria-hidden="true">{inner}</svg>'


def origin_key(origins: Sequence[str], ui: UiFn, *, short: bool = False) -> list[str]:
    """One legend item per origin: its colour and its name (`short`: the name used in headings and columns)."""
    classes = origin_classes(origins)
    return [_swatch(f'<rect class="nc-o {classes[o]}" x="0" y="1" width="18" height="10"/>')
            + origin_label(o, ui, short=short) for o in origins]


def legend(origins: Sequence[str], ui: UiFn) -> str:
    """The key of every cell chart: the origin colours (with the options as the models were asked them), the
    gender bar, the threshold line and the two marks."""
    items = origin_key(origins, ui)
    items.append(_swatch('<rect class="nc-track" x="0" y="1" width="18" height="10" rx="2"/>'
                         '<rect class="nc-gender" x="0" y="1" width="13" height="10" rx="2"/>') + ui("names.legend.gender"))
    items.append(_swatch('<rect class="nc-track" x="0" y="1" width="18" height="10" rx="2"/>'
                         '<line class="nc-thr" x1="12" x2="12" y1="0" y2="12"/>') + ui("names.legend.threshold"))
    items.append(_swatch(mark(True, 0, 0), 12) + ui("names.pass"))
    items.append(_swatch(mark(False, 0, 0), 12) + ui("names.fail"))
    return '<ul class="names-legend">' + "".join(f"<li>{item}</li>" for item in items) + "</ul>"


def short_legend(origins: Sequence[str], ui: UiFn) -> str:
    """The origin colours again, in one line, for the charts of one origin: the full key is above the first."""
    return ('<ul class="names-legend names-legend-short">'
            + "".join(f"<li>{item}</li>" for item in origin_key(origins, ui, short=True)) + "</ul>")


# --- re-asks -----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Chain:
    """One answer (name x model x repeat) whose all-hedge attempts were asked again (reasked.csv, ADR 0011)."""

    name: str
    gender: str
    origin: str
    model: str
    repeat: str
    replaced: tuple[tuple[str, ...], ...]  # per all-hedge attempt, in order: the questions it gave no estimate for
    ended: str  # "answered", "hedge" (still all hedge), "missing" (the re-ask failed), or "" (perceptions.csv does not say)
    final: dict[str, str] | None  # the answer on record (its perceptions.csv row)

    @property
    def reasks(self) -> int:
        return len(self.replaced)


def _blank(row: dict[str, str], columns: Sequence[str]) -> bool:
    return all(not row.get(c, "").strip() for c in columns)


def _all_hedge(row: dict[str, str], origins: Sequence[str]) -> tuple[str, ...]:
    """The questions a valid answer gave no estimate for: its scored columns are blank."""
    questions = (("gender", [SCORED + g for g in SCORED_GENDERS]), ("origin", [SCORED + o for o in origins]))
    return tuple(q for q, columns in questions if columns and _blank(row, columns))


def _answer(row: dict[str, str]) -> tuple[str, str, str]:
    """What one answer is the answer of: its name, model and repeat."""
    return row.get("name", ""), row.get("model", ""), row.get("repeat", "")


def reask_chains(record: NamesRecord) -> list[Chain]:
    """The answers that were asked again, in the order of reasked.csv, each with how it ended according to the
    answer on record in perceptions.csv: answered (an estimate for both questions), still all hedge, or missing
    (the re-ask failed or was not valid)."""
    if REASKED_FILE not in record.tables:
        return []
    origins = scored_origins(record.tables[REASKED_FILE].header)
    on_record = {_answer(r): r for r in record.rows(PERCEPTIONS)}
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in record.rows(REASKED_FILE):
        groups.setdefault(_answer(row), []).append(row)
    chains = []
    for (name, model, repeat), rows in groups.items():
        rows = sorted(rows, key=lambda r: count(r, "attempt"))
        final = on_record.get((name, model, repeat))
        if final is None:
            ended = ""
        elif _blank(final, ["gender_hedge", "origin_hedge"]):
            ended = "missing"
        else:
            ended = "hedge" if _all_hedge(final, origins) else "answered"
        chains.append(Chain(name, rows[0].get("intended_gender", ""), rows[0].get("intended_origin", ""), model,
                            repeat, tuple(_all_hedge(r, origins) for r in rows), ended, final))
    return chains
