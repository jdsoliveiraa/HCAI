"""The name check on the project site (`pilot/names.html`, ADRs 0010 and 0011): the record, the charts, the tables.

The fixture record is written by `hcai.namecheck` itself (`score_probabilities` + `write_outputs`) from a few
hand-made answers, so the site is tested against the columns and the pass flags the check really writes.
"""

from __future__ import annotations

import csv
import html
import re
import shutil
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

import pytest

from conftest import ROOT
from hcai.namecheck import GENDER_OPTIONS, ORIGIN_KEYS, Perception, score_probabilities, write_outputs
from hcai.site import NAMES_PAGE, UI, Site, build_site, load_site_config
from hcai.site_names import (
    GENDER_W, LABEL_MAX, MARK_W, MISSING, MODEL_PX, NAME_PX, W, cell_chart, load_record, model_labels, reask_chains,
    shown, text_width,
)
from test_site import broken_links, parse, write_project

MODELS = ("model-a", "model-b")
HOSTILE = 'Zé <b>"Faria" & Cª'  # a name that must be shown as text everywhere, never as markup


def answer(name: str, intended: tuple[str, str], model: str, repeat: int, gender: tuple[float, ...] | None,
           origin: tuple[float, ...] | None, *, attempt: int = 0, reasked: tuple[Perception, ...] = ()) -> Perception:
    """One probability-mode answer: `gender` over (male, female, unsure), `origin` over the four origins and
    "Other or unsure"; None for a missing answer."""
    return Perception(name, intended[0], intended[1], model, repeat, "", "", mode="probability",
                      gender_p=dict(zip(GENDER_OPTIONS, gender)) if gender else None,
                      origin_p=dict(zip(ORIGIN_KEYS, origin)) if origin else None,
                      gender_sum=1.0 if gender else None, origin_sum=1.0 if origin else None,
                      prompt_version="prob-v2", attempt=attempt, reasked=reasked)


MALE, FEMALE, UNSURE = (0.96, 0.02, 0.02), (0.02, 0.96, 0.02), (0.0, 0.0, 1.0)
NO_ORIGIN = (0.0, 0.0, 0.0, 0.0, 1.0)


def fixture_answers() -> list[Perception]:
    """Three names, two models, two repeats. Gonçalo Figueiredo (male, Portuguese) passes; one of his model-a
    answers was all hedge on both questions and answered when asked again. The hostile name (female, Brazilian)
    passes for model-a and fails origin for model-b, which reads it as PALOP. Ana Lima (female, Brazilian) fails
    for model-b: an answer all hedge on origin stayed so after two re-asks, and the other answer is missing."""
    pt_m, br_f = ("male", "portuguese"), ("female", "brazilian")
    out = []
    for model in MODELS:
        first = answer("Gonçalo Figueiredo", pt_m, model, 0, MALE, (0.8, 0.1, 0.05, 0.01, 0.04))
        if model == "model-a":
            first = answer("Gonçalo Figueiredo", pt_m, model, 0, MALE, (0.9, 0.05, 0.02, 0.01, 0.02), attempt=1,
                           reasked=(answer("Gonçalo Figueiredo", pt_m, model, 0, UNSURE, NO_ORIGIN),))
        out += [first, answer("Gonçalo Figueiredo", pt_m, model, 1, MALE, (0.7, 0.2, 0.05, 0.01, 0.04))]
    for repeat in (0, 1):
        out.append(answer(HOSTILE, br_f, "model-a", repeat, FEMALE, (0.5, 0.4, 0.04, 0.01, 0.05)))
        out.append(answer(HOSTILE, br_f, "model-b", repeat, FEMALE, (0.3, 0.2, 0.45, 0.01, 0.04)))
        out.append(answer("Ana Lima", br_f, "model-a", repeat, FEMALE, (0.55, 0.4, 0.02, 0.01, 0.02)))
    stuck = [answer("Ana Lima", br_f, "model-b", 0, FEMALE, NO_ORIGIN, attempt=k) for k in range(3)]
    out.append(answer("Ana Lima", br_f, "model-b", 0, FEMALE, NO_ORIGIN, attempt=2, reasked=tuple(stuck[:2])))
    out.append(answer("Ana Lima", br_f, "model-b", 1, None, None))
    return out


def write_record(directory: Path) -> Path:
    write_outputs(score_probabilities(fixture_answers(), max_reasks=3), directory)
    return directory


def rows_of(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A site project whose committed name check record (docs/pilot/names) is the fixture, with no results/."""
    config = write_project(tmp_path / "project", results=False)
    write_record(tmp_path / "project" / "docs" / "pilot" / "names")
    return config


class _Page(HTMLParser):
    """What the tests read off the names page: the chart rows (name, model, <title>, mark) and every text node."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[dict[str, str]] = []
        self.svgs: list[dict[str, str | None]] = []
        self.text: list[str] = []
        self._row: dict[str, str] | None = None
        self._title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        classes = (a.get("class") or "").split()
        if tag == "svg" and "names-chart" in classes:
            self.svgs.append(a)
        if tag == "g" and "nc-row" in classes:
            self._row = {"name": a.get("data-name") or "", "model": a.get("data-model") or "", "title": "", "mark": ""}
            self.rows.append(self._row)
        elif self._row is not None and tag == "title":
            self._title = True
        elif self._row is not None and tag == "path" and "nc-mark" in classes:
            self._row["mark"] += "pass" if "nc-pass" in classes else "fail" if "nc-fail" in classes else "?"

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._title = False
        if tag == "g":
            self._row = None

    def handle_data(self, data: str) -> None:
        self.text.append(data)
        if self._row is not None and self._title:
            self._row["title"] += data


def read_page(path: Path) -> _Page:
    page = _Page()
    page.feed(path.read_text(encoding="utf-8"))
    return page


def first_table(fragment: str) -> list[list[str]]:
    """The body rows of the first table in `fragment`, each cell as the text a reader sees."""
    body = fragment.split("<tbody>", 1)[1].split("</tbody>", 1)[0]

    def cell(inner: str) -> str:
        return html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", inner))).strip()

    return [[cell(c) for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S)]
            for row in re.findall(r"<tr>(.*?)</tr>", body, re.S)]


def table_rows(html_text: str, heading_id: str) -> list[list[str]]:
    """The body rows of the first table after the heading `heading_id`."""
    return first_table(html_text.split(f'id="{heading_id}"', 1)[1])


# --- the record --------------------------------------------------------------------------------------------------


def test_the_fixture_record_is_what_the_check_writes(tmp_path: Path):
    record = load_record(write_record(tmp_path / "names"))
    assert record is not None and record.probability and not record.problems
    assert record.origins == ["portuguese", "brazilian", "palop", "eastern_european"]
    assert record.names == ["Gonçalo Figueiredo", HOSTILE, "Ana Lima"] and record.failing == [HOSTILE, "Ana Lima"]
    assert load_record(tmp_path / "nothing") is None


def test_numbers_are_rounded_from_the_csv_text_half_up():
    assert [shown(x) for x in ("0.996176", "0.0125", "0.7995", "1", "0.000400")] == ["0.996", "0.013", "0.800", "1.000", "0.000"]
    assert [shown(x) for x in ("", None, "n/a", "nan", "inf", "1e999999")] == [MISSING] * 6


def test_reask_chains_tell_how_each_answer_ended(tmp_path: Path):
    record = load_record(write_record(tmp_path / "names"))
    assert record is not None
    chains = {(c.name, c.model, c.repeat): c for c in reask_chains(record)}
    assert set(chains) == {("Gonçalo Figueiredo", "model-a", "0"), ("Ana Lima", "model-b", "0")}
    answered, stuck = chains[("Gonçalo Figueiredo", "model-a", "0")], chains[("Ana Lima", "model-b", "0")]
    assert (answered.reasks, answered.replaced, answered.ended) == (1, (("gender", "origin"),), "answered")
    assert answered.final is not None and answered.final["attempt"] == "1"
    assert (stuck.reasks, stuck.replaced, stuck.ended) == (2, (("origin",), ("origin",)), "hedge")
    # without perceptions.csv there is no answer on record to tell how a chain ended
    (record.path / "perceptions.csv").unlink()
    again = load_record(record.path)
    assert again is not None and {c.ended for c in reask_chains(again)} == {""}
    # a re-ask that failed leaves a missing answer on record
    failed = answer("Rui Reis", ("male", "portuguese"), "model-a", 0, None, None, attempt=1, reasked=(
        answer("Rui Reis", ("male", "portuguese"), "model-a", 0, UNSURE, NO_ORIGIN),))
    write_outputs(score_probabilities([failed], max_reasks=3), tmp_path / "failed")
    lost = load_record(tmp_path / "failed")
    assert lost is not None and [c.ended for c in reask_chains(lost)] == ["missing"]


# --- where the record is read from -----------------------------------------------------------------------------------


def test_the_site_reads_the_committed_record_and_shows_it(project: Path, tmp_path: Path):
    out = tmp_path / "site"
    pages = build_site(project, out)
    assert NAMES_PAGE in pages and broken_links(out) == []
    text = (out / NAMES_PAGE).read_text()
    assert text.count("<h1") == 1 and '<a href="../pilot.html" aria-current="page">' in text
    assert "docs/pilot/names/" in text and "committed with the repository" in text
    for filename in ("cells.csv", "summary.csv", "perceptions.csv", "reasked.csv"):  # the repo of [site] repo
        assert f'href="https://github.com/jdsoliveiraa/HCAI/blob/main/docs/pilot/names/{filename}"' in text
    # the pass rule in plain words, with the parameters the record was scored with
    assert "at least</span> <strong>0.8</strong>" in text and "at least</span> <strong>0.25</strong>" in text
    pilot = (out / "pilot.html").read_text()
    assert 'href="pilot/names.html"' in pilot and "<strong>1 / 3</strong>" in pilot
    home = (out / "index.html").read_text()
    assert "3 names, 2 failing" in home


def test_the_site_falls_back_to_the_local_results(tmp_path: Path):
    config = write_project(tmp_path / "project", results=False)
    write_record(tmp_path / "project" / "results" / "names")
    out = tmp_path / "site"
    build_site(config, out)
    text = (out / NAMES_PAGE).read_text()
    assert "results/names/" in text and "local results, not committed" in text
    assert "github.com/jdsoliveiraa/HCAI/blob/main/docs/pilot/names" not in text  # nothing there to link to
    assert "Gonçalo Figueiredo" in text and len(read_page(out / NAMES_PAGE).svgs) == 2
    assert "3 names, 2 failing" in (out / "index.html").read_text()
    # the committed record wins as soon as it exists
    committed = tmp_path / "project" / "docs" / "pilot" / "names"
    committed.mkdir(parents=True)
    passing = [a for a in fixture_answers() if a.name == "Gonçalo Figueiredo"]
    write_outputs(score_probabilities(passing, max_reasks=3), committed)
    build_site(config, out)
    text = (out / NAMES_PAGE).read_text()
    assert "committed with the repository" in text and "Ana Lima" not in text
    assert "1 names, 0 failing" in (out / "index.html").read_text()


def test_the_record_path_is_a_site_config_key(tmp_path: Path):
    config = write_project(tmp_path / "project", results=False)
    assert load_site_config(config).names == "docs/pilot/names"
    write_record(tmp_path / "project" / "records" / "name check")
    config.write_text(config.read_text() + '\n[site]\nnames = "records/name check"\nrepo = "someone/fork"\n')
    out = tmp_path / "site"
    build_site(config, out)
    text = (out / NAMES_PAGE).read_text()
    assert 'href="https://github.com/someone/fork/blob/main/records/name%20check/summary.csv"' in text


def test_not_generated_when_there_is_no_record_anywhere(tmp_path: Path):
    config = write_project(tmp_path / "project", results=False)
    out = tmp_path / "site"
    pages = build_site(config, out)
    assert NAMES_PAGE in pages and broken_links(out) == []
    text = (out / NAMES_PAGE).read_text()
    assert "Not generated yet." in text and "<svg" not in text and "docs/pilot/names/" in text
    pilot = (out / "pilot.html").read_text()
    names_section = pilot.split('<section id="names"', 1)[1].split("</section>", 1)[0]
    assert "Not generated yet." in names_section and "pilot/names.html" not in names_section
    assert "Not generated yet." in (out / "index.html").read_text()


def test_a_record_without_scored_probabilities_shows_its_tables_only(tmp_path: Path):
    config = write_project(tmp_path / "project")  # results/names: a bare choice-style summary.csv and cells.csv
    out = tmp_path / "site"
    build_site(config, out)
    text = (out / NAMES_PAGE).read_text()
    assert "no scored probabilities" in text and "<svg" not in text
    assert "Tiago Martins" in text and "Rita Sousa" in text  # every row here, the failing ones on the pilot page
    assert broken_links(out) == []


def test_a_choice_mode_record_lists_every_row_on_its_page(tmp_path: Path):
    config = write_project(tmp_path / "project", results=False)
    names = tmp_path / "project" / "docs" / "pilot" / "names"
    names.mkdir(parents=True)
    rows = [f"Name {k},model-a,{k < 10}" for k in range(64)]  # 54 failing rows
    (names / "summary.csv").write_text("name,model,pass\n" + "\n".join(rows) + "\n")
    out = tmp_path / "site"
    build_site(config, out)
    text = (out / NAMES_PAGE).read_text()
    assert len(first_table(text)) == 64 and "Showing the first rows only" not in text
    assert "probability to" not in text  # the introduction fits a record of single choices too
    section = (out / "pilot.html").read_text().split('<section id="names"', 1)[1].split("</section>", 1)[0]
    assert len(first_table(section)) == 40 and "Showing the first rows only" in section and "(54 rows)" in section
    assert "every name and model</span>" in section and "charts" not in section  # the link promises what is there


def test_a_spreadsheet_resave_is_read_or_refused_never_misread(tmp_path: Path):
    config = write_project(tmp_path / "project", results=False)
    names = write_record(tmp_path / "project" / "docs" / "pilot" / "names")
    out = tmp_path / "site"
    for f in names.glob("*.csv"):  # a byte order mark is dropped
        f.write_bytes(b"\xef\xbb\xbf" + f.read_bytes())
    record = load_record(names)
    assert record is not None and record.probability and record.names[0] == "Gonçalo Figueiredo"
    build_site(config, out)
    assert "3 names, 2 failing" in (out / "index.html").read_text() and len(read_page(out / NAMES_PAGE).svgs) == 2
    # a semicolon export is refused with its reason: read as one column it would be a record of one nameless name
    summary = names / "summary.csv"
    summary.write_text(summary.read_text(encoding="utf-8-sig").replace(",", ";"), encoding="utf-8")
    record = load_record(names)
    assert record is not None and "summary.csv" not in record.tables and "not comma-separated" in record.problems[0]
    build_site(config, out)
    text = (out / NAMES_PAGE).read_text()
    assert '<p class="error">summary.csv: not comma-separated' in text and "<svg" not in text
    assert "names, " not in (out / "index.html").read_text()  # no count from a file that was not read
    assert '<p class="error">summary.csv: not comma-separated' in (out / "pilot.html").read_text()


def test_an_empty_committed_summary_does_not_hide_the_local_results(tmp_path: Path):
    config = write_project(tmp_path / "project", results=False)
    root = tmp_path / "project"
    write_record(root / "results" / "names")
    committed = root / "docs" / "pilot" / "names"
    committed.mkdir(parents=True)
    (committed / "summary.csv").write_text("")
    assert load_record(committed) is None
    header = (root / "results" / "names" / "summary.csv").read_text().splitlines()[0]
    (committed / "summary.csv").write_text(header + "\n")
    assert load_record(committed) is None  # a header with no row is no record either
    out = tmp_path / "site"
    build_site(config, out)
    assert "local results, not committed" in (out / NAMES_PAGE).read_text()
    assert "3 names, 2 failing" in (out / "index.html").read_text()


def test_a_record_outside_the_repository_is_not_called_committed(tmp_path: Path):
    config = write_project(tmp_path / "project", results=False)
    write_record(tmp_path / "elsewhere" / "names")
    config.write_text(config.read_text() + '\n[site]\nnames = "../elsewhere/names"\n')
    out = tmp_path / "site"
    build_site(config, out)
    text = (out / NAMES_PAGE).read_text()
    assert "Gonçalo Figueiredo" in text and "local results, not committed" in text
    assert "blob/main" not in text and "committed with the repository" not in text


# --- the page ----------------------------------------------------------------------------------------------------


def test_marks_and_tables_match_summary_csv(project: Path, tmp_path: Path):
    out = tmp_path / "site"
    build_site(project, out)
    record = project.parents[1] / "docs" / "pilot" / "names"
    summary = rows_of(record / "summary.csv")
    expected = {(r["name"], r["model"]): "pass" if r["pass"] == "True" else "fail" for r in summary}
    assert sorted(expected.values()) == ["fail", "fail", "pass", "pass", "pass", "pass"]  # the fixture has both
    page = read_page(out / NAMES_PAGE)
    assert {(r["name"], r["model"]): r["mark"] for r in page.rows} == expected  # one mark per row, the CSV's
    for row in page.rows:  # the row's <title> reads the CSV's own numbers
        source = next(r for r in summary if (r["name"], r["model"]) == (row["name"], row["model"]))
        assert row["title"].startswith(f'{row["name"]} · {row["model"]}: p({source["intended_gender"]}) '
                                       f'{shown(source["p_intended_gender"])}; ')
        assert row["title"].endswith("; pass") == (source["pass"] == "True")
        for origin in ("portuguese", "brazilian", "palop", "eastern_european"):
            assert f'{origin} {shown(source[f"scored_p_{origin}"])}' in row["title"]
    by_row = {(r["name"], r["model"]): r["title"] for r in page.rows}
    assert by_row[(HOSTILE, "model-b")].endswith("; fail (origin)")
    assert by_row[("Ana Lima", "model-b")].endswith("; fail (origin, missing answers, all-hedge answers)")
    # the summary table: one row per CSV row, in its order, with its numbers and its three flags
    text = (out / NAMES_PAGE).read_text()
    table = table_rows(text, "h-summary")
    assert len(table) == len(summary)
    word = {"True": "pass", "False": "fail"}
    for cells, source in zip(table, summary):
        assert cells[0] == source["name"] and cells[2] == source["model"]
        assert cells[3] == word[source["pass"]] and cells[5] == word[source["gender_pass"]]
        assert cells[4] == shown(source["p_intended_gender"])
        assert cells[6:10] == [shown(source[f"scored_p_{o}"]) for o in ("portuguese", "brazilian", "palop", "eastern_european")]
        assert cells[10] == word[source["origin_pass"]]
        assert cells[11:13] == [shown(source["gender_hedge"]), shown(source["origin_hedge"])]
        assert cells[13:] == [source["answers"], source["missing"],
                              f'{source["gender_all_hedge"]} / {source["origin_all_hedge"]}', source["reasks"]]
    assert table[-1][4] == "0.980" and table[-1][6:10] == [MISSING] * 4  # Ana Lima, model-b: no origin estimate
    # the cells table: cells.csv as it is
    cells_csv = rows_of(record / "cells.csv")
    cells_table = table_rows(text, "h-cells")
    assert len(cells_table) == len(cells_csv) == 6
    for cells, source in zip(cells_table, cells_csv):
        assert cells[2] == ("all models" if source["model"] == "all" else source["model"])
        assert cells[3:7] == [shown(source[k]) for k in ("mean_p_intended_gender", "mean_p_intended_origin",
                                                         "mean_gender_hedge", "mean_origin_hedge")]
        assert cells[-1] == f'{source["names_passing"]} / {source["names"]}'
    # the captions count the names of each cell that pass for every model
    assert "(1 / 1 <span" in text and "(0 / 2 <span" in text


def test_the_reask_record_is_shown_per_model_and_per_answer(project: Path, tmp_path: Path):
    out = tmp_path / "site"
    build_site(project, out)
    text = (out / NAMES_PAGE).read_text()
    section = text.split('id="h-reasks"', 1)[1].split('id="h-files"', 1)[0]
    per_model, each = (first_table(part) for part in section.split("<h3", 2)[1:])
    # model, answers, re-asks, answers asked again, ended answered / still all hedge / missing, names
    # the names come with their answers asked again: Ana Lima has one, asked twice (2 re-asks)
    assert per_model == [["model-a", "6", "1", "1", "1", "0", "0", "Gonçalo Figueiredo (1)"],
                         ["model-b", "6", "2", "1", "0", "1", "0", "Ana Lima (1)"]]
    assert each == [
        ["Gonçalo Figueiredo", "Male · Portuguese", "model-a", "0", "gender and origin", "1", "answered", "1",
         "0.980", "0.918"],
        ["Ana Lima", "Female · Brazilian", "model-b", "0", "origin → origin", "2", "still all hedge", "2", "0.980", MISSING],
    ]
    assert "summary.csv counts" not in section  # reasked.csv and summary.csv agree


def test_charts_are_accessible_static_svg(project: Path, tmp_path: Path):
    out = tmp_path / "site"
    build_site(project, out)
    text = (out / NAMES_PAGE).read_text()
    page, parsed = read_page(out / NAMES_PAGE), parse(out / NAMES_PAGE)
    assert len(page.svgs) == 2  # one per gender x origin cell of the record
    for svg in page.svgs:
        assert svg["role"] == "img" and str(svg["viewbox"]).startswith("0 0 320 ")
        named, described = str(svg["aria-labelledby"]).split(), str(svg["aria-describedby"])
        assert set(named) | {described} <= parsed.ids  # the caption, the <title> and the <desc> exist
    assert text.count("<title data-i18n=\"names.chart.title\"") == 2 and text.count("<desc data-i18n=\"names.chart.desc\"") == 2
    assert text.count("<figcaption") == 2 and 'class="names-legend"' in text
    # the origin colours are named again beside the charts of each origin, not only above the first chart
    assert text.count('class="names-legend names-legend-short"') == 2
    male_pt, female_br = (svg for svg in re.findall(r'<svg class="names-chart".*?</svg>', text, re.S))
    # text labels: every name and model is written in the chart itself, not only encoded in a mark
    for chart, labels in ((male_pt, ("Gonçalo Figueiredo", "model-a", "model-b", "p(male)", "origin shares")),
                          (female_br, ("Ana Lima", "model-a", "model-b", "p(female)", "origin shares"))):
        written = [t.text for t in ElementTree.fromstring(chart).iter("text")]
        assert set(labels) <= set(written), written
    # the thresholds: 0.8 on every chart; the origin minimum only where the rule has one (not for Portuguese)
    assert male_pt.count('class="nc-tick"') == 1 and ">0.8</text>" in male_pt and ">0.25</text>" not in male_pt
    assert female_br.count('class="nc-tick"') == 2 and ">0.8</text>" in female_br and ">0.25</text>" in female_br
    # the intended origin is the first segment of every bar: Portuguese (1st column) there, Brazilian (2nd) here
    assert re.findall(r'<rect class="nc-o (nc-o\d)"', male_pt.split("</g>")[0]) == ["nc-o1", "nc-o2", "nc-o3", "nc-o4"]
    assert re.findall(r'<rect class="nc-o (nc-o\d)"', female_br.split("</g>")[0]) == ["nc-o2", "nc-o1", "nc-o3", "nc-o4"]
    assert ">no estimate</text>" in female_br and ">no estimate</text>" not in male_pt  # Ana Lima, model-b
    # a threshold line lies over a bar only: 2 gender bars there; here 4 gender bars and the 3 origin bars drawn
    assert male_pt.count('class="nc-thr"') == 2 and female_br.count('class="nc-thr"') == 4 + 3
    # the tables that hold the charts' values are named regions, and their rows are named by their first cell
    for label in ("cap-cells", "cap-summary", "cap-per-model", "cap-each"):
        assert f'role="region" aria-labelledby="{label}"' in text and f'<caption id="{label}">' in text
    assert text.count('<th scope="row">') == 6 + 2 + 2  # summary rows, models, answers asked again
    # static: no script but the site's own, nothing loaded from elsewhere, and a table of the same values follows
    assert re.findall(r'<script[^>]*src="([^"]+)"', text) == ["../assets/site.js"]
    assert "<img" not in text and "<link rel=\"stylesheet\" href=\"../assets/style.css\">" in text
    assert text.index('id="h-chart"') < text.index("<svg class=\"names-chart\"") < text.index('id="h-summary"')
    # the styles the charts need exist, in both themes
    css = (out / "assets" / "style.css").read_text()
    light, dark = css.split("@media (prefers-color-scheme: dark)", 1)
    for token in ("--origin-1", "--origin-2", "--origin-3", "--origin-4"):
        assert token in light and token in dark.split("}", 1)[0], token  # the dark block's own :root
    for cls in ("nc-o1", "nc-o4", "nc-gender", "nc-track", "nc-thr", "nc-mark", "nc-fail", "names-grid"):
        assert f".{cls}" in css, cls


def test_names_are_escaped_everywhere(project: Path, tmp_path: Path):
    out = tmp_path / "site"
    build_site(project, out)
    text = (out / NAMES_PAGE).read_text()
    assert "<b>" not in text and '"Faria"' not in text  # never as markup, never closing an attribute
    escaped = "Zé &lt;b&gt;&quot;Faria&quot; &amp; Cª"
    assert f'data-name="{escaped}"' in text and f">{escaped}</text>" in text
    assert f'<th scope="row">{escaped}</th>' in text  # its rows in the summary table
    assert f"<title>{escaped} · model-a: " in text
    page = read_page(out / NAMES_PAGE)
    assert HOSTILE in page.text and {r["name"] for r in page.rows} == {"Gonçalo Figueiredo", HOSTILE, "Ana Lima"}
    assert text.rstrip().endswith("</html>")
    # each chart is well-formed on its own (a strict XML parse), and gives the names back as they are in the CSV
    charts = [ElementTree.fromstring(svg) for svg in re.findall(r'<svg class="names-chart".*?</svg>', text, re.S)]
    assert len(charts) == 2
    assert {g.get("data-name") for chart in charts for g in chart.iter("g")} == {"Gonçalo Figueiredo", HOSTILE, "Ana Lima"}
    assert {t.text for chart in charts for t in chart.iter("text") if t.get("class") == "nc-name"} == {
        "Gonçalo Figueiredo", HOSTILE, "Ana Lima"}
    # the same through the plain tables of a record without probabilities
    plain = tmp_path / "plain"
    config = write_project(plain, results=False)
    names = plain / "docs" / "pilot" / "names"
    names.mkdir(parents=True)
    (names / "summary.csv").write_text('name,model,pass\n"<i>x</i>",model-a,False\n')
    build_site(config, tmp_path / "plain-site")
    for page_path in (NAMES_PAGE, "pilot.html"):
        shown_text = (tmp_path / "plain-site" / page_path).read_text()
        assert "<i>" not in shown_text and "&lt;i&gt;x&lt;/i&gt;" in shown_text


def fake_ui(seen: list[str] | None = None):
    """A stand-in for hcai.site.ui that writes the key as the text (and notes it in `seen`)."""
    def ui(key: str, tag: str = "span", **attrs: str) -> str:
        if seen is not None:
            seen.append(key)
        return f"<{tag}>{key}</{tag}>"
    return ui


def test_bars_and_threshold_lines_are_drawn_to_the_csv_values(project: Path, tmp_path: Path):
    """The geometry of every chart row against summary.csv: the gender bar is p long, each origin segment ends
    where its cumulative share ends (the intended one drawn from 0 to exactly its share), and the lines stand at
    0.8 and 0.25 of their bars."""
    out = tmp_path / "site"
    build_site(project, out)
    summary = {(r["name"], r["model"]): r for r in rows_of(project.parents[1] / "docs" / "pilot" / "names" / "summary.csv")}
    text = (out / NAMES_PAGE).read_text()
    charts = [ElementTree.fromstring(svg) for svg in re.findall(r'<svg class="names-chart".*?</svg>', text, re.S)]
    checked = []
    for chart in charts:
        ticks = {t.text: float(t.get("x", "nan")) for t in chart.iter("text") if t.get("class") == "nc-tick"}
        starts = set()
        for g in chart.iter("g"):
            source = summary[(g.get("data-name", ""), g.get("data-model", ""))]
            rects: dict[str, list[tuple[float, float]]] = {}
            for rect in g.iter("rect"):
                kind = (rect.get("class") or "").split()[0]
                rects.setdefault(kind, []).append((float(rect.get("x", "nan")), float(rect.get("width", "nan"))))
            lines = [float(line.get("x1", "nan")) for line in g.iter("line")]
            words = [t.text for t in g.iter("text")]
            (gx, track), = rects["nc-track"]
            assert track == GENDER_W and rects["nc-gender"] == [(gx, pytest.approx(GENDER_W * float(source["p_intended_gender"]), abs=0.01))]
            assert lines[0] == pytest.approx(gx + 0.8 * GENDER_W, abs=0.01) and ticks["0.8"] == pytest.approx(lines[0])
            intended = source["intended_origin"]
            order = [intended] + [o for o in ("portuguese", "brazilian", "palop", "eastern_european") if o != intended]
            if not source["scored_p_portuguese"]:  # no origin estimate: words, no bar, no line
                assert "nc-o" not in rects and "no estimate" in words and len(lines) == 1
                checked.append("none")
                continue
            shares = [float(source[f"scored_p_{o}"]) for o in order]
            segments = rects["nc-o"]
            assert len(segments) == 4 and "no estimate" not in words
            ox = segments[0][0]
            starts.add(ox)
            ow = W - MARK_W - ox
            assert segments[0][1] == pytest.approx(ow * shares[0], abs=0.01)  # the intended share, exactly
            end = ox
            for (x, width), share in zip(segments, shares):
                end += ow * share
                assert x + width == pytest.approx(end, abs=0.02) and width <= ow * share + 0.01
            assert end == pytest.approx(ox + ow, abs=0.02)  # the shares fill the bar
            if intended == "portuguese":
                assert len(lines) == 1 and "0.25" not in ticks
            else:
                assert lines[1:] == [pytest.approx(ox + 0.25 * ow, abs=0.01)] and ticks["0.25"] == pytest.approx(lines[1])
            checked.append(intended)
        assert len(starts) == 1  # every bar of a chart starts at the same x
    assert sorted(checked) == ["brazilian"] * 3 + ["none"] + ["portuguese"] * 2
    # the failing origin row is seen to fail: its intended share ends left of the minimum line, the passing ones right
    hostile = {g.get("data-model"): g for chart in charts for g in chart.iter("g") if g.get("data-name") == HOSTILE}
    for model, passes in (("model-a", True), ("model-b", False)):
        first = next(r for r in hostile[model].iter("rect") if (r.get("class") or "").startswith("nc-o "))
        line = [float(line.get("x1", "nan")) for line in hostile[model].iter("line")][1]
        assert (float(first.get("x", "nan")) + float(first.get("width", "nan")) > line) is passes


def chart_of(rows: list[dict[str, str]], origin: str = "palop", seen: list[str] | None = None) -> str:
    names: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        names.setdefault(row["name"], []).append(row)
    return cell_chart(names, gender="male", origin=origin, origins=["portuguese", "palop"], uid="c", ui=fake_ui(seen))


def test_long_labels_are_cut_in_the_chart_and_whole_in_its_title():
    row = {"name": "N" * 80, "model": "a-very-long-model-name-indeed", "intended_gender": "male",
           "intended_origin": "palop", "p_intended_gender": "1.2", "scored_p_portuguese": "0.5", "scored_p_palop": "0.7",
           "gender_threshold": "0.8", "origin_minimum": "0.25", "pass": "False"}
    seen: list[str] = []
    svg = chart_of([row], seen=seen)
    chart = ElementTree.fromstring(svg)
    title = next(chart.iter("g")).find("title")
    assert title is not None and title.text is not None and title.text.startswith("N" * 80 + " · a-very-long-model-name-indeed: ")
    name, = [t.text or "" for t in chart.iter("text") if t.get("class") == "nc-name"]
    assert name.endswith("…") and ("N" * 80).startswith(name[:-1]) and 20 < len(name) < 45
    assert text_width(name, NAME_PX, bold=True) <= W  # by the chart's own (wide) estimate
    label, = [t.text or "" for t in chart.iter("text") if t.get("class") == "nc-model"]
    assert label.startswith("a-very") and label.endswith("indeed") and "…" in label  # cut in the middle
    assert text_width(label, MODEL_PX) <= LABEL_MAX
    widths = [float(w) for w in re.findall(r'<rect class="nc-(?:gender|o nc-o\d)"[^>]* width="([\d.]+)"', svg)]
    assert widths[0] == GENDER_W  # a probability above 1 is drawn as 1
    assert sum(widths[1:]) <= W - (LABEL_MAX + 6 + GENDER_W + 12) - MARK_W  # shares over 1 never run past the bar's end
    assert {"names.chart.title", "names.chart.desc", "names.chart.p.male", "names.chart.origins"} <= set(seen)


def test_two_models_never_get_the_same_label():
    base = {"intended_gender": "male", "intended_origin": "palop", "p_intended_gender": "0.9",
            "scored_p_portuguese": "0.5", "scored_p_palop": "0.5", "pass": "True"}
    # ids of one vendor differ at their end: cut in the middle, they stay apart
    models = ["anthropic/claude-3.5-sonnet", "anthropic/claude-3.5-haiku"]
    labels = model_labels(models, LABEL_MAX)
    assert len(set(labels.values())) == 2 and labels[models[0]].endswith("sonnet") and labels[models[1]].endswith("haiku")
    assert all(text_width(label, MODEL_PX) <= LABEL_MAX for label in labels.values())
    # ids alike at both ends are numbered in their order
    twins = ["vendor/model-alpha-2026-01-01-preview", "vendor/model-bravo-2026-01-01-preview"]
    numbered = model_labels(twins, LABEL_MAX)
    assert [numbered[m][:2] for m in twins] == ["1:", "2:"] and len(set(numbered.values())) == 2
    assert model_labels(["model-a", "model-b"], LABEL_MAX) == {"model-a": "model-a", "model-b": "model-b"}
    # and so in the chart: one label per model, each on its rows
    chart = ElementTree.fromstring(chart_of([base | {"name": "A B", "model": m} for m in models]))
    drawn = {g.get("data-model"): [t.text for t in g.iter("text") if t.get("class") == "nc-model"] for g in chart.iter("g")}
    assert drawn == {m: [labels[m]] for m in models}


def test_a_question_with_no_estimate_has_words_and_no_bar():
    """An empty gender bar would read as a probability of 0, and a line across nothing as a threshold failed."""
    row = {"name": "A B", "model": "m", "intended_gender": "male", "intended_origin": "palop", "p_intended_gender": "",
           "scored_p_portuguese": "0.2", "scored_p_palop": "0.8", "gender_threshold": "0.8", "origin_minimum": "0.25",
           "pass": "False"}
    row_g, = ElementTree.fromstring(chart_of([row])).iter("g")
    classes = [(e.get("class") or "").split()[0] for e in row_g.iter() if e.tag in ("rect", "line")]
    assert "nc-track" not in classes and "nc-gender" not in classes
    assert classes.count("nc-o") == 2 and classes.count("nc-thr") == 1  # the origin bar and its line only
    assert [t.text for t in row_g.iter("text")] == ["m", "names.chart.none.short"]
    nothing, = ElementTree.fromstring(chart_of([row | {"scored_p_portuguese": "", "scored_p_palop": ""}])).iter("g")
    assert [(e.get("class") or "").split()[0] for e in nothing.iter() if e.tag in ("rect", "line")] == ["nc-hit"]
    assert [t.text for t in nothing.iter("text")] == ["m", "names.chart.none.short", "names.chart.none"]


def test_every_interface_string_has_both_languages():
    assert all(len(pair) == 2 and all(pair) for pair in UI.values())
    used = set(re.findall(r'"(names\.[a-z_.]+)"', (ROOT / "src" / "hcai" / "site.py").read_text()
                          + (ROOT / "src" / "hcai" / "site_names.py").read_text()))
    assert {k for k in used if not k.endswith(".")} <= set(UI)  # dynamic keys end with a dot (names.origin.<key>)


# --- the committed record ----------------------------------------------------------------------------------------


def test_the_committed_record_renders_and_its_marks_match(tmp_path: Path):
    """The real site config and the record committed in docs/pilot/names (the Pages build has nothing else)."""
    record = ROOT / "docs" / "pilot" / "names"
    assert load_site_config(ROOT / "configs" / "core.toml").names == "docs/pilot/names"
    loaded = load_record(record)
    if loaded is None or not loaded.probability:  # the site shows such a record too, but without marks to compare
        pytest.skip("no committed probability-mode name check record")
    summary = rows_of(record / "summary.csv")
    root = tmp_path / "repo"  # the committed inputs only: no results/, as in CI
    shutil.copytree(ROOT / "configs", root / "configs")
    shutil.copytree(ROOT / "data" / "names", root / "data" / "names")
    shutil.copytree(record, root / "docs" / "pilot" / "names")
    site = Site(root / "configs" / "core.toml", tmp_path / "site")
    site.names_page()
    page = read_page(tmp_path / "site" / NAMES_PAGE)
    assert {(r["name"], r["model"]): r["mark"] for r in page.rows} == {
        (r["name"], r["model"]): "pass" if r["pass"] == "True" else "fail" for r in summary}
    assert len(page.svgs) == len({(r["intended_gender"], r["intended_origin"]) for r in summary})
    text = (tmp_path / "site" / NAMES_PAGE).read_text()
    names = list(dict.fromkeys(r["name"] for r in summary))
    failing = {r["name"] for r in summary if r["pass"] != "True"}
    assert f"<strong>{len(names) - len(failing)} / {len(names)}</strong>" in text
    assert "committed with the repository" in text and "class=\"error\"" not in text
    if (record / "reasked.csv").is_file():  # a run with --reasks: its re-asks, per model, add up to the files'
        section = text.split('id="h-reasks"', 1)[1].split('id="h-files"', 1)[0]
        per_model = first_table(section.split("<h3", 2)[1])  # model, answers, re-asks, ...
        assert sum(int(row[2]) for row in per_model) == len(rows_of(record / "reasked.csv")) == sum(
            int(r["reasks"]) for r in summary)
