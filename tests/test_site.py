"""The project site (`hcai site`, ADR 0010) and the realism review (spec #1 story 12)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

from hcai.cli import main
from hcai.review import (
    FORMAT, FORMAT_VERSION, SampleCV, SampleIdentity, blank_review, check_review, review_markdown, review_sample,
)
from hcai.site import OPEN, CLOSE, HEAD_SCRIPT, MARKER, UI, Site, build_site, review_payload, themed_css
from hcai.variants import render_variant

ASSETS = Path(__file__).parents[1] / "src" / "hcai" / "site_assets"
REVIEW_JS = ASSETS / "review.js"

ROLES = ["nurse", "accountant"]
QUALITIES = ["strong", "weak"]

JOB_AD = {
    "en": "# {role} (Lisbon)\n\nA {role} job.\n\n## Mandatory requirements\n- Degree\n\n## Desirable requirements\n- Experience\n",
    "pt": "# {role} (Lisboa)\n\nUm emprego.\n\n## Requisitos obrigatórios\n- Licenciatura\n\n## Requisitos preferenciais\n- Experiência\n",
}
CV = {
    "en": "# $name\n\nBased in Portugal · Nationality: $nationality\n$work_eligibility\n\n## Profile\n"
          "Nurse with 5 years <b>in</b> wards & clinics ({n}).\n",
    "pt": "# $name\n\nResidente em Portugal · Nacionalidade: $nationality\n$work_eligibility\n\n## Perfil\n"
          "{{{{Enfermeiro|Enfermeira}}}} com 5 anos em enfermarias ({n}).\n",
}
IDENTITIES = """
[[identities]]
id = "pt_m"
gender = "male"
origin = "portuguese"
names = ["Tiago Martins"]
nationality = "Portuguese"
nationalities = { en = "Portuguese", pt = "Portuguesa" }

[[identities]]
id = "pt_f"
gender = "female"
origin = "portuguese"
names = ["Rita Sousa"]
nationality = "Portuguese"
nationalities = { en = "Portuguese", pt = "Portuguesa" }

[[identities]]
id = "br_m"
gender = "male"
origin = "brazilian"
names = ["Leandro Lima"]
nationality = "Brazilian"
nationalities = { en = "Brazilian", pt = "Brasileira" }

[[identities]]
id = "br_f"
gender = "female"
origin = "brazilian"
names = ["Jéssica Oliveira"]
nationality = "Brazilian"
nationalities = { en = "Brazilian", pt = "Brasileira" }
"""


def write_project(root: Path, *, corpus: bool = True, results: bool = True) -> Path:
    """A tiny project: 2 roles x 2 quality levels x 2 base CVs (the second of each weak level borderline), EN + PT."""
    config = root / "configs" / "site.toml"
    config.parent.mkdir(parents=True)
    config.write_text(
        '[design]\nroles = ["nurse", "accountant"]\nquality_levels = ["strong", "weak"]\nbase_cvs_per_role = 2\n'
        'languages = ["en", "pt"]\nconditions = ["C0"]\nmodels = ["model-a"]\nseed = 7\n'
        'corpus_dir = "../data/corpus/v1"\n' + IDENTITIES
    )
    (root / "configs" / "other.toml").write_text("# another config <b>\n")
    if corpus:
        out = root / "data" / "corpus" / "v1"
        entries = []
        for role in ROLES:
            for lang, ad in JOB_AD.items():
                (out / role).mkdir(parents=True, exist_ok=True)
                (out / role / f"job_ad.{lang}.md").write_text(ad.format(role=role))
            for quality in QUALITIES:
                (out / role / quality).mkdir(parents=True, exist_ok=True)
                for n in (1, 2):
                    for lang, cv in CV.items():
                        (out / role / quality / f"cv0{n}.{lang}.md").write_text(cv.format(n=f"{role}-{quality}-{n}"))
                    borderline = quality == "weak" and n == 2
                    entries.append({"id": f"{role}/{quality}/cv0{n}", "role": role, "quality": quality,
                                    "borderline": borderline, "borderline_note": "only just" if borderline else ""})
        (out / "manifest.json").write_text(json.dumps({"version": "v1", "generator": {"model": "gen-x"},
                                                       "base_cvs": entries}))
    adr = root / "docs" / "adr"
    adr.mkdir(parents=True)
    (adr / "0001-first-decision.md").write_text("# 1. First decision\n\nDate: 2026-11-01 · Status: accepted\n\nWe chose `<x>`.\n")
    pilot = root / "docs" / "pilot"
    pilot.mkdir(parents=True)
    (pilot / "check-2026-11-14.json").write_text(json.dumps([
        {"model": "model-a", "provider": "opencode-go", "listed": True, "endpoint_ok": True, "json_schema_ok": True,
         "deterministic": False, "temperature_honoured": True, "seed_honoured": None, "calls": 11,
         "prompt_tokens": 10, "completion_tokens": 20, "error": ""}]))
    proposal = root / "proposal"
    proposal.mkdir()
    (proposal / "ijcai27.pdf").write_bytes(b"%PDF-1.4\n")
    (proposal / "ijcai27.tex").write_text(
        "\\begin{abstract}An \\emph{audit} of bias~\\cite{x}.\\end{abstract}\n"
        "\\item[\\textbf{RQ1}] Do scores differ by gender $\\times$ origin?\n")
    if results:
        names = root / "results" / "names"
        names.mkdir(parents=True)
        (names / "cells.csv").write_text("gender,origin,model,names\nmale,portuguese,model-a,1\n")
        (names / "summary.csv").write_text("name,model,pass\nTiago Martins,model-a,True\nRita Sousa,model-a,False\n")
        analysis = root / "results" / "analysis"
        analysis.mkdir()
        (analysis / "gaps.csv").write_text("identity,gap\npt_f,-1.5\n")
        (analysis / "forest_plot.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
    return config


@pytest.fixture
def project(tmp_path: Path) -> Path:
    return write_project(tmp_path / "project")


# --- link checker ------------------------------------------------------------------------------------------------


class _Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if a.get("id"):
            self.ids.add(str(a["id"]))
        for key in ("href", "src"):
            if a.get(key) is not None:
                self.links.append(str(a[key]))


class _ScriptOnly(HTMLParser):
    """Which elements with an id (and which "All hold" buttons) are inside `.js-only`, or carry it."""

    VOID = {"input", "br", "hr", "meta", "link", "img"}

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[bool] = []
        self.script_only: dict[str, bool] = {}
        self.all_hold: list[bool] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        hidden = "js-only" in (a.get("class") or "").split() or any(self.stack)
        if a.get("id"):
            self.script_only[str(a["id"])] = hidden
        if "data-all" in a:
            self.all_hold.append(hidden)
        if tag not in self.VOID:
            self.stack.append("js-only" in (a.get("class") or "").split())

    def handle_endtag(self, tag: str) -> None:
        if tag not in self.VOID and self.stack:
            self.stack.pop()


def parse(page: Path) -> _Links:
    p = _Links()
    p.feed(page.read_text(encoding="utf-8"))
    return p


def broken_links(site: Path) -> list[str]:
    """Every internal link of every page that is absolute or points at a missing file or anchor."""
    problems = []
    pages = {p: parse(p) for p in site.rglob("*.html")}
    for page, parsed in pages.items():
        for link in parsed.links:
            parts = urlsplit(link)
            if parts.scheme in ("http", "https", "mailto"):
                continue
            if parts.scheme or parts.netloc or link.startswith("/"):
                problems.append(f"{page.relative_to(site)}: not relative: {link}")
                continue
            target = (page.parent / unquote(parts.path)).resolve() if parts.path else page
            if not target.is_file() or site.resolve() not in target.parents:
                problems.append(f"{page.relative_to(site)}: missing {link}")
                continue
            if parts.fragment and target.suffix == ".html":
                ids = pages[target].ids if target in pages else parse(target).ids
                if parts.fragment not in ids:
                    problems.append(f"{page.relative_to(site)}: missing anchor {link}")
    return problems


# --- the generator -----------------------------------------------------------------------------------------------


def test_site_has_every_page_with_valid_relative_links(project: Path, tmp_path: Path):
    out = tmp_path / "site"
    pages = build_site(project, out)
    expected = {"index.html", "corpus/index.html", "corpus/nurse.html", "corpus/accountant.html",
                "corpus/nurse/strong/cv01.html", "corpus/accountant/weak/cv02.html", "review.html", "pilot.html",
                "results.html", "decisions/index.html", "decisions/0001-first-decision.html", "configs.html"}
    assert expected <= set(pages)
    assert len(pages) == len(set(pages))
    for page in pages:
        assert (out / page).is_file()
    assert (out / "files" / "ijcai27.pdf").is_file()
    assert (out / "figures" / "analysis" / "forest_plot.svg").is_file()
    assert broken_links(out) == []
    for page in pages:  # one h1 per page: documents shown inside a page have their headings demoted
        assert (out / page).read_text().count("<h1") == 1, page


def test_pages_share_the_nav_and_mark_the_current_page(project: Path, tmp_path: Path):
    out = tmp_path / "site"
    build_site(project, out)
    page = (out / "corpus" / "nurse" / "strong" / "cv01.html").read_text()
    assert 'href="../../../review.html"' in page and 'href="../../../assets/style.css"' in page
    assert re.search(r'<a href="../../index.html" aria-current="page">', page)
    home = (out / "index.html").read_text()
    assert '<a href="index.html" aria-current="page">' in home
    assert "An audit of bias." in home and "RQ1" in home and "gender × origin" in home
    assert "not written yet" in home  # no notebook, no report
    assert 'lang="en"' in home and 'name="viewport"' in home


def test_home_is_a_hub_with_every_part_and_its_state_before_the_prose(project: Path, tmp_path: Path):
    out = tmp_path / "site"
    build_site(project, out)
    home = (out / "index.html").read_text()
    cards = re.findall(r'<li class="card"><h3><a href="([^"]+)">.*?</a></h3><div class="card-state">(.*?)</div></li>', home)
    assert [href for href, _ in cards] == ["corpus/index.html", "review.html", "pilot.html", "pilot.html#names",
                                           "results.html", "decisions/index.html", "configs.html"]
    states = {href: re.sub(r"<[^>]+>", "", state) for href, state in cards}
    assert states["corpus/index.html"] == "v1: 4 job ads, 8 base CVs (en, pt), generator gen-x"
    assert states["review.html"] == "not committed yet"
    assert states["pilot.html"] == "model probes of 2026-11-14; no pilot run yet"
    assert states["pilot.html#names"] == "2 names, 1 failing"
    assert states["decisions/index.html"] == "ADRs: 1"
    assert states["configs.html"] == "other.toml, site.toml"
    # the cards' own words switch language with the rest of the interface
    assert dict(cards)["pilot.html"] == ('<span data-i18n="home.card.probes">model probes of</span> 2026-11-14; '
                                         '<span data-i18n="home.card.no_run">no pilot run yet</span>')
    # the lede and the cards come first, then the research questions, the summary and the links
    order = [home.index(mark) for mark in ('class="lede"', 'class="cards"', 'id="h-rqs"', 'id="h-summary"', 'id="h-links"')]
    assert order == sorted(order)
    bare = tmp_path / "bare"
    build_site(write_project(tmp_path / "empty", corpus=False, results=False), bare)
    empty = (bare / "index.html").read_text()
    assert empty.count('<li class="card">') == 7  # a part with nothing yet still has its card
    assert 'data-i18n="home.card.no_run"' in empty and "ADRs:</span> 1" in empty
    (tmp_path / "empty" / "docs" / "pilot" / "check-2026-11-14.json").unlink()
    build_site(tmp_path / "empty" / "configs" / "site.toml", bare)
    assert 'data-i18n="home.card.no_probes"' in (bare / "index.html").read_text()


def test_deep_pages_say_where_they_are_and_link_to_their_neighbours(project: Path, tmp_path: Path):
    out = tmp_path / "site"
    build_site(project, out)

    def crumbs(page: str) -> list[str]:
        nav = re.search(r'<nav class="crumbs"[^>]*><ol>(.*?)</ol></nav>', (out / page).read_text(), re.S)
        return re.findall(r'href="([^"]+)"', nav.group(1)) if nav else []

    assert crumbs("corpus/nurse.html") == ["index.html"]
    assert crumbs("corpus/nurse/weak/cv02.html") == ["../../index.html", "../../nurse.html"]
    assert crumbs("decisions/0001-first-decision.html") == ["index.html"]
    assert crumbs("index.html") == [] and crumbs("corpus/index.html") == []  # top-level pages are in the main nav

    def pager(page: str) -> dict[str, str]:
        return dict(re.findall(r'<a class="pager-(prev|next)" rel="(?:prev|next)" href="([^"]+)"', (out / page).read_text()))

    # base CVs follow each other in corpus order, across quality levels and roles
    assert pager("corpus/nurse/strong/cv01.html") == {"next": "cv02.html"}  # the first base CV has no previous one
    assert pager("corpus/nurse/strong/cv02.html") == {"prev": "cv01.html", "next": "../weak/cv01.html"}
    assert pager("corpus/nurse/weak/cv02.html") == {"prev": "cv01.html", "next": "../../accountant/strong/cv01.html"}
    assert pager("corpus/accountant/weak/cv02.html") == {"prev": "cv01.html"}
    # a page with several sections lists them (the links are checked by broken_links)
    for page, anchors in (("corpus/index.html", ["#h-nurse", "#h-accountant"]),
                          ("corpus/nurse/weak/cv02.html", ["#h-ids", "#h-tpl", "#h-var"]),
                          ("pilot.html", ["#h-probes", "#names", "#h-run", "#h-usage"]),
                          ("configs.html", ["#h-other", "#h-site"])):
        toc = re.search(r'<nav class="toc" aria-labelledby="toc-label">(.*?)</nav>', (out / page).read_text(), re.S)
        assert toc is not None and re.findall(r'href="([^"]+)"', toc.group(1)) == anchors, page
    assert broken_links(out) == []


def test_every_page_reads_without_scripts_and_no_script_blocks_it(project: Path, tmp_path: Path):
    out = tmp_path / "site"
    pages = build_site(project, out)
    for page in pages:
        text = (out / page).read_text()
        head, body = text.split("</head>")
        # the one inline script (theme before the first paint, and the mark that scripts run); the rest is deferred
        assert head.count("<script>") == 1 and f"<script>{HEAD_SCRIPT}</script>" in head, page
        assert re.findall(r"<script(?! defer src=| type=\"application/json\")[^>]*>", text) == ["<script>"], page
        assert "<script" not in body.replace('<script type="application/json"', ""), page
        # the switches do nothing without scripts, so they are only shown with them
        assert re.search(r'<div class="prefs js-only">\s*<button type="button" id="theme-toggle"', body), page
        assert body.index('id="lang-toggle"') < body.index("</header>")
        assert 'href="#main"' in body and '<main id="main"' in body and "<footer" in body, page
    css = (out / "assets" / "style.css").read_text()
    assert "html:not(.js) .js-only { display: none !important; }" in css
    assert 'className="js"' in HEAD_SCRIPT and "hcai-theme" in HEAD_SCRIPT
    review = (out / "review.html").read_text()
    shown = _ScriptOnly()
    shown.feed(review)
    for control in ("export", "export-md", "export-json", "import-file", "clear", "date-today", "keys-on",
                    "review-progress", "review-current", "save-state", "export-state"):
        assert shown.script_only[control], control
    assert shown.all_hold == [True] * 8
    for field in ("reviewer", "date", "verdict-freeze", "verdict-notes", "show-slots", "item-0-notes", "review-status"):
        assert not shown.script_only[field], field
    # the texts, the jump links and the slot highlighting need no script; the form says that saving does
    assert ".review:has(#show-slots:checked) mark.slot" in css
    assert 'href="#item-0"' in review and 'href="#finish"' in review
    assert f'<noscript><p class="error">{UI["review.nojs"][0]}</p></noscript>' in review
    # the variant viewer says what is on the page either way: a choice with scripts, the whole list without
    viewer = (out / "corpus" / "nurse" / "weak" / "cv02.html").read_text()
    assert '<span data-i18n="corpus.variants.choose" class="js-only">' in viewer
    assert '<span data-i18n="corpus.variants.all" class="no-js-only">' in viewer
    assert "html.js .no-js-only { display: none !important; }" in css


def test_theme_switch_reuses_the_dark_tokens(project: Path, tmp_path: Path):
    source = (ASSETS / "style.css").read_text()
    css = themed_css(source)
    system = re.search(r'@media \(prefers-color-scheme: dark\) \{\s*:root:not\(\[data-theme="light"\]\) \{(.*?)\}', css, re.S)
    forced = re.search(r':root\[data-theme="dark"\] \{(.*?)\}', css, re.S)
    assert system is not None and forced is not None

    def tokens(block: str) -> dict[str, str]:
        return dict(re.findall(r"(--[a-z0-9-]+|color-scheme):\s*([^;]+);", block))

    assert tokens(system.group(1)) == tokens(forced.group(1)) and len(tokens(forced.group(1))) > 10
    assert tokens(forced.group(1))["color-scheme"] == "dark"
    # a token added to the dark block of style.css reaches the switch without being written twice
    more = themed_css(source.replace("    --focus: #ffb366;\n", "    --focus: #ffb366;\n    --new-token: #123456;\n"))
    assert more.count("--new-token: #123456;") == 2
    with pytest.raises(ValueError):
        themed_css(":root { --bg: #fff; }")
    out = tmp_path / "site"
    build_site(project, out)
    assert (out / "assets" / "style.css").read_text() == css
    script = (out / "assets" / "site.js").read_text()
    assert '"hcai-theme"' in script and "/*STRINGS*/" not in script


def contrast(a: str, b: str) -> float:
    """The WCAG 2 contrast ratio of two #rrggbb colours."""
    def luminance(colour: str) -> float:
        parts = [int(colour[i: i + 2], 16) / 255 for i in (1, 3, 5)]
        r, g, b = (c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in parts)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_colour_tokens_meet_wcag_aa_in_both_themes():
    light_css, dark_css = (ASSETS / "style.css").read_text().split("@media (prefers-color-scheme: dark)", 1)
    themes = {"light": dict(re.findall(r"--([a-z-]+): (#[0-9a-f]{6});", light_css)),
              "dark": dict(re.findall(r"--([a-z-]+): (#[0-9a-f]{6});", dark_css.split("}", 1)[0]))}
    assert set(themes["light"]) == set(themes["dark"]) and "bg" in themes["light"]
    text = [("text", "bg"), ("text", "surface"), ("text", "tag"), ("muted", "bg"), ("muted", "surface"), ("link", "bg"),
            ("link", "surface"), ("accent-text", "accent"), ("mark-text", "mark"), ("borderline", "tag"),
            ("borderline", "surface"), ("error", "bg"), ("error", "surface")]
    # what marks a control or a state without text: field, button and chip borders (muted), a filled chip (accent),
    # the focus ring
    marks = [("muted", "bg"), ("accent", "bg"), ("focus", "bg"), ("focus", "surface")]
    for name, t in themes.items():
        for fg, bg in text:
            assert contrast(t[fg], t[bg]) >= 4.5, (name, fg, bg)
        for fg, bg in marks:
            assert contrast(t[fg], t[bg]) >= 3, (name, fg, bg)


def test_interface_strings_exist_in_both_languages(project: Path, tmp_path: Path):
    out = tmp_path / "site"
    pages = build_site(project, out)
    used: set[str] = set()
    for page in pages:
        used |= set(re.findall(r'data-i18n(?:-label)?="([^"]+)"', (out / page).read_text()))
    for script in ("site.js", "review.js", "variants.js"):  # every key a script names, however it is passed on
        used |= set(re.findall(r'"((?:review|corpus|home|nav|lang|theme|pager|toc|crumbs)\.[a-z_.]*[a-z])"',
                               (ASSETS / script).read_text()))
    assert {"review.item.done", "review.item.todo", "review.export.only.md", "review.said.all", "lang.switch"} <= used
    used |= {"theme.auto", "theme.light", "theme.dark"}  # built as "theme." + the theme
    assert used and used <= set(UI), sorted(used - set(UI))
    for key, (en, pt) in UI.items():
        assert en.strip() and pt.strip(), key
    assert UI["review.progress"][0].count("{done}") == UI["review.progress"][1].count("{done}") == 1
    assert UI["review.progress"][0].count("{total}") == UI["review.progress"][1].count("{total}") == 1


def test_corpus_pages_show_labels_and_highlight_demographic_differences(project: Path, tmp_path: Path):
    out = tmp_path / "site"
    build_site(project, out)
    index = (out / "corpus" / "index.html").read_text()
    assert "nurse/weak/cv02" in index and "only just" in index and "Borderline" in index
    page = (out / "corpus" / "nurse" / "weak" / "cv02.html").read_text()
    # every identity in every language, with its name and (PT) gendered word highlighted
    assert len(re.findall(r'<section class="variant doc"', page)) == 4 * 2
    assert '<mark class="slot">Rita Sousa</mark>' in page and '<mark class="slot">Enfermeira</mark>' in page
    assert '<mark class="slot">Brasileira</mark>' in page
    # generated text is escaped, never raw HTML
    assert "<b>in</b>" not in page and "&lt;b&gt;in&lt;/b&gt;" in page
    # the viewer's controls need scripts (hidden until they run); a second identity can be put beside the first
    assert '<div class="variant-controls" hidden>' in page and 'src="../../../assets/variants.js"' in page
    compare = re.search(r'<select id="v-compare">(.*?)</select>', page, re.S)
    assert compare is not None and 'for="v-compare"' in page
    assert re.findall(r'<option value="([^"]*)"', compare.group(1)) == ["", "pt_m", "pt_f", "br_m", "br_f"]
    shown = json.loads(node_run(f"""
        const v = require({json.dumps(str(ASSETS / "variants.js"))});
        process.stdout.write(JSON.stringify([v.shown("en", "pt_m", ""), v.shown("pt", "pt_m", "br_f"),
                                             v.shown("pt", "pt_m", "pt_m")]));
    """))
    assert shown == [{"lang": "en", "identities": ["pt_m"]}, {"lang": "pt", "identities": ["pt_m", "br_f"]},
                     {"lang": "pt", "identities": ["pt_m"]}]


def test_marked_variant_is_the_variant_with_markers(project: Path):
    template = CV["pt"].format(n="x")
    for gender in ("male", "female"):
        marked, plain = (render_variant(template, name="Rita Sousa", nationality="Portuguesa", gender=gender,
                                        language="pt", mark=mark) for mark in ((OPEN, CLOSE), None))
        assert marked.replace(OPEN, "").replace(CLOSE, "") == plain
        assert f"{OPEN}Rita Sousa{CLOSE}" in marked
    assert OPEN not in render_variant("$name\nNationality: $nationality\n$work_eligibility\n{{the|the}} x\n",
                                      name="A", nationality="B", gender="male", language="en",
                                      mark=(OPEN, CLOSE)).split("\n", 2)[2]


def test_pilot_and_results_pages_read_what_exists(project: Path, tmp_path: Path):
    out = tmp_path / "site"
    build_site(project, out)
    pilot = (out / "pilot.html").read_text()
    assert "model-a" in pilot and "opencode-go" in pilot
    assert "Rita Sousa" in pilot and "Tiago Martins" not in pilot  # only failing names are listed
    results = (out / "results.html").read_text()
    assert "pt_f" in results and "-1.5" in results
    assert "Not generated yet." in results  # the other analysis tables


def test_site_builds_with_no_corpus_and_no_results(tmp_path: Path):
    config = write_project(tmp_path / "project", corpus=False, results=False)
    out = tmp_path / "site"
    pages = build_site(config, out)
    for page in ("corpus/index.html", "review.html", "results.html"):
        assert "Not generated yet." in (out / page).read_text()
    assert "Not generated yet." in (out / "pilot.html").read_text()  # name check
    assert set(pages) >= {"index.html", "decisions/index.html"}
    assert broken_links(out) == []


def test_site_replaces_its_own_output_but_never_a_foreign_directory(project: Path, tmp_path: Path):
    out = tmp_path / "site"
    build_site(project, out)
    (out / "stale.html").write_text("old")
    build_site(project, out)
    assert not (out / "stale.html").exists() and (out / MARKER).exists()
    foreign = tmp_path / "mine"
    foreign.mkdir()
    (foreign / "keep.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        build_site(project, foreign)
    assert (foreign / "keep.txt").read_text() == "keep"


def test_cli_site_command(project: Path, tmp_path: Path, capsys):
    out = tmp_path / "site"
    assert main(["site", str(project), "--out", str(out)]) == 0
    assert "pages" in capsys.readouterr().out and (out / "index.html").exists()
    assert main(["site", str(project), "--out", str(project.parent)]) == 1  # a directory it did not write


def test_unknown_site_key_is_an_error(project: Path, tmp_path: Path):
    project.write_text(project.read_text() + '\n[site]\nnotebok = "x"\n')
    assert main(["site", str(project), "--out", str(tmp_path / "s")]) == 1


# --- the review sample ---------------------------------------------------------------------------------------------


def sample_inputs():
    cvs = [SampleCV(f"{role}/{q}/cv0{n}", role, q, q == "weak" and n == 2)
           for role in ["software_engineer", "nurse", "accountant"] for q in ["strong", "average", "weak"]
           for n in (1, 2)]
    identities = [SampleIdentity(f"{o}_{g[0]}", g, o) for o in ["pt", "br", "palop", "ee"] for g in ("male", "female")]
    return cvs, identities


def test_review_sample_is_deterministic_and_stratified():
    cvs, identities = sample_inputs()
    roles = ["software_engineer", "nurse", "accountant"]
    args = (roles, ["strong", "average", "weak"], ["en", "pt"], roles, cvs, identities)
    sample = review_sample(*args, seed=42)
    assert sample == review_sample(*args, seed=42)
    assert [i.id for i in sample.items[:6]] == [f"job_ad/{r}/{lang}" for r in roles for lang in ("en", "pt")]
    picked = sample.cvs
    assert len(picked) == 6
    assert [i.language for i in picked] == ["en", "pt"] * 3
    assert [i.borderline for i in picked] == [False, True] * 3
    assert [i.quality for i in picked[::2]] == ["strong", "average", "weak"]
    assert [i.gender for i in picked] == ["female", "male", "male", "female", "female", "male"]
    assert {(i.language, i.gender) for i in picked} == {(lang, g) for lang in ("en", "pt") for g in ("male", "female")}
    assert {i.origin for i in picked} == {"pt", "br", "palop", "ee"}
    others = {tuple(i.id for i in review_sample(*args, seed=s).cvs) for s in range(10)}
    assert len(others) > 1  # the seed matters


def test_review_sample_degrades_without_borderline_or_identities():
    cvs = [SampleCV("r/strong/cv01", "r", "strong", False), SampleCV("r/strong/cv02", "r", "strong", False)]
    ids = [SampleIdentity("x_f", "female", "x")]
    sample = review_sample(["r"], ["strong"], ["en"], ["r"], cvs, ids, seed=1)
    assert [i.base_cv for i in sample.cvs] in (["r/strong/cv01", "r/strong/cv02"], ["r/strong/cv02", "r/strong/cv01"])
    assert all(i.identity == "x_f" for i in sample.cvs)
    assert review_sample(["r"], ["strong"], ["en"], ["r"], cvs, [], seed=1).cvs == []


def test_site_review_sample_matches_the_rule(project: Path, tmp_path: Path):
    data = Site(project, tmp_path / "s").review_data()
    assert data is not None
    cvs = [i for i in data["items"] if i["kind"] == "cv"]
    assert [i["language"] for i in cvs] == ["en", "pt", "en", "pt"]
    assert [i["borderline"] for i in cvs] == [False, True, False, True]
    assert all(i["title"].startswith(f"CV: {i['base_cv']} (") for i in cvs)
    assert data["meta"] == {"corpus_version": "v1", "generator": "gen-x", "seed": 7, "languages": ["en", "pt"]}


# --- the review record ---------------------------------------------------------------------------------------------


def filled(data: dict) -> dict:
    review = blank_review(data["meta"], data["items"])
    review["reviewer"], review["date"] = "J. O.", "2026-11-17"
    review["verdict"] = {"freeze": True, "notes": "Good enough.\nOne odd date.  \n"}
    first, last = review["items"][0], review["items"][-1]
    first["checks"]["realistic"] = True
    first["notes"] = "Fine"
    last["checks"] = {c: True for c in last["checks"]}
    last["checks"]["no_stray_cues"] = False
    last["notes"] = "Line one\r\n\r\nline three"
    return review


def test_markdown_record_keeps_the_pilot_layout(project: Path, tmp_path: Path):
    data = Site(project, tmp_path / "s").review_data()
    assert data is not None
    blank = review_markdown(data["meta"], data["items"], blank_review(data["meta"], data["items"]))
    lines = blank.split("\n")
    assert lines[0] == "# CV realism review (pilot, #11 Part B)"
    assert "Reviewer: ______  Date: ______" in lines
    i = lines.index("## Overall verdict")
    assert lines[i + 1: i + 6] == ["", "- [ ] Corpus is realistic enough to freeze", "- Notes:", "", ""]
    j = lines.index("## Job ad: nurse (en)")
    assert lines[j + 1: j + 6] == ["", "- [ ] Realistic  - [ ] Language OK  - Notes:", "",
                                   "<details><summary>Text</summary>", ""]
    k = next(n for n, line in enumerate(lines) if line.startswith("## CV: nurse/"))
    assert lines[k + 2] == ("- [ ] Realistic  - [ ] Quality matches label  - [ ] Meets mandatory  - [ ] Language OK"
                            "  - [ ] No stray cues")
    assert lines[k + 3] == "- Notes:"
    assert lines[k + 4: k + 7] == ["", "<details><summary>Text</summary>", ""]
    assert blank.count("</details>") == len(data["items"])

    md = review_markdown(data["meta"], data["items"], filled(data))
    assert "Reviewer: J. O.  Date: 2026-11-17" in md
    assert "- [x] Corpus is realistic enough to freeze\n- Notes: Good enough.\n  One odd date.\n" in md
    assert "- [x] Realistic  - [ ] Language OK  - Notes: Fine\n" in md
    assert "- [x] Language OK  - [ ] No stray cues\n- Notes: Line one\n  \n  line three\n" in md


def test_check_review_refuses_another_sample(project: Path, tmp_path: Path):
    data = Site(project, tmp_path / "s").review_data()
    assert data is not None
    meta, items = data["meta"], data["items"]
    good = filled(data)
    assert check_review(good, meta, items) is None
    assert check_review({"format": "x"}, meta, items) == "not a realism review export"
    assert "corpus version" in str(check_review(good | {"corpus_version": "v2"}, meta, items))
    assert "seed" in str(check_review(good | {"seed": 8}, meta, items))
    assert check_review(good | {"items": good["items"][1:]}, meta, items) == "its items differ from this sample's"
    assert check_review(good | {"items": None}, meta, items) == "its items differ from this sample's"


def node_run(script: str) -> str:
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_browser_export_matches_python_and_round_trips(project: Path, tmp_path: Path):
    """The review page's JS writes the same Markdown as review_markdown, and its JSON imports back unchanged."""
    data = Site(project, tmp_path / "s").review_data()
    assert data is not None
    payload = review_payload(data)
    review = filled(data)
    (tmp_path / "payload.json").write_text(json.dumps(payload))
    (tmp_path / "review.json").write_text(json.dumps(review))
    out = node_run(f"""
        const fs = require("fs");
        const r = require({json.dumps(str(REVIEW_JS))});
        const data = JSON.parse(fs.readFileSync({json.dumps(str(tmp_path / "payload.json"))}, "utf8"));
        const review = JSON.parse(fs.readFileSync({json.dumps(str(tmp_path / "review.json"))}, "utf8"));
        const exported = JSON.stringify(review, null, 2) + "\\n";
        const imported = r.normalize(JSON.parse(exported), data);
        const bad = Object.assign({{}}, review, {{seed: 8}});
        process.stdout.write(JSON.stringify({{
          md: r.reviewMarkdown(data, review), imported: imported, check: r.checkReview(review, data),
          bad: r.checkReview(bad, data), notReview: r.checkReview([], data),
          blank: r.normalize(null, data), fewer: r.checkReview(Object.assign({{}}, review, {{items: review.items.slice(1)}}), data),
          keys: Object.keys(imported), itemKeys: Object.keys(imported.items[0]), storageKey: r.storageKey(data.meta),
        }}));
    """)
    result = json.loads(out)
    assert result["md"] == review_markdown(data["meta"], data["items"], review)
    assert result["imported"] == review
    assert result["check"] is None
    assert result["bad"] and "seed" in result["bad"]
    assert result["notReview"] == "not a realism review export"
    assert result["fewer"] == "its items differ from this sample's"
    assert result["blank"] == blank_review(data["meta"], data["items"])
    # the record is still format version 1, field for field, and a draft is still found under the same key: the
    # page's progress and shortcuts add nothing to either
    assert result["keys"] == list(blank_review(data["meta"], data["items"])) == [
        "format", "format_version", "corpus_version", "generator", "seed", "reviewer", "date", "verdict", "items"]
    assert result["itemKeys"] == ["id", "title", "checks", "notes"] and FORMAT_VERSION == 1
    assert result["storageKey"] == "hcai-review:v1:7"


def test_review_page_logic_for_export_state_current_item_and_keys():
    """What the review page decides, as its script decides it (the functions the page calls)."""
    out = json.loads(node_run(f"""
        const r = require({json.dumps(str(REVIEW_JS))});
        const tops = [-2600, -900, 150, 1400];  // four items; the third has reached the line
        process.stdout.write(JSON.stringify({{
          // which answers each exported file holds: both, one, an older state, or none
          exp: [r.exportState("A", null, null), r.exportState("A", null, {{md: "A", json: "A"}}),
                r.exportState("B", null, {{md: "B", json: "A"}}), r.exportState("B", null, {{md: "A", json: "B"}}),
                r.exportState("B", null, {{md: "A", json: "A"}}), r.exportState("A", "A", null),
                r.exportState("A", "A", {{md: "B", json: "B"}}), r.exportState("A", "C", {{json: "A"}})],
          marker: [r.parseExported(null), r.parseExported("not json"), r.parseExported('"A"'), r.parseExported("[]"),
                   r.parseExported('{{"format": "hcai-realism-review"}}'), r.parseExported('{{"md": "A", "json": 3}}'),
                   r.parseExported('{{"md": "A", "json": "B", "x": 1}}')],
          // the current item follows what is on screen
          cur: [r.currentIndex(tops, 4000, 300, false, 800), r.currentIndex([400, 1200], 4000, 300, false, 800),
                r.currentIndex([-5000, -3000], 250, 300, false, 800),
                r.currentIndex([-5000, -300], 700, 384, true, 1080),   // a tall screen at the end of the page
                r.currentIndex([-5000, -300], 700, 384, false, 1080),  // the same place, not at the end
                r.currentIndex([], 900, 300, false, 800)],
          keys: [r.keyAction("j", false), r.keyAction("K", false), r.keyAction("1", false), r.keyAction("5", false),
                 r.keyAction("a", false), r.keyAction("n", false), r.keyAction("Escape", false), r.keyAction("x", false),
                 r.keyAction("0", false), r.keyAction("j", true), r.keyAction("a", true), r.keyAction("3", true),
                 r.keyAction("Escape", true), r.keyAction("Enter", false), r.keyAction(" ", false)],
        }}));
    """))
    assert out["exp"] == ["review.export.none", "review.export.same", "review.export.only.md", "review.export.only.json",
                          "review.export.changed", "review.export.committed", "review.export.committed",
                          "review.export.only.json"]
    assert set(out["exp"]) <= set(UI)
    assert out["marker"] == [None, None, None, None, None, {"md": "A"}, {"md": "A", "json": "B"}]
    assert out["cur"] == [2, -1, 2, 2, 1, -1]  # 2 of two items is "Verdict and export"; -1 is before the first item
    act = [k and k["act"] for k in out["keys"]]
    assert act == ["next", "prev", "check", "check", "all", "notes", None, None, None, None, None, None, "leave", None, None]
    assert [k["n"] for k in out["keys"][2:4]] == [0, 4]


def test_an_item_is_reviewed_when_every_check_is_ticked_or_it_has_a_note(project: Path, tmp_path: Path):
    """The page's progress is read off the answers (nothing is stored for it)."""
    data = Site(project, tmp_path / "s").review_data()
    assert data is not None
    review = filled(data)  # first item: one of two checks and a note; last item: four of five checks and a note
    review["items"][1]["checks"] = {c: True for c in review["items"][1]["checks"]}  # every check, no note
    review["items"][2]["checks"]["realistic"] = True  # one of two checks, no note
    review["items"][3]["notes"] = "  \n "  # a blank note is no note
    (tmp_path / "review.json").write_text(json.dumps(review))
    out = json.loads(node_run(f"""
        const fs = require("fs");
        const r = require({json.dumps(str(REVIEW_JS))});
        const review = JSON.parse(fs.readFileSync({json.dumps(str(tmp_path / "review.json"))}, "utf8"));
        process.stdout.write(JSON.stringify({{
          done: review.items.map(r.itemDone), progress: r.progress(review),
          odd: [r.itemDone(null), r.itemDone({{}}), r.itemDone({{checks: {{}}, notes: ""}})],
        }}));
    """))
    assert out["done"] == [True, True, False, False, False, False, False, True]
    assert out["progress"] == {"done": 3, "total": 8}
    assert out["odd"] == [False, False, False]


def test_committed_review_is_preloaded_only_when_it_matches(project: Path, tmp_path: Path):
    site = Site(project, tmp_path / "s")
    data = site.review_data()
    assert data is not None
    record = project.parents[1] / "docs" / "pilot" / "cv-realism-review.json"
    record.write_text(json.dumps(filled(data)))
    again = Site(project, tmp_path / "s").review_data()
    assert again is not None and again["committed"]["reviewer"] == "J. O." and again["committed_mismatch"] == ""
    record.write_text(json.dumps(filled(data) | {"corpus_version": "v0"}))
    stale = Site(project, tmp_path / "s").review_data()
    assert stale is not None and stale["committed"] is None and "corpus version" in stale["committed_mismatch"]
    out = tmp_path / "site"
    build_site(project, out)
    page = (out / "review.html").read_text()
    assert "corpus version" in page and FORMAT in page


def test_review_page_is_an_accessible_form(project: Path, tmp_path: Path):
    out = tmp_path / "site"
    build_site(project, out)
    page = (out / "review.html").read_text()
    parsed = parse(out / "review.html")
    # every textarea and text input has a label
    for field_id in re.findall(r'<(?:textarea|input) id="([^"]+)"', page):
        assert f'for="{field_id}"' in page, field_id
    assert page.count('type="checkbox" data-check=') == 2 * 4 + 5 * 4  # 4 job ads, 4 CVs
    assert "Export review" in page and "Import JSON" in page and 'role="status"' in page
    assert "review-data" in parsed.ids
    blob = re.search(r'<script type="application/json" id="review-data">(.*?)</script>', page, re.S)
    assert blob is not None
    assert len(json.loads(blob.group(1))["items"]) == 8
    # landmarks and names: the strip's jump links are a labelled nav, each item a labelled article, messages are
    # status regions, and the buttons are grouped under a name
    assert '<nav aria-label="Review items" data-i18n-label="review.items.label"><ol class="chips">' in page
    # messages are status regions: on load and on the first save, after an export or import, and (for a screen
    # reader only) what a key or "All hold" just did
    assert re.findall(r'<p id="([a-z-]+)" class="[a-z-]+" role="status" aria-live="polite">', page) == [
        "review-status", "review-said", "finish-status"]
    assert '<p id="review-said" class="visually-hidden"' in page
    assert len(re.findall(r'<article class="item" id="item-\d+" data-item="[^"]+" aria-labelledby="item-\d+-h" tabindex="-1">', page)) == 8
    assert len(re.findall(r'role="group" aria-label="[^"]+" data-i18n-label="[^"]+"', page)) == 2


def test_review_page_follows_the_reviewers_order(project: Path, tmp_path: Path):
    out = tmp_path / "site"
    build_site(project, out)
    page = (out / "review.html").read_text()
    # the strip: one jump link per item, numbered from 1, then the verdict and export
    chips = re.findall(r'<a class="chip[^"]*" href="(#[^"]+)"', page)
    assert chips == [f"#item-{n}" for n in range(8)] + ["#finish"]
    assert re.search(r'<a class="chip" href="#item-0" data-chip="0" title="Job ad: nurse \(en\)">.*? 1<span', page)
    articles = re.findall(r'<article class="item".*?</article>', page, re.S)
    assert len(articles) == 8
    for n, article in enumerate(articles):
        # the text first, then its checklist and notes (beside it on a wide screen, below it on a narrow one)
        assert article.index('class="doc"') < article.index('class="answers"') < article.index('class="checks"') \
            < article.index("data-notes"), n
        assert f'<span class="item-n">{n + 1}</span>' in article
        assert article.count("data-all") == 1  # "All hold" is per item ...
    assert page.count("data-all") == 8  # ... and there is no page-wide one
    # who and when, the items, then the verdict and the export
    order = [page.index(mark) for mark in ('id="reviewer"', 'id="review-bar"', 'id="item-0"', 'id="item-7"', 'id="finish"',
                                           'id="verdict-freeze"', 'id="export"', 'id="import-file"', 'id="review-data"')]
    assert order == sorted(order)
    # a key can land in the last section too, so that Tab goes on from there
    assert '<section class="finish" id="finish" aria-labelledby="h-finish" tabindex="-1">' in page
    # the keys are listed, and can be turned off
    keys = re.search(r'<details class="keys js-only">(.*?)</details>', page, re.S)
    assert keys is not None and 'id="keys-on" checked' in keys.group(1)
    assert re.findall(r"<kbd>([^<]+)</kbd>", keys.group(1)) == ["j", "k", "1", "5", "a", "n", "Esc"]


def test_generated_text_cannot_inject_markup_links_or_break_the_review_data(tmp_path: Path):
    config = write_project(tmp_path / "project")
    ad = tmp_path / "project" / "data" / "corpus" / "v1" / "nurse" / "job_ad.en.md"
    ad.write_text(ad.read_text() + "\nNote <!-- <script>alert(1)</script> [click](javascript:alert(1)) "
                  "![t](https://evil.example/t.png) `a & b <c>`\n")
    out = tmp_path / "site"
    build_site(config, out)
    for page in ("corpus/nurse.html", "review.html"):
        text = (out / page).read_text()
        assert "<script>alert" not in text and 'href="javascript:' not in text and "<img" not in text, page
        assert "<code>a &amp; b &lt;c&gt;</code>" in text, page
    review = (out / "review.html").read_text()
    blob = re.search(r'<script type="application/json" id="review-data">(.*?)</script>', review, re.S)
    assert blob is not None and "<" not in blob.group(1)
    assert "alert(1)" in json.dumps(json.loads(blob.group(1)))
    assert review.rstrip().endswith("</html>") and 'src="assets/review.js"' in review


def test_bad_inputs_are_reported_not_fatal(tmp_path: Path):
    config = write_project(tmp_path / "project")
    root = tmp_path / "project"
    (root / "results" / "names" / "summary.csv").write_text("")
    (root / "docs" / "pilot" / "cv-realism-review.json").write_text('{"format": "hcai-realism-review", "items": null}')
    for cv in (root / "data" / "corpus" / "v1").rglob("cv0*.pt.md"):  # a slot error in every PT template
        cv.write_text(cv.read_text() + "\nO engenheiro.\n")
    out = tmp_path / "site"
    build_site(config, out)
    review = (out / "review.html").read_text()
    assert "Left out of the sample" in review and "unsupported format version" in review
    assert "does not match the current sample" in (out / "index.html").read_text()


def test_the_audit_page_shows_each_items_quotes_with_a_choice_per_quote_and_no_judge_verdict(project: Path, tmp_path: Path):
    audit = project.parent.parent / "data" / "runs" / "core-v1" / "analysis" / "audit"
    audit.mkdir(parents=True)
    items = [{"item_id": f"J00{k}", "language": "pt", "cv": f"# Ana\n\nCV text {k} <b>x</b>", "points": 10,
              "criteria": [{"criterion": "Licenciatura", "score": 10, "quote": "Licenciatura em Gestão"},
                           {"criterion": "OCC", "score": 0, "quote": ""}]} for k in (1, 2)]
    (audit / "audit_items.json").write_text(json.dumps({"n": 2, "seed": 0, "items": items}), encoding="utf-8")
    (audit / "audit_key.csv").write_text("item_id,judge_flag,judge_label\nJ001,True,unfaithful\n")
    out = tmp_path / "site"
    pages = build_site(project, out)

    assert "audit.html" in pages
    page = (out / "audit.html").read_text()
    assert page.count('<article class="audit-item"') == 2
    assert page.count('type="radio"') == 2 * 2 * 2  # 2 items x 2 quotes x faithful/unfaithful
    assert 'name="J001-2" value="unfaithful"' in page and "Licenciatura em Gestão" in page
    assert "&lt;b&gt;x&lt;/b&gt;" in page  # CV text is escaped, never markup
    assert "judge_flag" not in page and "judge_label" not in page
    assert 'src="assets/audit.js"' in page and (out / "assets" / "audit.js").is_file()
    assert broken_links(out) == []
