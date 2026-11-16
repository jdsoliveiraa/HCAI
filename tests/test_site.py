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
    FORMAT, SampleCV, SampleIdentity, blank_review, check_review, review_markdown, review_sample,
)
from hcai.site import OPEN, CLOSE, MARKER, Site, build_site, review_payload
from hcai.variants import render_variant

REVIEW_JS = Path(__file__).parents[1] / "src" / "hcai" / "site_assets" / "review.js"

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
    assert [i.gender for i in picked] == ["female", "male"] * 3
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
