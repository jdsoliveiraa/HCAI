"""The master command (#13): `hcai reproduce` on a tiny frozen run made with the fake LLM, offline.

The fixture is the skeleton design (1 role, 1 base CV, 2 identities, EN) under C0, C1 and C2, run once with
`hcai run --fake-llm` and laid out like `data/runs/core-v1`. The smoke test then runs the master command with
every socket refused, from a working directory that has a `.env` and a `results/` folder it must not read.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
from pathlib import Path

import pandas as pd
import pytest

from hcai.audit import read_criterion_labels, read_key, read_labels, rater_agreement
from hcai.cli import main
from hcai.reproduce import NetworkBlocked, no_network

SKELETON = Path(__file__).resolve().parent.parent / "configs" / "skeleton.toml"
CORPUS = SKELETON.parent.parent / "data" / "corpus"


@pytest.fixture
def frozen_run(tmp_path: Path, monkeypatch) -> Path:
    """A tiny run folder: results.csv, stability.json, proxy_leaks.json, run_summary.json and core.toml."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    config = tmp_path / "tiny.toml"
    config.write_text(
        SKELETON.read_text()
        .replace('conditions = ["C0"]', 'conditions = ["C0", "C1", "C2"]')
        .replace('corpus_dir = "../data/corpus"', f'corpus_dir = "{CORPUS}"')
        .replace("repeat_fraction = 0.0", "repeat_fraction = 1.0")
        + '\n[oversight]\njudge_model = "judge-model"\ncutoff = 50\nmargin = 5\n')
    work = tmp_path / "work"
    assert main(["run", str(config), "--out", str(work), "--cache", str(tmp_path / "cache"), "--fake-llm"]) == 0
    run = tmp_path / "runs" / "tiny-v1"
    run.mkdir(parents=True)
    for name in ("results.csv", "repeats.csv", "stability.json", "proxy_leaks.json", "proxy_leaks.csv",
                 "run_summary.json", "review_queue.csv"):
        if (work / name).exists():
            shutil.copy(work / name, run / name)
    shutil.copy(config, run / "core.toml")
    return run


def test_the_master_command_reproduces_everything_offline_from_a_frozen_run(frozen_run, tmp_path, monkeypatch, capsys):
    # a working directory with a key file and a results/ folder: the command must read neither
    cwd = tmp_path / "cwd"
    (cwd / "results").mkdir(parents=True)
    (cwd / "results" / "results.csv").write_text("not,a,results,table\n")
    (cwd / ".env").write_text("OPENCODE_API_KEY=must-not-be-read\n")
    monkeypatch.chdir(cwd)
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    # Proof that the guard is on while the analysis runs: a connection tried from inside it is refused.
    import hcai.reproduce as reproduce_module

    refused: list[bool] = []
    real_analyse = reproduce_module.analyse

    def analyse_trying_the_network(*args, **kwargs):
        try:
            socket.create_connection(("openrouter.ai", 443), timeout=1)
            refused.append(False)
        except NetworkBlocked:
            refused.append(True)
        return real_analyse(*args, **kwargs)

    monkeypatch.setattr(reproduce_module, "analyse", analyse_trying_the_network)
    before = {p: p.read_bytes() for p in frozen_run.iterdir()}

    code = main(["reproduce", str(frozen_run), "--labels", str(tmp_path / "no-labels"), "--config-base", str(tmp_path),
                 "--n-boot", "50", "--n-model-boot", "0", "--n-perm", "99"])

    assert code == 0
    assert refused and all(refused)
    assert "OPENCODE_API_KEY" not in os.environ
    assert {p: p.read_bytes() for p in frozen_run.iterdir() if p.is_file()} == before  # inputs untouched
    out = frozen_run / "analysis"
    tables = {p.stem for p in (out / "tables").glob("*.csv")}
    assert {"gaps", "gaps_by", "equivalence", "impact_ratios", "model_notes", "rank_identity", "rank_position",
            "ranking_quality", "oversight", "data_accounting", "screener_pass", "stability", "proxy_leaks"} <= tables
    # the post-hoc analyses (hcai.extra), under their own names
    assert {"extra_gaps_reference_free", "extra_gaps_subsets", "extra_rank_adjusted", "extra_repeat_noise",
            "extra_noise_vs_gap"} <= tables
    accounting = pd.read_csv(out / "tables" / "data_accounting.csv")
    assert set(accounting["condition"]) == {"C0", "C1", "C2"}
    figures = {p.name for p in (out / "figures").iterdir()}
    assert {"forest_models.pdf", "forest_models.png", "impact_heatmap.pdf", "impact_heatmap.png"} <= figures
    headline = json.loads((out / "headline.json").read_text())
    assert headline["data"]["rows"] == len(pd.read_csv(frozen_run / "results.csv"))
    assert {"rq1_fairness", "rq2_per_agent", "rq3_mitigation", "rq4_oversight", "extra_not_preregistered"} <= set(headline)
    assert headline["extra_not_preregistered"]["label"] == "post hoc, not pre-registered"
    assert headline["rq4_oversight"]["audit"]["status"]  # exported, or why not (too few flagged in a tiny run)
    manifest = (out / "MANIFEST.txt").read_text().splitlines()
    assert any(line.endswith("tables/gaps.csv") for line in manifest)
    assert "->" in capsys.readouterr().out

    # a rerun writes the same bytes: tables, figures and the headline are deterministic
    assert main(["reproduce", str(frozen_run), "--labels", str(tmp_path / "no-labels"), "--config-base", str(tmp_path),
                 "--n-boot", "50", "--n-model-boot", "0", "--n-perm", "99"]) == 0
    assert (out / "MANIFEST.txt").read_text().splitlines() == manifest


def test_the_network_guard_refuses_connections_and_lookups_then_restores_them():
    saved = (socket.socket.connect, socket.getaddrinfo, socket.create_connection)
    with no_network():
        with pytest.raises(NetworkBlocked):
            socket.create_connection(("example.org", 443), timeout=1)
        with pytest.raises(NetworkBlocked):
            socket.getaddrinfo("example.org", 443)
        with pytest.raises(NetworkBlocked):
            socket.socket().connect(("127.0.0.1", 9))
    assert (socket.socket.connect, socket.getaddrinfo, socket.create_connection) == saved


# --- per-quote labels (the audit page's export) ---------------------------------------------------------------


def write_key(path: Path) -> Path:
    verdicts = [[("faithful", "faithful")], [("faithful", "unfaithful")], [("unfaithful", "faithful")]]
    rows = []
    for k, item in enumerate(verdicts, start=1):
        quotes = [v for pair in item for v in pair]
        flag = "unfaithful" in quotes
        rows.append({"item_id": f"J{k:03d}", "n_criteria": len(quotes), "judge_flag": flag,
                     "judge_label": "unfaithful" if flag else "faithful",
                     "judge_verdicts": json.dumps([{"criterion": "c", "verdict": v, "reason": ""} for v in quotes])})
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_quote_sheet(path: Path, labels: dict[tuple[str, int], str]) -> Path:
    lines = ["item_id,criterion,label,comment"] + [f"{i},{c},{label}," for (i, c), label in labels.items()]
    path.write_text("﻿" + "\r\n".join(lines) + "\r\n", encoding="utf-8")
    return path


def test_a_per_quote_sheet_gives_item_labels_and_per_quote_kappa(tmp_path):
    key = read_key(write_key(tmp_path / "audit_key.csv"))
    judge = {("J001", 1): "faithful", ("J001", 2): "faithful", ("J002", 1): "faithful", ("J002", 2): "unfaithful",
             ("J003", 1): "unfaithful", ("J003", 2): "faithful"}
    same = write_quote_sheet(tmp_path / "labels_ana.csv", judge)
    other = write_quote_sheet(tmp_path / "labels_rui.csv", {**judge, ("J003", 1): "faithful"})

    items = read_labels(same, list(key["item_id"]), {"J001": 2, "J002": 2, "J003": 2})
    assert list(items) == ["faithful", "unfaithful", "unfaithful"]
    by_item, by_quote = rater_agreement(key, [same, other], n_boot=50)
    assert by_quote is not None
    judge_ana = by_quote[(by_quote["a"] == "judge") & (by_quote["b"] == "labels_ana")].iloc[0]
    assert judge_ana["kappa"] == pytest.approx(1.0) and judge_ana["n"] == 6
    judge_rui = by_item[(by_item["a"] == "judge") & (by_item["b"] == "labels_rui")].iloc[0]
    assert judge_rui["p_observed"] == pytest.approx(2 / 3)  # rui calls J003 faithful


def test_a_per_quote_sheet_with_an_unlabelled_or_unknown_criterion_is_refused(tmp_path):
    from hcai.audit import AuditLabelError

    sheet = write_quote_sheet(tmp_path / "labels_x.csv", {("J001", 1): "faithful", ("J001", 3): "faithful",
                                                          ("J002", 1): "maybe"})
    with pytest.raises(AuditLabelError) as e:
        read_criterion_labels(sheet, ["J001", "J002"], {"J001": 2, "J002": 1})
    message = str(e.value)
    assert "J001: criteria 2 unlabelled; criteria 3 unknown" in message
    assert "'maybe'" in message and "item J002 is missing" in message


def test_the_audit_is_exported_and_kappa_computed_once_two_sheets_are_in(frozen_run, tmp_path):
    # The fake judge flags nothing, so mark one of the two C2 justifications flagged, as a judge would.
    import csv

    path = frozen_run / "results.csv"
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    c2 = [r for r in rows if r["agent"] == "evaluator" and r["condition"] == "C2"]
    assert len(c2) == 2
    c2[0]["judge_flag"], c2[1]["judge_flag"] = "True", "False"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields or []))
        writer.writeheader()
        writer.writerows(rows)
    labels = tmp_path / "labels"
    labels.mkdir()
    out = tmp_path / "out"
    args = ["reproduce", str(frozen_run), "--out", str(out), "--labels", str(labels), "--config-base", str(tmp_path),
            "--n-boot", "20", "--n-model-boot", "0", "--n-perm", "19", "--audit-n", "2"]

    assert main(args) == 0
    exported = json.loads((out / "audit" / "audit_items.json").read_text())
    assert [i["item_id"] for i in exported["items"]] == ["J001", "J002"]
    assert "judge" not in json.dumps(exported)
    assert json.loads((out / "headline.json").read_text())["rq4_oversight"]["audit"]["status"].startswith("pending: 0")

    key = read_key(out / "audit" / "audit_key.csv")
    for rater in ("ana", "rui"):  # sheets as audit.js writes them: BOM, CRLF, one row per item x criterion
        rows = {(i, c): "faithful" for i, n in zip(key["item_id"], key["n_criteria"]) for c in range(1, int(n) + 1)}
        write_quote_sheet(labels / f"labels_{rater}.csv", rows)
    assert main(args) == 0
    agreement = pd.read_csv(out / "audit" / "agreement.csv")
    assert set(zip(agreement["a"], agreement["b"])) == {("judge", "labels_ana"), ("judge", "labels_rui"),
                                                        ("labels_ana", "labels_rui")}
    assert (out / "audit" / "agreement_criteria.csv").exists()
    assert json.loads((out / "headline.json").read_text())["rq4_oversight"]["audit"]["status"] == "computed"
