"""RQ4 oversight metrics and the human audit: flag-rate table, blind export, label import and Cohen's κ.

The export is exercised on a real C2 + C3 run through the fake client; the metrics on synthetic tables.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from hcai.analysis import load_results, results_frame
from hcai.audit import (
    AuditLabelError, agreement, audit_sample, cohens_kappa, export_audit, main, oversight_table, read_key, read_labels,
)
from hcai.blinding import fake_blinding
from hcai.config import Oversight
from hcai.llm import FakeLLMClient, LLMRequest
from hcai.runner import execute
from test_analysis import _row, one
from test_rubric import FABRICATED, JOB_AD, REAL_QUOTE, rubric

FLAG_REASON = "quote not in the CV"


def fake_crew(calls: list[LLMRequest]):
    """Everyone passes; model-a's rubric quotes a fabricated line (the judge flags it), model-b's quotes the CV."""

    def respond(request: LLMRequest):
        props = (request.response_schema or {}).get("properties", {})
        text = " ".join(m["content"] for m in request.messages)
        if "blinded_cv" in props:
            return fake_blinding(request)
        if "passed" in props:
            return {"passed": True, "reason": "meets the mandatory requirements"}
        if "ranking" in props:
            return {"ranking": props["ranking"]["items"]["enum"], "rationale": "kept the order shown"}
        if "criteria" in props:
            if request.model == "model-a":
                return rubric([10, 10, 8, 0], [REAL_QUOTE, REAL_QUOTE, FABRICATED, ""])
            return rubric([10, 10, 0, 0])
        if "verdicts" in props:
            calls.append(request)
            n = len(props["verdicts"]["items"]["properties"]["criterion"]["enum"])
            return {"verdicts": [
                {"criterion": i, "verdict": "unfaithful" if FABRICATED in text and i == 3 else "faithful",
                 "reason": FLAG_REASON if FABRICATED in text and i == 3 else "quote found"}
                for i in range(1, n + 1)]}
        raise AssertionError(f"unexpected request: {list(props)}")

    return respond


@pytest.fixture
def audited(design, corpus_dir, tmp_path):
    """A C2 + C3 run on 2 identities x 2 models: 8 judged decisions, the 4 of model-a flagged."""
    (corpus_dir / "data_analyst" / "job_ad.en.md").write_text(JOB_AD)
    design = replace(design, conditions=["C2", "C3"], models=["model-a", "model-b"],
                     oversight=Oversight(judge_model="judge-model", cutoff=0, margin=0))
    judge_calls: list[LLMRequest] = []
    execute(design, FakeLLMClient(fake_crew(judge_calls)), out_dir=tmp_path / "run", cache_dir=tmp_path / "cache")
    return design, load_results(tmp_path / "run" / "results.csv"), judge_calls


def read_sheet(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_labels(path: Path, labels: dict[str, str], extra: Sequence[tuple[str, str]] = ()) -> Path:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "cv", "criteria", "label", "comment"])
        for item_id, label in [*labels.items(), *extra]:
            writer.writerow([item_id, "cv text", "criteria", label, ""])
    return path


# --- export --------------------------------------------------------------------------------------


def test_the_audit_sheet_shows_no_judge_verdict_and_the_key_is_stratified_half_flagged(audited, tmp_path):
    design, results, _ = audited

    paths = export_audit(results, design, tmp_path / "audit", n=4, seed=0)

    items = read_sheet(paths.items)
    assert list(items[0]) == ["item_id", "cv", "criteria", "label", "comment"]
    assert len(items) == 4 and all(i["label"] == "" for i in items)
    sheet = paths.items.read_text(encoding="utf-8-sig")
    for leak in ("faithful", "unfaithful", FLAG_REASON, "quote found", "flag", "escalat", "judge"):
        assert leak not in sheet.lower()
    key = read_key(paths.key)
    assert list(key["item_id"]) == [i["item_id"] for i in items]
    assert sorted(key["judge_flag"]) == [False, False, True, True]
    assert set(key["judge_label"]) == {"faithful", "unfaithful"}


def test_each_item_shows_the_cv_the_judge_saw_with_every_score_and_quote(audited, tmp_path):
    design, results, judge_calls = audited

    paths = export_audit(results, design, tmp_path / "audit", n=8, seed=0)

    judged_texts = [m["content"] for c in judge_calls for m in c.messages]
    key = read_key(paths.key).set_index("item_id")
    for item in read_sheet(paths.items):
        assert any(item["cv"] in t for t in judged_texts)
        condition = key.loc[item["item_id"], "condition"]
        assert ("[redacted]" in item["cv"]) == (condition == "C3")
        assert ("João Silva" in item["cv"] or "Ana Souza" in item["cv"]) == (condition == "C2")
        assert item["criteria"].count("Score: ") == 4 and "Quote: «" in item["criteria"]
        fabricated = key.loc[item["item_id"], "model"] == "model-a"
        assert (FABRICATED in item["criteria"]) == fabricated


def test_the_sample_is_fifty_items_half_flagged_by_default_and_reproducible():
    rows = [_row(base_cv=f"nurse/strong/cv{k:02d}", identity="pt_m", gender="male", origin="portuguese", name="n",
                 language="en", condition="C2", model="model-a", agent="evaluator", score=50, evidence="[]",
                 judge_flag=k % 5 == 0, escalated=k % 5 == 0) for k in range(200)]
    results = results_frame(rows)

    sample = audit_sample(results, seed=1)

    assert len(sample) == 50 and int(sample["judge_flag"].sum()) == 25
    assert sample["base_cv"].is_unique
    assert list(audit_sample(results, seed=1)["base_cv"]) == list(sample["base_cv"])
    assert not sample["judge_flag"].iloc[:25].all()  # shuffled: the flagged half is not listed first


def test_too_few_flagged_justifications_for_a_stratified_sample_is_an_error(audited):
    _, results, _ = audited

    with pytest.raises(ValueError, match="only 4 flagged"):
        audit_sample(results, n=10)
    with pytest.raises(ValueError, match="even"):
        audit_sample(results, n=3)


# --- import --------------------------------------------------------------------------------------

IDS = ["J001", "J002", "J003"]


def test_labels_are_read_by_item_id_in_any_case_and_spacing(tmp_path):
    path = write_labels(tmp_path / "rater_a.csv", {"J002": " Unfaithful ", "J001": "faithful", "J003": "FAITHFUL"})

    labels = read_labels(path, IDS)

    assert labels.name == "rater_a"
    assert labels.to_dict() == {"J001": "faithful", "J002": "unfaithful", "J003": "faithful"}


@pytest.mark.parametrize("labels, extra, problem", [
    ({"J001": "faithful", "J002": "faithful"}, [], "item J003 is missing"),
    ({"J001": "faithful", "J002": "faithful", "J003": "faithful"}, [("J999", "faithful")], "unknown item id 'J999'"),
    ({"J001": "faithful", "J002": "faithful", "J003": "faithful"}, [("J001", "faithful")], "J001 is labelled more than once"),
    ({"J001": "faithful", "J002": "maybe", "J003": "faithful"}, [], "label 'maybe' is not one of faithful, unfaithful"),
    ({"J001": "faithful", "J002": "", "J003": "faithful"}, [], "label '' is not one of"),
])
def test_a_sheet_that_does_not_match_the_key_or_uses_other_labels_is_refused(tmp_path, labels, extra, problem):
    path = write_labels(tmp_path / "rater.csv", labels, extra)

    with pytest.raises(AuditLabelError, match=problem):
        read_labels(path, IDS)


# --- agreement -----------------------------------------------------------------------------------


def labels_from_counts(both_f: int, f_u: int, u_f: int, both_u: int) -> tuple[list[str], list[str]]:
    """Two raters' labels with the given 2x2 agreement matrix (rows: rater a, columns: rater b)."""
    cells = [("faithful", "faithful", both_f), ("faithful", "unfaithful", f_u),
             ("unfaithful", "faithful", u_f), ("unfaithful", "unfaithful", both_u)]
    a = [x for x, _, k in cells for _ in range(k)]
    b = [y for _, y, k in cells for _ in range(k)]
    return a, b


@pytest.mark.parametrize("counts, p_o, p_e, kappa", [
    ((20, 5, 10, 15), 0.70, 0.50, 0.40),  # the textbook example: kappa = (0.7 - 0.5) / (1 - 0.5)
    ((25, 0, 0, 25), 1.00, 0.50, 1.00),  # perfect agreement
    ((10, 15, 15, 10), 0.40, 0.50, -0.20),  # worse than chance
    ((45, 5, 0, 0), 0.90, 0.90, 0.00),  # b says faithful throughout: agreement no better than chance
])
def test_known_agreement_matrices_give_the_expected_kappa(counts, p_o, p_e, kappa):
    k = cohens_kappa(*labels_from_counts(*counts))

    assert k["p_observed"] == pytest.approx(p_o)
    assert k["p_expected"] == pytest.approx(p_e)
    assert k["kappa"] == pytest.approx(kappa)
    assert (k["both_faithful"], k["a_faithful_b_unfaithful"], k["a_unfaithful_b_faithful"], k["both_unfaithful"]) == counts


def test_kappa_is_undefined_when_both_raters_use_one_same_label():
    assert pd.isna(cohens_kappa(["faithful"] * 5, ["faithful"] * 5)["kappa"])


def test_agreement_reports_judge_against_each_rater_and_rater_against_rater():
    judge, rater_a = labels_from_counts(20, 5, 10, 15)
    rater_b = list(rater_a)  # b agrees with a everywhere
    ids = [f"J{k:03d}" for k in range(1, 51)]
    key = pd.DataFrame({"item_id": ids, "judge_label": judge})

    table = agreement(key, {"rater_a": pd.Series(rater_a, index=ids), "rater_b": pd.Series(rater_b, index=ids)},
                      n_boot=200)

    assert list(zip(table["a"], table["b"], table["kind"])) == [
        ("judge", "rater_a", "judge-human"), ("judge", "rater_b", "judge-human"), ("rater_a", "rater_b", "human-human")]
    assert one(table, a="judge", b="rater_a")["kappa"] == pytest.approx(0.4)
    human = one(table, a="rater_a", b="rater_b")
    assert human["kappa"] == 1.0 and human["kappa_ci95_low"] == 1.0
    judge_a = one(table, a="judge", b="rater_a")
    assert judge_a["kappa_ci95_low"] < 0.4 < judge_a["kappa_ci95_high"]


def test_the_command_line_exports_then_imports_two_raters_and_writes_kappa(audited, tmp_path, capsys):
    design, _, _ = audited
    config = tmp_path / "design.toml"
    config.write_text(f"""
[design]
roles = ["data_analyst"]
quality_levels = ["high"]
base_cvs_per_role = 1
languages = ["en"]
conditions = ["C2", "C3"]
models = ["model-a", "model-b"]
corpus_dir = "{design.corpus_dir}"

[[identities]]
id = "pt_m"
gender = "male"
origin = "portuguese"
names = ["João Silva"]
nationality = "Portuguese"

[[identities]]
id = "br_f"
gender = "female"
origin = "brazilian"
names = ["Ana Souza"]
nationality = "Brazilian"
""")
    out = tmp_path / "audit"

    assert main(["export", str(tmp_path / "run" / "results.csv"), "--config", str(config), "--out", str(out),
                 "--n", "8"]) == 0
    key = read_key(out / "audit_key.csv")
    truth = dict(zip(key["item_id"], key["judge_label"]))
    flipped = {i: ("faithful" if lab == "unfaithful" else "unfaithful") for i, lab in truth.items()}
    a = write_labels(tmp_path / "rater_a.csv", truth)
    b = write_labels(tmp_path / "rater_b.csv", flipped)

    assert main(["agreement", str(out / "audit_key.csv"), str(a), str(b), "--out", str(out), "--n-boot", "50"]) == 0
    table = pd.read_csv(out / "agreement.csv")
    assert one(table, a="judge", b="rater_a")["kappa"] == 1.0
    assert one(table, a="rater_a", b="rater_b")["kappa"] == -1.0
    assert "kappa judge vs rater_a (judge-human): 1.000" in capsys.readouterr().out

    write_labels(tmp_path / "rater_c.csv", {**truth, "J001": "perhaps"})
    assert main(["agreement", str(out / "audit_key.csv"), str(tmp_path / "rater_c.csv"), "--out", str(out)]) == 1
    assert "perhaps" in capsys.readouterr().err


# --- oversight table -----------------------------------------------------------------------------


def test_the_oversight_table_counts_flags_and_escalations_and_what_the_judge_cost():
    common = dict(identity="pt_m", gender="male", origin="portuguese", name="n", language="en", condition="C2")

    def decision(k: int, model: str, flag: bool | None, reason: str, status: str = "ok") -> list:
        base = dict(base_cv=f"nurse/strong/cv{k:02d}", model=model, **common)
        rows = [_row(**base, agent="evaluator", score=60 if status == "ok" else None, status=status,
                     judge_flag=flag, escalated=bool(reason), escalation_reason=reason)]
        if status == "ok":
            rows.append(_row(**base, agent="judge", judge_flag=flag, cost=0.01, attempts=1 if flag is not None else 3,
                             status="ok" if flag is not None else "missing", judge_model="judge-model"))
        return rows

    rows = (decision(1, "model-a", True, "flagged quote") + decision(2, "model-a", False, "")
            + decision(3, "model-a", False, "near cutoff") + decision(4, "model-a", True, "flagged quote; near cutoff")
            + decision(5, "model-b", None, "judge missing") + decision(6, "model-b", False, "")
            + decision(7, "model-b", None, "evaluator missing", status="missing"))
    rows.append(_row(base_cv="nurse/strong/cv01", model="model-a", **{**common, "condition": "C0"},
                     agent="evaluator", score=60))  # C0 is not overseen

    table = oversight_table(results_frame(rows))

    assert set(table["condition"]) == {"C2"}
    a = one(table, model="model-a")
    assert (a["decisions"], a["judged"], a["flagged"], a["escalated"], a["near_cutoff"]) == (4, 4, 2, 3, 2)
    assert a["flag_rate"] == 0.5 and a["escalation_rate"] == 0.75
    assert a["judge_calls"] == 4 and a["judge_cost"] == pytest.approx(0.04)
    b = one(table, model="model-b")
    assert (b["decisions"], b["judged"], b["flagged"], b["judge_missing"], b["evaluator_missing"]) == (3, 1, 0, 1, 1)
    assert b["escalated"] == 2 and b["judge_calls"] == 4
    pooled = one(table, model="all")
    assert (pooled["decisions"], pooled["flagged"], pooled["escalated"]) == (7, 2, 5)
    assert pooled["flag_rate"] == pytest.approx(2 / 5) and pooled["judge_cost"] == pytest.approx(0.06)


def test_the_oversight_table_of_a_run_matches_its_run_summary(audited, tmp_path):
    _, results, _ = audited
    summary = json.loads((tmp_path / "run" / "run_summary.json").read_text())["oversight"]

    table = oversight_table(results)

    for condition, by_model in summary.items():
        for model, counts in by_model.items():
            row = one(table, condition=condition, model=model)
            assert {k: int(row[k]) for k in counts} == counts


def test_a_sheet_saved_by_excel_in_windows_1252_with_semicolons_and_blank_rows_is_read(tmp_path):
    path = tmp_path / "rater_pt.csv"
    text = ("item_id;cv;criteria;label;comment\r\nJ001;Formação: João;x;Faithful;\r\nJ002;c;x;unfaithful;não\r\n"
            "J003;c;x;faithful;\r\n;;;;\r\n")
    path.write_bytes(text.encode("cp1252"))

    labels = read_labels(path, IDS)

    assert labels.to_dict() == {"J001": "faithful", "J002": "unfaithful", "J003": "faithful"}


def test_raters_with_the_same_file_name_or_the_judges_name_are_refused(tmp_path, capsys):
    key = tmp_path / "audit_key.csv"
    pd.DataFrame({"item_id": IDS, "judge_label": ["faithful"] * 3}).to_csv(key, index=False)
    labels = {i: "faithful" for i in IDS}
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = write_labels(tmp_path / "a" / "labels.csv", labels)
    b = write_labels(tmp_path / "b" / "labels.csv", labels)
    judge = write_labels(tmp_path / "judge.csv", labels)

    assert main(["agreement", str(key), str(a), str(b), "--out", str(tmp_path)]) == 1
    assert main(["agreement", str(key), str(judge), str(a), "--out", str(tmp_path)]) == 1
    assert "rename" in capsys.readouterr().err


def test_exporting_a_c3_item_without_its_blinder_row_is_a_clear_error(audited, tmp_path):
    design, results, _ = audited

    with pytest.raises(ValueError, match="no Blinder row"):
        export_audit(results[results["agent"] != "blinder"], design, tmp_path / "audit", n=8)
