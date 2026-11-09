from hcai.report import format_report, score_gaps
from hcai.runner import ResultRow, run_experiment


def row(base_cv: str, identity: str, score: int | None, model: str = "model-a") -> ResultRow:
    return ResultRow(
        base_cv=base_cv, identity=identity, gender="", origin="", name="", language="en", condition="C0",
        model=model, agent="evaluator", score=score, rationale="", prompt_tokens=0, completion_tokens=0,
        cost=0.0, cached=False, status="ok" if score is not None else "missing", error="",
    )


def test_planted_gap_is_reported_against_the_reference_identity(design, fake_client, tmp_path):
    rows = run_experiment(design, fake_client, cache_dir=tmp_path / "cache")

    gaps = score_gaps(rows, reference="pt_m")

    assert [(g.identity, g.mean_gap, g.n_pairs) for g in gaps] == [("br_f", -5.0, 1)]
    assert "br_f" in format_report(gaps, reference="pt_m")
    assert "-5.0" in format_report(gaps, reference="pt_m")


def test_gap_is_averaged_over_pairs_and_skips_pairs_with_a_missing_score():
    rows = [
        row("cv1", "pt_m", 80), row("cv1", "br_f", 76),
        row("cv2", "pt_m", 60), row("cv2", "br_f", 54),
        row("cv3", "pt_m", 50), row("cv3", "br_f", None),
        row("cv1", "pt_m", 70, model="model-b"), row("cv1", "br_f", 71, model="model-b"),
    ]

    gaps = score_gaps(rows, reference="pt_m")

    assert [(g.identity, g.model, g.mean_gap, g.n_pairs) for g in gaps] == [
        ("br_f", "model-a", -5.0, 2),
        ("br_f", "model-b", 1.0, 1),
    ]
