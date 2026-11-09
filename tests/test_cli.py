from pathlib import Path

from conftest import scores_by_name
from hcai.cli import main, run_command
from hcai.config import load_design
from hcai.llm import FakeLLMClient
from hcai.runner import read_results

SKELETON = Path(__file__).parent.parent / "configs" / "skeleton.toml"


def test_skeleton_config_holds_the_design_factors():
    design = load_design(SKELETON)

    assert len(design.roles) == 1
    assert design.base_cvs_per_role == 1
    assert len(design.identities) == 2
    assert design.languages == ["en"]
    assert design.conditions == ["C0"]
    assert design.quality_levels and design.models
    assert design.corpus_dir.is_dir()


def test_run_command_writes_results_and_returns_gap_report(tmp_path):
    design = load_design(SKELETON)
    a, b = (i.names[0] for i in design.identities)
    client = FakeLLMClient(scores_by_name({a: 72, b: 69}))

    report = run_command(SKELETON, out_dir=tmp_path, cache_dir=tmp_path / "cache", client=client)

    rows = read_results(tmp_path / "results.csv")
    assert len(rows) == 2 * len(design.models)
    assert "-3.0" in report


def test_dry_run_runs_end_to_end_without_api_key(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    exit_code = main(["run", str(SKELETON), "--out", str(tmp_path), "--cache", str(tmp_path / "cache"), "--dry-run"])

    assert exit_code == 0
    assert (tmp_path / "results.csv").exists()
    assert "Paired score gap" in capsys.readouterr().out


def test_dry_run_does_not_poison_the_default_cache_used_by_real_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    real = FakeLLMClient(lambda request: {"score": 90, "rationale": "real"})
    monkeypatch.setattr("hcai.cli.OpenRouterClient", lambda: real)

    main(["run", str(SKELETON), "--dry-run"])
    main(["run", str(SKELETON)])

    assert len(real.calls) == 2
    assert {r.score for r in read_results(tmp_path / "results" / "results.csv")} == {90}
