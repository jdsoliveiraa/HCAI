from dataclasses import replace
from pathlib import Path

from conftest import scores_by_name
from hcai.cli import main, run_command
from hcai.config import load_design
from hcai.llm import FakeLLMClient, Usage
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
    assert len(rows) == 2 * 3 * len(design.models)  # 2 CVs x 3 agents (one pool of 2)
    assert "-3.0" in report


def test_fake_llm_runs_end_to_end_without_api_key(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    exit_code = main(["run", str(SKELETON), "--out", str(tmp_path), "--cache", str(tmp_path / "cache"), "--fake-llm"])

    assert exit_code == 0
    assert (tmp_path / "results.csv").exists()
    assert "Paired score gap" in capsys.readouterr().out


def test_fake_llm_does_not_poison_the_default_cache_used_by_real_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a, b = (i.names[0] for i in load_design(SKELETON).identities)
    real = FakeLLMClient(scores_by_name({a: 90, b: 90}))
    monkeypatch.setattr("hcai.cli.OpenRouterClient", lambda: real)

    main(["run", str(SKELETON), "--fake-llm"])
    main(["run", str(SKELETON)])

    assert len(real.calls) == 5  # 2 Screener + 2 Evaluator + 1 Ranker
    rows = read_results(tmp_path / "results" / "results.csv")
    assert {r.score for r in rows if r.agent == "evaluator"} == {90}


def test_rerun_summary_reports_no_new_spend(tmp_path):
    a, b = (i.names[0] for i in load_design(SKELETON).identities)
    priced = scores_by_name({a: 60, b: 60})

    class PricedClient(FakeLLMClient):
        def complete(self, request):
            r = super().complete(request)
            return replace(r, usage=Usage(r.usage.prompt_tokens, r.usage.completion_tokens, cost=0.25))

    first = run_command(SKELETON, out_dir=tmp_path, cache_dir=tmp_path / "cache", client=PricedClient(priced))
    second = run_command(SKELETON, out_dir=tmp_path, cache_dir=tmp_path / "cache", client=PricedClient(priced))

    assert "spent $1.2500 this run" in first  # 5 calls: 2 Screener + 2 Evaluator + 1 Ranker
    assert "spent $0.0000 this run" in second


def test_pilot_flag_runs_the_slice_into_its_own_folder(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    exit_code = main(["run", str(SKELETON), "--pilot", "--fake-llm"])

    assert exit_code == 0
    assert (tmp_path / "results" / "pilot" / "results.csv").exists()
    assert (tmp_path / "results" / "pilot" / "run_summary.json").exists()
    assert not (tmp_path / "results" / "results.csv").exists()
    assert "estimated full-run cost" in capsys.readouterr().out


def test_run_prints_the_stability_report_when_the_config_asks_for_repeats(tmp_path):
    config = tmp_path / "with_repeats.toml"
    text = SKELETON.read_text().replace('corpus_dir = "../data/corpus"', f'corpus_dir = "{SKELETON.parent.parent / "data/corpus"}"')
    config.write_text(text.replace("repeat_fraction = 0.0", "repeat_fraction = 1.0"))
    a, b = (i.names[0] for i in load_design(SKELETON).identities)

    report = run_command(config, out_dir=tmp_path, cache_dir=tmp_path / "cache", client=FakeLLMClient(scores_by_name({a: 60, b: 60})))

    assert "Stability" in report and " 1.00 " in report
    assert (tmp_path / "stability.json").exists()


def test_fake_llm_runs_condition_c2_and_reports_escalations(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    config = tmp_path / "c2.toml"
    corpus = SKELETON.parent.parent / "data" / "corpus"
    config.write_text(
        SKELETON.read_text()
        .replace('conditions = ["C0"]', 'conditions = ["C0", "C2"]')
        .replace('corpus_dir = "../data/corpus"', f'corpus_dir = "{corpus}"')
        + '\n[oversight]\njudge_model = "judge-model"\ncutoff = 50\nmargin = 5\n'
    )

    exit_code = main(["run", str(config), "--out", str(tmp_path / "out"), "--cache", str(tmp_path / "cache"),
                      "--fake-llm"])

    assert exit_code == 0
    assert "oversight C2" in capsys.readouterr().out
    queue = read_results(tmp_path / "out" / "review_queue.csv")
    assert len(queue) == 2 and all(r.condition == "C2" and r.escalated for r in queue)  # fake scores sit at 50


def test_a_pilot_that_drops_a_condition_says_its_estimate_is_a_lower_bound(tmp_path, monkeypatch, capsys):
    config = tmp_path / "c1.toml"
    corpus = SKELETON.parent.parent / "data" / "corpus"
    config.write_text(
        SKELETON.read_text()
        .replace('conditions = ["C0"]', 'conditions = ["C0", "C1"]')
        .replace('corpus_dir = "../data/corpus"', f'corpus_dir = "{corpus}"')
        .replace("mixed_quality = 1", "mixed_quality = 1\nconditions = 1")
    )

    exit_code = main(["run", str(config), "--pilot", "--out", str(tmp_path / "out"), "--cache", str(tmp_path / "cache"),
                      "--fake-llm"])

    assert exit_code == 0
    assert "a lower bound: the pilot ran no condition like blinder C1" in capsys.readouterr().out
