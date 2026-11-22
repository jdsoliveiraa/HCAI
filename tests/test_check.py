"""`hcai check`: the per-model probes of endpoint, JSON schema, temperature and seed, on fake models."""

import json
import random
from pathlib import Path

import pytest

from hcai.check import format_probes, probe_model
from hcai.cli import check_command, main
from hcai.llm import FakeLLMClient, LLMRequest, ProviderError

SKELETON = Path(__file__).parent.parent / "configs" / "skeleton.toml"


def model(*, honours_seed: bool = True, honours_temperature: bool = True, json_ok: bool = True):
    """A fake model: at temperature > 0 its answer depends on the seed (if honoured) or is fresh every time."""
    fresh = random.Random(0)

    def respond(request: LLMRequest) -> str | dict:
        if request.response_schema is not None:
            return {"word": "apple"} if json_ok else "not json"
        if request.temperature == 0 or not honours_temperature:
            return "Brightwater is a quiet town."
        draw = random.Random(request.seed).random() if honours_seed else fresh.random()
        return f"Town number {draw:.6f}."

    return FakeLLMClient(respond)


def test_a_well_behaved_model_passes_every_probe():
    client = model()

    r = probe_model(client, "m", provider="opencode-go", listed=True, repeats=3)

    assert (r.endpoint_ok, r.json_schema_ok, r.deterministic, r.temperature_honoured, r.seed_honoured) == (
        True, True, True, True, True)
    assert r.calls == len(client.calls) == 3 * 3 + 2  # repeated identical requests, none served from a cache
    assert r.prompt_tokens > 0 and r.error == ""


def test_a_model_that_ignores_the_seed_is_reported():
    r = probe_model(model(honours_seed=False), "m")

    assert (r.deterministic, r.temperature_honoured, r.seed_honoured) == (True, True, False)


def test_a_model_that_ignores_temperature_leaves_the_seed_undecided():
    r = probe_model(model(honours_temperature=False), "m")

    assert (r.deterministic, r.temperature_honoured, r.seed_honoured) == (True, False, None)


def test_a_model_without_json_schema_support_is_reported():
    assert probe_model(model(json_ok=False), "m").json_schema_ok is False


def test_a_model_the_endpoint_refuses_is_reported_without_raising():
    def refuse(request):
        raise RuntimeError("model not found")

    r = probe_model(FakeLLMClient(refuse), "m")

    assert r.endpoint_ok is False and "model not found" in r.error
    assert "NO" in format_probes([r]) and "m: RuntimeError: model not found" in format_probes([r])


def test_check_probes_every_configured_model_and_flags_unlisted_ones(tmp_path):
    results = check_command(SKELETON, model(), {"openrouter": ["other/model"]}, repeats=2)

    assert [(r.model, r.provider, r.listed) for r in results] == [("openai/gpt-4o-mini", "openrouter", False)]


def test_check_cli_writes_json_and_does_not_touch_the_cache(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("hcai.cli.make_client", lambda design, models: model())

    class Listing:
        def list_models(self):
            return ["openai/gpt-4o-mini"]

    monkeypatch.setattr("hcai.cli.http_client", lambda provider, **kw: Listing())

    code = main(["check", str(SKELETON), "--out", str(tmp_path / "check.json"), "--repeats", "2"])

    assert code == 0
    saved = json.loads((tmp_path / "check.json").read_text())
    assert saved[0]["model"] == "openai/gpt-4o-mini" and saved[0]["listed"] is True
    assert "t=0 deterministic" in capsys.readouterr().out
    assert not (tmp_path / ".cache").exists()


def test_repeats_below_two_are_refused():
    with pytest.raises(ValueError):
        probe_model(model(), "m", repeats=1)


def test_a_provider_that_refuses_the_response_format_is_a_no():
    def respond(request):
        if request.response_schema is not None:
            raise ProviderError("HTTP 400: response_format is not supported")
        return "same"

    r = probe_model(FakeLLMClient(respond), "m")

    assert r.json_schema_ok is False and "response_format" in r.error
