"""Providers (#11 Part A): per-model provider config, keys and `.env`, rate limits, unknown cost, resume.

Every HTTP call goes to an in-process `httpx.MockTransport`; nothing leaves the machine.
"""

import json
import uuid
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from conftest import scores_by_name
from hcai.cli import main, models_command, run_command
from hcai.config import load_design
from hcai.llm import (
    USER_AGENT,
    CachedClient,
    FakeLLMClient,
    LLMRequest,
    OpenAICompatibleClient,
    ProviderError,
    ProviderExhausted,
    RateLimited,
    RateLimitGuard,
)
from hcai.providers import SESSION_ID, BUILTIN_PROVIDERS, MissingKey, check_keys, load_dotenv, provider_client
from hcai.runner import execute, read_results

SKELETON = Path(__file__).parent.parent / "configs" / "skeleton.toml"
CORPUS = SKELETON.parent.parent / "data" / "corpus"
SECRET = "sk-SECRET-do-not-leak-123"
REQ = LLMRequest(model="m", messages=[{"role": "user", "content": "hi"}])


def config(tmp_path: Path, extra: str = "", *, models_line: str | None = None) -> Path:
    """The skeleton config with its model on the default provider (no [model_providers]) plus `extra`."""
    text = SKELETON.read_text().replace('corpus_dir = "../data/corpus"', f'corpus_dir = "{CORPUS}"')
    text = text[: text.index("[model_providers]")]
    if models_line:
        text = text.replace('models = ["openai/gpt-4o-mini"]', models_line)
    path = tmp_path / "design.toml"
    path.write_text(text + extra)
    return path


def chat_answer(content: str, usage: dict | None = None) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}],
                                     "usage": usage or {"prompt_tokens": 10, "completion_tokens": 2}})


def crew_answer(request: httpx.Request) -> httpx.Response:
    """A provider that answers the crew like `scores_by_name` (no price in its usage block, as OpenCode Go)."""
    body = json.loads(request.content)
    schema = body.get("response_format", {}).get("json_schema", {}).get("schema")
    llm_request = LLMRequest(model=body["model"], messages=body["messages"], response_schema=schema)
    names = [i.names[0] for i in load_design(SKELETON).identities]
    out = scores_by_name({names[0]: 70, names[1]: 64})(llm_request)
    return chat_answer(json.dumps(out))


# --- config -------------------------------------------------------------------------------------------------


def test_an_unlisted_model_uses_opencode_go_and_one_line_switches_it_to_openrouter(tmp_path):
    default = load_design(config(tmp_path))
    switched = load_design(config(tmp_path, '\n[model_providers]\n"openai/gpt-4o-mini" = "openrouter"\n'))

    assert default.provider_for("openai/gpt-4o-mini") == BUILTIN_PROVIDERS["opencode-go"]
    assert default.provider_for("openai/gpt-4o-mini").base_url == "https://opencode.ai/zen/go/v1"
    assert default.provider_for("openai/gpt-4o-mini").key_env == "OPENCODE_API_KEY"
    assert switched.provider_for("openai/gpt-4o-mini").base_url == "https://openrouter.ai/api/v1"
    assert switched.provider_for("openai/gpt-4o-mini").key_env == "OPENROUTER_API_KEY"


def test_a_provider_can_be_overridden_or_added(tmp_path):
    design = load_design(config(tmp_path, (
        '\n[providers.opencode-go]\nbase_url = "https://proxy.example/v1"\n'
        '\n[providers.local]\nbase_url = "http://localhost:8000/v1"\nkey_env = "LOCAL_KEY"\nbilling = "subscription"\n'
        '\n[model_providers]\n"openai/gpt-4o-mini" = "local"\n')))

    assert design.providers["opencode-go"].base_url == "https://proxy.example/v1"
    assert design.providers["opencode-go"].key_env == "OPENCODE_API_KEY"
    assert design.provider_for("openai/gpt-4o-mini").key_env == "LOCAL_KEY"


@pytest.mark.parametrize("extra, message", [
    ('\n[model_providers]\n"openai/gpt-4o-mini" = "opencodego"\n', "unknown provider"),
    ('\n[model_providers]\n"openai/gpt4o-mini" = "openrouter"\n', "not a model of the config"),
    ('\n[providers.mine]\nbase_url = "https://x.example/v1"\n', "key_env"),
    ('\n[providers.openrouter]\nbilling = "free"\n', "billing"),
    ('\n[providers.openrouter]\nbase_ulr = "https://x.example/v1"\n', "unknown keys"),
])
def test_config_mistakes_in_providers_are_refused(tmp_path, extra, message):
    with pytest.raises(ValueError, match=message):
        load_design(config(tmp_path, extra))


def test_the_generator_and_judge_can_name_their_provider_too(tmp_path):
    design = load_design(config(tmp_path, (
        '\n[generator]\nmodel = "gen-model"\n\n[oversight]\njudge_model = "judge-model"\n'
        '\n[model_providers]\n"gen-model" = "openrouter"\n')))

    assert design.all_models == ["openai/gpt-4o-mini", "gen-model", "judge-model"]
    assert design.provider_for("gen-model").name == "openrouter"
    assert design.provider_for("judge-model").name == "opencode-go"


# --- keys and .env ------------------------------------------------------------------------------------------


def test_dotenv_sets_missing_variables_only(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# keys\nexport OPENCODE_API_KEY='sk-oc'\nOPENROUTER_API_KEY=sk-or # inline comment\n\nnot a line\n"
        'QUOTED="a b"\nALREADY=from-file\n'
    )
    env = {"ALREADY": "from-env", "OPENCODE_API_KEY": ""}  # an empty export does not hide the file's key

    loaded = load_dotenv(env_file, env)

    assert env == {"ALREADY": "from-env", "OPENCODE_API_KEY": "sk-oc", "OPENROUTER_API_KEY": "sk-or", "QUOTED": "a b"}
    assert sorted(loaded) == ["OPENCODE_API_KEY", "OPENROUTER_API_KEY", "QUOTED"]
    assert load_dotenv(tmp_path / "missing.env", env) == []


def test_dotenv_is_gitignored_and_the_example_is_not():
    ignore = (SKELETON.parent.parent / ".gitignore").read_text().splitlines()

    assert ".env" in ignore and "!.env.example" in ignore
    assert "OPENCODE_API_KEY=" in (SKELETON.parent.parent / ".env.example").read_text()


def test_a_missing_key_names_its_variable_and_fails_before_any_call():
    with pytest.raises(MissingKey, match="OPENCODE_API_KEY \\(provider opencode-go\\)"):
        provider_client(lambda m: BUILTIN_PROVIDERS["opencode-go"], ["m"], environ={})


def test_only_the_providers_in_use_need_a_key():
    check_keys([BUILTIN_PROVIDERS["openrouter"]], {"OPENROUTER_API_KEY": "sk-or"})  # no OpenCode key: fine


def test_run_without_keys_exits_with_the_missing_variable(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)  # no .env here
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)

    assert main(["run", str(config(tmp_path))]) == 1
    assert "OPENCODE_API_KEY" in capsys.readouterr().err


def test_main_reads_keys_from_a_dotenv_in_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.setattr("hcai.cli.load_dotenv", load_dotenv)  # the real one (conftest stubs it out)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    (tmp_path / ".env").write_text(f"OPENCODE_API_KEY={SECRET}\n")
    seen = {}

    def fake_make_client(design, models):
        import os
        seen["key"] = os.environ.get("OPENCODE_API_KEY")
        return FakeLLMClient(lambda r: {"score": 1})

    monkeypatch.setattr("hcai.cli.make_client", fake_make_client)
    main(["run", str(config(tmp_path)), "--out", str(tmp_path / "out"), "--cache", str(tmp_path / "c")])

    assert seen["key"] == SECRET


# --- the HTTP client ----------------------------------------------------------------------------------------


def routed_client(handler, *, sleep=lambda s: None, max_wait=6 * 3600.0, design=None, environ=None):
    design = design or load_design(SKELETON)
    return provider_client(
        design.provider_for, design.models, environ=environ or {"OPENCODE_API_KEY": SECRET, "OPENROUTER_API_KEY": SECRET},
        http=httpx.Client(transport=httpx.MockTransport(handler)), sleep=sleep, max_wait=max_wait,
    )


def test_each_model_goes_to_its_providers_url_with_its_key(tmp_path):
    seen: list[httpx.Request] = []

    def handler(request):
        seen.append(request)
        return chat_answer("ok")

    design = load_design(config(tmp_path, '\n[model_providers]\n"b" = "openrouter"\n', models_line='models = ["a", "b"]'))
    client = provider_client(design.provider_for, design.models,
                             environ={"OPENCODE_API_KEY": "sk-oc", "OPENROUTER_API_KEY": "sk-or"},
                             http=httpx.Client(transport=httpx.MockTransport(handler)))

    client.complete(replace(REQ, model="a"))
    client.complete(replace(REQ, model="b"))

    assert str(seen[0].url) == "https://opencode.ai/zen/go/v1/chat/completions"
    assert seen[0].headers["Authorization"] == "Bearer sk-oc"
    assert "usage" not in json.loads(seen[0].content)  # OpenRouter's price request is not sent to OpenCode Go
    assert str(seen[1].url) == "https://openrouter.ai/api/v1/chat/completions"
    assert seen[1].headers["Authorization"] == "Bearer sk-or"
    assert json.loads(seen[1].content)["usage"] == {"include": True}


def test_a_provider_that_reports_no_price_gives_an_unknown_cost_with_tokens():
    client = OpenAICompatibleClient("https://p.example/v1", "k", http=httpx.Client(
        transport=httpx.MockTransport(lambda r: chat_answer("hi", {"prompt_tokens": 7, "completion_tokens": 3}))))

    usage = client.complete(REQ).usage

    assert (usage.prompt_tokens, usage.completion_tokens, usage.cost) == (7, 3, None)


def test_models_are_listed_from_the_providers_models_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENCODE_API_KEY", "sk-oc")

    def handler(request):
        assert str(request.url) == "https://opencode.ai/zen/go/v1/models"
        return httpx.Response(200, json={"object": "list", "data": [{"id": "kimi-k2"}, {"id": "glm-4.6"}]})

    out = models_command(config(tmp_path), http=httpx.Client(transport=httpx.MockTransport(handler)))

    assert "opencode-go (https://opencode.ai/zen/go/v1): 2 models" in out
    assert out.index("glm-4.6") < out.index("kimi-k2")


# --- rate and usage limits ----------------------------------------------------------------------------------


def limited_then_ok(n_limited: int, headers: dict | None = None, status: int = 429):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) <= n_limited:
            return httpx.Response(status, headers=headers or {}, json={"error": {"message": "Rate limit exceeded"}})
        return chat_answer("ok")

    return handler, calls


def guarded(handler, **kw):
    sleeps: list[float] = []
    inner = OpenAICompatibleClient("https://p.example/v1", "k", http=httpx.Client(transport=httpx.MockTransport(handler)))
    return RateLimitGuard(inner, sleep=sleeps.append, log=lambda m: None, **kw), sleeps


def test_a_rate_limit_is_waited_out_for_retry_after_then_the_same_call_is_repeated():
    handler, calls = limited_then_ok(2, headers={"Retry-After": "12"})
    client, sleeps = guarded(handler)

    assert client.complete(REQ).text == "ok"
    assert sleeps == [12.0, 12.0]
    assert len(calls) == 3 and calls[0].content == calls[2].content


def test_without_retry_after_the_wait_doubles_up_to_the_cap():
    handler, _ = limited_then_ok(4)
    client, sleeps = guarded(handler, backoff=10, max_backoff=30)

    client.complete(REQ)

    assert sleeps == [10, 20, 30, 30]


def test_a_limit_that_outlasts_the_maximum_wait_stops_cleanly():
    handler, _ = limited_then_ok(100)
    client, sleeps = guarded(handler, backoff=10, max_backoff=10, max_wait=35)

    with pytest.raises(ProviderExhausted, match="still limited"):
        client.complete(REQ)
    assert sleeps == [10, 10, 10]


def test_a_retry_after_beyond_the_maximum_wait_stops_at_once():
    handler, _ = limited_then_ok(1, headers={"Retry-After": "20000"})
    client, sleeps = guarded(handler, max_wait=3600)

    with pytest.raises(ProviderExhausted):
        client.complete(REQ)
    assert sleeps == []


def test_no_credits_left_stops_at_once():
    handler, _ = limited_then_ok(1, status=402)
    client, sleeps = guarded(handler)

    with pytest.raises(ProviderExhausted, match="402"):
        client.complete(REQ)
    assert sleeps == []


@pytest.mark.parametrize("status, body, error", [
    (403, {"error": {"message": "Usage limit reached for this 5-hour window"}}, RateLimited),
    (200, {"error": {"code": 429, "message": "upstream busy"}}, RateLimited),
    (200, {"error": {"code": 502, "message": "upstream failed"}}, ProviderError),
    (500, {"error": "boom"}, ProviderError),
])
def test_limit_errors_are_told_apart_from_other_errors(status, body, error):
    client = OpenAICompatibleClient("https://p.example/v1", "k", http=httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(status, json=body))))

    with pytest.raises(error):
        client.complete(REQ)


def test_a_rate_limited_run_waits_and_finishes_with_every_row_ok(tmp_path):
    state = {"n": 0}

    def handler(request):
        state["n"] += 1
        if state["n"] in (2, 3):
            return httpx.Response(429, headers={"Retry-After": "5"})
        return crew_answer(request)

    sleeps: list[float] = []
    run = execute(load_design(SKELETON), routed_client(handler, sleep=sleeps.append), out_dir=tmp_path,
                  cache_dir=tmp_path / "cache")

    assert sleeps == [5.0, 5.0]
    assert all(r.status == "ok" for r in run.rows)
    assert run.totals["calls"] == 5 and run.totals["attempts"] == 5  # waits are not retries


# --- clean stop and resume ----------------------------------------------------------------------------------


class StopsAfter(FakeLLMClient):
    """Answers `limit` calls, then raises `ProviderExhausted` as a provider out of budget would."""

    def __init__(self, responder, limit: int):
        super().__init__(responder)
        self.limit = limit

    def complete(self, request):
        if len(self.calls) >= self.limit:
            raise ProviderExhausted("usage limit for the 5 h window")
        return super().complete(request)


def test_a_stopped_run_records_progress_and_a_rerun_resumes_without_repeating_calls(tmp_path):
    design = load_design(SKELETON)
    a, b = (i.names[0] for i in design.identities)
    responder = scores_by_name({a: 70, b: 64})
    out, cache = tmp_path / "out", tmp_path / "cache"
    first = StopsAfter(responder, limit=3)

    with pytest.raises(ProviderExhausted):
        execute(design, first, out_dir=out, cache_dir=cache)

    progress = json.loads((out / "progress.json").read_text())
    assert progress["status"] == "stopped" and "usage limit" in progress["reason"]
    assert sum(progress["calls_this_run"].values()) == 4  # 3 answered, the 4th refused
    assert not (out / "results.csv").exists()  # no partial results table

    second = FakeLLMClient(responder)
    run = execute(design, second, out_dir=out, cache_dir=cache)

    assert len(second.calls) == 2  # 5 calls in all: the 3 completed ones come from the cache
    assert {c.cache_key() for c in second.calls}.isdisjoint({c.cache_key() for c in first.calls})
    assert all(r.status == "ok" for r in run.rows)
    assert not (out / "progress.json").exists()


def test_the_cli_reports_a_stop_and_exits_with_code_3(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("hcai.cli.make_client", lambda design, models: StopsAfter(lambda r: {}, limit=0))

    code = main(["run", str(SKELETON), "--out", str(tmp_path / "out"), "--cache", str(tmp_path / "cache")])

    err = capsys.readouterr().err
    assert code == 3
    assert "rerun the same command to resume" in err and str(tmp_path / "out" / "progress.json") in err


# --- unknown cost and secrets in a whole run ----------------------------------------------------------------


def test_a_run_on_a_subscription_logs_tokens_and_marks_the_cost_unknown(tmp_path):
    design = load_design(config(tmp_path))
    run_command(config(tmp_path), out_dir=tmp_path / "out", cache_dir=tmp_path / "cache",
                client=routed_client(crew_answer, design=design))

    rows = read_results(tmp_path / "out" / "results.csv")
    called = [r for r in rows if r.attempts]
    assert called and all(r.cost is None and r.prompt_tokens > 0 for r in called)
    summary = json.loads((tmp_path / "out" / "run_summary.json").read_text())
    model = summary["by_model"]["openai/gpt-4o-mini"]
    assert (model["provider"], model["billing"]) == ("opencode-go", "subscription")
    assert model["unpriced_calls"] == model["calls"] == 5 and model["cost_spent"] == 0


def test_the_cli_summary_says_the_cost_is_unknown_and_the_pilot_estimates_tokens(tmp_path):
    design = load_design(config(tmp_path))
    report = run_command(config(tmp_path), out_dir=tmp_path / "out", cache_dir=tmp_path / "cache",
                         client=routed_client(crew_answer, design=design), pilot=True)

    assert "cost unknown for 5 calls this run" in report and "subscription" in report
    assert "estimated full-run cost unknown" in report and "completion tokens" in report
    pilot = json.loads((tmp_path / "out" / "run_summary.json").read_text())["pilot"]
    assert pilot["estimated_full_cost"] is None and pilot["estimated_full_tokens"]["prompt"] > 0


def test_a_metered_provider_still_logs_its_price(tmp_path):
    def priced(request):
        r = crew_answer(request)
        data = r.json()
        data["usage"]["cost"] = 0.002
        return httpx.Response(200, json=data)

    run = execute(load_design(SKELETON), routed_client(priced), out_dir=tmp_path, cache_dir=tmp_path / "cache")

    assert run.totals["cost_spent"] == pytest.approx(0.01) and run.totals["unpriced_calls"] == 0
    assert run.totals["by_model"]["openai/gpt-4o-mini"]["billing"] == "metered"


def test_the_key_never_reaches_results_cache_or_output(tmp_path, capsys):
    state = {"n": 0}

    def handler(request):
        assert request.headers["Authorization"] == f"Bearer {SECRET}"
        state["n"] += 1
        if state["n"] <= 3:  # a provider error that echoes the key back, until the Screener gives up
            return httpx.Response(500, text=f"invalid request from key {SECRET}")
        if state["n"] == 4:
            return httpx.Response(429, text=f"rate limit for {SECRET}")
        return crew_answer(request)

    path = config(tmp_path)
    path.write_text(path.read_text().replace("retry_backoff = 2.0", "retry_backoff = 0"))
    design = load_design(path)
    run_command(path, out_dir=tmp_path / "out", cache_dir=tmp_path / "cache", client=routed_client(handler, design=design))

    written = [p for p in tmp_path.rglob("*") if p.is_file() and p.name != "design.toml"]
    assert any(p.suffix == ".json" for p in (tmp_path / "cache").iterdir())
    assert "***" in (tmp_path / "out" / "results.csv").read_text()  # the echoed error, redacted
    for p in written:  # CrewAI's own local store (a SQLite file under the test's data dir) included
        assert SECRET.encode() not in p.read_bytes(), p
    captured = capsys.readouterr()
    assert SECRET not in captured.out + captured.err


# --- review fixes -------------------------------------------------------------------------------------------


def test_a_transport_error_that_quotes_the_header_is_redacted():
    def handler(request):
        raise httpx.LocalProtocolError(f"Illegal header value b'Bearer {SECRET}'")

    client = OpenAICompatibleClient("https://p.example/v1", SECRET, http=httpx.Client(transport=httpx.MockTransport(handler)))

    with pytest.raises(ProviderError) as info:
        client.complete(REQ)
    assert SECRET not in str(info.value) and "***" in str(info.value)


def test_the_key_is_stripped_of_stray_whitespace():
    seen = []

    def handler(request):
        seen.append(request.headers["Authorization"])
        return chat_answer("ok")

    OpenAICompatibleClient("https://p.example/v1", f" {SECRET}\n",
                           http=httpx.Client(transport=httpx.MockTransport(handler))).complete(REQ)

    assert seen == [f"Bearer {SECRET}"]


def test_an_echoed_key_across_the_cut_off_is_fully_redacted():
    body = "x" * 480 + f" key {SECRET} tail"
    client = OpenAICompatibleClient("https://p.example/v1", SECRET, http=httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(500, text=body))))

    with pytest.raises(ProviderError) as info:
        client.complete(REQ)
    assert SECRET[:8] not in str(info.value)


def test_retry_after_zero_still_waits_so_the_maximum_wait_stops_the_run():
    handler, _ = limited_then_ok(10_000, headers={"Retry-After": "0"})
    client, sleeps = guarded(handler, max_wait=20, min_wait=5)

    with pytest.raises(ProviderExhausted):
        client.complete(REQ)
    assert sleeps == [5, 5, 5, 5]


def test_a_refused_key_stops_the_run():
    client = OpenAICompatibleClient("https://p.example/v1", "k", http=httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(401, json={"error": "invalid api key"}))))

    with pytest.raises(ProviderExhausted, match="401"):
        client.complete(REQ)


# --- OpenCode Go session header and User-Agent ---------------------------------------------------------------


def test_opencode_go_sends_one_stable_session_id_and_the_hcai_user_agent_and_openrouter_no_session(tmp_path):
    seen: list[httpx.Request] = []

    def handler(request):
        seen.append(request)
        return chat_answer("ok")

    design = load_design(config(tmp_path, '\n[model_providers]\n"b" = "openrouter"\n', models_line='models = ["a", "b"]'))
    client = provider_client(design.provider_for, design.models,
                             environ={"OPENCODE_API_KEY": "sk-oc", "OPENROUTER_API_KEY": "sk-or"},
                             http=httpx.Client(transport=httpx.MockTransport(handler)))

    client.complete(replace(REQ, model="a"))
    client.complete(replace(REQ, model="a", messages=[{"role": "user", "content": "again"}]))
    client.complete(replace(REQ, model="b"))

    opencode, again, openrouter = seen
    session = opencode.headers["x-opencode-session"]
    assert session == SESSION_ID and str(uuid.UUID(session)) == session
    assert again.headers["x-opencode-session"] == session  # stable across the calls of a run
    assert opencode.headers["User-Agent"] == USER_AGENT and USER_AGENT.startswith("hcai/")
    assert "x-opencode-session" not in openrouter.headers
    assert openrouter.headers["User-Agent"] == USER_AGENT
    assert session not in opencode.content.decode()  # a header, never part of the body


def test_the_session_header_stays_out_of_the_cache_key_and_the_cache(tmp_path):
    def handler(request):
        return chat_answer("ok")

    key_before = REQ.cache_key()  # the request alone: the cached file's name and content are the real check
    plain = OpenAICompatibleClient("https://p.example/v1", "k", http=httpx.Client(transport=httpx.MockTransport(handler)))
    with_session = OpenAICompatibleClient("https://p.example/v1", "k", headers={"x-opencode-session": SESSION_ID},
                                          http=httpx.Client(transport=httpx.MockTransport(handler)))
    cached = CachedClient(with_session, tmp_path / "cache")

    cached.complete(REQ)

    assert REQ.cache_key() == key_before
    assert [p.name for p in (tmp_path / "cache").iterdir()] == [f"{key_before}.json"]
    assert SESSION_ID not in (tmp_path / "cache" / f"{key_before}.json").read_text()
    assert plain.complete(REQ).text == with_session.complete(REQ).text  # same request, same answer


def test_a_missing_session_answer_is_a_provider_error_not_a_silent_miss():
    body = {"error": {"type": "MissingSessionID", "message": "Request is missing x-opencode-session header"}}
    client = OpenAICompatibleClient("https://p.example/v1", "k", http=httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(400, json=body))))

    with pytest.raises(ProviderError, match="MissingSessionID"):
        client.complete(REQ)


def test_the_session_id_is_masked_in_error_text():
    client = OpenAICompatibleClient("https://p.example/v1", "k", headers={"x-opencode-session": SESSION_ID},
                                    http=httpx.Client(transport=httpx.MockTransport(
                                        lambda r: httpx.Response(400, text=f"bad session {SESSION_ID}"))))

    with pytest.raises(ProviderError) as info:
        client.complete(REQ)
    assert SESSION_ID not in str(info.value)


def test_config_headers_are_sent_but_cannot_override_the_key_or_user_agent(tmp_path):
    design = load_design(config(tmp_path, '\n[providers.opencode-go.headers]\nx-extra = "1"\n'))
    assert design.providers["opencode-go"].http_headers() == {"x-extra": "1", "x-opencode-session": SESSION_ID}

    with pytest.raises(ValueError, match="Authorization"):
        load_design(config(tmp_path, '\n[providers.opencode-go.headers]\nAuthorization = "Bearer x"\n'))


def test_an_echoed_key_is_masked_even_when_a_header_value_is_part_of_it():
    key = "sk-fake1abc-header-value-xyz"
    client = OpenAICompatibleClient("https://p.example/v1", key, headers={"x-extra": "1", "x-long": "header-value"},
                                    http=httpx.Client(transport=httpx.MockTransport(
                                        lambda r: httpx.Response(400, text=f"bad key {key} 1"))))

    with pytest.raises(ProviderError) as info:
        client.complete(REQ)
    assert "fake" not in str(info.value) and "xyz" not in str(info.value) and "***" in str(info.value)
    assert str(info.value).endswith(" 1")  # a value too short to be a secret is not masked everywhere


def test_the_client_never_lets_extra_headers_replace_the_key_or_user_agent():
    seen = []

    def handler(request):
        seen.append(request)
        return chat_answer("ok")

    OpenAICompatibleClient("https://p.example/v1", "k", headers={"authorization": "Bearer x", "user-agent": "y"},
                           http=httpx.Client(transport=httpx.MockTransport(handler))).complete(REQ)

    assert seen[0].headers.get_list("authorization") == ["Bearer k"]
    assert seen[0].headers.get_list("user-agent") == [USER_AGENT]


@pytest.mark.parametrize("extra", [
    '\n[providers.opencode-go]\nsession_header = 1\n',
    '\n[providers.opencode-go]\nsession_header = ""\n',
    '\n[providers.opencode-go]\nheaders = "x"\n',
    '\n[providers.opencode-go.headers]\nx-extra = 1\n',
    '\n[providers.opencode-go.headers]\n"bad header" = "1"\n',
    '\n[providers.opencode-go.headers]\nX-OpenCode-Session = "a"\n',
])
def test_bad_provider_headers_fail_at_load(tmp_path, extra):
    with pytest.raises(ValueError):
        load_design(config(tmp_path, extra))


def test_header_values_stay_out_of_the_provider_repr():
    provider = replace(BUILTIN_PROVIDERS["opencode-go"], headers=(("x-api-token", SECRET),))
    assert SECRET not in repr(provider)


def test_a_missing_session_run_warns_once_records_missing_rows_and_counts_the_refusals(tmp_path, capsys):
    body = {"error": {"type": "MissingSessionID", "message": "Request is missing x-opencode-session header"}}
    state = {"n": 0}

    def handler(request):
        state["n"] += 1
        return httpx.Response(400, json=body)

    design = load_design(config(tmp_path))  # the model on opencode-go
    out = tmp_path / "out"
    run = execute(design, routed_client(handler, design=design), out_dir=out, cache_dir=tmp_path / "cache")

    err = capsys.readouterr().err
    assert err.count("MissingSessionID") == 1  # one warning per run, however many calls fail
    assert "opencode-go" in err and "session header is missing or rejected" in err
    assert state["n"] > 1 and all(r.status == "missing" for r in run.rows)  # retried, then missing: no stop
    summary = json.loads((out / "run_summary.json").read_text())
    assert summary["missing_session_errors"] == state["n"] == summary["calls"]
    assert not list((tmp_path / "cache").glob("*.json"))  # refused calls are never cached
