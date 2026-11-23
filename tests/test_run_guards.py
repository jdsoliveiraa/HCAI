"""Run guards for the full run (#12): billing stop, consecutive failures, transient backoff, limit words, empty
answers, progress lines, early CSVs, timeouts, missing share, corrupt cache files. Fake clients and
`httpx.MockTransport` only; nothing leaves the machine.
"""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest

from conftest import scores_by_name
from hcai.cli import _fake_response, main, run_command
from hcai.config import RunSettings, load_design
from hcai.llm import (
    BillingGuard, BillingStop, CachedClient, FailureGuard, FakeLLMClient, LLMRequest, LLMResponse,
    MeteredClient, OpenAICompatibleClient, ProviderError, ProviderExhausted, RateLimited, TooManyFailures, Usage,
    is_limit_error,
)
from hcai.pipeline import ClientLLM, Retry, screen_and_evaluate
from hcai.runner import execute, read_results
from test_go_allowance import billed
from test_providers import SKELETON, config, crew_answer, routed_client

REQ = LLMRequest(model="m", messages=[{"role": "user", "content": "hi"}])
CORPUS = SKELETON.parent.parent / "data" / "corpus"


def with_run(path: Path, *lines: str) -> Path:
    """The config at `path` with extra `[run]` settings (placed after `retry_backoff`)."""
    text = path.read_text()
    assert "retry_backoff = 2.0" in text
    path.write_text(text.replace("retry_backoff = 2.0", "retry_backoff = 2.0\n" + "\n".join(lines), 1))
    return path


def http(handler, **kw) -> OpenAICompatibleClient:
    return OpenAICompatibleClient("https://p.example/v1", "k", http=httpx.Client(transport=httpx.MockTransport(handler)),
                                  **kw)


def all_conditions(tmp_path: Path) -> Path:
    """The skeleton with C0, C1 and C2 (so every agent: Screener, Evaluator, Ranker, Blinder, judge)."""
    path = tmp_path / "all.toml"
    path.write_text(
        SKELETON.read_text()
        .replace('conditions = ["C0"]', 'conditions = ["C0", "C1", "C2"]')
        .replace('corpus_dir = "../data/corpus"', f'corpus_dir = "{CORPUS}"')
        + '\n[oversight]\njudge_model = "judge-model"\ncutoff = 50\nmargin = 5\n')
    return path


AGENT_KEYS = {"screener": "passed", "evaluator": "score", "ranker": "ranking", "blinder": "blinded_cv",
              "judge": "verdicts"}


def agent_of(request: LLMRequest) -> str:
    props = (request.response_schema or {}).get("properties", {})
    if "criteria" in props:
        return "evaluator"
    return next(a for a, k in AGENT_KEYS.items() if k in props)


class Billed(FakeLLMClient):
    """The fake crew, with every answer of `agent` billed to `source` (all others to "go")."""

    def __init__(self, agent: str | None, source: str = "balance"):
        super().__init__(_fake_response)
        self.agent, self.source = agent, source

    def complete(self, request):
        out = super().complete(request)
        source = self.source if agent_of(request) == self.agent else "go"
        return replace(out, usage=Usage(1, 1, cost=0.0001, billing_source=source, allowance=0.0001 * (source == "go"),
                                        list_cost=0.0001))


class Failing(FakeLLMClient):
    """The fake crew, with every call of `agent` failing with a provider error."""

    def __init__(self, agent: str, error: ProviderError | None = None):
        super().__init__(_fake_response)
        self.agent = agent
        self.error = error or ProviderError("HTTP 400: bad request")

    def complete(self, request):
        if agent_of(request) == self.agent:
            self.calls.append(request)
            raise self.error
        return super().complete(request)


# --- B1: billing guard --------------------------------------------------------------------------------------


def test_the_first_call_billed_outside_allowed_billing_is_cached_then_the_run_stops(tmp_path):
    design = load_design(config(tmp_path))
    out, cache = tmp_path / "out", tmp_path / "cache"

    with pytest.raises(BillingStop, match="'balance'"):
        execute(design, routed_client(billed(lambda n: "go" if n < 2 else "balance")), out_dir=out, cache_dir=cache)

    progress = json.loads((out / "progress.json").read_text())
    assert progress["stop"] == "billing" and progress["billing_source"] == "balance"
    assert progress["billing_sources_this_run"] == {"go": 2, "balance": 1}
    assert len(list(cache.glob("*.json"))) == 3  # the paid answer is cached with the two go ones
    assert not (out / "results.csv").exists()

    rerun = execute(design, routed_client(billed(lambda n: "go")), out_dir=out, cache_dir=cache)
    assert rerun.totals["calls"] == 2 and rerun.totals["cache_hits"] == 3  # the paid call is not paid again
    assert rerun.totals["not_allowed_billing_calls"] == 0 and rerun.totals["allowed_billing"] == ["go"]


def test_the_cli_exits_3_naming_the_billing_source(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("hcai.cli.make_client", lambda design, models: Billed("screener"))

    code = main(["run", str(config(tmp_path)), "--out", str(tmp_path / "out"), "--cache", str(tmp_path / "cache")])

    err = capsys.readouterr().err
    assert code == 3 and "'balance'" in err and "allowed_billing" in err and "progress.json" in err


def test_an_empty_allowed_billing_checks_nothing_and_counts_what_was_paid_elsewhere(tmp_path):
    design = load_design(with_run(config(tmp_path), "allowed_billing = []"))

    run = execute(design, routed_client(billed(lambda n: "balance")), out_dir=tmp_path / "o", cache_dir=tmp_path / "c")

    assert run.totals["not_go_calls"] == 5 and run.totals["not_allowed_billing_calls"] == 0
    assert run.totals["allowed_billing"] == []


def test_an_allowed_source_besides_go_is_not_a_stop(tmp_path):
    design = load_design(with_run(config(tmp_path), 'allowed_billing = ["go", "balance"]'))

    run = execute(design, routed_client(billed(lambda n: "balance")), out_dir=tmp_path / "o", cache_dir=tmp_path / "c")

    assert run.totals["billing_sources"] == {"balance": 5} and all(r.status == "ok" for r in run.rows)


def test_answers_without_a_billing_source_are_never_checked(tmp_path):
    run = execute(load_design(config(tmp_path)), routed_client(crew_answer), out_dir=tmp_path / "o",
                  cache_dir=tmp_path / "c")

    assert all(r.status == "ok" for r in run.rows) and run.totals["billing_sources"] == {}


def test_a_paid_answer_replayed_from_the_cache_does_not_stop_the_run(tmp_path):
    cached = CachedClient(FakeLLMClient(lambda r: "ok"), tmp_path)
    paid = LLMResponse("ok", usage=Usage(billing_source="balance"))
    (tmp_path / f"{REQ.cache_key()}.json").write_text(json.dumps(
        {"text": paid.text, "parsed": None, "usage": {"billing_source": "balance", "cost": 0.1}}))

    assert BillingGuard(cached, ["go"]).complete(REQ).cached


def test_allowed_billing_must_be_a_list(tmp_path):
    with pytest.raises(ValueError, match="allowed_billing"):
        load_design(with_run(config(tmp_path), 'allowed_billing = "go"'))
    assert load_design(config(tmp_path)).run.allowed_billing == ("go",)


@pytest.mark.parametrize("agent", list(AGENT_KEYS))
def test_a_billing_stop_reaches_a_clean_stop_through_every_agent(agent, tmp_path):
    client = Billed(agent)

    with pytest.raises(BillingStop):
        execute(load_design(all_conditions(tmp_path)), client, out_dir=tmp_path / "o", cache_dir=tmp_path / "c")

    assert json.loads((tmp_path / "o" / "progress.json").read_text())["stop"] == "billing"
    assert agent_of(client.calls[-1]) == agent  # stopped at that agent's first paid call
    assert (tmp_path / "c" / f"{client.calls[-1].cache_key()}.json").exists()


# --- S1: consecutive failures and transient backoff ---------------------------------------------------------


def req(i: int, model: str = "m") -> LLMRequest:
    return LLMRequest(model=model, messages=[{"role": "user", "content": f"call {i}"}])


def test_consecutive_failed_calls_on_one_model_stop_after_the_limit_and_an_answer_resets_them():
    results = iter([*([ProviderError("down")] * 9), "ok", *([ProviderError("down 2")] * 3)])

    def respond(request):
        r = next(results)
        if isinstance(r, Exception):
            raise r
        return r

    guard = FailureGuard(FakeLLMClient(respond), limit=10)
    for i in range(9):
        with pytest.raises(ProviderError):
            guard.complete(req(i))
    assert guard.complete(req(9)).text == "ok" and guard.in_a_row["m"] == 0

    guard.limit = 3
    for i in range(2):
        with pytest.raises(ProviderError):
            guard.complete(req(10 + i))
    with pytest.raises(TooManyFailures, match=r"3 failed calls in a row.*down 2"):
        guard.complete(req(12))


def test_a_call_repeated_as_is_counts_once():
    guard = FailureGuard(FakeLLMClient(lambda r: (_ for _ in ()).throw(ProviderError("timeout", timeout=True))), limit=2)
    for _ in range(5):  # one call that keeps failing (its retries are the identical request)
        with pytest.raises(ProviderError):
            guard.complete(REQ)
    assert guard.in_a_row["m"] == 1
    with pytest.raises(TooManyFailures):
        guard.complete(req(1))


def test_failures_are_counted_per_model():
    guard = FailureGuard(FakeLLMClient(lambda r: (_ for _ in ()).throw(ProviderError("x"))), limit=2)
    for r in (req(0), req(0, "other")):
        with pytest.raises(ProviderError):
            guard.complete(r)
    with pytest.raises(TooManyFailures):
        guard.complete(req(1))


def test_in_a_run_failures_carry_across_calls_and_another_agents_answer_resets_them(tmp_path):
    c0 = load_design(config(tmp_path))
    c0 = replace(c0, run=replace(c0.run, retry_backoff=0.0, max_consecutive_failures=2))

    # the Evaluator fails on every CV, but the Screener answers in between: never 2 in a row, no stop (only answers
    # from the provider reset the count; a cache hit makes no call)
    run = execute(c0, Failing("evaluator"), out_dir=tmp_path / "o", cache_dir=tmp_path / "c", log=lambda m: None)
    assert run.totals["missing"]["missing"] > 0

    design = load_design(all_conditions(tmp_path))
    design = replace(design, run=replace(design.run, retry_backoff=0.0, max_consecutive_failures=2))

    # the Blinder runs on every CV of a group back to back: its 2nd failed call stops the run
    with pytest.raises(TooManyFailures, match="2 failed calls in a row"):
        execute(design, Failing("blinder"), out_dir=tmp_path / "o2", cache_dir=tmp_path / "c2", log=lambda m: None)


@pytest.mark.parametrize("agent", list(AGENT_KEYS))
def test_too_many_failures_reach_a_clean_stop_through_every_agent(agent, tmp_path):
    design = load_design(all_conditions(tmp_path))
    design = replace(design, run=replace(design.run, retry_backoff=0.0, max_consecutive_failures=1))

    with pytest.raises(TooManyFailures):
        execute(design, Failing(agent), out_dir=tmp_path / "o", cache_dir=tmp_path / "c")

    progress = json.loads((tmp_path / "o" / "progress.json").read_text())
    assert progress["stop"] == "failures" and progress["failures_in_a_row"] == 1
    assert "bad request" in progress["reason"]


def test_a_run_stopped_by_failures_exits_3_and_a_rerun_resumes(tmp_path, monkeypatch, capsys):
    path = with_run(config(tmp_path), "max_consecutive_failures = 1")
    path.write_text(path.read_text().replace("retry_backoff = 2.0", "retry_backoff = 0.0"))
    monkeypatch.setattr("hcai.cli.make_client", lambda design, models: Failing("evaluator"))
    args = ["run", str(path), "--out", str(tmp_path / "out"), "--cache", str(tmp_path / "cache")]

    assert main(args) == 3
    assert "failed calls in a row" in capsys.readouterr().err

    monkeypatch.setattr("hcai.cli.make_client", lambda design, models: FakeLLMClient(_fake_response))
    assert main(args) == 0
    assert all(r.status == "ok" for r in read_results(tmp_path / "out" / "results.csv"))


def llm(client, retry: Retry) -> ClientLLM:
    return ClientLLM(client, model="m", temperature=0.0, seed=None, schema={"type": "object"},
                     validate=lambda p: "" if p == {"ok": True} else "bad", retry=retry)


def fails_then_ok(errors: list[Exception]):
    left = list(errors)

    def respond(request):
        if left:
            raise left.pop(0)
        return {"ok": True}
    return respond


def test_transient_failures_wait_the_longer_schedule_and_get_as_many_repeats_as_it_has():
    sleeps: list[float] = []
    retry = Retry(retries=2, backoff=2.0, sleep=sleeps.append, transient_backoff=(30, 120, 480))
    client = FakeLLMClient(fails_then_ok([ProviderError("503", transient=True)] * 3))

    a = llm(client, retry)
    a.call("hi")

    assert sleeps == [30, 120, 480] and a.output().parsed == {"ok": True} and a.output().attempts == 4


def test_a_transient_failure_that_outlasts_the_schedule_is_raised():
    sleeps: list[float] = []
    retry = Retry(retries=2, backoff=2.0, sleep=sleeps.append, transient_backoff=(30, 120, 480))

    with pytest.raises(ProviderError):
        llm(FakeLLMClient(fails_then_ok([ProviderError("t", timeout=True)] * 9)), retry).call("hi")
    assert sleeps == [30, 120, 480]


def test_other_failures_and_malformed_answers_keep_their_backoff():
    sleeps: list[float] = []
    retry = Retry(retries=2, backoff=2.0, sleep=sleeps.append, transient_backoff=(30, 120, 480))

    with pytest.raises(ProviderError):
        llm(FakeLLMClient(fails_then_ok([ProviderError("400")] * 9)), retry).call("hi")
    assert sleeps == [2.0, 4.0]

    sleeps.clear()
    bad = llm(FakeLLMClient(lambda r: {"ok": False}), retry)
    bad.call("hi")
    assert sleeps == [] and bad.output().attempts == 3  # parse retries: no wait, as before


@pytest.mark.parametrize("make, transient, timeout", [
    (lambda r: httpx.Response(503, json={"error": {"message": "upstream unavailable"}}), True, False),
    (lambda r: httpx.Response(500, text="Internal Server Error"), True, False),
    (lambda r: httpx.Response(200, json={"error": {"code": 502, "message": "upstream failed"}}), True, False),
    (lambda r: httpx.Response(400, json={"error": {"message": "bad request"}}), False, False),
    (lambda r: (_ for _ in ()).throw(httpx.ReadTimeout("timed out")), True, True),
    (lambda r: (_ for _ in ()).throw(httpx.ConnectError("connection reset")), True, False),
    (lambda r: (_ for _ in ()).throw(httpx.RemoteProtocolError("peer closed")), True, False),
    (lambda r: (_ for _ in ()).throw(httpx.LocalProtocolError("bad header")), False, False),
    (lambda r: (_ for _ in ()).throw(httpx.UnsupportedProtocol("ftp://")), False, False),
    (lambda r: httpx.Response(502, text="<html>502 Bad Gateway: upstream connect error, connection limit</html>"),
     True, False),
])
def test_server_and_transport_errors_are_transient(make, transient, timeout):
    with pytest.raises(ProviderError) as e:
        http(make).complete(REQ)
    assert (e.value.transient, e.value.timeout) == (transient, timeout)


def test_a_short_outage_is_ridden_out_in_a_run(tmp_path):
    n = [0]

    def handler(request):
        n[0] += 1
        return httpx.Response(503, text="down") if n[0] in (2, 3, 4) else crew_answer(request)

    design = load_design(with_run(config(tmp_path), "transient_backoff = [0, 0, 0]"))
    run = execute(design, routed_client(handler), out_dir=tmp_path / "o", cache_dir=tmp_path / "c")

    assert all(r.status == "ok" for r in run.rows) and run.totals["failed_calls"] == 3


@pytest.mark.parametrize("line", ["max_consecutive_failures = -1", "transient_backoff = [-1]",
                                  'transient_backoff = "30"', "missing_target = 2"])
def test_invalid_guard_settings_are_refused(tmp_path, line):
    with pytest.raises(ValueError, match=r"\[run\]"):
        load_design(with_run(config(tmp_path), line))


def test_the_defaults():
    r = RunSettings()
    assert (r.allowed_billing, r.max_consecutive_failures, r.transient_backoff, r.missing_target) == (
        ("go",), 10, (30.0, 120.0, 480.0), 0.02)


# --- S2: limit words ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("status, body", [
    (403, {"error": {"message": "Subscription usage limit reached. Resets in 2h"}}),
    (400, {"error": {"message": "Monthly allowance exceeded"}}),
    (400, {"error": {"message": "Insufficient credits"}}),
    (400, {"error": {"message": "Your credit balance is too low"}}),
    (400, {"error": {"type": "SubscriptionUsageLimitError", "message": "Please try again later"}}),
    (400, {"name": "SubscriptionUsageLimitError", "message": "try later"}),
    (400, {"error": {"code": "insufficient_quota", "message": "see your plan"}}),
    (500, {"detail": "Rate limit exceeded"}),
    (403, "Quota exceeded for this key"),
    (403, {"error": {"message": "You've reached your usage limit. It'll reset in 2 hours."}}),
    (403, {"message": "Forbidden", "detail": "Weekly usage limit reached"}),
    (400, {"error": {"code": 429, "message": "Provider returned error", "metadata": {"raw": "upstream busy"}}}),
    (400, {"error": {"code": 400, "message": "Provider returned error", "metadata": {"raw": "Rate limit exceeded"}}}),
    (400, {"error": {"code": "rate_limit_exceeded", "message": "Rate limit for context caching"}}),
    (400, {"error": {"message": "Monthly limit exceeded, resets on the 1st"}}),
])
def test_limit_answers_are_waited_out(status, body):
    make = (lambda r: httpx.Response(status, json=body)) if not isinstance(body, str) else (
        lambda r: httpx.Response(status, text=body))
    with pytest.raises(RateLimited):
        http(make).complete(REQ)


@pytest.mark.parametrize("body", [
    {"error": {"message": "Invalid value for 'quota': expected a number", "type": "invalid_request_error"}},
    {"error": {"message": 'Unknown parameter "credit_limit"', "type": "invalid_request_error"}},
    {"error": {"message": "Unrecognized field credit_limit in response_format"}},
    {"error": {"message": "This request exceeds the model's context limit of 128000 tokens"}},
    {"error": {"message": "MissingSessionID"}},
    {"error": {"message": "max_tokens exceeded the model maximum of 32768"}},
    {"error": {"message": "The request exceeded the maximum allowed size"}},
    {"error": {"message": "Input length 40000 exceeds the maximum limit of 32768 tokens"}},
    {"error": {"message": "Your request exceeded model token limit: 262144"}},
    {"error": {"message": "Model qwen3.8-flash is not available on your subscription plan"}},
    {"error": {"message": "Insufficient permissions for this model"}},
    {"error": {"message": "Your credit card was declined"}},
    {"error": {"code": "limit_out_of_range", "message": "bad value"}},
])
def test_validation_errors_that_echo_a_limit_word_are_not_limits(body):
    with pytest.raises(ProviderError) as e:
        http(lambda r: httpx.Response(400, json=body)).complete(REQ)
    assert not isinstance(e.value, RateLimited)


def test_a_weak_word_counts_only_in_the_message_and_a_strong_phrase_anywhere():
    weak = {"error": {"message": "bad request", "param": "limit exceeded"}}
    strong = {"error": {"message": "bad request", "metadata": {"raw": "usage limit reached"}}}
    assert not is_limit_error(weak, json.dumps(weak)) and is_limit_error(strong, json.dumps(strong))
    assert is_limit_error(None, "usage limit reached")


def test_401_still_stops_and_402_still_stops():
    for status in (401, 402):
        with pytest.raises(ProviderExhausted, match=str(status)):
            http(lambda r: httpx.Response(status, json={"error": {"message": "subscription limit"}})).complete(REQ)


# --- S3: empty answers --------------------------------------------------------------------------------------


@pytest.mark.parametrize("finish", ["length", None, "content_filter"])
def test_an_empty_answer_not_finished_is_an_error_and_not_cached(tmp_path, finish):
    choice = {"message": {"content": None}, **({"finish_reason": finish} if finish else {})}
    client = http(lambda r: httpx.Response(200, json={"choices": [choice]}))

    with pytest.raises(ProviderError, match="empty answer"):
        CachedClient(client, tmp_path).complete(REQ)
    assert list(tmp_path.iterdir()) == []


def test_an_empty_answer_billed_outside_go_still_stops_the_run_and_is_counted(tmp_path):
    from test_go_allowance import go_usage
    body = {"choices": [{"message": {"content": ""}, "finish_reason": "length"}],
            "usage": go_usage(allowance=0.0, source="balance", list_cost=0.002)}
    meter = MeteredClient(http(lambda r: httpx.Response(200, json=body), subscription=True))

    with pytest.raises(BillingStop, match="'balance'"):
        BillingGuard(CachedClient(meter, tmp_path), ["go"]).complete(REQ)
    assert meter.billing == {"m": {"balance": 1}} and meter.balance["m"] == pytest.approx(0.002)
    assert list(tmp_path.iterdir()) == []  # still not cached: a rerun asks again


def test_an_empty_answer_the_model_finished_is_an_answer_and_cached(tmp_path):
    client = http(lambda r: httpx.Response(200, json={"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}))

    assert CachedClient(client, tmp_path).complete(REQ).text == ""
    assert len(list(tmp_path.iterdir())) == 1


# --- S4, S5, S6, S7: progress, early CSVs, timeouts, missing share -----------------------------------------


def test_each_block_prints_a_progress_line(tmp_path):
    lines: list[str] = []
    design = load_design(all_conditions(tmp_path))

    execute(design, Billed(None), out_dir=tmp_path / "o", cache_dir=tmp_path / "c", log=lines.append)

    progress = [x for x in lines if x.startswith("[block ")]
    assert len(progress) == 3 and progress[0].startswith("[block 1/3] ") and progress[-1].startswith("[block 3/3] ")
    assert all(k in progress[-1] for k in ("calls", "cache hits", "missing so far", "allowance $", "balance $",
                                           "billing go x", "elapsed 0:00:"))


def test_the_cli_prints_progress_to_stderr(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("hcai.cli.make_client", lambda design, models: FakeLLMClient(_fake_response))
    assert main(["run", str(config(tmp_path)), "--out", str(tmp_path / "o"), "--cache", str(tmp_path / "c")]) == 0
    captured = capsys.readouterr()
    assert "[block 1/1]" in captured.err and "[block" not in captured.out


def test_the_tables_of_paid_answers_are_on_disk_before_any_summary_runs(tmp_path, monkeypatch):
    path = config(tmp_path)
    path.write_text(path.read_text().replace("repeat_fraction = 0.0", "repeat_fraction = 1.0"))

    def boom(*a, **k):
        raise RuntimeError("summary crashed")

    monkeypatch.setattr("hcai.runner.run_totals", boom)
    with pytest.raises(RuntimeError, match="summary crashed"):
        execute(load_design(path), FakeLLMClient(_fake_response), out_dir=tmp_path / "o", cache_dir=tmp_path / "c")

    assert read_results(tmp_path / "o" / "results.csv") and read_results(tmp_path / "o" / "repeats.csv")
    assert not (tmp_path / "o" / "run_summary.json").exists()


@pytest.mark.parametrize("step", ["_pilot_estimate", "stability_report", "proxy_leaks"])
def test_a_crash_in_any_later_step_leaves_the_results(tmp_path, monkeypatch, step):
    monkeypatch.setattr(f"hcai.runner.{step}", lambda *a, **k: (_ for _ in ()).throw(RuntimeError(step)))
    design = load_design(all_conditions(tmp_path))
    from hcai.config import pilot_design
    with pytest.raises(RuntimeError):
        execute(pilot_design(design), FakeLLMClient(_fake_response), out_dir=tmp_path / "o", cache_dir=tmp_path / "c")
    assert read_results(tmp_path / "o" / "results.csv")


def test_timed_out_calls_are_counted(tmp_path):
    n = [0]

    def handler(request):
        n[0] += 1
        if n[0] == 1:
            raise httpx.ReadTimeout("timed out")
        return crew_answer(request)

    path = with_run(config(tmp_path), "transient_backoff = [0]")
    design = load_design(path)
    report = run_command(path, out_dir=tmp_path / "o", cache_dir=tmp_path / "c", client=routed_client(handler))
    run_totals = json.loads((tmp_path / "o" / "run_summary.json").read_text())
    assert "1 calls this run timed out" in report
    assert run_totals["by_model"][design.models[0]]["timed_out_calls"] == 1
    summary = json.loads((tmp_path / "o" / "run_summary.json").read_text())
    assert summary["timed_out_calls"] == 1 and summary["failed_calls"] == 1


def test_the_core_config_waits_1200_seconds():
    from conftest import core_config
    assert load_design(core_config()).run.request_timeout == 1200


def test_missing_share_is_reported_overall_and_per_cell_and_warned_above_target(tmp_path, monkeypatch, capsys):
    design = load_design(all_conditions(tmp_path))
    design = replace(design, run=replace(design.run, retry_backoff=0.0, max_consecutive_failures=0))
    lines: list[str] = []

    run = execute(design, Failing("judge"), out_dir=tmp_path / "o", cache_dir=tmp_path / "c", log=lines.append)

    m = run.totals["missing"]
    assert m["missing"] == 2 and m["rows"] == run.totals["rows"] and m["above_target"]
    assert m["by_cell"][f"{design.models[0]}/C2/en"]["missing"] == 2
    assert m["by_cell"][f"{design.models[0]}/C0/en"]["missing"] == 0
    assert any(x.startswith("WARNING: 2 of") and "missing_target" in x for x in lines)
    summary = json.loads((tmp_path / "o" / "run_summary.json").read_text())
    assert summary["missing"]["above_target"] is True


def test_the_cli_prints_the_missing_share_and_exits_0(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("hcai.cli.make_client", lambda design, models: FakeLLMClient(_fake_response))
    assert main(["run", str(config(tmp_path)), "--out", str(tmp_path / "o"), "--cache", str(tmp_path / "c")]) == 0
    out = capsys.readouterr()
    assert "missing: 0 of" in out.out and "(0.00%; target under 2.00%)" in out.out and "/C0/en: 0 of" in out.out
    assert "WARNING" not in out.err


# --- N1: corrupt cache files --------------------------------------------------------------------------------


@pytest.mark.parametrize("content", [b"", b'{"text": "ok", "par', b"\xff\xfe", b"null", b"{}", b'{"text": "x"}'])
def test_a_cache_file_that_does_not_parse_is_deleted_and_asked_again(tmp_path, content):
    path = tmp_path / f"{REQ.cache_key()}.json"
    path.write_bytes(content)
    warnings: list[str] = []
    inner = FakeLLMClient(lambda r: "fresh")

    response = CachedClient(inner, tmp_path, log=warnings.append).complete(REQ)

    assert response.text == "fresh" and not response.cached and len(inner.calls) == 1
    assert json.loads(path.read_text())["text"] == "fresh"  # replaced by the new answer
    assert len(warnings) == 1 and "does not parse" in warnings[0]


# --- cache keys ---------------------------------------------------------------------------------------------


def test_the_requests_and_cache_keys_of_every_agent_are_unchanged():
    """Keys computed before #12: the pilot's cache must replay. A retry note is part of the request too."""
    from hcai.blinding import blind
    from hcai.oversight import judge
    from hcai.pipeline import rank

    seen = {"ev": 0}

    def respond(r):
        props = r.response_schema["properties"]
        if "passed" in props:
            return {"passed": True, "reason": "ok"}
        if "ranking" in props:
            return {"ranking": props["ranking"]["items"]["enum"], "rationale": "x"}
        if "blinded_cv" in props:
            return {"blinded_cv": "CV"}
        if "verdicts" in props:
            return {"verdicts": [{"criterion": 1, "verdict": "faithful", "reason": "r"}]}
        seen["ev"] += 1
        return "not json" if seen["ev"] == 1 else {"score": 50, "rationale": "r"}

    c = FakeLLMClient(respond)
    screen_and_evaluate(c, model="qwen3.8-flash", job_ad="Nurse. Mandatory: licence.", cv="Ana\nLicence",
                        temperature=0.0, seed=42, retry=Retry(retries=2))
    rank(c, model="gpt-5.6-luna", job_ad="Nurse.", cvs=["A", "B"], temperature=0.0, seed=42)
    blind(c, model="qwen3.8-flash", cv="Ana Souza\nNationality: Brazilian", temperature=0.0, seed=42)
    judge(c, model="gemini-4-argon", cv="CV", evidence=[{"criterion": "licence", "score": 5, "quote": "Licence"}],
          temperature=0.0, seed=42)

    assert [r.cache_key() for r in c.calls] == [
        "ad0344fe3998159c4152a9c0b082fc438f507ab8511056631b0bc1418c99c470",  # screener
        "9dcbf3dfd5a5287798dc4fa0337143320d8795c74bfe22bba2c0d2052f73b485",  # evaluator
        "716da7f34cfa05a871135dc04dc8ddbe922a152bbbed0275021af17c9b1840a9",  # evaluator retry, with the note
        "289701285508fbd2d58dda549a8611e2c4c98ca1f79b676ba1e96849fbc7d69c",  # ranker
        "33f5d5ff4c3cd25f193a73bc1d6efeda88b5d9703511cc878e1f02b7fdacd386",  # blinder
        "b680e85f3de92efaa7e3d27ca461367699011fab0f609f8159dd6ea182596542",  # judge
    ]


def test_screened_out_rows_are_not_counted_in_the_missing_share():
    from hcai.runner import ResultRow, missing_counts
    base: dict[str, Any] = dict(base_cv="b", identity="i", gender="g", origin="o", name="n", language="en", condition="C0",
                model="m", score=None, rationale="", prompt_tokens=0, completion_tokens=0, cost=None, cached=False,
                error="")
    rows = ([ResultRow(agent="screener", status="ok", **base)] * 2 + [ResultRow(agent="evaluator", status="rejected", **base)] * 6
            + [ResultRow(agent="evaluator", status="missing", **base)] * 2)

    m = missing_counts(rows, 0.02)
    assert (m["rows"], m["missing"], m["share"]) == (4, 2, 0.5) and m["by_cell"]["m/C0/en"]["rows"] == 4
