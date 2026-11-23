"""Run logs (#12): run.log, calls.jsonl, warnings, heartbeat and progress.json, start and end lines.

Fake clients and `httpx.MockTransport` only; nothing leaves the machine.
"""

import hashlib
import json
import logging
import re
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from hcai import runlog
from hcai.cli import _fake_response, main
from hcai.config import load_design
from hcai.llm import (
    BillingStop, FailureGuard, FakeLLMClient, LLMRequest, ProviderError, RateLimited, RateLimitGuard, TooManyFailures,
)
from hcai.pipeline import Retry
from hcai.providers import SESSION_ID
from hcai.runner import execute
from test_providers import SECRET, config, crew_answer, routed_client
from test_run_guards import Billed, Failing, all_conditions, with_run

REQ = LLMRequest(model="m", messages=[{"role": "user", "content": "hi"}])
FIELDS = {"ts", "run", "block", "role", "lang", "condition", "model", "agent", "attempt", "cache", "key",
          "latency_s", "prompt_tokens", "completion_tokens", "total_tokens", "finish_reason", "allowance",
          "balance_cost", "billing_source", "http_status", "outcome", "error_class", "error", "retry", "wait_s",
          "prompt_chars"}


def records(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / "calls.jsonl").read_text().splitlines() if line.strip()]


def cli_run(tmp_path: Path, path: Path, *extra: str) -> int:
    return main(["run", str(path), "--out", str(tmp_path / "out"), "--cache", str(tmp_path / "cache"), *extra])


# --- calls.jsonl --------------------------------------------------------------------------------------------


def test_every_agent_attempt_is_one_record_with_its_context(tmp_path):
    design = load_design(all_conditions(tmp_path))
    client = FakeLLMClient(_fake_response)

    run = execute(design, client, out_dir=tmp_path / "o", cache_dir=tmp_path / "c")

    recs = records(tmp_path / "o")
    misses = [r for r in recs if r["cache"] == "miss"]
    assert len(recs) == run.totals["attempts"] and len(misses) == len(client.calls) == run.totals["calls"]
    assert len(recs) - len(misses) == run.totals["cache_hits"] > 0  # C2's Screener asks what C0's did
    assert all(FIELDS <= set(r) for r in recs)
    assert {r["agent"] for r in recs} == {"screener", "evaluator", "ranker", "blinder", "judge"}
    assert {r["outcome"] for r in recs} == {"ok"}
    assert len({r["run"] for r in recs}) == 1
    assert [r["key"] for r in misses] == [c.cache_key()[:16] for c in client.calls]
    cv = next(r for r in recs if r["agent"] == "screener")
    assert cv["variant"] and cv["identity"] and cv["base_cv"] and cv["block"].endswith("/3")
    assert cv["role"] == design.roles[0] and cv["lang"] == "en" and cv["condition"] == "C0"
    assert cv["total_tokens"] == cv["prompt_tokens"] + cv["completion_tokens"] and cv["prompt_chars"] > 0
    ranker = next(r for r in recs if r["agent"] == "ranker")
    assert ranker["pool"] and "variant" not in ranker
    judge = next(r for r in recs if r["agent"] == "judge")
    assert judge["model"] == "judge-model" and judge["block_model"] == design.models[0]


def test_no_prompt_or_answer_text_is_recorded(tmp_path):
    design = load_design(all_conditions(tmp_path))
    client = FakeLLMClient(_fake_response)
    execute(design, client, out_dir=tmp_path / "o", cache_dir=tmp_path / "c")

    written = (tmp_path / "o" / "calls.jsonl").read_text()
    assert "fake LLM" not in written  # the answers
    assert "Mandatory" not in written and "## CV" not in written and "Nationality" not in written  # the prompts
    first = records(tmp_path / "o")[0]
    answer = _fake_response(client.calls[0])
    assert first["answer_sha"] == hashlib.sha256(json.dumps(answer).encode()).hexdigest()[:16]


def test_a_rerun_appends_and_marks_cache_hits(tmp_path):
    design = load_design(config(tmp_path))
    execute(design, FakeLLMClient(_fake_response), out_dir=tmp_path / "o", cache_dir=tmp_path / "c")
    first = records(tmp_path / "o")

    execute(design, FakeLLMClient(_fake_response), out_dir=tmp_path / "o", cache_dir=tmp_path / "c")

    recs = records(tmp_path / "o")
    again = recs[len(first):]
    assert recs[: len(first)] == first and len(again) == len(first)
    assert {r["cache"] for r in again} == {"hit"} and len({r["run"] for r in recs}) == 2
    assert [r["key"] for r in again] == [r["key"] for r in first]


def test_a_partial_last_line_from_a_killed_run_does_not_break_the_next_record(tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    (out / "calls.jsonl").write_text('{"ts": "x"}\n{"ts": "cut sho')
    execute(load_design(config(tmp_path)), FakeLLMClient(_fake_response), out_dir=out, cache_dir=tmp_path / "c")

    lines = (out / "calls.jsonl").read_text().splitlines()
    assert lines[1] == '{"ts": "cut sho' and all(json.loads(x) for x in lines[2:])


def test_retries_and_timeouts_are_recorded_with_their_wait(tmp_path, caplog, monkeypatch):
    path = with_run(config(tmp_path), "transient_backoff = [30, 120]")
    design = replace(load_design(path), run=replace(load_design(path).run, max_consecutive_failures=0))
    timeout = ProviderError("p: ReadTimeout: timed out", timeout=True)
    caplog.set_level(logging.INFO, logger="hcai")
    sleeps: list[float] = []
    monkeypatch.setattr("hcai.runner.Retry", lambda **kw: Retry(**kw, sleep=sleeps.append))

    execute(design, Failing("screener", timeout), out_dir=tmp_path / "o", cache_dir=tmp_path / "c")

    screener = [r for r in records(tmp_path / "o") if r["agent"] == "screener"]
    assert [r["outcome"] for r in screener[:3]] == ["timeout"] * 3
    assert [(r["retry"], r["wait_s"]) for r in screener[:3]] == [(True, 30), (True, 120), (False, 0)]
    assert screener[0]["error_class"] == "ProviderError" and "ReadTimeout" in screener[0]["error"]
    assert sleeps[:2] == [30, 120]
    lines = [r for r in caplog.records if r.name == "hcai.calls" and r.levelno == logging.WARNING]
    assert any("-> timeout" in r.getMessage() and "retry in 30s" in r.getMessage() for r in lines)
    missing = [r.getMessage() for r in caplog.records if r.getMessage().startswith("missing: screener")]
    assert missing and "ReadTimeout" in missing[0]


def test_a_malformed_answer_is_a_parse_error_retried_at_once(tmp_path):
    answers = {"n": 0}

    def respond(r):
        if "passed" in r.response_schema["properties"]:
            answers["n"] += 1
            if answers["n"] == 1:
                return "not json"
        return _fake_response(r)

    execute(load_design(config(tmp_path)), FakeLLMClient(respond), out_dir=tmp_path / "o", cache_dir=tmp_path / "c")

    first, second = [r for r in records(tmp_path / "o") if r["agent"] == "screener"][:2]
    assert (first["outcome"], first["retry"], first["wait_s"], first["error_class"]) == ("parse_error", True, 0, "ParseError")
    assert (second["outcome"], second["attempt"]) == ("ok", 2)


def test_a_rate_limit_wait_is_a_record_and_a_warning_with_the_resume_time(tmp_path, caplog):
    state = {"n": 0}

    def handler(request):
        state["n"] += 1
        if state["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "60"}, text="rate limit")
        return crew_answer(request)

    design = load_design(config(tmp_path))
    caplog.set_level(logging.INFO, logger="hcai")
    execute(design, routed_client(handler, design=design), out_dir=tmp_path / "o", cache_dir=tmp_path / "c")

    recs = records(tmp_path / "o")
    limited = recs[0]
    assert limited["outcome"] == "rate_limited" and limited["wait_s"] == 60 and limited["http_status"] == 429
    assert limited["resume_at"] and limited["retry"] is True
    assert recs[1]["outcome"] == "ok" and recs[1]["attempt"] == 1 and recs[1]["http_status"] == 200
    warning = next(r.getMessage() for r in caplog.records if "rate/usage limit" in r.getMessage())
    assert "waiting 60s, resuming at 20" in warning
    ok = next(r for r in recs if r["outcome"] == "ok")
    assert ok["finish_reason"] is None or isinstance(ok["finish_reason"], str)
    assert ok["latency_s"] >= 0


def test_a_stop_is_an_exhausted_record_an_error_line_and_progress_numbers(tmp_path, caplog):
    design = load_design(config(tmp_path))
    with pytest.raises(BillingStop):
        execute(design, Billed("evaluator"), out_dir=tmp_path / "o", cache_dir=tmp_path / "c")

    last = records(tmp_path / "o")[-1]
    assert last["outcome"] == "exhausted" and last["error_class"] == "BillingStop"
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR and r.name == "hcai.runner"]
    assert len(errors) == 1 and "stopped (billing)" in errors[0] and "rerun the same command" in errors[0]
    progress = json.loads((tmp_path / "o" / "progress.json").read_text())
    assert progress["status"] == "stopped" and progress["stop"] == "billing"
    assert progress["rows_total"] > 0 and "cache_misses" in progress and progress["run_id"] == last["run"]


def test_too_many_failures_warn_from_five_in_a_row(caplog):
    calls = iter(range(100))

    class Down:
        def complete(self, request):
            raise ProviderError(f"down {next(calls)}")

    guard = FailureGuard(Down(), 7)
    for i in range(7):
        with pytest.raises((ProviderError, TooManyFailures)):
            guard.complete(replace(REQ, messages=[{"role": "user", "content": str(i)}]))
        with pytest.raises((ProviderError, TooManyFailures)):  # a retry of the same call: no second warning
            guard.complete(replace(REQ, messages=[{"role": "user", "content": str(i)}]))
    warned = [r.getMessage() for r in caplog.records if "failed calls in a row" in r.getMessage()]
    assert [w.split(":")[1].split()[0] for w in warned] == ["5", "6", "7"]  # the 7th also stops
    assert "max_consecutive_failures = 7" in warned[0]


def test_a_new_run_of_failures_after_an_answer_warns_again(caplog):
    fail = {"on": True}

    class Flaky:
        def complete(self, request):
            if fail["on"]:
                raise ProviderError("down")
            return FakeLLMClient(lambda r: "ok").complete(request)

    guard = FailureGuard(Flaky(), 0)
    for streak in range(2):
        fail["on"] = True
        for i in range(5):
            with pytest.raises(ProviderError):
                guard.complete(replace(REQ, messages=[{"role": "user", "content": f"{streak}-{i}"}]))
        fail["on"] = False
        guard.complete(REQ)
    assert sum("5 failed calls in a row" in r.getMessage() for r in caplog.records) == 2


def test_logging_failures_never_stop_the_run(tmp_path, monkeypatch):
    out = tmp_path / "o"
    (out / "calls.jsonl").mkdir(parents=True)  # cannot be opened as a file
    monkeypatch.setattr(runlog, "_line", lambda r: 1 / 0)  # and every log line raises

    run = execute(load_design(config(tmp_path)), FakeLLMClient(_fake_response), out_dir=out, cache_dir=tmp_path / "c")

    assert run.totals["calls"] > 0 and (out / "results.csv").exists()


def test_a_full_disk_drops_the_records_with_one_warning(tmp_path, caplog):
    rl = runlog.RunLog(tmp_path, heartbeat_s=0)
    with rl:
        rl._file.close()  # writes now fail like a full disk
        rl.write({"a": 1})
        rl.write({"a": 2})
    assert sum("cannot write calls.jsonl" in r.getMessage() for r in caplog.records) == 1


# --- heartbeat and progress.json ----------------------------------------------------------------------------


def test_the_heartbeat_logs_rows_rate_and_eta_and_rewrites_progress(tmp_path, caplog):
    now = [0.0]
    caplog.set_level(logging.INFO, logger="hcai")
    rl = runlog.RunLog(tmp_path, heartbeat_s=0, clock=lambda: now[0],
                       stats=lambda: dict(calls=12, allowance_spent=0.5, balance_spent=0.0))
    rl.rows_total = 1000
    rl.block(2, 36, role="nurse", language="pt", condition="C1", model="m")
    now[0] = 1800.0  # half an hour, 100 rows
    rl.rows_done = 100

    rl.heartbeat()

    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("heartbeat:"))
    assert "rows 100/1000 (10.0%)" in line and "12 calls" in line and "200 rows/h" in line and "ETA 20" in line
    assert "allowance $0.5000" in line and "block 2/36 nurse/pt/C1/m" in line
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["status"] == "running" and progress["rows_done"] == 100 and progress["percent"] == 10.0
    assert progress["rows_per_hour"] == 200.0 and progress["eta"] and progress["calls"] == 12
    assert progress["block"] == "2/36 nurse/pt/C1/m"


def test_the_heartbeat_thread_runs_during_a_run_and_stops_with_it(tmp_path):
    import threading
    import time as t

    path = with_run(config(tmp_path), "heartbeat_s = 0.05")

    def slow(r):
        t.sleep(0.03)
        return _fake_response(r)

    execute(load_design(path), FakeLLMClient(slow), out_dir=tmp_path / "o", cache_dir=tmp_path / "c")

    assert not (tmp_path / "o" / "progress.json").exists()  # a finished run removes it, as before
    assert not any(th.name == "hcai-heartbeat" for th in threading.enumerate())


def test_progress_is_running_while_the_run_is(tmp_path):
    seen: list[dict] = []

    def peek(r):
        seen.append(json.loads((tmp_path / "o" / "progress.json").read_text()))
        return _fake_response(r)

    execute(load_design(config(tmp_path)), FakeLLMClient(peek), out_dir=tmp_path / "o", cache_dir=tmp_path / "c")
    assert seen[0]["status"] == "running" and seen[0]["rows_total"] > 0


@pytest.mark.parametrize("line", ["heartbeat_s = 0", "heartbeat_s = -5", 'heartbeat_s = "x"', "heartbeat_s = true"])
def test_an_invalid_heartbeat_is_refused(tmp_path, line):
    with pytest.raises(ValueError, match="heartbeat_s"):
        load_design(with_run(config(tmp_path), line))


# --- run.log through the CLI --------------------------------------------------------------------------------


def test_the_cli_appends_run_log_with_iso_timestamps_and_the_start_and_end(tmp_path, monkeypatch):
    monkeypatch.setattr("hcai.cli.make_client", lambda design, models: FakeLLMClient(_fake_response))
    path = config(tmp_path)
    assert cli_run(tmp_path, path) == 0
    assert cli_run(tmp_path, path) == 0

    text = (tmp_path / "out" / "run.log").read_text()
    lines = text.splitlines()
    assert all(re.match(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}[+-]\d\d:\d\d (DEBUG|INFO|WARNING|ERROR) +hcai\.", x)
               for x in lines), lines[:3]
    assert text.count("starting: out") == 2 and text.count(" done in ") == 2
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert f"sha256 {sha}" in text and "git " in text and "plan: 1 blocks" in text
    assert "https://opencode.ai/zen/go/v1" in text and "request_timeout" in text and "allowed_billing" in text
    assert "hcai.calls: screener" in text  # the first run's calls (misses) at INFO
    assert " hit " not in text  # the rerun's cache hits are DEBUG only
    assert len([r for r in records(tmp_path / "out") if r["cache"] == "hit"]) > 0


def test_debug_logs_cache_hits_and_quiet_keeps_stderr_to_warnings(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("hcai.cli.make_client", lambda design, models: FakeLLMClient(_fake_response))
    path = config(tmp_path)
    assert cli_run(tmp_path, path, "--quiet") == 0
    assert "INFO" not in capsys.readouterr().err
    assert cli_run(tmp_path, path, "--log-level", "debug") == 0
    err = capsys.readouterr().err
    assert " hit " in (tmp_path / "out" / "run.log").read_text()
    assert "INFO" in err and "DEBUG" not in err  # stderr stays at INFO


def test_names_writes_its_own_log(tmp_path, monkeypatch):
    from conftest import core_config
    out = tmp_path / "names"
    main(["names", str(core_config(tmp_path)), "--fake-llm", "--repeats", "1", "--out", str(out),
          "--cache", str(tmp_path / "cache")])
    assert (out / "names.log").exists()


def test_no_key_header_or_session_id_reaches_any_log(tmp_path, monkeypatch, capsys):
    state = {"n": 0}

    def handler(request):
        state["n"] += 1
        if state["n"] <= 3:  # errors that echo the key and the session id back
            return httpx.Response(500, text=f"bad key {SECRET} session {SESSION_ID}")
        if state["n"] == 4:
            return httpx.Response(429, text=f"rate limit for {SECRET} {SESSION_ID}")
        return crew_answer(request)

    path = with_run(config(tmp_path), "transient_backoff = [0, 0]")
    path.write_text(path.read_text().replace("retry_backoff = 2.0", "retry_backoff = 0"))
    design = load_design(path)
    monkeypatch.setattr("hcai.cli.make_client", lambda d, models: routed_client(handler, design=design))
    assert cli_run(tmp_path, path, "--log-level", "DEBUG") == 0

    out = tmp_path / "out"
    for name in ("run.log", "calls.jsonl"):
        text = (out / name).read_text()
        assert SECRET not in text and SESSION_ID not in text and "Bearer" not in text, name
    assert "***" in (out / "calls.jsonl").read_text()  # the echoed key, redacted
    captured = capsys.readouterr()
    assert SECRET not in captured.err + captured.out and SESSION_ID not in captured.err + captured.out


def test_rate_limited_answers_reach_the_guard_not_the_agent():
    """A RateLimited that escapes (no guard) is still recorded as rate_limited."""
    assert runlog.outcome_of(RateLimited("x")) == "rate_limited"
    assert runlog.outcome_of(ProviderError("x", timeout=True)) == "timeout"
    assert runlog.outcome_of(ProviderError("x")) == "provider_error"
    assert runlog.outcome_of(TooManyFailures("x", model="m", failures=3)) == "exhausted"


def test_the_guard_logs_through_the_logger_by_default(caplog):
    class Once:
        n = 0

        def complete(self, request):
            self.n += 1
            if self.n == 1:
                raise RateLimited("HTTP 429", retry_after=None)
            return FakeLLMClient(lambda r: "ok").complete(request)

    RateLimitGuard(Once(), sleep=lambda s: None, backoff=30).complete(REQ)
    assert any(r.name == "hcai.llm" and "waiting 30s, resuming at" in r.getMessage() for r in caplog.records)
