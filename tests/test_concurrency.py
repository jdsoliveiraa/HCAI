"""Bounded concurrency (`[run] concurrency`, #12): the same requests, cache keys and results as a sequential run;
guards, logs and stops that stay correct with several workers. Fake clients and `httpx.MockTransport` only."""

import _thread
import hashlib
import json
import random
import threading
import time
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from hcai.cli import _fake_response
from hcai.concurrency import Task, run_tasks
from hcai.config import load_design
from hcai.llm import (
    CachedClient, FailureGuard, FakeLLMClient, LLMRequest, LLMResponse, ProviderError, ProviderExhausted,
    RateLimitGuard, RateLimited, Stopping, stop_scope,
)
from hcai.runner import execute, read_results
from test_providers import SECRET, routed_client
from test_run_guards import Billed, Failing, all_conditions, with_run

REQ = LLMRequest(model="m", messages=[{"role": "user", "content": "hi"}])


def wide(tmp_path: Path, concurrency: int, *extra: str) -> Path:
    """Every condition (every agent), four identities, a stability sample, and `concurrency` workers."""
    path = all_conditions(tmp_path)
    text = path.read_text().replace("repeat_fraction = 0.0", "repeat_fraction = 0.5")
    for ident, gender, origin, name in (("pt_f", "female", "portuguese", "Inês Carvalho"),
                                        ("br_m", "male", "brazilian", "Matheus Costa")):
        text = text.replace("[pools]", f'[[identities]]\nid = "{ident}"\ngender = "{gender}"\norigin = "{origin}"\n'
                                       f'names = ["{name}"]\nnationality = "X"\n\n[pools]', 1)
    text = text.replace("mixed_quality = 0", "mixed_quality = 2").replace("mixed_size = 8", "mixed_size = 1")
    path.write_text(text)
    return with_run(path, f"concurrency = {concurrency}", "transient_backoff = [0, 0, 0]", *extra)


class Slow(FakeLLMClient):
    """The fake crew with a random latency per call (thread-safe)."""

    def __init__(self, seed: int = 0, latency: float = 0.01):
        super().__init__(_fake_response)
        self.rng, self.latency, self.lock = random.Random(seed), latency, threading.Lock()

    def complete(self, request):
        with self.lock:
            wait = self.rng.uniform(0, self.latency)
        time.sleep(wait)
        return super().complete(request)


def cache_files(cache: Path) -> set[str]:
    return {str(p.relative_to(cache)) for p in cache.rglob("*.json")}


def table(path: Path) -> list[str]:
    return path.read_text().splitlines()


def run_at(tmp_path: Path, name: str, concurrency: int, client, *extra: str):
    d = tmp_path / name
    d.mkdir()
    run = execute(load_design(wide(d, concurrency, *extra)), client, out_dir=d / "out", cache_dir=d / "cache")
    return run, d


def test_concurrency_3_makes_the_same_requests_and_results_as_1(tmp_path):
    one, two = Slow(1), Slow(2)
    run1, d1 = run_at(tmp_path, "c1", 1, one)
    run3, d3 = run_at(tmp_path, "c3", 3, two)

    assert sorted(r.cache_key() for r in one.calls) == sorted(r.cache_key() for r in two.calls)  # no call twice
    assert cache_files(d1 / "cache") == cache_files(d3 / "cache")
    assert any(p.startswith("repeats/") for p in cache_files(d3 / "cache"))  # the stability sample ran
    # the same rows in the same (sequential) order; rows sorted too, in case a field order ever differs
    assert table(d1 / "out" / "results.csv") == table(d3 / "out" / "results.csv")
    assert sorted(table(d1 / "out" / "results.csv")) == sorted(table(d3 / "out" / "results.csv"))
    assert table(d1 / "out" / "repeats.csv") == table(d3 / "out" / "repeats.csv")
    assert table(d1 / "out" / "review_queue.csv") == table(d3 / "out" / "review_queue.csv")
    assert {r.agent for r in run3.rows} == {"screener", "evaluator", "ranker", "blinder", "judge"}
    assert {r.pool_kind for r in run3.rows if r.agent == "ranker"} == {"equal", "mixed"}
    assert run1.totals["calls"] == run3.totals["calls"] and run1.totals["missing"] == run3.totals["missing"]


def test_a_rerun_at_concurrency_3_replays_a_sequential_runs_cache(tmp_path):
    first, replay1, replay3 = Slow(1), Slow(2), Slow(3)
    _, d = run_at(tmp_path, "c1", 1, first)
    execute(load_design(d / "all.toml"), replay1, out_dir=d / "replay1", cache_dir=d / "cache")
    execute(load_design(wide(tmp_path / "c1", 3)), replay3, out_dir=d / "replay3", cache_dir=d / "cache")
    assert replay1.calls == [] and replay3.calls == []
    assert table(d / "replay1" / "results.csv") == table(d / "replay3" / "results.csv")
    assert {r.cached for r in read_results(d / "replay3" / "results.csv") if r.attempts} == {True}


def test_calls_jsonl_lines_are_whole_and_name_their_own_cv_or_pool(tmp_path):
    _, d = run_at(tmp_path, "c3", 3, Slow(3))
    records = [json.loads(line) for line in (d / "out" / "calls.jsonl").read_text().splitlines()]
    assert records
    for r in records:
        if r["agent"] == "ranker":
            assert r.get("pool") and not r.get("variant"), r
        else:
            assert r.get("variant") and not r.get("pool"), r
            assert r["variant"].startswith(r["base_cv"])
    log_lines = (d / "out" / "calls.jsonl").read_text().count("\n")
    assert log_lines == len(records)


def flaky_provider(seed: int):
    """An HTTP provider with random latency answering the fake crew; the first try of some requests gets a 429 or
    a 500 first (the same requests whatever the order, so both runs see the same faults)."""
    seen: set[str] = set()
    lock = threading.Lock()
    rng = random.Random(seed)

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        digest = hashlib.sha256(request.content).hexdigest()
        with lock:
            first = digest not in seen
            seen.add(digest)
            wait = rng.uniform(0, 0.01)
        time.sleep(wait)
        fault = int(digest[:2], 16) % 5  # 0: a 429, 1: a 500, else answered at once
        if first and fault == 0:
            return httpx.Response(429, json={"error": {"message": "Rate limit exceeded"}})
        if first and fault == 1:
            return httpx.Response(500, json={"error": {"message": "upstream failed"}})
        schema = body.get("response_format", {}).get("json_schema", {}).get("schema")
        out = _fake_response(LLMRequest(model=body["model"], messages=body["messages"], response_schema=schema))
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(out)}, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 3, "completion_tokens": 2}})

    return handler


def test_injected_429s_and_5xx_with_random_latency_give_the_sequential_results(tmp_path):
    sleeps: list[float] = []
    design = load_design(wide(tmp_path, 1))
    _, d1 = run_at(tmp_path, "c1", 1, routed_client(flaky_provider(1), design=design, sleep=sleeps.append))
    _, d3 = run_at(tmp_path, "c3", 3, routed_client(flaky_provider(2), design=design, sleep=sleeps.append))

    assert sleeps  # some 429s were waited out
    assert cache_files(d1 / "cache") == cache_files(d3 / "cache")
    assert table(d1 / "out" / "results.csv") == table(d3 / "out" / "results.csv")
    rows = read_results(d3 / "out" / "results.csv")
    assert all(r.status != "missing" for r in rows)
    records = [json.loads(x) for x in (d3 / "out" / "calls.jsonl").read_text().splitlines()]
    assert {r["outcome"] for r in records} >= {"ok", "rate_limited", "provider_error"}


# --- the guards --------------------------------------------------------------------------------------------


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_one_429_pauses_every_worker_once_instead_of_each_backing_off():
    """Two calls in flight both get a 429: the second joins the first one's pause (no doubled backoff of its own),
    and a third call sent during the pause waits for it before it is sent."""
    clock = Clock()
    both_sent = threading.Barrier(2)
    first_paused = threading.Event()
    sleeps: list[tuple[str, float]] = []
    tries: dict[str, int] = {}
    lock = threading.Lock()

    class Inner:
        def complete(self, request):
            who = request.messages[0]["content"]
            with lock:
                tries[who] = tries.get(who, 0) + 1
                n = tries[who]
            if who in ("a", "b") and n == 1:
                both_sent.wait(timeout=5)
                if who == "b":
                    assert first_paused.wait(timeout=5)
                raise RateLimited("HTTP 429: Rate limit exceeded")
            return LLMResponse(text="ok")

    def sleep(s):
        sleeps.append((threading.current_thread().name, s))
        if threading.current_thread().name == "a":
            first_paused.set()

    guard = RateLimitGuard(Inner(), backoff=30, max_backoff=900, sleep=sleep, log=lambda m: None, clock=clock)
    req = {w: LLMRequest(model="m", messages=[{"role": "user", "content": w}]) for w in "abc"}
    threads = [threading.Thread(target=guard.complete, args=(req[w],), name=w) for w in "ab"]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert sorted(s for _, s in sleeps) == [30, 30]  # one pause, joined: not 30 and 60
    assert tries == {"a": 2, "b": 2}
    sleeps.clear()
    third = threading.Thread(target=guard.complete, args=(req["c"],), name="c")
    third.start()
    third.join(timeout=5)
    assert sleeps == []  # a retry that answered lifted the pause (the provider serves again)


def test_a_call_sent_during_a_pause_waits_for_it_before_it_is_sent():
    clock = Clock()
    sleeps: list[float] = []
    sent: list[str] = []

    class Inner:
        def complete(self, request):
            sent.append(request.messages[0]["content"])
            if sent == ["a"]:
                raise RateLimited("HTTP 429: Rate limit exceeded")
            return LLMResponse(text="ok")

    def sleep(s):
        sleeps.append(s)
        if len(sleeps) == 1:  # while "a" waits out its pause, "b" is asked
            clock.now += 10
            guard.complete(LLMRequest(model="m", messages=[{"role": "user", "content": "b"}]))
            clock.now += 20

    guard = RateLimitGuard(Inner(), backoff=30, sleep=sleep, log=lambda m: None, clock=clock)
    guard.complete(LLMRequest(model="m", messages=[{"role": "user", "content": "a"}]))
    assert sleeps == [30, 20] and sent == ["a", "b", "a"]  # "b" waited the 20 s left of the pause, then went


def test_a_stop_ends_a_rate_limit_wait_and_the_call_is_not_repeated():
    calls = []

    class Inner:
        def complete(self, request):
            calls.append(request)
            raise RateLimited("HTTP 429: Rate limit exceeded")

    event = threading.Event()
    guard = RateLimitGuard(Inner(), backoff=30, log=lambda m: None)  # the default, interruptible sleep
    with stop_scope(event):
        threading.Timer(0.1, event.set).start()
        started = time.monotonic()
        with pytest.raises(Stopping):
            guard.complete(REQ)
    assert time.monotonic() - started < 5 and len(calls) == 1


def test_failures_count_distinct_requests_since_the_last_answer_whatever_the_interleaving():
    class Down:
        def complete(self, request):
            raise ProviderError("HTTP 503", transient=True)

    guard = FailureGuard(Down(), limit=3)
    a, b = (LLMRequest(model="m", messages=[{"role": "user", "content": x}]) for x in "ab")
    for r in (a, b, a, b, a):  # two workers retrying their own calls, interleaved
        with pytest.raises(ProviderError):
            guard.complete(r)
    assert guard.in_a_row["m"] == 2


def test_two_workers_asking_the_same_request_make_one_call(tmp_path):
    inner = Slow(latency=0)
    gate = threading.Event()

    class Held:
        def complete(self, request):
            gate.wait(timeout=5)
            return inner.complete(request)

    client = CachedClient(Held(), tmp_path)
    req = LLMRequest(model="m", messages=[{"role": "user", "content": "x"}], response_schema={"properties": {"score": {}}})
    out: list[LLMResponse] = []
    threads = [threading.Thread(target=lambda: out.append(client.complete(req))) for _ in range(3)]
    for t in threads:
        t.start()
    time.sleep(0.1)
    gate.set()
    for t in threads:
        t.join(timeout=5)
    assert len(inner.calls) == 1 and sorted(r.cached for r in out) == [False, True, True]
    assert [p.name for p in tmp_path.iterdir()] == [f"{req.cache_key()}.json"]  # no temporary file left


def test_a_billing_stop_under_concurrency_keeps_every_paid_answer_and_a_rerun_resumes(tmp_path):
    client = Billed("evaluator")
    path = wide(tmp_path, 3)
    with pytest.raises(BaseException) as stop:
        execute(load_design(path), client, out_dir=tmp_path / "out", cache_dir=tmp_path / "cache")
    assert type(stop.value).__name__ == "BillingStop"
    progress = json.loads((tmp_path / "out" / "progress.json").read_text())
    assert progress["status"] == "stopped" and progress["stop"] == "billing"
    paid = {r.cache_key() for r in client.calls}
    assert {p.stem for p in (tmp_path / "cache").glob("*.json")} == paid  # every answered call is cached
    records = [json.loads(x) for x in (tmp_path / "out" / "calls.jsonl").read_text().splitlines()]
    assert all(r["outcome"] in ("ok", "exhausted", "abandoned") for r in records)

    again = Billed(None)
    execute(load_design(with_run(path, 'allowed_billing = ["go", "balance"]')), again, out_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache")
    reference = Billed(None)
    _, d1 = run_at(tmp_path, "ref", 1, reference)
    assert len(client.calls) + len(again.calls) == len(reference.calls)  # nothing paid twice
    assert cache_files(tmp_path / "cache") == cache_files(d1 / "cache")
    # the same answers as an uninterrupted run (`cached` and the billing differ: part was replayed, one call was
    # billed to the balance)
    strip = lambda rows: [replace(r, cached=False, cost=None, allowance=None, billing_source="") for r in rows]  # noqa: E731
    assert strip(read_results(tmp_path / "out" / "results.csv")) == strip(read_results(d1 / "out" / "results.csv"))
    assert not (tmp_path / "out" / "progress.json").exists()


def test_too_many_failures_under_concurrency_stop_the_run_cleanly(tmp_path):
    with pytest.raises(BaseException) as stop:
        execute(load_design(wide(tmp_path, 3, "max_consecutive_failures = 3")), Failing("screener"),
                out_dir=tmp_path / "out", cache_dir=tmp_path / "cache")
    assert type(stop.value).__name__ == "TooManyFailures"
    progress = json.loads((tmp_path / "out" / "progress.json").read_text())
    assert progress["stop"] == "failures"


# --- the scheduler -----------------------------------------------------------------------------------------


def test_tasks_respect_their_dependencies_and_return_results_in_list_order():
    started: list[int] = []
    lock = threading.Lock()

    def task(i: int, wait: float):
        def fn():
            with lock:
                started.append(i)
            time.sleep(wait)
            return i
        return fn

    tasks = [Task(task(0, 0.05)), Task(task(1, 0.0)), Task(task(2, 0.0), deps=(0,), priority=-1), Task(task(3, 0.0))]
    assert run_tasks(tasks, 3, threading.Event()) == [0, 1, 2, 3]
    assert started.index(2) > started.index(0)
    assert run_tasks(tasks, 1, threading.Event()) == [0, 1, 2, 3]


def test_a_stop_in_one_task_lets_the_others_finish_and_starts_no_new_one():
    event = threading.Event()
    finished: list[int] = []

    def stopper():
        raise ProviderExhausted("limit")

    def worker(i):
        def fn():
            time.sleep(0.1)
            finished.append(i)
        return fn

    tasks = [Task(stopper), Task(worker(1)), Task(worker(2)), Task(worker(3)), Task(worker(4))]
    with pytest.raises(BaseException) as e:
        run_tasks(tasks, 3, event)
    assert type(e.value).__name__ == "ProviderExhausted" and event.is_set()
    assert sorted(finished) == [1, 2]  # the two in flight finished; none started after the stop


def test_ctrl_c_waits_for_the_tasks_in_flight_then_raises():
    event = threading.Event()
    finished: list[int] = []

    def interrupt():
        time.sleep(0.2)  # both tasks are in flight
        _thread.interrupt_main()
        event.wait(timeout=5)
        finished.append(0)

    def worker():
        event.wait(timeout=5)  # an interruptible wait, as the run's backoffs are
        finished.append(1)

    with pytest.raises(KeyboardInterrupt):
        run_tasks([Task(interrupt), Task(worker), Task(worker), Task(worker)], 2, event)
    assert event.is_set() and sorted(finished) == [0, 1]  # the third and fourth never started


def test_an_invalid_concurrency_is_refused(tmp_path):
    for bad in ("0", "1.5", "true", "100"):
        with pytest.raises(ValueError, match="concurrency"):
            load_design(with_run(all_conditions(tmp_path), f"concurrency = {bad}"))


def test_no_key_reaches_the_logs_under_concurrency(tmp_path):
    design = load_design(wide(tmp_path, 1))
    _, d = run_at(tmp_path, "c3", 3, routed_client(flaky_provider(3), design=design))
    assert SECRET not in (d / "out" / "calls.jsonl").read_text()
