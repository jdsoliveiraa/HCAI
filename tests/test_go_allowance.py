"""OpenCode Go allowance (#11 Part B): `usage.opencode` is the cost, billed per source, summed and estimated.

Every HTTP call goes to an in-process `httpx.MockTransport`; nothing leaves the machine.
"""

import csv
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import httpx
import pytest

from hcai.cli import run_command
from hcai.config import load_design
from hcai.llm import CachedClient, LLMRequest, LLMResponse, OpenAICompatibleClient, Usage, add_usage, parse_usage
from hcai.runner import execute, read_results
from test_providers import config, crew_answer, routed_client

REQ = LLMRequest(model="m", messages=[{"role": "user", "content": "hi"}])
MODEL = "openai/gpt-4o-mini"  # the skeleton's model, on opencode-go in `config(tmp_path)`


def go_usage(allowance=0.00016, source="go", list_cost=8e-05, cost=0.0):
    """A usage block as OpenCode Go reports it."""
    return {"prompt_tokens": 35, "completion_tokens": 1, "total_tokens": 36, "cost": cost,
            "opencode": {"billing_source": source, "list_cost": list_cost, "allowance": allowance,
                         "tokens": {"input": 35, "output": 1, "reasoning": 0, "cache_read": 0,
                                    "cache_write_5m": 0, "cache_write_1h": 0}}}


def client(usage, *, subscription=True):
    answer = httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}], "usage": usage})
    return OpenAICompatibleClient("https://p.example/v1", "k", subscription=subscription,
                                  http=httpx.Client(transport=httpx.MockTransport(lambda r: answer)))


def billed(source_of):
    """A crew provider billing each call to `source_of(n)` (n = 0, 1, ...), 0.0001 of allowance per Go call."""
    n = [0]

    def handler(request):
        data = crew_answer(request).json()
        data["usage"] = go_usage(allowance=0.0001, source=source_of(n[0]), list_cost=0.00005)
        n[0] += 1
        return httpx.Response(200, json=data)
    return handler


# --- parsing the usage block ------------------------------------------------------------------------------


def test_a_go_call_costs_the_allowance_it_took_and_keeps_its_billing_source_and_list_cost():
    usage = client(go_usage()).complete(REQ).usage

    assert usage == Usage(prompt_tokens=35, completion_tokens=1, cost=0.00016, billing_source="go",
                          allowance=0.00016, list_cost=8e-05)


def test_a_call_billed_to_the_balance_costs_its_price_and_takes_no_allowance():
    at_list = parse_usage(go_usage(source="balance", allowance=0.0))
    priced = parse_usage(go_usage(source="balance", cost=0.0003))

    assert (at_list.cost, at_list.allowance, at_list.billing_source) == (8e-05, 0.0, "balance")
    assert (priced.cost, priced.allowance) == (0.0003, 0.0)
    unpriced = parse_usage({"cost": 0, "opencode": {"billing_source": "balance"}})
    assert (unpriced.cost, unpriced.allowance) == (None, 0.0)  # no price and no list price: unknown


def test_a_call_naming_no_billing_source_costs_its_allowance_and_is_flagged():
    usage = parse_usage({"prompt_tokens": 1, "cost": 0, "opencode": {"allowance": 0.0002}})

    assert (usage.cost, usage.allowance, usage.billing_source) == (0.0002, 0.0002, "unknown")


def test_a_go_answer_without_its_allowance_has_an_unknown_cost():
    usage = parse_usage({"prompt_tokens": 1, "opencode": {"billing_source": "go"}})

    assert (usage.cost, usage.allowance, usage.billing_source) == (None, None, "go")


def test_without_the_opencode_block_a_zero_cost_is_unknown_on_a_subscription_only():
    assert client({"prompt_tokens": 7, "completion_tokens": 3, "cost": 0}).complete(REQ).usage.cost is None
    metered = client({"prompt_tokens": 7, "cost": 0}, subscription=False).complete(REQ).usage
    assert (metered.cost, metered.billing_source, metered.allowance) == (0.0, None, None)
    assert client({"prompt_tokens": 7, "cost": 0.002}).complete(REQ).usage.cost == 0.002


def test_the_attempts_behind_one_output_add_their_allowance_and_name_every_source():
    go, balance = parse_usage(go_usage()), parse_usage(go_usage(source="balance", allowance=0.0))

    both = add_usage([go, balance, go])

    assert both.billing_source == "balance+go"
    assert both.allowance == pytest.approx(0.00032) and both.cost == pytest.approx(0.0004)
    assert add_usage([Usage(cost=None), Usage(cost=None)]).billing_source is None
    assert add_usage([go, Usage(cost=0.0)]).allowance is None  # one attempt reported none: unknown


# --- the cache ---------------------------------------------------------------------------------------------


def test_the_cache_stores_the_allowance_and_loads_old_entries_without_it(tmp_path):
    cached = CachedClient(client(go_usage()), tmp_path)
    cached.complete(REQ)
    entry = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert entry["usage"]["billing_source"] == "go" and entry["usage"]["allowance"] == 0.00016

    entry["usage"] = {"prompt_tokens": 35, "completion_tokens": 1, "cost": None}  # written before #11 Part B
    next(tmp_path.glob("*.json")).write_text(json.dumps(entry))
    old = cached.complete(REQ)
    assert old.cached and old.usage == Usage(prompt_tokens=35, completion_tokens=1, cost=None)

    entry["usage"] = {"prompt_tokens": 35, "completion_tokens": 1, "cost": 0.0}  # Go's 0, read as a price
    next(tmp_path.glob("*.json")).write_text(json.dumps(entry))
    assert cached.complete(REQ).usage.cost is None  # unknown, not free

    entry["usage"] = {"prompt_tokens": 35, "completion_tokens": 1, "cost": 0.002}  # an old metered price
    next(tmp_path.glob("*.json")).write_text(json.dumps(entry))
    assert cached.complete(REQ).usage.cost == 0.002

    entry["usage"] = {"prompt_tokens": 35, "cost": 0.0, "billing_source": None, "future_field": 1}
    next(tmp_path.glob("*.json")).write_text(json.dumps(entry))
    assert cached.complete(REQ).usage.cost == 0.0  # written after the change: a real 0; unknown fields ignored


def test_an_output_mixing_an_old_cache_entry_and_a_go_call_has_an_unknown_cost():
    old = Usage.from_dict({"prompt_tokens": 35, "completion_tokens": 1, "cost": 0.0})

    both = add_usage([old, parse_usage(go_usage())])

    assert both.cost is None and both.allowance is None and both.billing_source == "go"


def test_results_keep_the_allowance_and_billing_source_and_older_tables_still_load(tmp_path):
    run = execute(load_design(config(tmp_path)), routed_client(billed(lambda n: "go")), out_dir=tmp_path / "out",
                  cache_dir=tmp_path / "cache")
    path = tmp_path / "out" / "results.csv"

    rows = read_results(path)
    assert rows == run.rows
    assert all(r.billing_source == "go" and r.allowance == pytest.approx(0.0001 * r.attempts)
               for r in rows if r.attempts)

    with path.open(newline="") as f:  # a table written before #11 Part B: no allowance, billing_source
        records = [{k: v for k, v in r.items() if k not in ("allowance", "billing_source")} for r in csv.DictReader(f)]
    old_path = tmp_path / "old.csv"
    with old_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    assert all(r.allowance is None and r.billing_source == "" for r in read_results(old_path))

    for r in records:  # Go rows on main: cost 0.0 misread as a price; read back as unknown
        r["cost"] = "0.0"
    with old_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    assert all((r.cost is None) == bool(r.prompt_tokens) for r in read_results(old_path))


# --- run totals and the printed summary --------------------------------------------------------------------


def no_billing_guard(path: Path) -> Path:
    """The config with `[run] allowed_billing = []`: a call billed outside "go" is counted, not a stop (#12)."""
    path.write_text(path.read_text().replace("retry_backoff = 2.0", "retry_backoff = 2.0\nallowed_billing = []"))
    return path


def test_a_go_run_reports_the_allowance_it_took_per_model_and_overall(tmp_path):
    report = run_command(config(tmp_path), out_dir=tmp_path / "out", cache_dir=tmp_path / "cache",
                         client=routed_client(billed(lambda n: "go")))

    summary = json.loads((tmp_path / "out" / "run_summary.json").read_text())
    model = summary["by_model"][MODEL]
    assert model["calls"] == 5 and model["unpriced_calls"] == 0
    assert model["allowance_spent"] == model["cost_spent"] == pytest.approx(0.0005)
    assert summary["allowance_spent"] == summary["allowance_total"] == pytest.approx(0.0005)
    assert summary["billing_sources"] == {"go": 5} and summary["not_go_calls"] == 0
    assert "Go allowance: $0.0005 spent this run" in report and f"{MODEL}: $0.0005 this run" in report
    assert "WARNING" not in report and "cost unknown" not in report

    rerun = run_command(config(tmp_path), out_dir=tmp_path / "out", cache_dir=tmp_path / "cache",
                        client=routed_client(billed(lambda n: "go")))
    again = json.loads((tmp_path / "out" / "run_summary.json").read_text())
    assert again["calls"] == 0 and again["allowance_spent"] == 0
    assert again["allowance_total"] == pytest.approx(0.0005)  # cache hits keep what they took
    assert "Go allowance: $0.0000 spent this run (the outputs took $0.0005 in all)" in rerun


def test_calls_not_billed_to_go_are_flagged_with_what_they_took_from_the_balance(tmp_path):
    report = run_command(no_billing_guard(config(tmp_path)), out_dir=tmp_path / "out", cache_dir=tmp_path / "cache",
                         client=routed_client(billed(lambda n: "go" if n < 3 else "balance")))

    summary = json.loads((tmp_path / "out" / "run_summary.json").read_text())
    assert summary["billing_sources"] == {"go": 3, "balance": 2}
    assert summary["not_go_calls"] == 2 and summary["not_go_attempts"] == 2
    assert summary["allowance_spent"] == pytest.approx(0.0003)
    assert summary["balance_spent"] == pytest.approx(0.0001)  # list cost 0.00005 x 2
    assert summary["cost_spent"] == pytest.approx(0.0004)
    assert "WARNING: 2 calls this run (2 attempts in all) were not billed to the Go allowance (balance x2)" in report


def test_calls_naming_no_billing_source_are_flagged_and_not_counted_as_balance(tmp_path):
    report = run_command(no_billing_guard(config(tmp_path)), out_dir=tmp_path / "out", cache_dir=tmp_path / "cache",
                         client=routed_client(billed(lambda n: "" if n == 0 else "go")))

    summary = json.loads((tmp_path / "out" / "run_summary.json").read_text())
    assert summary["billing_sources"] == {"unknown": 1, "go": 4} and summary["not_go_calls"] == 1
    assert summary["allowance_spent"] == pytest.approx(0.0005) and summary["balance_spent"] == 0
    assert "1 calls named no billing source (counted at their allowance)" in report


def test_a_provider_without_the_opencode_block_prints_no_allowance(tmp_path):
    report = run_command(config(tmp_path), out_dir=tmp_path / "out", cache_dir=tmp_path / "cache",
                         client=routed_client(crew_answer))

    summary = json.loads((tmp_path / "out" / "run_summary.json").read_text())
    assert summary["allowance_spent"] == 0 and summary["billing_sources"] == {}
    assert "Go allowance" not in report and "cost unknown for 5 calls" in report


def test_the_pilot_estimates_the_full_runs_allowance_in_dollars(tmp_path):
    report = run_command(config(tmp_path), out_dir=tmp_path / "out", cache_dir=tmp_path / "cache",
                         client=routed_client(billed(lambda n: "go")), pilot=True)

    pilot = json.loads((tmp_path / "out" / "run_summary.json").read_text())["pilot"]
    assert pilot["estimated_full_allowance"] is not None and pilot["estimated_full_allowance"] > 0
    assert pilot["estimated_full_allowance"] == pilot["estimated_full_cost"]  # all billed to go
    assert f"(${pilot['estimated_full_allowance']:.2f} of it from the Go allowance)" in report


def test_the_pilot_allowance_is_unknown_when_some_calls_did_not_report_it(tmp_path):
    n = [0]

    def handler(request):  # the first call has no opencode block (as a cache entry from before the field)
        n[0] += 1
        return crew_answer(request) if n[0] == 1 else billed(lambda k: "go")(request)

    run_command(config(tmp_path), out_dir=tmp_path / "out", cache_dir=tmp_path / "cache",
                client=routed_client(handler), pilot=True)

    pilot = json.loads((tmp_path / "out" / "run_summary.json").read_text())["pilot"]
    assert pilot["estimated_full_allowance"] is None and pilot["estimated_full_cost"] is None


# --- scripts/opencode_go_usage.py --spent -------------------------------------------------------------------


def _script():
    path = Path(__file__).parent.parent / "scripts" / "opencode_go_usage.py"
    spec = importlib.util.spec_from_file_location("opencode_go_usage", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _entry(directory: Path, name: str, model: str, usage: Usage) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    request = LLMRequest(model=model, messages=[{"role": "user", "content": name}])
    response = LLMResponse(text="ok", usage=usage)
    (directory / f"{name}.json").write_text(json.dumps({"request": asdict(request), **asdict(response)}))


def test_spent_sums_the_recorded_allowance_per_model_and_says_what_it_could_not_count(tmp_path, capsys):
    go, balance = parse_usage(go_usage()), parse_usage(go_usage(source="balance", allowance=0.0))
    _entry(tmp_path, "a", "m1", go)
    _entry(tmp_path / "repeats" / "r1", "b", "m1", go)
    _entry(tmp_path, "c", "m1", balance)
    _entry(tmp_path, "d", "m2", Usage(prompt_tokens=3, cost=None))  # before the field
    (tmp_path / "e.json").write_text("{}")  # not an entry

    script = _script()
    spent = script.spent_in_cache(tmp_path)

    assert spent["m1"]["entries"] == 3 and spent["m1"]["uncounted"] == 0
    assert spent["m1"]["allowance"] == pytest.approx(0.00032) and spent["m1"]["elsewhere"] == pytest.approx(8e-05)
    assert spent["m1"]["sources"] == {"go": 2, "balance": 1}
    assert spent["m2"] == dict(entries=1, uncounted=1, allowance=0.0, elsewhere=0.0, sources={})
    assert script.print_spent(tmp_path) == 0
    out = capsys.readouterr().out
    assert "m2: $0.0000 of Go allowance over 0 entries (none recorded); 1 entries without the field, uncounted" in out
    assert "total: $0.0003 of Go allowance, $0.0001 billed elsewhere; 1 entries without the field" in out
