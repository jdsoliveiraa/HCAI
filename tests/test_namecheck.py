"""`hcai names`: the name manipulation check, on fake models only."""

import csv
from dataclasses import replace
from pathlib import Path

import pytest

from hcai.cli import main, names_command
from hcai.config import load_design
from hcai.llm import FakeLLMClient, LLMRequest, ProviderError
from hcai.namecheck import (
    GENDER_OPTIONS, ORIGIN_LABELS, OTHER, OTHER_LABEL, Perception, bank_names, build_request, option_order,
    parse_answer, perceive, score,
)

CONFIGS = Path(__file__).parent.parent / "configs"
CORE = CONFIGS / "core.toml"  # the full name bank, two pipeline models
SKELETON = CONFIGS / "skeleton.toml"  # [[identities]]: Portuguese and Brazilian only


def _intended() -> dict[str, tuple[str, str]]:
    return {name: (g, o) for name, g, o in bank_names(load_design(CORE))}


def oracle(wrong: dict[tuple[str, str], tuple[str, str]] | None = None):
    """A fake model that perceives every name as intended, except (name, model) -> (gender, origin label) in `wrong`."""
    intended = _intended()
    wrong = wrong or {}

    def respond(request: LLMRequest) -> dict:
        prompt = request.messages[0]["content"]
        name = next(n for n in intended if f"full name is: {n}\n" in prompt)
        if (name, request.model) in wrong:
            g, o = wrong[(name, request.model)]
            return {"gender": g, "origin": o}
        g, o = intended[name]
        return {"gender": g, "origin": ORIGIN_LABELS[o]}

    return FakeLLMClient(respond)


def test_every_name_of_the_bank_is_asked_of_every_pipeline_model_once_per_repeat():
    design = load_design(CORE)
    client = oracle()

    perceptions = perceive(design, client, repeats=3)

    names = bank_names(design)
    assert len(names) == 32
    assert len(client.calls) == len(perceptions) == 32 * len(design.models) * 3
    assert {r.model for r in client.calls} == set(design.models)  # pipeline models only: no generator, no judge
    keys = [r.cache_key() for r in client.calls]
    assert len(set(keys)) == len(keys)  # each repeat is its own request (its own sample and cache entry)
    assert all(p.perceived_gender == p.intended_gender and p.perceived_origin == p.intended_origin
               for p in perceptions)


def test_parse_answer_maps_labels_to_bank_origins_and_rejects_anything_else():
    assert parse_answer({"gender": "female", "origin": "Cape Verdean or other PALOP"}) == ("female", "palop")
    assert parse_answer({"gender": " Male ", "origin": "ukrainian or other eastern european"}) == (
        "male", "eastern_european")
    assert parse_answer({"gender": "unsure", "origin": OTHER_LABEL}) == ("unsure", OTHER)
    for bad in [None, "male", {"gender": "male"}, {"gender": "man", "origin": "Portuguese"},
                {"gender": "male", "origin": "Angolan"}, {"gender": 1, "origin": "Portuguese"}]:
        assert parse_answer(bad) == ("", "")


def _p(name: str, model: str, repeat: int, g: str, o: str, intended: tuple[str, str] = ("male", "portuguese")):
    return Perception(name, intended[0], intended[1], model, repeat, g, o)


def test_hit_rates_and_pass_flags_per_name_and_model():
    ps = [
        # A: 5/5 gender, 4/5 origin on m1 -> passes at 0.8
        *[_p("A", "m1", r, "male", "portuguese" if r else "brazilian") for r in range(5)],
        # A on m2: 3/5 origin -> fails
        *[_p("A", "m2", r, "male", "portuguese" if r < 3 else OTHER) for r in range(5)],
        # B: gender 3/5, female names
        *[_p("B", "m1", r, "female" if r < 3 else "unsure", "brazilian", ("female", "brazilian")) for r in range(5)],
        *[_p("B", "m2", r, "female", "brazilian", ("female", "brazilian")) for r in range(5)],
    ]

    check = score(ps, threshold=0.8)

    by = {(s.name, s.model): s for s in check.scores}
    assert (by["A", "m1"].gender_hit_rate, by["A", "m1"].origin_hit_rate, by["A", "m1"].passed) == (1.0, 0.8, True)
    assert (by["A", "m2"].origin_hit_rate, by["A", "m2"].passed) == (0.6, False)
    assert (by["B", "m1"].gender_hit_rate, by["B", "m1"].passed) == (0.6, False)
    assert by["B", "m2"].passed
    assert check.failing == ["A", "B"]
    assert score(ps, threshold=0.6).failing == []  # the threshold is configurable


def test_a_missing_answer_is_left_out_of_the_rates_but_fails_the_name():
    ps = [_p("A", "m1", 0, "male", "portuguese"), _p("A", "m1", 1, "", "")]

    s = score(ps).scores[0]

    assert (s.answers, s.missing, s.gender_hit_rate, s.origin_hit_rate, s.passed) == (2, 1, 1.0, 1.0, False)
    none = score([_p("A", "m1", 0, "", "")]).scores[0]
    assert (none.gender_hit_rate, none.origin_hit_rate, none.passed) == (None, None, False)


def test_cells_report_the_mean_hit_rates_of_their_names():
    pt_m = ("male", "portuguese")
    ps = [_p("A", "m1", 0, "male", "portuguese", pt_m), _p("A", "m1", 1, "male", "brazilian", pt_m),
          _p("C", "m1", 0, "male", "portuguese", pt_m), _p("C", "m1", 1, "female", "portuguese", pt_m),
          _p("A", "m2", 0, "male", "portuguese", pt_m), _p("C", "m2", 0, "male", "portuguese", pt_m)]

    cells = {c.model: c for c in score(ps, threshold=0.5).cells if (c.gender, c.origin) == pt_m}

    assert (cells["m1"].gender_hit_rate, cells["m1"].origin_hit_rate, cells["m1"].passed) == (0.75, 0.75, 2)
    assert (cells["m2"].gender_hit_rate, cells["m2"].origin_hit_rate) == (1.0, 1.0)
    assert (cells["all"].names, cells["all"].gender_hit_rate, cells["all"].origin_hit_rate) == (2, 0.875, 0.875)


def test_the_prompt_never_reveals_the_intended_labels():
    design = load_design(CORE)
    client = oracle()
    perceive(design, client, repeats=4)
    intended = _intended()

    for request in client.calls:
        prompt = request.messages[0]["content"]
        name = next(n for n in intended if f"full name is: {n}\n" in prompt)
        g, o = intended[name]
        identity = next(i for i in design.identities if name in i.names)
        # nothing in the request is about this identity but the name: no id, key, country or nationality
        text = prompt + str(request.response_schema)
        for leak in [identity.id, o, identity.nationality_in("pt"), f" {g} name", "intended"]:
            assert leak not in text
        # the gender and origin words appear only as options in the lists (the context sentence has none)
        context = prompt.split("\n")[0].replace(name, "<NAME>")
        assert context == "Consider a job applicant in Portugal whose full name is: <NAME>"
        assert not any(w in context.lower() for w in [*GENDER_OPTIONS, "brazil", "cape verde", "ukrain", "palop"])
    # at each repeat, the request is identical for every name but the name itself
    by_repeat: dict[int, set] = {}
    for request in client.calls:
        prompt = request.messages[0]["content"]
        name = next(n for n in intended if f"full name is: {n}\n" in prompt)
        stripped = replace(request, messages=[{"role": "user", "content": prompt.replace(name, "<NAME>")}])
        by_repeat.setdefault(request.seed or 0, set()).add((stripped.model, stripped.cache_key()))
    assert all(len(keys) == len(design.models) for keys in by_repeat.values())


def test_origin_options_are_never_in_bank_order():
    bank = list(ORIGIN_LABELS)  # the bank's origin order: portuguese, brazilian, palop, eastern_european
    for repeat in range(200):
        genders, origins = option_order(repeat, bank)
        labels = origins[:-1]
        assert labels != [ORIGIN_LABELS[k] for k in bank]
        assert sorted(labels) == sorted(ORIGIN_LABELS.values()) and origins[-1] == OTHER_LABEL
        assert sorted(genders[:2]) == ["female", "male"] and genders[-1] == "unsure"
        two = option_order(repeat, ["portuguese", "brazilian"])[1]
        assert two.index("Portuguese") > two.index("Brazilian")
    # the order varies across repeats (position bias is spread), and is the same for a repeat on every call
    assert len({tuple(option_order(r, bank)[1]) for r in range(10)}) > 1
    assert option_order(3, bank) == option_order(3, bank)
    request = build_request("X Y", "m", 2, bank_origins=bank, temperature=0.0, seed=42)
    assert request.response_schema is not None
    assert request.response_schema["properties"]["origin"]["enum"] == option_order(2, bank)[1]
    assert request.seed == 44


def test_names_command_writes_the_tables_and_a_second_run_makes_no_calls(tmp_path):
    design = load_design(CORE)
    first = list(design.models)[0]
    client = oracle({("Tiago Martins", first): ("male", "Brazilian"),
                     ("Olena Melnyk", first): ("unsure", "Ukrainian or other Eastern European")})

    report, all_pass = names_command(CORE, client, out_dir=tmp_path / "out", cache_dir=tmp_path / "cache", repeats=2)

    calls = len(client.calls)
    assert calls == 32 * len(design.models) * 2
    assert not all_pass
    assert "2 of 32 names fail" in report and "Tiago Martins" in report and "Olena Melnyk" in report
    assert "Inês Martins" not in report.split("names fail")[1]
    rows = list(csv.DictReader((tmp_path / "out" / "perceptions.csv").open(encoding="utf-8")))
    assert list(rows[0]) == ["name", "intended_gender", "intended_origin", "model", "repeat", "perceived_gender",
                             "perceived_origin"]
    assert len(rows) == calls
    tiago = [r for r in rows if r["name"] == "Tiago Martins" and r["model"] == first]
    assert {(r["intended_origin"], r["perceived_origin"]) for r in tiago} == {("portuguese", "brazilian")}
    summary = list(csv.DictReader((tmp_path / "out" / "summary.csv").open(encoding="utf-8")))
    failing = {(r["name"], r["model"]) for r in summary if r["pass"] == "False"}
    assert failing == {("Tiago Martins", first), ("Olena Melnyk", first)}
    olena = next(r for r in summary if r["name"] == "Olena Melnyk" and r["model"] == first)
    assert (olena["gender_hit_rate"], olena["origin_hit_rate"]) == ("0.000", "1.000")
    cells = list(csv.DictReader((tmp_path / "out" / "cells.csv").open(encoding="utf-8")))
    pt_m = next(r for r in cells if (r["gender"], r["origin"], r["model"]) == ("male", "portuguese", first))
    assert (pt_m["mean_origin_hit_rate"], pt_m["names_passing"]) == ("0.750", "3")

    again, _ = names_command(CORE, client, out_dir=tmp_path / "out2", cache_dir=tmp_path / "cache", repeats=2)

    assert len(client.calls) == calls  # every answer came from the cache
    assert f"0 LLM calls, {calls} from cache" in again
    assert (tmp_path / "out2" / "perceptions.csv").read_text() == (tmp_path / "out" / "perceptions.csv").read_text()


def test_missing_answers_are_not_cached_so_a_rerun_asks_them_again(tmp_path):
    good = oracle()
    state = {"flaky": True}

    def respond(request: LLMRequest):
        if state["flaky"] and "Tiago Martins" in request.messages[0]["content"]:
            return "not json" if request.seed == 42 else {"gender": "male", "origin": "Spanish"}
        return good.responder(request)

    client = FakeLLMClient(respond)
    report, all_pass = names_command(SKELETON, client, out_dir=tmp_path / "o", cache_dir=tmp_path / "c", repeats=2)
    assert not all_pass and "Tiago Martins" in report and "(2 missing)" in report

    before = len(client.calls)
    state["flaky"] = False
    report, all_pass = names_command(SKELETON, client, out_dir=tmp_path / "o", cache_dir=tmp_path / "c", repeats=2)

    assert all_pass and "(0 missing)" in report
    assert len(client.calls) - before == 2  # only Tiago Martins's two missing answers were asked again


def test_a_failed_call_is_a_missing_answer(tmp_path):
    def respond(request: LLMRequest):
        raise ProviderError("boom")

    _, all_pass = names_command(SKELETON, FakeLLMClient(respond), out_dir=tmp_path, cache_dir=tmp_path / "c",
                                repeats=1)

    rows = list(csv.DictReader((tmp_path / "summary.csv").open(encoding="utf-8")))
    assert not all_pass and all(r["missing"] == "1" and r["pass"] == "False" for r in rows)


def test_an_identity_whose_origin_has_no_option_is_an_error(tmp_path):
    config = tmp_path / "c.toml"
    config.write_text(SKELETON.read_text().replace('origin = "brazilian"', 'origin = "martian"'))
    with pytest.raises(ValueError, match="martian"):
        perceive(load_design(config), oracle(), repeats=1)


def test_cli_names_with_the_fake_llm_reports_every_name_failing(tmp_path, capsys):
    code = main(["names", str(SKELETON), "--repeats", "2", "--threshold", "0.5", "--out", str(tmp_path / "out"),
                 "--cache", str(tmp_path / "cache")] + ["--fake-llm"])

    out = capsys.readouterr().out
    assert code == 2  # some name fails
    assert "names fail" in out and (tmp_path / "out" / "summary.csv").exists()
    assert main(["names", str(SKELETON), "--repeats", "0", "--fake-llm", "--cache", str(tmp_path / "c")]) == 1
