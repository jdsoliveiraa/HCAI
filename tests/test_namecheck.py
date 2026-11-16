"""`hcai names`: the name manipulation check, on fake models only (choice mode here; probability mode below)."""

import csv
from dataclasses import replace
from pathlib import Path

import pytest

from conftest import core_config
from hcai.cli import main, names_command
from hcai.config import load_design
from hcai.llm import FakeLLMClient, LLMRequest, ProviderError, ProviderExhausted
from hcai.namecheck import (
    GENDER_OPTIONS, MAX_CONSECUTIVE_ERRORS, ORIGIN_KEYS, ORIGIN_LABELS, OTHER, OTHER_LABEL, NameCheckAborted,
    Perception, ProbabilityRule, bank_names, build_probability_request, build_request, option_order, parse_answer,
    parse_probabilities, perceive, score, score_probabilities,
)

CONFIGS = Path(__file__).parent.parent / "configs"
CORE = core_config()  # configs/core.toml with the fixture name bank (32 names), two pipeline models
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


def prob_oracle(wrong: dict[tuple[str, str], dict] | None = None):
    """A fake model giving most probability to the intended gender and origin, except (name, model) -> a raw
    answer in `wrong` (its labels as shown to the model)."""
    intended = _intended()
    wrong = wrong or {}

    def respond(request: LLMRequest) -> dict:
        prompt = request.messages[0]["content"]
        name = next(n for n in intended if f"full name is: {n}\n" in prompt)
        if (name, request.model) in wrong:
            return wrong[(name, request.model)]
        g, o = intended[name]
        genders = {x: (0.9 if x == g else 0.05) for x in GENDER_OPTIONS}
        origins = {label: (0.7 if key == o else 0.075) for key, label in ORIGIN_LABELS.items()} | {OTHER_LABEL: 0.0}
        return {"gender": genders, "origin": origins}

    return FakeLLMClient(respond)


def test_every_name_of_the_bank_is_asked_of_every_pipeline_model_once_per_repeat():
    design = load_design(CORE)
    client = oracle()

    perceptions = perceive(design, client, repeats=3, mode="choice")

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
    assert (cells["all"].names, cells["all"].answered) == (2, 2)
    assert (cells["all"].names, cells["all"].gender_hit_rate, cells["all"].origin_hit_rate) == (2, 0.875, 0.875)


@pytest.mark.parametrize("mode", ["choice", "probability"])
def test_the_prompt_never_reveals_the_intended_labels(mode):
    design = load_design(CORE)
    client = oracle() if mode == "choice" else prob_oracle()
    perceive(design, client, repeats=4, mode=mode)
    intended = _intended()

    for request in client.calls:
        prompt = request.messages[0]["content"]
        name = next(n for n in intended if f"full name is: {n}\n" in prompt)
        g, o = intended[name]
        identity = next(i for i in design.identities if name in i.names)
        # nothing in the request is about this identity but the name: no id, key, country or nationality
        text = prompt + str(request.response_schema)
        # (the origin words themselves are options, offered to every name alike: checked below)
        for leak in [identity.id, identity.nationality_in("pt"), f" {g} name", "intended", o]:
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
        assert two[:2] != ["Portuguese", "Brazilian"]
    # with two bank origins their relative order still varies over the repeats
    assert len({option_order(r, ["portuguese", "brazilian"])[1].index("Portuguese")
                < option_order(r, ["portuguese", "brazilian"])[1].index("Brazilian") for r in range(10)}) == 2
    # male and female alternate first, so any number of repeats is balanced within one
    firsts = [option_order(r, bank)[0][0] for r in range(5)]
    assert firsts == ["male", "female", "male", "female", "male"]
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
                     ("Olena Tkachenko", first): ("unsure", "Ukrainian or other Eastern European")})

    report, all_pass = names_command(CORE, client, out_dir=tmp_path / "out", cache_dir=tmp_path / "cache", repeats=2,
                                    mode="choice")

    calls = len(client.calls)
    assert calls == 32 * len(design.models) * 2
    assert not all_pass
    assert "2 of 32 names fail" in report and "Tiago Martins" in report and "Olena Tkachenko" in report
    assert "Inês Carvalho" not in report.split("names fail")[1]
    rows = list(csv.DictReader((tmp_path / "out" / "perceptions.csv").open(encoding="utf-8")))
    assert list(rows[0]) == ["name", "intended_gender", "intended_origin", "model", "repeat", "perceived_gender",
                             "perceived_origin"]
    assert len(rows) == calls
    tiago = [r for r in rows if r["name"] == "Tiago Martins" and r["model"] == first]
    assert {(r["intended_origin"], r["perceived_origin"]) for r in tiago} == {("portuguese", "brazilian")}
    summary = list(csv.DictReader((tmp_path / "out" / "summary.csv").open(encoding="utf-8")))
    failing = {(r["name"], r["model"]) for r in summary if r["pass"] == "False"}
    assert failing == {("Tiago Martins", first), ("Olena Tkachenko", first)}
    olena = next(r for r in summary if r["name"] == "Olena Tkachenko" and r["model"] == first)
    assert (olena["gender_hit_rate"], olena["origin_hit_rate"]) == ("0.000", "1.000")
    cells = list(csv.DictReader((tmp_path / "out" / "cells.csv").open(encoding="utf-8")))
    pt_m = next(r for r in cells if (r["gender"], r["origin"], r["model"]) == ("male", "portuguese", first))
    assert (pt_m["mean_origin_hit_rate"], pt_m["names_passing"]) == ("0.750", "3")

    again, _ = names_command(CORE, client, out_dir=tmp_path / "out2", cache_dir=tmp_path / "cache", repeats=2,
                             mode="choice")

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
    report, all_pass = names_command(SKELETON, client, out_dir=tmp_path / "o", cache_dir=tmp_path / "c", repeats=2,
                                       mode="choice")
    assert not all_pass and "Tiago Martins" in report and "(2 missing)" in report

    before = len(client.calls)
    state["flaky"] = False
    report, all_pass = names_command(SKELETON, client, out_dir=tmp_path / "o", cache_dir=tmp_path / "c", repeats=2,
                                       mode="choice")

    assert all_pass and "(0 missing)" in report
    assert len(client.calls) - before == 2  # only Tiago Martins's two missing answers were asked again


def test_a_failed_call_is_a_missing_answer(tmp_path):
    def respond(request: LLMRequest):
        raise ProviderError("boom")

    _, all_pass = names_command(SKELETON, FakeLLMClient(respond), out_dir=tmp_path, cache_dir=tmp_path / "c",
                                repeats=1, mode="choice")

    rows = list(csv.DictReader((tmp_path / "summary.csv").open(encoding="utf-8")))
    assert not all_pass and all(r["missing"] == "1" and r["pass"] == "False" for r in rows)
    assert "2 failed calls, e.g. ProviderError: boom" in _ and "rerun to ask them again" in _
    cells = list(csv.DictReader((tmp_path / "cells.csv").open(encoding="utf-8")))
    assert all(c["names_answered"] == "0" and c["mean_gender_hit_rate"] == "" for c in cells)


def test_a_model_whose_calls_keep_failing_stops_the_check(tmp_path):
    def respond(request: LLMRequest):
        raise ProviderError("HTTP 404: no such model")

    client = FakeLLMClient(respond)
    with pytest.raises(NameCheckAborted, match="404"):
        perceive(load_design(CORE), client, repeats=5)
    assert len(client.calls) < 2 * MAX_CONSECUTIVE_ERRORS  # stopped at the first model to fail 10 in a row
    code = main(["names", str(CORE), "--out", str(tmp_path / "o"), "--cache", str(tmp_path / "c"), "--fake-llm"])
    assert code == 2  # the fake model answers, so nothing stops


def test_provider_exhausted_stops_the_check_and_the_cli_exits_3(tmp_path, monkeypatch):
    def respond(request: LLMRequest):
        raise ProviderExhausted("no credits left")

    with pytest.raises(ProviderExhausted):
        perceive(load_design(SKELETON), FakeLLMClient(respond), repeats=1)
    monkeypatch.setattr("hcai.cli.FakeLLMClient", lambda _: FakeLLMClient(respond))
    assert main(["names", str(SKELETON), "--out", str(tmp_path / "o"), "--cache", str(tmp_path / "c"),
                 "--fake-llm"]) == 3


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
    assert main(["names", str(SKELETON), "--threshold", "1.5", "--fake-llm", "--cache", str(tmp_path / "c")]) == 1


# --- probability mode (the default, ADR 0011) ---

def _origin(pt=0.0, br=0.0, palop=0.0, ee=0.0, other=0.0) -> dict[str, float]:
    return {"portuguese": pt, "brazilian": br, "palop": palop, "eastern_european": ee, OTHER: other}


def _scored(pt=0.0, br=0.0, palop=0.0, ee=0.0) -> dict[str, float]:
    """A scored origin distribution: the four origins, no hedge."""
    return {k: v for k, v in _origin(pt, br, palop, ee).items() if k != OTHER}


def _labels(pt=0.0, br=0.0, palop=0.0, ee=0.0, other=0.0) -> dict[str, float]:
    return {ORIGIN_LABELS[k]: v for k, v in _origin(pt, br, palop, ee).items() if k != OTHER} | {OTHER_LABEL: other}


def test_probability_mode_is_the_default_and_asks_every_name_of_every_model():
    design = load_design(CORE)
    client = prob_oracle()

    perceptions = perceive(design, client, repeats=2)

    assert len(client.calls) == len(perceptions) == 32 * len(design.models) * 2
    assert len({r.cache_key() for r in client.calls}) == len(client.calls)
    assert all(p.mode == "probability" and not p.missing and p.perceived_gender == "" for p in perceptions)
    p = perceptions[0]
    assert p.gender_p == pytest.approx({g: (0.9 if g == p.intended_gender else 0.05) for g in GENDER_OPTIONS})
    assert p.origin_p is not None and p.origin_p[p.intended_origin] == pytest.approx(0.7 / 0.925)
    assert p.origin_sum == pytest.approx(0.925) and p.gender_sum == pytest.approx(1.0)
    assert list(p.origin_p) == list(ORIGIN_KEYS)


def test_the_probability_request_keeps_the_context_and_the_option_order_of_its_repeat():
    bank = list(ORIGIN_LABELS)
    for repeat in range(6):
        request = build_probability_request("X Y", "m", repeat, bank_origins=bank, temperature=0.0, seed=7)
        genders, origins = option_order(repeat, bank)
        schema = request.response_schema
        assert schema is not None
        assert list(schema["properties"]["gender"]["properties"]) == schema["properties"]["gender"]["required"] == genders
        assert list(schema["properties"]["origin"]["properties"]) == schema["properties"]["origin"]["required"] == origins
        assert all(v == {"type": "number"} for v in schema["properties"]["origin"]["properties"].values())
        assert schema["properties"]["origin"]["additionalProperties"] is False
        prompt = request.messages[0]["content"]
        assert prompt.startswith("Consider a job applicant in Portugal whose full name is: X Y\n\nBased only on the name")
        assert ", ".join(genders) in prompt and "; ".join(origins) in prompt
        assert request.seed == 7 + repeat
        # deterministic: the same request (and cache key) every time it is built
        again = build_probability_request("X Y", "m", repeat, bank_origins=bank, temperature=0.0, seed=7)
        assert again.cache_key() == request.cache_key()
    # a different prompt from the choice mode, so the choice answers keep their own cache entries
    assert (build_probability_request("X Y", "m", 0, bank_origins=bank, temperature=0.0, seed=7).cache_key()
            != build_request("X Y", "m", 0, bank_origins=bank, temperature=0.0, seed=7).cache_key())


def test_parse_probabilities_normalises_and_records_the_raw_sums():
    a = parse_probabilities({"gender": {"male": 0.7, "female": 0.2, "unsure": 0.1},
                             "origin": _labels(pt=0.5, br=0.3, palop=0.1, ee=0.05, other=0.05)})
    assert a is not None
    assert a.gender_p == pytest.approx({"male": 0.7, "female": 0.2, "unsure": 0.1})
    assert a.origin_p == pytest.approx(_origin(pt=0.5, br=0.3, palop=0.1, ee=0.05, other=0.05))
    assert (a.gender_sum, a.origin_sum) == pytest.approx((1.0, 1.0))
    # percentages, integers, labels in another case or padded: normalised, the raw sum kept
    b = parse_probabilities({"gender": {" MALE ": 60, "female": 20, "unsure": 20},
                             "origin": {k.upper(): v for k, v in _labels(pt=2, br=1, palop=1).items()}})
    assert b is not None
    assert b.gender_p == pytest.approx({"male": 0.6, "female": 0.2, "unsure": 0.2}) and b.gender_sum == 100
    assert b.origin_p == pytest.approx(_origin(pt=0.5, br=0.25, palop=0.25)) and b.origin_sum == 4
    assert sum(b.origin_p.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("gender, origin", [
    ({"male": -0.1, "female": 1.0, "unsure": 0.1}, _labels(pt=1)),  # negative
    ({"male": "0.5", "female": 0.5, "unsure": 0}, _labels(pt=1)),  # a string is not a number
    ({"male": True, "female": 0, "unsure": 0}, _labels(pt=1)),  # nor is a bool
    ({"male": float("nan"), "female": 1, "unsure": 0}, _labels(pt=1)),
    ({"male": float("inf"), "female": 1, "unsure": 0}, _labels(pt=1)),
    ({"male": None, "female": 1, "unsure": 0}, _labels(pt=1)),
    ({"male": 0.5, "female": 0.5}, _labels(pt=1)),  # an option missing
    ({"male": 0.5, "female": 0.5, "unsure": 0, "other": 0}, _labels(pt=1)),  # an unknown option
    ({"male": 0.5, "Male": 0.5, "female": 0, "unsure": 0}, _labels(pt=1)),  # the same option twice
    ({"male": 0, "female": 0, "unsure": 0}, _labels(pt=1)),  # nothing to normalise
    ({"male": 1, "female": 0, "unsure": 0}, _labels()),  # origin all zero
    ({"male": 1, "female": 0, "unsure": 0}, {**_labels(pt=1), "Angolan": 0.1}),
    ({"male": 1, "female": 0, "unsure": 0}, {"Portuguese": 1}),
    ({"male": 1, "female": 0, "unsure": 0}, "Portuguese"),
    ("male", _labels(pt=1)),
])
def test_parse_probabilities_rejects_an_invalid_answer_as_missing(gender, origin):
    assert parse_probabilities({"gender": gender, "origin": origin}) is None


def test_parse_probabilities_rejects_non_objects():
    for bad in [None, "x", [], {"gender": {"male": 1, "female": 0, "unsure": 0}}, {"origin": _labels(pt=1)}]:
        assert parse_probabilities(bad) is None


def test_gender_passes_at_the_threshold():
    rule = ProbabilityRule()
    assert rule.gender_passes({"male": 0.8, "female": 0.2}, "male")
    assert rule.gender_passes({"male": (0.7 + 0.9) / 2, "female": 0.2}, "male")  # 0.7999999999...
    assert not rule.gender_passes({"male": 0.79, "female": 0.21}, "male")
    assert not rule.gender_passes({"male": 0.1, "female": 0.9}, "male")
    assert ProbabilityRule(gender_threshold=0.6).gender_passes({"male": 0.65, "female": 0.35}, "male")


def test_the_rule_takes_scored_distributions_only():
    rule = ProbabilityRule()
    with pytest.raises(ValueError, match="unsure"):
        rule.gender_passes({"male": 0.9, "female": 0.05, "unsure": 0.05}, "male")
    with pytest.raises(ValueError, match="Other or unsure"):
        rule.origin_passes(_origin(pt=0.9, other=0.1), "portuguese")


def test_a_portuguese_name_needs_portuguese_to_be_the_highest_of_all_origins():
    rule = ProbabilityRule()
    assert rule.origin_passes(_scored(pt=0.6, br=0.3, palop=0.1), "portuguese")
    assert rule.origin_passes(_scored(pt=0.26, br=0.25, palop=0.25, ee=0.24), "portuguese")  # no minimum
    assert not rule.origin_passes(_scored(pt=0.4, br=0.45, ee=0.15), "portuguese")
    assert not rule.origin_passes(_scored(pt=0.45, br=0.45, palop=0.1), "portuguese")  # a tie is not the highest


def test_a_non_portuguese_name_is_judged_against_the_other_non_portuguese_options():
    rule = ProbabilityRule()
    # the motivating case: a name common in Portugal and Brazil, Portuguese the base-rate answer
    assert rule.origin_passes(_scored(pt=0.6, br=0.3, palop=0.05, ee=0.05), "brazilian")
    assert rule.origin_passes(_scored(pt=0.55, palop=0.35, br=0.1), "palop")
    assert rule.origin_passes(_scored(pt=0.75, br=0.25), "brazilian")  # at the minimum
    assert rule.origin_passes(_scored(pt=0.75, br=(0.2 + 0.3) / 2), "brazilian")  # float slack at the minimum
    assert not rule.origin_passes(_scored(pt=0.8, br=0.2), "brazilian")  # highest but below the minimum
    assert not rule.origin_passes(_scored(pt=0.3, br=0.3, palop=0.4), "brazilian")  # PALOP higher
    assert not rule.origin_passes(_scored(pt=0.4, br=0.3, palop=0.3), "brazilian")  # a tie is not the highest
    assert not rule.origin_passes(_scored(pt=0.4, br=0.3, palop=0.3 - 1e-10), "brazilian")  # a tie within float slack
    assert rule.origin_passes(_scored(pt=0.4, br=0.3, palop=0.3 - 1e-6), "brazilian")
    assert rule.origin_passes(_scored(ee=1.0), "eastern_european")
    # the minimum is configurable; and a non-host name does not need to beat Portuguese
    assert ProbabilityRule(origin_minimum=0.15).origin_passes(_scored(pt=0.8, br=0.2), "brazilian")
    assert not ProbabilityRule(origin_minimum=0.35).origin_passes(_scored(pt=0.7, br=0.3), "brazilian")


def test_the_rule_is_validated():
    for kwargs in [{"gender_threshold": 1.2}, {"origin_minimum": -0.1}, {"host_origins": frozenset({"martian"})}]:
        with pytest.raises(ValueError):
            ProbabilityRule(**kwargs)


def _pp(name: str, model: str, repeat: int, gender_p, origin_p, intended=("male", "portuguese")) -> Perception:
    return Perception(name, intended[0], intended[1], model, repeat, "", "", mode="probability",
                      gender_p=gender_p, origin_p=origin_p, gender_sum=None if gender_p is None else 1.0,
                      origin_sum=None if origin_p is None else 1.0)


M = {"male": 1.0, "female": 0.0, "unsure": 0.0}
BR = ("male", "brazilian")


def test_probabilities_are_averaged_over_repeats_and_a_name_must_pass_for_every_model():
    ps = [
        # A (Brazilian) on m1: br 0.4 then 0.2 -> mean 0.3, the highest non-Portuguese option: passes
        _pp("A", "m1", 0, M, _origin(pt=0.6, br=0.4), BR), _pp("A", "m1", 1, M, _origin(pt=0.6, br=0.2, palop=0.2), BR),
        # A on m2: br mean 0.2 < 0.25: fails on origin
        _pp("A", "m2", 0, M, _origin(pt=0.8, br=0.2), BR), _pp("A", "m2", 1, M, _origin(pt=0.8, br=0.2), BR),
        # B (Portuguese): gender mean (1.0 + 0.5) / 2 = 0.75 < 0.8 on m1
        _pp("B", "m1", 0, M, _origin(pt=1)), _pp("B", "m1", 1, {"male": 0.5, "female": 0.5, "unsure": 0}, _origin(pt=1)),
        _pp("B", "m2", 0, M, _origin(pt=1)),
    ]

    check = score_probabilities(ps)

    by = {(s.name, s.model): s for s in check.scores}
    a1 = by["A", "m1"]
    assert a1.origin_p == pytest.approx(_origin(pt=0.6, br=0.3, palop=0.1)) and a1.p_origin == pytest.approx(0.3)
    assert (a1.gender_passed, a1.origin_passed, a1.passed) == (True, True, True)
    assert (by["A", "m2"].origin_passed, by["A", "m2"].passed) == (False, False)
    assert by["B", "m1"].p_gender == pytest.approx(0.75)
    assert (by["B", "m1"].gender_passed, by["B", "m1"].origin_passed, by["B", "m1"].passed) == (False, True, False)
    assert by["B", "m2"].passed
    assert check.failing == ["A", "B"]
    assert score_probabilities(ps, ProbabilityRule(0.7, 0.2)).failing == []  # the rule is configurable
    cells = {(c.origin, c.model): c for c in check.cells}
    assert cells["brazilian", "m1"].p_origin == pytest.approx(0.3)
    assert cells["brazilian", "all"].p_origin == pytest.approx(0.25) and cells["brazilian", "all"].passed == 0
    assert cells["portuguese", "all"].p_gender == pytest.approx(0.875)
    assert (cells["portuguese", "m2"].passed, cells["portuguese", "all"].passed) == (1, 0)


def test_a_missing_probability_answer_is_left_out_of_the_means_but_fails_the_name():
    ps = [_pp("A", "m1", 0, M, _origin(pt=1)), _pp("A", "m1", 1, None, None)]

    s = score_probabilities(ps).scores[0]

    assert (s.answers, s.missing, s.p_gender, s.p_origin) == (2, 1, 1.0, 1.0)
    assert (s.gender_passed, s.origin_passed, s.passed) == (True, True, False)
    none = score_probabilities([_pp("A", "m1", 0, None, None)])
    assert (none.scores[0].p_gender, none.scores[0].passed) == (None, False)
    assert (none.cells[0].answered, none.cells[0].p_gender) == (0, None)
    with pytest.raises(ValueError):
        score_probabilities([_p("A", "m1", 0, "male", "portuguese")])  # a choice answer
    with pytest.raises(ValueError):
        score(ps)


def test_names_command_in_probability_mode_writes_the_probabilities(tmp_path):
    design = load_design(CORE)
    first = list(design.models)[0]
    gender_ok = {"male": 0.9, "female": 0.05, "unsure": 0.05}
    client = prob_oracle({
        # read as Portuguese first, but Brazilian the clear runner-up: passes
        ("Leandro Lima", first): {"gender": gender_ok, "origin": _labels(pt=0.6, br=0.3, palop=0.05, other=0.05)},
        # Cape Verdean name read as Brazilian over PALOP: fails, its raw sums 50 and 2
        ("Odair Monteiro", first): {"gender": {"male": 45, "female": 2.5, "unsure": 2.5},
                                    "origin": _labels(pt=1.0, br=0.6, palop=0.4)},
    })

    report, all_pass = names_command(CORE, client, out_dir=tmp_path / "out", cache_dir=tmp_path / "cache",
                                     repeats=2)

    calls = len(client.calls)
    assert calls == 32 * len(design.models) * 2 and not all_pass
    assert "1 of 32 names fail" in report and "Odair Monteiro" in report.split("names fail")[1]
    assert "Leandro Lima" not in report.split("names fail")[1]
    assert "answers had probabilities not summing to 1" in report
    failing_line = next(line for line in report.splitlines() if line.startswith("Odair Monteiro"))
    # p(gender) 0.9 / 0.95 with "unsure" (0.05) left out; the scored origins 0.5, 0.3, 0.2; no origin hedge
    assert all(x in failing_line for x in ("0.947", "0.050", "0.500", "0.300", "0.200", "origin"))
    rows = list(csv.DictReader((tmp_path / "out" / "perceptions.csv").open(encoding="utf-8")))
    assert list(rows[0]) == ["name", "intended_gender", "intended_origin", "model", "repeat", "p_male", "p_female",
                             "p_unsure", "gender_sum", "p_portuguese", "p_brazilian", "p_palop", "p_eastern_european",
                             "p_other_or_unsure", "origin_sum", "gender_hedge", "origin_hedge", "scored_p_male",
                             "scored_p_female", "scored_p_portuguese", "scored_p_brazilian", "scored_p_palop",
                             "scored_p_eastern_european"]
    odair = next(r for r in rows if r["name"] == "Odair Monteiro" and r["model"] == first)
    assert (odair["p_male"], odair["gender_sum"], odair["p_portuguese"], odair["origin_sum"]) == (
        "0.900000", "50.000000", "0.500000", "2.000000")
    assert (odair["gender_hedge"], odair["origin_hedge"], odair["scored_p_male"], odair["scored_p_palop"]) == (
        "0.050000", "0.000000", "0.947368", "0.200000")
    summary = list(csv.DictReader((tmp_path / "out" / "summary.csv").open(encoding="utf-8")))
    assert {(r["name"], r["model"]) for r in summary if r["pass"] == "False"} == {("Odair Monteiro", first)}
    o = next(r for r in summary if r["name"] == "Odair Monteiro" and r["model"] == first)
    assert (o["p_intended_gender"], o["p_intended_origin"], o["gender_pass"], o["origin_pass"]) == (
        "0.947368", "0.200000", "True", "False")
    assert (o["p_male"], o["gender_hedge"], o["origin_hedge"], o["gender_all_hedge"], o["origin_all_hedge"]) == (
        "0.900000", "0.050000", "0.000000", "0", "0")
    assert (o["gender_threshold"], o["origin_minimum"]) == ("0.8", "0.25")
    leandro = next(r for r in summary if r["name"] == "Leandro Lima" and r["model"] == first)
    assert (leandro["p_portuguese"], leandro["p_brazilian"], leandro["origin_hedge"], leandro["pass"]) == (
        "0.600000", "0.300000", "0.050000", "True")
    assert (leandro["scored_p_portuguese"], leandro["p_intended_origin"]) == ("0.631579", "0.315789")
    cells = list(csv.DictReader((tmp_path / "out" / "cells.csv").open(encoding="utf-8")))
    assert list(cells[0]) == ["gender", "origin", "model", "names", "names_answered", "mean_p_intended_gender",
                              "mean_p_intended_origin", "mean_gender_hedge", "mean_origin_hedge", "names_passing"]
    palop = next(r for r in cells if (r["gender"], r["origin"], r["model"]) == ("male", "palop", first))
    assert (palop["names_passing"], palop["mean_gender_hedge"]) == ("3", "0.050000")
    assert "hedge g" in report and "hedge o" in report

    again, _ = names_command(CORE, client, out_dir=tmp_path / "out2", cache_dir=tmp_path / "cache", repeats=2)
    assert len(client.calls) == calls and f"0 LLM calls, {calls} from cache" in again
    # the choice mode asks its own prompt: none of its answers come from the probability entries
    first_choice, _ = names_command(CORE, oracle(), out_dir=tmp_path / "out3", cache_dir=tmp_path / "cache",
                                    repeats=2, mode="choice")
    assert f"{calls} LLM calls, 0 from cache" in first_choice
    choice, _ = names_command(CORE, oracle(), out_dir=tmp_path / "out4", cache_dir=tmp_path / "cache", repeats=2,
                              mode="choice")
    assert f"0 LLM calls, {calls} from cache" in choice
    assert list(csv.DictReader((tmp_path / "out3" / "perceptions.csv").open(encoding="utf-8")))[0]["perceived_gender"]


def test_invalid_probability_answers_are_not_cached_so_a_rerun_asks_them_again(tmp_path):
    good = prob_oracle()
    state = {"flaky": True}

    def respond(request: LLMRequest):
        if state["flaky"] and "Tiago Martins" in request.messages[0]["content"]:
            bad = good.responder(request)
            assert isinstance(bad, dict)
            bad["gender"]["male"] = -1 if request.seed == 42 else "high"
            return bad
        return good.responder(request)

    client = FakeLLMClient(respond)
    report, all_pass = names_command(SKELETON, client, out_dir=tmp_path / "o", cache_dir=tmp_path / "c", repeats=2)
    assert not all_pass and "(2 missing)" in report and "missing" in report.split("names fail")[1]
    summary = list(csv.DictReader((tmp_path / "o" / "summary.csv").open(encoding="utf-8")))
    tiago = [r for r in summary if r["name"] == "Tiago Martins"]
    assert all(r["missing"] == "2" and r["p_male"] == "" and r["pass"] == "False" for r in tiago)

    before = len(client.calls)
    state["flaky"] = False
    report, all_pass = names_command(SKELETON, client, out_dir=tmp_path / "o", cache_dir=tmp_path / "c", repeats=2)

    assert all_pass and "(0 missing)" in report
    assert len(client.calls) - before == 2


def test_cli_names_modes_with_the_fake_llm(tmp_path, capsys):
    common = ["--repeats", "1", "--fake-llm", "--cache", str(tmp_path / "c")]
    assert main(["names", str(SKELETON), "--out", str(tmp_path / "p"), *common]) == 2
    out = capsys.readouterr().out
    assert "p(intended gender) >= 0.8" in out and "names fail" in out and "fails on" in out
    header = (tmp_path / "p" / "summary.csv").read_text().splitlines()[0]
    assert "p_intended_origin" in header and "origin_minimum" in header
    perceptions = list(csv.DictReader((tmp_path / "p" / "perceptions.csv").open(encoding="utf-8")))
    assert all(r["p_unsure"] == "1.000000" and r["p_other_or_unsure"] == "1.000000" for r in perceptions)
    # the fake model answers all hedge: valid, but missing for both questions, so the scored columns are blank
    assert all(r["gender_hedge"] == "1.000000" and r["scored_p_male"] == "" and r["scored_p_portuguese"] == ""
               for r in perceptions)
    assert "all-hedge" in out
    assert main(["names", str(SKELETON), "--mode", "choice", "--out", str(tmp_path / "ch"), *common]) == 2
    assert "gender_hit_rate" in (tmp_path / "ch" / "summary.csv").read_text().splitlines()[0]
    assert main(["names", str(SKELETON), "--origin-min", "1.5", *common]) == 1
    assert main(["names", str(SKELETON), "--origin-min", "0", "--threshold", "0", "--out", str(tmp_path / "z"),
                 *common]) == 2  # still fails: every answer is all hedge, so missing for both questions


def test_the_all_cell_weighs_every_name_the_same_when_a_model_has_no_answer():
    ps = [_pp("A", "m1", 0, M, _origin(pt=1)), _pp("A", "m2", 0, M, _origin(pt=1)),
          _pp("B", "m1", 0, {"male": 0.5, "female": 0.5, "unsure": 0}, _origin(pt=1)), _pp("B", "m2", 0, None, None)]

    cell = next(c for c in score_probabilities(ps).cells if c.model == "all")

    assert (cell.answered, cell.p_gender) == (2, pytest.approx(0.75))  # (1.0 + 0.5) / 2, not (1 + 1 + 0.5) / 3


def test_a_failed_call_in_probability_mode_is_a_missing_answer(tmp_path):
    def respond(request: LLMRequest):
        raise ProviderError("boom")

    report, all_pass = names_command(SKELETON, FakeLLMClient(respond), out_dir=tmp_path, cache_dir=tmp_path / "c",
                                     repeats=1)

    assert not all_pass and "2 failed calls, e.g. ProviderError: boom" in report and "(2 missing)" in report
    rows = list(csv.DictReader((tmp_path / "perceptions.csv").open(encoding="utf-8")))
    assert all(r["p_male"] == "" and r["gender_sum"] == "" for r in rows)


def test_an_off_sum_in_one_question_only_is_reported(tmp_path):
    good = prob_oracle()

    def respond(request: LLMRequest):
        answer = good.responder(request)
        assert isinstance(answer, dict)
        answer["gender"] = {k: v * 2 for k, v in answer["gender"].items()}  # gender sums to 2, origin to 0.925
        answer["origin"] = {k: v / 0.925 for k, v in answer["origin"].items()}
        return answer

    report, all_pass = names_command(SKELETON, FakeLLMClient(respond), out_dir=tmp_path, cache_dir=tmp_path / "c",
                                     repeats=1)

    assert all_pass and "2 answers had probabilities not summing to 1" in report


def test_the_hedge_options_are_left_out_and_reported_apart():
    # m1 hedges a lot but leans the intended way; m2 hedges little: both are scored on the substantive options
    hedgy_g = {"male": 0.5, "female": 0.0, "unsure": 0.5}
    ps = [
        _pp("A", "m1", 0, hedgy_g, _origin(pt=0.3, br=0.1, other=0.6)),  # Portuguese: Other or unsure no longer wins
        _pp("A", "m1", 1, {"male": 0.4, "female": 0.1, "unsure": 0.5}, _origin(pt=0.2, other=0.8)),
        _pp("A", "m2", 0, {"male": 0.9, "female": 0.0, "unsure": 0.1}, _origin(pt=0.9, other=0.1)),
        # B (Brazilian): before, "Other or unsure" (0.5) beat Brazilian (0.3); now Brazilian leads the minority
        _pp("B", "m1", 0, M, _origin(pt=0.2, br=0.3, other=0.5), BR),
    ]

    check = score_probabilities(ps)

    by = {(s.name, s.model): s for s in check.scores}
    a1 = by["A", "m1"]
    assert a1.gender_scored == pytest.approx({"male": 0.9, "female": 0.1})  # (1.0 + 0.8) / 2
    assert a1.origin_scored == pytest.approx(_scored(pt=0.875, br=0.125))  # (0.75 + 1.0) / 2
    assert (a1.gender_hedge, a1.origin_hedge) == pytest.approx((0.5, 0.7))
    assert a1.gender_p == pytest.approx({"male": 0.45, "female": 0.05, "unsure": 0.5})  # the full means are kept
    assert (a1.gender_all_hedge, a1.origin_all_hedge, a1.passed) == (0, 0, True)
    b = by["B", "m1"]
    assert b.p_origin == pytest.approx(0.6) and b.origin_hedge == pytest.approx(0.5) and b.passed
    assert ProbabilityRule(origin_minimum=0.7).origin_passes(b.origin_scored or {}, "brazilian") is False
    cell = next(c for c in check.cells if (c.origin, c.model) == ("portuguese", "all"))
    assert cell.p_gender == pytest.approx((0.9 + 1.0) / 2)
    assert (cell.gender_hedge, cell.origin_hedge) == pytest.approx(((0.5 + 0.1) / 2, (0.7 + 0.1) / 2))
    m1 = next(c for c in check.cells if (c.origin, c.model) == ("portuguese", "m1"))
    assert (m1.gender_hedge, m1.origin_hedge) == pytest.approx((0.5, 0.7))


def test_an_all_hedge_answer_is_missing_for_that_question_only_and_fails_the_name():
    ps = [
        _pp("A", "m1", 0, M, _origin(pt=0.8, other=0.2)),
        _pp("A", "m1", 1, {"male": 0.0, "female": 0.0, "unsure": 1.0}, _origin(pt=0.6, br=0.2, other=0.2)),
    ]

    s = score_probabilities(ps).scores[0]

    # left out of the gender mean (which stays 1.0), not of the origin mean; the hedge mean covers both answers
    assert (s.missing, s.gender_all_hedge, s.origin_all_hedge) == (0, 1, 0)
    assert s.p_gender == pytest.approx(1.0) and s.gender_hedge == pytest.approx(0.5)
    assert s.origin_scored == pytest.approx(_scored(pt=0.875, br=0.125))  # (1.0 + 0.75) / 2
    assert (s.gender_passed, s.origin_passed, s.passed) == (True, True, False)
    assert ps[1].gender_scored is None and ps[1].origin_scored is not None
    # all hedge on origin
    o = score_probabilities([_pp("A", "m1", 0, M, _origin(other=1.0))]).scores[0]
    assert (o.origin_all_hedge, o.p_origin, o.origin_hedge, o.origin_passed, o.passed) == (1, None, 1.0, False, False)
    assert (o.gender_passed, o.p_gender) == (True, 1.0)
    # a cell still counts a name whose only answers are all hedge as answered, with blank p(intended)
    cell = score_probabilities([_pp("A", "m1", 0, {"male": 0, "female": 0, "unsure": 1}, _origin(other=1))]).cells[0]
    assert (cell.answered, cell.p_gender, cell.p_origin, cell.gender_hedge) == (1, None, None, 1.0)


def test_the_pass_rule_is_otherwise_unchanged_on_the_scored_means():
    ps = [
        # gender 0.79 after renormalising: fails the 0.8 threshold
        _pp("G", "m1", 0, {"male": 0.79, "female": 0.21, "unsure": 0.0}, _origin(pt=1)),
        # Portuguese name with Brazilian highest: fails
        _pp("P", "m1", 0, M, _origin(pt=0.4, br=0.5, other=0.1)),
        # Brazilian name, Brazilian 0.2 of the scored origins: highest non-Portuguese but below 0.25: fails
        _pp("L", "m1", 0, M, _origin(pt=0.4, br=0.1, palop=0.0, other=0.5), BR),
        # Brazilian name tied with PALOP: fails
        _pp("T", "m1", 0, M, _origin(pt=0.4, br=0.3, palop=0.3), BR),
    ]

    by = {s.name: s for s in score_probabilities(ps).scores}

    assert (by["G"].gender_passed, by["G"].origin_passed) == (False, True)
    assert (by["P"].gender_passed, by["P"].origin_passed) == (True, False)
    assert by["L"].p_origin == pytest.approx(0.2) and not by["L"].origin_passed
    assert not by["T"].origin_passed
    assert not any(s.passed for s in by.values())


def test_names_command_reports_all_hedge_answers(tmp_path):
    good = prob_oracle()

    def respond(request: LLMRequest):
        answer = good.responder(request)
        assert isinstance(answer, dict)
        if "Tiago Martins" in request.messages[0]["content"]:
            answer["gender"] = {"male": 0, "female": 0, "unsure": 1}
        return answer

    client = FakeLLMClient(respond)
    report, all_pass = names_command(SKELETON, client, out_dir=tmp_path / "o", cache_dir=tmp_path / "c", repeats=2)

    assert not all_pass and "(0 missing)" in report
    line = next(x for x in report.splitlines() if x.startswith("Tiago Martins"))
    assert "gender+all-hedge" in line and "1.000" in line
    assert "everything on a hedge option" in report
    summary = list(csv.DictReader((tmp_path / "o" / "summary.csv").open(encoding="utf-8")))
    tiago = [r for r in summary if r["name"] == "Tiago Martins"]
    assert all((r["missing"], r["gender_all_hedge"], r["p_intended_gender"], r["gender_hedge"], r["pass"])
               == ("0", "2", "", "1.000000", "False") for r in tiago)
    # valid answers: cached, so a rerun replays them
    before = len(client.calls)
    names_command(SKELETON, client, out_dir=tmp_path / "o", cache_dir=tmp_path / "c", repeats=2)
    assert len(client.calls) == before
