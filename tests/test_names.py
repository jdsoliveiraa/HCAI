import json

import pytest

from conftest import FIXTURE_BANK
from hcai.names import GENDERS, load_name_bank, name_parts

# The fixture, not data/names/name_bank.toml: the real bank's names are the researcher's to pick.
BANK = FIXTURE_BANK


def test_name_bank_has_several_names_per_gender_and_origin_with_sources():
    bank = load_name_bank(BANK)

    assert {o.key for o in bank.origins} == {"portuguese", "brazilian", "palop", "eastern_european"}
    for origin in bank.origins:
        for gender in GENDERS:
            names = origin.names[gender]
            assert len(names) >= 3, (origin.key, gender)
            assert len(set(names)) == len(names)
        assert origin.sources and all(s.title and s.url for s in origin.sources)
        assert origin.nationality["en"] and origin.nationality["pt"]


def test_no_name_is_shared_between_identities():
    bank = load_name_bank(BANK)

    all_names = [n for o in bank.origins for g in GENDERS for n in o.names[g]]
    assert len(all_names) == len(set(all_names))


def _write_bank(tmp_path, pt_male: list[str], pt_female: list[str]):
    """A two-origin bank: the given Portuguese names and one Brazilian name per gender."""
    def origin(key: str, code: str, male: list[str], female: list[str]) -> str:
        return (f'[origins.{key}]\ncode = "{code}"\ncountry = "{key}"\nnationality = {{ en = "X", pt = "X" }}\n'
                f'male = {json.dumps(male, ensure_ascii=False)}\nfemale = {json.dumps(female, ensure_ascii=False)}\n'
                'sources = [{ title = "t", url = "u" }]\n')

    path = tmp_path / "bank.toml"
    path.write_text(origin("portuguese", "pt", pt_male, pt_female)
                    + origin("brazilian", "br", ["Leandro Lima"], ["Fernanda Araújo"]))
    return path


def test_every_name_has_its_own_first_name_and_surname_across_the_bank():
    bank = load_name_bank(BANK)

    names = [n for o in bank.origins for g in GENDERS for n in o.names[g]]
    firsts = [name_parts(n)[0].casefold() for n in names]
    lasts = [name_parts(n)[1].casefold() for n in names]
    assert len(set(firsts)) == len(firsts) and len(set(lasts)) == len(lasts)
    assert name_parts("Ana Clara Souza") == ("Ana", "Souza")


def test_a_bank_with_unique_parts_loads(tmp_path):
    bank = load_name_bank(_write_bank(tmp_path, ["Tiago Martins"], ["Inês Carvalho"]))
    assert [o.key for o in bank.origins] == ["portuguese", "brazilian"]


def test_a_repeated_surname_is_rejected_naming_the_duplicates(tmp_path):
    # the same surname for a male and a female name of one origin (the old surname pool)
    with pytest.raises(ValueError, match=r"surname 'Martins' in Tiago Martins, Inês Martins") as e:
        load_name_bank(_write_bank(tmp_path, ["Tiago Martins"], ["Inês Martins"]))
    assert "first name" not in str(e.value).split("repeated:")[1]
    # across origins too, ignoring case
    with pytest.raises(ValueError, match=r"surname 'LIMA' in Rui LIMA, Leandro Lima"):
        load_name_bank(_write_bank(tmp_path, ["Rui LIMA"], ["Inês Carvalho"]))


def test_a_repeated_first_name_is_rejected_naming_the_duplicates(tmp_path):
    with pytest.raises(ValueError, match=r"first name 'Leandro' in Leandro Martins, Leandro Lima") as e:
        load_name_bank(_write_bank(tmp_path, ["Leandro Martins"], ["Inês Carvalho"]))
    assert "surname" not in str(e.value).split("repeated:")[1]
    # every duplicate is named at once, with the bank's path
    with pytest.raises(ValueError) as e:
        load_name_bank(_write_bank(tmp_path, ["Leandro Martins"], ["Inês Martins"]))
    assert str(tmp_path / "bank.toml") in str(e.value)
    assert "surname 'Martins'" in str(e.value) and "first name 'Leandro'" in str(e.value)


def test_a_name_without_a_surname_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="needs a first name and a surname"):
        load_name_bank(_write_bank(tmp_path, ["Tiago"], ["Inês Carvalho"]))


def test_the_config_loader_rejects_a_bank_with_repeated_parts(tmp_path):
    from conftest import core_config
    from hcai.config import load_design

    bank = _write_bank(tmp_path, ["Tiago Martins"], ["Inês Martins"])
    config = core_config(tmp_path)
    config.write_text(config.read_text().replace(str(FIXTURE_BANK), str(bank)))
    with pytest.raises(ValueError, match="surname 'Martins'"):
        load_design(config)
