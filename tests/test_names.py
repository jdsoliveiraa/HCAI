from pathlib import Path

from hcai.names import GENDERS, load_name_bank

BANK = Path(__file__).parents[1] / "data" / "names" / "name_bank.toml"


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
