import socket
import tempfile
from pathlib import Path

import pytest

from hcai.config import Design, Identity
from hcai.llm import FakeLLMClient, LLMRequest

ROOT = Path(__file__).parents[1]
# A name bank with the real bank's structure and placeholder names, every first name and surname used once: tests
# do not depend on the names the researcher picks for data/names/name_bank.toml.
FIXTURE_BANK = Path(__file__).parent / "fixtures" / "name_bank.toml"


def fixture_config(name: str, directory: Path | None = None, **replacements: str) -> Path:
    """configs/<name>.toml written to `directory` (a new temporary one by default) with the fixture name bank and
    its other ../data paths made absolute; each `old=new` of `replacements` must occur in it."""
    directory = Path(tempfile.mkdtemp(prefix=f"hcai-{name}-")) if directory is None else directory
    text = ((ROOT / "configs" / f"{name}.toml").read_text()
            .replace('name_bank = "../data/names/name_bank.toml"', f'name_bank = "{FIXTURE_BANK}"')
            .replace('"../data/', f'"{ROOT}/data/'))
    assert str(FIXTURE_BANK) in text
    for old, new in replacements.items():
        assert old in text, old
        text = text.replace(old, new)
    config = directory / f"{name}.toml"
    config.write_text(text)
    return config


def core_config(directory: Path | None = None, **replacements: str) -> Path:
    """configs/core.toml with the fixture name bank (see `fixture_config`)."""
    return fixture_config("core", directory, **replacements)


JOB_AD = "Data Analyst. Mandatory: SQL, Python. Desirable: dashboards."
BASE_CV = "$name\nNationality: $nationality\n$work_eligibility\n\nSQL, Python, 3 years of experience."


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail any test that tries to open a real network connection."""

    def refuse(*args, **kwargs):
        raise RuntimeError("tests must not touch the network; inject FakeLLMClient")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


@pytest.fixture(autouse=True)
def no_real_keys(monkeypatch):
    """Keep the developer's keys and .env out of tests: a forgotten mock then fails with MissingKey."""
    for name in ("OPENCODE_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("hcai.cli.load_dotenv", lambda *a, **k: [])


@pytest.fixture(autouse=True)
def crewai_storage(monkeypatch, tmp_path):
    """CrewAI keeps a local store of its last kickoff under the user data dir; keep tests out of it."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    role = root / "data_analyst"
    (role / "high").mkdir(parents=True)
    (role / "job_ad.en.md").write_text(JOB_AD)
    (role / "high" / "cv1.en.md").write_text(BASE_CV)
    return root


@pytest.fixture
def design(corpus_dir: Path) -> Design:
    return Design(
        roles=["data_analyst"],
        quality_levels=["high"],
        base_cvs_per_role=1,
        identities=[
            Identity(id="pt_m", gender="male", origin="portuguese", names=["João Silva"], nationality="Portuguese"),
            Identity(id="br_f", gender="female", origin="brazilian", names=["Ana Souza"], nationality="Brazilian"),
        ],
        languages=["en"],
        conditions=["C0"],
        models=["model-a"],
        corpus_dir=corpus_dir,
    )


def scores_by_name(scores: dict[str, int], default: int | None = None):
    """Fake responder: the Evaluator's score depends on which name appears in the prompt (else `default`).

    The Screener passes everyone and the Ranker keeps the order it was shown.
    """

    def respond(request: LLMRequest) -> dict:
        props = (request.response_schema or {}).get("properties", {})
        if "passed" in props:
            return {"passed": True, "reason": "meets the mandatory requirements"}
        if "ranking" in props:
            return {"ranking": props["ranking"]["items"]["enum"], "rationale": "kept the order shown"}
        text = " ".join(m["content"] for m in request.messages)
        for name, score in scores.items():
            if name in text:
                return {"score": score, "rationale": f"seen {name}"}
        if default is not None:
            return {"score": default, "rationale": "default score"}
        raise AssertionError("no known name in prompt")

    return respond


@pytest.fixture
def fake_client() -> FakeLLMClient:
    return FakeLLMClient(scores_by_name({"João Silva": 70, "Ana Souza": 65}))
