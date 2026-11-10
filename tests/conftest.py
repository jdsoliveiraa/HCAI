import socket
from pathlib import Path

import pytest

from hcai.config import Design, Identity
from hcai.llm import FakeLLMClient, LLMRequest

JOB_AD = "Data Analyst. Mandatory: SQL, Python. Desirable: dashboards."
BASE_CV = "$name\nNationality: $nationality\nRight to work in Portugal.\n\nSQL, Python, 3 years of experience."


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail any test that tries to open a real network connection."""

    def refuse(*args, **kwargs):
        raise RuntimeError("tests must not touch the network; inject FakeLLMClient")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


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


def scores_by_name(scores: dict[str, int]):
    """Fake responder: the Evaluator's score depends on which name appears in the prompt.

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
        raise AssertionError("no known name in prompt")

    return respond


@pytest.fixture
def fake_client() -> FakeLLMClient:
    return FakeLLMClient(scores_by_name({"João Silva": 70, "Ana Souza": 65}))
