"""LLM providers: which OpenAI-compatible endpoint serves each model, their keys, and the `.env` file.

Two providers are built in: `opencode-go` (the default: the OpenCode Go subscription) and `openrouter` (the
pay-per-token fallback). A config can override either one's `base_url` or `key_env`, or add another, under
`[providers.<name>]`; `[model_providers]` names the provider of each model (unlisted models use the default),
so switching a model to OpenRouter is one line. Keys are read from the environment, which `load_dotenv`
fills from a gitignored `.env` file; they are only ever sent in the Authorization header.

A provider may also send extra HTTP headers (`headers`) and a session header (`session_header`) whose value
is `SESSION_ID`, one random id per process: OpenCode Go refuses calls without `x-opencode-session`. Neither
is part of a request, so neither reaches the cache key, the cache or the results. The same holds for
`request_timeout`, the seconds every provider's client waits for an answer (`[run] request_timeout`).
"""

from __future__ import annotations

import os
import re
import threading
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from hcai.llm import (
    DEFAULT_REQUEST_TIMEOUT, LLMClient, LLMRequest, LLMResponse, OpenAICompatibleClient, RateLimitGuard,
    http_timeout,
)

BILLING = ("metered", "subscription")
_HEADER_NAME = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+")  # an RFC 9110 token


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    key_env: str  # the environment variable holding the API key
    # "metered": each call has a price the provider reports; "subscription": a flat fee, no per-call price
    billing: str = "metered"
    usage_include: bool = False  # ask for the price in the usage block (OpenRouter's `usage: {include: true}`)
    # extra HTTP headers sent with every call (name, value); kept out of the repr, as one may hold a token
    headers: tuple[tuple[str, str], ...] = field(default=(), repr=False)
    session_header: str | None = None  # a header that carries `SESSION_ID` (OpenCode Go: x-opencode-session)

    def __post_init__(self) -> None:
        if self.billing not in BILLING:
            raise ValueError(f"provider {self.name}: billing must be one of {BILLING}, not {self.billing!r}")
        if not self.base_url.startswith(("https://", "http://")):
            raise ValueError(f"provider {self.name}: base_url must be an http(s) URL, not {self.base_url!r}")
        if self.session_header is not None and not isinstance(self.session_header, str):
            raise ValueError(f"provider {self.name}: session_header must be a header name")
        if not all(isinstance(v, str) for _, v in self.headers):
            raise ValueError(f"provider {self.name}: header values must be strings")
        names = [n for n, _ in self.headers] + ([self.session_header] if self.session_header is not None else [])
        for n in names:
            if not isinstance(n, str) or not _HEADER_NAME.fullmatch(n):
                raise ValueError(f"provider {self.name}: {n!r} is not a valid HTTP header name")
        lowered = [n.lower() for n in names]
        if any(n in ("authorization", "user-agent") for n in lowered):
            raise ValueError(f"provider {self.name}: headers cannot set Authorization or User-Agent")
        if len(set(lowered)) != len(lowered):
            raise ValueError(f"provider {self.name}: a header is set twice (names are case-insensitive)")

    def http_headers(self) -> dict[str, str]:
        """The extra headers of every call: `headers`, plus `session_header: SESSION_ID` if it has one."""
        out = dict(self.headers)
        if self.session_header:
            out[self.session_header] = SESSION_ID
        return out


# One stable session id per process (so per run), sent in a provider's `session_header`. Not a secret, but a
# transport detail: it is never written to the cache, the results or the logs (and masked in error text).
SESSION_ID = str(uuid.uuid4())


DEFAULT_PROVIDER = "opencode-go"
BUILTIN_PROVIDERS: dict[str, Provider] = {
    "opencode-go": Provider("opencode-go", "https://opencode.ai/zen/go/v1", "OPENCODE_API_KEY", billing="subscription",
                            session_header="x-opencode-session"),
    "openrouter": Provider("openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", billing="metered",
                           usage_include=True),
}


class MissingKey(RuntimeError):
    """An API key the command needs is not set (names the variables, never a value)."""


def load_dotenv(path: str | Path = ".env", environ: os._Environ[str] | dict[str, str] | None = None) -> list[str]:
    """Set the variables of a `.env` file that are not already in the environment; returns their names.

    Lines are `NAME=value` or `export NAME=value`; blank lines and `#` comments are skipped, and a value may
    be quoted with ' or ". A missing file is not an error; a variable set to an empty string counts as
    unset. Values are never printed.
    """
    env = os.environ if environ is None else environ
    path = Path(path)
    if not path.is_file():
        return []
    loaded = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        name, sep, value = line.partition("=")
        name, value = name.strip(), value.strip()
        if not sep or not name.isidentifier():
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        elif " #" in value:  # an inline comment after an unquoted value
            value = value.split(" #", 1)[0].rstrip()
        if not env.get(name):  # unset or empty: an empty export would otherwise hide the file's key
            env[name] = value
            loaded.append(name)
    return loaded


class ProviderRouter:
    """Sends each request to the client of its model's provider; clients are made on first use."""

    def __init__(self, provider_of: Callable[[str], Provider], make: Callable[[Provider], LLMClient]):
        self._provider_of = provider_of
        self._make = make
        self._clients: dict[str, LLMClient] = {}
        self._lock = threading.Lock()  # workers asking at once still share one client (and rate-limit pause)

    def client_for(self, model: str) -> LLMClient:
        provider = self._provider_of(model)
        with self._lock:
            if provider.name not in self._clients:
                self._clients[provider.name] = self._make(provider)
            return self._clients[provider.name]

    def complete(self, request: LLMRequest) -> LLMResponse:
        return self.client_for(request.model).complete(request)


def check_keys(providers: Iterable[Provider], environ: Mapping[str, str] | None = None) -> None:
    """Raise `MissingKey` naming every provider whose key variable is unset or empty."""
    env = os.environ if environ is None else environ
    missing = sorted({f"{p.key_env} (provider {p.name})" for p in providers if not env.get(p.key_env)})
    if missing:
        raise MissingKey("API key not set: " + ", ".join(missing)
                         + "; export it or put it in a .env file (see .env.example)")


def http_client(provider: Provider, environ: Mapping[str, str] | None = None,
                http: httpx.Client | None = None, *,
                request_timeout: float = DEFAULT_REQUEST_TIMEOUT) -> OpenAICompatibleClient:
    """The raw HTTP client of `provider`, with its key from the environment.

    Each of its calls waits `request_timeout` seconds for the answer (`[run] request_timeout`).
    """
    env = os.environ if environ is None else environ
    check_keys([provider], env)
    return OpenAICompatibleClient(provider.base_url, env[provider.key_env], name=provider.name,
                                  usage_include=provider.usage_include,
                                  subscription=provider.billing == "subscription", http=http,
                                  timeout=request_timeout, headers=provider.http_headers())


def provider_client(
    provider_of: Callable[[str], Provider], models: Iterable[str], *, environ: Mapping[str, str] | None = None,
    http: httpx.Client | None = None, backoff: float = 30.0, max_backoff: float = 900.0,
    max_wait: float = 6 * 3600.0, sleep: Callable[[float], None] | None = None,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
) -> ProviderRouter:
    """A client for `models`, each routed to its provider behind a `RateLimitGuard`.

    The keys of every provider `models` need are checked up front, so a missing key fails before any call.
    Every provider's calls wait `request_timeout` seconds for the answer (`[run] request_timeout`).
    """
    env = os.environ if environ is None else environ
    check_keys({provider_of(m) for m in models}, env)
    http_timeout(request_timeout)  # an invalid one fails here, not at the first call (clients are made lazily)

    def make(provider: Provider) -> LLMClient:
        return RateLimitGuard(http_client(provider, env, http, request_timeout=request_timeout), backoff=backoff,
                              max_backoff=max_backoff, max_wait=max_wait, sleep=sleep, name=provider.name)

    return ProviderRouter(provider_of, make)
