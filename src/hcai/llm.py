"""The LLM client interface, its OpenAI-compatible HTTP and fake implementations, and the request cache.

`OpenAICompatibleClient` talks to any provider's `/chat/completions` and `/models` (OpenCode Go, OpenRouter;
see `hcai.providers`). A rate-limit or usage-limit answer raises `RateLimited`; `RateLimitGuard` waits it out
or, when it cannot, raises `ProviderExhausted`, which stops the run cleanly (see ADR 0005). Every HTTP call
waits `[run] request_timeout` seconds for the answer (`http_timeout`).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from email.utils import parsedate_to_datetime
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Protocol

import httpx


def _package_version() -> str:
    try:
        from importlib.metadata import version
        return version("hcai")
    except Exception:  # noqa: BLE001 - not installed (e.g. run from a source tree)
        return "0+unknown"


USER_AGENT = f"hcai/{_package_version()}"

# Seconds a provider call waits for the answer: the default of `[run] request_timeout`. The pilot's Blinder
# answers (up to about 11,000 completion tokens) took longer than the 120 s this used to be fixed at.
DEFAULT_REQUEST_TIMEOUT = 600.0
# Seconds to open the connection: short, so an unreachable provider fails fast whatever the request timeout.
CONNECT_TIMEOUT = 20.0


def http_timeout(request_timeout: float) -> httpx.Timeout:
    """The HTTP timeouts of a provider call that waits `request_timeout` seconds for the answer.

    `request_timeout` is httpx's read timeout: the longest silence while waiting for or reading the answer
    (a chat completion is sent whole, so in practice the time the model takes to write it). Sending the
    request and waiting for a free connection get the same limit; connecting gets `CONNECT_TIMEOUT`, or
    `request_timeout` if that is shorter. Raises `ValueError` unless it is a finite number above 0.
    """
    if (isinstance(request_timeout, bool) or not isinstance(request_timeout, (int, float))
            or not 0 < request_timeout < math.inf):
        raise ValueError(f"request_timeout must be a number of seconds > 0, not {request_timeout!r}")
    return httpx.Timeout(float(request_timeout), connect=min(CONNECT_TIMEOUT, float(request_timeout)))


@dataclass(frozen=True)
class LLMRequest:
    model: str
    messages: list[dict[str, str]]
    temperature: float = 0.0
    seed: int | None = None
    response_schema: dict[str, Any] | None = None

    def cache_key(self) -> str:
        canonical = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # What the call cost in dollars, from whichever source paid it; None: no price reported (cost unknown).
    # On OpenCode Go it is what the call took from the Go allowance, or, billed elsewhere, from the balance.
    cost: float | None = 0.0
    # OpenCode Go's `usage.opencode` (None when the answer has none, e.g. OpenRouter or a cache entry from
    # before these fields): `billing_source` is "go" (the subscription allowance) or what paid instead (e.g.
    # "balance" once Extra Usage kicks in), "+"-joined when the attempts behind one output differ;
    # `allowance` the dollars taken from the Go allowance (0 when billed elsewhere); `list_cost` the price
    # at list rates.
    billing_source: str | None = None
    allowance: float | None = None
    list_cost: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Usage:
        """A `Usage` from its `asdict` form; missing fields (an older cache entry) take their defaults.

        An entry cached before `billing_source` was recorded (#11 Part B) with a cost of 0 is an OpenCode Go
        call whose 0 was misread as a price: its cost is unknown (None), not free.
        """
        known = {f.name for f in fields(cls)}
        usage = cls(**{k: v for k, v in data.items() if k in known})
        if "billing_source" not in data and usage.cost == 0:
            usage = replace(usage, cost=None)
        return usage


@dataclass(frozen=True)
class LLMResponse:
    text: str
    parsed: Any = None
    usage: Usage = field(default_factory=Usage)
    cached: bool = False  # served from the cache: no call was made and nothing was spent


class LLMClient(Protocol):
    def complete(self, request: LLMRequest) -> LLMResponse: ...


def _parse_if_schema(request: LLMRequest, text: str) -> Any:
    if request.response_schema is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


class FakeLLMClient:
    """Scriptable client for tests: `responder` maps a request to text or a JSON-able object.

    Every request is recorded in `calls`. Token counts are word counts so they are non-zero.
    """

    def __init__(self, responder: Callable[[LLMRequest], str | dict | list]):
        self.responder = responder
        self.calls: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        out = self.responder(request)
        text = out if isinstance(out, str) else json.dumps(out)
        prompt_words = sum(len(m["content"].split()) for m in request.messages)
        usage = Usage(prompt_tokens=prompt_words, completion_tokens=len(text.split()))
        return LLMResponse(text=text, parsed=_parse_if_schema(request, text), usage=usage)


def add_costs(costs: list[float | None]) -> float | None:
    """The sum of `costs`, or None (unknown) when any of them is unknown."""
    return None if any(c is None for c in costs) else sum(c for c in costs if c is not None)


def add_usage(usages: list[Usage]) -> Usage:
    """The usage of several attempts behind one output: tokens summed, costs summed (None if any is unknown).

    The Go allowance and list cost are None when no attempt reported them, and are otherwise summed like
    costs (an attempt without them, e.g. from an older cache entry, makes them unknown).
    """
    reported = [u for u in usages if u.billing_source is not None]
    sources = sorted({s for u in reported for s in (u.billing_source or "").split("+") if s})
    return Usage(
        prompt_tokens=sum(u.prompt_tokens for u in usages),
        completion_tokens=sum(u.completion_tokens for u in usages),
        cost=add_costs([u.cost for u in usages]),
        billing_source="+".join(sources) if reported else None,
        allowance=add_costs([u.allowance for u in usages]) if reported else None,
        list_cost=add_costs([u.list_cost for u in usages]) if reported else None,
    )


def _number(value: Any) -> float | None:
    try:
        return None if value is None or isinstance(value, bool) else float(value)
    except (TypeError, ValueError):
        return None


def parse_usage(u: Any, *, subscription: bool = False) -> Usage:
    """The `Usage` of a chat-completions `usage` block.

    With OpenCode Go's `usage.opencode`, the cost is what the call took from its billing source: the
    `allowance` when `billing_source` is "go", else the reported `cost`, or `list_cost` when that is 0 or
    missing (the pay-as-you-go balance is charged at list price). Without it, a missing `cost` is unknown
    (None), and so is a `cost` of 0 from a `subscription` provider, which prices nothing per call.
    """
    u = u if isinstance(u, dict) else {}
    tokens: dict[str, Any] = dict(prompt_tokens=int(u.get("prompt_tokens", 0) or 0),
                                  completion_tokens=int(u.get("completion_tokens", 0) or 0))
    cost = _number(u.get("cost"))
    oc = u.get("opencode")
    if isinstance(oc, dict):
        source = str(oc.get("billing_source") or "unknown")
        allowance, list_cost = _number(oc.get("allowance")), _number(oc.get("list_cost"))
        if source == "go":
            return Usage(**tokens, cost=allowance, billing_source=source, allowance=allowance, list_cost=list_cost)
        if source == "unknown":  # billed somewhere unnamed: what the allowance says it took, flagged as not go
            return Usage(**tokens, cost=allowance if allowance is not None else cost, billing_source=source,
                         allowance=allowance, list_cost=list_cost)
        # billed elsewhere: its price, or the list price; a 0 with no list price is unknown, not free
        return Usage(**tokens, cost=cost if cost else list_cost, billing_source=source, allowance=0.0,
                     list_cost=list_cost)
    if subscription and not cost:
        cost = None
    return Usage(**tokens, cost=cost)


class RateLimited(Exception):
    """The provider refused a call for its rate or usage limit; `retry_after` is its hint in seconds, if any."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class ProviderExhausted(BaseException):
    """The provider cannot serve more calls for now (a limit that outlasts the wait, or no credits left).

    A `BaseException`, like `KeyboardInterrupt`, so it passes the `except Exception` handlers that record a
    failed agent call as missing data: the run stops instead of filling the results with missing rows.
    Every call completed before it is in the cache, so rerunning the same command resumes with no repeats.
    """


class ProviderError(Exception):
    """The provider answered with an error that is not a rate or usage limit."""


class MissingSession(ProviderError):
    """The provider refused the call for a missing or rejected session header (OpenCode Go's HTTP 400
    `MissingSessionID`). Handled like any `ProviderError` (retried, then a missing row); the client warns once
    per run and `MeteredClient` counts it, so the cause is not lost among the missing rows."""


_LIMIT_WORDS = ("rate limit", "rate-limit", "usage limit", "too many requests", "quota")


def _retry_after(headers: httpx.Headers) -> float | None:
    value = headers.get("retry-after")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
    except (TypeError, ValueError):
        return None


class OpenAICompatibleClient:
    """Chat-completions client for an OpenAI-compatible provider at `base_url` (e.g. `https://host/v1`).

    The key is sent only in the Authorization header: it is never part of a request, response or error text
    (an error body that echoes it is redacted). `usage_include` asks for the price in the usage block
    (OpenRouter); a provider that reports none, or a `subscription` one that reports 0 without OpenCode Go's
    `usage.opencode` block, gives `Usage.cost = None` (see `parse_usage`).

    Every call also sends `User-Agent: hcai/<version>` and the `headers` given (e.g. OpenCode Go's
    `x-opencode-session`). They are transport details: never part of the `LLMRequest`, so never in the cache
    key, the cache or the results, and their values are masked in error text like the key.

    `timeout` is the seconds a call waits for the answer (`[run] request_timeout`, see `http_timeout`). It is
    set on every request, so it also holds on a client passed in as `http`, and like the headers it is no
    part of the `LLMRequest` or of the request body.
    """

    def __init__(self, base_url: str, api_key: str, *, name: str = "", usage_include: bool = False,
                 subscription: bool = False, http: httpx.Client | None = None,
                 timeout: float = DEFAULT_REQUEST_TIMEOUT,
                 headers: dict[str, str] | None = None, log: Callable[[str], None] | None = None):
        api_key = api_key.strip()  # a stray space or newline would make httpx echo the key in its error
        if not api_key:
            raise RuntimeError(f"no API key for provider {name or base_url}")
        self.name = name or base_url
        self.base_url = base_url.rstrip("/")
        self.usage_include = usage_include
        self.subscription = subscription
        self._timeout = http_timeout(timeout)
        self._http = http or httpx.Client(timeout=self._timeout)
        self._key = api_key
        self._log = log or _stderr
        self._warned_session = False  # the MissingSessionID warning is printed once per client (one per run)
        extra = {k: v for k, v in (headers or {}).items() if k.lower() not in ("authorization", "user-agent")}
        # masked in error text after the key, longest first; values too short to be secrets are left alone
        self._masked = sorted((v for v in extra.values() if len(v) >= 8), key=len, reverse=True)
        self._headers = {"User-Agent": USER_AGENT, **extra, "Authorization": f"Bearer {api_key}"}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r}, {self.base_url!r})"

    def _redact(self, text: str, limit: int = 500) -> str:
        """`text` with the key masked, then cut to `limit` characters (in that order, so no part of it survives)."""
        text = text.replace(self._key, "***")  # the key first, so no header value can split it
        for value in self._masked:
            text = text.replace(value, "<header>")
        return text[:limit]

    def _send(self, method: str, url: str, **kw: Any) -> httpx.Response:
        """One HTTP request; a transport failure is re-raised as `ProviderError` with its message redacted."""
        try:
            return self._http.request(method, url, headers=self._headers, timeout=self._timeout, **kw)
        except Exception as e:  # noqa: BLE001 - httpx/h11 messages can quote the Authorization header
            hint = (f" (no answer within [run] request_timeout = {self._timeout.read:g}s)"
                    if isinstance(e, httpx.ReadTimeout) else "")
            raise ProviderError(f"{self.name}: {type(e).__name__}: {self._redact(str(e))}{hint}") from None

    def _check(self, resp: httpx.Response) -> Any:
        """The JSON body of a successful answer; raises `RateLimited`, `ProviderExhausted` or `ProviderError`."""
        text = self._redact(resp.text)
        lowered = text.lower()
        if resp.status_code == 401:
            raise ProviderExhausted(f"{self.name}: HTTP 401 (the API key was refused): {text}")
        if resp.status_code == 402:
            raise ProviderExhausted(f"{self.name}: HTTP 402 (no credits or usage left): {text}")
        if resp.status_code == 429 or (resp.status_code >= 400 and any(w in lowered for w in _LIMIT_WORDS)):
            raise RateLimited(f"{self.name}: HTTP {resp.status_code}: {text}", _retry_after(resp.headers))
        if resp.status_code == 400 and "missingsessionid" in lowered:
            if not self._warned_session:
                self._warned_session = True
                self._log(f"WARNING: {self.name}: the session header is missing or rejected (HTTP 400 "
                          f"MissingSessionID); these calls fail and their rows are recorded as missing. "
                          f"This warning is shown once per run.")
            raise MissingSession(f"{self.name}: HTTP {resp.status_code}: {text}")
        if resp.status_code >= 400:
            raise ProviderError(f"{self.name}: HTTP {resp.status_code}: {text}")
        try:
            data = resp.json()
        except json.JSONDecodeError as e:
            raise ProviderError(f"{self.name}: answer is not JSON: {text}") from e
        error = data.get("error") if isinstance(data, dict) else None
        if error:  # some providers report an upstream failure with HTTP 200
            message = self._redact(json.dumps(error, ensure_ascii=False))
            code = error.get("code") if isinstance(error, dict) else None
            if code == 402:
                raise ProviderExhausted(f"{self.name}: {message}")
            if code == 429 or any(w in message.lower() for w in _LIMIT_WORDS):
                raise RateLimited(f"{self.name}: {message}", _retry_after(resp.headers))
            raise ProviderError(f"{self.name}: {message}")
        return data

    def complete(self, request: LLMRequest) -> LLMResponse:
        body: dict[str, Any] = {
            "model": request.model,
            "messages": request.messages,
            "temperature": request.temperature,
        }
        if self.usage_include:
            body["usage"] = {"include": True}
        if request.seed is not None:
            body["seed"] = request.seed
        if request.response_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "strict": True, "schema": request.response_schema},
            }
        data = self._check(self._send("POST", f"{self.base_url}/chat/completions", json=body))
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise ProviderError(f"{self.name}: no message in the answer: {self._redact(str(data), 300)}") from e
        usage = parse_usage(data.get("usage"), subscription=self.subscription)
        return LLMResponse(text=text, parsed=_parse_if_schema(request, text), usage=usage)

    def list_models(self) -> list[str]:
        """The model ids the provider offers (`GET <base_url>/models`)."""
        data = self._check(self._send("GET", f"{self.base_url}/models"))
        return sorted(str(m["id"]) for m in data.get("data", []) if isinstance(m, dict) and "id" in m)


class OpenRouterClient(OpenAICompatibleClient):
    """OpenRouter chat-completions client. Reads the key from OPENROUTER_API_KEY."""

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str | None = None, http: httpx.Client | None = None,
                 timeout: float = DEFAULT_REQUEST_TIMEOUT):
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        super().__init__(self.BASE_URL, key, name="openrouter", usage_include=True, http=http, timeout=timeout)


def _stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


class RateLimitGuard:
    """Wraps a provider client: waits out `RateLimited` answers, then repeats the identical call.

    It waits the provider's `Retry-After` if given (at least `min_wait` seconds, so a `Retry-After: 0` cannot
    spin), else `backoff` seconds doubling up to `max_backoff`.
    When one call would wait more than `max_wait` seconds in all (or the provider asks for longer), it
    raises `ProviderExhausted`: the run stops cleanly and a rerun resumes from the cache.
    """

    def __init__(self, inner: LLMClient, *, backoff: float = 30.0, max_backoff: float = 900.0,
                 max_wait: float = 6 * 3600.0, sleep: Callable[[float], None] = time.sleep,
                 log: Callable[[str], None] = _stderr, name: str = "", min_wait: float = 5.0):
        self.inner = inner
        self.min_wait = min_wait
        self.backoff = backoff
        self.max_backoff = max_backoff
        self.max_wait = max_wait
        self.sleep = sleep
        self.log = log
        self.name = name

    def complete(self, request: LLMRequest) -> LLMResponse:
        waited = 0.0
        failures = 0
        while True:
            try:
                return self.inner.complete(request)
            except RateLimited as e:
                delay = max(self.min_wait, e.retry_after) if e.retry_after is not None else min(
                    self.backoff * 2 ** failures, self.max_backoff)
                if waited + delay > self.max_wait:
                    raise ProviderExhausted(
                        f"{self.name or 'provider'}: still limited after waiting {waited:.0f}s "
                        f"(next wait {delay:.0f}s exceeds [run] rate_limit_max_wait = {self.max_wait:.0f}s): {e}"
                    ) from e
                self.log(f"{self.name or 'provider'} rate/usage limit on {request.model}: "
                         f"waiting {delay:.0f}s before trying again")
                self.sleep(delay)
                waited += delay
                failures += 1


class MeteredClient:
    """Wraps a client and counts the calls it makes and what they cost, per model.

    Placed below the cache, it measures what a run actually spends: cache hits never reach it. On OpenCode
    Go it also sums the dollars taken from the Go allowance (`allowance`) and from elsewhere (`balance`: the
    cost of calls whose `billing_source` is not "go"), and counts the calls per billing source.
    """

    def __init__(self, inner: LLMClient):
        self.inner = inner
        self.calls: dict[str, int] = {}
        self.cost: dict[str, float] = {}  # the known prices only
        self.unpriced: dict[str, int] = {}  # calls the provider gave no price for (cost unknown)
        self.missing_session: dict[str, int] = {}  # calls refused for a missing session header (MissingSession)
        self.allowance: dict[str, float] = {}  # dollars taken from the Go allowance
        self.balance: dict[str, float] = {}  # dollars billed outside the Go allowance (known prices only)
        self.billing: dict[str, dict[str, int]] = {}  # calls per billing source (OpenCode Go only)

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls[request.model] = self.calls.get(request.model, 0) + 1  # a call that raises counts too
        try:
            response = self.inner.complete(request)
        except MissingSession:
            self.missing_session[request.model] = self.missing_session.get(request.model, 0) + 1
            raise
        if response.usage.cost is None:
            self.unpriced[request.model] = self.unpriced.get(request.model, 0) + 1
        else:
            self.cost[request.model] = self.cost.get(request.model, 0.0) + response.usage.cost
        u = response.usage
        if u.billing_source is not None:
            by_source = self.billing.setdefault(request.model, {})
            by_source[u.billing_source] = by_source.get(u.billing_source, 0) + 1
            if u.allowance is not None:
                self.allowance[request.model] = self.allowance.get(request.model, 0.0) + u.allowance
            # an "unknown" source is counted at its allowance (above), not as balance
            if u.billing_source not in ("go", "unknown") and u.cost is not None:
                self.balance[request.model] = self.balance.get(request.model, 0.0) + u.cost - (u.allowance or 0.0)
        return response


class CachedClient:
    """Wraps a client; responses are stored on disk keyed by the hash of the full request.

    By default every response is cached, malformed ones included, so a rerun replays exactly what the
    first run saw (and costs nothing); the pipeline asks a malformed answer again with a different request
    (see `hcai.pipeline.Retry` and ADR 0002). With `cache_unparseable=False` a structured answer that
    failed to parse is not cached, so a rerun draws a fresh one (the corpus generator, at temperature > 0).
    Calls that raise are never cached.
    """

    def __init__(self, inner: LLMClient, cache_dir: str | Path, *, cache_unparseable: bool = True):
        self.inner = inner
        self.cache_unparseable = cache_unparseable
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def complete(self, request: LLMRequest) -> LLMResponse:
        path = self.cache_dir / f"{request.cache_key()}.json"
        if path.exists():
            cached = json.loads(path.read_text())
            return LLMResponse(text=cached["text"], parsed=cached["parsed"], usage=Usage.from_dict(cached["usage"]), cached=True)
        response = self.inner.complete(request)
        if not self.cache_unparseable and request.response_schema is not None and response.parsed is None:
            return response
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"request": asdict(request), **asdict(response)}, ensure_ascii=False))
        tmp.replace(path)
        return response
