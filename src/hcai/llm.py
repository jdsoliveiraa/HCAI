"""The LLM client interface, its OpenAI-compatible HTTP and fake implementations, and the request cache.

`OpenAICompatibleClient` talks to any provider's `/chat/completions` and `/models` (OpenCode Go, OpenRouter;
see `hcai.providers`). A rate-limit or usage-limit answer raises `RateLimited`; `RateLimitGuard` waits it out
or, when it cannot, raises `ProviderExhausted`, which stops the run cleanly (see ADR 0005). Every HTTP call
waits `[run] request_timeout` seconds for the answer (`http_timeout`).

Run guards (ADR 0005, 2026-11-23 amendment): `BillingGuard` stops a run on the first fresh answer billed outside
`[run] allowed_billing` (after caching it), `FailureGuard` stops it after `[run] max_consecutive_failures`
provider errors in a row on one model, and `CachedClient` drops a cache file that does not parse.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Protocol

import httpx

from hcai import runlog

logger = logging.getLogger("hcai.llm")


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
# The most `[run] request_timeout` may be: a day. A longer wait is a mistake, and a number of seconds too large
# for the platform's clock makes every call fail with an OverflowError instead.
MAX_REQUEST_TIMEOUT = 24 * 3600.0
# Seconds to open the connection (the TCP connection, then again the TLS handshake): short, so an unreachable
# provider fails fast whatever the request timeout.
CONNECT_TIMEOUT = 20.0


def http_timeout(request_timeout: float) -> httpx.Timeout:
    """The HTTP timeouts of a provider call that waits `request_timeout` seconds for the answer.

    `request_timeout` is httpx's read timeout: the longest silence while waiting for or reading the answer
    (a chat completion is sent whole, so in practice the time the model takes to write it). Sending the
    request and waiting for a free connection get the same limit; connecting gets `CONNECT_TIMEOUT`, or
    `request_timeout` if that is shorter. Raises `ValueError` unless it is a number above 0 and at most
    `MAX_REQUEST_TIMEOUT`.
    """
    if (isinstance(request_timeout, bool) or not isinstance(request_timeout, (int, float))
            or not 0 < request_timeout <= MAX_REQUEST_TIMEOUT):
        raise ValueError(f"request_timeout must be a number of seconds > 0 and at most {MAX_REQUEST_TIMEOUT:.0f} "
                         f"(a day), not {request_timeout!r}")
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

    kind = "limit"  # what stopped the run, as recorded in progress.json


class BillingStop(ProviderExhausted):
    """A call was billed outside `[run] allowed_billing` (e.g. OpenCode Go's pay-as-you-go `balance`)."""

    kind = "billing"

    def __init__(self, message: str, *, model: str, billing_source: str):
        super().__init__(message)
        self.model = model
        self.billing_source = billing_source


class TooManyFailures(ProviderExhausted):
    """`[run] max_consecutive_failures` provider errors in a row on one model: an outage, a wrong id or key."""

    kind = "failures"

    def __init__(self, message: str, *, model: str, failures: int):
        super().__init__(message)
        self.model = model
        self.failures = failures


class ProviderError(Exception):
    """The provider answered with an error that is not a rate or usage limit.

    `transient`: a server error (HTTP 5xx) or a transport failure (timeout, connection reset), which the pipeline
    retries after `[run] transient_backoff` rather than `retry_backoff`; `timeout`: the call timed out.
    """

    def __init__(self, message: str, *, transient: bool = False, timeout: bool = False, usage: Usage | None = None):
        super().__init__(message)
        self.transient = transient or timeout
        self.timeout = timeout
        self.usage = usage  # what a billed answer that is still an error (an empty one) cost, if reported


class MissingSession(ProviderError):
    """The provider refused the call for a missing or rejected session header (OpenCode Go's HTTP 400
    `MissingSessionID`). Handled like any `ProviderError` (retried, then a missing row); the client warns once
    per run and `MeteredClient` counts it, so the cause is not lost among the missing rows."""


# What makes an HTTP error (status >= 400, or an `error` in a 200 answer) a rate or usage limit (`RateLimited`,
# waited out) rather than a failed call. Matched case-insensitively at the start of a word, CamelCase and
# snake_case error codes split into words (`SubscriptionUsageLimitError` -> "subscription usage limit error").
# Strong phrases name a limit on their own and are matched in every string of the body and in the code fields.
_STRONG_LIMIT = (r"rate[- ]?limit", r"usage limit", r"too many requests", r"quota", r"allowance",
                 r"insufficient (?:credit|balance|fund|quota)", r"(?:out of|no) credit",
                 r"credits? (?:exhausted|exceeded|depleted|remaining|balance|limit)", r"credit balance",
                 r"subscription (?:usage|limit)")
# Weak words name a limit only in the message or a code of a 4xx answer that is not about the request itself
# (`_REQUEST_SHAPE`): "Monthly limit exceeded" is a limit, "Input exceeds the maximum limit of 32768 tokens" or
# "Insufficient permissions" is not, as it fails the same way on every retry and waiting would hold the run up.
_WEAK_LIMIT = ("limit", "exceeded", "insufficient", "credit", "subscription")
_REQUEST_SHAPE = ("token", "max", "size", "length", "context", "input", "output", "prompt", "payload", "range",
                  "permission", "card", "available", "support", "param", "field", "schema", "invalid")
_STRONG_RE = re.compile(r"\b(?:" + "|".join(_STRONG_LIMIT) + ")", re.IGNORECASE)
_WEAK_RE = re.compile(r"\b(?:" + "|".join(_WEAK_LIMIT) + ")", re.IGNORECASE)
_SHAPE_RE = re.compile(r"\b(?:" + "|".join(_REQUEST_SHAPE) + ")", re.IGNORECASE)
# quoted text, where a validation error echoes the request back; an apostrophe inside a word ("You've") is no quote
_QUOTED = re.compile(r"(?<!\w)'[^']*'(?!\w)|\"[^\"]*\"|`[^`]*`|<<<.*?>>>")
_CODE_FIELDS = ("code", "type", "name", "status")


def _words(code: str) -> str:
    """An error code as words: `SubscriptionUsageLimitError` / `insufficient_quota` -> "...Usage Limit Error"."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", code).replace("_", " ").replace("-", " ")


def _prose(text: str) -> str:
    """`text` without quoted text and snake_case identifiers (`credit_limit`): what the provider itself wrote."""
    return " ".join(w for w in _QUOTED.sub(" ", text).split() if "_" not in w)


def _strings(value: Any) -> list[str]:
    """Every string in a JSON value, keys left out."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def _error_fields(body: Any, text: str) -> tuple[str, list[Any]]:
    """The message of an error answer and its code-like fields (strings or numbers).

    The message is `error.message` (or `error` itself when it is a string, or a top-level `message`); without
    one, every string of the JSON body, or the body text when it is not JSON.
    """
    if body is None:
        return text, []
    error = body.get("error") if isinstance(body, dict) else None
    sources = [d for d in (error, body) if isinstance(d, dict)]
    codes = [v for d in sources for k in _CODE_FIELDS for v in [d.get(k)] if isinstance(v, (str, int))]
    message: Any = error if isinstance(error, str) else None
    for d in sources:
        if message is None and isinstance(d.get("message"), str):
            message = d["message"]
    if message is None:
        message = " ".join(_strings(error if error is not None else body))
    return message, codes


def is_limit_error(body: Any, text: str, status: int = 400) -> bool:
    """Whether an error answer (HTTP `status`; `body` its JSON or None, `text` its text) is a rate or usage limit.

    - An error code 429 (e.g. `{"error": {"code": 429}}`) is one.
    - A strong phrase (`_STRONG_LIMIT`: rate limit, usage limit, quota, allowance, insufficient credits, ...) in
      any string of the body (e.g. `detail`, `metadata.raw`) or in a code field is one.
    - A weak word (`_WEAK_LIMIT`: limit, exceeded, insufficient, credit, subscription) in the message or a code
      is one only for a 4xx answer that says nothing about the request itself (`_REQUEST_SHAPE`: tokens,
      maximum, size, length, context, permissions, ...). A 5xx answer stays a transient server error.
    Quoted text ('...', "...", `...`) and snake_case identifiers in the body are left out first, as that is where a
    validation error echoes the request ("Invalid value for 'quota'"). The trade-off (ADR 0005): a limit worded only
    with weak words next to a request-shape word, or only inside quotes, is missed and handled as a failed call
    (retried, then missing; `FailureGuard` stops the run if it persists).
    """
    message, codes = _error_fields(body, text)
    if any(c == 429 and not isinstance(c, bool) for c in codes):
        return True
    words = [_words(c) for c in codes if isinstance(c, str)]
    everything = [_prose(t) for t in (_strings(body) if body is not None else [text])]
    if any(_STRONG_RE.search(t) for t in everything + words):
        return True
    if status >= 500:
        return False
    return any(_WEAK_RE.search(t) and not _SHAPE_RE.search(t) for t in [_words(_prose(message)), *words])


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
        self._timeout = http_timeout(timeout)  # sent with every request (`_send`), whichever client carries it
        self._http = http or httpx.Client()
        self._key = api_key
        self._log = log or _warn
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
        started = time.monotonic()
        try:
            return self._http.request(method, url, headers=self._headers, timeout=self._timeout, **kw)
        except Exception as e:  # noqa: BLE001 - httpx/h11 messages can quote the Authorization header
            hint = (f" (no answer within [run] request_timeout = {self._timeout.read:.10g}s)"
                    if isinstance(e, httpx.ReadTimeout) else "")
            raise ProviderError(f"{self.name}: {type(e).__name__}: {self._redact(str(e))}{hint}",
                                transient=isinstance(e, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)),
                                timeout=isinstance(e, httpx.TimeoutException)) from None
        finally:
            runlog.note(http_s=time.monotonic() - started)

    def _check(self, resp: httpx.Response) -> Any:
        """The JSON body of a successful answer; raises `RateLimited`, `ProviderExhausted` or `ProviderError`."""
        runlog.note(http_status=resp.status_code)
        text = self._redact(resp.text)
        lowered = text.lower()
        if resp.status_code == 401:
            raise ProviderExhausted(f"{self.name}: HTTP 401 (the API key was refused): {text}")
        if resp.status_code == 402:
            raise ProviderExhausted(f"{self.name}: HTTP 402 (no credits or usage left): {text}")
        if resp.status_code == 429 or (resp.status_code >= 400 and is_limit_error(_json_or_none(resp), text, resp.status_code)):
            raise RateLimited(f"{self.name}: HTTP {resp.status_code}: {text}", _retry_after(resp.headers))
        if resp.status_code == 400 and "missingsessionid" in lowered:
            if not self._warned_session:
                self._warned_session = True
                self._log(f"{self.name}: the session header is missing or rejected (HTTP 400 "
                          f"MissingSessionID); these calls fail and their rows are recorded as missing. "
                          f"This warning is shown once per run.")
            raise MissingSession(f"{self.name}: HTTP {resp.status_code}: {text}")
        if resp.status_code >= 400:
            raise ProviderError(f"{self.name}: HTTP {resp.status_code}: {text}", transient=resp.status_code >= 500)
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
            if code == 429 or is_limit_error(data, message, code if isinstance(code, int) else 400):
                raise RateLimited(f"{self.name}: {message}", _retry_after(resp.headers))
            raise ProviderError(f"{self.name}: {message}",
                                transient=isinstance(code, int) and not isinstance(code, bool) and code >= 500)
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
            choice = data["choices"][0]
            text = choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise ProviderError(f"{self.name}: no message in the answer: {self._redact(str(data), 300)}") from e
        finish = choice.get("finish_reason") if isinstance(choice, dict) else None
        runlog.note(finish_reason=finish if isinstance(finish, str) else None)
        if not text.strip() and finish != "stop":
            # cut off (e.g. "length") or no reason given: not an answer, so not cached and asked again on a rerun;
            # an empty answer the model finished ("stop") is an answer, cached and validated like any (ADR 0002)
            raise ProviderError(f"{self.name}: empty answer (finish_reason {finish!r}) from {request.model}",
                                usage=parse_usage(data.get("usage"), subscription=self.subscription))
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


def _warn(message: str) -> None:
    logger.warning(message)


def _json_or_none(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


class RateLimitGuard:
    """Wraps a provider client: waits out `RateLimited` answers, then repeats the identical call.

    It waits the provider's `Retry-After` if given (at least `min_wait` seconds, so a `Retry-After: 0` cannot
    spin), else `backoff` seconds doubling up to `max_backoff`.
    When one call would wait more than `max_wait` seconds in all (or the provider asks for longer), it
    raises `ProviderExhausted`: the run stops cleanly and a rerun resumes from the cache.
    """

    def __init__(self, inner: LLMClient, *, backoff: float = 30.0, max_backoff: float = 900.0,
                 max_wait: float = 6 * 3600.0, sleep: Callable[[float], None] = time.sleep,
                 log: Callable[[str], None] = _warn, name: str = "", min_wait: float = 5.0):
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
                resume = datetime.now().astimezone() + timedelta(seconds=delay)
                self.log(f"{self.name or 'provider'} rate/usage limit on {request.model}: waiting {delay:.0f}s, "
                         f"resuming at {resume.isoformat(timespec='seconds')} (waited {waited:.0f}s of at most "
                         f"[run] rate_limit_max_wait = {self.max_wait:.0f}s for this call): {str(e)[:200]}")
                runlog.rate_limited(request.model, wait=delay, resume_at=resume, waited=waited, error=e)
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
        self.timed_out: dict[str, int] = {}  # calls that got no answer within `[run] request_timeout`
        self.failed: dict[str, int] = {}  # calls that raised a provider error (timeouts and missing sessions too)

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls[request.model] = self.calls.get(request.model, 0) + 1  # a call that raises counts too
        try:
            response = self.inner.complete(request)
        except ProviderError as e:
            self.failed[request.model] = self.failed.get(request.model, 0) + 1
            if isinstance(e, MissingSession):
                self.missing_session[request.model] = self.missing_session.get(request.model, 0) + 1
            if e.timeout:
                self.timed_out[request.model] = self.timed_out.get(request.model, 0) + 1
            if e.usage is not None:  # billed although it is no answer (an empty one): spent all the same
                self._spend(request.model, e.usage)
            raise
        self._spend(request.model, response.usage)
        return response

    def _spend(self, model: str, u: Usage) -> None:
        if u.cost is None:
            self.unpriced[model] = self.unpriced.get(model, 0) + 1
        else:
            self.cost[model] = self.cost.get(model, 0.0) + u.cost
        if u.billing_source is not None:
            by_source = self.billing.setdefault(model, {})
            by_source[u.billing_source] = by_source.get(u.billing_source, 0) + 1
            if u.allowance is not None:
                self.allowance[model] = self.allowance.get(model, 0.0) + u.allowance
            # an "unknown" source is counted at its allowance (above), not as balance
            if u.billing_source not in ("go", "unknown") and u.cost is not None:
                self.balance[model] = self.balance.get(model, 0.0) + u.cost - (u.allowance or 0.0)


class CachedClient:
    """Wraps a client; responses are stored on disk keyed by the hash of the full request.

    By default every response is cached, malformed ones included, so a rerun replays exactly what the
    first run saw (and costs nothing); the pipeline asks a malformed answer again with a different request
    (see `hcai.pipeline.Retry` and ADR 0002). With `cache_unparseable=False` a structured answer that
    failed to parse is not cached, so a rerun draws a fresh one (the corpus generator, at temperature > 0).
    Calls that raise are never cached. A cache file that does not parse (zero bytes, cut short) is deleted with
    a warning and counts as a miss.
    """

    def __init__(self, inner: LLMClient, cache_dir: str | Path, *, cache_unparseable: bool = True,
                 log: Callable[[str], None] = _warn):
        self.inner = inner
        self.log = log
        self.cache_unparseable = cache_unparseable
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def complete(self, request: LLMRequest) -> LLMResponse:
        key = request.cache_key()
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            try:
                cached = json.loads(path.read_text())
                response = LLMResponse(text=cached["text"], parsed=cached["parsed"],
                                       usage=Usage.from_dict(cached["usage"]), cached=True)
            except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError, AttributeError) as e:
                # a file cut short (zero bytes, a crash mid-write) or not an entry: a miss, dropped and asked again
                self.log(f"cache file {path} does not parse ({type(e).__name__}); deleted, asking again")
                path.unlink(missing_ok=True)
            else:
                runlog.note(cache="hit", key=key[:16])
                return response
        runlog.note(cache="miss", key=key[:16])
        response = self.inner.complete(request)
        if not self.cache_unparseable and request.response_schema is not None and response.parsed is None:
            return response
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"request": asdict(request), **asdict(response)}, ensure_ascii=False))
        tmp.replace(path)
        return response


class BillingGuard:
    """Wraps a `CachedClient`: stops the run on the first fresh answer billed outside `allowed`.

    OpenCode Go names what paid for each call (`Usage.billing_source`): "go" while the subscription allowance
    lasts, e.g. "balance" once Extra Usage bills the pay-as-you-go balance. The paid answer is already in the
    cache (the guard sits above it), so nothing paid for is lost; then `BillingStop` stops the run cleanly and
    a rerun resumes from the cache. Answers served from the cache were paid by an earlier run and pass, and so
    does any answer that names no billing source (e.g. OpenRouter). An empty `allowed` checks nothing. An
    error the provider billed (an empty answer, `ProviderError.usage`) is checked too, though it is not cached.
    """

    def __init__(self, inner: LLMClient, allowed: tuple[str, ...] | list[str]):
        self.inner = inner
        self.allowed = tuple(allowed)

    def complete(self, request: LLMRequest) -> LLMResponse:
        try:
            response = self.inner.complete(request)
        except ProviderError as e:  # an error the provider billed (an empty answer): checked like an answer
            if e.usage is not None:
                self._check(request, e.usage.billing_source, cached=False)
            raise
        self._check(request, response.usage.billing_source, cached=response.cached)
        return response

    def _check(self, request: LLMRequest, source: str | None, *, cached: bool) -> None:
        if self.allowed and not cached and source is not None and source not in self.allowed:
            logger.warning(f"billing guard: a call to {request.model} was billed to {source!r}, outside [run] "
                           f"allowed_billing = {list(self.allowed)}; stopping the run")
            raise BillingStop(
                f"{request.model}: a call was billed to {source!r}, not to [run] allowed_billing = "
                f"{list(self.allowed)}; its answer is cached. Stopped so nothing more is paid for that way; add "
                f"{source!r} to allowed_billing to go on paying for it",
                model=request.model, billing_source=source)


class FailureGuard:
    """Wraps a provider client (below the cache): stops the run after `limit` provider errors in a row on one model.

    Each failed call counts once: repeating the identical request (a retry of a raised call) adds nothing, so
    the count is of distinct calls, and one call that keeps failing (an answer that always times out, a
    content filter) adds 1, not 3 or 4. Any answer from that model resets its count. At `limit`,
    `TooManyFailures` stops the run cleanly with the last error, instead of recording row after row as missing
    through an outage, a withdrawn model id or a refused session. Failed calls are not cached, so a rerun asks
    them again. `limit = 0` never stops.
    """

    def __init__(self, inner: LLMClient, limit: int):
        self.inner = inner
        self.limit = limit
        self.in_a_row: dict[str, int] = {}
        self._last_failed: dict[str, str] = {}  # per model, the cache key of the last failed request
        self._warned_at: dict[str, int] = {}  # per model, the count last warned about (one warning per count)

    def _warned(self, model: str, n: int) -> bool:
        if self._warned_at.get(model) == n:
            return True
        self._warned_at[model] = n
        return False

    def complete(self, request: LLMRequest) -> LLMResponse:
        try:
            response = self.inner.complete(request)
        except ProviderError as e:
            key = request.cache_key()
            if self._last_failed.get(request.model) != key:
                self.in_a_row[request.model] = self.in_a_row.get(request.model, 0) + 1
                self._last_failed[request.model] = key
            n = self.in_a_row[request.model]
            if n >= 5 and not self._warned(request.model, n):
                logger.warning(f"{request.model}: {n} failed calls in a row"
                               + (f" (the run stops at [run] max_consecutive_failures = {self.limit})"
                                  if self.limit else "") + f"; last: {str(e)[:200]}")
            if self.limit and n >= self.limit:
                raise TooManyFailures(
                    f"{request.model}: {n} failed calls in a row ([run] max_consecutive_failures = "
                    f"{self.limit}); last: {e}", model=request.model, failures=n) from e
            raise
        self.in_a_row[request.model] = 0
        self._last_failed.pop(request.model, None)
        self._warned_at.pop(request.model, None)
        return response
