"""Run logs (ADR 0005, amendment of 2026-11-23 on logs, #12): what a long run is doing, while it does it.

Everything logs through the stdlib `logging` tree `hcai.*`. A library user sees nothing (the `hcai` logger has a
`NullHandler`); the CLI calls `configure`, which sends INFO and up to stderr and, for `hcai run`, appends a
human-readable `<out>/run.log` (ISO timestamps with ms, level, logger) at `--log-level`.

`RunLog` is one run's record keeper, active while `hcai.runner.execute` runs:

- each agent attempt (`Attempt`, opened by `hcai.pipeline.ClientLLM`) becomes one line in the log (INFO when it
  answered, WARNING when it failed, ERROR when it stopped the run, DEBUG for a cache hit) and, while a `RunLog`
  is active, one JSON record appended to `<out>/calls.jsonl` and flushed per line. The layers below the agent
  add what only they see through `note` (cache hit and key, HTTP status, finish reason, HTTP latency); a
  rate-limit wait adds its own record (`rate_limited`).
- a heartbeat thread logs where the run is every `[run] heartbeat_s` seconds and rewrites `progress.json`.

Under `[run] concurrency` several workers record at once: the CV or pool a call is for (`RunLog.item`) is kept per
worker (a context variable), the counters and `calls.jsonl` are written under one lock (one whole line per record),
and the `logging` handlers lock per line.

Nothing here touches a request, so no cache key changes. No prompt or answer text is written: a prompt's length,
an answer's length and hash, and the cache key prefix (the full text stays in the cache). Error texts come from
`ProviderError`, whose messages are already redacted of the key and header values, and are cut short. No key,
header or session id is ever logged. Every method that records something swallows its own failures: logging
must never stop or slow the run.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

ROOT = "hcai"
logging.getLogger(ROOT).addHandler(logging.NullHandler())  # silent unless the CLI (or a caller) configures it
log = logging.getLogger("hcai.runlog")
calls_log = logging.getLogger("hcai.calls")

LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
DEFAULT_HEARTBEAT = 300.0
ERROR_CHARS = 300  # error messages in the logs are cut to this many characters

P = ParamSpec("P")
R = TypeVar("R")


def _safe(fn: Callable[P, R]) -> Callable[P, R | None]:
    """Run `fn`, but never let a logging failure reach the run (it is reported once at DEBUG, then dropped)."""
    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R | None:
        try:
            return fn(*args, **kwargs)
        except Exception:  # noqa: BLE001 - logging must never crash the run
            try:
                log.debug("logging failed in %s", fn.__name__, exc_info=True)
            except Exception:  # noqa: BLE001
                pass
            return None
    return wrapper


# --- configuration -------------------------------------------------------------------------------------------


class _IsoFormatter(logging.Formatter):
    """`2026-11-23T14:05:09.123+00:00 INFO    hcai.runner: ...`: local time with its offset, to the millisecond."""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        return datetime.fromtimestamp(record.created).astimezone().isoformat(timespec="milliseconds")


class _CurrentStderr(logging.StreamHandler):  # type: ignore[type-arg]
    """A stream handler that writes to whatever `sys.stderr` is when the record is emitted (tests swap it)."""

    def __init__(self) -> None:
        super().__init__(sys.stderr)

    @property  # type: ignore[override]
    def stream(self) -> Any:  # pyright: ignore[reportIncompatibleVariableOverride]
        return sys.stderr

    @stream.setter
    def stream(self, value: Any) -> None:
        pass


class _LazyFile(logging.FileHandler):
    """Appends to its file, created (with its directory) only when the first record is written: a command that
    fails before logging anything leaves nothing behind."""

    def __init__(self, path: Path):
        super().__init__(path, mode="a", encoding="utf-8", delay=True)

    def _open(self) -> Any:
        Path(self.baseFilename).parent.mkdir(parents=True, exist_ok=True)
        return super()._open()


FILE_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
STDERR_FORMAT = "%(asctime)s %(levelname)s %(message)s"


@contextmanager
def configure(log_file: str | Path | None = None, *, level: str = "INFO", quiet: bool = False) -> Iterator[None]:
    """Log `hcai.*` to stderr (INFO and up; WARNING and up when `quiet`) and, if given, append `log_file` at `level`.

    The handlers are removed and closed on exit, so a test (or a second command in one process) starts clean.
    """
    file_level = logging.getLevelNamesMapping()[level.upper()]
    stderr_level = logging.WARNING if quiet else logging.INFO
    handlers: list[logging.Handler] = []
    err = _CurrentStderr()
    err.setLevel(stderr_level)
    err.setFormatter(logging.Formatter(STDERR_FORMAT, "%H:%M:%S"))
    handlers.append(err)
    if log_file is not None:
        fh = _LazyFile(Path(log_file))
        fh.setLevel(file_level)
        fh.setFormatter(_IsoFormatter(FILE_FORMAT))
        handlers.append(fh)
    root = logging.getLogger(ROOT)
    old_level = root.level
    root.setLevel(min(h.level for h in handlers))
    for h in handlers:
        root.addHandler(h)
    try:
        yield
    finally:
        for h in handlers:
            root.removeHandler(h)
            h.close()
        root.setLevel(old_level)


# --- what a run is about to do -------------------------------------------------------------------------------


def file_sha256(path: str | Path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def git_state(where: str | Path) -> tuple[str | None, bool | None]:
    """The commit checked out at `where` and whether tracked files differ from it; (None, None) when unknown."""
    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(["git", "-C", str(where), *args], capture_output=True, text=True, timeout=10,
                                  check=False)
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    if commit is None:
        return None, None
    status = git("status", "--porcelain", "--untracked-files=no")
    return commit, None if status is None else bool(status)


# --- the current attempt -------------------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _short(text: str, limit: int = ERROR_CHARS) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# The start of a malformed answer that `ClientLLM.output` appends to a missing row's error (`...: 'answer'`),
# possibly nested ("blinding failed: <error>: '...'"): cut before the error reaches the log.
_ANSWER_SNIPPET = re.compile(r": (?:'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")$", re.S)


def without_answer(error: str) -> str:
    """`error` without the answer snippet a missing row's error ends with (answer text never reaches the log)."""
    cut = _ANSWER_SNIPPET.sub("", error)
    return cut if cut == error else cut + " (answer start in results.csv)"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


def outcome_of(error: BaseException) -> str:
    """The `outcome` of an attempt that raised `error`."""
    from hcai.llm import ProviderExhausted, RateLimited  # here, as hcai.llm imports this module

    if getattr(error, "kind", None) == "stopping":  # `hcai.llm.Stopping`: not made, the run was stopping
        return "abandoned"
    if isinstance(error, ProviderExhausted):
        return "exhausted"
    if isinstance(error, RateLimited):
        return "rate_limited"
    if getattr(error, "timeout", False):
        return "timeout"
    return "provider_error"


_CURRENT: contextvars.ContextVar[Attempt | None] = contextvars.ContextVar("hcai_attempt", default=None)


@_safe
def note(**fields: Any) -> None:
    """Add what a layer below the agent knows (cache hit and key, HTTP status, finish reason) to the current attempt."""
    attempt = _CURRENT.get()
    if attempt is not None:
        attempt.notes.update(fields)


class Attempt:
    """One request an agent sends down the client stack: logged and recorded once it answered or raised.

    `agent` is the agent's name (screener, evaluator, ranker, blinder, judge), `n` the attempt number (1-based)
    of this agent call. `answered` / `failed` log one line and append one record to `calls.jsonl` when a `RunLog`
    is active; neither ever raises.
    """

    def __init__(self, agent: str, n: int, request: Any):
        self.agent = agent
        self.n = n
        self.model = getattr(request, "model", "")
        self.notes: dict[str, Any] = {}
        self.started = time.monotonic()
        try:
            self.prompt_chars = sum(len(m.get("content", "")) for m in request.messages)
        except Exception:  # noqa: BLE001
            self.prompt_chars = None
        self._token = _CURRENT.set(self)
        run = active()
        if run is not None:
            run.activity(f"{agent} call to {self.model}, attempt {n}")

    def _close(self) -> None:
        try:
            _CURRENT.reset(self._token)
        except (ValueError, RuntimeError):  # another context (a thread): just clear it
            _CURRENT.set(None)

    @_safe
    def answered(self, response: Any, *, error: str = "", retry: bool = False) -> None:
        """The attempt got an answer: `ok`, or `parse_error` (with the validator's `error`) when it is malformed."""
        self._close()
        usage = response.usage
        text = response.text or ""
        cached = bool(response.cached)
        self.notes.setdefault("cache", "hit" if cached else None)
        record = self._record(
            outcome="parse_error" if error else "ok", usage=usage, retry=retry, wait=0.0,
            error_class="ParseError" if error else None, error=_short(error) if error else None,
            answer_chars=len(text), answer_sha=_sha(text))
        level = logging.DEBUG if cached and not error else logging.WARNING if error else logging.INFO
        _emit(level, record)

    @_safe
    def failed(self, error: BaseException, *, retry: bool = False, wait: float = 0.0) -> None:
        """The attempt raised `error`; `retry`: the agent repeats it after `wait` seconds."""
        self._close()
        outcome = outcome_of(error)
        record = self._record(outcome=outcome, usage=getattr(error, "usage", None), retry=retry, wait=wait,
                              error_class=type(error).__name__, error=_short(str(error)))
        _emit(logging.ERROR if outcome == "exhausted" else logging.INFO if outcome == "abandoned" else logging.WARNING,
              record)

    def _record(self, *, outcome: str, usage: Any, retry: bool, wait: float, error_class: str | None,
                error: str | None, **extra: Any) -> dict[str, Any]:
        run = active()
        latency = self.notes.get("http_s")
        record: dict[str, Any] = dict(
            ts=_now_iso(), run=run.run_id if run else None, **(run.scope_fields() if run else {}),
            model=self.model, agent=self.agent, attempt=self.n, cache=self.notes.get("cache"),
            key=self.notes.get("key"),
            latency_s=round(latency if latency is not None else time.monotonic() - self.started, 3),
            **usage_fields(usage), finish_reason=self.notes.get("finish_reason"),
            http_status=self.notes.get("http_status"), outcome=outcome, error_class=error_class, error=error,
            retry=retry, wait_s=round(wait, 3), prompt_chars=self.prompt_chars, **extra)
        return record


def usage_fields(usage: Any) -> dict[str, Any]:
    """The token and billing fields of a record (None when the attempt reported no usage)."""
    if usage is None:
        return dict(prompt_tokens=None, completion_tokens=None, total_tokens=None, allowance=None,
                    balance_cost=None, billing_source=None)
    source = usage.billing_source
    balance = (usage.cost - (usage.allowance or 0.0)
               if source is not None and source not in ("go", "unknown") and usage.cost is not None else None)
    return dict(prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens,
                total_tokens=usage.prompt_tokens + usage.completion_tokens, allowance=usage.allowance,
                balance_cost=None if balance is None else round(balance, 8), billing_source=source)


@_safe
def rate_limited(model: str, *, wait: float, resume_at: datetime, waited: float, error: BaseException) -> None:
    """A rate- or usage-limit answer that `RateLimitGuard` waits out (`wait` seconds): one `rate_limited` record.

    The human-readable WARNING line is the guard's own; this adds the record (no second line).
    """
    attempt = _CURRENT.get()
    run = active()
    if run is not None:
        run.activity(f"waiting out a rate/usage limit on {model} until {resume_at.isoformat(timespec='seconds')}")
    if attempt is None or run is None:
        return
    record = attempt._record(outcome="rate_limited", usage=None, retry=True, wait=wait,
                             error_class=type(error).__name__, error=_short(str(error)),
                             resume_at=resume_at.isoformat(timespec="seconds"), waited_s=round(waited, 3))
    run.write(record)
    attempt.notes.pop("http_status", None)  # the next try's status is its own


def _line(r: dict[str, Any]) -> str:
    where = r.get("variant") or r.get("pool") or ""
    if r.get("repeat"):
        where += f" repeat {r['repeat']}"
    tokens = f" tok {r['prompt_tokens']}+{r['completion_tokens']}" if r.get("prompt_tokens") is not None else ""
    billing = ""
    if r.get("billing_source") is not None:
        billing = f" {r['billing_source']}"
        if r.get("allowance") is not None:
            billing += f" ${r['allowance']:.5f}"
        if r.get("balance_cost"):
            billing += f" balance ${r['balance_cost']:.5f}"
    finish = f" finish={r['finish_reason']}" if r.get("finish_reason") else ""
    status = f" http {r['http_status']}" if r.get("http_status") else ""
    cache = f" {r['cache']}" if r.get("cache") else ""
    out = (f"{r['agent']} {r['model']} {where} #{r['attempt']}{cache} {r['latency_s']:.1f}s{status}{tokens}"
           f"{finish}{billing} -> {r['outcome']}")
    if r.get("error"):
        out += f" ({r['error_class']}: {_short(r['error'], 200)})"
    if r.get("retry"):
        out += f"; retry in {r['wait_s']:g}s"
    return out


def _emit(level: int, record: dict[str, Any]) -> None:
    calls_log.log(level, _line(record))
    run = active()
    if run is not None:
        run.write(record)


# --- the run ------------------------------------------------------------------------------------------------


_ACTIVE: RunLog | None = None
# The CV variant or pool the calls of this worker are for (`RunLog.item`): one per thread / context.
_ITEM: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar("hcai_item", default={})


def active() -> RunLog | None:
    return _ACTIVE


def new_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + hashlib.sha256(
        f"{time.time_ns()}".encode()).hexdigest()[:6]


class RunLog:
    """One run's records: `calls.jsonl`, progress counters, the heartbeat and `progress.json`.

    `out_dir=None` keeps the counters only (no files, no heartbeat): what `run_experiment` uses.
    `stats` returns this run's spending so far (calls, allowance, balance, failed calls); it is read from the
    heartbeat thread, so it must only copy plain counters.
    """

    def __init__(self, out_dir: str | Path | None, *, run_id: str | None = None,
                 heartbeat_s: float = DEFAULT_HEARTBEAT, stats: Callable[[], dict[str, Any]] | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.run_id = run_id or new_run_id()
        self.out_dir = None if out_dir is None else Path(out_dir)
        self.heartbeat_s = heartbeat_s
        self._stats = stats or (lambda: {})
        self._clock = clock
        self._lock = threading.Lock()
        self._file: Any = None
        self._write_failed = False
        self._stop = threading.Event()
        self._closed = False  # set on exit, under the lock: the heartbeat writes nothing after it
        self._thread: threading.Thread | None = None
        self.started = clock()
        self.started_at = datetime.now(UTC)
        self.scope: dict[str, Any] = {}
        self.block_label = ""
        self.rows_total = 0
        self.rows_done = 0  # planned rows whose CV or pool is done (a judge row counts even when not due)
        self.rows = 0  # rows written to the results so far
        self.missing = 0
        self.records = 0
        self.hits = 0
        self.misses = 0
        self._count_lock = threading.Lock()  # the counters above, updated by concurrent workers
        self._activity = "starting"
        self._activity_since = clock()
        self._window: tuple[float, int] = (self.started, 0)  # (time, rows_done) at the last heartbeat

    # lifecycle

    def __enter__(self) -> RunLog:
        global _ACTIVE
        _ACTIVE = self
        if self.out_dir is not None:
            self._open()
            if self.heartbeat_s > 0:
                self._thread = threading.Thread(target=self._beat, name="hcai-heartbeat", daemon=True)
                self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        global _ACTIVE
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        with self._lock:  # waits for a write in progress; a heartbeat still running after the join writes nothing
            self._closed = True
        if _ACTIVE is self:
            _ACTIVE = None
        with self._lock:
            if self._file is not None:
                try:
                    self._file.close()
                except OSError:
                    pass
                self._file = None

    @_safe
    def _open(self) -> None:
        assert self.out_dir is not None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / "calls.jsonl"
        # a run killed mid-line leaves a partial last line: start ours on a fresh one
        partial = False
        if path.exists() and path.stat().st_size > 0:
            with path.open("rb") as f:
                f.seek(-1, os.SEEK_END)
                partial = f.read(1) != b"\n"
        self._file = path.open("a", encoding="utf-8")
        if partial:
            self._file.write("\n")

    @_safe
    def write(self, record: dict[str, Any]) -> None:
        """Append one record to `calls.jsonl`, flushed; counts cache hits and misses."""
        if record.get("outcome") != "rate_limited":  # a wait is no attempt of its own
            with self._count_lock:
                self.records += 1
                if record.get("cache") == "hit":
                    self.hits += 1
                elif record.get("cache") == "miss":
                    self.misses += 1
        with self._lock:
            if self._file is None:
                return
            try:
                self._file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                self._file.flush()
            except (OSError, ValueError) as e:
                if not self._write_failed:
                    self._write_failed = True
                    log.warning("cannot write calls.jsonl (%s: %s); the run goes on without call records",
                                type(e).__name__, e)
                try:
                    self._file.close()
                except (OSError, ValueError):
                    pass
                self._file = None

    # where the run is

    def scope_fields(self) -> dict[str, Any]:
        return {**self.scope, **_ITEM.get()}

    def activity(self, what: str) -> None:
        self._activity = what
        self._activity_since = self._clock()

    @_safe
    def plan(self, rows: int, *, blocks: int, variants: int, pools: int) -> None:
        self.rows_total = rows
        self.write_progress("running")
        log.info("plan: %d blocks (role x language x condition x model), %d CV variants, %d Ranker pools per "
                 "condition and model, %d result rows (upper bound: a judge row only follows a valid Evaluator "
                 "score; the stability sample comes on top)", blocks, variants, pools, rows)

    @_safe
    def block(self, i: int, n: int, *, role: str, language: str, condition: str, model: str) -> None:
        self.block_label = f"{i}/{n} {role}/{language}/{condition}/{model}"
        self.scope = dict(block=f"{i}/{n}", role=role, lang=language, condition=condition, block_model=model)
        log.info("block %s started", self.block_label)

    @contextmanager
    def item(self, **fields: Any) -> Iterator[None]:
        """The CV variant (`variant`, `base_cv`, `identity`) or pool (`pool`) the calls inside are for.

        Kept per worker (a context variable), on top of the block's `scope`.
        """
        token = _ITEM.set({**_ITEM.get(), **fields})
        try:
            yield
        finally:
            _ITEM.reset(token)

    @_safe
    def done(self, rows: list[Any], planned: int) -> None:
        """`rows` were added to the results for a CV or pool that was due `planned` rows; warns about missing ones."""
        missing = sum(r.status == "missing" for r in rows)
        with self._count_lock:
            self.rows_done += planned
            self.rows += len(rows)
            self.missing += missing
        by: dict[tuple[str, str, str], int] = {}
        for r in rows:
            if r.status == "missing":
                where = r.pool or f"{r.base_cv}:{r.identity}"
                key = (r.agent, where, _short(without_answer(r.error), 200))
                by[key] = by.get(key, 0) + 1
        for (agent, where, error), k in by.items():
            log.warning("missing: %s%s row%s of %s in block %s: %s", agent, f" x{k}" if k > 1 else "",
                        "s" if k > 1 else "", where, self.block_label, error or "no answer")

    # heartbeat and progress.json

    def snapshot(self) -> dict[str, Any]:
        """The numbers of the heartbeat and of `progress.json`."""
        now = self._clock()
        elapsed = now - self.started
        stats = {}
        try:
            stats = dict(self._stats())
        except Exception:  # noqa: BLE001
            pass
        then, done_then = self._window
        recent = (self.rows_done - done_then) / (now - then) * 3600 if now - then >= 1 else None
        overall = self.rows_done / elapsed * 3600 if elapsed >= 1 else None
        rate = recent if recent else overall
        left = max(self.rows_total - self.rows_done, 0)
        eta = (datetime.now(UTC) + timedelta(hours=left / rate)) if rate and left else None
        return dict(
            run_id=self.run_id, updated_at=_now_iso(), started_at=self.started_at.isoformat(timespec="seconds"),
            elapsed_s=round(elapsed), block=self.block_label, rows_done=self.rows_done, rows_total=self.rows_total,
            percent=round(100 * self.rows_done / self.rows_total, 2) if self.rows_total else None,
            rows=self.rows, missing=self.missing, agent_attempts=self.records, cache_hits=self.hits,
            cache_misses=self.misses, **stats,
            rows_per_hour=None if rate is None else round(rate, 1),
            rows_per_hour_overall=None if overall is None else round(overall, 1),
            eta=None if eta is None else eta.isoformat(timespec="minutes"),
            now=self._activity, now_for_s=round(now - self._activity_since))

    @_safe
    def heartbeat(self) -> None:
        s = self.snapshot()
        self._window = (self._clock(), self.rows_done)
        pct = f" ({s['percent']:.1f}%)" if s["percent"] is not None else ""
        rate = f"{s['rows_per_hour']:.0f} rows/h" if s["rows_per_hour"] is not None else "rate n/a"
        eta = f", ETA {s['eta']}" if s["eta"] else ""
        log.info("heartbeat: rows %d/%d%s, %s calls (%d cache hits, %d misses), %d missing so far, allowance "
                 "$%.4f, balance $%.4f, %s%s; block %s; now: %s (%ds)", s["rows_done"], s["rows_total"], pct,
                 s.get("calls", "?"), s["cache_hits"], s["cache_misses"], s["missing"],
                 s.get("allowance_spent", 0.0), s.get("balance_spent", 0.0), rate, eta, s["block"] or "-",
                 s["now"], s["now_for_s"])
        self.write_progress("running", s)

    @_safe
    def write_progress(self, status: str, snapshot: dict[str, Any] | None = None, **extra: Any) -> None:
        """Rewrite `progress.json` atomically (a reader never sees half a file).

        Once the run log is closed, a "running" write (a late heartbeat) is dropped, so it cannot overwrite the
        stop or recreate the file a finished run deleted.
        """
        if self.out_dir is None:
            return
        data = dict(status=status, **(snapshot or self.snapshot()), **extra)
        path = self.out_dir / "progress.json"
        with self._lock:
            if self._closed and status == "running":
                return
            tmp = path.with_suffix(f".json.{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps(data, indent=2, default=str) + "\n")
            tmp.replace(path)

    def _beat(self) -> None:
        while not self._stop.wait(self.heartbeat_s):
            self.heartbeat()
