"""Bounded concurrency for a run (`[run] concurrency`, ADR 0005 amendment of 2026-11-23 on concurrency).

`run_tasks` runs a list of tasks, each a function with the tasks it depends on, with at most `n` in flight, and
returns their results in list order, whatever order they finished in. With `n = 1` the tasks run one after the
other in list order on the calling thread, exactly as a plain loop would.

With `n > 1` each task runs in a daemon thread, in a copy of the caller's context (so the run's stop event and
the run log's scope reach it). Among the tasks whose dependencies are done, the one with the lowest `priority`,
then the lowest index, starts first.

Stops: when a task raises (a `ProviderExhausted` stop, or a crash), the run's stop `event` is set, no new task
starts, and the tasks in flight are waited for: their calls under way finish (and are cached), and their next
call or wait raises `hcai.llm.Stopping`. Then the first error is raised (a real stop wins over a `Stopping`).
On Ctrl-C (KeyboardInterrupt in the calling thread) the same happens; a second Ctrl-C abandons the tasks in
flight (their threads are daemons, so the process can exit; an abandoned call is not cached, nothing else is lost).
"""

from __future__ import annotations

import contextvars
import logging
import queue
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from hcai.llm import Stopping

logger = logging.getLogger("hcai.runner")


@dataclass(frozen=True)
class Task:
    fn: Callable[[], Any]
    deps: tuple[int, ...] = ()  # indices of the tasks this one needs done first (all lower than its own)
    priority: int = 0  # among ready tasks, lower starts first (n > 1 only)


def run_tasks(tasks: Sequence[Task], n: int, event: threading.Event) -> list[Any]:
    for i, t in enumerate(tasks):
        if any(not 0 <= d < i for d in t.deps):
            raise ValueError(f"task {i} depends on a task that does not come before it: {t.deps}")
    if n <= 1:
        return [t.fn() for t in tasks]
    results: list[Any] = [None] * len(tasks)
    done: set[int] = set()
    pending = list(range(len(tasks)))
    running: dict[int, threading.Thread] = {}
    finished: queue.Queue[tuple[int, bool, Any]] = queue.Queue()
    errors: list[BaseException] = []

    def launch(i: int) -> None:
        ctx = contextvars.copy_context()

        def body() -> None:
            try:
                out = ctx.run(tasks[i].fn)
            except BaseException as e:  # noqa: BLE001 - handed to the calling thread
                finished.put((i, False, e))
            else:
                finished.put((i, True, out))

        thread = threading.Thread(target=body, name=f"hcai-worker-{i}", daemon=True)
        running[i] = thread
        thread.start()

    def collect() -> None:
        while True:
            try:
                i, ok, value = finished.get(timeout=0.2)  # a timeout, so Ctrl-C reaches this thread
            except queue.Empty:
                continue
            running.pop(i, None)
            if ok:
                results[i] = value
                done.add(i)
            else:
                errors.append(value)
                event.set()
            return

    try:
        while True:
            if not errors and not event.is_set():
                ready = sorted((i for i in pending if all(d in done for d in tasks[i].deps)),
                               key=lambda i: (tasks[i].priority, i))
                for i in ready[: n - len(running)]:
                    pending.remove(i)
                    launch(i)
            if not running:
                break
            collect()
    except KeyboardInterrupt:
        event.set()
        if running:
            logger.warning(f"interrupted: waiting for the {len(running)} task(s) in flight to finish their call under "
                           f"way (their answers are cached); press Ctrl-C again to abandon them")
        # A second KeyboardInterrupt propagates from here: the daemon workers are abandoned. Live threads, not the
        # bookkeeping, say when to stop waiting, so an interrupt that landed mid-bookkeeping cannot hang this.
        while any(t.is_alive() for t in running.values()):
            try:
                i, _, _ = finished.get(timeout=0.2)
            except queue.Empty:
                continue
            running.pop(i, None)
        raise
    if errors:
        real = [e for e in errors if not isinstance(e, Stopping)]
        raise (real or errors)[0]
    if event.is_set() and pending:  # stopped from outside with tasks left: not a complete result
        raise Stopping("the run is stopping")
    return results
