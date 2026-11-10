# 2. Malformed answers are cached; a retry asks again with a note

Date: 2026-11-10 · Status: accepted (ticket #6); supersedes the "unparseable outputs are not cached" rule from #5

## Context

Agent calls run at temperature 0 with a fixed seed. Malformed output (unparseable JSON, a score outside 0–100, a ranking that is not a permutation) has to be retried and, if it still fails, recorded as missing (#6). Before this, unparseable answers were left out of the cache so that a rerun would retry them. But then a rerun was neither free nor a replay: the provider could answer differently the second time, and parseable-but-invalid answers were cached anyway, so the two kinds of malformed output behaved differently.

## Decision

- Every pipeline response is cached, malformed ones included. Calls that raise (HTTP errors, timeouts) are not cached.
- The corpus generator is the exception: it runs at temperature 0.7 with a fixed number of attempts, so it keeps unparseable answers out of the cache (`CachedClient(cache_unparseable=False)`) and a rerun of a failed generation draws fresh ones.
- A retry after a malformed answer is a new request: the last message gets a note, `(Attempt k. The previous answer was rejected: <reason>. Respond only with JSON matching the required schema.)`. Re-sending the identical request would return the same answer from the cache, and with temperature 0 most likely from the provider too.
- A call that raised is repeated as is, after `[run] retry_backoff` seconds (default 2), doubling each time.
- After `[run] retries` retries, the agent's rows are `missing`, with the error. `attempts` records how many calls are behind each row.
- To ask the missing ones again, raise `retries`. The earlier attempts replay from the cache and only the new attempts are sent.

## Consequences

A pipeline rerun replays exactly what the first run saw and costs nothing (US33). A retried answer comes from a slightly different prompt (the note). The note is the same for every identity, and `attempts` lets the analysis flag or exclude retried rows.
