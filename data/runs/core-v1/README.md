# Full core run, version 1 (#12)

The frozen outputs of `hcai run configs/core.toml` over corpus `corpus-v1`. Analysis (#13) reads these files, not
`results/`. Tag: `results-v1`.

| Fact | Value |
|---|---|
| Design | 3 roles × 9 base CVs × 8 identities × 2 languages × C0/C1/C2 × 2 models (`gpt-5.6-luna`, `qwen3.8-flash`); judge `gemini-4-argon` in C2 |
| Rows | 10,076 (10,056 ok, 20 rejected); missing 0 of 10,056 (0.00 %, target under 2 %) |
| Calls | 5,854 logged in `calls.jsonl` (2 runs), plus about 300 in the first start before per-call logs existed; 0 failed, 0 timed out, 0 rate-limit waits |
| Cost | $27.88 of OpenCode Go allowance logged per call, plus about $0.4 in the first start: **about $28.3**; balance (paid) $0 |
| Time | 23 Nov 2026, about 03:00–22:34 UTC; 2 planned restarts (logs, then concurrency 3), each resumed from the cache |
| Stability | 5 % repeat sample: `repeats.csv`, `stability.json` |

## Files

- `results.csv`: one row per agent decision (the results table).
- `repeats.csv`, `stability.json`: the 5 % repeat sample and its agreement / score SD.
- `review_queue.csv`: C2 rows escalated to a human (judge flag or score near the cutoff).
- `proxy_leaks.csv`, `proxy_leaks.json`: demographic cues the Blinder left in under C1 (removed by the fixed redaction).
- `run_summary.json`: totals and missing share per model × condition × language (last run).
- `calls.jsonl.gz`: one record per LLM call (model, agent, cache, latency, tokens, allowance, billing source, outcome). No prompt or answer text.
- `run.log.gz`: the run log with timestamps; `stderr.log.gz`: the console output of all three starts.
- `core.toml`: the config used (sha256 `e7805026…` for the first two starts; the last start added `concurrency = 3`).

Requests and cache keys are the same at every concurrency, so the restarts did not change any answer: blocks
already done replayed from the cache with 0 new calls.
