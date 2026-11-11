# 5. Per-model providers; rate limits are waited out, then the run stops and resumes from the cache

Date: 2026-11-11 · Status: accepted (ticket #11, Part A)

## Context

The pilot and full run should cost nothing beyond the researcher's OpenCode Go subscription, with OpenRouter as a fallback if the subscription runs short (#11). OpenCode Go meters a usage budget over a rolling window of about 5 h (about $12). When the budget runs out the provider refuses calls until the window moves on. It reports no price per call.

## Decision

- **Providers.** Each model is served by a named provider, an OpenAI-compatible endpoint with its own `base_url` and key variable. The built-in providers are `opencode-go` (the default) and `openrouter`. `[model_providers]` names a model's provider, so a switch is one line. The ids are each provider's own, so a switch may also mean a new id, which counts as a new model in the results.
- **The cache key holds no provider.** It is the request (model id, messages, temperature, seed, schema). The provider is a way of reaching a model id, not part of the experiment. A rerun after a switch therefore replays every call it already completed.
- **Keys** are read from the environment, filled from a gitignored `.env`. They are sent only in the Authorization header and redacted from any error text the provider echoes back. They never reach requests, responses, the cache or the results.
- **Rate and usage limits** (HTTP 429, or an error that names a rate or usage limit) are waited out below the cache and the call meter, so a wait counts as neither a retry nor a call. The wait is `Retry-After` if the provider sends it, else `[run] rate_limit_backoff` doubling up to `rate_limit_max_backoff`. They are not retried by the pipeline's `[run] retries`. Retried, they would turn a budget gap into `missing` rows.
- **Clean stop.** A call that would wait more than `[run] rate_limit_max_wait` seconds in all (default 6 h), or HTTP 402 (no credits), raises `ProviderExhausted`. It is a `BaseException`, so it passes the `except Exception` handlers that record agent failures as missing data. The run writes `progress.json` and no results table, and the CLI exits with code 3. Every completed call is already cached, so rerunning the same command carries on from the first call not yet made.
- **Unknown cost.** A provider that reports no price gives `cost = None`, a blank in `results.csv`. Totals sum the known prices and count `unpriced_attempts` and `unpriced_calls`. Each model in `run_summary.json` names its `provider` and `billing` (`metered` or `subscription`). The pilot also scales tokens up to the full run (`estimated_full_tokens`), which is what a subscription budget is spent on.

## Consequences

A run that outlasts the subscription window either waits, unattended, or stops and resumes later with nothing repeated. A crew interrupted midway replays its earlier agents' calls from the cache. Responses served by different providers under the same model id share cache entries. To compare providers, use a separate `--cache`.
