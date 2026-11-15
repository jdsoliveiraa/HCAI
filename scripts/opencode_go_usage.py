"""Show how much of the OpenCode Go allowance is used, per model.

OpenCode Go has no usage or balance endpoint, and the remaining 5-hour / weekly / monthly budget is only
shown in the console (https://opencode.ai/console). What each response does carry is `usage.opencode`:
`billing_source` (go, or balance once Extra Usage kicks in), `allowance` (dollars this request took from
the Go allowance) and `list_cost`. This sends one tiny chat request (max_tokens=16) per model and prints
those, which confirms the key works, which models still bill to Go (an exhausted cap answers with an error
or switches billing_source), and each model's allowance multiplier (allowance / list_cost). Each probe
costs a fraction of a cent of allowance. The key comes from .env (OPENCODE_API_KEY) and is never printed.

    uv run python scripts/opencode_go_usage.py                     # models of configs/core.toml
    uv run python scripts/opencode_go_usage.py gemini-4-argon      # just these models
    uv run python scripts/opencode_go_usage.py --raw gpt-5.6-luna  # also dump the whole response body
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx

from hcai.config import load_design
from hcai.providers import BUILTIN_PROVIDERS, load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def config_models(config: Path) -> list[str]:
    design = load_design(config)
    models = list(design.models)
    for extra in (design.generator.model if design.generator else None, design.judge_model):
        if extra and extra not in models:
            models.append(extra)
    return models


def find_allowance(body: object) -> object:
    """The `opencode.allowance` value, or any `allowance` key anywhere in the body."""
    if isinstance(body, dict):
        oc = body.get("opencode")
        if isinstance(oc, dict) and "allowance" in oc:
            return oc["allowance"]
        for v in body.values():
            if (found := find_allowance(v)) is not None:
                return found
        return body.get("allowance")
    if isinstance(body, list):
        for v in body:
            if (found := find_allowance(v)) is not None:
                return found
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", help="model ids (default: the models of --config)")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "core.toml")
    ap.add_argument("--raw", action="store_true", help="print the whole response body (key never included)")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    provider = BUILTIN_PROVIDERS["opencode-go"]
    key = os.environ.get(provider.key_env, "").strip()
    if not key:
        print(f"{provider.key_env} is not set (add it to {ROOT / '.env'})", file=sys.stderr)
        return 2
    headers = {"Authorization": f"Bearer {key}", "User-Agent": "hcai-usage/0.1", **provider.http_headers()}

    models = args.models or config_models(args.config)
    status = 0
    with httpx.Client(timeout=60) as http:
        for model in models:
            try:
                r = http.post(f"{provider.base_url}/chat/completions", headers=headers, json={
                    "model": model, "max_tokens": 16, "temperature": 0,
                    "messages": [{"role": "user", "content": "Reply with: ok"}]})
            except httpx.HTTPError as e:
                print(f"{model}: request failed: {type(e).__name__}", file=sys.stderr)
                status = 1
                continue
            limit_headers = {k: v for k, v in r.headers.items() if any(s in k.lower() for s in ("ratelimit", "limit", "opencode", "allowance", "retry-after"))}
            try:
                body = r.json()
            except ValueError:
                body = {"text": r.text[:500]}
            print(f"== {model}  HTTP {r.status_code}")
            usage = body.get("usage") if isinstance(body, dict) else None
            oc = usage.get("opencode") if isinstance(usage, dict) else None
            if isinstance(oc, dict):
                allowance, list_cost = oc.get("allowance"), oc.get("list_cost")
                ratio = f"{allowance / list_cost:.2f}x" if allowance and list_cost else "?"
                print(f"billing_source: {oc.get('billing_source')}  allowance: ${allowance}  list_cost: ${list_cost}"
                      f"  multiplier: {ratio}")
            else:
                allowance = find_allowance(body)
                print("allowance:", allowance if allowance is not None else "(not in response)")
            if limit_headers:
                print("headers:", json.dumps(limit_headers, indent=2))
            if r.status_code >= 400:
                status = 1
                print("error:", json.dumps(body, ensure_ascii=False)[:500].replace(key, "***"))
            if args.raw:
                print("body:", json.dumps(body, indent=2, ensure_ascii=False).replace(key, "***"))
    return status


if __name__ == "__main__":
    sys.exit(main())
