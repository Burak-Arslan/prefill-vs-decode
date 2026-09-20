"""Prefill vs decode benchmark against a local LM Studio server.

Measures each phase separately for three workload shapes that match real
on-device jobs:

  classify  short prompt, very short output   (intent, yes/no verification)
  extract   long prompt, short output         (summarize, pull fields as JSON)
  generate  short prompt, long output         (chat, drafting)

Method:
  - one warm-up run per model and workload, discarded
  - RUNS timed runs, median reported
  - time to first token measured client side from the stream
  - prefill rate  = prompt_tokens / time_to_first_token
  - decode rate   = (completion_tokens - 1) / (total_time - time_to_first_token)
  - token counts come from the server usage block, not from our own guess

Usage:
  python3 bench.py                 run everything, write results.json
  python3 bench.py --models a,b    limit to some models
"""

import argparse
import json
import random
import statistics
import string
import time
import urllib.request

BASE = "http://localhost:1234"
RUNS = 3

# A filler paragraph repeated to build a long prompt. Deterministic on purpose,
# so every model and every run reads exactly the same input.
FILLER = (
    "The technician arrived at the substation at 08:14 and logged the panel "
    "identifier, the meter serial number, the tariff class, and the last "
    "recorded index value before opening the cabinet door. "
)


def no_think(messages, model):
    """Qwen3 family keeps a thinking phase on by default. A classification or
    extraction job on a device would ship with it off, so switch it off with the
    documented soft switch and record whether any reasoning tokens still appear."""
    if "qwen3" not in model.lower():
        return messages
    out = []
    for m in messages:
        if m["role"] == "user":
            out.append({**m, "content": m["content"] + " /no_think"})
        else:
            out.append(m)
    return out


def _decode_rate(completion_tokens, decode_time):
    if decode_time <= 0 or completion_tokens < 24:
        return None
    rate = (completion_tokens - 1) / decode_time
    if rate > 2000:
        return None
    return rate


def defeat_prompt_cache(messages):
    """The server reuses a cached prefill when a prompt repeats, which makes a
    repeated benchmark prompt look up to six times faster than it is. Measured on
    this machine: same prompt 98 ms to first token, unique prefix 599 ms. Every
    request therefore starts with a fresh nonce so no prefix is ever reused."""
    nonce = "".join(random.choices(string.ascii_lowercase + string.digits, k=12))
    out = []
    patched = False
    for m in messages:
        if m["role"] == "user" and not patched:
            out.append({**m, "content": f"Reference {nonce}. " + m["content"]})
            patched = True
        else:
            out.append(m)
    return out


def post_stream(model, messages, max_tokens):
    """Send one streaming chat request, return timing and usage."""
    body = json.dumps(
        {
            "model": model,
            "messages": defeat_prompt_cache(no_think(messages, model)),
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()

    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )

    start = time.perf_counter()
    first_token_at = None
    usage = None

    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            produced = (
                delta.get("content")
                or delta.get("reasoning_content")
                or delta.get("reasoning")
            )
            if produced and first_token_at is None:
                first_token_at = time.perf_counter()

    end = time.perf_counter()
    if first_token_at is None or usage is None:
        raise RuntimeError("no content or no usage returned")

    ttft = first_token_at - start
    decode_time = end - first_token_at
    details = usage.get("completion_tokens_details") or {}
    return {
        "reasoning_tokens": details.get("reasoning_tokens", 0),
        "ttft_s": ttft,
        "decode_s": decode_time,
        "total_s": end - start,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "prefill_tok_s": usage["prompt_tokens"] / ttft if ttft > 0 else None,
        # A rate above this means the server buffered the stream instead of
        # delivering tokens as they were produced, so the number is not a decode
        # measurement and must not be reported.
        "decode_tok_s": _decode_rate(usage["completion_tokens"], decode_time),
    }


def workloads():
    long_prompt = FILLER * 40
    return {
        "classify": {
            "messages": [
                {
                    "role": "user",
                    "content": "Answer with one word, yes or no. Is an electricity "
                    "meter an industrial measurement device?",
                }
            ],
            "max_tokens": 16,
        },
        "extract": {
            "messages": [
                {
                    "role": "user",
                    "content": long_prompt
                    + "\n\nReturn JSON with the fields panel_id, meter_serial, "
                    "tariff_class. Answer with JSON only.",
                }
            ],
            "max_tokens": 60,
        },
        "generate": {
            "messages": [
                {
                    "role": "user",
                    "content": "Explain to a junior engineer why reading a model "
                    "weight from memory is the slow part of text generation.",
                }
            ],
            "max_tokens": 400,
        },
    }


def list_models():
    with urllib.request.urlopen(f"{BASE}/api/v0/models", timeout=30) as resp:
        data = json.load(resp)
    out = []
    for m in data["data"]:
        if m.get("type") in ("llm", "vlm"):
            out.append(
                {
                    "id": m["id"],
                    "arch": m.get("arch"),
                    "quantization": m.get("quantization"),
                    "format": m.get("compatibility_type"),
                }
            )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="")
    ap.add_argument("--runs", type=int, default=RUNS)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    models = list_models()
    if args.models:
        wanted = {m.strip() for m in args.models.split(",")}
        models = [m for m in models if m["id"] in wanted]

    jobs = workloads()
    results = []

    for model in models:
        for name, job in jobs.items():
            print(f"{model['id']} / {name}: warmup", flush=True)
            try:
                post_stream(model["id"], job["messages"], job["max_tokens"])
            except Exception as exc:
                print(f"  warmup failed: {exc}", flush=True)
                continue

            runs = []
            for i in range(args.runs):
                try:
                    r = post_stream(model["id"], job["messages"], job["max_tokens"])
                except Exception as exc:
                    print(f"  run {i + 1} failed: {exc}", flush=True)
                    continue
                runs.append(r)
                dec = (
                    f"{r['decode_tok_s']:.1f} tok/s"
                    if r["decode_tok_s"] is not None
                    else "n/a, output too short"
                )
                print(
                    f"  run {i + 1}: ttft {r['ttft_s'] * 1000:.0f} ms, "
                    f"prefill {r['prefill_tok_s']:.0f} tok/s, decode {dec}",
                    flush=True,
                )

            if not runs:
                continue

            results.append(
                {
                    "model": model["id"],
                    "arch": model["arch"],
                    "quantization": model["quantization"],
                    "format": model["format"],
                    "workload": name,
                    "prompt_tokens": runs[0]["prompt_tokens"],
                    "reasoning_tokens_median": statistics.median(
                        r["reasoning_tokens"] for r in runs
                    ),
                    "completion_tokens_median": statistics.median(
                        r["completion_tokens"] for r in runs
                    ),
                    "ttft_ms_median": statistics.median(r["ttft_s"] for r in runs)
                    * 1000,
                    "prefill_tok_s_median": statistics.median(
                        r["prefill_tok_s"] for r in runs
                    ),
                    "decode_tok_s_median": statistics.median(
                        r["decode_tok_s"] for r in runs
                    )
                    if all(r["decode_tok_s"] is not None for r in runs)
                    else None,
                    "runs": runs,
                }
            )

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {args.out} with {len(results)} rows")


if __name__ == "__main__":
    main()
