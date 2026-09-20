"""Does the server reuse a cached prefill for an identical prompt?
Same prompt three times, then the same prompt with a unique prefix three times."""
import json, time, urllib.request, random, string, statistics

BASE = "http://localhost:1234"
MODEL = "qwen/qwen3-1.7b"
FILLER = ("The technician arrived at the substation at 08:14 and logged the panel "
          "identifier, the meter serial number, the tariff class, and the last "
          "recorded index value before opening the cabinet door. ")
BODY = FILLER * 40 + "\n\nReturn JSON with the fields panel_id, meter_serial, tariff_class. /no_think"

def ttft(prompt):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 8, "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter(); first = None; usage = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "): continue
            p = line[6:]
            if p == "[DONE]": break
            c = json.loads(p)
            if c.get("usage"): usage = c["usage"]
            ch = c.get("choices") or []
            if ch:
                d = ch[0].get("delta") or {}
                if (d.get("content") or d.get("reasoning_content")) and first is None:
                    first = time.perf_counter()
    return (first - start) * 1000, usage["prompt_tokens"]

print("ayni prompt, uc kez:")
same = []
for i in range(3):
    ms, pt = ttft(BODY); same.append(ms)
    print(f"  {i+1}. ttft {ms:.0f} ms  ({pt} prompt token, prefill {pt/(ms/1000):.0f} tok/s)")

print("her seferinde farkli onek:")
uniq = []
for i in range(3):
    nonce = "".join(random.choices(string.ascii_lowercase, k=12))
    ms, pt = ttft(f"Reference {nonce}. " + BODY); uniq.append(ms)
    print(f"  {i+1}. ttft {ms:.0f} ms  ({pt} prompt token, prefill {pt/(ms/1000):.0f} tok/s)")

print(f"\nmedyan ayni: {statistics.median(same):.0f} ms | medyan farkli: {statistics.median(uniq):.0f} ms")
