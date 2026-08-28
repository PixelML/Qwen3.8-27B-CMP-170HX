#!/usr/bin/env python3
"""Prefill throughput: long prompt, 1 output token, non-streaming.
prefill_tok/s = prompt_completion_tokens / request latency."""
import json, sys, time
import urllib.request

BASE, KEY = sys.argv[1], sys.argv[2]
TOKENS = int(sys.argv[3]) if len(sys.argv) > 3 else 8192
SAMPLES = 2

words = ("the quick brown fox jumps over the lazy dog while the sun sets "
         "behind the mountains and rivers flow through valleys ") * (TOKENS // 20 + 1)
prompt = " ".join(words.split()[:TOKENS])

results = []
for i in range(1 + SAMPLES):
    body = json.dumps({
        "model": "qwen3.8-27b",
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0.0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        BASE.rstrip("/") + "/v1/completions", data=body,
        headers={"Authorization": "Bearer " + KEY,
                 "Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    dt = time.perf_counter() - t0
    pt = d["usage"]["prompt_tokens"]
    rec = {"run": "warmup" if i == 0 else f"run{i}",
           "prompt_tokens_reported": pt,
           "latency_s": round(dt, 3),
           "prefill_tok_s": round(pt / dt, 1)}
    print(json.dumps(rec))
    if i > 0:
        results.append(pt / dt)

print(json.dumps({"summary": {"samples": SAMPLES,
                              "mean_prefill_tok_s": round(sum(results)/len(results), 1)}}))
