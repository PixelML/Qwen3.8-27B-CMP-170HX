#!/usr/bin/env python3
"""Legacy SSE-event-counting benchmark against the vLLM OpenAI API.

Do not use this harness for speculative-decoding throughput. One SSE event can
contain multiple accepted tokens, so this script undercounts output. Use
``scripts/bench-usage.py`` for usage-token-counted measurements.

Protocol matches the LocalMaxxing reference: ~88-token prompt, 256 output
tokens, greedy, streaming, one request at a time."""
import json, sys, time
import urllib.request

BASE = sys.argv[1]
KEY = sys.argv[2]
WARMUP, SAMPLES = 1, 3
PROMPT = "Write a story about a robot who learns to paint."

def one_request(stream=True):
    body = json.dumps({
        "model": "qwen3.8-27b",
        "prompt": PROMPT,
        "max_tokens": 256,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": stream,
    }).encode()
    req = urllib.request.Request(
        BASE.rstrip("/") + "/v1/completions", data=body,
        headers={"Authorization": "Bearer " + KEY,
                 "Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    chunks = 0
    with urllib.request.urlopen(req, timeout=300) as r:
        for line in r:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                break
            if ttft is None:
                ttft = time.perf_counter() - t0
            chunks += 1
    total = time.perf_counter() - t0
    return ttft, total, chunks

results = []
for i in range(WARMUP + SAMPLES):
    tag = "warmup" if i == 0 else f"run{i}"
    ttft, total, chunks = one_request()
    out_tps = (chunks - 1) / (total - ttft) if chunks > 1 else 0
    tot_tps = chunks / total
    print(json.dumps({"run": tag, "ttft_ms": round(ttft * 1000, 1),
                      "total_s": round(total, 3), "output_tokens": chunks,
                      "output_tok_s": round(out_tps, 2),
                      "total_tok_s": round(tot_tps, 2)}))
    sys.stdout.flush()
    if i > 0:
        results.append((ttft, total, chunks))

ttft = sum(r[0] for r in results) / len(results) * 1000
total = sum(r[1] for r in results) / len(results)
chunks = sum(r[2] for r in results) / len(results)
print(json.dumps({"summary": {"runs": len(results),
    "mean_ttft_ms": round(ttft, 1), "mean_total_s": round(total, 3),
    "mean_output_tokens": round(chunks, 1),
    "mean_output_tok_s": round((chunks - 1) / (total - ttft / 1000), 2) if chunks > 1 else 0,
    "mean_total_tok_s": round(chunks / total, 2)}}))
