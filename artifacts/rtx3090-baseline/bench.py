#!/usr/bin/env python3
"""v9 benchmark protocol against a local server: tokenize for prompt size,
SSE completions with include_usage, temp 0, ignore_eos; 256/900-token decode
and 8k prefill; GPU1 peak sampling. Usage: bench.py <port> <api_key> <tag>"""
import json, subprocess, sys, threading, time, urllib.request

BASE_URL = "http://127.0.0.1:%s" % sys.argv[1]
KEY = sys.argv[2]
TAG = sys.argv[3] if len(sys.argv) > 3 else "run"
HDR = {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"}
peak = {"util": 0, "power": 0, "mem": 0, "clk": 0, "temp": 0}
stop = False

def sample():
    global stop
    while not stop:
        try:
            out = subprocess.run(["nvidia-smi", "-i", "1",
                                  "--query-gpu=utilization.gpu,power.draw,memory.used,clocks.sm,temperature.gpu",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.strip()
            u, p, m, c, t = map(float, out.split(","))
            peak["util"] = max(peak["util"], u); peak["power"] = max(peak["power"], p)
            peak["mem"] = max(peak["mem"], m); peak["clk"] = max(peak["clk"], c)
            peak["temp"] = max(peak["temp"], t)
        except Exception:
            pass
        time.sleep(1)

threading.Thread(target=sample, daemon=True).start()

def post(path, body, timeout=600):
    req = urllib.request.Request(BASE_URL + path, data=json.dumps(body).encode(), headers=HDR)
    return urllib.request.urlopen(req, timeout=timeout)

def ntokens(prompt):
    with post("/tokenize", {"model": "qwen3.8-27b", "prompt": prompt}) as r:
        return len(json.load(r)["tokens"])

def req_stream(prompt, maxtok):
    body = {"model": "qwen3.8-27b", "prompt": prompt, "max_tokens": maxtok,
            "temperature": 0.0, "ignore_eos": True, "stream": True,
            "stream_options": {"include_usage": True}}
    t0 = time.perf_counter(); ttft = None; n = 0
    ctok = None
    with post("/v1/completions", body) as resp:
        for line in resp:
            if line.startswith(b"data: "):
                payload = line[6:].strip()
                if payload == b"[DONE]":
                    break
                if ttft is None:
                    ttft = time.perf_counter() - t0
                n += 1
                try:
                    obj = json.loads(payload)
                    u = obj.get("usage") or {}
                    if u.get("completion_tokens") is not None:
                        ctok = u["completion_tokens"]
                except Exception:
                    pass
    if ctok is None:
        ctok = n
    return ttft, time.perf_counter() - t0, ctok

def bench(name, prompt, maxtok, warm, samples, ptok):
    for i in range(warm):
        req_stream(prompt, maxtok)
    res = []
    for i in range(samples):
        ttft, total, n = req_stream(prompt, maxtok)
        print(json.dumps({"tag": TAG, "run": name, "i": i, "ttft_ms": round(ttft * 1000, 1), "total_s": round(total, 3),
                          "out_tok": n, "decode_tok_s": round((n - 1) / (total - ttft), 2)}), flush=True)
        res.append((ttft, total, n))
    ttft = sum(r[0] for r in res) / len(res) * 1000
    tot = sum(r[1] for r in res) / len(res)
    nn = sum(r[2] for r in res) / len(res)
    print(json.dumps({"tag": TAG, "summary": name, "prompt_tokens": ptok, "mean_ttft_ms": round(ttft, 1),
                      "mean_total_s": round(tot, 3), "mean_out_tok": round(nn, 1),
                      "decode_tok_s": round((nn - 1) / (tot - ttft / 1000), 2),
                      "prefill_tok_s": round(ptok / (ttft / 1000), 1)}), flush=True)

P256 = "Write a story about a robot who learns to paint."
LONG = ("summarize the following text. " +
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega ") * 200
t256 = ntokens(P256); tlong = ntokens(LONG)
print(json.dumps({"tag": TAG, "prompt_tokens": {"story": t256, "long": tlong}}), flush=True)
bench("decode256", P256, 256, 1, 3, t256)
bench("decode900", P256, 900, 1, 3, t256)
bench("prefill_long", LONG, 8, 1, 3, tlong)
time.sleep(2); stop = True; time.sleep(1)
print(json.dumps({"tag": TAG, "gpu1_peak": peak}), flush=True)
