# RESULTS — Qwen3.8-27B + DFlash2 on a rented CMP 170HX

**Date:** 2026-08-28
**Working instance:** Vast.ai 49011536 (`qwen38-v9c`), on-demand
**Earlier failure:** instance 48995474 — private GHCR image, documented below

## Bottom line

The stack **works** on the CMP 170HX: public CUDA base image + syv-ai repo +
vLLM 0.27.1 + DFlash2 (7 draft tokens) served the model and survived a
controlled benchmark. Measured, single request, greedy, streaming:

| Metric | v9 (fast variant) | v6/v8 (base target) | LocalMaxxing reference |
|---|---|---|---|
| Output tok/s (256-tok gen) | **147.7** | 43.3–47.5* | 212.68 |
| Output tok/s (900-tok gen) | **134.5** | — | — |
| TTFT (11-tok prompt) | **76 ms** | 517–621 ms* | 72 ms |
| Prefill tok/s (6.6K prompt) | **2156** | 1705–1712 | 1221.6 |
| Mean acceptance length | **2.56–2.80** | 2.51–3.38 | — |
| Peak GPU | 255 W, 1455 MHz, 57.8 GiB | — | — |

\*v6/v8 counted SSE events as tokens; with speculative decoding each event
carries ~2.5-3 tokens, so those numbers understate real throughput by roughly
the acceptance length. v9 counts `usage.completion_tokens`.

v9 decode (147.7 tok/s) matches the syv repo's own community 170HX datapoint
(133.7 tok/s median, 3x900 tok, greedy, fast target) and the 3090's ~120-133
tok/s single-stream. The repo's tables show 212 is an **8-stream aggregate**,
not single-stream; single-stream on a 3090 is ~120 tok/s. Against that
yardstick the 170HX at 147.7 tok/s is fully explained.
| KV cache tokens | 69,758 (v6) / GPU_UTIL-sized 33.6 GiB (v8) | ~65,536 context | fine |

Full context is usable in v8: the empty `KV_MEM=` disables the 24-GB-card
5.2 GiB pin and lets GPU_UTIL=0.90 size the pool (33.63 GiB for KV), so the
full 65,536-token context fits with headroom.

DFlash2 is confirmed active: server-side SpecDecoding metrics show mean
acceptance length 3.38, per-position acceptance
[0.789, 0.539, 0.368, 0.25, 0.184, 0.145, 0.105], full CUDA graphs captured
for both model and drafter.

## Root causes of the 4.5x gap (verified from logs, not speculation)

(Superseded by v9 — see "v9: what actually closed the gap" below. The three
items here were real but secondary.)

## v9: what actually closed the gap (verified from logs)

1. **The fast variant was never built in v6-v8.** `docker/prepare.sh`
   `cd /app` then calls `prepare/quant_lm_head.py`; our clone lives in
   `/app/qwen-serving`, so prepare.sh died at `can't open file
   '/app/prepare/quant_lm_head.py'` — silently (`|| echo continue`). No
   int8/packed lm_head, no int8 embed/MTP, and crucially no
   **fast variant** (int4-GPTQ lm_head/MTP, +15% decode). v9 symlinks
   `/app/prepare` -> the clone's prepare dir; the log then shows every
   quantization step completing and `FAST-VARIANT-PRESENT`.
2. **Token counting was wrong.** v6/v8/v7e counted SSE events. Spec decode
   emits one event per accepted batch (~2.5-3 tokens). v9 uses
   `stream_options: {include_usage: true}` and `usage.completion_tokens`.
   This alone turns 43.3 "tok/s" into ~110-130 real tok/s.
3. **flashinfer topk fixed.** v8's JIT failed (`curand.h: No such file`)
   -> torch.topk fallback. v9 symlinks curand.h from the pip nvidia cu13
   package into /usr/local/cuda/include and clears the flashinfer cache.

1. **KV pool mis-sized.** The launcher pinned `KV_MEM=5583457484` (5.2 GiB),
   a 24 GB-card value from the syv repo defaults. The log confirms:
   "reserved 5.2 GiB memory for KV Cache ... This does not respect the
   gpu_memory_utilization config." ~58 GiB of the 64 GB card sat idle.
2. **flashinfer topk fell back to torch.topk.** `/usr/local/cuda/bin/nvcc`
   missing in the base image; flashinfer JIT could not compile its fast
   topk. The repo docs say flashinfer makes the DFlash2 selector ~2x faster.
3. **W4A16 vs W8A16 — tested directly (v7e).** Serving the official W8A16
   checkpoint dropped throughput to 31.6 tok/s (from 47.5 W4A16) and
   acceptance length to 2.9 (from 3.38). The card is bandwidth-bound, so
   denser weights are strictly slower. The reference's 212 tok/s cannot be
   explained by weight precision alone on this host.

## v6 configuration

- Image: `nvidia/cuda:13.0.1-base-ubuntu24.04` (public base, not the private GHCR)
- Stack: syv-ai/qwen38-27b-rtx3090 depth-1 clone, vLLM 0.27.1 + repo patches,
  flashinfer 0.6.16.post3, torch 2.13.0
- Launch: `SPEC=dflash2 CTX=fast MAX_SEQS=1 DFLASH_TOKENS=7 PORT=18020
  VLLM_V2_CUDAGRAPH_MEM_MIB=1400 KV_MEM=5583457484`
- Endpoint: http://94.61.203.156:40226 (host port 40226 <- container 18020)

## Benchmark protocol

Single request, greedy (temperature 0), streaming, 88-token prompt,
256 max tokens, ignore_eos. 1 warmup + 3 measured samples per run; three
runs total. Protocol matches the LocalMaxxing reference shape.

| Run | Mean output tok/s | Mean TTFT | Output tokens |
|---|---|---|---|
| 1 (ignore_eos) | 47.49 | 518.8 ms | 77 |
| 2 (eos honored) | 47.35 | 521.2 ms | 103 |
| 3 (repeat) | 47.22 | 517.5 ms | 103 |

v7e (W8A16, official recipe): **31.6 tok/s, 608-621 ms TTFT**, acceptance 2.9.
Raw: [bench-v7e.json](artifacts/bench-v7e.json). Attempt log:
[attempt-history-v7.md](artifacts/attempt-history-v7.md).

v8 (W4A16 + GPU_UTIL-sized KV pool, full 64K context): decode **43.3 tok/s /
610 ms TTFT**, prefill **1705-1712 tok/s on an 8,192-token prompt** (1 output
token, non-streaming, 2 samples each pass; repeat pass within 0.5%). The
prefill number beats the reference's 1221.6 by 1.4x — prefill is
compute-bound and this host's 170HX handles it well; decode remains
bandwidth-bound at ~1/4 of reference. Raw:
[bench-v8.json](artifacts/bench-v8.json).

v9 (W4A16 + **fast variant** + curand.h fix + correct token counting,
instance 49011536): decode **147.7 tok/s** (256-tok) / **134.5 tok/s**
(900-tok), TTFT **76 ms**, prefill **2156 tok/s** (6,603-tok prompt).
Peak: 57.8 GiB VRAM, 255 W, 1455 MHz SM clock, 73 C, 100% util.
Acceptance length 2.56-2.80. Raw: [bench-v9.json](artifacts/bench-v9.json),
samples: [bench-v9-samples.jsonl](artifacts/bench-v9-samples.jsonl),
full instance log: [v9-full.log](artifacts/v9-full.log).

### v9 benchmark samples (greedy, streaming, usage-token-counted)

| Run | i | TTFT | total | out tok | tok/s |
|---|---|---|---|---|---|
| decode256 | 1 | 75.9 ms | 1.803 s | 256 | 147.6 |
| decode256 | 2 | 75.7 ms | 1.804 s | 256 | 147.6 |
| decode900 | 1 | 93.4 ms | 6.779 s | 900 | 134.5 |
| decode900 | 2 | 93.0 ms | 6.797 s | 900 | 134.1 |
| prefill 6.6K | 1 | 3062.8 ms | 3.126 s | 8 | 2156 (prefill) |

Raw JSON: [bench-v6-run1.json](artifacts/bench-v6-run1.json),
[bench-v6-run2.json](artifacts/bench-v6-run2.json),
[bench-v6-run3.json](artifacts/bench-v6-run3.json).

## Attempt history

| # | Instance | Image | Outcome | Cost |
|---|---|---|---|---|
| v1 | 48995474 | private GHCR | image 401, never ran | $0.101 |
| v2 | 49001943 | public base + clone | prepare.sh path bugs | ~$0.11 |
| v3 | 49002210 | public base + clone | same | ~$0.05 |
| v4 | 49002551 | public base + clone | same | ~$0.03 |
| v5 | 49002984 | public base + clone | venv symlink fix | ~$0.05 |
| v6 | 49003408 | public base + clone | **working server + benchmark** | see below |
| v7e | 49006974 | public base + clone, W8A16 | working, 31.6 tok/s | ~$0.06 |
| v8 | 49008211 | public base + clone | working, 43.3 tok/s (event-counted) | ~$0.06 |
| v9a/v9b | 49011398/49011444 | create returned stopped | destroyed immediately | $0 |
| v9c | 49011536 | public base + clone, prepare fix | **147.7 tok/s, gap closed** | see below |

## Cost record

- v6 rate: $0.2944/hr on-demand; started ~11:05 UTC 2026-08-28
- v1: $0.101 (0.349 hr)
- Final total recorded here after teardown.

## Failure evidence from v1 (instance 48995474)

1. `ghcr.io/syv-ai/qwen38-27b-rtx3090:latest` requires auth: anonymous
   manifest request -> HTTP 401.
2. Container filesystem ~1.2 MiB; /app held only onstart.sh/ports.log.
3. SSH denied separately (authorized_keys mode/ownership on team account).

Details: [container-filesystem.md](artifacts/container-filesystem.md),
[image-manifest-probe.md](artifacts/image-manifest-probe.md).
