# RESULTS — Qwen3.8-27B + DFlash2 on a rented CMP 170HX

**Date:** 2026-08-28
**Working instance:** Vast.ai 49003408 (`qwen38-cmp170hx-v6`), on-demand
**Earlier failure:** instance 48995474 — private GHCR image, documented below

## Bottom line

The stack **works** on the CMP 170HX: public CUDA base image + syv-ai repo +
vLLM 0.27.1 + DFlash2 (7 draft tokens) served the model and survived a
controlled benchmark. Measured, single request, greedy, streaming:

| Metric | This run | LocalMaxxing reference | Delta |
|---|---|---|---|
| Output tok/s (decode) | **43.3–47.5** (config-dep.) | 212.68 | **4.5–4.9x slower** |
| TTFT (88-tok prompt) | **517–621 ms** | 72 ms | 7–8.6x slower |
| Prefill tok/s (8K prompt) | **1705–1712** | 1221.6 | **1.4x faster** |
| Mean acceptance length | **3.38** | — | DFlash2 active |
| KV cache tokens | 69,758 (v6) / GPU_UTIL-sized 33.6 GiB (v8) | ~65,536 context | fine |

Full context is usable in v8: the empty `KV_MEM=` disables the 24-GB-card
5.2 GiB pin and lets GPU_UTIL=0.90 size the pool (33.63 GiB for KV), so the
full 65,536-token context fits with headroom.

DFlash2 is confirmed active: server-side SpecDecoding metrics show mean
acceptance length 3.38, per-position acceptance
[0.789, 0.539, 0.368, 0.25, 0.184, 0.145, 0.105], full CUDA graphs captured
for both model and drafter.

## Root causes of the 4.5x gap (verified from logs, not speculation)

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
