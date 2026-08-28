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
| Output tok/s | **47.2–47.5** | 212.68 | **4.5x slower** |
| TTFT | **517–521 ms** | 72 ms | 7x slower |
| Prefill tok/s | not isolated | 1221.6 | — |
| Mean acceptance length | **3.38** | — | DFlash2 active |
| KV cache tokens | 69,758 | ~65,536 context | fine |

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
3. **W4A16 requant vs reference W8A16.** prepare.sh requantizes the INT8
   checkpoint to W4A16 for 24 GB cards. The reference used W8A16 — different
   kernel paths and possibly acceptance behavior.

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
