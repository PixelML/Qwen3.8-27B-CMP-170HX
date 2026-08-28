# Qwen3.8-27B INT8 + DFlash2 speculative decoding on a Vast.ai CMP 170HX — failed run report

**Date:** 2026-08-28
**Status:** ❌ Benchmark not executed — rental never reached a healthy state
**Cost basis:** on-demand, ~$0.289/hr (includes 80 GB disk)

## Summary

We attempted to reproduce the LocalMaxxing CMP 170HX result (212.68 output tok/s,
1221.6 prefill tok/s, 72 ms TTFT) for **lued/Qwen3.8-27B-INT8-W8A16-MTP** with
**syvai/Qwen3.8-27B-DFlash2-W4A16** as the draft model, on a single on-demand
Vast.ai instance (ID 48995474, Portugal, 1× CMP 170HX 64 GB).

The server never became healthy. The verified root cause:

> **The image `ghcr.io/syv-ai/qwen38-27b-rtx3090:latest` is private on GHCR.**
> An anonymous manifest request returned **HTTP 401**. Vast still created and
> started the instance, but the container filesystem was only **~1.2 MiB**, with
>`/app` containing only `onstart.sh` and `ports.log`; `/tmp/qwen38` was absent.
> No model, no vLLM, no server — nothing to benchmark.

## What we measured

Nothing. No vLLM server started, so there are no tok/s, TTFT, prefill, VRAM,
power, temperature, or speculation-acceptance numbers. We are reporting the
failure evidence rather than inventing metrics. See [RESULTS.md](RESULTS.md).

## Evidence index

| File | What it shows |
|---|---|
| [artifacts/image-manifest-probe.md](artifacts/image-manifest-probe.md) | Anonymous GHCR manifest request → 401 (image is private) |
| [artifacts/container-filesystem.md](artifacts/container-filesystem.md) | ~1.2 MiB filesystem; /app only onstart.sh + ports.log; /tmp/qwen38 absent |
| [artifacts/health-probes.md](artifacts/health-probes.md) | http://94.61.203.156:18020/health unreachable (connection refused / timeout) |
| [artifacts/ssh-auth.md](artifacts/ssh-auth.md) | SSH Permission denied; authorized_keys mode/ownership issue; fix attempted, still failing |
| [artifacts/instance-metadata.md](artifacts/instance-metadata.md) | Instance config as known; parent-supplied metadata pending where marked |
| [artifacts/vast-instance-48995474.log](artifacts/vast-instance-48995474.log) | Raw container log (87 KB): SSH shim only; no image pull, no vLLM |
| [artifacts/benchmark-spec.md](artifacts/benchmark-spec.md) | Intended launch flags, env, and request payload (never executed) |

## Hardware caveats (why this GPU was interesting)

The CMP 170HX is a mining card repurposed for LLM inference:

- 64 GB HBM2e with very high memory bandwidth (~1.5 TB/s class)
- **PCIe Gen2 x4 host link only** — model download/weight load is slow; fine
  once weights are resident, painful for cold starts
- **No FP8/FP4 tensor-core paths** (compute capability 8.6-ish, stripped SKU) —
  W8A16 main + W4A16 draft is roughly the right quantization envelope
- No display outputs, fan/power quirks vary by vendor

## Reproducing the *diagnosis*

```bash
# 1. Confirm the image is private (expect 401 for anonymous access)
TOKEN=$(curl -s "https://ghcr.io/token?scope=repository:syv-ai/qwen38-27b-rtx3090:pull" | jq -r .token)
curl -s -o /dev/null -w "%{http_code}\n" \
  -H "Authorization: Bearer $TOKEN" \
  https://ghcr.io/v2/syv-ai/qwen38-27b-rtx3090/manifests/latest

# 2. Probe the mapped public port (expect connection refused/timeout)
curl -sS --max-time 8 -w "HTTP=%{http_code} time=%{time_total}\n" \
  http://94.61.203.156:18020/health
```

See [scripts/probe.sh](scripts/probe.sh) for the full probe sequence used.

## Lessons / what a retry needs

1. **Verify image pullability before renting.** A 401 from the anonymous
   manifest check (above) predicts this exact failure in seconds, for free.
2. **Skip the image entirely.** Start from any public CUDA base image, then
   `pip install vllm==0.27.1` and apply the patches from
   github.com/syv-ai/qwen38-27b-rtx3090 in the onstart script — the author's
   own instructions. No private registry involved.
3. If the image must be used, attach a GHCR pull credential to the instance
   (Vast supports registry auth via Docker config on creation).

## Reference target (from LocalMaxxing, not reproduced here)

| Metric | Reference |
|---|---|
| Output throughput | 212.68 tok/s |
| Prefill throughput | 1221.6 tok/s |
| TTFT | 72 ms |
| VRAM used | 54.5 GB of 64 GB |
| llama.cpp stock baseline | ~70 tok/s (for contrast) |
| Output tokens / request | 256 |
| Prompt tokens | 88 |
| Context length | 65,536 |
| Stack | vLLM 0.27.1 syv overlay, DFlash2, 7 draft tokens |
| Main / draft model | lued/Qwen3.8-27B-INT8-W8A16-MTP / syvai/Qwen3.8-27B-DFlash2-W4A16 |
| KV cache | BF16; FLASH_ATTN; GPU util 0.90; max seqs 1 |

No numbers on this page were measured by us. This run produced no metrics.

## References

- LocalMaxxing run page:
  <https://www.localmaxxing.com/en/models/lued/Qwen3.8-27B-INT8-W8A16-MTP?run=cmt7y3mm301v1nn01sxj7ptwi>
- Dual Channel Labs announcement (origin of the target numbers):
  <https://x.com/bob_hw_store/status/2092962836934705501> and
  <https://x.com/bob_hw_store/status/2092962838822150508>
- The overlay/patches repo the image was built from:
  <https://github.com/syv-ai/qwen38-27b-rtx3090>

Per the author, the intended install is **not** the GHCR image: it's
`pip install vllm==0.27.1` plus applying the patches from the GitHub repo
above (that's what backports DFlash2 and the Ampere-specific fixes). The GHCR
image that defeated this run was only a prebuilt convenience artifact — and it
is private.
