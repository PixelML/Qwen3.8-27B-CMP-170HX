# RESULTS — failed run (no benchmark numbers)

**Run window:** 2026-08-28
**Instance:** Vast.ai 48995474 (on-demand)
**Outcome:** Server never healthy; benchmark never issued.

## Bottom line

No throughput, latency, VRAM, power, or speculation metrics were captured —
the container never ran the model server because **the GHCR image is private
and was never actually pulled**. Reporting zeros or estimates would be fiction;
the honest result is "no result."

## Root cause (verified)

1. `ghcr.io/syv-ai/qwen38-27b-rtx3090:latest` requires authentication:
   anonymous manifest request → **HTTP 401**.
2. The instance container filesystem was **~1.2 MiB** total — consistent with
   a stub/fallback filesystem, not the ~tens-of-GB image.
3. `/app` contained only `onstart.sh` and `ports.log`; `/tmp/qwen38` (the
   documented weight cache location for this image) **did not exist**.
4. The onstart script therefore had no vLLM to launch; port 18020 never
   listened; health checks failed; SSH access was separately broken
   (`authorized_keys` mode/ownership, "Permission denied").

## Failure timeline

| Step | Result |
|---|---|
| Create instance 48995474 (80 GB disk, port 18020→18020) | Instance entered running state |
| Initial env `EXTRA_ARGS` | Malformed (bad quoting); replaced by an explicit onstart script |
| Onstart applied, instance rebooted | Onstart could not matter — image content absent |
| Health probes `http://94.61.203.156:18020/health` | Connection refused (sandboxed probes) and timeout (unsandboxed probes) |
| SSH via mapped proxy `ssh9.vast.ai:35474` | Permission denied (publickey) |
| authorized_keys fix attempt (chmod 700/600 on home/.ssh) | Applied, SSH still refused |
| GHCR anonymous manifest probe | **401 — image private (root cause)** |
| Container filesystem inspection | ~1.2 MiB; /app = onstart.sh, ports.log; /tmp/qwen38 absent |

## Metric table (honest version)

| Metric | Value | Notes |
|---|---|---|
| Output tok/s | — not measured | no server |
| Prefill tok/s | — not measured | no server |
| TTFT | — not measured | no server |
| Total tok/s | — not measured | no server |
| Peak VRAM | — not measured | no server |
| Power / temp | — not measured | no server |
| DFlash2 acceptance | — not measured | no server |
| Errors | image-private 401; health unreachable; SSH denied | see artifacts |

## Comparison vs LocalMaxxing reference

Reference: 212.68 output tok/s / 1221.6 prefill tok/s / 72 ms TTFT.
This run: **not comparable** — zero requests were served. The CMP 170HX
hardware itself was never exercised by the model stack.

## Cost / runtime record

- Rate: $0.2889/hr on-demand ($0.2667 GPU + $0.0222 disk), 80 GB disk
- Billed runtime: **0.349 hr** (started 2026-08-28 09:22:40 UTC, stopped ≈09:43:36 UTC)
- Charges while running: ≈ **$0.101**
- Disk while stopped: ≈ $0.00037/min until destroy
- Instance is **stopped** (GPU billing ended); destroy pending parent decision

## What to change before retrying

1. Pre-check the image with an anonymous manifest request (2 seconds, free).
2. Either publish a public mirror of the image or attach GHCR pull creds.
3. Only then rent; keep the same launch flags captured in
   [artifacts/benchmark-spec.md](artifacts/benchmark-spec.md).
