# Instance metadata — 48995474

Captured from the Vast API (`/api/v1/instances/`) on 2026-08-28 10:00 UTC.

| Field | Value |
|---|---|
| Instance ID | 48995474 |
| Label | qwen38-cmp170hx-repro |
| Host ID / machine ID | 512364 / 148507 |
| Status at capture | **stopped** (GPU billing ended) |
| Region | Portugal, PT |
| GPU | 1× NVIDIA CMP 170HX, 64 GB HBM2e |
| PCIe | Gen2 x4 (pcie_bw 1.5 GB/s) |
| Compute cap | 8.0 (total_flops 12.15 TFLOPS, dlperf 42.0) |
| GPU mem bandwidth | 1518.9 GB/s |
| Public IP | 94.61.203.156 |
| Port mapping | 18020 → 18020 |
| SSH proxy | ssh9.vast.ai:35474 |
| Image | ghcr.io/syv-ai/qwen38-27b-rtx3090:latest (**private — root cause**) |
| Image digest | n/a — image content never pulled (401); container FS ~1.2 MiB |
| Driver / CUDA (host) | 610.43.02 / CUDA 13.3 max |
| Host CPU | AMD Ryzen 9 3950X, 32 threads (16 effective), AVX |
| Host RAM | 64,216 MiB (instance mem limit 44.6 GiB) |
| Disk | 80 GB (billed with instance) |
| Host board / OS | B450 TOMAHAWK MAX / Ubuntu 24.04 |
| Pricing | on-demand: GPU $0.2667/hr + disk $0.0222/hr = $0.2889/hr |
| Host net | 891.5 Mbps down / 390.3 Mbps up |
| vLLM version | not reached (0.27.1 syv overlay intended) |
| Model revision | not reached (lued/Qwen3.8-27B-INT8-W8A16-MTP intended) |
| Draft revision | not reached (syvai/Qwen3.8-27B-DFlash2-W4A16 intended) |
| Download sizes/times | none — no download ever began |
| Restarts | ≥1 manual reboot; onstart applied twice (first EXTRA_ARGS malformed) |

## Config as captured (env + onstart)

- `extra_env`: MODEL/DRAFT/SPEC=dflash2/GPU_UTIL=0.90/MAX_LEN=65536/
  MAX_SEQS=1/DFLASH_TOKENS=7/PREFIX_CACHE=0/SPEC_ATTN=1, plus the malformed
  `EXTRA_ARGS="--dtype\\"` from the first attempt. (A throwaway benchmark
  `VLLM_API_KEY` value was present; redacted on principle.)
- The corrected onstart script launches `/app/venv/bin/vllm serve` with
  FLASH_ATTN, BF16 KV, 0.90 util, max-num-seqs 1, and the DFlash2 7-token
  speculative config — it never had a binary to launch because the image
  content was never pulled.

## Cost / runtime record (timestamped)

| Event | Timestamp (UTC) |
|---|---|
| Instance started | 2026-08-28 09:22:40 |
| Billed runtime (API `duration`) | 0.349 hr (≈ 20 min 56 s) |
| Container stopped | ≈ 2026-08-28 09:43:36 |
| Status at 10:00 UTC | stopped — GPU billing ended; disk accrues until destroy |

Charges while running: 0.349 hr × $0.2889/hr ≈ **$0.101**.
Disk while stopped: ≈ $0.00037/min until the instance is destroyed.
