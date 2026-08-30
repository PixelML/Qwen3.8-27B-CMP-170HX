# MANIFEST — Qwen3.8-27B Q8 PCIe scaling baseline

One manifest per baseline round. Pins below are fixed for the round; TODO
fields are filled in **at run time** from the live environment and committed
with the receipts. Fields marked TODO must never be pre-filled from memory.

## Model

| Field | Value |
|---|---|
| Model family | Qwen3.8-27B |
| Quantization | Q8 / INT8 W8A16 (int8 weights, BF16 compute activations) |
| MTP head | BF16 MTP head present in the checkpoint; **speculative decoding disabled for this round** |
| Model revision | **TODO** (record exact revision/commit of the checkpoint before the first run) |
| Resolved commit hash(es) | **TODO** (record the resolved storage hash(es) of the checkpoint directory contents at run time) |
| Storage location class | canonical shared model storage (exact mount recorded locally only) |

## Runtime stack (pinned)

| Component | Version |
|---|---|
| Serving runtime | vLLM 0.27.1 |
| Python | 3.10 |
| Torch | 2.13.0 |
| Flash Infer | 0.6.16.post3 |
| NVIDIA driver | 610.43.03 |

Runtime provenance: **TODO** (record the exact commit of the serving recipe
and the package provenance — image or environment — actually used, at run
time; do not copy from a previous round).

## Hardware

| Field | Value |
|---|---|
| Node class | four-card CMP node |
| Accelerator | CMP 170HX class, 64 GiB per card |
| Card count topologies | 1, 2, 4 (even tensor parallelism) |
| Effective per-card PCIe state | **TODO** (record negotiated link generation/width per card from the live monitor output at run time — do not assume from the hardware datasheet) |
| Power policy | 180 W per card for the entire benchmark round |

## Client / protocol (fixed)

| Field | Value |
|---|---|
| API surface | OpenAI-compatible completions endpoint on the pinned server |
| Context buckets (nominal) | 1,024 / 8,192 / 32,768 prompt tokens |
| Requested output tokens | exactly 256 per request (EOS suppressed) |
| Sampling | temperature 0.0, top_p 1.0, fixed seed 1234, single stream |
| Repetitions | 1 designated cold + 3 warm per bucket per topology |
| Receipt schema | `qwen38-q8-pcie-baseline.v1` (JSONL, sanitized before commit) |

## Run record (filled at run time)

| Field | Value |
|---|---|
| Round start / end timestamps (UTC) | **TODO** |
| Topology run timestamps (UTC, 1 / 2 / 4 card) | **TODO** |
| Operator lease acknowledgement recorded | **TODO** (yes/no + mechanism) |
| Deviations from `PLAN.md`, if any | **TODO** (must be empty for a clean round) |

## Cross-references

- Fixed contract and run order: `PLAN.md`
- Measurement definitions and aggregation: `METHODOLOGY.md`
- Receipt layout and sanitation: `receipts/README.md`
