# PLAN — Qwen3.8-27B Q8 (INT8 W8A16, BF16 MTP head) whole-node throughput baseline

**Status:** contract fixed; no measurements taken yet.
**Harness:** this directory (`run_matrix.sh`, `bench_client.py`, `monitor_gpu.py`,
`summarize.py`, `make_charts.py`).
**Environment contract:** four-card CMP node, 64 GiB per card, canonical
shared model storage, 180 W benchmark power policy. See `MANIFEST.md` for the
pinned versions and `METHODOLOGY.md` for measurement definitions.

## Hypothesis

On a four-card CMP node with a weak per-card PCIe host link, serving
Qwen3.8-27B Q8 (INT8 W8A16, BF16 MTP head) as four simultaneous card-local
TP1 workers changes aggregate node throughput in a measurable, roughly
predictable way:

- **Whole-node generation throughput** (four workers, 256 output tokens
  each, single stream per worker) should approach roughly 4x the single-card
  control, with the shortfall quantifying cross-card interference and shared
  host-link contention.
- **Fairness** across the four identical workers should be high; per-card
  spread is reported explicitly rather than hidden by the mean.
- **Warm TTFT** at 32K prompt tokens is the most PCIe-sensitive metric and is
  expected to degrade relative to the single-card control when four workers
  share the host link at once.

The purpose of this run is to **measure** one short single-card control and
one four-worker whole-node round under one frozen configuration — not to tune
anything. The former sequential 1-to-2-to-4 TP scaling sweep and TP4
optimization are parked and out of scope for this pass.

## Fixed variable contract (frozen for the entire round)

Every run in this round uses exactly these settings. Changing any of them
invalidates comparability and requires a new plan version:

| Variable | Fixed value |
|---|---|
| Model | Qwen3.8-27B Q8 / INT8 W8A16, BF16 MTP head, one revision for the whole round (recorded in `MANIFEST.md` at run time) |
| Topology | Two phases only: (A) short single-card TP1 control; (B) four simultaneous card-local TP1 workers, one per physical card. No TP2/TP4 sweep. |
| Speculative decoding | **Off** for the entire round |
| Model length / context cap | 36,864 tokens (32,768-token prompt bucket + output + fixed margin) |
| Max concurrent sequences | 1 (single stream) |
| Memory utilization | 0.90 |
| Sampling (server) | Pinned server seed; sampling fixed client-side |
| Sampling (client) | temperature 0.0, top_p 1.0, fixed seed, exactly 256 requested output tokens, EOS suppression so the count is exact |
| Context buckets | nominal 1,024 / 8,192 / 32,768 prompt tokens (exact counts from server usage object) |
| Repetitions | 3 warm per bucket + 1 designated cold per bucket after startup |
| Power policy | 180 W per card for the whole round |
| Runtime stack | Pinned in `MANIFEST.md` (vLLM / Torch / Flash Infer / driver); one stack for the whole round |
| Prompt suite | Deterministic public text, identical across all topologies |

Explicitly out of scope for this round: tuning of any kind, alternate
checkpoints or revisions, power-limit changes, runtime or serving-stack
variants, speculative decoding of any configuration, concurrency sweeps
beyond the pinned single stream per worker, and any asymmetric topology.

## Phase order

Phases run **strictly one at a time, control first, then node4**:

1. `run_matrix.sh control` — short single-card control
2. `run_matrix.sh node4` — four card-local workers measured in one shared window

Rules:

- One phase must fully complete (or be classified as failed) and all of its
  workers must be shut down before the next invocation starts. The harness
  enforces this with a lock and refuses concurrent operation.
- control launches exactly one TP1 server; node4 launches exactly four
  card-local TP1 servers simultaneously. Servers are never reused across
  phases.
- node4 workers are started together and measured in one shared time window;
  per-card rows plus the shared window yield whole-node aggregate throughput
  and fairness/interference evidence against the control.
- If a phase fails, the operator decides whether to re-run it after
  recording the classified failure; receipts are always preserved.

## Cold / warm measurement policy

- After the server passes its health gate and a fixed settle delay, the
  **first request in each context bucket is the designated cold
  measurement** for that bucket (first-touch weights page-ins, KV pool
  allocation, compile caches).
- The following **3 requests per bucket are warm repetitions** with byte-
  identical payloads. Headline aggregates use warm repetitions only; cold
  measurements are reported separately and never mixed into warm means.
- Warm repetitions are expected to be near-identical at temperature 0;
  divergence beyond the spread documented in `METHODOLOGY.md` is a flag for
  the quality smoke, not a re-roll trigger. Re-rolls to "get a better
  number" are prohibited — failed or anomalous rows are preserved as-is.

## Safety stop thresholds

The harness watchdog aborts a phase run (classified `safety_stop`,
receipts preserved) when any active card shows:

| Signal | Threshold | Basis |
|---|---|---|
| Core temperature | ≥ 88 °C sustained (3 consecutive samples, 10 s apart) | thermal margin |
| Power draw | > 200 W sustained (3 consecutive samples, 10 s apart) | 180 W policy with known sub-second transients; sustained > 200 W means the cap is not effective |
| Uncorrected ECC | count increases during the run | memory integrity |
| Server process | exits or fails its health gate | classified `server_start_failed` / `server_unhealthy` |

Operator-level stop rule: any kernel Xid event during the run → stop the
round, quarantine the node, and record the classified failure. Do not
continue the matrix after a hardware-level stop.

## Classified failure policy

Every failure gets a stable class recorded in the receipts; raw logs stay on
the local run volume and are never committed. Classes:

- `preflight_*` — preflight gate refused the run (GPU visibility, compute
  busy, storage missing/unreadable, free space, power policy, lease
  acknowledgement, port busy). Nothing is measured; no server starts.
- `server_start_failed` — server never became healthy; partial receipts (if
  any) preserved.
- `server_unhealthy` — server died mid-round; remaining requests are not
  attempted; completed rows preserved.
- `safety_stop` — watchdog threshold tripped.
- `client_error` classes per request (`http_error`, `timeout`,
  `connection_error`, `stream_truncated`, `missing_usage`, `no_first_token`,
  `invalid_response`) — failed rows stay in the CSV with empty metrics.
- `interrupted` — operator or signal terminated the round; partial receipts
  preserved and usable.

Negative and partial results are first-class outputs: they are committed
like successful rounds and disclosed in any publication.

## Scope statement

This fixed matrix — 3 topologies × 3 context buckets × (1 cold + 3 warm) —
**is the complete baseline round**. There is no phase 2 hidden in this plan;
follow-up work requires a new plan and a new manifest.
