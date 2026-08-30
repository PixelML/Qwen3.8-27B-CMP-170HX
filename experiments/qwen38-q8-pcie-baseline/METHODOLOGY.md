# METHODOLOGY — Qwen3.8-27B Q8 whole-node throughput baseline

Complete measurement contract for the fixed baseline round described in
`PLAN.md`. If this document and an implementation disagree, this document
wins and the deviation is recorded in `MANIFEST.md`.

## 1. Measurement definitions

All timings are client-observed wall-clock over one OpenAI-compatible
completions request with streaming enabled:

| Metric | Definition |
|---|---|
| TTFT | Time from request dispatch to the first SSE event carrying non-empty generated text. Streaming exists **only** to observe TTFT. |
| E2E latency | Time from request dispatch to the terminal SSE event. |
| Prompt-processing throughput | `prompt_tokens / TTFT` (tok/s) — prefill-rate proxy. |
| Generation throughput | `(output_tokens − 1) / (E2E − TTFT)` (tok/s) — post-first-token decode rate. |
| Aggregate output throughput | `output_tokens / E2E` (tok/s) — whole-call rate. |

Token counting policy: **always from the final usage object**
(`stream_options.include_usage`), never from SSE event counts — event
counting is a known failure mode of speculative stacks and is prohibited
here (see repo history for why). `ignore_eos` is set so output length is
exactly the requested 256 tokens; `usage.completion_tokens` remains the
recorded truth.

Prompt identity: deterministic public text, sized to nominal 1,024 / 8,192 /
32,768-token buckets with a documented 4-chars-per-token heuristic. The
nominal bucket is a *label*; the exact per-request `usage.prompt_tokens` is
recorded and used in every computed metric.

Phase policy: per phase, per bucket — the first request after the health
gate and settle delay is the **cold** measurement (first-touch weight
page-ins, KV pool allocation, compile caches); the next 3 byte-identical
requests are **warm repetitions**. The designated cold measurement is
exactly one per bucket; there is no "warm-up discard" beyond it and no
re-rolls.

## 2. Aggregation rules

- Headline numbers are means over the successful **warm** repetitions of a
  (phase, bucket) cell. Cold measurements are reported separately and
  never averaged into warm aggregates.
- Min–max spread across warm repetitions accompanies every mean (charts
  render it as a band/error bars).
- Failed rows are never dropped and never imputed. A cell with 2 observed
  warm reps is aggregated over those 2 and **disclosed** via
  `cell_warm_samples` in `results.csv` and `n=` labels in charts.
- A cell with 0 successful warm reps contributes no aggregate at all — the
  failed rows remain visible in the CSV with their error class and empty
  metrics.
- Cross-phase comparisons (control vs node4) use the same bucket and phase only.
- Resource aggregates (peak/mean power, peak memory, peak utilization,
  peak temperature, nonzero-throttle sample count, min effective PCIe
  link gen/width) are computed per phase from the paired monitor JSONL and
  attached to every row of that phase.
- node4 whole-node aggregate: within one shared request window (identical
  bucket/phase/repetition across the four workers), node throughput equals
  the sum of worker generation rates divided by the shared window duration;
  it is reported per bucket in `node_aggregate_output_tok_s`.
- Fairness: per-window min, max, and percent spread across the four workers
  accompany every node4 mean; interference is the per-bucket ratio of node
  aggregate throughput to 4x the single-card control.
- Aggregate throughput across cells (e.g. a single "the 4-card number") is
  not defined by this methodology; always report per bucket.

## 3. Quality-smoke method and limitations

Method: after each phase completes, read the recorded `response_text`
for the 32,768-token bucket's warm repetitions end-to-end and check:

1. determinism — warm repetitions at temperature 0 should be byte-identical
   (a mismatch is recorded as a deviation, not silently re-run),
2. coherence — a human skim for degenerate loops, immediate EOS artifacts,
   or corrupted segments in the first and last ~50 tokens,
3. length sanity — `output_tokens == 256` and `response_chars` in a
   plausible band vs the warm median.

Limitations (explicit): this is a human smoke check, not a benchmark
score. It has no recall guarantee, it cannot detect subtle quality loss,
and it is reader-dependent. It exists to catch gross serving corruption,
never to rank runs. No perplexity or downstream-task score is produced by
this round.

## 4. Environment provenance

Recorded per round in `MANIFEST.md`:

- pinned versions (vLLM, Python, Torch, Flash Infer, driver) — the values
  there are verified pins for this round,
- TODO-at-runtime fields: model revision, resolved checkpoint hash(es),
  serving recipe commit, package provenance (image or environment),
  effective per-card PCIe link state from the monitor output, actual UTC
  timestamps for each phase run, and any deviations.

The operator lease acknowledgement mechanism (env var contract in
`run_matrix.sh`) is recorded as acknowledged/mechanism — the value itself is
not a secret and is fixed in the script.

## 5. Safety thresholds

Enforced by the harness watchdog (see `PLAN.md`); restated here as the
authoritative reference:

| Signal | Threshold | Action |
|---|---|---|
| Core temperature | ≥ 88 °C on any active card, 3 consecutive samples 10 s apart | abort run (`safety_stop`), preserve receipts |
| Power draw | > 200 W sustained, 3 consecutive samples 10 s apart | abort run (`safety_stop`) — the 180 W policy allows sub-second transients, not sustained excursions |
| Uncorrected ECC | any increase over the run baseline | abort run (`safety_stop`) |
| Kernel Xid | any occurrence | operator stops the whole round and quarantines the node |
| Power policy | card power limit above 180 W at preflight | refuse to start (override records a deviation) |

## 6. Failure classification

Stable classes (recorded per row in receipts / CSV):

- Preflight (run never starts): `preflight_*` — gpu visibility, compute
  busy, storage missing/unreadable, free space, power policy, class
  mismatch, lease missing, port busy, lock held.
- Server lifecycle: `server_start_failed` (never healthy),
  `server_unhealthy` (died mid-round).
- Per-request client classes: `http_error`, `timeout`, `connection_error`,
  `stream_truncated`, `missing_usage`, `no_first_token`,
  `invalid_response`, `interrupted`.
- Round-level: `safety_stop`, `interrupted`, `completed`.

Preservation rule: every classified failure produces committed artifacts
(receipts + CSV rows + a note), regardless of whether the round eventually
succeeded on retry.

## 7. Data sanitation procedure

Receipts are sanitized by construction, but everything is re-checked before
commit:

1. Copy only `.receipt.jsonl` and `.monitor.jsonl` from the run directory —
   never `server.log`, `metadata.txt`, or `watchdog.txt`.
2. Review the receipt: confirm zero occurrences of hostnames, addresses,
   UUIDs, serials, tokens, and absolute private paths.
3. Run the mechanical scan over staged files:
   ```bash
   grep -rniE '([0-9]{1,3}\.){3}[0-9]{1,3}|uuid|serial|serial=|bearer |api[_-]?key|/home/|/Users/|hostname' \
     experiments/qwen38-q8-pcie-baseline/receipts/
   ```
   Expected result: no matches (a match blocks the commit).
4. Confirm `results.csv` and charts are derived only from sanitized inputs.
5. Record the commit in `MANIFEST.md` run record (timestamps + deviations).

Sanitization must never edit measured values; if a value is sensitive, the
whole file is withheld and the withholding is disclosed.

## 8. Publication checklist

- [ ] All three topologies attempted in ascending order; attempts and
      outcomes (including failures) committed.
- [ ] `MANIFEST.md` TODO fields filled from the live run; no field
      pre-filled from memory.
- [ ] Receipts committed per `receipts/README.md` naming and retention.
- [ ] `results.csv` regenerated by `summarize.py` from the committed
      receipts (header identical to
      `results/qwen38-q8-pcie-baseline-PLACEHOLDER.csv`).
- [ ] Charts regenerated by `make_charts.py` from the committed CSV;
      reduced sample counts labeled.
- [ ] Sanitation grep pass clean over everything staged.
- [ ] `RESULTS.md` updated with: per-bucket warm means ± spread, cold
      values separately, deviations, quality-smoke outcome and its
      limitations, and links to receipts.
- [ ] Negative results stated as plainly as positive ones.
