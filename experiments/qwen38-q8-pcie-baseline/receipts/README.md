# Receipts — Qwen3.8-27B Q8 PCIe scaling baseline

This directory holds **sanitized, committed receipts only**. Raw run output
(server logs, local metadata, watchdog notes) stays on the run volume in the
harness's untracked run directory and is never committed.

## Layout

```
receipts/
  qwen38-q8-pcie-baseline.cards1.<UTC-stamp>.receipt.jsonl
  qwen38-q8-pcie-baseline.cards1.<UTC-stamp>.monitor.jsonl
  qwen38-q8-pcie-baseline.cards2.<UTC-stamp>.receipt.jsonl
  qwen38-q8-pcie-baseline.cards2.<UTC-stamp>.monitor.jsonl
  qwen38-q8-pcie-baseline.cards4.<UTC-stamp>.receipt.jsonl
  qwen38-q8-pcie-baseline.cards4.<UTC-stamp>.monitor.jsonl
```

- `<UTC-stamp>` is the run start time in `YYYYMMDDTHHMMSSZ` form, matching
  the run directory stamp and the `run_id` inside the file.
- `.receipt.jsonl` is the benchmark client output (sanitized by
  construction: `run_start`, per-request `request`, and `run_summary`
  records).
- `.monitor.jsonl` is the GPU telemetry (`monitor_start`, `gpu_sample`,
  optional `sample_error`, `monitor_stop` records; index-only GPU
  identification).

## Naming rules

- Never rename after commit; never overwrite an existing receipt. The UTC
  stamp makes collisions implausible — if one occurs, do not clobber:
  keep both and record the deviation in `MANIFEST.md`.
- Copy receipts from the run directory into this directory only after the
  sanitation checklist in `METHODOLOGY.md` passes.

## Retention rules

- Keep every receipt forever, including negative, partial, and interrupted
  runs — they are first-class results.
- Each committed receipt must be referenced by at least one committed
  `results.csv` row set or an accompanying note in `RESULTS.md` /
  `METHODOLOGY.md` explaining why it produced no CSV rows.
- Do not delete receipts to "clean up" history; supersede them with newer
  rounds instead.

## Prohibited content

A receipt must never contain:

- hostnames, node names, domain names, or network addresses of any kind,
- GPU UUIDs, serial numbers, PCI bus ids, or board identifiers,
- API keys, tokens, credentials, or their fragments,
- raw server logs or stack traces (classified error codes only),
- absolute private filesystem paths (storage roots and mount points are
  described generically, e.g. "canonical shared model storage"),
- model bytes or byte-derived blobs (counts and hashes recorded in
  `MANIFEST.md` are fine).

The client writes receipts that satisfy these rules by construction; the
monitor writes index-only telemetry by construction. Human copying is the
step that can introduce violations, which is why the pre-commit checklist in
`METHODOLOGY.md` includes a grep pass over everything staged.

## Negative / partial runs

If a phase fails or is interrupted, its receipt still gets committed with
the same naming scheme. Do not "retry until clean" and commit only the good
one: every attempt that produced receipts is preserved, and the summary
discloses how many attempts exist.
