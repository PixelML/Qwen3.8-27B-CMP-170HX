#!/usr/bin/env python3
"""Convert sanitized JSONL receipts into a tidy results.csv for the
Qwen3.8-27B Q8 PCIe scaling baseline.

Rules (see METHODOLOGY.md):
  * one CSV row per request receipt row (failed rows are preserved, never
    dropped),
  * missing values are written as empty cells — nothing is invented, no
    imputation, no interpolation,
  * cell_cold_samples / cell_warm_samples disclose how many requests were
    actually observed per (run, bucket, phase), so reduced repetitions are
    visible in the committed artifact,
  * monitor aggregates are matched to runs by card count (explicit
    `file.jsonl:run_id` pairing wins when provided); unmatched runs leave
    the resource columns empty.

Standard library only.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

CSV_COLUMNS = [
    "run_id",
    "schema_version",
    "card_count",
    "context_bucket",
    "repetition",
    "phase",
    "cell_cold_samples",
    "cell_warm_samples",
    "success",
    "error_class",
    "prompt_tokens",
    "output_tokens",
    "requested_output_tokens",
    "latency_s",
    "ttft_s",
    "prompt_throughput_tok_s",
    "generation_throughput_tok_s",
    "aggregate_output_throughput_tok_s",
    "response_chars",
    "monitor_samples",
    "gpu_count_monitored",
    "peak_power_w",
    "mean_power_w",
    "peak_memory_used_mib",
    "peak_gpu_util_pct",
    "peak_temperature_c",
    "throttle_samples_nonzero",
    "pcie_link_gen_min",
    "pcie_link_width_min",
]

BUCKET_ORDER = {"1024": 0, "8192": 1, "32768": 2}


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise SystemExit(
                    f"{path}:{lineno}: invalid JSON ({error}); refusing to "
                    "summarize a corrupt receipt — preserve it and classify "
                    "the run instead"
                )
    return records


def aggregate_monitor(records: list[dict]) -> dict[str, object]:
    samples = [r for r in records if r.get("type") == "gpu_sample"]
    powers = [r["power_w"] for r in samples if isinstance(r.get("power_w"), (int, float))]
    mems = [
        r["memory_used_mib"]
        for r in samples
        if isinstance(r.get("memory_used_mib"), (int, float))
    ]
    utils = [
        r["utilization_gpu_pct"]
        for r in samples
        if isinstance(r.get("utilization_gpu_pct"), (int, float))
    ]
    temps = [
        r["temperature_gpu_c"]
        for r in samples
        if isinstance(r.get("temperature_gpu_c"), (int, float))
    ]
    gens = [
        r["pcie_link_gen_current"]
        for r in samples
        if isinstance(r.get("pcie_link_gen_current"), (int, float))
    ]
    widths = [
        r["pcie_link_width_current"]
        for r in samples
        if isinstance(r.get("pcie_link_width_current"), (int, float))
    ]
    throttled = [
        r
        for r in samples
        if isinstance(r.get("throttle_reasons_bitmask"), int)
        and r["throttle_reasons_bitmask"] != 0
    ]
    selected = records[0].get("gpus_selected") if records else None
    distinct_gpus = {r.get("gpu_index") for r in samples if "gpu_index" in r}
    return {
        "monitor_samples": len(samples),
        "gpu_count_monitored": (
            len(selected)
            if isinstance(selected, list) and selected
            else len(distinct_gpus)
        ),
        "peak_power_w": max(powers) if powers else None,
        "mean_power_w": round(sum(powers) / len(powers), 2) if powers else None,
        "peak_memory_used_mib": max(mems) if mems else None,
        "peak_gpu_util_pct": max(utils) if utils else None,
        "peak_temperature_c": max(temps) if temps else None,
        "throttle_samples_nonzero": len(throttled),
        "pcie_link_gen_min": min(gens) if gens else None,
        "pcie_link_width_min": min(widths) if widths else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "receipts", nargs="+", type=Path, help="sanitized receipt JSONL file(s)"
    )
    parser.add_argument(
        "--monitor",
        action="append",
        default=[],
        type=Path,
        help="monitor JSONL; optionally `file.jsonl:run_id` to pin the pairing",
    )
    parser.add_argument("--csv", required=True, type=Path)
    args = parser.parse_args()

    # --- receipts -----------------------------------------------------------
    request_rows: list[dict] = []
    run_card_counts: dict[str, int] = {}
    run_order: list[str] = []
    for path in args.receipts:
        for record in read_jsonl(path):
            if record.get("type") == "run_start":
                run_id = record.get("run_id")
                if run_id and run_id not in run_card_counts:
                    run_card_counts[run_id] = int(record.get("card_count", 0))
                    run_order.append(run_id)
            elif record.get("type") == "request":
                request_rows.append(record)
                run_id = record.get("run_id")
                if run_id and run_id not in run_card_counts:
                    run_card_counts[run_id] = int(record.get("card_count", 0))
                    run_order.append(run_id)

    if not request_rows:
        raise SystemExit("no request rows found in the provided receipts")

    # observed samples per (run, bucket, phase) — disclosed, never padded
    cell_counts: dict[tuple[str, str, str], int] = {}
    for row in request_rows:
        key = (
            str(row.get("run_id")),
            str(row.get("context_bucket")),
            str(row.get("phase")),
        )
        cell_counts[key] = cell_counts.get(key, 0) + 1

    # --- monitor aggregates, paired to runs ---------------------------------
    monitor_pool: list[dict[str, object]] = []
    pinned: dict[str, dict[str, object]] = {}
    for entry in args.monitor:
        spec = str(entry)
        run_id = None
        if ":" in spec:
            file_part, run_id = spec.rsplit(":", 1)
            path = Path(file_part)
        else:
            path = Path(spec)
        agg = aggregate_monitor(read_jsonl(path))
        if run_id:
            pinned[run_id] = agg
        else:
            monitor_pool.append(agg)

    monitor_by_run: dict[str, dict[str, object]] = dict(pinned)
    for run_id in run_order:
        if run_id in monitor_by_run:
            continue
        cards = run_card_counts.get(run_id, 0)
        for index, agg in enumerate(monitor_pool):
            if agg.get("gpu_count_monitored") == cards:
                monitor_by_run[run_id] = monitor_pool.pop(index)
                break

    # --- tidy rows ----------------------------------------------------------
    request_rows.sort(
        key=lambda r: (
            str(r.get("run_id")),
            int(r.get("card_count", 0)),
            BUCKET_ORDER.get(str(r.get("context_bucket")), 99),
            0 if r.get("phase") == "cold" else 1,
            int(r.get("repetition", 0)),
        )
    )

    def cell(value: object) -> object:
        return "" if value is None else value

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(CSV_COLUMNS)
        for row in request_rows:
            run_id = str(row.get("run_id"))
            bucket = str(row.get("context_bucket"))
            phase = str(row.get("phase"))
            agg = monitor_by_run.get(run_id, {})
            writer.writerow(
                [
                    run_id,
                    row.get("schema_version", ""),
                    row.get("card_count", ""),
                    bucket,
                    row.get("repetition", ""),
                    phase,
                    cell_counts.get((run_id, bucket, "cold"), 0),
                    cell_counts.get((run_id, bucket, "warm"), 0),
                    "True" if row.get("success") else "False",
                    cell(row.get("error_class")),
                    cell(row.get("prompt_tokens")),
                    cell(row.get("output_tokens")),
                    cell(row.get("requested_output_tokens")),
                    cell(row.get("latency_s")),
                    cell(row.get("ttft_s")),
                    cell(row.get("prompt_throughput_tok_s")),
                    cell(row.get("generation_throughput_tok_s")),
                    cell(row.get("aggregate_output_throughput_tok_s")),
                    cell(row.get("response_chars")),
                    cell(agg.get("monitor_samples")),
                    cell(agg.get("gpu_count_monitored")),
                    cell(agg.get("peak_power_w")),
                    cell(agg.get("mean_power_w")),
                    cell(agg.get("peak_memory_used_mib")),
                    cell(agg.get("peak_gpu_util_pct")),
                    cell(agg.get("peak_temperature_c")),
                    cell(agg.get("throttle_samples_nonzero")),
                    cell(agg.get("pcie_link_gen_min")),
                    cell(agg.get("pcie_link_width_min")),
                ]
            )
    print(f"wrote {args.csv} ({len(request_rows)} request rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
