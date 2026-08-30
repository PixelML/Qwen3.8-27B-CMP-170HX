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
    "worker_id",
    "worker_endpoint",
    "mode",
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
    "node_aggregate_output_tok_s",
    "fairness_min_tok_s",
    "fairness_max_tok_s",
    "fairness_spread_pct",
    "node_energy_j",
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
    per_gpu_powers: dict[object, list[float]] = {}
    for sample in samples:
        value = sample.get("power_w")
        if isinstance(value, (int, float)):
            per_gpu_powers.setdefault(sample.get("gpu_index"), []).append(value)
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
        "per_gpu_mean_power_w": {
            gpu: round(sum(values) / len(values), 2)
            for gpu, values in sorted(per_gpu_powers.items(), key=lambda item: str(item[0]))
        },
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
    node_windows: dict[tuple[str, str, str, int], dict[str, float]] = {}
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
            elif record.get("type") == "window":
                window_key = (
                    str(record.get("run_id")),
                    str(record.get("context_bucket")),
                    str(record.get("phase")),
                    int(record.get("repetition", 0)),
                )
                node_windows[window_key] = {
                    "start": float(record.get("window_started_perf", 0.0)),
                    "end": float(record.get("window_ended_perf", 0.0)),
                }

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

    # --- node4 per-window aggregates: throughput, fairness, energy ----------
    node4_groups: dict[tuple[str, str, str, int], list[dict]] = {}
    for row in request_rows:
        if row.get("mode") != "node4_card_local":
            continue
        group_key = (
            str(row.get("run_id")),
            str(row.get("context_bucket")),
            str(row.get("phase")),
            int(row.get("repetition", 0)),
        )
        node4_groups.setdefault(group_key, []).append(row)

    node4_metrics: dict[tuple[str, str, str, int], dict[str, object]] = {}
    for group_key, rows in node4_groups.items():
        metrics: dict[str, object] = {
            "node_aggregate_output_tok_s": None,
            "fairness_min_tok_s": None,
            "fairness_max_tok_s": None,
            "fairness_spread_pct": None,
        }
        generation = [
            float(row["generation_throughput_tok_s"])
            for row in rows
            if row.get("success")
            and isinstance(row.get("generation_throughput_tok_s"), (int, float))
        ]
        window = node_windows.get(group_key)
        if window and generation:
            duration = window["end"] - window["start"]
            if duration > 0:
                metrics["node_aggregate_output_tok_s"] = round(
                    sum(generation) / duration, 2
                )
        if generation:
            gen_min = min(generation)
            gen_max = max(generation)
            gen_mean = sum(generation) / len(generation)
            metrics["fairness_min_tok_s"] = round(gen_min, 2)
            metrics["fairness_max_tok_s"] = round(gen_max, 2)
            if gen_mean > 0:
                metrics["fairness_spread_pct"] = round(
                    (gen_max - gen_min) / gen_mean * 100, 2
                )
        node4_metrics[group_key] = metrics

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
            int(r.get("worker_id", 0)),
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
            mode = str(row.get("mode", ""))
            worker_id = row.get("worker_id")
            node_metrics: dict[str, object] = {}
            node_energy_j: object = None
            if mode == "node4_card_local":
                group_key = (
                    run_id,
                    bucket,
                    phase,
                    int(row.get("repetition", 0)),
                )
                node_metrics = node4_metrics.get(group_key, {})
                window = node_windows.get(group_key)
                if window and worker_id is not None:
                    duration = window["end"] - window["start"]
                    per_gpu_power = agg.get("per_gpu_mean_power_w") or {}
                    ordinals = sorted(per_gpu_power, key=lambda item: str(item))
                    if duration > 0 and len(ordinals) >= int(worker_id):
                        power = per_gpu_power[ordinals[int(worker_id) - 1]]
                        if isinstance(power, (int, float)):
                            node_energy_j = round(power * duration, 1)
            writer.writerow(
                [
                    run_id,
                    row.get("schema_version", ""),
                    row.get("card_count", ""),
                    bucket,
                    row.get("repetition", ""),
                    phase,
                    cell(worker_id),
                    cell(row.get("worker_endpoint")),
                    mode,
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
                    cell(node_metrics.get("node_aggregate_output_tok_s")),
                    cell(node_metrics.get("fairness_min_tok_s")),
                    cell(node_metrics.get("fairness_max_tok_s")),
                    cell(node_metrics.get("fairness_spread_pct")),
                    node_energy_j,
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
