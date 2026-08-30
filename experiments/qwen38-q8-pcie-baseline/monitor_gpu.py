#!/usr/bin/env python3
"""Per-GPU telemetry sampler for the Qwen3.8-27B Q8 PCIe scaling baseline.

Samples every visible-and-selected GPU on a fixed interval and appends one
JSONL record per GPU per interval with: memory used, utilization, power
draw, core temperature, SM clock, active throttle-reason bitmask, uncorrected
ECC counter (when the stack exposes it), and the EFFECTIVE PCIe link
generation/width (current, plus the max the slot/card would allow).

Identifier policy: device serials and UUIDs are deliberately NEVER sampled —
only the small integer index is recorded. Any post-processing that needs to
correlate against external inventories must redact explicitly (see
METHODOLOGY.md, data sanitation).

Sampling must never abort a measurement round: per-sample failures are
recorded as {"type": "sample_error"} rows and the loop continues.
Standard library only.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_DEFAULT = "qwen38-q8-pcie-baseline.v1"

# Base fields sampled every interval (no UUID, no serial, no bus id).
BASE_FIELDS = (
    "index",
    "memory.used",
    "utilization.gpu",
    "power.draw",
    "temperature.gpu",
    "clocks.sm",
    "clocks_throttle_reasons.active",
    "pcie.link.gen.current",
    "pcie.link.width.current",
    "pcie.link.gen.max",
    "pcie.link.width.max",
)
# Optional fields appended when the driver supports them.
ECC_FIELD = "ecc.errors.uncorrected.volatile.total"

NUMERIC_FIELDS = {
    "memory_used_mib",
    "utilization_gpu_pct",
    "power_w",
    "temperature_gpu_c",
    "sm_clock_mhz",
    "pcie_link_gen_current",
    "pcie_link_width_current",
    "pcie_link_gen_max",
    "pcie_link_width_max",
}
THROTTLE_FIELD = "throttle_reasons_bitmask"
ECC_NAME = "ecc_uncorrected_volatile_total"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_cell(cell: str) -> str:
    return cell.strip().replace("[", "").replace("]", "").replace("N/A", "")


def build_query(fields: tuple[str, ...]) -> list[str]:
    return [
        "nvidia-smi",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]


def sample(fields: tuple[str, ...], gpus: list[int]) -> list[dict[str, object]]:
    """One sampling pass; returns one record per selected GPU (index order)."""
    try:
        completed = subprocess.run(
            build_query(fields),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as error:  # sampling must not abort the round
        return [
            {"type": "sample_error", "timestamp": utc_now(), "detail": str(error)}
        ]

    names = []
    for field in fields:
        if field == ECC_FIELD:
            names.append(ECC_NAME)
        elif field == "index":
            names.append("gpu_index")
        else:
            names.append(field.replace(".", "_").replace("-", "_"))
    records: list[dict[str, object]] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        cells = [sanitize_cell(cell) for cell in line.split(",")]
        if len(cells) != len(names):
            records.append(
                {
                    "type": "sample_error",
                    "timestamp": utc_now(),
                    "detail": f"field count mismatch: {len(cells)} != {len(names)}",
                }
            )
            continue
        record: dict[str, object] = {"type": "gpu_sample", "timestamp": utc_now()}
        for name, cell in zip(names, cells):
            if name == "gpu_index":
                record[name] = int(cell) if cell.isdigit() else cell
            elif name in NUMERIC_FIELDS:
                try:
                    record[name] = float(cell)
                except ValueError:
                    record[name] = None
            elif name == THROTTLE_FIELD:
                try:
                    record[name] = int(float(cell))
                except ValueError:
                    record[name] = None
            elif name == ECC_NAME:
                try:
                    record[name] = int(float(cell))
                except ValueError:
                    record[name] = None
            else:
                record[name] = cell
        if not gpus or record.get("gpu_index") in gpus:
            records.append(record)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpus",
        default="",
        help="comma-separated GPU indices to sample (default: all visible)",
    )
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--schema", default=SCHEMA_DEFAULT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gpus = [int(g) for g in args.gpus.split(",") if g.strip()] if args.gpus else []

    # Probe once for ECC support so field order stays stable across the run.
    fields = BASE_FIELDS
    probe = subprocess.run(
        build_query(BASE_FIELDS + (ECC_FIELD,)),
        capture_output=True,
        text=True,
        timeout=5,
    )
    if probe.returncode == 0:
        fields = BASE_FIELDS + (ECC_FIELD,)

    stop = False

    def handler(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:

        def emit(record: dict[str, object]) -> None:
            record["schema_version"] = args.schema
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()

        emit(
            {
                "type": "monitor_start",
                "timestamp": utc_now(),
                "gpus_selected": gpus,
                "interval_s": args.interval,
                "ecc_sampled": fields == BASE_FIELDS + (ECC_FIELD,),
                "identifier_policy": "no serials, no UUIDs, index only",
            }
        )
        while not stop:
            started = time.monotonic()
            for record in sample(fields, gpus):
                emit(record)
            elapsed = time.monotonic() - started
            sleep_for = max(args.interval - elapsed, 0.1)
            # Sleep in small slices so signals land promptly.
            deadline = time.monotonic() + sleep_for
            while not stop and time.monotonic() < deadline:
                time.sleep(min(0.2, max(deadline - time.monotonic(), 0.01)))
        emit({"type": "monitor_stop", "timestamp": utc_now()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
