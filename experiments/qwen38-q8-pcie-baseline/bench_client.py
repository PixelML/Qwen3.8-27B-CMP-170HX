#!/usr/bin/env python3
"""Deterministic benchmark client for the Qwen3.8-27B Q8 PCIe scaling baseline.

Fixed protocol (see PLAN.md / METHODOLOGY.md — do not tune in place):
  * public, deterministic prompt suite at nominal 1,024 / 8,192 / 32,768
    prompt tokens (exact counts come from the server usage object),
  * exactly 256 requested output tokens (EOS suppressed via the serving
    runtime's ignore_eos extension so the count is exact),
  * temperature 0.0, top_p 1.0, fixed seed, single stream,
  * streaming is used ONLY to observe time-to-first-token; prompt/output
    token counts are taken from the FINAL usage object, never from SSE
    event counts,
  * per context bucket: 1 designated cold measurement (first request after
    startup) + N warm repetitions (contract: 3),
  * interruption-safe JSONL: every record is flushed and fsynced on write,
    and a run_summary record is always emitted on normal completion,
    classification failure, or signal-driven interruption.

Receipts are sanitized by construction: no hostnames, addresses, UUIDs,
serials, API keys, or raw server logs are ever written. Standard library
only.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# --- fixed contract constants -------------------------------------------------
SCHEMA_DEFAULT = "qwen38-q8-pcie-baseline.v1"
CONTEXT_BUCKETS: tuple[int, ...] = (1024, 8192, 32768)
REQUESTED_OUTPUT_TOKENS = 256
TEMPERATURE = 0.0
TOP_P = 1.0
SEED = 1234
SERVED_MODEL_NAME_DEFAULT = "qwen3.8-27b-q8"
CHARS_PER_TOKEN_HEURISTIC = 4.0  # sizing heuristic only; usage object is truth
NODE_WORKER_IDS: tuple[int, ...] = (1, 2, 3, 4)

# Public, deterministic, content-free prompt text (no private data).
PROMPT_SENTENCE = (
    "The quiet archivist catalogs every returned letter by the season of its "
    "postmark, then reads one paragraph aloud to the empty reading room "
    "before locking the bronze doors at dusk. "
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JsonlWriter:
    """Interruption-safe JSONL: flush + fsync on every record."""

    def __init__(self, path: Path) -> None:
        self._handle = path.open("a", encoding="utf-8")

    def write(self, record: dict[str, object]) -> None:
        self._handle.write(json.dumps(record, sort_keys=True) + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        try:
            self._handle.close()
        except Exception:
            pass


def build_prompt(nominal_tokens: int) -> str:
    """Deterministic prompt sized with a documented chars-per-token heuristic.

    The nominal bucket label is approximate by construction; the exact prompt
    token count for every request is recorded from the server usage object.
    """
    target_chars = int(nominal_tokens * CHARS_PER_TOKEN_HEURISTIC)
    repeats = target_chars // len(PROMPT_SENTENCE) + 1
    return (PROMPT_SENTENCE * repeats)[:target_chars]


def concurrent_node4_run(args: argparse.Namespace) -> int:
    """Phase B (node4): identical requests on four card-local workers, in lockstep.

    Buckets run strictly in ascending order; the four per-worker requests inside
    one bucket start together (threading.Barrier) and finish before the next
    bucket begins. The shared wall window makes per-card throughput directly
    comparable and yields an honest whole-node aggregate. Cold/warm structure,
    sampling, and token limits match the single-card control exactly.
    """
    out_path: Path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = JsonlWriter(out_path)
    worker_urls = [u.strip() for u in args.node4_urls.split(",") if u.strip()]
    if len(worker_urls) != 4:
        writer.close()
        raise SystemExit("--node4-urls requires exactly 4 comma-separated base URLs")
    endpoints = {
        wid: (worker_urls[wid - 1], f"worker-{wid}")
        for wid in NODE_WORKER_IDS
    }
    streamers = {
        wid: (CompletionStreamer(url, args.api_key, args.model), label)
        for wid, (url, label) in endpoints.items()
    }
    lock = threading.Lock()

    def write(record: dict[str, object]) -> None:
        with lock:
            writer.write(record)

    write(
        {
            "type": "run_start",
            "schema_version": args.schema,
            "run_id": args.run_id,
            "card_count": 4,
            "mode": "node4_card_local",
            "started_at_utc": utc_now(),
            "contract": {
                "context_buckets_nominal": list(CONTEXT_BUCKETS),
                "requested_output_tokens": REQUESTED_OUTPUT_TOKENS,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "seed": SEED,
                "warm_repetitions_target": args.warm_repetitions,
                "cold_per_bucket": 1,
                "speculative_decoding": "off",
                "served_model_name": args.model,
                "worker_endpoints": [label for _, label in endpoints.values()],
            },
        }
    )

    counts = {"cold_ok": 0, "warm_ok": 0, "cold_failed": 0, "warm_failed": 0}
    status = "completed"
    exit_code = 0

    try:
        for bucket in CONTEXT_BUCKETS:
            prompt = build_prompt(bucket)
            plan = [("cold", 0)] + [
                ("warm", rep) for rep in range(1, args.warm_repetitions + 1)
            ]
            for phase, repetition in plan:
                barrier = threading.Barrier(4, timeout=120)
                window_started = time.perf_counter()

                def worker_slot(worker_id: int) -> None:
                    streamer, label = streamers[worker_id]
                    row: dict[str, object] = {
                        "type": "request",
                        "schema_version": args.schema,
                        "run_id": args.run_id,
                        "card_count": 1,
                        "mode": "node4_card_local",
                        "worker_id": worker_id,
                        "worker_endpoint": label,
                        "context_bucket": str(bucket),
                        "nominal_prompt_tokens": bucket,
                        "phase": phase,
                        "repetition": repetition,
                        "finished_at_utc": utc_now(),
                    }
                    try:
                        barrier.wait()
                        result = streamer.stream(prompt, args.request_timeout)
                    except threading.BrokenBarrierError:
                        result = {
                            "success": False,
                            "error_class": "barrier_broken",
                        }
                    except RunInterrupted:
                        result = {"success": False, "error_class": "interrupted"}
                    merged = {**row, **result}
                    write(merged)
                    if result.get("success"):
                        key = f"{phase}_ok"
                    else:
                        key = f"{phase}_failed"
                    with lock:
                        counts[key] += 1

                threads = [
                    threading.Thread(target=worker_slot, args=(wid,))
                    for wid in NODE_WORKER_IDS
                ]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()

                # Shared-window receipt: per-phase node aggregate is derived
                # downstream from rows sharing (bucket, phase, repetition).
                write(
                    {
                        "type": "window",
                        "schema_version": args.schema,
                        "run_id": args.run_id,
                        "context_bucket": str(bucket),
                        "phase": phase,
                        "repetition": repetition,
                        "window_started_perf": round(window_started, 6),
                        "window_ended_perf": round(time.perf_counter(), 6),
                    }
                )
                if any(t.is_alive() for t in threads):  # pragma: no cover
                    status = "thread_join_timeout"
                    exit_code = 4
                    raise RunInterrupted
    except RunInterrupted:
        status = "interrupted"
        exit_code = 130
    finally:
        write(
            {
                "type": "run_summary",
                "schema_version": args.schema,
                "run_id": args.run_id,
                "card_count": 4,
                "mode": "node4_card_local",
                "status": status,
                "counts": counts,
                "ended_at_utc": utc_now(),
            }
        )
        writer.close()
    return exit_code


class RunInterrupted(Exception):
    """Raised by signal handlers to abort the run at a safe point."""


def install_signal_handlers() -> None:
    def handler(signum: int, _frame: object) -> None:
        raise RunInterrupted(f"signal {signum}")

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


class CompletionStreamer:
    """Stream one /v1/completions request; TTFT from first token event,
    token counts from the final usage object."""

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.url = base_url.rstrip("/") + "/v1/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.model = model

    def stream(
        self, prompt: str, request_timeout: float
    ) -> dict[str, object]:
        """Returns a flat metrics dict; never raises for protocol issues —
        failures are returned as classified error rows instead."""
        body = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": REQUESTED_OUTPUT_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "seed": SEED,
            "ignore_eos": True,  # exactly 256 output tokens, EOS suppressed
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        record: dict[str, object] = {
            "requested_output_tokens": REQUESTED_OUTPUT_TOKENS,
        }
        started = time.perf_counter()
        ttft: float | None = None
        done = False
        usage: dict | None = None
        pieces: list[str] = []
        try:
            request = urllib.request.Request(
                self.url,
                data=json.dumps(body).encode("utf-8"),
                headers=self.headers,
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=request_timeout) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        done = True
                        break
                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    choices = event.get("choices") or []
                    text_piece = ""
                    if choices:
                        text_piece = choices[0].get("text") or ""
                        if ttft is None and text_piece:
                            ttft = time.perf_counter() - started
                    if text_piece:
                        pieces.append(text_piece)
                    event_usage = event.get("usage")
                    if event_usage:
                        usage = event_usage
        except RunInterrupted:
            raise
        except socket.timeout as error:
            record.update(
                success=False, error_class="timeout", error_detail=str(error)
            )
            return record
        except urllib.error.HTTPError as error:
            record.update(
                success=False,
                error_class="http_error",
                error_detail=f"status={error.code}",
            )
            return record
        except urllib.error.URLError as error:
            record.update(
                success=False, error_class="connection_error", error_detail=str(error)
            )
            return record
        except TimeoutError as error:
            record.update(
                success=False, error_class="timeout", error_detail=str(error)
            )
            return record

        latency = time.perf_counter() - started
        record["latency_s"] = round(latency, 4)
        if ttft is not None:
            record["ttft_s"] = round(ttft, 4)

        if ttft is None:
            record.update(success=False, error_class="no_first_token")
            return record
        if not done:
            record.update(success=False, error_class="stream_truncated")
            return record
        if not usage:
            record.update(success=False, error_class="missing_usage")
            return record

        prompt_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        if prompt_tokens is None or output_tokens is None:
            record.update(success=False, error_class="missing_usage")
            return record

        prompt_tokens = int(prompt_tokens)
        output_tokens = int(output_tokens)
        record.update(
            success=True,
            error_class=None,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            # Prompt-processing throughput: prompt tokens over TTFT window.
            prompt_throughput_tok_s=round(prompt_tokens / ttft, 2),
            # Generation throughput: post-first-token decode rate.
            generation_throughput_tok_s=round(
                (output_tokens - 1) / max(latency - ttft, 1e-9), 2
            ),
            # Aggregate output throughput: output tokens over the whole call.
            aggregate_output_throughput_tok_s=round(output_tokens / latency, 2),
        )
        record["response_text"] = "".join(pieces)
        record["response_chars"] = len(record["response_text"])
        return record


def run(args: argparse.Namespace) -> int:
    out_path: Path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = JsonlWriter(out_path)
    streamer = CompletionStreamer(args.base_url, args.api_key, args.model)

    writer.write(
        {
            "type": "run_start",
            "schema_version": args.schema,
            "run_id": args.run_id,
            "card_count": args.card_count,
            "started_at_utc": utc_now(),
            "contract": {
                "context_buckets_nominal": list(CONTEXT_BUCKETS),
                "requested_output_tokens": REQUESTED_OUTPUT_TOKENS,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "seed": SEED,
                "warm_repetitions_target": args.warm_repetitions,
                "cold_per_bucket": 1,
                "speculative_decoding": "off",
                "served_model_name": args.model,
            },
        }
    )

    counts = {"cold_ok": 0, "warm_ok": 0, "cold_failed": 0, "warm_failed": 0}
    status = "completed"
    exit_code = 0

    try:
        for bucket in CONTEXT_BUCKETS:
            prompt = build_prompt(bucket)
            plan = [("cold", 0)] + [
                ("warm", rep) for rep in range(1, args.warm_repetitions + 1)
            ]
            for phase, repetition in plan:
                row: dict[str, object] = {
                    "type": "request",
                    "schema_version": args.schema,
                    "run_id": args.run_id,
                    "card_count": args.card_count,
                    "context_bucket": str(bucket),
                    "nominal_prompt_tokens": bucket,
                    "phase": phase,
                    "repetition": repetition,
                    "finished_at_utc": utc_now(),
                }
                try:
                    result = streamer.stream(prompt, args.request_timeout)
                except RunInterrupted:
                    row.update(success=False, error_class="interrupted")
                    writer.write(row)
                    counts[f"{phase}_failed"] += 1
                    raise
                merged = {**row, **result}
                writer.write(merged)
                if result.get("success"):
                    counts[f"{phase}_ok"] += 1
                else:
                    counts[f"{phase}_failed"] += 1
    except RunInterrupted:
        status = "interrupted"
        exit_code = 130
    finally:
        writer.write(
            {
                "type": "run_summary",
                "schema_version": args.schema,
                "run_id": args.run_id,
                "card_count": args.card_count,
                "status": status,
                "counts": counts,
                "ended_at_utc": utc_now(),
            }
        )
        writer.close()
    return exit_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default=os.environ.get("BENCH_API_KEY", ""))
    parser.add_argument("--card-count", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--schema", default=SCHEMA_DEFAULT)
    parser.add_argument("--model", default=SERVED_MODEL_NAME_DEFAULT)
    parser.add_argument("--warm-repetitions", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("control", "node4"),
        default="control",
        help="control: one endpoint; node4: four card-local endpoints in lockstep",
    )
    parser.add_argument(
        "--node4-urls",
        default="",
        help="comma-separated base URLs for the four card-local workers",
    )
    return parser.parse_args()


if __name__ == "__main__":
    install_signal_handlers()
    parsed = parse_args()
    if parsed.mode == "node4":
        sys.exit(concurrent_node4_run(parsed))
    sys.exit(run(parsed))
