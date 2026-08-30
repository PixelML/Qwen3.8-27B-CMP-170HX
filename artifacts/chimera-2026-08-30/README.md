# Chimera three-card benchmark evidence

These files capture the 2026-08-30 sequential v9-recipe replication on the
three CMP 170HX cards passed through to Chimera VM 215.

| Card | Decode 256 | Decode 900 | TTFT 256 | Prefill 6.6K | Core / memory peak |
|---|---:|---:|---:|---:|---:|
| GPU0 | 135.31 tok/s | 121.28 tok/s | 201.2 ms | 1957.3 tok/s | 51 / 52 C |
| GPU1 | 140.27 tok/s | 124.78 tok/s | 189.7 ms | 1954.7 tok/s | 51 / 59 C |
| GPU2 | 133.57 tok/s | 119.94 tok/s | 181.4 ms | 1926.0 tok/s | 51 / 61 C |

Each GPU directory contains:

- `bench.jsonl`: prompt token counts, SSE-event counts, all measured samples,
  summaries, and observed GPU peaks
- `telemetry.jsonl`: raw half-second power, clock, utilization, VRAM, and
  temperature samples
- `metadata.txt`: model paths, runtime versions, recipe commit, UUID, PCI bus,
  driver, configured power limit, and run timestamps
- `nvidia-before.txt` and `nvidia-after.txt`: raw power, temperature, and clock
  event state before and after the run
- `specdec.log`: server-side DFlash2 acceptance metrics
- `server-evidence.log`: filtered configuration, kernel, graph-capture, and
  server evidence from the full temporary server log

`health.txt` records the manual post-run kernel-journal check. No model weights,
API credentials, or full private environment dumps are included.
