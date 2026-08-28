# Intended benchmark spec (never executed)

Captured so a retry uses the exact same target configuration.

## Launch configuration (target)

| Knob | Value |
|---|---|
| vLLM | 0.27.1 syv overlay |
| Main model | lued/Qwen3.8-27B-INT8-W8A16-MTP |
| Draft model | syvai/Qwen3.8-27B-DFlash2-W4A16 |
| Speculative decoding | DFlash2, 7 draft tokens |
| KV cache dtype | BF16 |
| Attention backend | FLASH_ATTN |
| GPU memory utilization | 0.90 |
| Max sequences | 1 |
| Context length | 65,536 |
| Port | 18020 |

## Planned request (OpenAI-compatible)

Single-request controlled measurement, modeled on the LocalMaxxing protocol:

- Endpoint: `POST /v1/chat/completions`
- Prompt: ~88 tokens (fixed reference prompt)
- `max_tokens: 256`
- `stream: true` (for TTFT and inter-token timing)
- `temperature: 0` (deterministic comparison)

## Planned measurements

Output tok/s, prefill tok/s, TTFT, total tok/s, peak VRAM, power, temp,
DFlash2 acceptance rate (from vLLM logs), and comparison against the
212.68 / 1221.6 / 72 ms reference.

None of these were captured in this run; see RESULTS.md.
