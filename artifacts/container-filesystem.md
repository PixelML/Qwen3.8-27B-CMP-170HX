# Container filesystem evidence

**Instance:** 48995474
**When:** 2026-08-28 (parent session, via Vast execute/logs)

Observed inside the running instance container:

- Total filesystem content: **~1.2 MiB**
- `/app` contained only:
  - `onstart.sh`
  - `ports.log`
- `/tmp/qwen38` (documented weight/cache location for this image): **absent**

Interpretation: the private GHCR image (401 on anonymous pull) was never
materialized. The onstart script ran (or attempted to run) in an effectively
empty container — there was no vLLM, no weights, no server binary to start.

Corroboration from the raw instance log (vast-instance-48995474.log): the only
non-SSH activity is apt "already newest" checks and sshd startup. There is no
image pull, no model download, no vLLM output — across the entire log.

Consequences observed downstream:

- Port 18020 never listened (health-probes.md)
- No model download occurred (no sizes/times to record)
- No startup logs from vLLM exist to preserve
