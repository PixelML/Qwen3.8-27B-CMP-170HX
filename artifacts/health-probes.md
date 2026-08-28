# Health endpoint probes — port 18020 unreachable

**Target:** `http://94.61.203.156:18020/health`
**Instance:** Vast.ai 48995474

## Probes from agent-sandbox (this session, 2026-08-28)

Sandboxed network:

```
curl -sS --max-time 8 -w 'HTTP=%{http_code} time=%{time_total}\n' http://94.61.203.156:18020/health
curl: (7) Couldn't connect to server
HTTP=000 time=0.000000
```

Unsandboxed network (escalated, one probe):

```
curl: (28) Connection timed out after 8001 milliseconds
HTTP=000 time=8.001655
```

Refused vs timeout difference is expected (different egress paths filtering vs
silently dropping). Either way: **nothing is listening that answers health
checks**, consistent with the empty-container root cause.

## Parent-session context

The parent observed the same outcome before and after applying the corrected
onstart script and rebooting the instance. A vLLM server that had started
would typically answer `/health` within seconds of weight load; no weight
download ever began (see container-filesystem.md).
