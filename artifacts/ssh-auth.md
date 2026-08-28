# SSH access failure — authorized_keys mode/ownership

**Instance:** 48995474
**Proxy:** `ssh9.vast.ai:35474`
**When:** 2026-08-28 (parent session; corroborated by agent-sandbox session)

Symptom:

```
Permission denied (publickey).
```

Diagnosis: the Vast-injected `authorized_keys` path had incorrect
mode/ownership (home and/or `.ssh`), so sshd refused to honor it.

**Confirmed by the raw instance log** (vast-instance-48995474.log):

```
Server listening on 0.0.0.0 port 22.
Authentication refused: bad ownership or modes for file /root/.ssh/authorized_keys
Failed publickey for root from 35.198.200.37 port 51632 ssh2: ED25519 SHA256:kCsq...
```

Fix attempted by parent: `chmod 700` on home and `chmod 600` on
`authorized_keys` (via onstart adjustments), then instance reboot.

Result: **SSH still refused** after the reboot. Likely because the container
root cause (image never pulled; see image-manifest-probe.md) prevented the
intended startup environment from existing at all. The Vast SSH shim's own
reverse-tunnel also failed continuously:

```
Error: remote port forwarding failed for listen port 35474
```

Note: direct SSH was never required for the benchmark plan (API/port probes
are sufficient); it mattered only for debugging and artifact collection.
