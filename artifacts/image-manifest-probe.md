# GHCR image manifest probe — image is private

**Image:** `ghcr.io/syv-ai/qwen38-27b-rtx3090:latest`
**When:** 2026-08-28 (parent session)

Anonymous (unauthenticated) manifest request against the GHCR v2 API returned:

```
HTTP/1.1 401 Unauthorized
```

Interpretation: the repository requires authentication to pull. A Vast instance
created without GHCR credentials cannot obtain the image content. This matches
the observed ~1.2 MiB stub filesystem (see container-filesystem.md).

Reproduce:

```bash
TOKEN=$(curl -s "https://ghcr.io/token?scope=repository:syv-ai/qwen38-27b-rtx3090:pull" | jq -r .token)
curl -s -o /dev/null -w "%{http_code}\n" \
  -H "Authorization: Bearer $TOKEN" \
  https://ghcr.io/v2/syv-ai/qwen38-27b-rtx3090/manifests/latest
# -> 401
```
