#!/usr/bin/env bash
# Reproduces the failure diagnosis for a Vast deployment of
# ghcr.io/syv-ai/qwen38-27b-rtx3090:latest — no credentials required.
set -u

IMAGE_REPO="syv-ai/qwen38-27b-rtx3090"
HEALTH_URL="${HEALTH_URL:-http://94.61.203.156:18020/health}"

echo "== 1. GHCR anonymous manifest probe (expect 401 if private) =="
TOKEN=$(curl -s "https://ghcr.io/token?scope=repository:${IMAGE_REPO}:pull" \
  | sed -n 's/.*"token":"\([^"]*\)".*/\1/p')
if [ -n "${TOKEN:-}" ]; then
  CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "Authorization: Bearer ${TOKEN}" \
    "https://ghcr.io/v2/${IMAGE_REPO}/manifests/latest")
  echo "manifest HTTP=${CODE}"
  [ "$CODE" = "401" ] && echo "-> IMAGE IS PRIVATE; rental will reproduce this failure"
else
  echo "could not obtain anonymous token (network blocked?)"
fi

echo "== 2. Health endpoint probe (expect refusal/timeout when server absent) =="
curl -sS --max-time 8 -w "HTTP=%{http_code} time=%{time_total}\n" "$HEALTH_URL" || true

echo "Done. If step 1 returned 401, fix image access before renting again."
