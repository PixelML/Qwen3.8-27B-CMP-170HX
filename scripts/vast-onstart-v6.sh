#!/usr/bin/env bash
# Working onstart for instance 49003408 (v6). See repo README for the v7 diffs.
set -ux
date -u
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
mkdir -p /root/.ssh && chown root:root /root/.ssh && chmod 700 /root/.ssh
touch /root/.ssh/authorized_keys && chown root:root /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys
apt-get update -qq || true
DEBIAN_FRONTEND=noninteractive apt-get install -y python3.12 python3.12-venv python3.12-dev build-essential patch git curl ca-certificates || true
mkdir -p /app && cd /app
ln -sf /app/qwen-serving/venv /app/venv
[ -d qwen-serving/.git ] || git clone --depth 1 https://github.com/syv-ai/qwen38-27b-rtx3090 qwen-serving
cd qwen-serving
if [ ! -x venv/bin/vllm ]; then
  python3.12 -m venv venv
  venv/bin/pip install --upgrade pip wheel
  venv/bin/pip install -r docker/requirements.txt
  venv/bin/pip install flashinfer-python flashinfer-cubin==0.6.13 || true
fi
SP=$(venv/bin/python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' | tail -n1)
grep -q dflash2-backport "$SP/vllm/engine/arg_utils.py" 2>/dev/null || for p in patches/*.patch; do echo "== $p"; patch -p1 -d "$SP" < "$p" || true; done
export HF_HOME=/cache/huggingface HF_HUB_ENABLE_HF_TRANSFER=1
export PATH=/app/venv/bin:$PATH
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_API_KEY=qwen38-bench-20260828
BASE_MODEL_DIR=/app/qwen-serving/models/Qwen3.8-27B-W4A16-AutoRound bash docker/prepare.sh || true
venv/bin/python prepare/fetch_dflash2.py || true
cd /app/qwen-serving
SPEC=dflash2 CTX=fast MAX_SEQS=1 DFLASH_TOKENS=7 PORT=18020 VLLM_V2_CUDAGRAPH_MEM_MIB=1400 KV_MEM=5583457484 \
  nohup bash single-user/start_qwen.sh >/tmp/qwen38-server.log 2>&1 &
echo "server pid $!" | tee /tmp/qwen38-server.pid
touch /tmp/qwen38-server.log
tail -n +1 -f /tmp/qwen38-server.log &
while :; do sleep 3600; done
