# Instance 49003408 metadata (qwen38-cmp170hx-v6)

Captured 2026-08-28 ~11:30 UTC via Vast API v1 /instances/.

- Image: `nvidia/cuda:13.0.1-base-ubuntu24.04` (public Docker Hub base)
- GPU: 1x CMP 170HX, 64 GB (65536 MiB), PCIe Gen4 x4 per offer listing (Vast plan fields pcie_rev/pcie_width empty in API)
- Host driver: 610.43.02
- Host CPU: AMD Ryzen 9 3950X (32 threads), 64 GB RAM, 100 GB disk
- Network: 891.5 Mbps down / 390.3 Mbps up
- Port mapping: requested 18020/tcp -> actual host port 40226 (Vast assigns random host ports)
- Public endpoint: http://94.61.203.156:40226
- Stack: syv-ai/qwen38-27b-rtx3090 @ git HEAD (depth-1 clone), vLLM 0.27.1 + repo patches, flashinfer-python 0.6.16.post3, torch 2.13.0, python 3.12 venv
- Launch env: SPEC=dflash2 CTX=fast MAX_SEQS=1 DFLASH_TOKENS=7 PORT=18020 VLLM_V2_CUDAGRAPH_MEM_MIB=1400 KV_MEM=5583457484
