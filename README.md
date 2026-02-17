# Tightwad

Mixed-vendor GPU inference cluster manager with speculative decoding proxy. Pools CUDA and ROCm GPUs across machines using [llama.cpp RPC](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc), and accelerates inference via application-layer speculative decoding across network-separated servers.

## Two Modes

### 1. RPC Cluster — Pool GPUs into one endpoint

Combine GPUs from different machines and vendors into a single OpenAI-compatible API. The coordinator distributes model layers across local and remote GPUs.

### 2. Speculative Decoding Proxy — Draft + Verify across machines

A fast small model (e.g., 8B on a consumer GPU) drafts candidate tokens, a large model (e.g., 72B on a server or cloud API) verifies them in batch. Output quality is **identical to running the large model alone**, but 2-3x faster because batch verification is much cheaper than autoregressive generation.

```
Client (OpenAI API)
        │
        ▼
┌──────────────────────────────┐
│   Tightwad Proxy (:8088)      │  Python async server
│   Speculation Loop:          │
│   1. Draft 8 tokens          │──► Draft: Qwen3-8B (fast, local)
│   2. Verify batch            │──► Target: Qwen3-72B (accurate, local or API)
│   3. Accept/reject           │
│   4. Stream to client        │
└──────────────────────────────┘
```

**Why not just use RPC?** RPC ships 100-300 MB of tensor data per step over the network. The speculative proxy ships token IDs (bytes). For models that fit on a single machine's VRAM, speculation is dramatically faster.

## Quick Start

```bash
# Install
git clone https://github.com/akivasolutions/hydra-inference.git
cd hydra-inference
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# Edit topology for your hardware
vim configs/cluster.yaml
```

### Speculative Decoding Proxy

```bash
# Start the proxy (draft + target servers must be running)
tightwad proxy start

# Check health and acceptance rate stats
tightwad proxy status

# Test it
curl http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}'

# Detailed stats
curl http://localhost:8088/v1/tightwad/status

# Stop
tightwad proxy stop
```

### RPC Cluster

```bash
# Check cluster status
tightwad status

# Start (after rpc-server instances are running on workers)
tightwad start

# Hot-swap to a different model (RPC workers persist)
tightwad swap deepseek-r1-70b

# Benchmark
tightwad benchmark

# Stop
tightwad stop
```

## Configuration

Edit `configs/cluster.yaml`:

```yaml
# Speculative decoding proxy
proxy:
  host: 0.0.0.0
  port: 8088
  max_draft_tokens: 8
  fallback_on_draft_failure: true
  draft:
    url: http://192.168.1.50:11434   # Ollama on a cheap GPU
    model_name: qwen3:8b
    backend: ollama                     # or "llamacpp"
  target:
    url: http://192.168.1.100:11434    # Bigger GPU or cloud API
    model_name: qwen3:32b
    backend: ollama

# RPC cluster (optional, for tensor-parallel across machines)
coordinator:
  host: 0.0.0.0
  port: 8080
  backend: hip
  gpus:
    - name: "7900 XTX #0"
      vram_gb: 24
    - name: "7900 XTX #1"
      vram_gb: 24

workers:
  - host: 192.168.1.100
    gpus:
      - name: "RTX 4070 Ti Super"
        vram_gb: 16
        rpc_port: 50052

models:
  qwen3-72b:
    path: /models/Qwen3-72B-Q4_K_M.gguf
    ctx_size: 8192
    flash_attn: true
    default: true
```

### Server Backends

The proxy supports two backend types for draft and target servers:

| Backend | Endpoint | Best for |
|---------|----------|----------|
| `ollama` | `/api/generate` (raw mode) | Quick setup, any Ollama instance |
| `llamacpp` | `/v1/completions` (with logprobs) | Best performance, full logprobs support |

## How Speculative Decoding Works

1. **Draft:** The small model generates N candidate tokens (fast, ~100+ tok/s)
2. **Verify:** The large model evaluates all N tokens in a single forward pass
3. **Accept/reject:** Keep tokens where both models agree, take the large model's token at the first disagreement
4. **Repeat** until done

The output is **provably identical** to running the large model alone — the small model just proposes shortcuts.

### Benchmark Results

Tested with Qwen3-8B (RTX 2070, Ollama) drafting for Qwen3-32B (RTX 4070 Ti Super, Ollama) across 5 prompt types:

| Prompt Type | Acceptance Rate | Rounds | Notes |
|-------------|:--------------:|:------:|-------|
| Reasoning   | **88%**        | 32     | Highest — deterministic math answers |
| Code        | **73%**        | 34     | High — structured syntax overlap |
| Factual     | 52%            | 18     | Moderate agreement |
| List        | 44%            | 40     | Varied phrasing causes divergence |
| Creative    | 34%            | 6      | Lowest — many valid outputs |
| **Average** | **58.3%**      | 26     | |

> **Current status:** Text-match verification proves acceptance rates but doesn't yet achieve wall-clock speedup (both models generate autoregressively). Logprobs-based batch verification — where the target scores all draft tokens in one forward pass — is the next milestone that will convert these rates into real 2-3x throughput gains.

Run the benchmark yourself: `python scripts/benchmark_proxy.py`

### Use Cases

- **Local multi-GPU:** Draft on a consumer GPU ($200), verify on a larger GPU/rig
- **Cloud cost reduction:** Draft locally, verify via cloud API — fewer API calls for the same output quality
- **Edge + datacenter:** Fast local responses with datacenter-grade accuracy

## CLI Reference

| Command | Description |
|---------|-------------|
| `tightwad proxy start` | Start speculative decoding proxy |
| `tightwad proxy stop` | Stop the proxy |
| `tightwad proxy status` | Show draft/target health + acceptance rate stats |
| `tightwad status` | Show RPC cluster status |
| `tightwad start [-m MODEL]` | Start RPC coordinator |
| `tightwad stop` | Stop the coordinator |
| `tightwad swap MODEL` | Hot-swap model (workers persist) |
| `tightwad benchmark` | Benchmark the running coordinator |

Global option: `-c /path/to/cluster.yaml` or `TIGHTWAD_CONFIG` env var.

## API Endpoints (Proxy)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/completions` | POST | Text completion (OpenAI-compatible) |
| `/v1/chat/completions` | POST | Chat completion (OpenAI-compatible) |
| `/v1/models` | GET | List available models |
| `/v1/tightwad/status` | GET | Proxy stats: acceptance rate, rounds, throughput |

All endpoints support `stream: true` for SSE streaming.

## Hardware Setup

### Worker (CUDA — Windows)

```bash
cmake -B build -DGGML_CUDA=ON -DGGML_RPC=ON
cmake --build build --config Release
build/bin/rpc-server.exe -p 50052  # GPU 0
```

Or use `scripts/install-worker.sh`

### Coordinator (ROCm — Ubuntu)

```bash
cmake -B build -DGGML_HIP=ON -DGGML_RPC=ON -DAMDGPU_TARGETS=gfx1100
cmake --build build --config Release -j$(nproc)
sudo cp build/bin/llama-server /usr/local/bin/
```

Or use `scripts/install-coordinator.sh`

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Project Structure

```
tightwad/
├── config.py        # YAML config loader (cluster + proxy)
├── cli.py           # Click CLI (cluster + proxy commands)
├── coordinator.py   # llama-server lifecycle management
├── worker.py        # RPC worker health checks
├── proxy.py         # Speculative decoding proxy server
└── speculation.py   # Verification algorithm (pure logic)
tests/
├── test_config.py
├── test_coordinator.py
├── test_speculation.py
└── test_proxy.py
configs/
└── cluster.yaml     # Hardware topology + proxy config
```
