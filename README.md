# Hydra-Inference

Mixed-vendor GPU inference cluster manager. Pools CUDA and ROCm GPUs across machines into a single inference endpoint using [llama.cpp's RPC backend](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc).

## Why

A single machine rarely has enough VRAM for large models. Hydra lets you combine GPUs from different machines and vendors — for example, AMD 7900 XTX cards on a Linux box with NVIDIA RTX cards on a Windows desktop — and present them as one OpenAI-compatible API.

## Architecture

```
┌──────────────────────────────────────────────────┐
│           ROCm Machine (Ubuntu)                  │
│  llama-server (HIP + RPC backends)               │
│  GPU0: 7900 XTX (24GB) ── local layers           │
│  GPU1: 7900 XTX (24GB) ── local layers           │
│  --rpc 192.168.1.100:50052,50053                 │
│  Serves OpenAI-compatible API on :8080           │
└──────────────┬───────────────────────────────────┘
               │ TCP / LAN
┌──────────────▼───────────────────────────────────┐
│        Windows Desktop (192.168.1.100)           │
│  rpc-server :50052 → RTX 4070 Ti Super (16GB)   │
│  rpc-server :50053 → RTX 3060 (12GB)            │
└──────────────────────────────────────────────────┘
```

The machine with the most VRAM acts as coordinator (runs `llama-server`). Remote GPUs contribute layers via `rpc-server` over TCP. Hydra calculates the optimal `--tensor-split` automatically from VRAM sizes.

## Quick Start

```bash
# Install
git clone https://github.com/akivasolutions/hydra-inference.git
cd hydra-inference
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# Edit topology for your hardware
vim configs/cluster.yaml

# Check cluster status
hydra status

# Start (after rpc-server instances are running on workers)
hydra start

# Hot-swap to a different model (RPC workers persist)
hydra swap deepseek-r1-70b

# Benchmark
hydra benchmark

# Stop
hydra stop
```

## Configuration

Edit `configs/cluster.yaml` to match your hardware:

```yaml
coordinator:
  host: 0.0.0.0
  port: 8080
  backend: hip       # hip or cuda
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
      - name: "RTX 3060"
        vram_gb: 12
        rpc_port: 50053

models:
  qwen3-72b:
    path: /models/Qwen3-72B-Q4_K_M.gguf
    ctx_size: 8192
    flash_attn: true
    default: true
```

Hydra auto-calculates tensor split from VRAM sizes. For the above config: `[0.32, 0.32, 0.21, 0.16]`.

## CLI Reference

| Command | Description |
|---------|-------------|
| `hydra status` | Show coordinator + worker status, VRAM, tensor split |
| `hydra start [-m MODEL]` | Start coordinator with health-checked workers |
| `hydra stop` | Stop the coordinator (workers keep running) |
| `hydra swap MODEL` | Hot-swap model (restart coordinator, workers persist) |
| `hydra benchmark` | Run pp/tg benchmark against running coordinator |

Global option: `-c /path/to/cluster.yaml` or `HYDRA_CONFIG` env var.

## Hardware Setup

### Worker (CUDA — Windows)

```bash
# Build llama.cpp with CUDA + RPC
cmake -B build -DGGML_CUDA=ON -DGGML_RPC=ON
cmake --build build --config Release

# Start one rpc-server per GPU
build/bin/rpc-server.exe -p 50052  # GPU 0
build/bin/rpc-server.exe -p 50053  # GPU 1
```

Or use the bootstrap script: `scripts/install-worker.sh`

### Coordinator (ROCm — Ubuntu)

```bash
# Build llama.cpp with HIP + RPC
cmake -B build -DGGML_HIP=ON -DGGML_RPC=ON -DAMDGPU_TARGETS=gfx1100
cmake --build build --config Release -j$(nproc)
sudo cp build/bin/llama-server /usr/local/bin/
```

Or use `scripts/install-coordinator.sh`

## How It Works

1. **Workers** run `rpc-server`, which exposes GPU compute over TCP. Each `rpc-server` instance binds to one GPU.
2. **Coordinator** runs `llama-server` with `--rpc <worker-addresses>`. It loads the model and distributes layers across local and remote GPUs based on `--tensor-split`.
3. **Hydra CLI** manages the lifecycle: checks worker health before starting, calculates split ratios, tracks the coordinator PID, and supports model hot-swapping.

The RPC backend serializes GGML tensor operations over the network. This works across vendors because the coordinator doesn't need to know whether a remote GPU is CUDA or ROCm — the `rpc-server` handles its own backend.

## Expected Performance

| Config | Model | tok/s (est.) |
|--------|-------|-------------|
| 2x XTX local only | Qwen3 72B Q4 | ~13 |
| 4 GPUs via RPC (GbE) | Qwen3 72B Q4 | ~10-15 |
| 4 GPUs via RPC (2.5GbE) | Qwen3 72B Q4 | ~12-16 |
| 4 GPUs via RPC | 120B+ Q4 | ~5-8 |

Network bandwidth is the bottleneck. 2.5GbE USB-C adapters (~$25) significantly reduce this.

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```
