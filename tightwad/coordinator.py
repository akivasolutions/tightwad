"""Coordinator: launches llama-server with RPC backend flags."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

from .config import ClusterConfig, ModelConfig
from .worker import check_all_workers, check_coordinator_health, wait_for_workers

PIDFILE = Path.home() / ".tightwad" / "coordinator.pid"


def build_server_args(config: ClusterConfig, model: ModelConfig) -> list[str]:
    """Build llama-server command-line arguments."""
    args = [
        config.coordinator_binary,
        "-m", model.path,
        "-ngl", "999",
        "--host", config.coordinator_host,
        "--port", str(config.coordinator_port),
        "--ctx-size", str(model.ctx_size),
        "-n", str(model.predict),
    ]

    if model.flash_attn:
        args.append("--flash-attn")

    # RPC workers
    rpc_addrs = config.rpc_addresses
    if rpc_addrs:
        args.extend(["--rpc", ",".join(rpc_addrs)])

    # Tensor split across all GPUs (coordinator locals first, then RPC workers)
    split = config.tensor_split()
    if len(split) > 1:
        args.extend(["--tensor-split", ",".join(str(s) for s in split)])

    return args


def start(config: ClusterConfig, model_name: str | None = None) -> int:
    """Start the coordinator llama-server.

    Returns the subprocess PID.
    """
    # Resolve model
    if model_name:
        model = config.models.get(model_name)
        if not model:
            raise ValueError(
                f"Model '{model_name}' not found. "
                f"Available: {', '.join(config.models)}"
            )
    else:
        model = config.default_model()
        if not model:
            raise ValueError("No models configured")

    # Check if already running
    if PIDFILE.exists():
        pid = int(PIDFILE.read_text().strip())
        try:
            os.kill(pid, 0)
            raise RuntimeError(
                f"Coordinator already running (PID {pid}). "
                "Use 'tightwad stop' first."
            )
        except ProcessLookupError:
            PIDFILE.unlink()

    # Health-check RPC workers
    worker_statuses = check_all_workers(config)
    dead = [s for s in worker_statuses if not s.alive]
    if dead:
        dead_str = ", ".join(f"{s.host}:{s.port}" for s in dead)
        raise RuntimeError(
            f"RPC workers not reachable: {dead_str}\n"
            "Start rpc-server on the worker machine first."
        )

    # Build and launch
    args = build_server_args(config, model)
    PIDFILE.parent.mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    PIDFILE.write_text(str(proc.pid))

    return proc.pid


def stop() -> bool:
    """Stop the coordinator llama-server."""
    if not PIDFILE.exists():
        return False

    pid = int(PIDFILE.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    PIDFILE.unlink(missing_ok=True)
    return True


def status(config: ClusterConfig) -> dict:
    """Get full cluster status."""
    # Coordinator
    coord_running = False
    coord_pid = None
    if PIDFILE.exists():
        coord_pid = int(PIDFILE.read_text().strip())
        try:
            os.kill(coord_pid, 0)
            coord_running = True
        except ProcessLookupError:
            PIDFILE.unlink(missing_ok=True)
            coord_pid = None

    coord_health = None
    if coord_running:
        coord_health = check_coordinator_health(
            "127.0.0.1", config.coordinator_port
        )

    # Workers
    worker_statuses = check_all_workers(config)

    return {
        "coordinator": {
            "running": coord_running,
            "pid": coord_pid,
            "port": config.coordinator_port,
            "health": coord_health,
        },
        "workers": [
            {
                "address": f"{s.host}:{s.port}",
                "alive": s.alive,
                "latency_ms": s.latency_ms,
                "error": s.error,
            }
            for s in worker_statuses
        ],
        "config": {
            "total_vram_gb": config.total_vram_gb,
            "gpu_count": len(config.all_gpus),
            "models": list(config.models.keys()),
            "tensor_split": config.tensor_split(),
        },
    }


def swap_model(config: ClusterConfig, model_name: str) -> int:
    """Hot-swap the active model (stop coordinator, restart with new model).

    RPC workers persist — only the coordinator restarts.
    """
    stop()
    return start(config, model_name)
