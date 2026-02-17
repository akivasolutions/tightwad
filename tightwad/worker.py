"""RPC worker health checks and lifecycle management."""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass

import httpx

from .config import ClusterConfig, Worker


@dataclass
class WorkerStatus:
    host: str
    port: int
    gpu_name: str
    alive: bool
    latency_ms: float | None = None
    error: str | None = None


def check_rpc_port(host: str, port: int, timeout: float = 2.0) -> WorkerStatus:
    """Check if an RPC worker is reachable via TCP connect."""
    gpu_name = f"{host}:{port}"
    start = time.monotonic()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            latency = (time.monotonic() - start) * 1000
            return WorkerStatus(
                host=host, port=port, gpu_name=gpu_name,
                alive=True, latency_ms=round(latency, 1),
            )
    except (ConnectionRefusedError, TimeoutError, OSError) as e:
        return WorkerStatus(
            host=host, port=port, gpu_name=gpu_name,
            alive=False, error=str(e),
        )


def check_all_workers(config: ClusterConfig) -> list[WorkerStatus]:
    """Check all RPC workers defined in the cluster config."""
    statuses = []
    for worker in config.workers:
        for gpu in worker.gpus:
            if gpu.rpc_port:
                statuses.append(check_rpc_port(worker.host, gpu.rpc_port))
    return statuses


def check_coordinator_health(
    host: str = "127.0.0.1", port: int = 8080, timeout: float = 5.0
) -> dict:
    """Check if the coordinator llama-server is healthy via /health endpoint."""
    try:
        resp = httpx.get(f"http://{host}:{port}/health", timeout=timeout)
        return {"alive": resp.status_code == 200, "status": resp.json()}
    except Exception as e:
        return {"alive": False, "error": str(e)}


def wait_for_workers(
    config: ClusterConfig,
    timeout: float = 30.0,
    interval: float = 2.0,
) -> list[WorkerStatus]:
    """Wait for all RPC workers to become available."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        statuses = check_all_workers(config)
        if all(s.alive for s in statuses):
            return statuses
        time.sleep(interval)
    return check_all_workers(config)
