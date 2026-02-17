"""Distribute GGUF models to worker machines via rsync/scp."""

from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID

from .config import ClusterConfig, Worker


@dataclass
class TransferTarget:
    host: str
    ssh_user: str | None
    remote_path: str
    worker_name: str  # for display


@dataclass
class TransferResult:
    target: TransferTarget
    success: bool
    message: str


def resolve_targets(
    config: ClusterConfig,
    model_name: str,
    specific_target: str | None = None,
) -> tuple[Path, list[TransferTarget]]:
    """Resolve model path and transfer targets from config.

    Args:
        config: Cluster config with workers and models.
        model_name: Model name key from config.
        specific_target: Optional 'host:/path' override.

    Returns:
        Tuple of (local model path, list of transfer targets).
    """
    model_cfg = config.models.get(model_name)
    if not model_cfg:
        available = ", ".join(config.models.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

    local_path = Path(model_cfg.path)

    if specific_target:
        host, _, remote_path = specific_target.partition(":")
        if not remote_path:
            raise ValueError(f"Target must be 'host:/path', got '{specific_target}'")
        return local_path, [TransferTarget(
            host=host,
            ssh_user=None,
            remote_path=remote_path,
            worker_name=host,
        )]

    targets: list[TransferTarget] = []
    for w in config.workers:
        if not w.model_dir:
            continue
        targets.append(TransferTarget(
            host=w.host,
            ssh_user=w.ssh_user,
            remote_path=str(Path(w.model_dir) / local_path.name),
            worker_name=f"{w.host} ({w.gpus[0].name})" if w.gpus else w.host,
        ))

    return local_path, targets


def build_transfer_cmd(local_path: Path, target: TransferTarget) -> list[str]:
    """Build rsync or scp command for a transfer."""
    dest_user = f"{target.ssh_user}@" if target.ssh_user else ""
    dest = f"{dest_user}{target.host}:{target.remote_path}"

    if shutil.which("rsync"):
        return [
            "rsync", "-avz", "--progress", "--partial",
            str(local_path), dest,
        ]
    else:
        return ["scp", "-C", str(local_path), dest]


async def transfer_one(
    local_path: Path,
    target: TransferTarget,
    progress: Progress,
    task_id: TaskID,
) -> TransferResult:
    """Transfer model to one target using rsync/scp."""
    cmd = build_transfer_cmd(local_path, target)

    progress.update(task_id, description=f"[cyan]{target.worker_name}[/cyan]: transferring...")

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode == 0:
        progress.update(task_id, description=f"[green]{target.worker_name}[/green]: done", completed=100)
        return TransferResult(target=target, success=True, message="OK")
    else:
        err = stderr.decode().strip() or stdout.decode().strip()
        progress.update(task_id, description=f"[red]{target.worker_name}[/red]: failed")
        return TransferResult(target=target, success=False, message=err)


async def distribute_async(
    local_path: Path,
    targets: list[TransferTarget],
    console: Console,
) -> list[TransferResult]:
    """Transfer model to all targets in parallel with progress display."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        tasks = []
        for target in targets:
            task_id = progress.add_task(f"[dim]{target.worker_name}[/dim]: queued", total=100)
            tasks.append(transfer_one(local_path, target, progress, task_id))

        results = await asyncio.gather(*tasks)

    return list(results)


def distribute(
    local_path: Path,
    targets: list[TransferTarget],
    console: Console,
) -> list[TransferResult]:
    """Synchronous wrapper for distribute_async."""
    return asyncio.run(distribute_async(local_path, targets, console))


def format_dry_run(local_path: Path, targets: list[TransferTarget]) -> str:
    """Show what would be transferred without executing."""
    lines = [f"Model: {local_path}"]
    if local_path.exists():
        size_gb = local_path.stat().st_size / (1024**3)
        lines.append(f"Size:  {size_gb:.2f} GB")
    else:
        lines.append(f"Size:  (file not found locally)")
    lines.append(f"\nTransfers ({len(targets)}):")
    for t in targets:
        cmd = build_transfer_cmd(local_path, t)
        lines.append(f"  {t.worker_name}:")
        lines.append(f"    {' '.join(cmd)}")
    return "\n".join(lines)
