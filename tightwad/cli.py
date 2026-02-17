"""CLI interface for Tightwad cluster management."""

from __future__ import annotations

import json
import sys

import click
from rich.console import Console
from rich.table import Table

from . import coordinator, worker
from .config import load_config
from . import proxy as proxy_mod

console = Console()


@click.group()
@click.option(
    "-c", "--config",
    envvar="TIGHTWAD_CONFIG",
    default=None,
    help="Path to cluster.yaml config file",
)
@click.pass_context
def cli(ctx, config):
    """Tightwad — Mixed-vendor GPU inference cluster manager with speculative decoding."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config


def _load(ctx) -> "ClusterConfig":
    return load_config(ctx.obj.get("config_path"))


@cli.command()
@click.option("-m", "--model", default=None, help="Model name from config")
@click.pass_context
def start(ctx, model):
    """Start the coordinator llama-server with RPC workers."""
    config = _load(ctx)

    console.print("[bold]Checking RPC workers...[/bold]")
    statuses = worker.check_all_workers(config)
    for s in statuses:
        icon = "[green]●[/green]" if s.alive else "[red]●[/red]"
        latency = f" ({s.latency_ms}ms)" if s.latency_ms else ""
        console.print(f"  {icon} {s.host}:{s.port}{latency}")

    dead = [s for s in statuses if not s.alive]
    if dead:
        console.print(
            "\n[red bold]Cannot start — RPC workers unreachable.[/red bold]"
        )
        console.print("Start rpc-server on the worker machine first.")
        sys.exit(1)

    model_cfg = (
        config.models.get(model) if model else config.default_model()
    )
    if not model_cfg:
        console.print("[red]No model specified and no default configured.[/red]")
        sys.exit(1)

    console.print(f"\n[bold]Starting coordinator with {model_cfg.name}...[/bold]")
    console.print(f"  Tensor split: {config.tensor_split()}")
    console.print(f"  Total VRAM: {config.total_vram_gb} GB across {len(config.all_gpus)} GPUs")

    try:
        pid = coordinator.start(config, model)
        console.print(f"\n[green bold]Coordinator started (PID {pid})[/green bold]")
        console.print(f"  API: http://localhost:{config.coordinator_port}/v1")
    except RuntimeError as e:
        console.print(f"\n[red]{e}[/red]")
        sys.exit(1)


@cli.command()
def stop():
    """Stop the coordinator llama-server."""
    if coordinator.stop():
        console.print("[green]Coordinator stopped.[/green]")
    else:
        console.print("[yellow]Coordinator was not running.[/yellow]")


@cli.command()
@click.pass_context
def status(ctx):
    """Show cluster status."""
    config = _load(ctx)
    st = coordinator.status(config)

    # Coordinator
    coord = st["coordinator"]
    if coord["running"]:
        console.print(
            f"[green bold]● Coordinator[/green bold] "
            f"PID {coord['pid']} on :{coord['port']}"
        )
        if coord["health"] and coord["health"].get("alive"):
            console.print("  Health: [green]OK[/green]")
        elif coord["health"]:
            console.print(f"  Health: [red]{coord['health'].get('error', 'unhealthy')}[/red]")
    else:
        console.print("[dim]○ Coordinator not running[/dim]")

    # Workers
    console.print()
    table = Table(title="RPC Workers")
    table.add_column("Address")
    table.add_column("Status")
    table.add_column("Latency")
    for w in st["workers"]:
        status_str = "[green]alive[/green]" if w["alive"] else f"[red]down[/red]"
        latency_str = f"{w['latency_ms']}ms" if w["latency_ms"] else "-"
        table.add_row(w["address"], status_str, latency_str)
    console.print(table)

    # Config summary
    cfg = st["config"]
    console.print(f"\nTotal VRAM: [bold]{cfg['total_vram_gb']} GB[/bold] across {cfg['gpu_count']} GPUs")
    console.print(f"Tensor split: {cfg['tensor_split']}")
    console.print(f"Models: {', '.join(cfg['models'])}")


@cli.command()
@click.argument("model_name")
@click.pass_context
def swap(ctx, model_name):
    """Hot-swap to a different model (restarts coordinator, keeps RPC workers)."""
    config = _load(ctx)
    try:
        pid = coordinator.swap_model(config, model_name)
        console.print(
            f"[green bold]Swapped to {model_name} (PID {pid})[/green bold]"
        )
    except (ValueError, RuntimeError) as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)


@cli.command()
@click.pass_context
def benchmark(ctx):
    """Run a quick benchmark against the running coordinator."""
    config = _load(ctx)
    health = worker.check_coordinator_health("127.0.0.1", config.coordinator_port)
    if not health.get("alive"):
        console.print("[red]Coordinator not running. Start it first.[/red]")
        sys.exit(1)

    import httpx
    import time

    base = f"http://127.0.0.1:{config.coordinator_port}"

    # Prompt processing benchmark
    console.print("[bold]Running benchmark...[/bold]\n")
    prompt = "Explain quantum computing in detail. " * 64  # ~512 tokens

    start_time = time.monotonic()
    resp = httpx.post(
        f"{base}/v1/completions",
        json={
            "prompt": prompt,
            "max_tokens": 128,
            "temperature": 0.0,
        },
        timeout=120.0,
    )
    elapsed = time.monotonic() - start_time

    if resp.status_code != 200:
        console.print(f"[red]Server returned {resp.status_code}[/red]")
        sys.exit(1)

    data = resp.json()
    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    pp_speed = prompt_tokens / elapsed if elapsed > 0 else 0
    tg_speed = completion_tokens / elapsed if elapsed > 0 else 0

    table = Table(title="Benchmark Results")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Prompt tokens", str(prompt_tokens))
    table.add_row("Completion tokens", str(completion_tokens))
    table.add_row("Total time", f"{elapsed:.1f}s")
    table.add_row("Prompt processing", f"~{pp_speed:.0f} tok/s")
    table.add_row("Generation", f"~{tg_speed:.1f} tok/s")
    console.print(table)


# --- Proxy subcommand group ---


@cli.group()
def proxy():
    """Speculative decoding proxy commands."""
    pass


@proxy.command("start")
@click.pass_context
def proxy_start(ctx):
    """Start the speculative decoding proxy server."""
    config = _load(ctx)
    if config.proxy is None:
        console.print("[red]No proxy section in config. Add it to cluster.yaml.[/red]")
        sys.exit(1)

    existing_pid = proxy_mod.read_pidfile()
    if existing_pid is not None:
        try:
            import os
            os.kill(existing_pid, 0)
            console.print(
                f"[yellow]Proxy already running (PID {existing_pid}). "
                f"Stop it first with: tightwad proxy stop[/yellow]"
            )
            sys.exit(1)
        except ProcessLookupError:
            proxy_mod.remove_pidfile()

    pc = config.proxy
    console.print(f"[bold]Starting speculative decoding proxy...[/bold]")
    console.print(f"  Draft:  {pc.draft.model_name} @ {pc.draft.url}")
    console.print(f"  Target: {pc.target.model_name} @ {pc.target.url}")
    console.print(f"  Max draft tokens: {pc.max_draft_tokens}")
    console.print(f"  Listening on: {pc.host}:{pc.port}")

    import uvicorn
    app = proxy_mod.create_app(pc)
    proxy_mod.write_pidfile()
    try:
        uvicorn.run(app, host=pc.host, port=pc.port, log_level="info")
    finally:
        proxy_mod.remove_pidfile()


@proxy.command("stop")
def proxy_stop():
    """Stop the speculative decoding proxy."""
    if proxy_mod.stop_proxy():
        console.print("[green]Proxy stopped.[/green]")
    else:
        console.print("[yellow]Proxy was not running.[/yellow]")


@proxy.command("status")
@click.pass_context
def proxy_status(ctx):
    """Show proxy health and acceptance rate stats."""
    config = _load(ctx)
    if config.proxy is None:
        console.print("[red]No proxy section in config.[/red]")
        sys.exit(1)

    import httpx

    pc = config.proxy
    url = f"http://127.0.0.1:{pc.port}/v1/tightwad/status"

    try:
        resp = httpx.get(url, timeout=5.0)
        data = resp.json()
    except Exception:
        console.print("[dim]○ Proxy not running[/dim]")
        return

    # Draft/target health
    for role in ("draft", "target"):
        info = data[role]
        alive = info["health"].get("alive", False)
        icon = "[green]●[/green]" if alive else "[red]●[/red]"
        console.print(f"  {icon} {role.title()}: {info['model']} @ {info['url']}")

    # Stats
    stats = data.get("stats", {})
    if stats.get("total_rounds", 0) > 0:
        console.print()
        table = Table(title="Speculation Stats")
        table.add_column("Metric")
        table.add_column("Value")
        table.add_row("Rounds", str(stats["total_rounds"]))
        table.add_row("Drafted", str(stats["total_drafted"]))
        table.add_row("Accepted", str(stats["total_accepted"]))
        table.add_row("Acceptance rate", f"{stats['acceptance_rate']:.1%}")
        table.add_row("Tokens/round", f"{stats['effective_tokens_per_round']:.1f}")
        table.add_row("Uptime", f"{stats['uptime_seconds']:.0f}s")
        console.print(table)
    else:
        console.print("\n[dim]No speculation rounds yet.[/dim]")
