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
    if pc.drafters:
        console.print(f"  [bold]Drafters ({len(pc.drafters)}):[/bold]")
        for d in pc.drafters:
            console.print(f"    - {d.model_name} @ {d.url} ({d.backend})")
    else:
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

    # Drafters or single draft health
    if "drafters" in data:
        console.print("  [bold]Drafters:[/bold]")
        drafter_table = Table()
        drafter_table.add_column("Model")
        drafter_table.add_column("URL")
        drafter_table.add_column("Backend")
        drafter_table.add_column("Health")
        drafter_table.add_column("Wins")
        for d in data["drafters"]:
            alive = d["health"].get("alive", False)
            health_str = "[green]alive[/green]" if alive else "[red]down[/red]"
            drafter_table.add_row(
                d["model"], d["url"], d["backend"],
                health_str, str(d["wins"]),
            )
        console.print(drafter_table)
    elif "draft" in data:
        info = data["draft"]
        alive = info["health"].get("alive", False)
        icon = "[green]●[/green]" if alive else "[red]●[/red]"
        console.print(f"  {icon} Draft: {info['model']} @ {info['url']}")

    # Target health
    target = data["target"]
    t_alive = target["health"].get("alive", False)
    t_icon = "[green]●[/green]" if t_alive else "[red]●[/red]"
    console.print(f"  {t_icon} Target: {target['model']} @ {target['url']}")

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


@cli.command("inspect")
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--plan", "show_plan", is_flag=True, help="Show distribution plan for current cluster")
@click.pass_context
def inspect_cmd(ctx, model_path, show_plan):
    """Inspect a GGUF model file: metadata, tensors, distribution plan."""
    try:
        from .gguf_inspect import inspect_model, plan_distribution, format_report
    except ImportError:
        console.print("[red]Missing gguf package. Install with: pip install tightwad[inspect][/red]")
        sys.exit(1)

    model_info = inspect_model(model_path)
    plan = None
    if show_plan:
        config = _load(ctx)
        plan = plan_distribution(model_info, config)

    output = format_report(model_info, plan)
    console.print(output)


@cli.command("distribute")
@click.argument("model_name")
@click.option("-t", "--target", "specific_target", default=None, help="Specific target as host:/path")
@click.option("--dry-run", is_flag=True, help="Preview transfers without executing")
@click.pass_context
def distribute_cmd(ctx, model_name, specific_target, dry_run):
    """Distribute a model to worker machines via rsync/scp."""
    from .distribute import resolve_targets, distribute, format_dry_run

    config = _load(ctx)
    try:
        local_path, targets = resolve_targets(config, model_name, specific_target)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    if not targets:
        console.print("[yellow]No targets with model_dir configured. "
                       "Add model_dir to workers in config or use -t host:/path.[/yellow]")
        sys.exit(1)

    if dry_run:
        console.print(format_dry_run(local_path, targets))
        return

    if not local_path.exists():
        console.print(f"[red]Model file not found: {local_path}[/red]")
        sys.exit(1)

    console.print(f"[bold]Distributing {local_path.name} to {len(targets)} target(s)...[/bold]\n")
    results = distribute(local_path, targets, console)

    failed = [r for r in results if not r.success]
    if failed:
        console.print(f"\n[red]{len(failed)} transfer(s) failed:[/red]")
        for r in failed:
            console.print(f"  {r.target.worker_name}: {r.message}")
        sys.exit(1)
    else:
        console.print(f"\n[green]All {len(results)} transfer(s) complete.[/green]")


@cli.command()
@click.option("--direct", is_flag=True, help="Chat directly with target (no speculation, for comparison)")
@click.pass_context
def chat(ctx, direct):
    """Interactive chat with the speculative decoding proxy."""
    import httpx
    config = _load(ctx)
    if config.proxy is None:
        console.print("[red]No proxy section in config. Add it to cluster.yaml.[/red]")
        sys.exit(1)
    pc = config.proxy
    if direct:
        base_url = pc.target.url
        console.print(f"\n[bold]Direct mode:[/bold] {pc.target.model_name} @ {pc.target.url}")
    else:
        base_url = f"http://127.0.0.1:{pc.port}"
        try:
            httpx.get(f"{base_url}/v1/tightwad/status", timeout=3.0)
        except Exception:
            console.print("[red]Proxy not running. Start it first:[/red]")
            console.print("  tightwad proxy start")
            sys.exit(1)
        console.print(f"\n[bold]Speculative mode:[/bold] proxy @ :{pc.port} -> {pc.target.model_name}")
    console.print("[dim]Type your message and press Enter. Ctrl+C to quit.[/dim]\n")
    messages: list[dict] = []
    while True:
        try:
            user_input = console.input("[bold green]You:[/bold green] ")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break
        if not user_input.strip():
            continue
        messages.append({"role": "user", "content": user_input})
        try:
            url = f"{base_url}/v1/chat/completions"
            body = {"messages": messages, "max_tokens": 1024, "temperature": 0.0, "stream": False}
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(url, json=body)
                resp.raise_for_status()
                data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not text:
                text = data.get("choices", [{}])[0].get("text", "")
            console.print(f"[bold cyan]AI:[/bold cyan] {text}\n")
            messages.append({"role": "assistant", "content": text})
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted.[/dim]\n")
            messages.pop()
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]\n")
            messages.pop()
