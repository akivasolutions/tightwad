"""CLI interface for Tightwad cluster management."""

from __future__ import annotations

import json
import os
import sys

import click
from rich.console import Console
from rich.table import Table

from pathlib import Path

from . import coordinator, worker
from .config import load_config
from .coordinator import LOGDIR, COORDINATOR_LOG
from . import doctor as doctor_mod
from . import proxy as proxy_mod
from . import manifest as manifest_mod
from . import swarm_transfer as swarm_mod
from . import init_wizard

PROXY_LOG = LOGDIR / "proxy.log"

console = Console()


@click.group()
@click.version_option(package_name="tightwad")
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
@click.option("--subnet", default=None, help="Subnet to scan (e.g. 192.168.1.0/24, auto-detected if omitted)")
@click.option("--port", "extra_ports", multiple=True, type=int, help="Additional ports to scan (repeatable)")
@click.option("-o", "--output", default="configs/cluster.yaml", type=click.Path(), help="Output config path")
@click.option("--draft-url", default=None, help="Draft server URL (non-interactive mode)")
@click.option("--draft-model", default=None, help="Draft model name (required with --draft-url)")
@click.option("--draft-backend", default=None, help="Draft backend: ollama or llamacpp (auto-detected from port)")
@click.option("--target-url", default=None, help="Target server URL (non-interactive mode)")
@click.option("--target-model", default=None, help="Target model name (required with --target-url)")
@click.option("--target-backend", default=None, help="Target backend: ollama or llamacpp (auto-detected from port)")
@click.option("--max-draft-tokens", default=32, type=int, help="Max tokens per draft round (default: 32)")
@click.option("-y", "--yes", is_flag=True, help="Overwrite existing config without prompting")
def init(subnet, extra_ports, output, draft_url, draft_model, draft_backend,
         target_url, target_model, target_backend, max_draft_tokens, yes):
    """Auto-discover LAN inference servers and generate cluster.yaml."""
    import asyncio
    from urllib.parse import urlparse

    output_path = Path(output)

    # Non-interactive mode: --draft-url + --target-url
    if draft_url and target_url:
        if not draft_model:
            raise click.UsageError("--draft-model is required when using --draft-url")
        if not target_model:
            raise click.UsageError("--target-model is required when using --target-url")

        draft_parsed = urlparse(draft_url)
        target_parsed = urlparse(target_url)
        d_backend = draft_backend or init_wizard.detect_backend(draft_url)
        t_backend = target_backend or init_wizard.detect_backend(target_url)

        draft_server = init_wizard.DiscoveredServer(
            host=draft_parsed.hostname,
            port=draft_parsed.port or 80,
            backend=d_backend,
            models=[draft_model],
        )
        target_server = init_wizard.DiscoveredServer(
            host=target_parsed.hostname,
            port=target_parsed.port or 80,
            backend=t_backend,
            models=[target_model],
        )

        yaml_str = init_wizard.generate_cluster_yaml(
            draft_server=draft_server,
            draft_model=draft_model,
            target_server=target_server,
            target_model=target_model,
            max_draft_tokens=max_draft_tokens,
        )

        if output_path.exists() and not yes:
            overwrite = input(f"{output_path} already exists. Overwrite? [y/N] ").strip().lower()
            if overwrite != "y":
                console.print("[dim]Cancelled.[/dim]")
                return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(yaml_str)
        console.print(f"[green]✓[/green] Config written to {output_path}")
        return

    # Interactive mode: scan LAN
    if output_path.exists() and not yes:
        overwrite = input(f"{output_path} already exists. Overwrite? [y/N] ").strip().lower()
        if overwrite != "y":
            console.print("[dim]Cancelled.[/dim]")
            return

    console.print("[bold]Scanning LAN for inference servers...[/bold]\n")
    result = asyncio.run(init_wizard.scan_lan(
        subnet=subnet,
        extra_ports=list(extra_ports) if extra_ports else None,
    ))

    init_wizard.run_wizard(console, result, output_path)


@cli.command()
@click.option("-m", "--model", default=None, help="Model name from config")
@click.option("--ram-reclaim", type=click.Choice(["off", "on", "auto"]), default=None,
              help="RAM reclaim mode (default: from config, usually 'auto')")
@click.pass_context
def start(ctx, model, ram_reclaim):
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

    mode = ram_reclaim or config.ram_reclaim

    try:
        if mode == "off":
            pid = coordinator.start(config, model)
            console.print(f"\n[green bold]Coordinator started (PID {pid})[/green bold]")
            console.print(f"  API: http://localhost:{config.coordinator_port}/v1")
        else:
            console.print(f"  RAM reclaim: {mode}")
            pid, result = coordinator.start_and_reclaim(
                config, model, ram_reclaim=mode,
            )
            console.print(f"\n[green bold]Coordinator started (PID {pid})[/green bold]")
            console.print(f"  API: http://localhost:{config.coordinator_port}/v1")
            if result:
                if result.method == "skipped":
                    console.print(f"  RAM reclaim: skipped ({result.error or 'not needed'})")
                else:
                    console.print(
                        f"  [green]Reclaimed {result.reclaimed_mb:,.0f} MB RAM "
                        f"({result.method})[/green]"
                    )
            elif mode == "auto":
                console.print("  RAM reclaim: skipped (sufficient RAM)")
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
@click.argument("service", default="coordinator", type=click.Choice(["coordinator", "proxy"]))
@click.option("-f", "--follow", is_flag=True, help="Live-tail the log (like tail -f)")
@click.option("--clear", is_flag=True, help="Truncate log files")
@click.option("-n", "--lines", default=50, help="Number of lines to show (default: 50)")
def logs(service, follow, clear, lines):
    """View coordinator or proxy logs."""
    log_file = COORDINATOR_LOG if service == "coordinator" else PROXY_LOG

    if clear:
        for lf in [COORDINATOR_LOG, PROXY_LOG]:
            if lf.exists():
                lf.write_text("")
        console.print("[green]Logs cleared.[/green]")
        return

    if not log_file.exists():
        console.print(f"[dim]No log file yet: {log_file}[/dim]")
        console.print(f"[dim]Start the {service} to generate logs.[/dim]")
        return

    if follow:
        import subprocess as sp
        try:
            sp.run(["tail", "-f", str(log_file)])
        except KeyboardInterrupt:
            pass
        return

    # Show last N lines
    text = log_file.read_text()
    tail = text.splitlines()[-lines:]
    if not tail:
        console.print("[dim]Log file is empty.[/dim]")
    else:
        for line in tail:
            console.print(line, highlight=False)


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


@cli.command()
@click.option("--fix", is_flag=True, help="Show suggested fix commands for failures")
@click.option("--json", "as_json", is_flag=True, help="Output machine-readable JSON report")
@click.pass_context
def doctor(ctx, fix, as_json):
    """Diagnose cluster configuration, connectivity, and version issues."""
    report = doctor_mod.run_doctor(ctx.obj.get("config_path"))

    if as_json:
        click.echo(json.dumps(report.to_dict(), indent=2))
    else:
        doctor_mod.render_report(console, report, show_fix=fix)

    if not report.passed:
        sys.exit(1)


@cli.command()
@click.option("--pid", "target_pid", type=int, default=None,
              help="PID of llama-server process (auto-detected from pidfile)")
@click.option("--model-path", type=click.Path(), default=None,
              help="Path to GGUF model file (auto-detected on Linux)")
def reclaim(target_pid, model_path):
    """Reclaim RAM from a running llama-server after model loading.

    Tells the OS to release file-backed pages from a llama-server process.
    On Linux uses posix_fadvise(DONTNEED), on Windows trims the working set.
    On macOS this is a no-op (unified memory).

    The coordinator PID is auto-detected from the pidfile if --pid is not given.
    On Linux, the model path is auto-detected from /proc/{pid}/maps.
    """
    from .reclaim import reclaim_ram
    from .coordinator import PIDFILE

    pid = target_pid
    if pid is None:
        if not PIDFILE.exists():
            console.print("[red]No coordinator running and no --pid specified.[/red]")
            console.print("Start the coordinator first, or provide --pid.")
            sys.exit(1)
        pid = int(PIDFILE.read_text().strip())
        # Verify it's alive
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            console.print(f"[red]PID {pid} from pidfile is not running.[/red]")
            sys.exit(1)
        except PermissionError:
            pass  # process exists but owned by another user — proceed

    console.print(f"[bold]Reclaiming RAM from PID {pid}...[/bold]")
    result = reclaim_ram(pid, model_path)

    if result.method == "skipped":
        msg = result.error or "not applicable on this platform"
        console.print(f"  RAM reclaim: [yellow]skipped[/yellow] ({msg})")
    else:
        console.print(f"  RSS before: {result.rss_before_mb:,.1f} MB")
        console.print(f"  RSS after:  {result.rss_after_mb:,.1f} MB")
        console.print(
            f"  [green]Reclaimed {result.reclaimed_mb:,.0f} MB "
            f"({result.method})[/green]"
        )


@cli.command()
@click.option("--model", "model_path", type=click.Path(exists=True), default=None,
              help="Model file to check RAM sufficiency against")
def tune(model_path):
    """Diagnose system RAM/swap and recommend tuning for large models."""
    from .tune import diagnose, recommend

    info = diagnose()

    console.print("[bold]System Resources:[/bold]")
    console.print(f"  RAM:        {info.total_ram_gb:.1f} GB ({info.available_ram_gb:.1f} GB available)")
    console.print(f"  Swap:       {info.swap_total_gb:.1f} GB ({info.swap_used_gb:.1f} GB used)")
    if info.vm_swappiness is not None:
        console.print(f"  Swappiness: {info.vm_swappiness}")
    if info.swap_on_nvme is not None:
        nvme_str = "[green]yes[/green]" if info.swap_on_nvme else "[yellow]no[/yellow]"
        console.print(f"  Swap NVMe:  {nvme_str}")

    model_size_gb = None
    if model_path:
        model_size = Path(model_path).stat().st_size
        model_size_gb = model_size / (1024**3)
        console.print(f"\n[bold]Model:[/bold] {Path(model_path).name} ({model_size_gb:.1f} GB)")

    recs = recommend(info, model_size_gb)
    console.print()

    severity_icons = {
        "critical": "[red bold][!] CRITICAL:[/red bold]",
        "warn": "[yellow][!] WARNING:[/yellow]",
        "info": "[dim][i][/dim]",
    }

    for rec in recs:
        icon = severity_icons.get(rec.severity, "[dim][i][/dim]")
        console.print(f"  {icon} {rec.message}")
        if rec.commands:
            console.print()
            for cmd in rec.commands:
                console.print(f"      {cmd}")
            console.print()


def _parse_size(s: str) -> int:
    """Parse a human-readable size string to bytes. E.g. '2G', '4096M', '2147483648'."""
    s = s.strip().upper()
    multipliers = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    if s and s[-1] in multipliers:
        return int(float(s[:-1]) * multipliers[s[-1]])
    return int(s)


@cli.command("load")
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--mem-limit", default=None, type=str,
              help="Memory limit (e.g. '2G', '4096M'). Reserved for v0.1.5 --force-constrain.")
@click.option("--no-prewarm", is_flag=True, help="Skip sequential pre-warming")
@click.option("--ram-reclaim", type=click.Choice(["off", "on", "auto"]), default=None,
              help="RAM reclaim mode after loading (default: auto)")
@click.option("--timeout", default=300.0, type=float, help="Health check timeout in seconds")
@click.pass_context
def load_cmd(ctx, model_path, mem_limit, no_prewarm, ram_reclaim, timeout):
    """Load a GGUF model with pre-warming and memory-aware startup.

    Pre-warms the page cache sequentially before llama-server mmaps the file,
    then reclaims RAM after the model loads to VRAM. Use this for standalone
    loading of any GGUF file.

    For models configured in cluster.yaml, use 'tightwad start' instead —
    it integrates pre-warming automatically when ram_reclaim is 'auto' or 'on'.
    """
    from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
    from .loader import (
        load_model, needs_streaming_load, prewarm_sequential,
    )
    from .reclaim import get_available_ram_bytes
    from .gguf_reader import read_header, model_summary

    model_path = Path(model_path)
    file_size = model_path.stat().st_size
    model_size_gb = file_size / (1024**3)

    if mem_limit is not None:
        console.print(
            "[yellow]--mem-limit is reserved for v0.1.5 (--force-constrain). "
            "Ignored in this release.[/yellow]"
        )

    # Parse GGUF header for display
    gguf_info = None
    try:
        header = read_header(model_path)
        gguf_info = model_summary(header)
    except Exception:
        pass

    console.print(f"\n[bold]Model:[/bold] {model_path.name}")
    if gguf_info:
        parts = []
        if gguf_info.get("arch"):
            parts.append(gguf_info["arch"])
        if gguf_info.get("layers"):
            parts.append(f"{gguf_info['layers']} layers")
        if gguf_info.get("quant"):
            parts.append(gguf_info["quant"])
        parts.append(f"{model_size_gb:.1f} GB")
        console.print(f"  {', '.join(parts)}")
    else:
        console.print(f"  Size: {model_size_gb:.1f} GB")

    available = get_available_ram_bytes()
    available_gb = available / (1024**3)
    console.print(f"\n[bold]System:[/bold] {available_gb:.1f} GB RAM available")

    needs_prewarm = needs_streaming_load(file_size, available)
    if needs_prewarm and not no_prewarm:
        console.print(f"  Strategy: pre-warm + reclaim (model > 80% of available RAM)")
    elif no_prewarm:
        console.print(f"  Strategy: skip pre-warm (--no-prewarm)")
    else:
        console.print(f"  Strategy: direct load (model fits in RAM)")

    # Pre-warm with progress bar
    if needs_prewarm and not no_prewarm:
        console.print()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed:.1f}/{task.total:.1f} GB"),
            console=console,
        ) as progress:
            task = progress.add_task("Pre-warming...", total=model_size_gb)

            def on_progress(bytes_read, total):
                progress.update(task, completed=bytes_read / (1024**3))

            elapsed = prewarm_sequential(
                model_path, file_size, progress_callback=on_progress,
            )
            throughput = model_size_gb / elapsed if elapsed > 0 else 0

        console.print(f"  Pre-warm: {elapsed:.1f}s ({throughput:.2f} GB/s)")

    # Start coordinator via config
    config = _load(ctx)
    mode = ram_reclaim or config.ram_reclaim

    console.print(f"\n[bold]Starting coordinator...[/bold]")
    try:
        result = load_model(
            config,
            model_name=str(model_path),
            prewarm=False,  # already pre-warmed above
            ram_reclaim=mode,
            wait_timeout=timeout,
        )
    except (ValueError, RuntimeError) as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    if result.healthy:
        console.print(f"  Health check: [green]OK[/green] ({result.load_time_seconds:.1f}s)")
    else:
        console.print(f"  Health check: [yellow]timeout[/yellow] ({timeout:.0f}s)")

    if result.reclaim_result:
        r = result.reclaim_result
        if r.method != "skipped":
            console.print(
                f"  [green]Reclaimed {r.reclaimed_mb:,.0f} MB RAM ({r.method})[/green]"
            )

    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  PID:       {result.pid}")
    console.print(f"  Peak RAM:  {result.peak_rss_mb:.1f} MB")
    console.print(f"  Model:     {result.model_size_gb:.1f} GB")
    console.print(f"  Load time: {result.load_time_seconds:.1f}s")


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
    console.print(f"  Dashboard: http://127.0.0.1:{pc.port}/dashboard")

    import uvicorn
    LOGDIR.mkdir(parents=True, exist_ok=True)
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "fmt": "%(asctime)s %(levelname)s %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "file": {
                "class": "logging.FileHandler",
                "filename": str(PROXY_LOG),
                "mode": "a",
                "formatter": "default",
            },
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["console", "file"], "level": "INFO"},
            "uvicorn.error": {"handlers": ["console", "file"], "level": "INFO"},
            "uvicorn.access": {"handlers": ["console", "file"], "level": "INFO"},
        },
    }
    app = proxy_mod.create_app(pc)
    proxy_mod.write_pidfile()
    try:
        uvicorn.run(app, host=pc.host, port=pc.port, log_level="info", log_config=log_config)
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
@click.option("--method", type=click.Choice(["auto", "rsync", "swarm"]), default="auto",
              help="Transfer method (default: auto-select by file size)")
@click.option("--token", default=None, help="Bearer token for swarm auth")
@click.option("--dry-run", is_flag=True, help="Preview transfers without executing")
@click.pass_context
def distribute_cmd(ctx, model_name, specific_target, method, token, dry_run):
    """Distribute a model to worker machines via rsync/scp or swarm P2P."""
    from .distribute import (
        resolve_targets, distribute, distribute_swarm,
        format_dry_run, auto_select_method,
    )

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

    # Resolve auto method
    if method == "auto":
        if local_path.exists():
            method = auto_select_method(local_path)
        else:
            method = "rsync"  # can't check size, fall back
        console.print(f"[dim]Auto-selected method: {method}[/dim]")

    if dry_run:
        console.print(format_dry_run(local_path, targets, method=method, token=token))
        return

    if not local_path.exists():
        console.print(f"[red]Model file not found: {local_path}[/red]")
        sys.exit(1)

    console.print(f"[bold]Distributing {local_path.name} to {len(targets)} target(s) via {method}...[/bold]\n")

    if method == "swarm":
        results = distribute_swarm(local_path, targets, console, token=token)
    else:
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
    prev_stats: dict | None = None
    status_url = f"{base_url}/v1/tightwad/status" if not direct else None
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
            import time as _time
            url = f"{base_url}/v1/chat/completions"
            body = {"messages": messages, "max_tokens": 1024, "temperature": 0.0, "stream": False}
            t0 = _time.monotonic()
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(url, json=body)
                resp.raise_for_status()
                data = resp.json()
            elapsed = _time.monotonic() - t0
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not text:
                text = data.get("choices", [{}])[0].get("text", "")
            console.print(f"[bold cyan]AI:[/bold cyan] {text}")
            messages.append({"role": "assistant", "content": text})

            # Inline speculation stats (speculative mode only)
            if status_url:
                try:
                    sr = httpx.get(status_url, timeout=3.0)
                    cur = sr.json().get("stats", {})
                    if prev_stats and cur.get("total_rounds", 0) > prev_stats.get("total_rounds", 0):
                        dr = cur["total_rounds"] - prev_stats["total_rounds"]
                        dd = cur["total_drafted"] - prev_stats["total_drafted"]
                        da = cur["total_accepted"] - prev_stats["total_accepted"]
                        rate = da / dd * 100 if dd > 0 else 0
                        tpr = da / dr if dr > 0 else 0
                        tok_s = da / elapsed if elapsed > 0 else 0
                        console.print(
                            f"  [dim]↳ {dr} round{'s' if dr != 1 else ''}, "
                            f"{dd} drafted, {da} accepted ({rate:.1f}%), "
                            f"{tpr:.1f} tok/round, {tok_s:.1f} tok/s[/dim]"
                        )
                    prev_stats = cur
                except Exception:
                    pass

            console.print()
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted.[/dim]\n")
            messages.pop()
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]\n")
            messages.pop()


# --- Manifest subcommand group ---


@cli.group()
def manifest():
    """Swarm manifest commands."""
    pass


@manifest.command("create")
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--piece-size", default=64, type=int, help="Piece size in MB (default: 64)")
@click.option("--no-inspect", is_flag=True, help="Skip GGUF metadata inspection")
@click.option("-o", "--output", "output_path", default=None, type=click.Path(), help="Output manifest path")
def manifest_create(model_path, piece_size, no_inspect, output_path):
    """Create a swarm manifest for a GGUF model file."""
    from pathlib import Path
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

    model_path = Path(model_path)
    piece_bytes = piece_size * 1024 * 1024
    total_size = model_path.stat().st_size
    est_pieces = (total_size + piece_bytes - 1) // piece_bytes

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} pieces"),
        console=console,
    ) as progress:
        task = progress.add_task("Hashing pieces...", total=est_pieces)

        def on_progress(done, _total):
            progress.update(task, completed=done)

        m = manifest_mod.create_manifest(
            model_path,
            piece_size=piece_bytes,
            use_gguf_inspect=not no_inspect,
            progress_callback=on_progress,
        )

    if output_path is None:
        output_path = model_path.parent / f"{model_path.name}.tightwad.manifest"
    else:
        output_path = Path(output_path)

    m.save(output_path)
    console.print(f"\n[green]Manifest created:[/green] {output_path}")
    console.print(f"  Model:    {m.model}")
    console.print(f"  Size:     {m.total_size / (1024**3):.2f} GB")
    console.print(f"  Pieces:   {m.num_pieces} x {piece_size} MB")
    if m.metadata:
        console.print(f"  Metadata: {json.dumps(m.metadata)}")


# --- Swarm subcommand group ---


@cli.group()
def swarm():
    """Swarm P2P transfer commands."""
    pass


@swarm.command("seed")
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--port", default=9080, type=int, help="Seeder port (default: 9080)")
@click.option("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
@click.option("--token", default=None, help="Require Bearer token for all requests")
@click.option("--allowed-ips", default=None, multiple=True, help="Restrict access to IP/CIDR (repeatable)")
def swarm_seed(model_path, port, host, token, allowed_ips):
    """Start a swarm seeder for a model file."""
    from pathlib import Path

    model_path = Path(model_path)

    # Load or create manifest
    m = manifest_mod.SwarmManifest.find_for_model(model_path)
    if m is None:
        console.print("[yellow]No manifest found. Creating one...[/yellow]")
        m = manifest_mod.create_manifest(model_path)
        manifest_file = model_path.parent / f"{model_path.name}.tightwad.manifest"
        m.save(manifest_file)
        console.print(f"[green]Manifest saved:[/green] {manifest_file}")

    # Build full bitfield (we have the complete file)
    bf = manifest_mod.PieceBitfield.load_or_create(
        model_path.parent / f"{model_path.name}.tightwad.pieces",
        m.num_pieces,
    )
    # Verify we have all pieces if bitfield is empty
    if not bf.have_all():
        for piece in m.pieces:
            if manifest_mod.verify_piece(model_path, piece):
                bf.mark_have(piece.index)
        bf.save()

    console.print(f"\n[bold]Starting swarm seeder...[/bold]")
    console.print(f"  Model:  {m.model} ({m.filename})")
    console.print(f"  Pieces: {len(bf.have)}/{m.num_pieces} ({bf.completion_pct():.0f}%)")
    console.print(f"  Listen: {host}:{port}")
    if token:
        console.print(f"  Auth:   Bearer token required")
    if allowed_ips:
        console.print(f"  IPs:    {', '.join(allowed_ips)}")

    swarm_mod.run_seeder(
        model_path, m, bf, host=host, port=port,
        token=token, allowed_ips=list(allowed_ips) if allowed_ips else None,
    )


@swarm.command("pull")
@click.argument("dest_path", type=click.Path())
@click.option("--manifest", "manifest_source", required=True, help="Path or URL to manifest")
@click.option("--peer", "peers", multiple=True, required=True, help="Peer URL (repeatable)")
@click.option("--parallel", default=4, type=int, help="Max concurrent downloads (default: 4)")
@click.option("--token", default=None, help="Bearer token for authenticated peers")
def swarm_pull(dest_path, manifest_source, peers, parallel, token):
    """Pull a model from swarm peers."""
    from pathlib import Path
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

    dest_path = Path(dest_path)

    # Load manifest from file or URL
    if manifest_source.startswith("http://") or manifest_source.startswith("https://"):
        import httpx
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        console.print(f"Fetching manifest from {manifest_source}...")
        resp = httpx.get(manifest_source, timeout=30.0, headers=headers)
        resp.raise_for_status()
        m = manifest_mod.SwarmManifest.from_dict(resp.json())
    else:
        m = manifest_mod.SwarmManifest.load(manifest_source)

    # Load or create bitfield for destination
    bf = manifest_mod.PieceBitfield.load_or_create(
        dest_path.parent / f"{dest_path.name}.tightwad.pieces",
        m.num_pieces,
    )

    missing = bf.missing_pieces()
    console.print(f"\n[bold]Pulling {m.filename}[/bold]")
    console.print(f"  Pieces: {m.num_pieces} total, {len(missing)} to download")
    console.print(f"  Peers:  {len(peers)}")
    console.print(f"  Parallel: {parallel}")

    if not missing:
        console.print("\n[green]Already complete![/green]")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading pieces...", total=len(missing))

        def on_progress(completed, total, piece_idx):
            progress.update(task, completed=completed, description=f"Piece {piece_idx}")

        ok = swarm_mod.run_puller(
            dest_path, m, bf, list(peers),
            max_concurrent=parallel,
            progress_callback=on_progress,
            token=token,
        )

    if ok:
        console.print(f"\n[green]Download complete:[/green] {dest_path}")
    else:
        console.print(f"\n[yellow]Download incomplete. Re-run to resume.[/yellow]")
        sys.exit(1)


@swarm.command("status")
@click.argument("model_path", type=click.Path(exists=True))
def swarm_status(model_path):
    """Show swarm status for a model file."""
    from pathlib import Path

    model_path = Path(model_path)

    m = manifest_mod.SwarmManifest.find_for_model(model_path)
    if m is None:
        console.print("[dim]No manifest found. Create one with:[/dim]")
        console.print(f"  tightwad manifest create {model_path}")
        return

    bf = manifest_mod.PieceBitfield.load_or_create(
        model_path.parent / f"{model_path.name}.tightwad.pieces",
        m.num_pieces,
    )

    # Check if we have the full file but empty bitfield
    if not bf.have and model_path.exists():
        console.print("[dim]Verifying pieces...[/dim]")
        for piece in m.pieces:
            if manifest_mod.verify_piece(model_path, piece):
                bf.mark_have(piece.index)
        bf.save()

    pct = bf.completion_pct()
    missing = bf.missing_pieces()

    console.print(f"\n[bold]Swarm Status:[/bold] {m.filename}")
    console.print(f"  Model:      {m.model}")
    console.print(f"  Size:       {m.total_size / (1024**3):.2f} GB")
    console.print(f"  Pieces:     {m.num_pieces} x {m.piece_size // (1024*1024)} MB")
    console.print(f"  Have:       {len(bf.have)}/{m.num_pieces} ({pct:.0f}%)")
    if missing:
        console.print(f"  Missing:    {len(missing)} pieces")
    else:
        console.print(f"  [green]Complete![/green]")
    if m.metadata:
        console.print(f"  Metadata:   {json.dumps(m.metadata)}")

    # Check for running seeder
    pid = swarm_mod.read_seeder_pidfile(m.model)
    if pid is not None:
        try:
            os.kill(pid, 0)
            console.print(f"  Seeder:     [green]running[/green] (PID {pid})")
        except ProcessLookupError:
            console.print(f"  Seeder:     [dim]not running[/dim]")
