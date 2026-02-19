"""Cluster configuration loader."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger("tightwad.config")


DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "cluster.yaml"


@dataclass
class GPU:
    name: str
    vram_gb: int
    rpc_port: int | None = None  # None for local coordinator GPUs


@dataclass
class Worker:
    host: str
    gpus: list[GPU] = field(default_factory=list)
    ssh_user: str | None = None
    model_dir: str | None = None

    @property
    def rpc_addresses(self) -> list[str]:
        return [f"{self.host}:{gpu.rpc_port}" for gpu in self.gpus]


@dataclass
class ModelConfig:
    name: str
    path: str
    ctx_size: int = 8192
    predict: int = 4096
    flash_attn: bool = True
    default: bool = False


@dataclass
class ServerEndpoint:
    url: str  # e.g. "http://192.168.1.100:8081"
    model_name: str  # display only
    backend: str = "llamacpp"  # "llamacpp" or "ollama"


@dataclass
class ProxyConfig:
    draft: ServerEndpoint
    target: ServerEndpoint
    host: str = "0.0.0.0"
    port: int = 8088
    max_draft_tokens: int = 8
    fallback_on_draft_failure: bool = True
    drafters: list[ServerEndpoint] = field(default_factory=list)
    #: Optional Bearer token that protects all /v1/ endpoints.
    #: When set, every request must include ``Authorization: Bearer <token>``.
    #: When unset the proxy operates in open (unauthenticated) mode and logs
    #: a startup warning — this preserves backward compatibility.
    auth_token: str | None = None
    #: Allow upstream URLs that resolve to private / internal IP ranges.
    #:
    #: Tightwad's most common deployment targets LAN servers (e.g.
    #: ``http://192.168.1.10:11434``) so the private-IP SSRF check defaults to
    #: ``True`` (opted in / allowed).  Set this to ``False`` in environments
    #: where the proxy should never reach internal infrastructure.
    #:
    #: The scheme check (http/https only) is **always** enforced regardless of
    #: this flag.
    #:
    #: Audit ref: SEC-5 / issue #7
    allow_private_upstream: bool = True
    #: Maximum value allowed for ``max_tokens`` in completion requests.
    #:
    #: Requests that exceed this limit are rejected with ``400 Bad Request``
    #: before the downstream server is contacted.  Very large ``max_tokens``
    #: values can force the downstream llama.cpp server into an extremely long
    #: generation — effectively a DoS against the backend.
    #:
    #: Configurable via ``TIGHTWAD_MAX_TOKENS_LIMIT`` env var or
    #: ``proxy.max_tokens_limit`` in cluster.yaml.
    #:
    #: Audit ref: CQ-1 / issue #8
    max_tokens_limit: int = 16384
    #: Maximum allowed request body size in bytes.
    #:
    #: Requests whose ``Content-Length`` header exceeds this value are rejected
    #: with ``413 Content Too Large`` *before* the body is buffered in memory,
    #: preventing memory-exhaustion DoS via multi-gigabyte payloads.
    #:
    #: Configurable via ``TIGHTWAD_MAX_BODY_SIZE`` env var (bytes) or
    #: ``proxy.max_body_size`` in cluster.yaml.
    #:
    #: Audit ref: CQ-5 / issue #8
    max_body_size: int = 10 * 1024 * 1024  # 10 MB default


@dataclass
class ClusterConfig:
    coordinator_host: str
    coordinator_port: int
    coordinator_backend: str
    coordinator_gpus: list[GPU]
    workers: list[Worker]
    models: dict[str, ModelConfig]
    coordinator_binary: str
    rpc_server_binary: str
    proxy: ProxyConfig | None = None
    ram_reclaim: str = "auto"  # "off", "on", "auto"

    @property
    def all_gpus(self) -> list[GPU]:
        gpus = list(self.coordinator_gpus)
        for w in self.workers:
            gpus.extend(w.gpus)
        return gpus

    @property
    def total_vram_gb(self) -> int:
        return sum(g.vram_gb for g in self.all_gpus)

    @property
    def rpc_addresses(self) -> list[str]:
        addrs: list[str] = []
        for w in self.workers:
            addrs.extend(w.rpc_addresses)
        return addrs

    def tensor_split(self) -> list[float]:
        """Calculate --tensor-split proportions from VRAM sizes."""
        gpus = self.all_gpus
        total = sum(g.vram_gb for g in gpus)
        return [round(g.vram_gb / total, 2) for g in gpus]

    def default_model(self) -> ModelConfig | None:
        for m in self.models.values():
            if m.default:
                return m
        # Return first model if none marked default
        return next(iter(self.models.values()), None)


def load_proxy_from_env() -> ProxyConfig | None:
    """Build a ProxyConfig from TIGHTWAD_* environment variables.

    Returns None if required env vars (TIGHTWAD_DRAFT_URL, TIGHTWAD_TARGET_URL)
    are not set.

    Raises
    ------
    ValueError
        If a supplied URL fails SSRF validation (bad scheme, private IP when
        not allowed, etc.).
    """
    draft_url = os.environ.get("TIGHTWAD_DRAFT_URL")
    target_url = os.environ.get("TIGHTWAD_TARGET_URL")
    if not draft_url or not target_url:
        return None

    # Token may be supplied via TIGHTWAD_PROXY_TOKEN (primary) or the legacy
    # TIGHTWAD_TOKEN alias used by the swarm seeder.
    auth_token = (
        os.environ.get("TIGHTWAD_PROXY_TOKEN")
        or os.environ.get("TIGHTWAD_TOKEN")
        or None
    )

    # TIGHTWAD_ALLOW_PRIVATE_UPSTREAM: set to "false" / "0" / "no" to
    # enforce private-IP blocking even in env-var mode.  Defaults to True
    # so that homelab LAN URLs continue to work without extra configuration.
    _priv_raw = os.environ.get("TIGHTWAD_ALLOW_PRIVATE_UPSTREAM", "true").lower()
    allow_private = _priv_raw not in ("false", "0", "no")

    # Validate URLs before building the config (SSRF: SEC-5)
    _validate_proxy_urls(
        draft_url=draft_url,
        target_url=target_url,
        allow_private=allow_private,
        source="environment variable",
    )

    max_tokens_limit = int(os.environ.get("TIGHTWAD_MAX_TOKENS_LIMIT", "16384"))
    max_body_size = int(os.environ.get("TIGHTWAD_MAX_BODY_SIZE", str(10 * 1024 * 1024)))

    return ProxyConfig(
        draft=ServerEndpoint(
            url=draft_url,
            model_name=os.environ.get("TIGHTWAD_DRAFT_MODEL", "draft"),
            backend=os.environ.get("TIGHTWAD_DRAFT_BACKEND", "ollama"),
        ),
        target=ServerEndpoint(
            url=target_url,
            model_name=os.environ.get("TIGHTWAD_TARGET_MODEL", "target"),
            backend=os.environ.get("TIGHTWAD_TARGET_BACKEND", "ollama"),
        ),
        host=os.environ.get("TIGHTWAD_HOST", "0.0.0.0"),
        port=int(os.environ.get("TIGHTWAD_PORT", "8088")),
        max_draft_tokens=int(os.environ.get("TIGHTWAD_MAX_DRAFT_TOKENS", "32")),
        auth_token=auth_token,
        allow_private_upstream=allow_private,
        max_tokens_limit=max_tokens_limit,
        max_body_size=max_body_size,
    )


def _validate_proxy_urls(
    *,
    draft_url: str,
    target_url: str,
    drafters: list[str] | None = None,
    allow_private: bool,
    source: str = "config",
) -> None:
    """Run SSRF validation on all proxy upstream URLs.

    Parameters
    ----------
    draft_url:
        The draft model's upstream URL.
    target_url:
        The target model's upstream URL.
    drafters:
        Optional list of additional drafter URLs to validate.
    allow_private:
        Forwarded to :func:`~tightwad.ssrf.validate_upstream_url`.
    source:
        Human-readable description of where the URLs came from (used in
        log messages and error context).

    Raises
    ------
    ValueError
        If any URL fails SSRF validation.
    """
    # Lazy import to avoid circular dependencies at module load time.
    from .ssrf import validate_upstream_url

    endpoints = [("proxy.draft", draft_url), ("proxy.target", target_url)]
    for label, url in (drafters or []):
        endpoints.append((label, url))

    for label, url in endpoints:
        try:
            validate_upstream_url(url, allow_private=allow_private)
        except ValueError as exc:
            raise ValueError(
                f"SSRF validation failed for {label} URL from {source}: {exc}"
            ) from exc
        logger.debug("ssrf: %s URL %r validated OK (allow_private=%s)", label, url, allow_private)


def load_config(path: str | Path | None = None) -> ClusterConfig:
    """Load cluster config from YAML file, falling back to env vars for proxy-only mode."""
    config_path = Path(path) if path else Path(
        os.environ.get("TIGHTWAD_CONFIG", DEFAULT_CONFIG)
    )

    if not config_path.exists():
        proxy = load_proxy_from_env()
        if proxy is not None:
            return ClusterConfig(
                coordinator_host="0.0.0.0",
                coordinator_port=8080,
                coordinator_backend="cuda",
                coordinator_gpus=[],
                workers=[],
                models={},
                coordinator_binary="llama-server",
                rpc_server_binary="rpc-server",
                proxy=proxy,
            )
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    coord = raw["coordinator"]
    coordinator_gpus = [
        GPU(name=g["name"], vram_gb=g["vram_gb"])
        for g in coord.get("gpus", [])
    ]

    workers = []
    for w in raw.get("workers", []):
        gpus = [
            GPU(name=g["name"], vram_gb=g["vram_gb"], rpc_port=g["rpc_port"])
            for g in w.get("gpus", [])
        ]
        workers.append(Worker(
            host=w["host"],
            gpus=gpus,
            ssh_user=w.get("ssh_user"),
            model_dir=w.get("model_dir"),
        ))

    models = {}
    for name, m in raw.get("models", {}).items():
        models[name] = ModelConfig(
            name=name,
            path=m["path"],
            ctx_size=m.get("ctx_size", 8192),
            predict=m.get("predict", 4096),
            flash_attn=m.get("flash_attn", True),
            default=m.get("default", False),
        )

    binaries = raw.get("binaries", {})

    proxy = None
    if "proxy" in raw:
        p = raw["proxy"]
        draft = p["draft"]
        target = p["target"]
        drafter_endpoints = []
        drafter_url_pairs: list[tuple[str, str]] = []
        for i, d in enumerate(p.get("drafters", [])):
            drafter_endpoints.append(ServerEndpoint(
                url=d["url"],
                model_name=d["model_name"],
                backend=d.get("backend", "llamacpp"),
            ))
            drafter_url_pairs.append((f"proxy.drafters[{i}]", d["url"]))

        # auth_token: read from YAML, then fall back to env vars so that
        # tokens can be injected at runtime without editing the config file.
        yaml_token = p.get("auth_token") or None
        env_token = (
            os.environ.get("TIGHTWAD_PROXY_TOKEN")
            or os.environ.get("TIGHTWAD_TOKEN")
            or None
        )
        # YAML value takes precedence; env var is the fallback.
        resolved_token = yaml_token or env_token

        # allow_private_upstream defaults to True so that common homelab
        # configs (targeting LAN addresses like 192.168.x.x) work without
        # any extra config.  Operators who want strict SSRF enforcement can
        # set allow_private_upstream: false in cluster.yaml.
        allow_private = p.get("allow_private_upstream", True)

        # Validate all upstream URLs before constructing clients (SSRF: SEC-5).
        _validate_proxy_urls(
            draft_url=draft["url"],
            target_url=target["url"],
            drafters=drafter_url_pairs,
            allow_private=allow_private,
            source=str(config_path),
        )

        proxy = ProxyConfig(
            draft=ServerEndpoint(url=draft["url"], model_name=draft["model_name"], backend=draft.get("backend", "llamacpp")),
            target=ServerEndpoint(url=target["url"], model_name=target["model_name"], backend=target.get("backend", "llamacpp")),
            host=p.get("host", "0.0.0.0"),
            port=p.get("port", 8088),
            max_draft_tokens=p.get("max_draft_tokens", 8),
            fallback_on_draft_failure=p.get("fallback_on_draft_failure", True),
            drafters=drafter_endpoints,
            auth_token=resolved_token,
            allow_private_upstream=allow_private,
            max_tokens_limit=p.get("max_tokens_limit", 16384),
            max_body_size=p.get("max_body_size", 10 * 1024 * 1024),
        )

    ram_reclaim = raw.get("ram_reclaim", "auto")
    if ram_reclaim not in ("off", "on", "auto"):
        logger.warning(
            "Invalid ram_reclaim value %r, defaulting to 'auto'", ram_reclaim
        )
        ram_reclaim = "auto"

    return ClusterConfig(
        coordinator_host=coord.get("host", "0.0.0.0"),
        coordinator_port=coord.get("port", 8080),
        coordinator_backend=coord.get("backend", "hip"),
        coordinator_gpus=coordinator_gpus,
        workers=workers,
        models=models,
        coordinator_binary=binaries.get("coordinator", "llama-server"),
        rpc_server_binary=binaries.get("rpc_server", "rpc-server"),
        proxy=proxy,
        ram_reclaim=ram_reclaim,
    )
