# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GitHub Actions CI workflow with Python 3.10–3.13 test matrix (#18)
- CHANGELOG.md, CONTRIBUTING.md, and CODE_OF_CONDUCT.md (#19)
- GitHub issue templates (bug report, feature request) and PR template (#20)
- Example configs: minimal spec decode, CPU draft, two-GPU, mixed-vendor, combined mode (#21)
- Documentation: quickstart guide, configuration reference, architecture overview (#22)
- README badges: PyPI version, CI status, license, Python versions (#23)

## [0.1.3] - 2026-02-18

### Added
- **`tightwad reclaim` command** — free RAM after model loads to VRAM. Cross-platform: `posix_fadvise(DONTNEED)` on Linux, `SetProcessWorkingSetSize` on Windows, no-op on macOS (unified memory)
- **`tightwad tune` command** — diagnose system RAM/swap readiness for large models with platform-specific fix commands
- **`--ram-reclaim` flag on `tightwad start`** — modes: `off`, `on`, `auto` (default). Auto reclaims if model > 50% of available RAM
- `ram_reclaim` config option in cluster.yaml (top-level, defaults to `auto`)
- `start_and_reclaim()` coordinator function — waits for `/health` 200 before reclaiming (prevents evicting pages the server still needs)
- Auto-detection of model path from `/proc/{pid}/maps` on Linux
- Auto-detection of coordinator PID from pidfile for `tightwad reclaim`
- Cross-platform RSS and available RAM reading without psutil dependency

## [0.1.2] - 2026-02-18

### Added
- `--version` flag to CLI
- Non-interactive `tightwad init` with `--draft-url` and `--target-url` flags
- SECURITY.md with vulnerability reporting guidelines
- Docker healthcheck and persistent log volume

### Security
- **Bearer-token authentication for proxy API** — optional `auth_token` config or `TIGHTWAD_PROXY_TOKEN` env var; logs warning if proxy starts without auth (#6)
- **SSRF protection on upstream URLs** — scheme allowlist (http/https only), optional private-IP blocking via `allow_private_upstream` config, DNS-rebinding protection (#7)
- **Input validation and request size limits** — `max_tokens_limit` (default 16384) rejects excessive generation requests; `max_body_size` (default 10 MB) rejects oversized payloads before buffering (#8)
- Fixed XSS vulnerability in live dashboard (SEC-2, #9)
- Code hardening: replaced assert guards with proper exceptions, fixed log file descriptor leak (#10)

### Fixed
- Broken wiki links in documentation
- Replaced nonexistent Qwen3-72B references with Qwen3-32B
- Sanitized example configs and scripts (removed hardcoded IPs)

## [0.1.1] - 2026-02-17

### Added
- `tightwad doctor` diagnostic command for cluster validation
- `tightwad logs` command with `--follow` and `--clear` options
- `tightwad chat` interactive mode with inline speculation stats
- Live web dashboard for proxy monitoring (`/dashboard`)
- Docker Compose support with environment variable configuration
- `tightwad init` wizard for auto-discovering LAN inference servers
- Swarm security: `--token` and `--allowed-ips` flags for seeder

### Fixed
- Mocked `draft_client.post` in verify_with_logprobs tests

## [0.1.0] - 2026-02-16

### Added
- Initial release
- Mixed-vendor GPU pooling via llama.cpp RPC
- Speculative decoding proxy (draft + verify across machines)
- Combined mode: speculation over RPC pool
- GGUF model inspection (`tightwad inspect`)
- Model distribution to workers (`tightwad distribute`)
- Swarm P2P transfer protocol with manifest and chunked downloads
- Benchmark scripts for local and OpenRouter models
- Support for CUDA, ROCm, and CPU backends
- OpenAI-compatible API endpoint

[Unreleased]: https://github.com/akivasolutions/tightwad/compare/v0.1.3...HEAD
[0.1.3]: https://github.com/akivasolutions/tightwad/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/akivasolutions/tightwad/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/akivasolutions/tightwad/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/akivasolutions/tightwad/releases/tag/v0.1.0
