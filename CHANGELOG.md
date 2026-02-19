# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-02-18

### Added
- `--version` flag to CLI
- Non-interactive `tightwad init` with `--draft-url` and `--target-url` flags
- SECURITY.md with vulnerability reporting guidelines
- Docker healthcheck and persistent log volume

### Fixed
- Broken wiki links in documentation
- XSS vulnerability in live dashboard (SEC-2)
- Code hardening: replaced assert guards, fixed log file descriptor leak
- Replaced nonexistent Qwen3-72B references with Qwen3-32B

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

[0.1.2]: https://github.com/akivasolutions/tightwad/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/akivasolutions/tightwad/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/akivasolutions/tightwad/releases/tag/v0.1.0
