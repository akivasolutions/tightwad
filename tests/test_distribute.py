"""Tests for model distribution to workers."""

from pathlib import Path
from unittest.mock import patch

import pytest

from tightwad.distribute import (
    TransferTarget,
    build_transfer_cmd,
    resolve_targets,
    format_dry_run,
)
from tightwad.config import (
    GPU,
    Worker,
    ClusterConfig,
    ModelConfig,
)


@pytest.fixture
def cluster_config():
    return ClusterConfig(
        coordinator_host="0.0.0.0",
        coordinator_port=8080,
        coordinator_backend="cuda",
        coordinator_gpus=[GPU(name="P400", vram_gb=0)],
        workers=[
            Worker(
                host="192.168.1.100",
                gpus=[GPU(name="4070", vram_gb=16, rpc_port=50052)],
                ssh_user="youruser",
                model_dir="/models",
            ),
            Worker(
                host="192.168.1.200",
                gpus=[GPU(name="2070", vram_gb=8, rpc_port=50052)],
                ssh_user=None,
                model_dir="/data/models",
            ),
            Worker(
                host="192.168.1.300",
                gpus=[GPU(name="M2", vram_gb=16, rpc_port=50052)],
                # No model_dir — should be skipped
            ),
        ],
        models={
            "llama-3.3-70b": ModelConfig(
                name="llama-3.3-70b",
                path="/mnt/models/Llama-3.3-70B-Instruct-Q4_K_M.gguf",
                default=True,
            ),
        },
        coordinator_binary="llama-server",
        rpc_server_binary="rpc-server",
    )


class TestResolveTargets:
    def test_resolve_from_config(self, cluster_config):
        local_path, targets = resolve_targets(cluster_config, "llama-3.3-70b")
        assert local_path == Path("/mnt/models/Llama-3.3-70B-Instruct-Q4_K_M.gguf")
        # Only 2 workers have model_dir set
        assert len(targets) == 2
        assert targets[0].host == "192.168.1.100"
        assert targets[0].ssh_user == "youruser"
        assert targets[0].remote_path == "/models/Llama-3.3-70B-Instruct-Q4_K_M.gguf"
        assert targets[1].host == "192.168.1.200"
        assert targets[1].ssh_user is None

    def test_resolve_specific_target(self, cluster_config):
        local_path, targets = resolve_targets(
            cluster_config, "llama-3.3-70b",
            specific_target="10.0.0.1:/data/model.gguf",
        )
        assert len(targets) == 1
        assert targets[0].host == "10.0.0.1"
        assert targets[0].remote_path == "/data/model.gguf"

    def test_unknown_model_raises(self, cluster_config):
        with pytest.raises(ValueError, match="Unknown model"):
            resolve_targets(cluster_config, "nonexistent")

    def test_bad_specific_target_raises(self, cluster_config):
        with pytest.raises(ValueError, match="host:/path"):
            resolve_targets(cluster_config, "llama-3.3-70b", specific_target="nopath")


class TestBuildTransferCmd:
    def test_rsync_preferred(self):
        target = TransferTarget(
            host="192.168.1.100",
            ssh_user="youruser",
            remote_path="/models/model.gguf",
            worker_name="desktop",
        )
        with patch("shutil.which", return_value="/usr/bin/rsync"):
            cmd = build_transfer_cmd(Path("/local/model.gguf"), target)
        assert cmd[0] == "rsync"
        assert "youruser@192.168.1.100:/models/model.gguf" in cmd
        assert "--partial" in cmd

    def test_scp_fallback(self):
        target = TransferTarget(
            host="192.168.1.100",
            ssh_user=None,
            remote_path="/models/model.gguf",
            worker_name="desktop",
        )
        with patch("shutil.which", return_value=None):
            cmd = build_transfer_cmd(Path("/local/model.gguf"), target)
        assert cmd[0] == "scp"
        assert "192.168.1.100:/models/model.gguf" in cmd

    def test_no_user_prefix_when_none(self):
        target = TransferTarget(
            host="10.0.0.1",
            ssh_user=None,
            remote_path="/models/m.gguf",
            worker_name="test",
        )
        with patch("shutil.which", return_value="/usr/bin/rsync"):
            cmd = build_transfer_cmd(Path("/local/m.gguf"), target)
        dest = cmd[-1]
        assert dest == "10.0.0.1:/models/m.gguf"
        assert "@" not in dest


class TestFormatDryRun:
    def test_dry_run_output(self, tmp_path):
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"\x00" * 1024)

        targets = [
            TransferTarget(
                host="192.168.1.100",
                ssh_user="youruser",
                remote_path="/models/model.gguf",
                worker_name="desktop (4070)",
            ),
        ]
        output = format_dry_run(model_file, targets)
        assert "model.gguf" in output
        assert "desktop (4070)" in output
        assert "192.168.1.100" in output

    def test_dry_run_missing_file(self):
        targets = [
            TransferTarget(
                host="10.0.0.1",
                ssh_user=None,
                remote_path="/m.gguf",
                worker_name="test",
            ),
        ]
        output = format_dry_run(Path("/nonexistent/model.gguf"), targets)
        assert "not found" in output
