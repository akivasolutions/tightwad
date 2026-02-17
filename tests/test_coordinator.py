"""Tests for coordinator command building."""

import pytest
import yaml

from tightwad.config import load_config
from tightwad.coordinator import build_server_args


@pytest.fixture
def config(tmp_path):
    cfg = {
        "coordinator": {
            "host": "0.0.0.0",
            "port": 8080,
            "backend": "hip",
            "gpus": [
                {"name": "XTX 0", "vram_gb": 24},
                {"name": "XTX 1", "vram_gb": 24},
            ],
        },
        "workers": [
            {
                "host": "192.168.1.100",
                "gpus": [
                    {"name": "4070", "vram_gb": 16, "rpc_port": 50052},
                    {"name": "3060", "vram_gb": 12, "rpc_port": 50053},
                ],
            }
        ],
        "models": {
            "qwen3-72b": {
                "path": "/models/qwen3-72b.gguf",
                "ctx_size": 8192,
                "predict": 4096,
                "flash_attn": True,
                "default": True,
            }
        },
        "binaries": {"coordinator": "llama-server"},
    }
    p = tmp_path / "cluster.yaml"
    p.write_text(yaml.dump(cfg))
    return load_config(p)


def test_build_args_basic(config):
    model = config.default_model()
    args = build_server_args(config, model)

    assert args[0] == "llama-server"
    assert "-m" in args
    assert args[args.index("-m") + 1] == "/models/qwen3-72b.gguf"
    assert "-ngl" in args
    assert "999" in args
    assert "--flash-attn" in args


def test_build_args_rpc(config):
    model = config.default_model()
    args = build_server_args(config, model)

    assert "--rpc" in args
    rpc_val = args[args.index("--rpc") + 1]
    assert "192.168.1.100:50052" in rpc_val
    assert "192.168.1.100:50053" in rpc_val


def test_build_args_tensor_split(config):
    model = config.default_model()
    args = build_server_args(config, model)

    assert "--tensor-split" in args
    split_val = args[args.index("--tensor-split") + 1]
    parts = split_val.split(",")
    assert len(parts) == 4
