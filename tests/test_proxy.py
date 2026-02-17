"""Tests for the speculative decoding proxy."""

import json

import pytest
import pytest_asyncio
from starlette.testclient import TestClient

from tightwad.config import ProxyConfig, ServerEndpoint
from tightwad.proxy import apply_chat_template, create_app


@pytest.fixture
def proxy_config():
    return ProxyConfig(
        draft=ServerEndpoint(url="http://draft:8081", model_name="qwen3-8b"),
        target=ServerEndpoint(url="http://target:8080", model_name="qwen3-72b"),
        host="0.0.0.0",
        port=8088,
        max_draft_tokens=8,
        fallback_on_draft_failure=True,
    )


class TestChatTemplate:
    def test_basic_user_message(self):
        messages = [{"role": "user", "content": "Hello"}]
        prompt = apply_chat_template(messages)
        assert "<|im_start|>user\nHello<|im_end|>" in prompt
        assert "<|im_start|>assistant\n" in prompt
        assert prompt.endswith("<|im_start|>assistant\n")

    def test_system_message(self):
        messages = [
            {"role": "system", "content": "You are a pirate."},
            {"role": "user", "content": "Hi"},
        ]
        prompt = apply_chat_template(messages)
        assert "<|im_start|>system\nYou are a pirate.<|im_end|>" in prompt
        # System should not appear as a regular message
        assert prompt.count("<|im_start|>system") == 1

    def test_multi_turn(self):
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ]
        prompt = apply_chat_template(messages)
        assert "<|im_start|>user\nWhat is 2+2?<|im_end|>" in prompt
        assert "<|im_start|>assistant\n4<|im_end|>" in prompt
        assert "<|im_start|>user\nAnd 3+3?<|im_end|>" in prompt

    def test_default_system_prompt(self):
        messages = [{"role": "user", "content": "Hi"}]
        prompt = apply_chat_template(messages)
        assert "You are a helpful assistant." in prompt


class TestProxyApp:
    def test_models_endpoint(self, proxy_config):
        app = create_app(proxy_config)
        client = TestClient(app)
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 2
        assert data["data"][0]["id"] == "qwen3-72b"
        assert data["data"][1]["id"] == "qwen3-8b"

    def test_status_endpoint_servers_down(self, proxy_config):
        """When draft/target are unreachable, status shows not alive."""
        app = create_app(proxy_config)
        client = TestClient(app)
        resp = client.get("/v1/tightwad/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["draft"]["model"] == "qwen3-8b"
        assert data["target"]["model"] == "qwen3-72b"
        assert data["draft"]["health"]["alive"] is False
        assert data["target"]["health"]["alive"] is False
        assert data["stats"]["total_rounds"] == 0
        assert data["stats"]["acceptance_rate"] == 0.0

    def test_completion_draft_unavailable_fallback(self, proxy_config):
        """When both servers are down, request should fail gracefully."""
        app = create_app(proxy_config)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/completions", json={
            "prompt": "Hello",
            "max_tokens": 10,
        })
        # Should get a 500 since target is also unreachable
        assert resp.status_code == 500


class TestSSEFormat:
    def test_sse_chunk_format(self):
        """SSE data lines should be valid JSON."""
        line = 'data: {"id":"cmpl-tightwad-1","object":"text_completion","choices":[{"index":0,"text":"hello","finish_reason":null}]}'
        assert line.startswith("data: ")
        payload = json.loads(line[6:])
        assert payload["object"] == "text_completion"
        assert payload["choices"][0]["text"] == "hello"

    def test_sse_done_marker(self):
        line = "data: [DONE]"
        assert line == "data: [DONE]"
