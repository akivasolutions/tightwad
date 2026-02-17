"""Async speculative decoding proxy server."""

from __future__ import annotations

import json
import logging
import os
import signal
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from .config import ProxyConfig
from .speculation import (
    DraftToken,
    TargetLogprob,
    VerificationResult,
    verify_draft_tokens,
)

logger = logging.getLogger("tightwad.proxy")

PIDFILE = Path.home() / ".tightwad" / "proxy.pid"


@dataclass
class ProxyStats:
    total_rounds: int = 0
    total_drafted: int = 0
    total_accepted: int = 0
    total_bonus: int = 0
    total_resampled: int = 0
    total_tokens_output: int = 0
    start_time: float = field(default_factory=time.monotonic)

    @property
    def acceptance_rate(self) -> float:
        if self.total_drafted == 0:
            return 0.0
        return self.total_accepted / self.total_drafted

    @property
    def effective_tokens_per_round(self) -> float:
        if self.total_rounds == 0:
            return 0.0
        return self.total_tokens_output / self.total_rounds

    @property
    def uptime_seconds(self) -> float:
        return time.monotonic() - self.start_time


class SpeculativeProxy:
    def __init__(self, config: ProxyConfig):
        self.config = config
        self.draft_client = httpx.AsyncClient(
            base_url=config.draft.url, timeout=30.0
        )
        self.target_client = httpx.AsyncClient(
            base_url=config.target.url, timeout=120.0
        )
        self.stats = ProxyStats()

    async def close(self):
        await self.draft_client.aclose()
        await self.target_client.aclose()

    async def check_server(self, url: str, backend: str) -> dict:
        try:
            # Ollama uses GET / which returns "Ollama is running"
            check_url = f"{url}/" if backend == "ollama" else f"{url}/health"
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(check_url)
            if resp.status_code == 200:
                try:
                    return {"alive": True, "status": resp.json()}
                except Exception:
                    return {"alive": True, "status": resp.text.strip()}
            return {"alive": False, "error": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"alive": False, "error": str(e)}

    async def draft_tokens(
        self, prompt: str, n: int, temperature: float = 0.0
    ) -> list[DraftToken]:
        """Generate n draft tokens from the draft model."""
        if self.config.draft.backend == "ollama":
            return await self._draft_ollama(prompt, n, temperature)
        return await self._draft_llamacpp(prompt, n, temperature)

    async def _draft_ollama(
        self, prompt: str, n: int, temperature: float
    ) -> list[DraftToken]:
        """Draft via Ollama's native /api/generate with raw:true."""
        body = {
            "model": self.config.draft.model_name,
            "prompt": prompt,
            "raw": True,
            "stream": False,
            "options": {
                "num_predict": n,
                "temperature": temperature,
            },
        }
        url = f"{self.config.draft.url}/api/generate"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=body)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("response", "")
        if not text:
            return []
        # Ollama doesn't return per-token info, so we treat the whole
        # response as one "token" for text-match verification
        return [DraftToken(token_id=0, logprob=0.0, text=text)]

    async def _draft_llamacpp(
        self, prompt: str, n: int, temperature: float
    ) -> list[DraftToken]:
        """Draft via llama-server /v1/completions with logprobs."""
        body: dict = {
            "prompt": prompt,
            "max_tokens": n,
            "temperature": temperature,
            "logprobs": 1,
            "stream": False,
        }
        resp = await self.draft_client.post("/v1/completions", json=body)
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        text = choice.get("text", "")

        tokens: list[DraftToken] = []
        logprobs_data = choice.get("logprobs")
        if logprobs_data and logprobs_data.get("content"):
            # llama-server format: logprobs.content[].{id, token, logprob}
            for entry in logprobs_data["content"]:
                tokens.append(DraftToken(
                    token_id=entry.get("id", 0),
                    logprob=entry.get("logprob", 0.0),
                    text=entry.get("token", ""),
                ))
        elif text:
            tokens.append(DraftToken(token_id=0, logprob=0.0, text=text))

        return tokens

    async def _generate_target(
        self, prompt: str, n: int, temperature: float
    ) -> str:
        """Generate text from target model."""
        if self.config.target.backend == "ollama":
            url = f"{self.config.target.url}/api/generate"
            body = {
                "model": self.config.target.model_name,
                "prompt": prompt,
                "raw": True,
                "stream": False,
                "options": {
                    "num_predict": n,
                    "temperature": temperature,
                },
            }
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(url, json=body)
            resp.raise_for_status()
            return resp.json().get("response", "")
        else:
            body: dict = {
                "prompt": prompt,
                "max_tokens": n,
                "temperature": temperature,
                "stream": False,
            }
            resp = await self.target_client.post("/v1/completions", json=body)
            resp.raise_for_status()
            return resp.json()["choices"][0].get("text", "")

    async def verify_and_accept(
        self, prompt: str, draft_text: str, temperature: float = 0.0
    ) -> tuple[str, int, int]:
        """Verify draft text against target model.

        Uses text-match greedy verification: target generates from the same
        prompt, and we find the longest matching prefix.

        Returns (accepted_text, n_accepted_chars, n_draft_chars).
        """
        n_draft_chars = len(draft_text)

        target_text = await self._generate_target(
            prompt, self.config.max_draft_tokens, temperature
        )

        # Find longest common prefix
        match_len = 0
        for i in range(min(len(draft_text), len(target_text))):
            if draft_text[i] == target_text[i]:
                match_len = i + 1
            else:
                break

        # Accept the matching prefix + target's continuation from divergence point
        if match_len == len(draft_text) and match_len == len(target_text):
            # Perfect match
            return target_text, match_len, n_draft_chars
        elif match_len == len(draft_text):
            # Draft is a prefix of target — accept all draft + target's extra
            return target_text, match_len, n_draft_chars
        else:
            # Divergence — accept common prefix + target from that point
            return target_text, match_len, n_draft_chars

    async def speculation_round(
        self, prompt: str, temperature: float = 0.0
    ) -> tuple[str, bool]:
        """One full draft → verify → accept cycle.

        Returns (accepted_text, is_done).
        is_done is True if we hit EOS or empty generation.
        """
        # Draft phase
        try:
            draft = await self.draft_tokens(
                prompt, self.config.max_draft_tokens, temperature
            )
        except Exception as exc:
            if self.config.fallback_on_draft_failure:
                logger.warning("Draft failed (%s: %s), falling back to target", type(exc).__name__, exc)
                return await self._fallback_generate(
                    prompt, self.config.max_draft_tokens, temperature
                )
            raise

        if not draft:
            return "", True

        draft_text = "".join(t.text for t in draft)
        if not draft_text:
            return "", True

        # Verify phase — text-match greedy
        accepted_text, match_len, draft_len = await self.verify_and_accept(
            prompt, draft_text, temperature
        )

        # Update stats
        self.stats.total_rounds += 1
        self.stats.total_drafted += draft_len
        self.stats.total_accepted += match_len
        self.stats.total_tokens_output += len(accepted_text)

        if match_len < draft_len:
            self.stats.total_resampled += 1
        elif len(accepted_text) > draft_len:
            self.stats.total_bonus += 1

        is_done = len(accepted_text) == 0
        return accepted_text, is_done

    async def _fallback_generate(
        self, prompt: str, n: int, temperature: float
    ) -> tuple[str, bool]:
        """Direct generation from target (no speculation)."""
        text = await self._generate_target(prompt, n, temperature)
        self.stats.total_tokens_output += len(text)
        return text, not text

    async def generate_completion(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stream: bool = False,
        stop: list[str] | None = None,
    ):
        """Full speculative generation loop."""
        generated = ""
        tokens_generated = 0

        if stream:
            async def sse_stream():
                nonlocal generated, tokens_generated
                while tokens_generated < max_tokens:
                    chunk, done = await self.speculation_round(
                        prompt + generated, temperature
                    )
                    if not chunk and done:
                        break
                    generated += chunk
                    tokens_generated += len(chunk.split())  # approximate

                    # Check stop sequences
                    if stop:
                        for s in stop:
                            idx = generated.find(s)
                            if idx != -1:
                                generated = generated[:idx]
                                chunk = ""
                                done = True
                                break

                    # SSE chunk
                    sse_data = {
                        "id": f"cmpl-tightwad-{id(self)}",
                        "object": "text_completion",
                        "choices": [{
                            "index": 0,
                            "text": chunk,
                            "finish_reason": None,
                        }],
                    }
                    yield f"data: {json.dumps(sse_data)}\n\n"

                    if done:
                        break

                # Final chunk with finish_reason
                final = {
                    "id": f"cmpl-tightwad-{id(self)}",
                    "object": "text_completion",
                    "choices": [{
                        "index": 0,
                        "text": "",
                        "finish_reason": "stop" if tokens_generated < max_tokens else "length",
                    }],
                }
                yield f"data: {json.dumps(final)}\n\n"
                yield "data: [DONE]\n\n"

            return sse_stream()
        else:
            while tokens_generated < max_tokens:
                chunk, done = await self.speculation_round(
                    prompt + generated, temperature
                )
                if not chunk and done:
                    break
                generated += chunk
                tokens_generated += len(chunk.split())

                if stop:
                    for s in stop:
                        idx = generated.find(s)
                        if idx != -1:
                            generated = generated[:idx]
                            done = True
                            break

                if done:
                    break

            return {
                "id": f"cmpl-tightwad-{id(self)}",
                "object": "text_completion",
                "choices": [{
                    "index": 0,
                    "text": generated,
                    "finish_reason": "stop" if tokens_generated < max_tokens else "length",
                }],
                "usage": {
                    "completion_tokens": tokens_generated,
                },
            }


# --- Chat template helpers ---

QWEN3_CHAT_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "{messages}"
    "<|im_start|>assistant\n"
)
QWEN3_MESSAGE_TEMPLATE = "<|im_start|>{role}\n{content}<|im_end|>\n"


def apply_chat_template(messages: list[dict]) -> str:
    """Convert OpenAI chat messages to Qwen3 prompt format."""
    system = "You are a helpful assistant."
    parts = []

    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        if role == "system":
            system = content
        else:
            parts.append(QWEN3_MESSAGE_TEMPLATE.format(role=role, content=content))

    return QWEN3_CHAT_TEMPLATE.format(system=system, messages="".join(parts))


# --- Starlette app ---

_proxy: SpeculativeProxy | None = None


def _get_proxy() -> SpeculativeProxy:
    assert _proxy is not None, "Proxy not initialized"
    return _proxy


async def handle_completion(request: Request):
    proxy = _get_proxy()
    body = await request.json()

    prompt = body.get("prompt", "")
    max_tokens = body.get("max_tokens", 256)
    temperature = body.get("temperature", 0.0)
    stream = body.get("stream", False)
    stop = body.get("stop")
    if isinstance(stop, str):
        stop = [stop]

    result = await proxy.generate_completion(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=stream,
        stop=stop,
    )

    if stream:
        return StreamingResponse(
            result,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return JSONResponse(result)


async def handle_chat_completion(request: Request):
    proxy = _get_proxy()
    body = await request.json()

    messages = body.get("messages", [])
    prompt = apply_chat_template(messages)
    max_tokens = body.get("max_tokens", 256)
    temperature = body.get("temperature", 0.0)
    stream = body.get("stream", False)
    stop = body.get("stop")
    if isinstance(stop, str):
        stop = [stop]

    # Add Qwen3 stop tokens
    chat_stop = list(stop) if stop else []
    chat_stop.append("<|im_end|>")

    result = await proxy.generate_completion(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=stream,
        stop=chat_stop,
    )

    if stream:
        return StreamingResponse(
            result,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Reformat as chat completion response
    text = result["choices"][0]["text"]
    chat_result = {
        "id": result["id"].replace("cmpl-", "chatcmpl-"),
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": result["choices"][0]["finish_reason"],
        }],
        "usage": result.get("usage", {}),
    }
    return JSONResponse(chat_result)


async def handle_status(request: Request):
    proxy = _get_proxy()
    draft_health = await proxy.check_server(proxy.config.draft.url, proxy.config.draft.backend)
    target_health = await proxy.check_server(proxy.config.target.url, proxy.config.target.backend)

    stats = proxy.stats
    return JSONResponse({
        "draft": {
            "url": proxy.config.draft.url,
            "model": proxy.config.draft.model_name,
            "health": draft_health,
        },
        "target": {
            "url": proxy.config.target.url,
            "model": proxy.config.target.model_name,
            "health": target_health,
        },
        "config": {
            "max_draft_tokens": proxy.config.max_draft_tokens,
            "fallback_on_draft_failure": proxy.config.fallback_on_draft_failure,
        },
        "stats": {
            "total_rounds": stats.total_rounds,
            "total_drafted": stats.total_drafted,
            "total_accepted": stats.total_accepted,
            "total_bonus": stats.total_bonus,
            "total_resampled": stats.total_resampled,
            "total_tokens_output": stats.total_tokens_output,
            "acceptance_rate": round(stats.acceptance_rate, 3),
            "effective_tokens_per_round": round(stats.effective_tokens_per_round, 2),
            "uptime_seconds": round(stats.uptime_seconds, 1),
        },
    })


async def handle_models(request: Request):
    proxy = _get_proxy()
    return JSONResponse({
        "object": "list",
        "data": [
            {
                "id": proxy.config.target.model_name,
                "object": "model",
                "owned_by": "tightwad",
            },
            {
                "id": proxy.config.draft.model_name,
                "object": "model",
                "owned_by": "tightwad",
                "meta": {"role": "draft"},
            },
        ],
    })


def create_app(config: ProxyConfig) -> Starlette:
    global _proxy
    _proxy = SpeculativeProxy(config)

    @asynccontextmanager
    async def lifespan(app):
        yield
        if _proxy:
            await _proxy.close()

    app = Starlette(
        routes=[
            Route("/v1/completions", handle_completion, methods=["POST"]),
            Route("/v1/chat/completions", handle_chat_completion, methods=["POST"]),
            Route("/v1/models", handle_models, methods=["GET"]),
            Route("/v1/tightwad/status", handle_status, methods=["GET"]),
        ],
        lifespan=lifespan,
    )
    return app


def write_pidfile():
    PIDFILE.parent.mkdir(parents=True, exist_ok=True)
    PIDFILE.write_text(str(os.getpid()))


def remove_pidfile():
    PIDFILE.unlink(missing_ok=True)


def read_pidfile() -> int | None:
    if PIDFILE.exists():
        try:
            return int(PIDFILE.read_text().strip())
        except (ValueError, OSError):
            return None
    return None


def stop_proxy() -> bool:
    pid = read_pidfile()
    if pid is None:
        return False
    try:
        os.kill(pid, signal.SIGTERM)
        remove_pidfile()
        return True
    except ProcessLookupError:
        remove_pidfile()
        return False
