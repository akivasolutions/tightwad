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

import asyncio

import httpx
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

from .config import ProxyConfig, ServerEndpoint
from .speculation import (
    DraftToken,
    TargetLogprob,
    VerificationResult,
    verify_draft_tokens,
)
from .validation import (
    ValidationError,
    parse_chat_completion_request,
    parse_completion_request,
)

logger = logging.getLogger("tightwad.proxy")

PIDFILE = Path.home() / ".tightwad" / "proxy.pid"


class TokenAuthMiddleware:
    """Starlette ASGI middleware that enforces Bearer-token authentication.

    When a token is configured every HTTP request must include::

        Authorization: Bearer <token>

    Requests missing or presenting an incorrect token receive a ``401
    Unauthorized`` response.  Non-HTTP scopes (WebSocket, lifespan) pass
    through unchanged.

    When no token is configured the middleware is a transparent no-op —
    this preserves backward compatibility with unauthenticated deployments.
    """

    def __init__(self, app: ASGIApp, token: str | None) -> None:
        self.app = app
        self.token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # Only enforce auth on HTTP requests; let lifespan/websocket through.
        if scope["type"] != "http" or not self.token:
            await self.app(scope, receive, send)
            return

        request = Request(scope)
        auth_header = request.headers.get("authorization", "")
        expected = f"Bearer {self.token}"

        if auth_header != expected:
            response = Response(
                content='{"detail":"Unauthorized"}',
                status_code=401,
                media_type="application/json",
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)


@dataclass
class RequestRecord:
    timestamp: float
    rounds: int
    drafted: int
    accepted: int
    acceptance_rate: float
    draft_ms: float
    verify_ms: float
    total_ms: float
    tokens_output: int
    model: str


MAX_REQUEST_HISTORY = 50


@dataclass
class ProxyStats:
    total_rounds: int = 0
    total_drafted: int = 0
    total_accepted: int = 0
    total_bonus: int = 0
    total_resampled: int = 0
    total_tokens_output: int = 0
    drafter_wins: dict[str, int] = field(default_factory=dict)
    start_time: float = field(default_factory=time.monotonic)
    request_history: list = field(default_factory=list)

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
        self._multi_drafter = bool(config.drafters)
        self._winning_drafter: ServerEndpoint | None = None

        # Per-drafter clients for multi-drafter mode
        self.draft_clients: list[tuple[ServerEndpoint, httpx.AsyncClient]] = []
        for drafter in config.drafters:
            client = httpx.AsyncClient(base_url=drafter.url, timeout=30.0)
            self.draft_clients.append((drafter, client))

    async def close(self):
        await self.draft_client.aclose()
        await self.target_client.aclose()
        for _, client in self.draft_clients:
            await client.aclose()

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
        # Qwen3 thinking mode puts output in "thinking" when "response" is empty
        text = data.get("response", "") or data.get("thinking", "")
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

    async def _draft_from_endpoint(
        self, endpoint: ServerEndpoint, client: httpx.AsyncClient,
        prompt: str, n: int, temperature: float,
    ) -> list[DraftToken]:
        """Draft tokens from a specific endpoint using its backend."""
        if endpoint.backend == "ollama":
            body = {
                "model": endpoint.model_name,
                "prompt": prompt,
                "raw": True,
                "stream": False,
                "options": {"num_predict": n, "temperature": temperature},
            }
            url = f"{endpoint.url}/api/generate"
            async with httpx.AsyncClient(timeout=30.0) as tmp:
                resp = await tmp.post(url, json=body)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("response", "") or data.get("thinking", "")
            if not text:
                return []
            return [DraftToken(token_id=0, logprob=0.0, text=text)]
        else:
            body: dict = {
                "prompt": prompt,
                "max_tokens": n,
                "temperature": temperature,
                "logprobs": 1,
                "stream": False,
            }
            resp = await client.post("/v1/completions", json=body)
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            text = choice.get("text", "")
            tokens: list[DraftToken] = []
            logprobs_data = choice.get("logprobs")
            if logprobs_data and logprobs_data.get("content"):
                for entry in logprobs_data["content"]:
                    tokens.append(DraftToken(
                        token_id=entry.get("id", 0),
                        logprob=entry.get("logprob", 0.0),
                        text=entry.get("token", ""),
                    ))
            elif text:
                tokens.append(DraftToken(token_id=0, logprob=0.0, text=text))
            return tokens

    async def draft_tokens_parallel(
        self, prompt: str, n: int, temperature: float = 0.0,
    ) -> list[DraftToken]:
        """Race all drafters in parallel. Take the first llamacpp result
        that arrives, or wait up to 0.5s for a better one after the first
        result comes in."""

        async def _run(endpoint: ServerEndpoint, client: httpx.AsyncClient):
            return endpoint, await self._draft_from_endpoint(
                endpoint, client, prompt, n, temperature
            )

        pending = {asyncio.ensure_future(_run(ep, cl)) for ep, cl in self.draft_clients}
        candidates: list[tuple[ServerEndpoint, list[DraftToken]]] = []
        has_llamacpp = False

        # Wait for the first result
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                try:
                    endpoint, tokens = task.result()
                    if tokens:
                        candidates.append((endpoint, tokens))
                        if endpoint.backend == "llamacpp" and len(tokens) > 1:
                            has_llamacpp = True
                except Exception as exc:
                    logger.warning("Drafter failed: %s", exc)

            # If we have a llamacpp result, give others 0.3s to finish
            if has_llamacpp and pending:
                done2, pending = await asyncio.wait(pending, timeout=0.3)
                for task in done2:
                    try:
                        endpoint, tokens = task.result()
                        if tokens:
                            candidates.append((endpoint, tokens))
                    except Exception as exc:
                        logger.warning("Drafter failed: %s", exc)
                # Cancel stragglers
                for task in pending:
                    task.cancel()
                break
            elif candidates:
                # Got a non-llamacpp result, give others 1s to provide a better one
                done2, pending = await asyncio.wait(pending, timeout=1.0)
                for task in done2:
                    try:
                        endpoint, tokens = task.result()
                        if tokens:
                            candidates.append((endpoint, tokens))
                    except Exception as exc:
                        logger.warning("Drafter failed: %s", exc)
                for task in pending:
                    task.cancel()
                break

        if not candidates:
            logger.warning("All drafters failed, no candidates")
            return []

        # Pick best: prefer candidates with more individual tokens (llamacpp
        # gives per-token logprobs, enabling batch verification). Among those,
        # pick longest total text. Ties broken by highest mean logprob.
        # Ollama returns 1 big text blob — deprioritize it since it forces
        # text-match fallback.
        def _score(pair):
            ep, toks = pair
            n_tokens = len(toks)
            has_logprobs = ep.backend == "llamacpp" and n_tokens > 1
            total_text = sum(len(t.text) for t in toks)
            mean_lp = sum(t.logprob for t in toks) / n_tokens if toks else -999.0
            return (has_logprobs, n_tokens, total_text, mean_lp)

        winner_endpoint, winner_tokens = max(candidates, key=_score)
        self._winning_drafter = winner_endpoint
        self.stats.drafter_wins[winner_endpoint.url] = (
            self.stats.drafter_wins.get(winner_endpoint.url, 0) + 1
        )
        logger.info(
            "Drafter winner: %s (%s) — %d tokens, %d candidates",
            winner_endpoint.model_name, winner_endpoint.url,
            len(winner_tokens), len(candidates),
        )
        return winner_tokens

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
            data = resp.json()
            # Qwen3 thinking mode puts output in "thinking" when "response" is empty
            return data.get("response", "") or data.get("thinking", "")
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

    async def verify_with_logprobs(
        self, prompt: str, draft_tokens: list[DraftToken], temperature: float = 0.0
    ) -> VerificationResult:
        """Verify draft tokens via prompt-append batch verification.

        Appends draft text to the prompt so the target processes draft tokens
        as prompt (parallel evaluation) rather than generating them one-by-one.
        KV cache means only new draft tokens are evaluated each round.

        Generates N+1 tokens with logprobs from the base prompt to compare
        token IDs. Falls back to prompt-append mode when the target supports
        fast prompt evaluation.

        Returns VerificationResult with accepted/rejected tokens.
        """
        n_draft = len(draft_tokens)
        draft_text = "".join(t.text for t in draft_tokens)

        # Prompt-append verification: feed draft text as prompt, generate 1 token.
        # Draft tokens are processed in parallel during prompt evaluation (~2x faster
        # than autoregressive generation). We then compare what the target generates
        # from the base prompt to find the first disagreement.
        #
        # Step 1: Target generates N+1 tokens from base prompt with logprobs
        # to do per-position comparison.
        body: dict = {
            "prompt": prompt + draft_text,
            "max_tokens": 1,
            "temperature": temperature,
            "logprobs": True,
            "stream": False,
        }
        resp = await self.target_client.post("/v1/completions", json=body)
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        bonus_text = choice.get("text", "")
        logprobs_data = choice.get("logprobs", {})
        content = logprobs_data.get("content", [])

        # Now tokenize the draft text using the target's tokenizer to get
        # the token IDs the target would have assigned
        tok_resp = await self.target_client.post("/tokenize", json={"content": draft_text})
        tok_resp.raise_for_status()
        target_token_ids = tok_resp.json().get("tokens", [])

        # Also tokenize using the draft model to compare
        draft_tok_resp = await self.draft_client.post("/tokenize", json={"content": draft_text})
        draft_tok_resp.raise_for_status()
        draft_token_ids = draft_tok_resp.json().get("tokens", [])

        # Same-family models share tokenizers, so tokenization should match.
        # If tokens match, the draft text is valid for the target.
        # Count matching token IDs from the start.
        n_match = 0
        for i in range(min(len(target_token_ids), len(draft_token_ids))):
            if target_token_ids[i] == draft_token_ids[i]:
                n_match += 1
            else:
                break

        # If tokenization matches completely, we accept all draft tokens
        if n_match == len(draft_token_ids) and n_match == len(target_token_ids):
            # All draft tokens accepted — bonus token is what target generates next
            bonus_token = None
            if content:
                entry = content[0]
                bonus_token = DraftToken(
                    token_id=entry.get("id", 0),
                    logprob=entry.get("logprob", 0.0),
                    text=entry.get("token", bonus_text),
                )

            return VerificationResult(
                accepted_tokens=list(draft_tokens),
                bonus_token=bonus_token,
                accepted_count=n_draft,
                rejected_at=None,
                resample_token=None,
            )
        else:
            # Tokenization diverged — fall back to generating from base prompt
            # to find the exact rejection point
            fallback_body: dict = {
                "prompt": prompt,
                "max_tokens": n_draft + 1,
                "temperature": temperature,
                "logprobs": True,
                "stream": False,
            }
            resp2 = await self.target_client.post("/v1/completions", json=fallback_body)
            resp2.raise_for_status()
            data2 = resp2.json()

            choice2 = data2["choices"][0]
            content2 = choice2.get("logprobs", {}).get("content", [])

            if not content2:
                return VerificationResult(
                    accepted_tokens=[], bonus_token=None, accepted_count=0, rejected_at=0,
                    resample_token=DraftToken(token_id=0, logprob=0.0, text=""),
                )

            target_logprobs: list[TargetLogprob] = []
            for i, entry in enumerate(content2):
                top_token_id = entry.get("id", 0)
                top_logprob = entry.get("logprob", -100.0)
                top_token_text = entry.get("token", "")

                draft_token_lp = None
                if i < n_draft:
                    if draft_tokens[i].token_id == top_token_id:
                        draft_token_lp = top_logprob
                    else:
                        for alt in entry.get("top_logprobs", []):
                            if alt.get("id") == draft_tokens[i].token_id:
                                draft_token_lp = alt.get("logprob")
                                break

                target_logprobs.append(TargetLogprob(
                    token_id=top_token_id,
                    logprob=top_logprob,
                    draft_token_logprob=draft_token_lp,
                ))
                target_logprobs[-1]._text = top_token_text  # type: ignore[attr-defined]

            result = verify_draft_tokens(draft_tokens, target_logprobs, temperature)

            if result.resample_token is not None and result.rejected_at is not None:
                idx = result.rejected_at
                if idx < len(target_logprobs):
                    result.resample_token.text = getattr(target_logprobs[idx], '_text', '')
            if result.bonus_token is not None:
                idx = n_draft
                if idx < len(target_logprobs):
                    result.bonus_token.text = getattr(target_logprobs[idx], '_text', '')

            return result

    async def verify_text_match(
        self, prompt: str, draft_text: str, temperature: float = 0.0
    ) -> tuple[str, int, int]:
        """Fallback: text-match verification for backends without logprobs.

        Returns (target_text, n_accepted_chars, n_draft_chars).
        """
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

        return target_text, match_len, len(draft_text)

    def _can_use_logprobs(self) -> bool:
        """Check if we can use logprobs-based batch verification."""
        # Need llamacpp for both: draft must provide per-token IDs,
        # target must support logprobs in /v1/completions.
        # Ollama returns text blobs without token IDs, so text-match is used.
        if self.config.target.backend != "llamacpp":
            return False
        if self._multi_drafter:
            # In multi-drafter mode, check the winning drafter's backend
            return (
                self._winning_drafter is not None
                and self._winning_drafter.backend == "llamacpp"
            )
        return self.config.draft.backend == "llamacpp"

    async def speculation_round(
        self, prompt: str, temperature: float = 0.0
    ) -> tuple[str, bool, float, float]:
        """One full draft → verify → accept cycle.

        Returns (accepted_text, is_done, draft_ms, verify_ms).
        is_done is True if we hit EOS or empty generation.
        """
        # Draft phase
        t0 = time.monotonic()
        try:
            if self._multi_drafter:
                draft = await self.draft_tokens_parallel(
                    prompt, self.config.max_draft_tokens, temperature
                )
            else:
                draft = await self.draft_tokens(
                    prompt, self.config.max_draft_tokens, temperature
                )
        except Exception as exc:
            if self.config.fallback_on_draft_failure:
                logger.warning("Draft failed (%s: %s), falling back to target", type(exc).__name__, exc)
                text, done = await self._fallback_generate(
                    prompt, self.config.max_draft_tokens, temperature
                )
                return text, done, 0.0, 0.0
            raise
        draft_ms = (time.monotonic() - t0) * 1000

        if not draft:
            return "", True, draft_ms, 0.0

        draft_text = "".join(t.text for t in draft)
        if not draft_text:
            return "", True, draft_ms, 0.0

        # Verify phase
        t1 = time.monotonic()
        if self._can_use_logprobs():
            # Logprobs batch verification — single forward pass on target
            result = await self.verify_with_logprobs(prompt, draft, temperature)
            verify_ms = (time.monotonic() - t1) * 1000
            accepted_text = "".join(t.text for t in result.accepted_tokens)
            if result.resample_token and result.resample_token.text:
                accepted_text += result.resample_token.text
            elif result.bonus_token and result.bonus_token.text:
                accepted_text += result.bonus_token.text

            n_accepted = result.accepted_count
            n_drafted = len(draft)

            self.stats.total_rounds += 1
            self.stats.total_drafted += n_drafted
            self.stats.total_accepted += n_accepted
            self.stats.total_tokens_output += result.total_tokens

            if result.rejected_at is not None:
                self.stats.total_resampled += 1
            elif result.bonus_token is not None:
                self.stats.total_bonus += 1

            is_done = not accepted_text
            return accepted_text, is_done, draft_ms, verify_ms
        else:
            # Text-match fallback (Ollama or mixed backends)
            target_text, match_len, draft_len = await self.verify_text_match(
                prompt, draft_text, temperature
            )
            verify_ms = (time.monotonic() - t1) * 1000

            self.stats.total_rounds += 1
            self.stats.total_drafted += draft_len
            self.stats.total_accepted += match_len
            self.stats.total_tokens_output += len(target_text)

            if match_len < draft_len:
                self.stats.total_resampled += 1

            is_done = len(target_text) == 0
            return target_text, is_done, draft_ms, verify_ms

    async def _fallback_generate(
        self, prompt: str, n: int, temperature: float
    ) -> tuple[str, bool]:
        """Direct generation from target (no speculation)."""
        text = await self._generate_target(prompt, n, temperature)
        self.stats.total_tokens_output += len(text)
        return text, not text

    def _record_request(
        self, rounds: int, drafted: int, accepted: int,
        draft_ms: float, verify_ms: float, total_ms: float,
        tokens_output: int,
    ):
        """Append a RequestRecord to history (ring buffer, max 50)."""
        rate = accepted / drafted if drafted > 0 else 0.0
        record = RequestRecord(
            timestamp=time.time(),
            rounds=rounds,
            drafted=drafted,
            accepted=accepted,
            acceptance_rate=round(rate, 3),
            draft_ms=round(draft_ms, 1),
            verify_ms=round(verify_ms, 1),
            total_ms=round(total_ms, 1),
            tokens_output=tokens_output,
            model=self.config.target.model_name,
        )
        self.stats.request_history.append(record)
        if len(self.stats.request_history) > MAX_REQUEST_HISTORY:
            self.stats.request_history.pop(0)

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
                req_rounds = 0
                req_drafted = 0
                req_accepted = 0
                req_draft_ms = 0.0
                req_verify_ms = 0.0
                req_t0 = time.monotonic()
                stats_before_drafted = self.stats.total_drafted
                stats_before_accepted = self.stats.total_accepted

                while tokens_generated < max_tokens:
                    chunk, done, d_ms, v_ms = await self.speculation_round(
                        prompt + generated, temperature
                    )
                    req_rounds += 1
                    req_draft_ms += d_ms
                    req_verify_ms += v_ms

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

                req_drafted = self.stats.total_drafted - stats_before_drafted
                req_accepted = self.stats.total_accepted - stats_before_accepted
                req_total_ms = (time.monotonic() - req_t0) * 1000
                self._record_request(
                    req_rounds, req_drafted, req_accepted,
                    req_draft_ms, req_verify_ms, req_total_ms,
                    tokens_generated,
                )

            return sse_stream()
        else:
            req_rounds = 0
            req_draft_ms = 0.0
            req_verify_ms = 0.0
            req_t0 = time.monotonic()
            stats_before_drafted = self.stats.total_drafted
            stats_before_accepted = self.stats.total_accepted

            while tokens_generated < max_tokens:
                chunk, done, d_ms, v_ms = await self.speculation_round(
                    prompt + generated, temperature
                )
                req_rounds += 1
                req_draft_ms += d_ms
                req_verify_ms += v_ms

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

            req_drafted = self.stats.total_drafted - stats_before_drafted
            req_accepted = self.stats.total_accepted - stats_before_accepted
            req_total_ms = (time.monotonic() - req_t0) * 1000
            self._record_request(
                req_rounds, req_drafted, req_accepted,
                req_draft_ms, req_verify_ms, req_total_ms,
                tokens_generated,
            )

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
    system = "You are a helpful assistant. /no_think"
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

# Module-level singleton for the active proxy instance.
#
# Design note (CQ-3): This global is an intentional trade-off for simplicity —
# the proxy is a heavyweight singleton (HTTP client pools, stats) that is only
# ever created once per process by ``create_app()``.  A future refactor could
# pass state through Starlette's ``request.state`` or a dependency-injection
# container to support multiple instances per process.  For now:
#   - ``reset_proxy_state()`` provides a reliable teardown path for tests.
#   - ``create_app()`` raises ``RuntimeError`` if called a second time without
#     resetting first to surface accidental re-initialization early.
_proxy: SpeculativeProxy | None = None


def reset_proxy_state() -> None:
    """Reset the module-level proxy singleton to ``None``.

    Call this in test teardown or before calling ``create_app()`` a second
    time to prevent stale state from a previous call leaking into subsequent
    requests.
    """
    global _proxy
    _proxy = None


def _get_proxy() -> SpeculativeProxy:
    """Return the active proxy, raising ``RuntimeError`` if not initialized.

    Unlike ``assert``, this check is never stripped by the ``-O`` flag.
    """
    if _proxy is None:
        raise RuntimeError("Proxy not initialized")
    return _proxy


async def _check_body_size(request: Request, max_body_size: int) -> Response | None:
    """Return a 413 Response if Content-Length exceeds *max_body_size*, else None.

    Checks the ``Content-Length`` header eagerly so that oversized payloads
    are rejected before the body is buffered in memory, preventing memory-
    exhaustion DoS (audit ref: CQ-5 / issue #8).

    When ``Content-Length`` is absent the check is skipped — the body will
    be read normally and Starlette's own limits apply.
    """
    content_length_raw = request.headers.get("content-length")
    if content_length_raw is not None:
        try:
            content_length = int(content_length_raw)
        except ValueError:
            return JSONResponse(
                {"detail": "Invalid Content-Length header"},
                status_code=400,
            )
        if content_length > max_body_size:
            return JSONResponse(
                {
                    "detail": (
                        f"Request body too large: {content_length} bytes "
                        f"(limit is {max_body_size} bytes)"
                    )
                },
                status_code=413,
            )
    return None


async def handle_completion(request: Request):
    proxy = _get_proxy()

    # --- Body size check (CQ-5) ---
    size_error = await _check_body_size(request, proxy.config.max_body_size)
    if size_error is not None:
        return size_error

    # --- Parse & validate JSON body (CQ-1) ---
    try:
        raw_body = await request.json()
    except Exception:
        return JSONResponse(
            {"detail": "Request body must be valid JSON with Content-Type: application/json"},
            status_code=400,
        )

    try:
        req = parse_completion_request(
            raw_body,
            max_tokens_limit=proxy.config.max_tokens_limit,
        )
    except ValidationError as exc:
        return JSONResponse(exc.to_dict(), status_code=400)

    result = await proxy.generate_completion(
        prompt=req.prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        stream=req.stream,
        stop=req.stop,
    )

    if req.stream:
        return StreamingResponse(
            result,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return JSONResponse(result)


async def handle_chat_completion(request: Request):
    proxy = _get_proxy()

    # --- Body size check (CQ-5) ---
    size_error = await _check_body_size(request, proxy.config.max_body_size)
    if size_error is not None:
        return size_error

    # --- Parse & validate JSON body (CQ-1) ---
    try:
        raw_body = await request.json()
    except Exception:
        return JSONResponse(
            {"detail": "Request body must be valid JSON with Content-Type: application/json"},
            status_code=400,
        )

    try:
        req = parse_chat_completion_request(
            raw_body,
            max_tokens_limit=proxy.config.max_tokens_limit,
        )
    except ValidationError as exc:
        return JSONResponse(exc.to_dict(), status_code=400)

    # Convert validated ChatMessage objects to dicts for apply_chat_template
    messages_dicts = [{"role": m.role, "content": m.content} for m in req.messages]
    prompt = apply_chat_template(messages_dicts)

    # Add Qwen3 stop tokens
    chat_stop = list(req.stop) if req.stop else []
    chat_stop.append("<|im_end|>")

    result = await proxy.generate_completion(
        prompt=prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        stream=req.stream,
        stop=chat_stop,
    )

    if req.stream:
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
    target_health = await proxy.check_server(proxy.config.target.url, proxy.config.target.backend)

    stats = proxy.stats
    response: dict = {
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
    }

    if proxy._multi_drafter:
        drafters_info = []
        for endpoint, _ in proxy.draft_clients:
            health = await proxy.check_server(endpoint.url, endpoint.backend)
            drafters_info.append({
                "url": endpoint.url,
                "model": endpoint.model_name,
                "backend": endpoint.backend,
                "health": health,
                "wins": stats.drafter_wins.get(endpoint.url, 0),
            })
        response["drafters"] = drafters_info
    else:
        draft_health = await proxy.check_server(proxy.config.draft.url, proxy.config.draft.backend)
        response["draft"] = {
            "url": proxy.config.draft.url,
            "model": proxy.config.draft.model_name,
            "health": draft_health,
        }

    return JSONResponse(response)


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


CHAT_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Tightwad Chat</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,system-ui,sans-serif;background:#1a1a2e;color:#e0e0e0;height:100vh;display:flex;flex-direction:column}
#header{padding:12px 20px;background:#16213e;border-bottom:1px solid #0f3460;display:flex;align-items:center;gap:12px}
#header h1{font-size:18px;color:#e94560}
#header span{font-size:13px;color:#888}
#status{font-size:12px;color:#4ecca3;margin-left:auto;cursor:pointer}
#messages{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:16px}
.msg{max-width:80%;padding:12px 16px;border-radius:12px;line-height:1.5;white-space:pre-wrap;word-wrap:break-word;font-size:15px}
.user{align-self:flex-end;background:#0f3460;color:#fff;border-bottom-right-radius:4px}
.ai{align-self:flex-start;background:#16213e;border:1px solid #0f3460;border-bottom-left-radius:4px}
.ai.streaming{border-color:#e94560}
#input-area{padding:16px 20px;background:#16213e;border-top:1px solid #0f3460;display:flex;gap:10px}
#input{flex:1;padding:12px 16px;border-radius:8px;border:1px solid #0f3460;background:#1a1a2e;color:#fff;font-size:15px;outline:none;font-family:inherit}
#input:focus{border-color:#e94560}
#send{padding:12px 24px;border-radius:8px;border:none;background:#e94560;color:#fff;font-size:15px;cursor:pointer;font-weight:600}
#send:hover{background:#c73e54}
#send:disabled{background:#555;cursor:not-allowed}
</style></head><body>
<div id="header"><h1>Tightwad</h1><span>Speculative Decoding Chat</span><span id="status" onclick="checkStatus()">checking...</span></div>
<div id="messages"></div>
<div id="input-area">
<input id="input" placeholder="Type a message..." autofocus>
<button id="send" onclick="send()">Send</button>
</div>
<script>
const msgs=document.getElementById('messages'),input=document.getElementById('input'),btn=document.getElementById('send');
let messages=[], busy=false;

input.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey&&!busy){e.preventDefault();send()}});

async function checkStatus(){
  const el=document.getElementById('status');
  try{
    const r=await fetch('/v1/tightwad/status');const d=await r.json();
    const n=d.drafters?d.drafters.length:1;
    const t=d.target?.model||'?';
    el.textContent=n+' drafter'+(n>1?'s':'')+' > '+t;
    el.style.color='#4ecca3';
  }catch{el.textContent='error';el.style.color='#e94560'}
}

function addMsg(role,text){
  const div=document.createElement('div');
  div.className='msg '+(role==='user'?'user':'ai');
  div.textContent=text;
  msgs.appendChild(div);
  msgs.scrollTop=msgs.scrollHeight;
  return div;
}

async function send(){
  const text=input.value.trim();if(!text||busy)return;
  busy=true;btn.disabled=true;input.value='';
  messages.push({role:'user',content:text});
  addMsg('user',text);
  const aiDiv=addMsg('ai','');
  aiDiv.classList.add('streaming');
  let full='';
  try{
    const r=await fetch('/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({messages:messages,max_tokens:1024,temperature:0,stream:true})});
    const reader=r.body.getReader();const dec=new TextDecoder();let buf='';
    while(true){
      const{done,value}=await reader.read();if(done)break;
      buf+=dec.decode(value,{stream:true});
      const lines=buf.split('\\n');buf=lines.pop();
      for(const line of lines){
        if(!line.startsWith('data: '))continue;
        const payload=line.slice(6);if(payload==='[DONE]')continue;
        try{
          const d=JSON.parse(payload);
          const c=d.choices?.[0];
          const chunk=c?.text||c?.delta?.content||'';
          if(chunk){full+=chunk;aiDiv.textContent=full;msgs.scrollTop=msgs.scrollHeight}
        }catch{}
      }
    }
  }catch(e){full='Error: '+e.message;aiDiv.textContent=full}
  aiDiv.classList.remove('streaming');
  messages.push({role:'assistant',content:full});
  busy=false;btn.disabled=false;input.focus();
}

checkStatus();
</script></body></html>"""


async def handle_chat_ui(request: Request):
    return HTMLResponse(CHAT_HTML)


def create_app(config: ProxyConfig) -> Starlette:
    """Create and configure the proxy ASGI application.

    .. note::
        This function sets the module-level ``_proxy`` singleton.  Only one
        proxy instance per process is supported.  If ``create_app()`` is
        called while a proxy already exists (i.e. the singleton is not
        ``None``), a warning is logged and the previous instance is replaced.
        Call :func:`reset_proxy_state` between calls to make re-initialization
        explicit and avoid accidental stale-state bugs.
    """
    global _proxy
    if _proxy is not None:
        logger.warning(
            "create_app() called while a proxy instance already exists; "
            "replacing the previous singleton.  Call reset_proxy_state() "
            "before re-initializing to make this explicit."
        )
    _proxy = SpeculativeProxy(config)

    from .dashboard import handle_dashboard, handle_events, handle_history

    @asynccontextmanager
    async def lifespan(app):
        # Security check: warn loudly if the proxy is unauthenticated.
        if not config.auth_token:
            logger.warning(
                "⚠️  SECURITY WARNING: No auth token configured. "
                "The proxy API is open to anyone who can reach %s:%s. "
                "Set TIGHTWAD_PROXY_TOKEN or add auth_token to cluster.yaml "
                "to require Bearer-token authentication.",
                config.host,
                config.port,
            )
        else:
            logger.info(
                "🔐 Proxy auth enabled — Bearer token required for all API requests."
            )
        yield
        if _proxy:
            await _proxy.close()

    app = Starlette(
        routes=[
            Route("/", handle_chat_ui, methods=["GET"]),
            Route("/dashboard", handle_dashboard, methods=["GET"]),
            Route("/v1/completions", handle_completion, methods=["POST"]),
            Route("/v1/chat/completions", handle_chat_completion, methods=["POST"]),
            Route("/v1/models", handle_models, methods=["GET"]),
            Route("/v1/tightwad/status", handle_status, methods=["GET"]),
            Route("/v1/tightwad/events", handle_events, methods=["GET"]),
            Route("/v1/tightwad/history", handle_history, methods=["GET"]),
        ],
        middleware=[
            # TokenAuthMiddleware is a no-op when auth_token is None,
            # preserving backward compatibility for unauthenticated deployments.
            Middleware(TokenAuthMiddleware, token=config.auth_token),
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
