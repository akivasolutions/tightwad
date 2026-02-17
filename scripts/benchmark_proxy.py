#!/usr/bin/env python3
"""Benchmark the speculative decoding proxy against different configurations.

Runs prompts through spec decoding and target-only, measures acceptance rate
and speedup. Outputs results as JSON for inclusion in docs.
"""

import asyncio
import json
import time

import httpx

# Test prompts — mix of factual, creative, and code
PROMPTS = [
    {
        "name": "factual",
        "messages": [{"role": "user", "content": "What are the three laws of thermodynamics? Explain each in one sentence."}],
    },
    {
        "name": "code",
        "messages": [{"role": "user", "content": "Write a Python function that checks if a string is a palindrome. Include a docstring."}],
    },
    {
        "name": "creative",
        "messages": [{"role": "user", "content": "Write a haiku about GPU inference."}],
    },
    {
        "name": "reasoning",
        "messages": [{"role": "user", "content": "If a train travels 120 miles in 2 hours, and then 180 miles in 3 hours, what is its average speed for the entire trip?"}],
    },
    {
        "name": "list",
        "messages": [{"role": "user", "content": "List 5 practical uses for speculative decoding in production AI systems."}],
    },
]


def ollama_extract_text(data: dict) -> str:
    """Extract text from Ollama response, handling thinking mode."""
    return data.get("response", "") or data.get("thinking", "")


async def test_target_only(target_url: str, target_model: str, backend: str,
                           prompt: dict, max_tokens: int = 256) -> dict:
    """Generate directly from target (baseline)."""
    start = time.monotonic()

    if backend == "ollama":
        body = {
            "model": target_model,
            "prompt": f"<|im_start|>system\nYou are a helpful assistant. /no_think<|im_end|>\n"
                      f"<|im_start|>user\n{prompt['messages'][0]['content']}<|im_end|>\n"
                      f"<|im_start|>assistant\n",
            "raw": True,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0},
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{target_url}/api/generate", json=body)
        resp.raise_for_status()
        data = resp.json()
        text = ollama_extract_text(data)
        # Clean stop token
        if "<|im_end|>" in text:
            text = text[:text.index("<|im_end|>")]
        eval_count = data.get("eval_count", len(text.split()))
        eval_duration = data.get("eval_duration", 0) / 1e9
        tok_s = eval_count / eval_duration if eval_duration > 0 else 0
    else:
        body = {
            "model": target_model,
            "messages": prompt["messages"],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{target_url}/v1/chat/completions", json=body)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        eval_count = data.get("usage", {}).get("completion_tokens", len(text.split()))
        tok_s = 0

    elapsed = time.monotonic() - start

    return {
        "text": text.strip(),
        "tokens": eval_count,
        "elapsed_s": round(elapsed, 2),
        "tok_s": round(tok_s, 1) if tok_s else round(eval_count / elapsed, 1),
    }


async def test_spec_decoding(draft_url: str, draft_model: str, draft_backend: str,
                              target_url: str, target_model: str, target_backend: str,
                              prompt: dict, max_tokens: int = 256,
                              max_draft_tokens: int = 8) -> dict:
    """Run speculative decoding manually (draft -> verify -> accept loop)."""
    user_content = prompt["messages"][0]["content"]
    # Use /no_think to disable Qwen3 thinking mode for clean text output
    base_prompt = (
        f"<|im_start|>system\nYou are a helpful assistant. /no_think<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    generated = ""
    total_rounds = 0
    total_drafted_chars = 0
    total_accepted_chars = 0
    start = time.monotonic()

    async with httpx.AsyncClient(timeout=30.0) as draft_client, \
               httpx.AsyncClient(timeout=120.0) as target_client:

        for _ in range(50):  # max rounds safety
            current_prompt = base_prompt + generated

            # Draft phase
            if draft_backend == "ollama":
                draft_body = {
                    "model": draft_model,
                    "prompt": current_prompt,
                    "raw": True,
                    "stream": False,
                    "options": {"num_predict": max_draft_tokens, "temperature": 0},
                }
                resp = await draft_client.post(f"{draft_url}/api/generate", json=draft_body)
                resp.raise_for_status()
                draft_text = ollama_extract_text(resp.json())
            else:
                draft_body = {
                    "prompt": current_prompt,
                    "max_tokens": max_draft_tokens,
                    "temperature": 0,
                    "stream": False,
                }
                resp = await draft_client.post(f"{draft_url}/v1/completions", json=draft_body)
                resp.raise_for_status()
                draft_text = resp.json()["choices"][0].get("text", "")

            if not draft_text:
                break

            # Check if draft hit EOS
            draft_done = False
            if "<|im_end|>" in draft_text:
                draft_text = draft_text[:draft_text.index("<|im_end|>")]
                draft_done = True
            if not draft_text:
                break

            # Verify phase — target generates from same prompt
            if target_backend == "ollama":
                target_body = {
                    "model": target_model,
                    "prompt": current_prompt,
                    "raw": True,
                    "stream": False,
                    "options": {"num_predict": max_draft_tokens, "temperature": 0},
                }
                resp = await target_client.post(f"{target_url}/api/generate", json=target_body)
                resp.raise_for_status()
                target_text = ollama_extract_text(resp.json())
            else:
                target_body = {
                    "prompt": current_prompt,
                    "max_tokens": max_draft_tokens,
                    "temperature": 0,
                    "stream": False,
                }
                resp = await target_client.post(f"{target_url}/v1/completions", json=target_body)
                resp.raise_for_status()
                target_text = resp.json()["choices"][0].get("text", "")

            # Check target EOS
            target_done = False
            if "<|im_end|>" in target_text:
                target_text = target_text[:target_text.index("<|im_end|>")]
                target_done = True

            # Text-match greedy verification
            match_len = 0
            for i in range(min(len(draft_text), len(target_text))):
                if draft_text[i] == target_text[i]:
                    match_len = i + 1
                else:
                    break

            total_rounds += 1
            total_drafted_chars += len(draft_text)
            total_accepted_chars += match_len

            # Accept target's output (always correct)
            generated += target_text
            word_count = len(generated.split())

            if target_done or draft_done or word_count >= max_tokens or not target_text:
                break

    elapsed = time.monotonic() - start
    acceptance_rate = total_accepted_chars / total_drafted_chars if total_drafted_chars > 0 else 0
    word_count = len(generated.split())

    return {
        "text": generated.strip(),
        "rounds": total_rounds,
        "total_drafted_chars": total_drafted_chars,
        "total_accepted_chars": total_accepted_chars,
        "acceptance_rate": round(acceptance_rate, 3),
        "elapsed_s": round(elapsed, 2),
        "word_count": word_count,
        "tok_s_approx": round(word_count / elapsed, 1) if elapsed > 0 else 0,
    }


async def run_config(name: str, draft_url: str, draft_model: str, draft_backend: str,
                     target_url: str, target_model: str, target_backend: str) -> dict:
    """Run all prompts against one configuration."""
    print(f"\n{'='*70}")
    print(f"Config: {name}")
    print(f"  Draft:  {draft_model} @ {draft_url} ({draft_backend})")
    print(f"  Target: {target_model} @ {target_url} ({target_backend})")
    print(f"{'='*70}")

    results = []
    for prompt in PROMPTS:
        print(f"\n  [{prompt['name']}] ", end="", flush=True)

        # Baseline: target only
        print("baseline...", end="", flush=True)
        try:
            baseline = await test_target_only(target_url, target_model, target_backend, prompt)
            print(f" {baseline['elapsed_s']}s ({baseline['tok_s']} tok/s)", end="", flush=True)
        except Exception as e:
            print(f" FAILED: {e}")
            baseline = {"error": str(e)}
            results.append({"prompt": prompt["name"], "baseline": baseline, "speculative": {"error": "skipped"}})
            continue

        # Speculative decoding
        print(" | spec...", end="", flush=True)
        try:
            spec = await test_spec_decoding(
                draft_url, draft_model, draft_backend,
                target_url, target_model, target_backend,
                prompt,
            )
            print(f" {spec['elapsed_s']}s ({spec['acceptance_rate']*100:.0f}% accept, {spec['rounds']} rounds)")
        except Exception as e:
            print(f" FAILED: {e}")
            spec = {"error": str(e)}

        speedup = round(baseline["elapsed_s"] / spec["elapsed_s"], 2) if "elapsed_s" in spec and spec["elapsed_s"] > 0 else None

        results.append({
            "prompt": prompt["name"],
            "baseline": baseline,
            "speculative": spec,
            "speedup": speedup,
        })

    # Aggregate stats
    spec_ok = [r for r in results if "error" not in r.get("speculative", {})]
    if spec_ok:
        avg_acceptance = sum(r["speculative"]["acceptance_rate"] for r in spec_ok) / len(spec_ok)
        avg_speedup = sum(r["speedup"] for r in spec_ok if r["speedup"]) / len(spec_ok)
        avg_rounds = sum(r["speculative"]["rounds"] for r in spec_ok) / len(spec_ok)
    else:
        avg_acceptance = avg_speedup = avg_rounds = 0

    summary = {
        "config": name,
        "draft": f"{draft_model} ({draft_backend})",
        "target": f"{target_model} ({target_backend})",
        "avg_acceptance_rate": round(avg_acceptance, 3),
        "avg_speedup": round(avg_speedup, 2),
        "avg_rounds": round(avg_rounds, 1),
        "prompts": results,
    }

    print(f"\n  Summary: {avg_acceptance*100:.1f}% acceptance, {avg_speedup:.2f}x speedup, {avg_rounds:.1f} avg rounds")
    return summary


async def main():
    configs = [
        # Config 1: Same family — Qwen3-8B → Qwen3-32B
        {
            "name": "Qwen3-8B → Qwen3-32B (same family, Ollama)",
            "draft_url": "http://192.168.1.101:11434",
            "draft_model": "qwen3:8b",
            "draft_backend": "ollama",
            "target_url": "http://192.168.1.100:11434",
            "target_model": "qwen3:32b",
            "target_backend": "ollama",
        },
    ]

    all_results = []
    for config in configs:
        try:
            result = await run_config(**config)
            all_results.append(result)
        except Exception as e:
            print(f"\n  CONFIG FAILED: {e}")
            all_results.append({"config": config["name"], "error": str(e)})

    # Write results
    output_path = "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults written to {output_path}")

    # Print summary table
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"{'Config':<50} {'Accept%':>8} {'Speedup':>8} {'Rounds':>7}")
    print("-"*80)
    for r in all_results:
        if "error" not in r:
            print(f"{r['config']:<50} {r['avg_acceptance_rate']*100:>7.1f}% {r['avg_speedup']:>7.2f}x {r['avg_rounds']:>7.1f}")
        else:
            print(f"{r['config']:<50} {'FAILED':>8}")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
