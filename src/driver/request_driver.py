from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from config_manager import RunConfig

logger = logging.getLogger(__name__)


@dataclass
class PromptRecord:
    prompt: str
    category: str = "unknown"


@dataclass
class RequestResult:
    prompt_index: int
    prompt: str
    category: str
    output: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    ttft_ms: float
    wall_clock_seconds: float
    tokens_per_second: float
    temperature: float
    finish_reason: str
    token_logprobs: list[float] = field(default_factory=list)
    error: str | None = None


def send_workload(base_url: str, run: RunConfig) -> list[RequestResult]:
    client = OpenAI(base_url=base_url, api_key="placeholder", timeout=300.0)
    prompts = _load_prompts(run.workload_path)
    results: list[RequestResult] = []

    for idx, record in enumerate(tqdm(prompts, desc=run.run_id, ncols=90)):
        result = _send_one(client, run, record, idx)
        results.append(result)
        if result.error:
            logger.error("Request %s failed: %s", idx, result.error)

    return results


def _send_one(
    client: OpenAI,
    run: RunConfig,
    record: PromptRecord,
    prompt_index: int,
) -> RequestResult:
    start = time.perf_counter()
    first_token_t: float | None = None
    output_parts: list[str] = []
    token_logprobs: list[float] = []
    finish_reason = "unknown"

    try:
        stream = client.completions.create(
            model=run.served_model_name,
            prompt=record.prompt,
            max_tokens=run.max_output_tokens,
            temperature=run.temperature,
            logprobs=1,
            stream=True,
        )

        for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.text:
                if first_token_t is None:
                    first_token_t = time.perf_counter()
                output_parts.append(choice.text)
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            if choice.logprobs and choice.logprobs.token_logprobs:
                token_logprobs.extend(
                    lp for lp in choice.logprobs.token_logprobs if lp is not None
                )
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return RequestResult(
            prompt_index=prompt_index,
            prompt=record.prompt,
            category=record.category,
            output="",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            ttft_ms=elapsed * 1000,
            wall_clock_seconds=elapsed,
            tokens_per_second=0,
            temperature=run.temperature,
            finish_reason="error",
            error=str(exc),
        )

    elapsed = time.perf_counter() - start
    output = "".join(output_parts)
    completion_tokens = len(token_logprobs) or max(1, len(output.split()))
    ttft_ms = ((first_token_t - start) * 1000) if first_token_t else elapsed * 1000

    return RequestResult(
        prompt_index=prompt_index,
        prompt=record.prompt,
        category=record.category,
        output=output,
        prompt_tokens=0,
        completion_tokens=completion_tokens,
        total_tokens=completion_tokens,
        ttft_ms=ttft_ms,
        wall_clock_seconds=elapsed,
        tokens_per_second=completion_tokens / elapsed if elapsed > 0 else 0,
        temperature=run.temperature,
        finish_reason=finish_reason,
        token_logprobs=token_logprobs,
    )


def _load_prompts(path: str) -> list[PromptRecord]:
    records: list[PromptRecord] = []
    with Path(path).open() as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt = obj.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(f"{path}:{line_number} must contain a non-empty 'prompt'")
            category = obj.get("category", "unknown")
            records.append(PromptRecord(prompt=prompt, category=str(category)))
    return records
