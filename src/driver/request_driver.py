from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from config_manager import RunConfig
from driver.lighteval_loader import load_lighteval_tasks

logger = logging.getLogger(__name__)


@dataclass
class PromptRecord:
    prompt: str
    category: str = "unknown"
    target: str = ""
    choices: list[str] = field(default_factory=list)
    task_name: str = ""
    instruction_ids: list[str] = field(default_factory=list)
    instruction_kwargs: list[dict] = field(default_factory=list)


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
    target: str = ""
    choices: list[str] = field(default_factory=list)
    task_name: str = ""
    instruction_ids: list[str] = field(default_factory=list)
    instruction_kwargs: list[dict] = field(default_factory=list)
    error: str | None = None


def send_workload(base_url: str, run: RunConfig) -> list[RequestResult]:
    client = OpenAI(base_url=base_url, api_key="placeholder", timeout=300.0)
    prompts = _load_prompts(run)
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
            target=record.target,
            choices=record.choices,
            task_name=record.task_name,
            instruction_ids=record.instruction_ids,
            instruction_kwargs=record.instruction_kwargs,
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
        target=record.target,
        choices=record.choices,
        task_name=record.task_name,
        instruction_ids=record.instruction_ids,
        instruction_kwargs=record.instruction_kwargs,
    )


def _load_prompts(run: RunConfig) -> list[PromptRecord]:
    if run.workload_source == "lighteval":
        return [
            PromptRecord(
                prompt=str(item["prompt"]),
                category=str(item.get("category", "unknown")),
                target=str(item.get("target", "")),
                choices=[str(choice) for choice in item.get("choices", [])],
                task_name=str(item.get("task_name", "")),
                instruction_ids=[str(iid) for iid in item.get("instruction_ids", [])],
                instruction_kwargs=[dict(kwargs) for kwargs in item.get("instruction_kwargs", [])],
            )
            for item in load_lighteval_tasks(
                run.workload_tasks,
                num_prompts_per_task=run.workload_num_prompts_per_task,
            )
        ]

    if run.workload_path is None:
        raise ValueError(f"JSONL workload {run.workload_id} has no workload_path")

    records: list[PromptRecord] = []
    with Path(run.workload_path).open() as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt = obj.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(f"{run.workload_path}:{line_number} must contain a non-empty 'prompt'")
            records.append(
                PromptRecord(
                    prompt=prompt,
                    category=str(obj.get("category", "unknown")),
                    target=str(obj.get("target", "")),
                    choices=[str(choice) for choice in obj.get("choices", [])],
                    task_name=str(obj.get("task_name", "")),
                    instruction_ids=[str(iid) for iid in obj.get("instruction_ids", [])],
                    instruction_kwargs=[dict(kwargs) for kwargs in obj.get("instruction_kwargs", [])],
                )
            )
    return records
