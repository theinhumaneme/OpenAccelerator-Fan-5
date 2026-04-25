import time
from dataclasses import dataclass, field
from openai import OpenAI


@dataclass
class RequestResult:
    prompt: str
    category: str
    output: str
    tokens_generated: int
    ttft_ms: float
    total_ms: float
    throughput_tps: float
    token_logprobs: list[float] = field(default_factory=list)
    target: str = ""
    instruction_ids: list[str] = field(default_factory=list)
    instruction_kwargs: list[dict] = field(default_factory=list)


def make_client(base_url: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key="placeholder")


def run_request(
    client: OpenAI,
    model: str,
    prompt: str,
    category: str,
    max_tokens: int = 256,
    target: str = "",
    instruction_ids: list[str] | None = None,
    instruction_kwargs: list[dict] | None = None,
) -> RequestResult:
    start = time.perf_counter()
    first_token_t: float | None = None
    output_parts: list[str] = []
    token_logprobs: list[float] = []

    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        logprobs=True,
        top_logprobs=1,
        stream=True,
    )

    for chunk in stream:
        choice = chunk.choices[0]
        if choice.delta.content:
            if first_token_t is None:
                first_token_t = time.perf_counter()
            output_parts.append(choice.delta.content)
        if choice.logprobs and choice.logprobs.content:
            token_logprobs.extend(item.logprob for item in choice.logprobs.content)

    total_ms = (time.perf_counter() - start) * 1000
    ttft_ms = ((first_token_t - start) * 1000) if first_token_t else total_ms
    output = "".join(output_parts)
    tokens_generated = len(token_logprobs) or max(1, len(output.split()))
    throughput_tps = tokens_generated / (total_ms / 1000)

    return RequestResult(
        prompt=prompt,
        category=category,
        output=output,
        tokens_generated=tokens_generated,
        ttft_ms=ttft_ms,
        total_ms=total_ms,
        throughput_tps=throughput_tps,
        token_logprobs=token_logprobs,
        target=target,
        instruction_ids=instruction_ids or [],
        instruction_kwargs=instruction_kwargs or [],
    )
