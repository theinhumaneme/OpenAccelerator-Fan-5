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


def make_client(base_url: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key="placeholder")


def run_request(
    client: OpenAI,
    model: str,
    prompt: str,
    category: str,
    max_tokens: int = 256,
    target: str = "",
) -> RequestResult:
    start = time.perf_counter()
    first_token_t: float | None = None
    output_parts: list[str] = []
    token_logprobs: list[float] = []

    stream = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        logprobs=1,
        stream=True,
    )

    for chunk in stream:
        choice = chunk.choices[0]
        if choice.text:
            if first_token_t is None:
                first_token_t = time.perf_counter()
            output_parts.append(choice.text)
        if choice.logprobs and choice.logprobs.token_logprobs:
            token_logprobs.extend(
                lp for lp in choice.logprobs.token_logprobs if lp is not None
            )

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
    )
