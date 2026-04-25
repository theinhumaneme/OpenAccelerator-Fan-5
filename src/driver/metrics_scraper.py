from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

ACCEPTANCE_RATE_METRIC = "vllm:spec_decode_draft_acceptance_rate"
INTERESTING_METRICS = {
    ACCEPTANCE_RATE_METRIC,
    "vllm:spec_decode_num_accepted_tokens",
    "vllm:spec_decode_num_draft_tokens",
    "vllm:num_generation_tokens_total",
    "vllm:gpu_cache_usage_perc",
    "vllm:avg_generation_throughput_toks_per_s",
    "vllm:request_success_total",
}
METRIC_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
)


@dataclass
class MetricSnapshot:
    metrics_url: str
    values: dict[str, float] = field(default_factory=dict)
    scrape_error: str | None = None

    @property
    def acceptance_rate(self) -> float | None:
        return self.values.get(ACCEPTANCE_RATE_METRIC)


def fetch_vllm_metrics(metrics_url: str) -> MetricSnapshot:
    try:
        resp = httpx.get(metrics_url, timeout=5.0)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("Could not scrape metrics endpoint %s: %s", metrics_url, exc)
        return MetricSnapshot(metrics_url=metrics_url, scrape_error=str(exc))

    values: dict[str, float] = {}
    for line in resp.text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = METRIC_RE.match(line)
        if not match:
            continue
        name = match.group("name")
        if name in INTERESTING_METRICS:
            values[name] = float(match.group("value"))

    return MetricSnapshot(metrics_url=metrics_url, values=values)
