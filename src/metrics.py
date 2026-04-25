import re
import httpx
import numpy as np


def compute_task_accuracy(results: list) -> dict:
    """Generation accuracy: target string must appear at the start of the model output."""
    correct = total = 0
    for r in results:
        if not r.target:
            continue
        total += 1
        # Normalize: strip non-alphanumeric, lowercase, compare prefix of output
        norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
        if norm(r.target) and norm(r.target) in norm(r.output[:len(r.target) + 30]):
            correct += 1
    return {
        "accuracy": correct / total if total else 0.0,
        "n_scored": total,
    }


# vLLM exposes this gauge in its Prometheus /metrics endpoint when
# speculative decoding is enabled.
_ACCEPTANCE_RATE_METRIC = "vllm:spec_decode_draft_acceptance_rate"


def fetch_acceptance_rate(metrics_url: str) -> float | None:
    try:
        resp = httpx.get(metrics_url, timeout=5.0)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [warn] Could not reach metrics endpoint: {e}")
        return None

    for line in resp.text.splitlines():
        if line.startswith(_ACCEPTANCE_RATE_METRIC) and not line.startswith("#"):
            m = re.search(r"\}\s*([\d.eE+\-]+)", line)
            if m:
                return float(m.group(1))
    return None


def bucket_by_difficulty(token_logprobs: list[float], n_buckets: int = 4) -> dict[str, float]:
    """
    Bucket tokens by their log-probability (proxy for difficulty).
    High logprob = model is confident = easy token.
    Returns the mean logprob per bucket.
    """
    if not token_logprobs:
        return {}
    arr = np.array(token_logprobs, dtype=float)
    # Sort ascending so bucket 0 = hardest (most negative logprob)
    thresholds = np.percentile(arr, np.linspace(0, 100, n_buckets + 1))
    labels = ["very_hard", "hard", "easy", "very_easy"]
    buckets: dict[str, float] = {}
    for i in range(n_buckets):
        lo, hi = thresholds[i], thresholds[i + 1]
        mask = (arr >= lo) & (arr <= hi)
        label = labels[i] if i < len(labels) else f"bucket_{i}"
        buckets[label] = float(np.mean(arr[mask])) if mask.any() else 0.0
    return buckets


def aggregate(results: list) -> dict[str, float]:
    if not results:
        return {}
    ttfts = np.array([r.ttft_ms for r in results])
    totals = np.array([r.total_ms for r in results])
    tps = np.array([r.throughput_tps for r in results])
    return {
        "n": len(results),
        "mean_ttft_ms": float(np.mean(ttfts)),
        "p50_ttft_ms": float(np.percentile(ttfts, 50)),
        "p95_ttft_ms": float(np.percentile(ttfts, 95)),
        "mean_total_ms": float(np.mean(totals)),
        "mean_throughput_tps": float(np.mean(tps)),
        "p50_throughput_tps": float(np.percentile(tps, 50)),
    }
