from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from config_manager import RunConfig
from driver.metrics_scraper import MetricSnapshot, fetch_vllm_metrics
from driver.request_driver import RequestResult, send_workload


def collect_run(
    *,
    run: RunConfig,
    base_url: str,
    metrics_url: str,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_before = fetch_vllm_metrics(metrics_url)
    results = send_workload(base_url, run)
    metrics_after = fetch_vllm_metrics(metrics_url)

    payload = {
        "run_id": run.run_id,
        "experiment": run.run_id,
        "run_config": run.model_dump(),
        "acceptance_rate": metrics_after.acceptance_rate,
        "metrics_before": _snapshot_to_dict(metrics_before),
        "metrics_after": _snapshot_to_dict(metrics_after),
        "stats": aggregate(results),
        "difficulty_buckets": bucket_by_difficulty(
            [lp for result in results for lp in result.token_logprobs]
        ),
        "per_category": _aggregate_by_category(results),
        "raw": [asdict(result) for result in results],
    }

    result_path = out_dir / "result.json"
    result_path.write_text(json.dumps(payload, indent=2))
    return payload


def aggregate(results: list[RequestResult]) -> dict[str, float]:
    successful = [result for result in results if result.error is None]
    if not successful:
        return {"n": len(results), "successful": 0, "failed": len(results)}

    ttfts = np.array([result.ttft_ms for result in successful], dtype=float)
    totals = np.array([result.wall_clock_seconds * 1000 for result in successful], dtype=float)
    tps = np.array([result.tokens_per_second for result in successful], dtype=float)
    output_tokens = np.array([result.completion_tokens for result in successful], dtype=float)

    return {
        "n": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "mean_ttft_ms": float(np.mean(ttfts)),
        "p50_ttft_ms": float(np.percentile(ttfts, 50)),
        "p95_ttft_ms": float(np.percentile(ttfts, 95)),
        "mean_total_ms": float(np.mean(totals)),
        "mean_throughput_tps": float(np.mean(tps)),
        "p50_throughput_tps": float(np.percentile(tps, 50)),
        "total_completion_tokens": float(np.sum(output_tokens)),
    }


def bucket_by_difficulty(token_logprobs: list[float], n_buckets: int = 4) -> dict[str, float]:
    if not token_logprobs:
        return {}

    arr = np.array(token_logprobs, dtype=float)
    thresholds = np.percentile(arr, np.linspace(0, 100, n_buckets + 1))
    labels = ["very_hard", "hard", "easy", "very_easy"]
    buckets: dict[str, float] = {}

    for idx in range(n_buckets):
        lo, hi = thresholds[idx], thresholds[idx + 1]
        if idx == n_buckets - 1:
            mask = (arr >= lo) & (arr <= hi)
        else:
            mask = (arr >= lo) & (arr < hi)
        label = labels[idx] if idx < len(labels) else f"bucket_{idx}"
        buckets[label] = float(np.mean(arr[mask])) if mask.any() else 0.0

    return buckets


def _aggregate_by_category(results: list[RequestResult]) -> dict[str, dict[str, float]]:
    categories = sorted({result.category for result in results})
    return {
        category: aggregate([result for result in results if result.category == category])
        for category in categories
    }


def _snapshot_to_dict(snapshot: MetricSnapshot) -> dict:
    return {
        "metrics_url": snapshot.metrics_url,
        "values": snapshot.values,
        "scrape_error": snapshot.scrape_error,
        "acceptance_rate": snapshot.acceptance_rate,
    }
