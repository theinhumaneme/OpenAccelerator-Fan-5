from __future__ import annotations

import logging

import httpx

from config_manager import RunConfig

logger = logging.getLogger(__name__)

WARMUP_PROMPT = "Write a short greeting."


def run_warmup(base_url: str, run: RunConfig, num_requests: int) -> None:
    if num_requests <= 0:
        return

    url = base_url.rstrip("/") + "/completions"
    failures = 0
    for idx in range(num_requests):
        try:
            resp = httpx.post(
                url,
                json={
                    "model": run.served_model_name,
                    "prompt": WARMUP_PROMPT,
                    "max_tokens": 64,
                    "temperature": 0.0,
                },
                timeout=120.0,
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            failures += 1
            logger.warning("Warmup request %s failed: %s", idx, exc)

    if failures:
        logger.warning("Warmup completed with %s/%s failures", failures, num_requests)
    else:
        logger.info("Warmup complete: %s requests", num_requests)
