from __future__ import annotations

import logging
import time

import httpx

logger = logging.getLogger(__name__)


def wait_for_healthy(
    health_url: str,
    *,
    timeout_seconds: int = 600,
    poll_interval: float = 5.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = "not checked"

    while time.monotonic() < deadline:
        try:
            resp = httpx.get(health_url, timeout=5.0)
            if resp.status_code == 200:
                logger.info("vLLM healthy at %s", health_url)
                return
            last_error = f"HTTP {resp.status_code}"
        except httpx.HTTPError as exc:
            last_error = str(exc)

        time.sleep(poll_interval)

    raise TimeoutError(
        f"vLLM did not become healthy at {health_url} within {timeout_seconds}s "
        f"(last error: {last_error})"
    )
