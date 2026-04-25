from __future__ import annotations

import json
import logging
import signal
import subprocess
from pathlib import Path
from typing import TextIO

from config_manager import RunConfig

logger = logging.getLogger(__name__)

VLLM_VERSION_TESTED = "0.8.x"


def build_vllm_command(run: RunConfig) -> list[str]:
    cmd = [
        "vllm",
        "serve",
        run.verifier_model,
        "--host",
        run.vllm_host,
        "--port",
        str(run.vllm_port),
        "--served-model-name",
        run.served_model_name,
        "--dtype",
        run.verifier_dtype,
        "--gpu-memory-utilization",
        str(run.gpu_memory_utilization),
        "--max-model-len",
        str(run.max_model_len),
        "--max-num-seqs",
        str(run.max_num_seqs),
        "--block-size",
        str(run.block_size),
    ]

    if run.enforce_eager:
        cmd.append("--enforce-eager")
    if run.enable_prefix_caching:
        cmd.append("--enable-prefix-caching")

    if not run.is_baseline:
        spec_config = json.dumps(
            {
                "model": run.draft_model,
                "num_speculative_tokens": run.num_speculative_tokens,
                "quantization": run.draft_quantization,
            },
            separators=(",", ":"),
        )
        cmd.extend(["--speculative-config", spec_config])

    return cmd


class VLLMServerProcess:
    def __init__(self, run: RunConfig, log_path: Path | None = None):
        self.run = run
        self.log_path = log_path
        self.process: subprocess.Popen[str] | None = None
        self._log_file: TextIO | None = None

    def start(self) -> None:
        cmd = build_vllm_command(self.run)
        logger.info("Starting vLLM for %s: %s", self.run.run_id, " ".join(cmd))

        stdout: int | TextIO = subprocess.PIPE
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = self.log_path.open("w")
            stdout = self._log_file

        self.process = subprocess.Popen(
            cmd,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def stop(self, timeout: int = 30) -> None:
        if self.process is None:
            self._close_log_file()
            return

        if self.process.poll() is None:
            logger.info("Stopping vLLM for %s (PID %s)", self.run.run_id, self.process.pid)
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning("vLLM did not exit cleanly, sending SIGKILL")
                self.process.kill()
                self.process.wait()

        self.process = None
        self._close_log_file()

    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def _close_log_file(self) -> None:
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def __enter__(self) -> "VLLMServerProcess":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
