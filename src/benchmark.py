from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from config_manager import ExperimentPlan, RunConfig, load_experiment, run_to_env
from driver.collector import collect_run
from server_manager import VLLMServerProcess, build_vllm_command, run_warmup, wait_for_healthy

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the speculative decoding benchmark matrix.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--run-id", help="Run one expanded run_id. Defaults to all runs.")
    parser.add_argument("--cell-id", help="Filter runs by cell id, e.g. dense-31b or moe-26b.")
    parser.add_argument("--workload-id", help="Filter runs by workload id.")
    parser.add_argument("--baseline-only", action="store_true", help="Run only no-spec baselines.")
    parser.add_argument("--spec-only", action="store_true", help="Run only speculative runs.")
    parser.add_argument("--limit", type=int, help="Run at most N matching configs.")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--list-runs", action="store_true", help="Print expanded run ids and exit.")
    parser.add_argument("--emit-env", metavar="RUN_ID", help="Print shell env for launching one run.")
    parser.add_argument("--dry-run", action="store_true", help="Print vLLM commands without running.")
    parser.add_argument(
        "--external-vllm",
        action="store_true",
        help="Do not start vLLM; assume it is already running at --base-url.",
    )
    parser.add_argument("--base-url", help="OpenAI-compatible base URL, e.g. http://localhost:8000/v1.")
    parser.add_argument("--metrics-url", help="Prometheus metrics URL, e.g. http://localhost:8000/metrics.")
    parser.add_argument("--health-url", help="vLLM health URL, e.g. http://localhost:8000/health.")
    parser.add_argument("--health-timeout", type=int, default=600)
    parser.add_argument(
        "--skip-vram-validation",
        action="store_true",
        help="Expand/run configs even if the configured VRAM budget is overcommitted.",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        plan = load_experiment(args.config, validate_vram=False)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    if args.emit_env:
        run = _find_run(plan.runs, args.emit_env)
        print(run_to_env(run))
        return

    runs = _filter_runs(
        plan.runs,
        run_id=args.run_id,
        cell_id=args.cell_id,
        workload_id=args.workload_id,
        baseline_only=args.baseline_only,
        spec_only=args.spec_only,
        limit=args.limit,
    )

    if args.list_runs:
        _print_runs(runs, plan)
        return

    if not runs:
        print("No runs matched the requested filters.", file=sys.stderr)
        sys.exit(1)

    vram_messages = _vram_messages_for_runs(plan, runs)
    if not args.skip_vram_validation and any(msg.startswith("ERROR:") for msg in vram_messages):
        print("VRAM validation failed for selected runs:", file=sys.stderr)
        print("\n".join(vram_messages), file=sys.stderr)
        sys.exit(2)

    for warning in vram_messages:
        logger.warning(warning)

    if args.dry_run:
        for run in runs:
            print(f"\n# {run.run_id}")
            print(" ".join(build_vllm_command(run)))
        return

    session_dir = _session_dir(Path(args.out_dir), plan)
    session_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Writing results under %s", session_dir)

    for run in runs:
        _run_one(
            run=run,
            plan=plan,
            session_dir=session_dir,
            external_vllm=args.external_vllm,
            base_url=args.base_url,
            metrics_url=args.metrics_url,
            health_url=args.health_url,
            health_timeout=args.health_timeout,
        )

    print(f"Completed {len(runs)} run(s). Results: {session_dir}")


def _run_one(
    *,
    run: RunConfig,
    plan: ExperimentPlan,
    session_dir: Path,
    external_vllm: bool,
    base_url: str | None,
    metrics_url: str | None,
    health_url: str | None,
    health_timeout: int,
) -> None:
    run_dir = session_dir / run.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    effective_base_url = base_url or os.environ.get("VLLM_BASE_URL") or _base_url_for_run(run)
    effective_metrics_url = metrics_url or os.environ.get("VLLM_METRICS_URL") or _metrics_url_for_base(
        effective_base_url
    )
    effective_health_url = health_url or os.environ.get("VLLM_HEALTH_URL") or _health_url_for_base(
        effective_base_url
    )

    logger.info("Starting run %s", run.run_id)
    if external_vllm:
        wait_for_healthy(effective_health_url, timeout_seconds=health_timeout)
        run_warmup(effective_base_url, run, plan.baselines.warmup_requests)
        collect_run(
            run=run,
            base_url=effective_base_url,
            metrics_url=effective_metrics_url,
            out_dir=run_dir,
        )
        return

    server = VLLMServerProcess(run, log_path=run_dir / "vllm.log")
    try:
        server.start()
        wait_for_healthy(effective_health_url, timeout_seconds=health_timeout)
        run_warmup(effective_base_url, run, plan.baselines.warmup_requests)
        collect_run(
            run=run,
            base_url=effective_base_url,
            metrics_url=effective_metrics_url,
            out_dir=run_dir,
        )
    finally:
        server.stop()


def _filter_runs(
    runs: list[RunConfig],
    *,
    run_id: str | None,
    cell_id: str | None,
    workload_id: str | None,
    baseline_only: bool,
    spec_only: bool,
    limit: int | None,
) -> list[RunConfig]:
    if baseline_only and spec_only:
        raise SystemExit("--baseline-only and --spec-only are mutually exclusive")

    filtered = runs
    if run_id:
        filtered = [run for run in filtered if run.run_id == run_id]
    if cell_id:
        filtered = [run for run in filtered if run.cell_id == cell_id]
    if workload_id:
        filtered = [run for run in filtered if run.workload_id == workload_id]
    if baseline_only:
        filtered = [run for run in filtered if run.is_baseline]
    if spec_only:
        filtered = [run for run in filtered if not run.is_baseline]
    if limit is not None:
        filtered = filtered[:limit]
    return filtered


def _find_run(runs: list[RunConfig], run_id: str) -> RunConfig:
    for run in runs:
        if run.run_id == run_id:
            return run
    choices = ", ".join(run.run_id for run in runs[:10])
    raise SystemExit(f"Unknown run_id '{run_id}'. First choices: {choices}")


def _print_runs(runs: list[RunConfig], plan: ExperimentPlan) -> None:
    print(f"Experiment: {plan.experiment.name} v{plan.experiment.version}")
    print(f"Hardware: {plan.hardware.gpu_count}x {plan.hardware.gpu_sku} ({plan.hardware.vram_gb:g}GB each)")
    vram_messages = _vram_messages_for_runs(plan, runs)
    if vram_messages:
        print("VRAM checks:")
        for warning in vram_messages:
            print(f"  {warning}")
    print()
    print(f"{'run_id':<58} {'cell':<10} {'k':>2} {'temp':>4} {'workload':<10} {'mode':<8}")
    print("-" * 100)
    for run in runs:
        mode = "baseline" if run.is_baseline else "spec"
        print(
            f"{run.run_id:<58} {run.cell_id:<10} "
            f"{run.num_speculative_tokens:>2} {run.temperature:>4.1f} "
            f"{run.workload_id:<10} {mode:<8}"
        )


def _session_dir(out_dir: Path, plan: ExperimentPlan) -> Path:
    if not plan.experiment.timestamp_prefix:
        return out_dir
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return out_dir / f"{timestamp}_{plan.experiment.name}"


def _base_url_for_run(run: RunConfig) -> str:
    return f"http://localhost:{run.vllm_port}/v1"


def _vram_messages_for_runs(plan: ExperimentPlan, runs: list[RunConfig]) -> list[str]:
    selected_cell_ids = {run.cell_id for run in runs}
    available = plan.hardware.total_vram_gb
    overhead = plan.vllm_defaults.kv_cache_overhead_gb
    min_headroom = plan.vllm_defaults.min_vram_headroom_gb
    messages: list[str] = []

    for cell in plan.cells:
        if cell.id not in selected_cell_ids:
            continue
        required = cell.total_vram_gb + overhead
        headroom = available - required
        if headroom < 0:
            messages.append(
                "ERROR: "
                f"cell '{cell.id}' needs ~{required:.1f}GB "
                f"({cell.total_vram_gb:.1f}GB model + {overhead:.1f}GB KV/cache overhead) "
                f"but {plan.hardware.gpu_count}x {plan.hardware.gpu_sku} provides "
                f"~{available:.1f}GB. Reduce max_model_len, use a smaller/quantized verifier, "
                "or increase gpu_count."
            )
        elif headroom < min_headroom:
            messages.append(
                "WARNING: "
                f"cell '{cell.id}' has only ~{headroom:.1f}GB VRAM headroom. "
                "Expect KV-cache pressure on longer sequences."
            )
    return messages


def _metrics_url_for_base(base_url: str) -> str:
    parsed = urlparse(base_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    return base + "/metrics"


def _health_url_for_base(base_url: str) -> str:
    parsed = urlparse(base_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    return base + "/health"


if __name__ == "__main__":
    main()
