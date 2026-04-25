"""
Main benchmark runner.

Usage:
  # Run all experiments (prompts for each one in sequence):
  python src/benchmark.py

  # Run a single named experiment:
  python src/benchmark.py --experiment dense_same_family

  # Override config or output directory:
  python src/benchmark.py --config configs/experiments.yaml --out-dir results/
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from dataclasses import asdict

import httpx
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from client import make_client, run_request
from metrics import fetch_acceptance_rate, bucket_by_difficulty, aggregate, compute_task_accuracy
from prompts import load_lighteval_tasks


def wait_for_vllm(base_url: str, timeout: int = 300) -> None:
    health_url = base_url.replace("/v1", "").rstrip("/") + "/health"
    print(f"  Waiting for vLLM at {health_url} (timeout={timeout}s) ...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(health_url, timeout=5.0)
            if r.status_code == 200:
                print("  vLLM is ready.")
                return
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError(f"vLLM did not become ready within {timeout}s")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_experiment(
    exp_name: str,
    exp_cfg: dict,
    global_cfg: dict,
    out_dir: Path,
) -> dict:
    vllm = global_cfg["vllm"]
    bench = global_cfg["benchmark"]

    client = make_client(vllm["base_url"])
    prompts = load_lighteval_tasks(
        bench["tasks"],
        num_prompts_per_task=bench.get("num_prompts_per_task"),
    )

    print(f"  Verifier : {exp_cfg['verifier']}")
    print(f"  Draft    : {exp_cfg['draft']}")
    print(f"  Prompts  : {len(prompts)} ({len(bench['tasks'])} tasks)")

    acc_before = fetch_acceptance_rate(vllm["metrics_url"])

    results = []
    for item in tqdm(prompts, desc=exp_name, ncols=80):
        result = run_request(
            client=client,
            model=exp_cfg["verifier"],
            prompt=item["prompt"],
            category=item["category"],
            max_tokens=vllm["max_tokens"],
            target=item["target"],
            instruction_ids=item.get("instruction_ids"),
            instruction_kwargs=item.get("instruction_kwargs"),
        )
        results.append(result)
        time.sleep(0.05)

    acc_after = fetch_acceptance_rate(vllm["metrics_url"])

    all_logprobs = [lp for r in results for lp in r.token_logprobs]
    categories = sorted(set(item["category"] for item in prompts))

    output = {
        "experiment": exp_name,
        "config": exp_cfg,
        "acceptance_rate": acc_after,
        "acceptance_rate_before": acc_before,
        "stats": aggregate(results),
        "accuracy": compute_task_accuracy(results),
        "difficulty_buckets": bucket_by_difficulty(all_logprobs),
        "per_category": {
            cat: aggregate([r for r in results if r.category == cat])
            for cat in categories
        },
        "raw": [asdict(r) for r in results],
    }

    out_path = out_dir / f"{exp_name}.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"  Saved → {out_path}")

    acc_str = f"{acc_after:.3f}" if acc_after is not None else "n/a"
    tps = output["stats"].get("mean_throughput_tps", 0)
    print(f"  acceptance_rate={acc_str}  throughput={tps:.1f} tok/s")

    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--experiment", help="Run only this experiment key")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Skip the interactive 'Press Enter' prompt (assumes vLLM is already up)",
    )
    parser.add_argument(
        "--wait-for-vllm",
        type=int,
        default=0,
        metavar="SECONDS",
        help="Poll vLLM /health until ready (use in Docker/CI instead of --no-wait)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Allow Docker / CI to override vLLM URLs via environment variables.
    if url := os.environ.get("VLLM_BASE_URL"):
        cfg["vllm"]["base_url"] = url
    if url := os.environ.get("VLLM_METRICS_URL"):
        cfg["vllm"]["metrics_url"] = url

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    experiments = cfg["experiments"]
    if args.experiment:
        if args.experiment not in experiments:
            print(f"Unknown experiment '{args.experiment}'. Choices: {list(experiments)}")
            sys.exit(1)
        experiments = {args.experiment: experiments[args.experiment]}

    for name, exp_cfg in experiments.items():
        print(f"\n{'='*60}")
        print(f"Experiment: {exp_cfg['name']}")
        if args.wait_for_vllm > 0:
            wait_for_vllm(cfg["vllm"]["base_url"], timeout=args.wait_for_vllm)
        elif not args.no_wait:
            print(f"Launch script: {exp_cfg.get('launch_script', 'n/a')}")
            print("Make sure vLLM is running with this model pair before continuing.")
            input("Press Enter when vLLM is ready... ")
        run_experiment(name, exp_cfg, cfg, out_dir)

    print("\nAll done. Run `python src/analyze.py` to generate charts.")


if __name__ == "__main__":
    main()
