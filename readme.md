# Speculative Decoding Benchmark: Dense vs MoE Self-Speculation

This repo benchmarks quantized self-speculation on Gemma 4 verifier/draft pairs with vLLM. The pipeline is:

1. Expand a validated experiment config.
2. Start vLLM with the correct verifier and optional draft model.
3. Drive a JSONL workload through the OpenAI-compatible completions API.
4. Scrape vLLM Prometheus metrics and generate comparison charts.

The configured matrix has 32 speculative runs:

- Cells: `dense-31b`, `moe-26b`
- Speculation length `k`: `1, 3, 5, 7`
- Temperatures: `0.0, 0.7`
- Workloads: `humaneval`, `sharegpt`

With `baselines.run_no_speculation: true`, the runner also adds 4 verifier-only baseline runs.

## Important A100-80GB Constraint

`configs/experiment.yaml` is set to `1x A100-80GB` because that is the target hardware. The dense cell estimates:

```text
78GB model footprint + 4GB KV/cache overhead = ~82GB required
```

That intentionally fails the VRAM gate before loading models. To run the full dense-vs-MoE matrix, change the config to a larger GPU budget, reduce memory pressure (`max_model_len`, verifier precision, or model choice), or use more GPUs. Use `--skip-vram-validation` only for listing runs, smoke testing, or intentionally running a subset after you have accounted for the memory risk.

## Setup

```bash
python3 -m pip install -r requirements.txt
cp .env.example .env
# edit .env and set HF_TOKEN
```

Gemma models are gated on Hugging Face. Accept the model licenses before running.

## Inspect Runs

```bash
./run.sh --list-runs
```

This prints all expanded run IDs. Example:

```text
run_0000_baseline_dense-31b_humaneval
run_0001_dense-31b_k1_t0p0_humaneval
run_0018_baseline_moe-26b_humaneval
```

## Run Locally

Local mode expects `vllm` to be installed on the host. The benchmark runner starts and stops one vLLM server per run.

```bash
./run.sh run_0018_baseline_moe-26b_humaneval
```

Equivalent direct command:

```bash
python3 src/benchmark.py \
  --config configs/experiment.yaml \
  --run-id run_0018_baseline_moe-26b_humaneval
```

For dry-run command inspection:

```bash
python3 src/benchmark.py --skip-vram-validation --dry-run --limit 2
```

## Run with Docker or Podman

Compose mode starts vLLM in the `vllm/vllm-openai` image and runs the benchmark in a separate Python container.

```bash
MODE=compose ./run.sh run_0018_baseline_moe-26b_humaneval
```

`run.sh` expands the selected `run_id` from `configs/experiment.yaml`, exports the vLLM model flags, then starts Compose. Results land under `results/`.

Podman GPU access requires CDI on many systems:

```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
```

## Analyze Results

```bash
python3 src/analyze.py --results-dir results
```

Charts are written to `results/charts/`:

- `acceptance_rates.png`
- `throughput.png`
- `ttft.png`
- `by_workload.png`
- `difficulty_buckets.png`

## Project Structure

```text
.
├── configs/
│   ├── experiment.yaml
│   ├── models/
│   │   ├── gemma4-31b-dense.yaml
│   │   └── gemma4-26b-moe.yaml
│   └── workloads/
│       ├── humaneval.yaml
│       └── sharegpt.yaml
├── workload_data/
│   ├── humaneval_prompts.jsonl
│   └── sharegpt_slice.jsonl
├── src/
│   ├── benchmark.py
│   ├── config_manager.py
│   ├── server_manager/
│   ├── driver/
│   └── analyze.py
├── docker-compose.yaml
├── Dockerfile
└── run.sh
```

## Result Layout

Each run writes a bundle:

```text
results/<timestamp>_gemma4-self-spec-dense-vs-moe/<run_id>/
├── result.json
└── vllm.log
```

`result.json` includes the expanded run config, request-level latency/throughput records, vLLM metrics before and after the workload, acceptance rate, and difficulty buckets from streamed token logprobs.
