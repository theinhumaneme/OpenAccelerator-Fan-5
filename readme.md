# Speculative Decoding: Dense vs MoE Verifier Benchmark

**Core question:** does the verifier's architecture (dense vs mixture-of-experts) change how well it agrees with a dense draft model — and by how much?

Speculative decoding works because a small draft model proposes tokens and a large verifier accepts or rejects them. Acceptance rate depends on how well their token distributions align. Dense draft → dense verifier is the standard setup. This benchmark tests what happens when the verifier is a MoE model, whose per-token distribution is shaped by dynamic expert routing the draft has no knowledge of.

---

## Hypotheses

1. **Dense verifier → higher acceptance rate.** Dense models share more distributional similarity with a dense draft than MoE models do, because MoE expert routing introduces per-token distribution shifts the draft can't predict.

2. **The gap is not uniform.** For high-probability tokens (common words, syntactic glue), all models agree and acceptance rates converge. The gap opens on low-probability tokens (reasoning steps, domain terms) where MoE routing diverges.

3. **MoE verifier may still win on throughput.** MoE models activate fewer parameters per token, so verification is cheaper per step. Even with lower acceptance rates, end-to-end tokens/sec might be competitive or better.

---

## To start we have two experiments do these first
Experiment 1: Dense vs. Dense (The Gemma 4 Ecosystem)
Your Draft Model Options:
Draft Option 1 (Fastest): google/gemma-4-2b-it
Draft Option 2 (Most Accurate): google/gemma-4-4b-it

Experiment 2: MoE vs. MoE (Brand New April 2026 Ecosystems)
Target Model: Qwen/Qwen3-235B-A22B-Instruct
Draft Model: Qwen/Qwen3.6-35B-A3B-Instruct (Released April 14, 2026)

## Benchmark Matrix

| Experiment | Draft | Verifier | Verifier Arch | Family |
|---|---|---|---|---|
| `dense_same_family` | gemma-2-2b-it | gemma-2-27b-it | Dense | Same |
| `dense_cross_family` | qwen2.5-3b | gemma-2-27b-it | Dense | Cross |
| `moe_cross_family` | qwen2.5-3b | Mixtral-8x7B | MoE | Cross |
| `moe_alt_draft` | gemma-2-2b-it | Mixtral-8x7B | MoE | Cross |

**Key comparisons:**
- `dense_cross_family` vs `moe_cross_family` — same draft, different verifier arch → **isolates the architecture effect**
- `dense_same_family` vs `dense_cross_family` — same verifier, different draft family → **isolates the family-mismatch effect**
- `dense_same_family` vs `moe_cross_family` — both effects together → **worst-case vs best-case**

---

## Metrics

| Metric | How it's measured |
|---|---|
| Acceptance rate | `vllm:spec_decode_draft_acceptance_rate` from vLLM Prometheus `/metrics` |
| Throughput (tok/s) | Output tokens ÷ total wall time, per request |
| TTFT | Time from request send to first streamed token |
| P95 TTFT | 95th percentile TTFT across all prompts |
| Token difficulty | Mean log-probability of output tokens, bucketed into quartiles |

Token difficulty bucketing uses the verifier's output logprobs as a proxy: tokens with very negative logprobs are ones where the verifier is uncertain (hard tokens). This lets you see whether acceptance rate gaps are uniform or concentrated at hard tokens.

---

## Setup

```bash
pip install -r requirements.txt
```

### Optional: quantize the draft model

Reduces VRAM footprint of the draft model from ~5GB (bf16) to ~1GB (INT4):

```bash
chmod +x scripts/quantize_draft.sh
./scripts/quantize_draft.sh google/gemma-2-2b-it ./quantized-draft
# Then update configs/experiments.yaml: draft: "./quantized-draft"
```

---

## Running

Three paths: Docker/Podman (recommended for Brev), bare-metal vLLM, or Docker with manual vLLM management.

---

### Option A — Docker or Podman (recommended on Brev)

This is the simplest path. One script handles everything.

```bash
# 1. Configure
cp .env.example .env
# Edit .env: set HF_TOKEN, VERIFIER_MODEL, DRAFT_MODEL, EXPERIMENT, TP_SIZE

# 2. Run one experiment end-to-end (auto-detects Docker vs Podman)
chmod +x run.sh
./run.sh dense_same_family

# 3. Swap model pair and run the next experiment
# Edit .env to change VERIFIER_MODEL / DRAFT_MODEL / EXPERIMENT, then:
./run.sh moe_cross_family
```

`run.sh` builds the benchmark image, starts vLLM in a container with GPU access, waits for it to be healthy, runs the benchmark, and exits. Results land in `./results/`.

**Podman-specific note:** Podman uses CDI for GPU access. On a fresh Brev instance run once:

```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
```

Then `run.sh` works identically with Podman.

#### Manual compose (if you want more control)

```bash
# Start vLLM only, watch it load:
docker compose up vllm

# In a second terminal, run the benchmark against the already-running vLLM:
docker compose run --rm benchmark \
  --experiment dense_same_family \
  --wait-for-vllm 60

# Tear down between experiments:
docker compose down
```

---

### Option B — Bare-metal vLLM (native install)

Each experiment requires vLLM to be running with the right model pair. The benchmark runner pauses between experiments and prompts you to restart vLLM.

```bash
# Step 1 — launch vLLM for the first experiment
chmod +x scripts/*.sh
./scripts/launch_dense_same.sh
# Wait until vLLM logs: Application startup complete

# Step 2 — run the benchmark (interactive: will prompt before each experiment)
python src/benchmark.py

# Or run a single experiment non-interactively:
python src/benchmark.py --experiment dense_same_family --no-wait
```

Results are written to `results/<experiment_name>.json`.

---

### Step — analyze (same for all options)

```bash
python src/analyze.py
# Charts saved to results/charts/
```

---

## Output

`src/analyze.py` produces five charts:

- **`acceptance_rates.png`** — bar chart of acceptance rate per experiment
- **`throughput.png`** — mean tokens/sec per experiment
- **`ttft.png`** — mean and P95 TTFT per experiment
- **`by_category.png`** — throughput broken down by task category (code / math / chat)
- **`difficulty_buckets.png`** — verifier confidence across easy/hard token buckets

And a summary table printed to stdout:

```
Experiment                          AccRate      TPS   TTFT_ms    P95_ms
--------------------------------------------------------------------------------
dense_same_family                     0.821    142.3     312.1     489.0
dense_cross_family                    0.743    138.7     318.4     501.2
moe_cross_family                      0.691    159.4     298.7     445.3
moe_alt_draft                         0.703    155.1     304.2     460.8
```

*(example values — replace with your actual results)*

---

## Project Structure

```
.
├── Dockerfile                 # benchmark runner image (Python + deps)
├── docker-compose.yaml        # vLLM + benchmark, GPU-enabled, works with Podman too
├── run.sh                     # one-liner wrapper: detects Docker/Podman, runs one experiment
├── .env.example               # template — copy to .env and fill in HF_TOKEN + model names
├── configs/
│   └── experiments.yaml       # all 4 experiment configs + vLLM/benchmark settings
├── scripts/
│   ├── launch_dense_same.sh   # bare-metal vLLM: Gemma 2B → Gemma 27B
│   ├── launch_dense_cross.sh  # bare-metal vLLM: Qwen 3B → Gemma 27B
│   ├── launch_moe_cross.sh    # bare-metal vLLM: Qwen 3B → Mixtral 8x7B
│   ├── launch_moe_alt.sh      # bare-metal vLLM: Gemma 2B → Mixtral 8x7B
│   └── quantize_draft.sh      # INT4 quantization via LLM Compressor
├── src/
│   ├── benchmark.py           # orchestrator: runs prompts, saves JSON
│   ├── client.py              # vLLM OpenAI-compatible client with TTFT timing
│   ├── metrics.py             # Prometheus scraper + token difficulty bucketing
│   ├── prompts.py             # 50 prompts across code / math / chat
│   └── analyze.py             # loads JSON results, generates charts
├── results/                   # benchmark outputs (gitignored)
└── requirements.txt
```

---

## Hardware Notes

Tested configuration (Brev A100 instance):

| Component | Spec |
|---|---|
| GPUs | 4× A100 80GB |
| Driver | CUDA 12.x |
| vLLM | ≥ 0.4.0 (speculative decoding support) |
| Tensor parallel | 4 (both verifiers) |
| Speculative tokens | 5 per step |

Mixtral-8x7B fits comfortably on 2× A100 80GB in bf16. Gemma-2-27B needs ~55GB, fits on 1× A100 80GB with some headroom for KV cache.

---

## The Finding That Would Matter

If MoE verifiers show **lower acceptance rates but equal or higher throughput**, that means the standard evaluation metric for speculative decoding (acceptance rate) is misleading for MoE architectures. The correct metric is effective tokens per GPU-second, and optimizing for acceptance rate alone would lead to wrong model selection decisions.

If acceptance rate gaps are **concentrated at hard tokens** (low logprob buckets), that implies speculation length `k` should be tuned differently for MoE verifiers — shorter sequences to avoid wasted computation on the hard tokens where rejection is likely.
