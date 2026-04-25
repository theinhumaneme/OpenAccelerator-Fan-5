"""
Generate comparison charts from benchmark result files.

Usage:
  python src/analyze.py                          # reads results/*.json
  python src/analyze.py --results-dir results    # explicit path
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DENSE_COLOR = "#2196F3"
MOE_COLOR = "#FF5722"


def load_results(results_dir: Path) -> dict[str, dict]:
    data: dict[str, dict] = {}
    for f in sorted(results_dir.glob("*.json")):
        d = json.loads(f.read_text())
        data[d["experiment"]] = d
    if not data:
        print(f"No *.json files found in {results_dir}")
    return data


def _color(exp_name: str) -> str:
    return DENSE_COLOR if "dense" in exp_name else MOE_COLOR


def _short_label(exp_name: str) -> str:
    return exp_name.replace("_", "\n")


def plot_acceptance_rates(data: dict, out_dir: Path) -> None:
    names = list(data.keys())
    rates = [data[n].get("acceptance_rate") or 0.0 for n in names]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(
        [_short_label(n) for n in names],
        rates,
        color=[_color(n) for n in names],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_ylabel("Acceptance Rate")
    ax.set_title("Speculative Decoding: Draft Token Acceptance Rate by Verifier Architecture")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.8, color="gray", linestyle="--", linewidth=0.8, label="0.8 reference")
    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{rate:.2f}",
            ha="center", va="bottom", fontsize=8,
        )
    legend_patches = [
        mpatches.Patch(color=DENSE_COLOR, label="Dense verifier"),
        mpatches.Patch(color=MOE_COLOR, label="MoE verifier"),
    ]
    ax.legend(handles=legend_patches, loc="upper right")
    plt.tight_layout()
    path = out_dir / "acceptance_rates.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_throughput(data: dict, out_dir: Path) -> None:
    names = list(data.keys())
    tps = [data[n]["stats"].get("mean_throughput_tps", 0) for n in names]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(
        [_short_label(n) for n in names],
        tps,
        color=[_color(n) for n in names],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_ylabel("Tokens / Second")
    ax.set_title("Speculative Decoding: Throughput by Verifier Architecture")
    for bar, t in zip(bars, tps):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{t:.1f}",
            ha="center", va="bottom", fontsize=8,
        )
    legend_patches = [
        mpatches.Patch(color=DENSE_COLOR, label="Dense verifier"),
        mpatches.Patch(color=MOE_COLOR, label="MoE verifier"),
    ]
    ax.legend(handles=legend_patches)
    plt.tight_layout()
    path = out_dir / "throughput.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_ttft(data: dict, out_dir: Path) -> None:
    names = list(data.keys())
    means = [data[n]["stats"].get("mean_ttft_ms", 0) for n in names]
    p95s = [data[n]["stats"].get("p95_ttft_ms", 0) for n in names]
    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - w / 2, means, w, label="Mean TTFT", color=[_color(n) for n in names])
    ax.bar(x + w / 2, p95s, w, label="P95 TTFT", color=[_color(n) for n in names], alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([_short_label(n) for n in names])
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("Time to First Token by Verifier Architecture")
    ax.legend()
    plt.tight_layout()
    path = out_dir / "ttft.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_task_accuracy(data: dict, out_dir: Path) -> None:
    names = list(data.keys())
    x = np.arange(len(names))
    w = 0.28

    prompt_accs = [data[n].get("accuracy", {}).get("ifeval", {}).get("prompt_accuracy", 0.0) for n in names]
    inst_accs = [data[n].get("accuracy", {}).get("ifeval", {}).get("instruction_accuracy", 0.0) for n in names]
    other_accs = [data[n].get("accuracy", {}).get("other_accuracy", 0.0) for n in names]

    fig, ax = plt.subplots(figsize=(11, 4))
    b1 = ax.bar(x - w, prompt_accs, w, label="IFEval prompt acc", color=[_color(n) for n in names])
    b2 = ax.bar(x,     inst_accs,   w, label="IFEval instruction acc", color=[_color(n) for n in names], alpha=0.6)
    b3 = ax.bar(x + w, other_accs,  w, label="Other task acc (GSM8K/MMLU-Pro)", color=[_color(n) for n in names], alpha=0.35)

    ax.set_xticks(x)
    ax.set_xticklabels([_short_label(n) for n in names])
    ax.set_ylabel("Accuracy")
    ax.set_title("Task Accuracy by Verifier Architecture")
    ax.set_ylim(0, 1.1)
    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}",
                        ha="center", va="bottom", fontsize=7)
    ax.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    path = out_dir / "task_accuracy.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_by_category(data: dict, out_dir: Path) -> None:
    categories = sorted(set(
        cat
        for d in data.values()
        for cat in d.get("per_category", {})
    ))
    if not categories:
        return
    exp_names = list(data.keys())
    x = np.arange(len(categories))
    w = 0.8 / len(exp_names)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, exp in enumerate(exp_names):
        per_cat = data[exp].get("per_category", {})
        tps = [per_cat.get(cat, {}).get("mean_throughput_tps", 0) for cat in categories]
        offset = x + (i - len(exp_names) / 2 + 0.5) * w
        ax.bar(offset, tps, w, label=exp, color=_color(exp), alpha=0.75 + 0.05 * i)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Tokens / Second")
    ax.set_title("Throughput by Task Category and Verifier Architecture")
    ax.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    path = out_dir / "by_category.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_difficulty_buckets(data: dict, out_dir: Path) -> None:
    """
    For each experiment, show mean logprob across difficulty buckets.
    Lower (more negative) logprob = harder tokens the verifier is less certain about.
    This proxies where the acceptance rate gap between dense and MoE would concentrate.
    """
    bucket_labels = ["very_hard", "hard", "easy", "very_easy"]
    exp_names = list(data.keys())
    x = np.arange(len(bucket_labels))
    w = 0.8 / len(exp_names)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, exp in enumerate(exp_names):
        buckets = data[exp].get("difficulty_buckets", {})
        vals = [buckets.get(b, 0) for b in bucket_labels]
        offset = x + (i - len(exp_names) / 2 + 0.5) * w
        ax.bar(offset, vals, w, label=exp, color=_color(exp), alpha=0.75 + 0.05 * i)

    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels)
    ax.set_ylabel("Mean Token Log-Probability")
    ax.set_title("Verifier Confidence by Token Difficulty Bucket")
    ax.legend(fontsize=7)
    plt.tight_layout()
    path = out_dir / "difficulty_buckets.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def print_summary(data: dict) -> None:
    header = (
        f"{'Experiment':<35} {'AccRate':>8} {'TPS':>8} {'TTFT_ms':>9}"
        f" {'P95_ms':>9} {'IF_Prompt':>10} {'IF_Inst':>8} {'Other':>7}"
    )
    w = len(header)
    print("\n" + "=" * w)
    print(header)
    print("-" * w)
    for name, d in data.items():
        acc = d.get("acceptance_rate")
        s = d.get("stats", {})
        accd = d.get("accuracy", {})
        if_prompt = accd.get("ifeval", {}).get("prompt_accuracy", 0.0)
        if_inst = accd.get("ifeval", {}).get("instruction_accuracy", 0.0)
        other = accd.get("other_accuracy", 0.0)
        print(
            f"{name:<35} "
            f"{acc or 0:>8.3f} "
            f"{s.get('mean_throughput_tps', 0):>8.1f} "
            f"{s.get('mean_ttft_ms', 0):>9.1f} "
            f"{s.get('p95_ttft_ms', 0):>9.1f} "
            f"{if_prompt:>10.3f} "
            f"{if_inst:>8.3f} "
            f"{other:>7.3f}"
        )
    print("=" * w)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="results/charts")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_results(results_dir)
    if not data:
        return

    print_summary(data)
    print(f"\nGenerating charts → {out_dir}/")
    plot_acceptance_rates(data, out_dir)
    plot_throughput(data, out_dir)
    plot_ttft(data, out_dir)
    plot_task_accuracy(data, out_dir)
    plot_by_category(data, out_dir)
    plot_difficulty_buckets(data, out_dir)


if __name__ == "__main__":
    main()
