from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

_tmp_dir = Path(os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(_tmp_dir / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_tmp_dir / "xdg-cache"))

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

DENSE_COLOR = "#2563EB"
MOE_COLOR = "#DC2626"


def load_results(results_dir: Path) -> dict[str, dict]:
    data: dict[str, dict] = {}
    files = sorted(results_dir.glob("**/result.json"))
    if not files:
        files = sorted(results_dir.glob("*.json"))

    for path in files:
        payload = json.loads(path.read_text())
        run_id = payload.get("run_id") or payload.get("experiment") or path.stem
        data[run_id] = payload

    if not data:
        print(f"No result JSON files found in {results_dir}")
    return data


def plot_acceptance_rates(data: dict[str, dict], out_dir: Path) -> None:
    names = list(data)
    rates = [data[name].get("acceptance_rate") or 0.0 for name in names]

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.45), 4.5))
    bars = ax.bar(
        [_short_label(name) for name in names],
        rates,
        color=[_color(data[name]) for name in names],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_ylabel("Acceptance Rate")
    ax.set_title("Speculative Decoding Draft Acceptance Rate")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", labelrotation=75, labelsize=7)
    ax.axhline(0.8, color="gray", linestyle="--", linewidth=0.8)
    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{rate:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    _add_arch_legend(ax)
    plt.tight_layout()
    _save(fig, out_dir / "acceptance_rates.png")


def plot_throughput(data: dict[str, dict], out_dir: Path) -> None:
    names = list(data)
    tps = [data[name].get("stats", {}).get("mean_throughput_tps", 0) for name in names]

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.45), 4.5))
    bars = ax.bar(
        [_short_label(name) for name in names],
        tps,
        color=[_color(data[name]) for name in names],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_ylabel("Output Tokens / Second")
    ax.set_title("Mean Request Throughput")
    ax.tick_params(axis="x", labelrotation=75, labelsize=7)
    for bar, value in zip(bars, tps):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    _add_arch_legend(ax)
    plt.tight_layout()
    _save(fig, out_dir / "throughput.png")


def plot_ttft(data: dict[str, dict], out_dir: Path) -> None:
    names = list(data)
    means = [data[name].get("stats", {}).get("mean_ttft_ms", 0) for name in names]
    p95s = [data[name].get("stats", {}).get("p95_ttft_ms", 0) for name in names]
    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.45), 4.5))
    ax.bar(x - width / 2, means, width, label="Mean TTFT", color=[_color(data[n]) for n in names])
    ax.bar(
        x + width / 2,
        p95s,
        width,
        label="P95 TTFT",
        color=[_color(data[n]) for n in names],
        alpha=0.55,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([_short_label(name) for name in names], rotation=75, fontsize=7)
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("Time to First Token")
    ax.legend()
    plt.tight_layout()
    _save(fig, out_dir / "ttft.png")


def plot_task_accuracy(data: dict[str, dict], out_dir: Path) -> None:
    names = [name for name, payload in data.items() if payload.get("accuracy")]
    if not names:
        return

    x = np.arange(len(names))
    width = 0.28
    prompt_accs = [
        data[name].get("accuracy", {}).get("ifeval", {}).get("prompt_accuracy", 0.0)
        for name in names
    ]
    inst_accs = [
        data[name].get("accuracy", {}).get("ifeval", {}).get("instruction_accuracy", 0.0)
        for name in names
    ]
    other_accs = [data[name].get("accuracy", {}).get("other_accuracy", 0.0) for name in names]

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.45), 4.5))
    bars = [
        ax.bar(x - width, prompt_accs, width, label="IFEval prompt acc", color=[_color(data[n]) for n in names]),
        ax.bar(x, inst_accs, width, label="IFEval instruction acc", color=[_color(data[n]) for n in names], alpha=0.6),
        ax.bar(x + width, other_accs, width, label="Other task acc", color=[_color(data[n]) for n in names], alpha=0.35),
    ]
    ax.set_xticks(x)
    ax.set_xticklabels([_short_label(name) for name in names], rotation=75, fontsize=7)
    ax.set_ylabel("Accuracy")
    ax.set_title("Task Accuracy")
    ax.set_ylim(0, 1.1)
    for group in bars:
        for bar in group:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.01,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
    ax.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    _save(fig, out_dir / "task_accuracy.png")


def plot_by_workload(data: dict[str, dict], out_dir: Path) -> None:
    workloads = sorted({_run_config(payload).get("workload_id", "unknown") for payload in data.values()})
    cells = sorted({_run_config(payload).get("cell_id", "unknown") for payload in data.values()})
    labels = [f"{cell}\n{workload}" for cell in cells for workload in workloads]
    values = []
    colors = []

    for cell in cells:
        for workload in workloads:
            matching = [
                payload
                for payload in data.values()
                if _run_config(payload).get("cell_id") == cell
                and _run_config(payload).get("workload_id") == workload
            ]
            values.append(_mean([payload.get("stats", {}).get("mean_throughput_tps", 0) for payload in matching]))
            colors.append(DENSE_COLOR if "dense" in cell else MOE_COLOR)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Output Tokens / Second")
    ax.set_title("Mean Throughput by Cell and Workload")
    _add_arch_legend(ax)
    plt.tight_layout()
    _save(fig, out_dir / "by_workload.png")


def plot_by_category(data: dict[str, dict], out_dir: Path) -> None:
    categories = sorted(
        {
            category
            for payload in data.values()
            for category in payload.get("per_category", {})
        }
    )
    if not categories:
        return

    names = list(data)
    x = np.arange(len(categories))
    width = 0.8 / max(1, len(names))

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, name in enumerate(names):
        per_category = data[name].get("per_category", {})
        tps = [per_category.get(category, {}).get("mean_throughput_tps", 0) for category in categories]
        offset = x + (idx - len(names) / 2 + 0.5) * width
        ax.bar(offset, tps, width, label=_short_label(name), color=_color(data[name]), alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Output Tokens / Second")
    ax.set_title("Throughput by Task Category")
    ax.legend(fontsize=6, ncols=2)
    plt.tight_layout()
    _save(fig, out_dir / "by_category.png")


def plot_difficulty_buckets(data: dict[str, dict], out_dir: Path) -> None:
    bucket_labels = ["very_hard", "hard", "easy", "very_easy"]
    names = list(data)
    x = np.arange(len(bucket_labels))
    width = 0.8 / max(1, len(names))

    fig, ax = plt.subplots(figsize=(11, 5))
    for idx, name in enumerate(names):
        buckets = data[name].get("difficulty_buckets", {})
        values = [buckets.get(label, 0) for label in bucket_labels]
        offset = x + (idx - len(names) / 2 + 0.5) * width
        ax.bar(offset, values, width, label=_short_label(name), color=_color(data[name]), alpha=0.65)

    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels)
    ax.set_ylabel("Mean Token Log-Probability")
    ax.set_title("Verifier Confidence by Token Difficulty Bucket")
    ax.legend(fontsize=6, ncols=2)
    plt.tight_layout()
    _save(fig, out_dir / "difficulty_buckets.png")


def print_summary(data: dict[str, dict]) -> None:
    print("\n" + "=" * 151)
    print(
        f"{'Run':<58} {'Cell':<10} {'Workload':<10} {'k':>2} {'Temp':>5} "
        f"{'AccRate':>8} {'TPS':>8} {'TTFT':>8} {'P95':>8} "
        f"{'IFPrompt':>8} {'IFInst':>8} {'Other':>7}"
    )
    print("-" * 151)
    for name, payload in data.items():
        cfg = _run_config(payload)
        stats = payload.get("stats", {})
        acc = payload.get("acceptance_rate")
        accuracy = payload.get("accuracy", {})
        ifeval = accuracy.get("ifeval", {})
        print(
            f"{name:<58} {cfg.get('cell_id', ''):<10} {cfg.get('workload_id', ''):<10} "
            f"{cfg.get('num_speculative_tokens', 0):>2} {cfg.get('temperature', 0):>5.1f} "
            f"{acc or 0:>8.3f} {stats.get('mean_throughput_tps', 0):>8.1f} "
            f"{stats.get('mean_ttft_ms', 0):>8.1f} {stats.get('p95_ttft_ms', 0):>8.1f} "
            f"{ifeval.get('prompt_accuracy', 0):>8.3f} "
            f"{ifeval.get('instruction_accuracy', 0):>8.3f} "
            f"{accuracy.get('other_accuracy', 0):>7.3f}"
        )
    print("=" * 151)


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
    print(f"\nGenerating charts -> {out_dir}/")
    plot_acceptance_rates(data, out_dir)
    plot_throughput(data, out_dir)
    plot_ttft(data, out_dir)
    plot_task_accuracy(data, out_dir)
    plot_by_workload(data, out_dir)
    plot_by_category(data, out_dir)
    plot_difficulty_buckets(data, out_dir)


def _run_config(payload: dict) -> dict:
    return payload.get("run_config") or payload.get("config") or {}


def _color(payload: dict) -> str:
    cfg = _run_config(payload)
    arch = cfg.get("verifier_architecture") or cfg.get("verifier_type") or ""
    cell = cfg.get("cell_id", "")
    return MOE_COLOR if arch == "moe" or "moe" in cell else DENSE_COLOR


def _short_label(name: str) -> str:
    return name.replace("run_", "r").replace("_", "\n")


def _mean(values: list[float]) -> float:
    clean = [value for value in values if value]
    return float(np.mean(clean)) if clean else 0.0


def _add_arch_legend(ax) -> None:
    patches = [
        mpatches.Patch(color=DENSE_COLOR, label="Dense verifier"),
        mpatches.Patch(color=MOE_COLOR, label="MoE verifier"),
    ]
    ax.legend(handles=patches)


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


if __name__ == "__main__":
    main()
