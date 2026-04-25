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


def plot_by_workload(data: dict[str, dict], out_dir: Path) -> None:
    workloads = sorted({_run_config(d).get("workload_id", "unknown") for d in data.values()})
    cells = sorted({_run_config(d).get("cell_id", "unknown") for d in data.values()})
    labels = [f"{cell}\n{workload}" for cell in cells for workload in workloads]
    values = []
    colors = []

    for cell in cells:
        for workload in workloads:
            matching = [
                d
                for d in data.values()
                if _run_config(d).get("cell_id") == cell
                and _run_config(d).get("workload_id") == workload
            ]
            values.append(_mean([d.get("stats", {}).get("mean_throughput_tps", 0) for d in matching]))
            colors.append(DENSE_COLOR if "dense" in cell else MOE_COLOR)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Output Tokens / Second")
    ax.set_title("Mean Throughput by Cell and Workload")
    _add_arch_legend(ax)
    plt.tight_layout()
    _save(fig, out_dir / "by_workload.png")


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
    print("\n" + "=" * 118)
    print(
        f"{'Run':<58} {'Cell':<10} {'Workload':<10} {'k':>2} "
        f"{'Temp':>5} {'AccRate':>8} {'TPS':>8} {'TTFT':>8} {'P95':>8}"
    )
    print("-" * 118)
    for name, payload in data.items():
        cfg = _run_config(payload)
        stats = payload.get("stats", {})
        acc = payload.get("acceptance_rate")
        print(
            f"{name:<58} {cfg.get('cell_id', ''):<10} {cfg.get('workload_id', ''):<10} "
            f"{cfg.get('num_speculative_tokens', 0):>2} {cfg.get('temperature', 0):>5.1f} "
            f"{acc or 0:>8.3f} {stats.get('mean_throughput_tps', 0):>8.1f} "
            f"{stats.get('mean_ttft_ms', 0):>8.1f} {stats.get('p95_ttft_ms', 0):>8.1f}"
        )
    print("=" * 118)


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
    plot_by_workload(data, out_dir)
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
