"""Plot per-scale comparison curves for all methods on the scale suite.

Methods: ours (3-seed mean +/- std), mappo, cbs_ta_ptc (2024 fixed), ga, ctas_d.
Reads results/scale_suite/eval/<method>_<scale>/episode_metrics.csv.
Outputs PNG (and PDF) figures into results/scale_suite/plots/scale_compare/.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
EVAL = ROOT / "results" / "scale_suite" / "eval"
OUT = ROOT / "results" / "scale_suite" / "plots" / "scale_compare"
OUT.mkdir(parents=True, exist_ok=True)

SCALES = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]
NUM_AGENTS = {"s1": 12, "s2": 16, "s3": 20, "s4": 24, "s5": 28, "s6": 32, "s7": 36, "s8": 40}
SEEDS = [1, 2, 3]

# method -> (display label, eval-dir resolver, color, marker, linestyle)
METHODS = {
    "ours": ("Ours", lambda s: f"ours_{s}_seed{{seed}}", "#d62728", "o", "-"),
    "mappo": ("MAPPO", lambda s: f"mappo_{s}", "#1f77b4", "s", "-"),
    "cbs": ("CBS-TA-PTC", lambda s: f"cbs_ta_ptc_{s}", "#2ca02c", "^", "--"),
    "ga": ("GA", lambda s: f"ga_{s}", "#ff7f0e", "D", "--"),
    "ctas_d": ("CTAS-D", lambda s: f"ctas_d_{s}", "#9467bd", "v", ":"),
}

METRICS = [
    ("return", "Return", False),
    ("completed_tasks", "Completed Tasks", False),
    ("cpsat_relative_done", "Oracle-Relative Completion", False),
    ("success_rate", "Success Rate", False),
    ("makespan", "Makespan", False),
    ("generalized_makespan", "Generalized Makespan", False),
    ("oracle_done_gap", "Oracle Done Gap (lower=better)", False),
]


def load_episodes(tag: str) -> Optional[List[Dict[str, str]]]:
    f = EVAL / tag / "episode_metrics.csv"
    if not f.is_file():
        return None
    with f.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def mean_of(rows: List[Dict[str, str]], key: str) -> Optional[float]:
    vals = []
    for r in rows:
        v = r.get(key)
        if v in ("", "nan", "None", None):
            continue
        try:
            vals.append(float(v))
        except Exception:
            pass
    return float(np.mean(vals)) if vals else None


def collect(metric: str):
    """Return dict method -> (mean list over scales, std list over scales)."""
    data: Dict[str, Dict[str, List[Optional[float]]]] = {}
    for mkey, (_, resolver, *_rest) in METHODS.items():
        means: List[Optional[float]] = []
        stds: List[Optional[float]] = []
        for s in SCALES:
            if mkey == "ours":
                per_seed = []
                for sd in SEEDS:
                    rows = load_episodes(resolver(s).format(seed=sd))
                    if rows is None:
                        continue
                    mv = mean_of(rows, metric)
                    if mv is not None:
                        per_seed.append(mv)
                if per_seed:
                    means.append(float(np.mean(per_seed)))
                    stds.append(float(np.std(per_seed)))
                else:
                    means.append(None)
                    stds.append(None)
            else:
                rows = load_episodes(resolver(s))
                mv = mean_of(rows, metric) if rows else None
                means.append(mv)
                stds.append(None)
        data[mkey] = {"mean": means, "std": stds}
    return data


def plot_metric(metric: str, ylabel: str, _flag: bool):
    data = collect(metric)
    x = np.array([NUM_AGENTS[s] for s in SCALES], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    any_data = False
    for mkey, (label, _r, color, marker, ls) in METHODS.items():
        ys = np.array([np.nan if v is None else v for v in data[mkey]["mean"]], dtype=float)
        if np.all(np.isnan(ys)):
            continue
        any_data = True
        sd = data[mkey]["std"]
        ax.plot(x, ys, label=label, color=color, marker=marker, linestyle=ls,
                linewidth=2.0, markersize=6)
        if mkey == "ours" and any(v is not None for v in sd):
            yerr = np.array([0.0 if v is None else v for v in sd], dtype=float)
            ax.fill_between(x, ys - yerr, ys + yerr, color=color, alpha=0.15)
    if not any_data:
        plt.close(fig)
        print(f"[skip] no data for {metric}")
        return
    ax.set_xlabel("Number of Agents (scale)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v)}\n({s})" for v, s in zip(x, SCALES)], fontsize=9)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(fontsize=9, loc="best", framealpha=0.9)
    ax.set_title(f"{ylabel} vs Scale", fontsize=12)
    fig.tight_layout()
    png = OUT / f"scale_{metric}.png"
    pdf = OUT / f"scale_{metric}.pdf"
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"[ok] {png.name}")


def plot_combined():
    """3x3 grid combining all metrics (7 metrics + legend + 1 empty)."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 12.5))
    x = np.array([NUM_AGENTS[s] for s in SCALES], dtype=float)
    for idx, (metric, ylabel, _f) in enumerate(METRICS):
        ax = axes.flat[idx]
        data = collect(metric)
        for mkey, (label, _r, color, marker, ls) in METHODS.items():
            ys = np.array([np.nan if v is None else v for v in data[mkey]["mean"]], dtype=float)
            if np.all(np.isnan(ys)):
                continue
            ax.plot(x, ys, label=label, color=color, marker=marker, linestyle=ls,
                    linewidth=1.8, markersize=5)
            if mkey == "ours":
                sd = data[mkey]["std"]
                if any(v is not None for v in sd):
                    yerr = np.array([0.0 if v is None else v for v in sd], dtype=float)
                    ax.fill_between(x, ys - yerr, ys + yerr, color=color, alpha=0.12)
        ax.set_xlabel("Agents", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(v)}" for v in x], fontsize=8)
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.tick_params(labelsize=8)
    # legend in the cell right after the last metric, hide remaining cells
    legend_ax = axes.flat[len(METRICS)]
    for j in range(len(METRICS), len(axes.flat)):
        axes.flat[j].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc="center", fontsize=12, frameon=False)
    fig.suptitle("Method Comparison across Scales (s1=12 agents ... s8=40 agents)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    png = OUT / "scale_compare_all.png"
    pdf = OUT / "scale_compare_all.pdf"
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"[ok] {png.name}")


def main():
    for metric, ylabel, flag in METRICS:
        plot_metric(metric, ylabel, flag)
    plot_combined()
    print(f"\nAll figures -> {OUT}")


if __name__ == "__main__":
    main()
