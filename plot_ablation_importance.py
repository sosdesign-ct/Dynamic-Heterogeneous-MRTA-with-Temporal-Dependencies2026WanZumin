# -*- coding: utf-8 -*-
"""Ablation importance figure (Plan A).

Horizontal bar chart showing, for each ablation variant, the drop in
CP-SAT relative completion vs Full, expressed as a relative percentage.
Variants are sorted by importance (largest drop first). Error bars use
seed-level std propagated as relative values.

Data source: paper Table tab:ablation_completion, backed by
results/eval_ablation_final_bottleneck/ablation_variant_summary.csv.

Output: ablation_analysis.png in the workspace root.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(ROOT, "results/eval_ablation_final_bottleneck/ablation_variant_summary.csv")

# variant key -> display label, ordered by importance (largest drop first)
VARIANTS = [
    ("wo_mask", "w/o mask"),
    ("wo_commitment", "w/o commitment"),
    ("self_only_graph", "self-only graph"),
    ("wo_rgat", "w/o RGAT"),
]


def main():
    df = pd.read_csv(CSV).set_index("variant")
    full = df.loc["full", "cpsat_relative_done_mean_avg"]

    rows = []
    for key, label in VARIANTS:
        if key not in df.index:
            continue
        mean = df.loc[key, "cpsat_relative_done_mean_avg"]
        std = df.loc[key, "cpsat_relative_done_mean_std"]
        drop = (full - mean) / full * 100.0
        drop_err = std / full * 100.0
        rows.append((label, drop, drop_err, mean))

    # sort descending by drop
    rows.sort(key=lambda r: r[1], reverse=True)

    labels = [r[0] for r in rows][::-1]
    drops = np.array([r[1] for r in rows][::-1])
    errs = np.array([r[2] for r in rows][::-1])
    means = np.array([r[3] for r in rows][::-1])

    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"][:len(rows)][::-1]

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    bars = ax.barh(labels, drops, xerr=errs, color=colors, alpha=0.85,
                   capsize=4, error_kw=dict(lw=1.2), height=0.55)

    # annotate completion value at the end of each bar
    for bar, mean in zip(bars, means):
        w = bar.get_width()
        ax.text(w + 1.0, bar.get_y() + bar.get_height() / 2,
                f"completion = {mean:.3f}", va="center", fontsize=8.5, color="dimgray")

    ax.set_xlabel("Relative drop in CP-SAT completion vs Full (%)")
    ax.set_title("Ablation importance: performance drop when each component is removed")
    ax.set_xlim(0, max(drops + errs) * 1.25)
    ax.grid(axis="x", alpha=0.3, ls="--")
    fig.tight_layout()
    out = os.path.join(ROOT, "ablation_analysis.png")
    fig.savefig(out, dpi=220)
    print("saved:", out)


if __name__ == "__main__":
    main()
