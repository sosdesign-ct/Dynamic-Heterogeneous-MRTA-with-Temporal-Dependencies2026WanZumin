# -*- coding: utf-8 -*-
"""Value-loss convergence: Ours vs MAPPO (3 seeds each).

Reads the 6 tensorboard event files listed in loss_step_sources.csv
and plots optimization/value_loss against environment steps.
MAPPO is trained for 2e6 steps; we truncate at 1e6 (Ours budget)
and mark the truncation with a vertical dashed line.

Output: value_loss_convergence.png in the workspace root.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

ROOT = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(ROOT, "loss_step_sources.csv")
TAG = "optimization/value_loss"
OURS_BUDGET = 1.0e6  # Ours 训练预算
MAPPO_BUDGET = 2.0e6  # MAPPO 训练预算


def load_curve(event_file, max_step):
    ea = event_accumulator.EventAccumulator(event_file, size_guidance={"scalars": 0})
    ea.Reload()
    # event step -> 真实环境步数 的映射
    prog = ea.Scalars("progress/step")
    step_map = {int(s.step): float(s.value) for s in prog}
    sc = ea.Scalars(TAG)
    steps = np.array([step_map.get(int(s.step), float(s.step)) for s in sc], dtype=float)
    vals = np.array([s.value for s in sc], dtype=float)
    # 只保留 max_step 以内的数据
    m = steps <= max_step
    return steps[m], vals[m]


def smooth(v, k=5):
    if len(v) < k:
        return v
    ker = np.ones(k) / k
    return np.convolve(v, ker, mode="same")


def main():
    series = []  # (family, seed_label, steps, vals)
    with open(CSV) as f:
        for row in csv.DictReader(f):
            fam = "Ours" if row["method"].startswith("Ours") else "MAPPO"
            max_step = OURS_BUDGET if fam == "Ours" else MAPPO_BUDGET
            s, v = load_curve(row["event_file"], max_step)
            series.append((fam, row["method"], s, v))

    colors = {"Ours": "#1f77b4", "MAPPO": "#d62728"}
    alphas = [0.55, 0.7, 0.85]

    fig, ax = plt.subplots(figsize=(7.5, 4.0))

    all_y = []
    for fam in ("Ours", "MAPPO"):
        fam_series = [x for x in series if x[0] == fam]
        for i, (_, label, s, v) in enumerate(sorted(fam_series, key=lambda x: x[1])):
            x, y = s, smooth(v, 5)
            all_y.append(y)
            ax.plot(x / 1e6, y, color=colors[fam], alpha=alphas[i % 3], lw=1.4,
                    label=f"{fam} (3 seeds)" if i == 0 else None)

    # 标记 Ours 训练预算
    ax.axvline(OURS_BUDGET / 1e6, color="gray", ls="--", lw=1.2, alpha=0.8)
    ax.text(OURS_BUDGET / 1e6, ax.get_ylim()[1] * 0.02, "Ours budget ($1\\times10^6$ steps)",
            rotation=90, va="bottom", ha="right", fontsize=9, color="gray")

    # 让 y 轴聚焦在数据集中区域，忽略初期的高峰值
    y_concat = np.concatenate(all_y)
    y_max = np.percentile(y_concat, 60) * 3.0
    ax.set_ylim(bottom=0, top=y_max)
    ax.set_xlim(0, MAPPO_BUDGET / 1e6)
    ax.set_xlabel("Environment Steps ($\\times 10^6$)", fontsize=11)
    ax.set_ylabel("Value Loss", fontsize=11)
    ax.set_title("Value-Function Loss Convergence", fontsize=12)
    ax.grid(alpha=0.3, ls="--")
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    ax.tick_params(axis="both", which="major", labelsize=10)
    fig.tight_layout()
    out = os.path.join(ROOT, "value_loss_convergence.png")
    fig.savefig(out, dpi=220)
    print("saved:", out)


if __name__ == "__main__":
    main()
