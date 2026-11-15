# -*- coding: utf-8 -*-
"""Seed-stability figure: Ours vs MAPPO on the 229-scene paper_fair benchmark.

Panel A: training mean_return curves (3 seeds each) -> training behaviour.
Panel B/C/D: per-seed EVAL metrics (return / recognized_done_rate /
cpsat_relative_done) -> Ours clusters tight, MAPPO shows seed-level drop.

Outputs PNG + PDF under --output_dir.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

EVAL_METRICS = [
    ("return", "Return"),
    ("recognized_done_rate", "Recognized Completion"),
    ("cpsat_relative_done", "CP-SAT Relative Completion"),
]


def load_train_curve(event_file, tag="performance/mean_return"):
    ea = event_accumulator.EventAccumulator(event_file, size_guidance={"scalars": 0})
    ea.Reload()
    sc = ea.Scalars(tag)
    steps = np.array([s.step for s in sc], dtype=float)
    vals = np.array([s.value for s in sc], dtype=float)
    return steps, vals


def smooth(v, k=9):
    if len(v) < k:
        return v
    ker = np.ones(k) / k
    return np.convolve(v, ker, mode="same")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss_sources", default="loss_step_sources.csv")
    ap.add_argument("--eval_base", default="results/paper_fair_d020_080_strict_v2_fixed_eval_fix")
    ap.add_argument("--output_dir", default="results/paper_fair_d020_080_strict_v2_fixed_eval_fix/seed_stability")
    ap.add_argument("--smooth", type=int, default=9)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- training curves ----
    train = {}  # method -> list of (steps, vals)
    with open(args.loss_sources) as f:
        for row in csv.DictReader(f):
            fam = "Ours" if row["method"].startswith("Ours") else "MAPPO"
            s, v = load_train_curve(row["event_file"])
            train.setdefault(fam, []).append((s, v))

    # ---- eval per-seed metrics ----
    evals = {}  # fam -> {metric: [seed1,seed2,seed3]}
    for fam_key, fam in [("ours", "Ours"), ("mappo", "MAPPO")]:
        evals[fam] = {m[0]: [] for m in EVAL_METRICS}
        for seed in (1, 2, 3):
            p = os.path.join(args.eval_base, f"eval_{fam_key}", f"full_seed{seed}", "episode_metrics.csv")
            df = pd.read_csv(p)
            for m, _ in EVAL_METRICS:
                evals[fam][m].append(float(df[m].mean()) if m in df.columns else np.nan)

    colors = {"Ours": "#1f77b4", "MAPPO": "#d62728"}

    fig = plt.figure(figsize=(15, 4.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.5, 1, 1, 1], wspace=0.28)

    # Panel A: training curves (x = log index as training progress)
    axA = fig.add_subplot(gs[0, 0])
    for fam in ("Ours", "MAPPO"):
        for i, (s, v) in enumerate(sorted(train[fam], key=lambda x: len(x[1]), reverse=True)):
            x = np.linspace(0.0, 100.0, len(v))  # normalize to % of training
            axA.plot(x, smooth(v, args.smooth), color=colors[fam], alpha=0.55,
                     lw=1.3, label=fam if i == 0 else None)
    axA.set_xlim(0, 100)
    axA.set_xlabel("Training Progress (%)")
    axA.set_ylabel("Mean Return (train)")
    axA.set_title("A. Training Curves (3 seeds each)")
    axA.grid(alpha=0.3)
    axA.legend(frameon=False, loc="lower right")

    # Panels B-D: eval metrics per seed
    for j, (m, label) in enumerate(EVAL_METRICS):
        ax = fig.add_subplot(gs[0, j + 1])
        x = np.arange(2)
        means = [np.nanmean(evals["Ours"][m]), np.nanmean(evals["MAPPO"][m])]
        stds = [np.nanstd(evals["Ours"][m]), np.nanstd(evals["MAPPO"][m])]
        ax.bar(x, means, yerr=stds, capsize=4, color=[colors["Ours"], colors["MAPPO"]],
               alpha=0.75, width=0.6, error_kw=dict(lw=1.2))
        # scatter individual seeds
        for xi, fam in [(0, "Ours"), (1, "MAPPO")]:
            ys = evals[fam][m]
            ax.scatter(np.full(len(ys), xi) + np.linspace(-0.12, 0.12, len(ys)), ys,
                       color="black", s=18, zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels(["Ours", "MAPPO"])
        ax.set_title(f"{chr(66+j)}. {label}")
        ax.grid(axis="y", alpha=0.3)
        # annotate per-seed spread (CV-like)
        for xi, fam in [(0, "Ours"), (1, "MAPPO")]:
            ys = np.array(evals[fam][m])
            spread = (np.nanmax(ys) - np.nanmin(ys))
            ax.annotate(f"spread={spread:.3f}", (xi, np.nanmax(ys)),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=7, color="dimgray")

    fig.suptitle("Seed Stability: Ours vs MAPPO (229-scene benchmark)", y=1.02, fontsize=13)
    png = os.path.join(args.output_dir, "seed_stability.png")
    pdf = os.path.join(args.output_dir, "seed_stability.pdf")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print("saved:", png)
    print("saved:", pdf)

    # dump a small summary
    for m, label in EVAL_METRICS:
        o = np.array(evals["Ours"][m]); p = np.array(evals["MAPPO"][m])
        print(f"{label:32s} Ours mean={o.mean():.4f} std={o.std():.4f} | "
              f"MAPPO mean={p.mean():.4f} std={p.std():.4f}")


if __name__ == "__main__":
    main()
