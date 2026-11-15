from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


DEFAULT_TAGS = [
    ("optimization/policy_loss", "Policy Loss"),
    ("optimization/value_loss", "Value Loss"),
    ("optimization/total_loss", "Total Loss"),
    ("optimization/approx_kl", "Approx. KL"),
]


def latest_event_file(run_dir: Path) -> Path:
    files = sorted(run_dir.glob("**/events.out.tfevents*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"no TensorBoard event file found under {run_dir}")
    return files[0]


def read_scalars(event_file: Path) -> Tuple[Dict[str, pd.DataFrame], Dict[int, float]]:
    acc = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    acc.Reload()
    scalars: Dict[str, pd.DataFrame] = {}
    for tag in acc.Tags().get("scalars", []):
        events = acc.Scalars(tag)
        scalars[tag] = pd.DataFrame(
            {
                "event_step": [int(e.step) for e in events],
                "wall_time": [float(e.wall_time) for e in events],
                "value": [float(e.value) for e in events],
            }
        )
    step_map: Dict[int, float] = {}
    if "progress/step" in scalars:
        prog = scalars["progress/step"]
        step_map = {int(r.event_step): float(r.value) for r in prog.itertuples(index=False)}
    return scalars, step_map


def smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) == 0:
        return y
    window = min(int(window), len(y))
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(y, kernel, mode="same")


def tag_frame(scalars: Dict[str, pd.DataFrame], step_map: Dict[int, float], tag: str) -> pd.DataFrame:
    if tag not in scalars:
        return pd.DataFrame(columns=["train_step", "value"])
    df = scalars[tag].copy()
    df["train_step"] = df["event_step"].map(step_map)
    if df["train_step"].isna().any():
        df["train_step"] = df["train_step"].fillna(df["event_step"])
    return df.loc[:, ["train_step", "value"]].sort_values("train_step")


def parse_method(items: Iterable[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"invalid method mapping: {item}; expected NAME=RUN_DIR")
        name, raw = item.split("=", 1)
        out[name.strip()] = Path(raw).expanduser().resolve()
    if not out:
        raise ValueError("at least one --method NAME=RUN_DIR is required")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot loss-vs-training-step convergence curves from TensorBoard events.")
    parser.add_argument("--method", nargs="+", required=True, help="method mapping: NAME=RUN_DIR")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--smooth_window", type=int, default=3)
    parser.add_argument("--max_step", type=float, default=0.0, help="optional x-axis cutoff; <=0 keeps all data")
    args = parser.parse_args()

    methods = parse_method(args.method)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = {}
    summary = []
    for name, run_dir in methods.items():
        event_file = latest_event_file(run_dir)
        scalars, step_map = read_scalars(event_file)
        loaded[name] = (scalars, step_map, event_file)
        last_step = max(step_map.values()) if step_map else float("nan")
        summary.append({"method": name, "event_file": str(event_file), "last_train_step": last_step})

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.2))
    axes = axes.ravel()
    for ax, (tag, label) in zip(axes, DEFAULT_TAGS):
        plotted = 0
        for name, (scalars, step_map, _event_file) in loaded.items():
            df = tag_frame(scalars, step_map, tag)
            if df.empty:
                continue
            if float(args.max_step) > 0:
                df = df[df["train_step"] <= float(args.max_step)]
            x = df["train_step"].to_numpy(dtype=np.float64)
            y = smooth(df["value"].to_numpy(dtype=np.float64), int(args.smooth_window))
            ax.plot(x, y, linewidth=2.0, label=name)
            plotted += 1
        ax.set_title(label)
        ax.set_xlabel("Training Step")
        ax.set_ylabel(label)
        ax.grid(True, linestyle="--", alpha=0.3)
        if plotted:
            ax.legend()
    fig.suptitle("Loss-Step Convergence Curves")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_dir / "loss_step_convergence.png", dpi=220)
    plt.close(fig)

    pd.DataFrame(summary).to_csv(output_dir / "loss_step_sources.csv", index=False)
    (output_dir / "loss_step_sources.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
