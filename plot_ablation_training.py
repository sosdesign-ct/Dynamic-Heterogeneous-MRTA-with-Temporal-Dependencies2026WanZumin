import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


UPDATE_RE = re.compile(
    r"update=(?P<update>\d+)\s+step=(?P<step>\d+)\s+mean_return=(?P<mean_return>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+makespan=(?P<makespan>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)
EPISODE_RE = re.compile(
    r"episode=(?P<episode>\d+)\s+step=(?P<step>\d+)\s+episode_return=(?P<episode_return>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)


def parse_mapping_items(items: List[str], kind: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"invalid {kind}: {item}. expected format METHOD=PATH")
        method, path = item.split("=", 1)
        method = method.strip()
        path = path.strip()
        if not method or not path:
            raise ValueError(f"invalid {kind}: {item}. METHOD and PATH must be non-empty")
        out[method] = path
    return out


def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or y.size == 0:
        return y
    w = int(window)
    if w > y.size:
        w = y.size
    kernel = np.ones(w, dtype=np.float64) / float(w)
    return np.convolve(y, kernel, mode="same")


def parse_log_file(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    updates = []
    episodes = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m_upd = UPDATE_RE.search(line)
            if m_upd:
                updates.append(
                    {
                        "update": int(m_upd.group("update")),
                        "step": int(m_upd.group("step")),
                        "mean_return": float(m_upd.group("mean_return")),
                        "makespan": float(m_upd.group("makespan")),
                    }
                )

            m_ep = EPISODE_RE.search(line)
            if m_ep:
                episodes.append(
                    {
                        "episode": int(m_ep.group("episode")),
                        "step": int(m_ep.group("step")),
                        "episode_return": float(m_ep.group("episode_return")),
                    }
                )

    return pd.DataFrame(updates), pd.DataFrame(episodes)


def read_update_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"step", "makespan"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} must contain columns: {sorted(required)}")

    if "update" not in df.columns:
        df["update"] = np.arange(1, len(df) + 1)
    if "mean_return" not in df.columns:
        df["mean_return"] = np.nan

    cols = ["update", "step", "mean_return", "makespan"]
    return df.loc[:, cols].copy()


def read_episode_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"episode", "episode_return"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} must contain columns: {sorted(required)}")

    if "step" not in df.columns:
        df["step"] = np.nan

    cols = ["episode", "step", "episode_return"]
    return df.loc[:, cols].copy()


def collect_method_data(
    method_logs: Dict[str, str],
    method_update_csv: Dict[str, str],
    method_episode_csv: Dict[str, str],
) -> Dict[str, Dict[str, pd.DataFrame]]:
    method_names = set(method_logs.keys()) | set(method_update_csv.keys()) | set(method_episode_csv.keys())
    if not method_names:
        raise ValueError("no method input provided")

    data: Dict[str, Dict[str, pd.DataFrame]] = {}
    for method in sorted(method_names):
        updates = pd.DataFrame(columns=["update", "step", "mean_return", "makespan"])
        episodes = pd.DataFrame(columns=["episode", "step", "episode_return"])

        if method in method_logs:
            p = Path(method_logs[method])
            if not p.exists():
                raise FileNotFoundError(f"log file not found: {p}")
            log_updates, log_episodes = parse_log_file(p)
            updates = pd.concat([updates, log_updates], ignore_index=True)
            episodes = pd.concat([episodes, log_episodes], ignore_index=True)

        if method in method_update_csv:
            p = Path(method_update_csv[method])
            if not p.exists():
                raise FileNotFoundError(f"update csv not found: {p}")
            updates = pd.concat([updates, read_update_csv(p)], ignore_index=True)

        if method in method_episode_csv:
            p = Path(method_episode_csv[method])
            if not p.exists():
                raise FileNotFoundError(f"episode csv not found: {p}")
            episodes = pd.concat([episodes, read_episode_csv(p)], ignore_index=True)

        if len(updates) > 0:
            updates = updates.drop_duplicates(subset=["update", "step", "makespan"], keep="last")
            updates = updates.sort_values(["step", "update"]).reset_index(drop=True)

        if len(episodes) > 0:
            episodes = episodes.drop_duplicates(subset=["episode", "step", "episode_return"], keep="last")
            episodes = episodes.sort_values(["episode", "step"]).reset_index(drop=True)

        data[method] = {"updates": updates, "episodes": episodes}

    return data


def plot_training_curves(data: Dict[str, Dict[str, pd.DataFrame]], output_dir: Path, smooth_window: int):
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.0))
    ax_mk, ax_ret = axes

    plotted_mk = 0
    plotted_ret = 0
    for method, dd in data.items():
        up = dd["updates"]
        if up.empty:
            continue

        x = up["step"].to_numpy(dtype=np.float64)
        y = up["makespan"].to_numpy(dtype=np.float64)
        y_s = moving_average(y, smooth_window)
        ax_mk.plot(x, y_s, linewidth=2.0, label=method)
        plotted_mk += 1

        if up["mean_return"].notna().any():
            yr = up["mean_return"].to_numpy(dtype=np.float64)
            yr_s = moving_average(yr, smooth_window)
            ax_ret.plot(up["update"].to_numpy(dtype=np.float64), yr_s, linewidth=2.0, label=method)
            plotted_ret += 1

    if plotted_mk == 0 and plotted_ret == 0:
        raise RuntimeError("no training-curve data found")

    ax_mk.set_title("Makespan During Optimization")
    ax_mk.set_xlabel("Training Step")
    ax_mk.set_ylabel("Makespan")
    ax_mk.grid(True, linestyle="--", alpha=0.35)
    if plotted_mk > 0:
        ax_mk.legend()

    ax_ret.set_title("Mean Return During Optimization")
    ax_ret.set_xlabel("Update")
    ax_ret.set_ylabel("Mean Return")
    ax_ret.grid(True, linestyle="--", alpha=0.35)
    if plotted_ret > 0:
        ax_ret.legend()

    out_path = output_dir / "C_training_curves.png"
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)


def plot_episode_reward(data: Dict[str, Dict[str, pd.DataFrame]], output_dir: Path, smooth_window: int):
    fig, ax = plt.subplots(figsize=(9.0, 5.0))

    plotted = 0
    used_fallback = False

    for method, dd in data.items():
        ep = dd["episodes"]
        if not ep.empty:
            x = ep["episode"].to_numpy(dtype=np.float64)
            y = ep["episode_return"].to_numpy(dtype=np.float64)
            y_s = moving_average(y, smooth_window)
            ax.plot(x, y_s, linewidth=2.0, label=method)
            plotted += 1
            continue

        up = dd["updates"]
        if not up.empty and up["mean_return"].notna().any():
            x = up["update"].to_numpy(dtype=np.float64)
            y = up["mean_return"].to_numpy(dtype=np.float64)
            y_s = moving_average(y, smooth_window)
            ax.plot(x, y_s, linewidth=2.0, linestyle="--", label=f"{method} (mean_return)")
            plotted += 1
            used_fallback = True

    if plotted == 0:
        raise RuntimeError(
            "no episode reward data found. provide episode logs with 'episode=... episode_return=...' or episode csv"
        )

    ax.set_title("Episode Reward")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Reward Per Episode")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    out_path = output_dir / "C_episode_reward.png"
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)

    if used_fallback:
        print("warning: some methods have no episode-level data, fallback to update-level mean_return.")


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot C figure training curves and optional episode reward curves."
    )
    p.add_argument(
        "--method_log",
        type=str,
        nargs="*",
        default=[],
        help="Method log files: METHOD=PATH. log should contain update lines and optionally episode lines.",
    )
    p.add_argument(
        "--method_update_csv",
        type=str,
        nargs="*",
        default=[],
        help="Method update csv: METHOD=PATH. required columns: step,makespan; optional: update,mean_return",
    )
    p.add_argument(
        "--method_episode_csv",
        type=str,
        nargs="*",
        default=[],
        help="Method episode csv: METHOD=PATH. required columns: episode,episode_return; optional: step",
    )
    p.add_argument("--smooth_window", type=int, default=5, help="Moving average window size")
    p.add_argument("--output_dir", type=str, default="model/vis_train")
    return p.parse_args()


def main():
    args = parse_args()

    method_logs = parse_mapping_items(args.method_log, "method_log")
    method_update_csv = parse_mapping_items(args.method_update_csv, "method_update_csv")
    method_episode_csv = parse_mapping_items(args.method_episode_csv, "method_episode_csv")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = collect_method_data(method_logs, method_update_csv, method_episode_csv)
    plot_training_curves(data, output_dir, args.smooth_window)
    plot_episode_reward(data, output_dir, args.smooth_window)

    print(f"saved: {output_dir / 'C_training_curves.png'}")
    print(f"saved: {output_dir / 'C_episode_reward.png'}")


if __name__ == "__main__":
    main()
