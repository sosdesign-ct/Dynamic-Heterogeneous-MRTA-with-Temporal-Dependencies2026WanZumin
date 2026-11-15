from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_METRICS = [
    "return",
    "recognized_done_rate",
    "cpsat_relative_done",
    "infer_time",
]

DEFAULT_DIFFICULTY_GROUPS = [
    ("D0.20-0.30", 0.20, 0.30),
    ("D0.30-0.40", 0.30, 0.40),
    ("D0.40-0.50", 0.40, 0.50),
    ("D0.50-0.60", 0.50, 0.60),
    ("D0.60-0.70", 0.60, 0.70),
    ("D0.70-0.80", 0.70, 0.81),
]

METRIC_LABELS = {
    "cpsat_relative_done": "CP-SAT Relative Completion",
    "completion": "Completion (Violation-aware)",
    "completion_raw": "Completion (Raw)",
    "recognized_done_tasks": "Recognized Done Tasks",
    "recognized_done_rate": "Recognized Done Rate",
    "violation_rate": "Violation Rate",
    "throughput": "Throughput",
    "infer_time": "Decision Runtime (s)",
    "runtime": "Decision Runtime (s)",
    "makespan": "Makespan",
    "generalized_makespan": "Generalized Makespan",
    "return": "Return",
}


def parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_input_item(item: str) -> Tuple[Optional[str], Path]:
    s = str(item).strip()
    if not s:
        raise ValueError("empty input item")
    if "=" in s:
        label, raw = s.split("=", 1)
        label = label.strip()
        raw = raw.strip()
        if not label or not raw:
            raise ValueError(f"invalid input item: {item}")
        return label, Path(raw)
    return None, Path(s)


def resolve_csv(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        p = path / "episode_metrics.csv"
        if p.is_file():
            return p
    raise FileNotFoundError(f"episode metrics csv not found: {path}")


def load_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def parse_float(v: object) -> float:
    try:
        return float(str(v))
    except Exception:
        return float("nan")


def finite_stats(values: Sequence[float]) -> Tuple[float, float, float, int]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    std = float(np.std(arr))
    sem = float(std / np.sqrt(float(arr.size)))
    return float(np.mean(arr)), std, sem, int(arr.size)


def scene_aggregated_values(rows: Sequence[Dict[str, object]], metric: str) -> List[float]:
    by_scene: Dict[str, List[float]] = {}
    for row in rows:
        scene_key = str(row.get("scene_path", "")).strip()
        if not scene_key:
            scene_key = str(row.get("scene_idx", "")).strip()
        value = parse_float(row.get(metric, "nan"))
        if scene_key and np.isfinite(value):
            by_scene.setdefault(scene_key, []).append(value)
    return [float(np.mean(values)) for values in by_scene.values() if values]


def replicate_aggregated_values(rows: Sequence[Dict[str, object]], metric: str) -> List[float]:
    by_replicate: Dict[str, List[float]] = {}
    for row in rows:
        replicate = str(row.get("_replicate", "")).strip()
        value = parse_float(row.get(metric, "nan"))
        if replicate and np.isfinite(value):
            by_replicate.setdefault(replicate, []).append(value)
    return [float(np.mean(values)) for values in by_replicate.values() if values]


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(str(metric), str(metric))


def enrich_derived_metrics(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows:
        rr: Dict[str, object] = dict(row)

        infer_time = parse_float(rr.get("infer_time", "nan"))
        runtime = parse_float(rr.get("runtime", "nan"))
        if not np.isfinite(infer_time) and np.isfinite(runtime):
            rr["infer_time"] = runtime
        elif np.isfinite(infer_time) and not np.isfinite(runtime):
            rr["runtime"] = infer_time

        recognized_done = parse_float(rr.get("recognized_done_tasks", "nan"))
        completed_tasks = parse_float(rr.get("completed_tasks", "nan"))
        if not np.isfinite(recognized_done) and np.isfinite(completed_tasks):
            rr["recognized_done_tasks"] = completed_tasks

        completion_raw = parse_float(rr.get("completion_raw", "nan"))
        recognized_done_rate = parse_float(rr.get("recognized_done_rate", "nan"))
        success_rate = parse_float(rr.get("success_rate", "nan"))
        if not np.isfinite(completion_raw):
            if np.isfinite(recognized_done_rate):
                completion_raw = recognized_done_rate
            elif np.isfinite(success_rate):
                completion_raw = success_rate
            if np.isfinite(completion_raw):
                rr["completion_raw"] = completion_raw

        completion = parse_float(rr.get("completion", "nan"))
        violation_rate_for_completion = parse_float(rr.get("violation_rate", "nan"))
        if not np.isfinite(completion) and np.isfinite(completion_raw):
            penalty = 1.0 - float(np.clip(violation_rate_for_completion, 0.0, 1.0)) if np.isfinite(violation_rate_for_completion) else 1.0
            rr["completion"] = completion_raw * penalty

        cpsat_relative_done = parse_float(rr.get("cpsat_relative_done", "nan"))
        oracle_done = parse_float(rr.get("oracle_max_recognized_done_tasks", "nan"))
        recognized_done = parse_float(rr.get("recognized_done_tasks", "nan"))
        if not np.isfinite(cpsat_relative_done) and np.isfinite(recognized_done) and np.isfinite(oracle_done) and oracle_done > 0:
            rr["cpsat_relative_done"] = recognized_done / oracle_done

        strict_success = parse_float(rr.get("strict_success", "nan"))
        if not np.isfinite(strict_success):
            completion = parse_float(rr.get("completion", "nan"))
            truncated = parse_float(rr.get("truncated", "nan"))
            violation_rate = parse_float(rr.get("violation_rate", "nan"))
            if np.isfinite(completion) and np.isfinite(truncated) and np.isfinite(violation_rate):
                strict_success = 1.0 if (completion >= 0.999999 and truncated < 0.5 and violation_rate <= 1e-12) else 0.0
            else:
                strict_success = float("nan")
        rr["strict_success"] = strict_success

        strict_completion = parse_float(rr.get("strict_completion", "nan"))
        if not np.isfinite(strict_completion):
            completed_tasks = parse_float(rr.get("completed_tasks", "nan"))
            truncated = parse_float(rr.get("truncated", "nan"))
            violation_rate = parse_float(rr.get("violation_rate", "nan"))

            # Artifact rows do not record which specific completed tasks were violation-free.
            # Use an episode-level strict count: any truncation or any recorded violation zeroes the count.
            if np.isfinite(completed_tasks) and np.isfinite(truncated) and np.isfinite(violation_rate):
                strict_completion = completed_tasks if (truncated < 0.5 and violation_rate <= 1e-12) else 0.0
            else:
                strict_completion = float("nan")
        rr["strict_completion"] = strict_completion
        out.append(rr)
    return out


def aggregate_exact(xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    uniq = np.unique(xs)
    xo: List[float] = []
    yo: List[float] = []
    for u in uniq:
        mask = xs == u
        vals = ys[mask]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        xo.append(float(u))
        yo.append(float(np.mean(vals)))
    return np.asarray(xo, dtype=np.float64), np.asarray(yo, dtype=np.float64)


def aggregate_binned(xs: np.ndarray, ys: np.ndarray, bins: int) -> Tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[finite]
    ys = ys[finite]
    if xs.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    lo = float(np.min(xs))
    hi = float(np.max(xs))
    if abs(hi - lo) < 1e-12:
        return np.array([lo], dtype=np.float64), np.array([float(np.mean(ys))], dtype=np.float64)

    edges = np.linspace(lo, hi, max(2, int(bins) + 1))
    inds = np.digitize(xs, edges, right=False) - 1
    inds = np.clip(inds, 0, len(edges) - 2)

    xo: List[float] = []
    yo: List[float] = []
    for bi in range(len(edges) - 1):
        mask = inds == bi
        if not np.any(mask):
            continue
        xo.append(float((edges[bi] + edges[bi + 1]) * 0.5))
        yo.append(float(np.mean(ys[mask])))
    return np.asarray(xo, dtype=np.float64), np.asarray(yo, dtype=np.float64)


def parse_difficulty_groups(text: str) -> List[Tuple[str, float, float]]:
    raw = str(text).strip()
    if not raw:
        return list(DEFAULT_DIFFICULTY_GROUPS)
    out: List[Tuple[str, float, float]] = []
    for item in raw.split(","):
        parts = [x.strip() for x in item.split(":")]
        if len(parts) != 3:
            raise ValueError(f"invalid difficulty group: {item}. expected LABEL:MIN:MAX")
        out.append((parts[0], float(parts[1]), float(parts[2])))
    if not out:
        raise ValueError("empty --difficulty_groups")
    return out


def write_csv(path: Path, rows: Sequence[Dict[str, object]], prefix: Optional[Sequence[str]] = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for r in rows for k in r.keys()})
    ordered: List[str] = []
    for k in list(prefix or []):
        if k in keys:
            ordered.append(k)
    ordered.extend([k for k in keys if k not in ordered])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_by_difficulty_groups(
    rows: Sequence[Dict[str, object]],
    methods: Sequence[str],
    metrics: Sequence[str],
    groups: Sequence[Tuple[str, float, float]],
) -> List[Dict[str, object]]:
    summary: List[Dict[str, object]] = []
    for method in methods:
        method_rows = [r for r in rows if str(r.get("method", "")).strip() == method]
        for group_label, lo, hi in groups:
            grp = []
            for row in method_rows:
                diff = parse_float(row.get("scenario_difficulty", "nan"))
                if np.isfinite(diff) and lo <= diff < hi:
                    grp.append(row)
            out: Dict[str, object] = {
                "method": method,
                "difficulty_group": group_label,
                "difficulty_min": float(lo),
                "difficulty_max": float(hi),
                "num_rows": int(len(grp)),
            }
            for metric in metrics:
                scene_values = scene_aggregated_values(grp, metric)
                mean, std, sem, count = finite_stats(scene_values)
                _, seed_std, seed_sem, seed_count = finite_stats(replicate_aggregated_values(grp, metric))
                out[f"{metric}_mean"] = mean
                out[f"{metric}_std"] = std
                out[f"{metric}_sem"] = sem
                out[f"{metric}_count"] = count
                out[f"{metric}_seed_std"] = seed_std
                out[f"{metric}_seed_sem"] = seed_sem
                out[f"{metric}_seed_count"] = seed_count
            summary.append(out)
    return summary


def summarize_runtime_tradeoff(
    rows: Sequence[Dict[str, object]],
    methods: Sequence[str],
    runtime_key: str,
    quality_metric: str,
) -> List[Dict[str, object]]:
    summary: List[Dict[str, object]] = []
    for method in methods:
        grp = [r for r in rows if str(r.get("method", "")).strip() == method]
        rt_mean, rt_std, _, rt_count = finite_stats(scene_aggregated_values(grp, runtime_key))
        q_mean, q_std, _, q_count = finite_stats(scene_aggregated_values(grp, quality_metric))
        _, rt_seed_std, rt_seed_sem, rt_seed_count = finite_stats(replicate_aggregated_values(grp, runtime_key))
        _, q_seed_std, q_seed_sem, q_seed_count = finite_stats(replicate_aggregated_values(grp, quality_metric))
        summary.append({
            "method": method,
            f"{runtime_key}_mean": rt_mean,
            f"{runtime_key}_std": rt_std,
            f"{runtime_key}_sem": rt_seed_sem,
            f"{runtime_key}_count": rt_count,
            f"{runtime_key}_seed_std": rt_seed_std,
            f"{runtime_key}_seed_count": rt_seed_count,
            f"{quality_metric}_mean": q_mean,
            f"{quality_metric}_std": q_std,
            f"{quality_metric}_sem": q_seed_sem,
            f"{quality_metric}_count": q_count,
            f"{quality_metric}_seed_std": q_seed_std,
            f"{quality_metric}_seed_count": q_seed_count,
            "num_rows": int(len(grp)),
        })
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot B/C paper figures from evaluate_mrta artifacts."
    )
    p.add_argument("--input", type=str, nargs="+", required=True,
                   help="LABEL=PATH or PATH where PATH is eval dir or episode_metrics.csv")
    p.add_argument("--metrics", type=str, default=",".join(DEFAULT_METRICS))
    p.add_argument("--x_key", type=str, default="scenario_difficulty", choices=["scale", "scenario_difficulty", "time_horizon"])
    p.add_argument("--output_dir", type=str, default="results/eval_bc_from_artifacts")
    p.add_argument("--title_b", type=str, default="B. Performance Across Scenario Difficulty")
    p.add_argument("--difficulty_bins", type=int, default=8)
    p.add_argument(
        "--difficulty_groups",
        type=str,
        default="D0.20-0.30:0.20:0.30,D0.30-0.40:0.30:0.40,D0.40-0.50:0.40:0.50,D0.50-0.60:0.50:0.60,D0.60-0.70:0.60:0.70,D0.70-0.80:0.70:0.81",
    )
    p.add_argument("--b_style", type=str, default="line", choices=["line", "grouped_bar"])
    p.add_argument("--make_c", action="store_true", help="also draw C runtime-quality tradeoff figure")
    p.add_argument("--runtime_key", type=str, default="infer_time")
    p.add_argument("--c_metric", type=str, default="cpsat_relative_done")
    p.add_argument("--title_c", type=str, default="C. Quality-Runtime Tradeoff")
    return p.parse_args()


def main():
    args = parse_args()
    metrics = parse_csv_list(args.metrics)
    if not metrics:
        raise ValueError("empty --metrics")

    rows: List[Dict[str, object]] = []
    for input_idx, item in enumerate(args.input):
        explicit_label, path = parse_input_item(item)
        csv_path = resolve_csv(path)
        src_rows = load_rows(csv_path)
        label = explicit_label if explicit_label else (path.name if path.is_dir() else path.stem)
        for r in src_rows:
            rr: Dict[str, object] = dict(r)
            rr["method"] = str(label)
            rr["_replicate"] = f"{label}:{input_idx}"
            rows.append(rr)

    if not rows:
        raise RuntimeError("no rows loaded")

    rows = enrich_derived_metrics(rows)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib unavailable: {exc}")

    methods = sorted({str(r.get("method", "")).strip() for r in rows if str(r.get("method", "")).strip()})
    colors = {
        "ours": "#1f77b4",
        "mappo": "#d62728",
        "cbs_ta_ptc": "#9467bd",
        "ga": "#8c564b",
        "ctas_d": "#2ca02c",
    }

    difficulty_groups = parse_difficulty_groups(args.difficulty_groups)
    fig_b, axes_b = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes_b = axes_b.ravel()
    x_labels: List[str] = []
    for i, metric in enumerate(metrics[:4]):
        ax = axes_b[i]
        if args.x_key == "scenario_difficulty" and args.b_style == "grouped_bar":
            group_labels = [g[0] for g in difficulty_groups]
            x = np.arange(len(group_labels), dtype=np.float64)
            width = 0.82 / max(1, len(methods))
            for mi, method in enumerate(methods):
                vals: List[float] = []
                errs: List[float] = []
                sub = [r for r in rows if str(r.get("method", "")).strip() == method]
                for _, lo, hi in difficulty_groups:
                    group_rows = [
                        r for r in sub
                        if lo <= parse_float(r.get("scenario_difficulty", "nan")) < hi
                    ]
                    grp_vals = scene_aggregated_values(group_rows, metric)
                    mean, _, _, _ = finite_stats(grp_vals)
                    _, _, sem, _ = finite_stats(replicate_aggregated_values(group_rows, metric))
                    vals.append(mean)
                    errs.append(sem)
                safe_vals = np.nan_to_num(np.asarray(vals, dtype=np.float64), nan=0.0)
                safe_errs = np.nan_to_num(np.asarray(errs, dtype=np.float64), nan=0.0)
                offset = (mi - (len(methods) - 1) / 2.0) * width
                ax.bar(x + offset, safe_vals, width=width, yerr=safe_errs, capsize=3, label=method, alpha=0.9)
            ax.set_xticks(x)
            ax.set_xticklabels(group_labels, rotation=0)
        else:
            for m in methods:
                sub = [r for r in rows if str(r.get("method", "")) == m]
                if args.x_key == "scale":
                    scale_vals = [str(r.get("scale", "")).strip() for r in sub]
                    uniq_scales = [s for s in dict.fromkeys(scale_vals) if s]
                    xs = np.arange(len(uniq_scales), dtype=np.float64)
                    ys = np.array([
                        np.nanmean([parse_float(r.get(metric, "nan")) for r in sub if str(r.get("scale", "")).strip() == sc])
                        for sc in uniq_scales
                    ], dtype=np.float64)
                    x_plot, y_plot = xs, ys
                    x_labels = uniq_scales
                else:
                    xs = np.array([parse_float(r.get(args.x_key, "nan")) for r in sub], dtype=np.float64)
                    ys = np.array([parse_float(r.get(metric, "nan")) for r in sub], dtype=np.float64)
                    if args.x_key == "scenario_difficulty":
                        x_plot, y_plot = aggregate_binned(xs, ys, bins=int(args.difficulty_bins))
                    else:
                        x_plot, y_plot = aggregate_exact(xs, ys)
                if x_plot.size == 0:
                    continue
                order = np.argsort(x_plot)
                ax.plot(
                    x_plot[order],
                    y_plot[order],
                    marker="o",
                    linewidth=2.0,
                    markersize=4.0,
                    color=colors.get(m, None),
                    label=m,
                )
        ax.set_title(metric_label(metric))
        ax.grid(alpha=0.25)
        if i >= 2:
            ax.set_xlabel("Scale" if args.x_key == "scale" else metric_label(args.x_key))
        if args.x_key == "scale" and x_labels:
            ax.set_xticks(np.arange(len(x_labels), dtype=np.float64))
            ax.set_xticklabels(x_labels, rotation=20)

    for j in range(min(4, len(metrics)), len(axes_b)):
        axes_b[j].set_axis_off()
    handles, labels = axes_b[0].get_legend_handles_labels()
    fig_b.legend(handles, labels, loc="lower center", ncol=max(2, len(labels)))
    fig_b.suptitle(str(args.title_b))
    fig_b.tight_layout(rect=(0.0, 0.06, 1.0, 0.95))
    b_path = out_dir / "B_metrics_vs_difficulty.png"
    fig_b.savefig(b_path, dpi=160)
    plt.close(fig_b)

    difficulty_summary = summarize_by_difficulty_groups(
        rows=rows,
        methods=methods,
        metrics=metrics,
        groups=difficulty_groups,
    )
    difficulty_summary_csv = out_dir / "B_difficulty_metric_summary.csv"
    write_csv(
        difficulty_summary_csv,
        difficulty_summary,
        prefix=["method", "difficulty_group", "difficulty_min", "difficulty_max", "num_rows"],
    )

    c_path: Optional[Path] = None
    runtime_summary_csv: Optional[Path] = None
    if bool(args.make_c):
        runtime_key = str(args.runtime_key)
        c_metric = str(args.c_metric)
        runtime_summary = summarize_runtime_tradeoff(
            rows=rows,
            methods=methods,
            runtime_key=runtime_key,
            quality_metric=c_metric,
        )
        runtime_summary_csv = out_dir / "C_runtime_tradeoff_summary.csv"
        write_csv(runtime_summary_csv, runtime_summary, prefix=["method", "num_rows"])

        fig_c, ax_c = plt.subplots(1, 1, figsize=(7.5, 5.2))
        for row in runtime_summary:
            method = str(row.get("method", ""))
            x = parse_float(row.get(f"{runtime_key}_mean", "nan"))
            y = parse_float(row.get(f"{c_metric}_mean", "nan"))
            xerr = parse_float(row.get(f"{runtime_key}_sem", "nan"))
            yerr = parse_float(row.get(f"{c_metric}_sem", "nan"))
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            ax_c.errorbar(
                x,
                y,
                xerr=0.0 if not np.isfinite(xerr) else xerr,
                yerr=0.0 if not np.isfinite(yerr) else yerr,
                marker="o",
                markersize=7,
                capsize=4,
                linestyle="none",
                label=method,
            )
            ax_c.annotate(method, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)
        ax_c.set_xlabel(metric_label(runtime_key))
        ax_c.set_ylabel(metric_label(c_metric))
        ax_c.set_title(str(args.title_c))
        ax_c.grid(alpha=0.25)
        handles, labels = ax_c.get_legend_handles_labels()
        if labels:
            ax_c.legend(loc="best")
        fig_c.tight_layout()
        c_path = out_dir / "C_runtime_tradeoff.png"
        fig_c.savefig(c_path, dpi=160)
        plt.close(fig_c)

    combined_csv = out_dir / "combined_episode_metrics.csv"
    write_csv(combined_csv, rows, prefix=["method", "scene_idx", "scene_path"])

    print(f"saved_combined_csv={combined_csv.as_posix()}")
    print(f"saved_B_summary={difficulty_summary_csv.as_posix()}")
    print(f"saved_B_figure={b_path.as_posix()}")
    if runtime_summary_csv is not None:
        print(f"saved_C_summary={runtime_summary_csv.as_posix()}")
    if c_path is not None:
        print(f"saved_C_figure={c_path.as_posix()}")


if __name__ == "__main__":
    main()
