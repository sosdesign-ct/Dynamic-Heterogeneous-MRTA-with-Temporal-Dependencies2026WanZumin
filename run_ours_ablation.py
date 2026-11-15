from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


# Train-time ablation switches for our method.
VARIANT_TRAIN_FLAGS: Dict[str, List[str]] = {
    "full": [],
    "wo_rgat": ["--wo_rgat"],
    "self_only_graph": ["--self_only_graph"],
    "wo_mask": ["--wo_mask"],
    "wo_commitment": ["--wo_commitment"],
}

# Runtime semantic switches that also need eval-time flags.
VARIANT_EVAL_FLAGS: Dict[str, List[str]] = {
    "full": [],
    "wo_rgat": [],
    "self_only_graph": [],
    "wo_mask": ["--wo_mask"],
    "wo_commitment": ["--wo_commitment"],
}


def parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_seed_list(text: str) -> List[int]:
    seeds: List[int] = []
    for tok in parse_csv_list(text):
        seeds.append(int(tok))
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def resolve_path(workspace: Path, raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (workspace / p)


def has_option(args: List[str], option: str) -> bool:
    return any(tok == option or tok.startswith(option + "=") for tok in args)


def derive_best_fixed_ckpt_path(save_path: Path) -> Path:
    suffix = save_path.suffix if save_path.suffix else ".pt"
    stem = save_path.stem if save_path.suffix else save_path.name
    return save_path.with_name(f"{stem}_best_fixed_eval{suffix}")


def option_int(args: List[str], option: str) -> Optional[int]:
    for idx, tok in enumerate(args):
        if tok == option and idx + 1 < len(args):
            return int(args[idx + 1])
        if tok.startswith(option + "="):
            return int(tok.split("=", 1)[1])
    return None


def checkpoint_step(path: Path) -> Optional[int]:
    if not path.is_file():
        return None
    try:
        import torch

        checkpoint = torch.load(path, map_location="cpu")
        if not isinstance(checkpoint, dict) or "step_count" not in checkpoint:
            return None
        return int(checkpoint["step_count"])
    except Exception as exc:
        print(f"checkpoint inspection failed, training will run: path={path} error={exc}", flush=True)
        return None


def run_command(cmd: List[str], cwd: Path, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("COMMAND: " + " ".join(cmd) + "\n\n")
        f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
            f.flush()
        rc = proc.wait()
    if rc != 0:
        reason = f"signal {-rc}" if rc < 0 else f"exit code {rc}"
        raise RuntimeError(f"command failed with {reason}; log={log_path}: {' '.join(cmd)}")


def read_aggregate_metrics(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(f"aggregate metrics file not found: {path}")

    out: Dict[str, Dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric = str(row.get("metric", "")).strip()
            if not metric:
                continue
            out[metric] = {
                "mean": float(row.get("mean", "nan")),
                "std": float(row.get("std", "nan")),
            }
    return out


def metric_value(metrics: Dict[str, Dict[str, float]], metric: str, field: str) -> float:
    raw = metrics.get(metric, {}).get(field, float("nan"))
    try:
        return float(raw)
    except Exception:
        return float("nan")


def mean_std(xs: List[float]) -> tuple[float, float]:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    if not vals:
        return float("nan"), float("nan")
    m = sum(vals) / float(len(vals))
    v = sum((x - m) ** 2 for x in vals) / float(len(vals))
    return m, math.sqrt(max(v, 0.0))


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_ablation_summary(path: Path, rows: List[Dict[str, object]]):
    if not rows:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        print(f"ablation plot skipped: matplotlib unavailable ({exc})", flush=True)
        return

    metrics = [
        ("recognized_done_rate_mean", "Completion"),
        ("cpsat_relative_done_mean", "CP-SAT Relative Completion"),
    ]
    variants = [str(r.get("variant", "")) for r in rows]
    x = np.arange(len(variants), dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    axes = np.asarray(axes).ravel()
    for ax, (metric, label) in zip(axes, metrics):
        vals = [float(str(r.get(f"{metric}_avg", float("nan")))) for r in rows]
        errs = [float(str(r.get(f"{metric}_std", float("nan")))) for r in rows]
        ax.bar(x, np.nan_to_num(vals, nan=0.0), yerr=np.nan_to_num(errs, nan=0.0), capsize=3)
        ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Ablation Study")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


TOP_LEVEL_OPTIONS = {
    "--workspace",
    "--python",
    "--seeds",
    "--variants",
    "--save_root",
    "--eval_output_root",
    "--log_root",
    "--fixed_set_path",
    "--oracle_metrics_csv",
    "--eval_episodes",
    "--policy_decode",
    "--policy_explore_eps",
    "--use_best_fixed_eval_ckpt",
    "--resume_existing_train",
    "--skip_completed_train",
    "--no-skip_completed_train",
    "--skip_existing",
    "--reuse_full_checkpoint_root",
    "--require_strict_protocol",
}


def split_passthrough_args(argv: List[str]) -> tuple[List[str], List[str], List[str]]:
    markers = {"--train_extra_args", "--eval_extra_args"}
    cleaned: List[str] = []
    train_extra: List[str] = []
    eval_extra: List[str] = []
    current: str | None = None
    for tok in argv:
        if tok in markers:
            current = tok
            continue
        if current is not None and tok in TOP_LEVEL_OPTIONS:
            current = None
            cleaned.append(tok)
            continue
        if current == "--train_extra_args":
            train_extra.append(tok)
        elif current == "--eval_extra_args":
            eval_extra.append(tok)
        else:
            cleaned.append(tok)
    return cleaned, train_extra, eval_extra


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run our-method ablations (train + fixed-set eval) and export summary CSVs "
            "for paper-ready reporting."
        )
    )
    p.add_argument("--workspace", type=str, default=".")
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument("--seeds", type=str, default="1,2,3")
    p.add_argument(
        "--variants",
        type=str,
        default="full,wo_rgat,self_only_graph,wo_mask,wo_commitment",
        help="comma-separated ablation variants",
    )
    p.add_argument("--save_root", type=str, default="model/save/ablation_ours")
    p.add_argument("--eval_output_root", type=str, default="results/eval_ablation_ours")
    p.add_argument("--log_root", type=str, default="results/eval_ablation_ours/logs")
    p.add_argument("--fixed_set_path", type=str, default="MRTAFixedTestSet")
    p.add_argument(
        "--oracle_metrics_csv",
        type=str,
        default="results/cpsat_oracle/fixed35_dynamic_enveta_exact_fullteams/episode_metrics.csv",
        help="CP-SAT episode_metrics.csv used to compute cpsat_relative_done; empty string disables oracle metrics",
    )
    p.add_argument(
        "--eval_episodes",
        type=int,
        default=0,
        help="episodes for evaluate_mrta; <=0 means all scenes in fixed set",
    )
    p.add_argument(
        "--policy_decode",
        type=str,
        default="greedy",
        choices=["greedy", "sample"],
        help="decode mode used in evaluation",
    )
    p.add_argument(
        "--policy_explore_eps",
        type=float,
        default=0.0,
        help="epsilon exploration used in evaluation under greedy decode",
    )
    p.add_argument(
        "--use_best_fixed_eval_ckpt",
        action="store_true",
        help="evaluate *_best_fixed_eval checkpoint if available",
    )
    p.add_argument(
        "--resume_existing_train",
        action="store_true",
        help="resume training from the variant checkpoint if it already exists and final eval is not present",
    )
    p.add_argument(
        "--skip_completed_train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="skip training when checkpoint step_count has reached --total_steps (default: enabled)",
    )
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument(
        "--reuse_full_checkpoint_root",
        type=str,
        default="",
        help=(
            "optional checkpoint root used only for the full variant; useful when "
            "new strict ablations are trained in a separate save_root"
        ),
    )
    p.add_argument(
        "--require_strict_protocol",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "require train_extra_args to enable strict difficulty enforcement and "
            "reject incompatible checkpoints before resuming"
        ),
    )
    p.add_argument("--train_extra_args", nargs="*", default=[])
    p.add_argument("--eval_extra_args", nargs="*", default=[])
    return p


def main():
    cleaned_argv, train_extra_args, eval_extra_args = split_passthrough_args(sys.argv[1:])
    args = build_parser().parse_args(cleaned_argv)
    args.train_extra_args = train_extra_args
    args.eval_extra_args = eval_extra_args

    workspace = Path(args.workspace).resolve()
    python_exec = str(args.python)

    train_entry = workspace / "train_mrta.py"
    eval_entry = workspace / "evaluate_mrta.py"
    if not train_entry.is_file() or not eval_entry.is_file():
        raise FileNotFoundError(
            f"train/eval entry not found under workspace={workspace}. "
            "expected train_mrta.py and evaluate_mrta.py"
        )

    seeds = parse_seed_list(args.seeds)
    variants = parse_csv_list(args.variants)
    if not variants:
        raise ValueError("at least one variant is required")

    invalid_variants = [v for v in variants if v not in VARIANT_TRAIN_FLAGS]
    if invalid_variants:
        valid = ", ".join(sorted(VARIANT_TRAIN_FLAGS.keys()))
        raise ValueError(f"invalid variants: {invalid_variants}; valid variants are: {valid}")

    save_root = resolve_path(workspace, args.save_root)
    reuse_full_root = (
        resolve_path(workspace, args.reuse_full_checkpoint_root)
        if str(args.reuse_full_checkpoint_root).strip()
        else None
    )
    eval_root = resolve_path(workspace, args.eval_output_root)
    log_root = resolve_path(workspace, args.log_root)
    oracle_csv = resolve_path(workspace, args.oracle_metrics_csv) if str(args.oracle_metrics_csv).strip() else None
    if oracle_csv is not None and not oracle_csv.is_file():
        raise FileNotFoundError(f"oracle metrics csv not found: {oracle_csv}")
    strict_requested = has_option(args.train_extra_args, "--enforce_env_difficulty")
    if bool(args.require_strict_protocol) and not strict_requested:
        raise ValueError(
            "strict protocol requested, but --train_extra_args does not contain "
            "--enforce_env_difficulty"
        )

    detail_rows: List[Dict[str, object]] = []

    print(
        f"start ablation: variants={variants} seeds={seeds} fixed_set={args.fixed_set_path} "
        f"policy_decode={args.policy_decode} eps={float(args.policy_explore_eps):.3f} "
        f"oracle={oracle_csv if oracle_csv is not None else 'disabled'}",
        flush=True,
    )

    for variant in variants:
        for seed in seeds:
            run_tag = f"{variant}_seed{seed}"
            checkpoint_root = reuse_full_root if variant == "full" and reuse_full_root is not None else save_root
            ckpt_path = checkpoint_root / variant / f"seed_{seed}.pt"
            train_log = log_root / f"{run_tag}_train.log"
            eval_log = log_root / f"{run_tag}_eval.log"

            eval_dir = eval_root / run_tag
            agg_csv = eval_dir / "aggregate_metrics.csv"

            if args.skip_existing and agg_csv.is_file():
                print(f"skip existing run: {run_tag}", flush=True)
            else:
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)

                train_cmd: List[str] = [
                    python_exec,
                    str(train_entry),
                    "--seed",
                    str(seed),
                    "--save_path",
                    str(ckpt_path),
                    "--no_auto_tune_runtime",
                ]
                train_cmd.extend(VARIANT_TRAIN_FLAGS[variant])
                if bool(args.resume_existing_train) and ckpt_path.is_file() and not has_option(args.train_extra_args, "--resume_path"):
                    if bool(args.require_strict_protocol):
                        try:
                            import torch

                            existing = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                            existing_cfg = existing.get("mrta_config", {}) if isinstance(existing, dict) else {}
                            if not bool(existing_cfg.get("enforce_difficulty_range", False)):
                                raise RuntimeError(
                                    f"refusing to resume non-strict checkpoint under strict protocol: {ckpt_path}"
                                )
                        except RuntimeError:
                            raise
                        except Exception as exc:
                            raise RuntimeError(f"cannot validate checkpoint protocol: {ckpt_path}: {exc}") from exc
                    train_cmd.extend(["--resume_path", str(ckpt_path)])
                train_cmd.extend(args.train_extra_args)

                target_steps = option_int(args.train_extra_args, "--total_steps")
                saved_steps = checkpoint_step(ckpt_path) if bool(args.skip_completed_train) else None
                train_complete = target_steps is not None and saved_steps is not None and saved_steps >= target_steps
                if train_complete:
                    print(
                        f"[train] skip completed {run_tag}: checkpoint_step={saved_steps} "
                        f"target_step={target_steps}",
                        flush=True,
                    )
                else:
                    print(f"[train] {run_tag}", flush=True)
                    run_command(train_cmd, cwd=workspace, log_path=train_log)

                eval_ckpt = ckpt_path
                if bool(args.use_best_fixed_eval_ckpt):
                    best_ckpt = derive_best_fixed_ckpt_path(ckpt_path)
                    if best_ckpt.is_file():
                        eval_ckpt = best_ckpt

                eval_cmd: List[str] = [
                    python_exec,
                    str(eval_entry),
                    "--checkpoint",
                    str(eval_ckpt),
                    "--fixed_set_path",
                    str(args.fixed_set_path),
                    "--episodes",
                    str(int(args.eval_episodes)),
                    "--method_alias",
                    variant,
                    "--output_dir",
                    str(eval_root),
                    "--output_tag",
                    run_tag,
                    "--policy_decode",
                    str(args.policy_decode),
                    "--policy_explore_eps",
                    str(float(args.policy_explore_eps)),
                ]
                eval_cmd.extend(VARIANT_EVAL_FLAGS[variant])
                if oracle_csv is not None and not has_option(args.eval_extra_args, "--oracle_metrics_csv"):
                    eval_cmd.extend(["--oracle_metrics_csv", str(oracle_csv)])
                eval_cmd.extend(args.eval_extra_args)

                print(f"[eval] {run_tag}", flush=True)
                run_command(eval_cmd, cwd=workspace, log_path=eval_log)

            metrics = read_aggregate_metrics(agg_csv)
            detail_rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "checkpoint": str(ckpt_path),
                    "return_mean": metric_value(metrics, "return", "mean"),
                    "cpsat_relative_done_mean": metric_value(metrics, "cpsat_relative_done", "mean"),
                    "recognized_done_tasks_mean": metric_value(metrics, "recognized_done_tasks", "mean"),
                    "recognized_done_rate_mean": metric_value(metrics, "recognized_done_rate", "mean"),
                    "success_mean": metric_value(metrics, "success", "mean"),
                    "makespan_mean": metric_value(metrics, "makespan", "mean"),
                    "violation_rate_mean": metric_value(metrics, "violation_rate", "mean"),
                    "throughput_mean": metric_value(metrics, "throughput", "mean"),
                    "infer_time_mean": metric_value(metrics, "infer_time", "mean"),
                }
            )

    detail_fields = [
        "variant",
        "seed",
        "checkpoint",
        "return_mean",
        "cpsat_relative_done_mean",
        "recognized_done_tasks_mean",
        "recognized_done_rate_mean",
        "success_mean",
        "makespan_mean",
        "violation_rate_mean",
        "throughput_mean",
        "infer_time_mean",
    ]
    detail_csv = eval_root / "ablation_seed_metrics.csv"
    write_csv(detail_csv, detail_rows, detail_fields)

    grouped_rows: List[Dict[str, object]] = []
    metric_keys = [
        "return_mean",
        "cpsat_relative_done_mean",
        "recognized_done_tasks_mean",
        "recognized_done_rate_mean",
        "success_mean",
        "makespan_mean",
        "violation_rate_mean",
        "throughput_mean",
        "infer_time_mean",
    ]
    for variant in variants:
        subset = [r for r in detail_rows if str(r.get("variant")) == variant]
        row: Dict[str, object] = {
            "variant": variant,
            "num_seeds": len(subset),
        }
        for key in metric_keys:
            vals = [float(str(r.get(key, float("nan")))) for r in subset]
            m, st = mean_std(vals)
            row[f"{key}_avg"] = m
            row[f"{key}_std"] = st
        grouped_rows.append(row)

    grouped_fields = [
        "variant",
        "num_seeds",
        "return_mean_avg",
        "return_mean_std",
        "cpsat_relative_done_mean_avg",
        "cpsat_relative_done_mean_std",
        "recognized_done_tasks_mean_avg",
        "recognized_done_tasks_mean_std",
        "recognized_done_rate_mean_avg",
        "recognized_done_rate_mean_std",
        "success_mean_avg",
        "success_mean_std",
        "makespan_mean_avg",
        "makespan_mean_std",
        "violation_rate_mean_avg",
        "violation_rate_mean_std",
        "throughput_mean_avg",
        "throughput_mean_std",
        "infer_time_mean_avg",
        "infer_time_mean_std",
    ]
    grouped_csv = eval_root / "ablation_variant_summary.csv"
    write_csv(grouped_csv, grouped_rows, grouped_fields)

    plot_path = eval_root / "ablation_variant_summary.png"
    plot_ablation_summary(plot_path, grouped_rows)

    protocol_summary = {
        "method_family": "ours",
        "policy_decode": str(args.policy_decode),
        "policy_explore_eps": float(args.policy_explore_eps),
        "fixed_set_path": str(args.fixed_set_path),
        "oracle_metrics_csv": str(oracle_csv) if oracle_csv is not None else "",
        "eval_episodes": int(args.eval_episodes),
        "use_best_fixed_eval_ckpt": bool(args.use_best_fixed_eval_ckpt),
        "variants": list(variants),
        "seeds": list(seeds),
        "train_extra_args": list(args.train_extra_args),
        "eval_extra_args": list(args.eval_extra_args),
    }
    protocol_json = eval_root / "protocol_summary.json"
    protocol_json.write_text(json.dumps(protocol_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("ablation finished", flush=True)
    print(f"seed_metrics={detail_csv}", flush=True)
    print(f"variant_summary={grouped_csv}", flush=True)
    print(f"variant_plot={plot_path}", flush=True)
    print(f"protocol_summary={protocol_json}", flush=True)


if __name__ == "__main__":
    main()
