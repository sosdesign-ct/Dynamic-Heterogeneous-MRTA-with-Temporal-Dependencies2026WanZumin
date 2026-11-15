from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


SCALE_PRESETS: Dict[str, Dict[str, int]] = {
    "s1": {
        "num_agents": 12,
        "init_min": 12,
        "init_max": 20,
        "total_floor": 24,
        "total_cap": 36,
        "max_tasks": 48,
        "dynamic_every": 10,
        "dynamic_min": 0,
        "dynamic_max": 1,
    },
    "s2": {
        "num_agents": 16,
        "init_min": 16,
        "init_max": 24,
        "total_floor": 32,
        "total_cap": 48,
        "max_tasks": 64,
        "dynamic_every": 10,
        "dynamic_min": 0,
        "dynamic_max": 2,
    },
    "s3": {
        "num_agents": 20,
        "init_min": 20,
        "init_max": 28,
        "total_floor": 40,
        "total_cap": 56,
        "max_tasks": 80,
        "dynamic_every": 10,
        "dynamic_min": 0,
        "dynamic_max": 2,
    },
    "s4": {
        "num_agents": 24,
        "init_min": 24,
        "init_max": 32,
        "total_floor": 48,
        "total_cap": 64,
        "max_tasks": 96,
        "dynamic_every": 10,
        "dynamic_min": 0,
        "dynamic_max": 2,
    },
    "s5": {
        "num_agents": 28,
        "init_min": 28,
        "init_max": 36,
        "total_floor": 56,
        "total_cap": 72,
        "max_tasks": 112,
        "dynamic_every": 10,
        "dynamic_min": 0,
        "dynamic_max": 3,
    },
    "s6": {
        "num_agents": 32,
        "init_min": 32,
        "init_max": 40,
        "total_floor": 64,
        "total_cap": 80,
        "max_tasks": 128,
        "dynamic_every": 8,
        "dynamic_min": 1,
        "dynamic_max": 3,
    },
    "s7": {
        "num_agents": 36,
        "init_min": 36,
        "init_max": 44,
        "total_floor": 72,
        "total_cap": 96,
        "max_tasks": 144,
        "dynamic_every": 8,
        "dynamic_min": 1,
        "dynamic_max": 3,
    },
    "s8": {
        "num_agents": 40,
        "init_min": 40,
        "init_max": 48,
        "total_floor": 80,
        "total_cap": 112,
        "max_tasks": 160,
        "dynamic_every": 8,
        "dynamic_min": 1,
        "dynamic_max": 4,
    },
}

MIXED_TRAIN_PRESET: Dict[str, int] = {
    "num_agents": 28,
    "init_min": 16,
    "init_max": 40,
    "total_floor": 32,
    "total_cap": 80,
    "max_tasks": 160,
    "dynamic_every": 10,
    "dynamic_min": 0,
    "dynamic_max": 3,
}

def parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def split_passthrough_args(argv: List[str]) -> tuple[List[str], List[str], List[str], List[str]]:
    """Split top-level options from child-command options.

    Top-level options must precede the first passthrough marker. Once a marker
    is encountered, every following token belongs to that passthrough section
    until another marker is encountered. This is intentional because child
    commands share option names such as ``--num_envs`` with this driver.
    """
    markers = {"--train_extra_args", "--eval_extra_args", "--benchmark_extra_args"}
    cleaned: List[str] = []
    train_extra: List[str] = []
    eval_extra: List[str] = []
    benchmark_extra: List[str] = []
    current: str | None = None
    for tok in argv:
        if tok in markers:
            current = tok
            continue
        if current == "--train_extra_args":
            train_extra.append(tok)
        elif current == "--eval_extra_args":
            eval_extra.append(tok)
        elif current == "--benchmark_extra_args":
            benchmark_extra.append(tok)
        else:
            cleaned.append(tok)
    return cleaned, train_extra, eval_extra, benchmark_extra


def parse_scale_names(text: str) -> List[str]:
    names = parse_csv_list(text)
    if not names:
        raise ValueError("at least one scale is required")
    bad = [n for n in names if n not in SCALE_PRESETS]
    if bad:
        raise ValueError(f"invalid scale names: {bad}; valid: {sorted(SCALE_PRESETS.keys())}")
    return names


def resolve_path(workspace: Path, raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (workspace / p)


def run_cmd(cmd: List[str], cwd: Path, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("COMMAND: " + " ".join(cmd) + "\n\n")
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
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"command failed (exit={rc}): {' '.join(cmd)}")


def read_aggregate_csv(path: Path) -> Dict[str, Dict[str, float]]:
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


def metric_val(metrics: Dict[str, Dict[str, float]], metric: str, key: str = "mean") -> float:
    try:
        return float(metrics.get(metric, {}).get(key, float("nan")))
    except Exception:
        return float("nan")


def has_option(args: Sequence[str], name: str) -> bool:
    return any(str(x) == name or str(x).startswith(name + "=") for x in args)


def oracle_tag_for_scale(scale: str) -> str:
    return f"{scale}_dynamic_enveta_fullteams"


def oracle_csv_for_scale(oracle_root: Path, scale: str) -> Path:
    return oracle_root / oracle_tag_for_scale(scale) / "episode_metrics.csv"


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def has_eval_artifact(run_dir: Path) -> bool:
    return (run_dir / "aggregate_metrics.csv").is_file() and (run_dir / "summary.json").is_file()


def count_csv_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return sum(1 for _ in reader)


def validate_bc_chunk(
    chunk_dir: Path,
    expected_scene_count: int,
    expected_methods: int,
) -> Tuple[bool, str]:
    ab_csv = chunk_dir / "records_ab.csv"
    summary_json = chunk_dir / "summary.json"
    if not ab_csv.is_file() or not summary_json.is_file():
        return False, "missing records_ab.csv / summary.json"

    ab_n = count_csv_rows(ab_csv)
    exp_ab = int(expected_scene_count) * int(expected_methods)

    if ab_n != exp_ab:
        return False, f"AB rows mismatch: got={ab_n} expected={exp_ab}"
    return True, "ok"


def has_fixed_set(dir_path: Path, expected_num_envs: int) -> bool:
    if not dir_path.is_dir():
        return False
    overview = dir_path / "overview.csv"
    if not overview.is_file():
        return False
    env_dirs = [p for p in dir_path.glob("env_*") if p.is_dir()]
    return len(env_dirs) >= int(expected_num_envs)


def train_ckpt_path(root: Path, method: str, scale: str) -> Path:
    scale_dir = root / method / scale
    preferred = [
        scale_dir / "checkpoint.pt",
        scale_dir / "checkpoint_best_fixed_eval.pt",
    ]
    for p in preferred:
        if p.is_file():
            return p

    if scale_dir.is_dir():
        pts = sorted(
            [p for p in scale_dir.glob("*.pt") if p.is_file()],
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        if pts:
            return pts[0]

    # Fallback to shared checkpoints under model/save.
    shared_root = root.parent
    if method == "ours":
        shared_candidates = [
            shared_root / "mrta_checkpoint_best_fixed_eval.pt",
            shared_root / "mrta_checkpoint.pt",
        ]
    elif method == "mappo":
        shared_candidates = [
            shared_root / "mappo_classic_seed1.pt",
            shared_root / "mappo_classic_best_fixed_eval.pt",
            shared_root / "mappo_classic.pt",
        ]
    else:
        shared_candidates = []

    for p in shared_candidates:
        if p.is_file():
            return p

    # Keep an expected path for downstream error messages if nothing exists.
    return scale_dir / "checkpoint.pt"


def eval_run_tag(method: str, scale: str) -> str:
    return f"{method}_{scale}"


def scale_order_value(scale: str) -> float:
    preset = SCALE_PRESETS[scale]
    return float(preset["total_cap"]) + 0.5 * float(preset["init_max"])


def scale_axis_label(scale: str) -> str:
    preset = SCALE_PRESETS[scale]
    return f'{preset["num_agents"]}-{preset["init_max"]}/{preset["total_floor"]}-{preset["total_cap"]}'


def plot_a_from_rows(rows: List[Dict[str, object]], out_dir: Path, title: str):
    if not rows:
        raise RuntimeError("no rows for A figure")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"A plot skipped: matplotlib unavailable ({exc})", flush=True)
        return

    scales = sorted({str(r["scale"]) for r in rows}, key=scale_order_value)
    scale_labels = [scale_axis_label(sc) for sc in scales]
    methods = sorted({str(r["method"]) for r in rows})

    metrics = [
        ("cpsat_relative_done_mean", "CP-SAT Relative Completion"),
        ("recognized_done_tasks_mean", "Recognized Done Tasks"),
        ("runtime_mean", "Decision Runtime (s)"),
        ("violation_rate_mean", "Violation Rate"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()

    color_map = {
        "ours": "#1f77b4",
        "mappo": "#d62728",
        "cbs_ta_ptc": "#9467bd",
        "ga": "#8c564b",
    }

    x = list(range(len(scales)))
    for idx, (mk, mt) in enumerate(metrics):
        ax = axes[idx]
        for method in methods:
            ys: List[float] = []
            for sc in scales:
                matched = [r for r in rows if str(r["method"]) == method and str(r["scale"]) == sc]
                if not matched:
                    ys.append(float("nan"))
                    continue
                ys.append(float(matched[0].get(mk, float("nan"))))
            ax.plot(x, ys, marker="o", linewidth=2.0, label=method, color=color_map.get(method))

        ax.set_title(mt)
        ax.grid(alpha=0.25)
        if idx >= 2:
            ax.set_xticks(x)
            ax.set_xticklabels(scale_labels, rotation=20)
            ax.set_xlabel("Agent range / task range")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=max(2, len(labels)))
    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.95))

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "A_metrics_vs_scale.png", dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train/evaluate the formal A-figure scaling suite using the current "
            "evaluate_mrta + CP-SAT-oracle artifact protocol."
        )
    )
    p.add_argument("--workspace", type=str, default=".")
    p.add_argument("--python", type=str, default=sys.executable)

    p.add_argument("--scales", type=str, default="s1,s2,s3,s4,s5,s6,s7,s8")
    p.add_argument("--num_envs", type=int, default=35, help="fixed-set scene count for each scale")
    p.add_argument("--base_seed", type=int, default=20260423)
    p.add_argument("--num_agents", type=int, default=20)
    p.add_argument("--cap_dim", type=int, default=3)
    p.add_argument("--edge_density", type=float, default=0.15)
    p.add_argument("--ours_train_ckpt", type=str, default="model/save/scale_suite/ours/mixed_scale/checkpoint_best_fixed_eval.pt")
    p.add_argument("--mappo_ckpt", type=str, default="model/save/mappo_fair_baseline/full/seed_1.pt",
                   help="MAPPO checkpoint from the formal strict protocol run")

    p.add_argument("--train_seeds", type=str, default="1",
                   help="comma-separated seeds for ours mixed-scale training; each seed gets its own checkpoint")
    p.add_argument("--chunk_size", type=int, default=5, help="scenes per benchmark chunk")
    p.add_argument("--cpu", action="store_true", help="force benchmark evaluation on CPU")
    p.add_argument("--mappo_wo_mask", action="store_true", help="evaluate MAPPO with action mask disabled")
    p.add_argument("--no_mappo_wo_mask", action="store_false", dest="mappo_wo_mask", help="keep MAPPO mask enabled")
    p.set_defaults(mappo_wo_mask=False)
    p.add_argument("--mappo_update_epochs", type=int, default=3, help="mildly reduced MAPPO PPO epochs during training")
    p.add_argument("--ours_policy_decode", type=str, default="greedy", choices=["greedy", "sample"],
                   help="decode mode for ours during eval phase")
    p.add_argument("--ours_policy_explore_eps", type=float, default=0.0,
                   help="epsilon exploration for ours during eval phase")

    p.add_argument("--fixed_root", type=str, default="results/scale_suite/fixed_sets")
    p.add_argument("--save_root", type=str, default="model/save/scale_suite")
    p.add_argument("--eval_root", type=str, default="results/scale_suite/eval")
    p.add_argument("--plot_root", type=str, default="results/scale_suite/plots")
    p.add_argument("--oracle_root", type=str, default="results/scale_suite/oracle")
    p.add_argument("--log_root", type=str, default="results/scale_suite/logs")
    p.add_argument("--oracle_time_limit_sec", type=float, default=120.0)
    p.add_argument("--oracle_workers", type=int, default=8)
    p.add_argument("--oracle_max_teams_per_task", type=int, default=0)

    p.add_argument(
        "--strict_train_protocol",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "apply the formal main-run difficulty range/bin protocol to the "
            "large-scale training command; disabled by default because scale "
            "training uses a separate mixed-size generation regime"
        ),
    )
    p.add_argument(
        "--scale_difficulty_min", type=float, default=None,
        help="optional difficulty lower bound for scale fixed-set generation; "
             "leave unset to accept any feasible scenario (recommended for large scales)",
    )
    p.add_argument(
        "--scale_difficulty_max", type=float, default=None,
        help="optional difficulty upper bound for scale fixed-set generation",
    )

    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--only_phase", type=str, default="all",
                   choices=["all", "generate", "oracle", "train", "eval", "plot_a"],
                   help="run only one phase")

    p.add_argument("--train_extra_args", nargs="*", default=[])
    p.add_argument("--eval_extra_args", nargs="*", default=[])
    p.add_argument("--benchmark_extra_args", nargs="*", default=[])
    p.add_argument("--preview_only", action="store_true", help="generate a minimal A-figure preview using one scene per scale")
    p.add_argument("--preview_num_envs", type=int, default=1, help="number of scenes per scale in preview mode")
    cleaned_argv, train_extra_args, eval_extra_args, benchmark_extra_args = split_passthrough_args(sys.argv[1:])
    args = p.parse_args(cleaned_argv)
    args.train_extra_args = train_extra_args
    args.eval_extra_args = eval_extra_args
    args.benchmark_extra_args = benchmark_extra_args
    return args


def should_run(only_phase: str, phase: str) -> bool:
    return only_phase == "all" or only_phase == phase


def main():
    args = parse_args()
    workspace = Path(args.workspace).resolve()
    py = str(args.python)
    scales = parse_scale_names(args.scales)

    gen_script = workspace / "generate_fixed_mrta_testset.py"
    oracle_script = workspace / "evaluate_cpsat_oracle.py"
    train_script = workspace / "train_mrta.py"
    eval_script = workspace / "evaluate_mrta.py"
    for s in [gen_script, oracle_script, train_script, eval_script]:
        if not s.is_file():
            raise FileNotFoundError(f"required script not found: {s}")

    fixed_root = resolve_path(workspace, args.fixed_root)
    save_root = resolve_path(workspace, args.save_root)
    eval_root = resolve_path(workspace, args.eval_root)
    plot_root = resolve_path(workspace, args.plot_root)
    oracle_root = resolve_path(workspace, args.oracle_root)
    log_root = resolve_path(workspace, args.log_root)

    fixed_root.mkdir(parents=True, exist_ok=True)
    save_root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)
    plot_root.mkdir(parents=True, exist_ok=True)
    oracle_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    # 1) Generate fixed sets for each scale.
    if should_run(args.only_phase, "generate"):
        for i, scale in enumerate(scales):
            preset = SCALE_PRESETS[scale]
            fixed_dir = fixed_root / scale
            if args.skip_existing and has_fixed_set(fixed_dir, int(args.num_envs)):
                print(f"[skip generate] scale={scale}", flush=True)
                continue

            cmd = [
                py,
                str(gen_script),
                "--output_dir",
                str(fixed_dir),
                "--num_envs",
                str(int(args.num_envs)),
                "--base_seed",
                str(int(args.base_seed) + int(i) * 1000),
                "--num_agents",
                str(int(preset["num_agents"])),
                "--cap_dim",
                str(int(args.cap_dim)),
                "--max_tasks",
                str(int(preset["max_tasks"])),
                "--min_init_tasks",
                str(int(preset["init_min"])),
                "--max_init_tasks",
                str(int(preset["init_max"])),
                "--total_tasks_floor",
                str(int(preset["total_floor"])),
                "--total_tasks_cap",
                str(int(preset["total_cap"])),
                "--dynamic_arrival_every",
                str(int(preset["dynamic_every"])),
                "--dynamic_arrival_min",
                str(int(preset["dynamic_min"])),
                "--dynamic_arrival_max",
                str(int(preset["dynamic_max"])),
                "--edge_density",
                str(float(args.edge_density)),
            ]
            # Optional difficulty filtering for fixed-set generation.
            # For large scales the natural difficulty is already high (0.6-0.8),
            # so the default is to accept any feasible scenario.
            if args.scale_difficulty_min is not None:
                cmd.extend(["--difficulty_min", str(float(args.scale_difficulty_min))])
            if args.scale_difficulty_max is not None:
                cmd.extend(["--difficulty_max", str(float(args.scale_difficulty_max))])
            print(f"[generate] scale={scale}", flush=True)
            run_cmd(cmd, cwd=workspace, log_path=log_root / f"generate_{scale}.log")

            # Write protocol manifest for reproducibility.
            manifest = {
                "protocol": "scale_generalization_v2",
                "scale": scale,
                "preset": preset,
                "num_envs": int(args.num_envs),
                "base_seed": int(args.base_seed) + int(i) * 1000,
                "cap_dim": int(args.cap_dim),
                "edge_density": float(args.edge_density),
                "difficulty_min": args.scale_difficulty_min,
                "difficulty_max": args.scale_difficulty_max,
                "fixed_set_max_steps": 800,
            }
            manifest_path = fixed_dir / "scale_protocol_manifest.json"
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # 2) Build the CP-SAT oracle denominator for each generated scale fixed set.
    if should_run(args.only_phase, "oracle"):
        for scale in scales:
            fixed_dir = fixed_root / scale
            out_csv = oracle_csv_for_scale(oracle_root, scale)
            if args.skip_existing and out_csv.is_file():
                print(f"[skip oracle] scale={scale}", flush=True)
                continue
            cmd = [
                py,
                str(oracle_script),
                "--fixed_set_path",
                str(fixed_dir),
                "--max_scenes",
                str(int(args.num_envs)),
                "--output_dir",
                str(oracle_root),
                "--output_tag",
                oracle_tag_for_scale(scale),
                "--fixed_set_max_steps",
                "800",
                "--time_limit_sec",
                str(float(args.oracle_time_limit_sec)),
                "--max_teams_per_task",
                str(int(args.oracle_max_teams_per_task)),
                "--workers",
                str(int(args.oracle_workers)),
                "--include_deterministic_arrivals",
                "--omniscient_prepositioning",
                "--write_schedules",
            ]
            print(f"[oracle] scale={scale}", flush=True)
            run_cmd(cmd, cwd=workspace, log_path=log_root / f"oracle_{scale}.log")

    # 3) Train ours in the mixed-scale regime, one checkpoint per seed.
    #    Large-scale scenarios have a different difficulty profile than the
    #    standard 20-agent setting, so strict difficulty bins from the main
    #    protocol are NOT applied here by default.  The --strict_train_protocol
    #    flag can be used to force them, but this is likely to fail on s3+.
    if should_run(args.only_phase, "train"):
        train_preset = MIXED_TRAIN_PRESET
        train_seeds = [int(s) for s in parse_csv_list(args.train_seeds)]
        for seed in train_seeds:
            if len(train_seeds) > 1:
                ours_ckpt = resolve_path(workspace, args.ours_train_ckpt).with_name(
                    f"checkpoint_best_fixed_eval_seed{seed}.pt"
                )
            else:
                ours_ckpt = resolve_path(workspace, args.ours_train_ckpt)
            if args.skip_existing and ours_ckpt.is_file():
                print(f"[skip train] ours mixed_scale seed={seed}", flush=True)
                continue
            ours_ckpt.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                py,
                str(train_script),
                "--seed",
                str(seed),
                "--save_path",
                str(ours_ckpt),
                "--no_auto_tune_runtime",
                "--env_num_agents",
                str(int(train_preset["num_agents"])),
                "--env_max_tasks",
                str(int(train_preset["max_tasks"])),
                "--env_init_tasks_min",
                str(int(train_preset["init_min"])),
                "--env_init_tasks_max",
                str(int(train_preset["init_max"])),
                "--env_total_tasks_floor",
                str(int(train_preset["total_floor"])),
                "--env_total_tasks_cap",
                str(int(train_preset["total_cap"])),
                "--env_max_steps",
                "800",
                "--env_dynamic_arrival_every",
                str(int(train_preset["dynamic_every"])),
                "--env_dynamic_arrival_min",
                str(int(train_preset["dynamic_min"])),
                "--env_dynamic_arrival_max",
                str(int(train_preset["dynamic_max"])),
                "--env_edge_density",
                str(float(args.edge_density)),
            ]
            if bool(args.strict_train_protocol):
                cmd.extend([
                    "--env_difficulty_min", "0.2",
                    "--env_difficulty_max", "0.8",
                    "--env_difficulty_bins",
                    "0.20:0.25,0.25:0.30,0.30:0.35,0.35:0.40,0.40:0.45,0.45:0.50,"
                    "0.50:0.55,0.55:0.60,0.60:0.65,0.65:0.70,0.70:0.75,0.75:0.80",
                    "--enforce_env_difficulty",
                    "--env_difficulty_max_attempts", "160",
                    "--curriculum_steps", "0",
                    "--env_hard_focus_prob", "0.0",
                ])
            else:
                # Large-scale training: enforce feasibility only (any difficulty).
                cmd.extend([
                    "--env_difficulty_min", "0.0",
                    "--env_difficulty_max", "1.0",
                    "--no_enforce_env_difficulty",
                    "--env_difficulty_max_attempts", "32",
                    "--curriculum_steps", "0",
                    "--env_hard_focus_prob", "0.0",
                ])
            # 大规模图(28 agents × 160 max_tasks)下 critic pair_encoder 的 4D 张量
            # [B, A, T, H] 显存随 A×T 急剧增长；默认 minibatch_size=1536 在此规模下 OOM
            # (实测需 ~22.8GiB)。将 PPO 更新 minibatch 降到 384(实测峰值 ~16.3GiB，留有余量)，
            # 保证三 seed 训练在单卡 24GB 上可完成。若用户已通过 train_extra_args 显式指定
            # minibatch_size 则尊重之。
            if not any("minibatch_size" in str(a) for a in args.train_extra_args):
                cmd.extend(["--minibatch_size", "384"])
            cmd.extend(args.train_extra_args)
            print(f"[train] ours mixed_scale seed={seed}", flush=True)
            run_cmd(cmd, cwd=workspace, log_path=log_root / f"train_ours_mixed_scale_seed{seed}.log")

    # 4) Evaluate trained models and online baselines on matching fixed sets.
    if should_run(args.only_phase, "eval"):
        eval_methods = ["ours", "mappo", "cbs_ta_ptc", "ga", "ctas_d"]
        train_seeds = [int(s) for s in parse_csv_list(args.train_seeds)]
        for scale in scales:
            fixed_dir = fixed_root / scale
            for method in eval_methods:
                # For ours with multiple seeds, evaluate each seed separately.
                seed_list = train_seeds if method == "ours" else [None]
                for seed in seed_list:
                    if seed is not None and len(train_seeds) > 1:
                        tag = eval_run_tag(method, scale) + f"_seed{seed}"
                        ours_ckpt_for_seed = resolve_path(workspace, args.ours_train_ckpt).with_name(
                            f"checkpoint_best_fixed_eval_seed{seed}.pt"
                        )
                    else:
                        tag = eval_run_tag(method, scale)
                        ours_ckpt_for_seed = resolve_path(workspace, args.ours_train_ckpt)

                    run_dir = eval_root / tag
                    if args.skip_existing and has_eval_artifact(run_dir):
                        print(f"[skip eval] {tag}", flush=True)
                        continue

                    cmd = [py, str(eval_script)]
                    if method in {"ours", "mappo"}:
                        if method == "ours":
                            ckpt = ours_ckpt_for_seed
                        else:
                            ckpt = resolve_path(workspace, args.mappo_ckpt)
                        if not ckpt.is_file():
                            raise FileNotFoundError(f"checkpoint not found for method={method}: {ckpt}")
                        cmd.extend(["--checkpoint", str(ckpt)])
                        # Use the env config stored in the checkpoint for fair evaluation.
                        cmd.append("--use_checkpoint_env_config")
                    else:
                        cmd.extend(["--baseline", "--baseline_method", method])

                    cmd.extend([
                        "--fixed_set_path",
                        str(fixed_dir),
                        "--episodes",
                        "0",
                        "--fixed_set_max_steps",
                        "800",
                        "--method_alias",
                        method,
                        "--output_dir",
                        str(eval_root),
                        "--output_tag",
                        tag,
                        "--no_plot",
                    ])

                    if not has_option(args.eval_extra_args, "--oracle_metrics_csv"):
                        oracle_csv = oracle_csv_for_scale(oracle_root, scale)
                        if oracle_csv.is_file():
                            cmd.extend(["--oracle_metrics_csv", str(oracle_csv)])
                        else:
                            raise FileNotFoundError(
                                f"oracle csv not found for scale={scale}: {oracle_csv}. Run --only_phase oracle first."
                            )

                    if method == "mappo" and bool(args.mappo_wo_mask):
                        cmd.append("--wo_mask")
                    if method == "ours":
                        cmd.extend(["--policy_decode", str(args.ours_policy_decode)])
                        cmd.extend(["--policy_explore_eps", str(float(args.ours_policy_explore_eps))])
                    # Traditional baselines (cbs_ta_ptc / ga / ctas_d) can stall on large-scale
                    # scenes. Cap each decision and the whole episode by wall-clock so a single
                    # slow scene cannot block the suite; timed-out episodes are truncated and
                    # still record partial recognized_done_rate. Respect explicit user overrides
                    # passed via eval_extra_args.
                    if method in {"cbs_ta_ptc", "ga", "ctas_d"}:
                        if not has_option(args.eval_extra_args, "--baseline_timeout_sec"):
                            cmd.extend(["--baseline_timeout_sec", "5.0"])
                        # 120s per-episode wall-clock cutoff: 4x the common 30s MAPF benchmark
                        # convention (Okumura 2023 / LaCAM) and aligned with the 1-minute bound
                        # used by Li et al.; keeps the suite within a 1-2 day budget while still
                        # generous to traditional baselines on large scales.
                        if not has_option(args.eval_extra_args, "--baseline_episode_timeout_sec"):
                            cmd.extend(["--baseline_episode_timeout_sec", "120.0"])
                    cmd.extend(args.eval_extra_args)

                    print(f"[eval] {tag}", flush=True)
                    run_cmd(cmd, cwd=workspace, log_path=log_root / f"eval_{tag}.log")

    # 5) Plot A from eval artifacts.
    if should_run(args.only_phase, "plot_a"):
        rows: List[Dict[str, object]] = []
        eval_methods = ["ours", "mappo", "cbs_ta_ptc", "ga", "ctas_d"]
        train_seeds = [int(s) for s in parse_csv_list(args.train_seeds)]
        for scale in scales:
            for method in eval_methods:
                seed_list = train_seeds if method == "ours" else [None]
                for seed in seed_list:
                    if seed is not None and len(train_seeds) > 1:
                        tag = eval_run_tag(method, scale) + f"_seed{seed}"
                    else:
                        tag = eval_run_tag(method, scale)
                    agg_csv = eval_root / tag / "aggregate_metrics.csv"
                    if not agg_csv.is_file():
                        print(f"warning: missing {agg_csv}, skipping", flush=True)
                        continue
                    metrics = read_aggregate_csv(agg_csv)
                    method_label = f"{method}_seed{seed}" if seed is not None and len(train_seeds) > 1 else method
                    rows.append(
                        {
                            "scale": scale,
                            "scale_label": scale_axis_label(scale),
                            "method": method_label,
                            "return_mean": metric_val(metrics, "return", "mean"),
                            "return_std": metric_val(metrics, "return", "std"),
                            "cpsat_relative_done_mean": metric_val(metrics, "cpsat_relative_done", "mean"),
                            "cpsat_relative_done_std": metric_val(metrics, "cpsat_relative_done", "std"),
                            "recognized_done_tasks_mean": metric_val(metrics, "recognized_done_tasks", "mean"),
                            "recognized_done_tasks_std": metric_val(metrics, "recognized_done_tasks", "std"),
                            "recognized_done_rate_mean": metric_val(metrics, "recognized_done_rate", "mean"),
                            "recognized_done_rate_std": metric_val(metrics, "recognized_done_rate", "std"),
                            "runtime_mean": metric_val(metrics, "infer_time", "mean"),
                            "runtime_std": metric_val(metrics, "infer_time", "std"),
                            "violation_rate_mean": metric_val(metrics, "violation_rate", "mean"),
                            "violation_rate_std": metric_val(metrics, "violation_rate", "std"),
                            "throughput_mean": metric_val(metrics, "throughput", "mean"),
                            "infer_time_mean": metric_val(metrics, "infer_time", "mean"),
                        }
                    )

        if not rows:
            raise RuntimeError("no eval artifacts found for plotting; run --only_phase eval first")

        a_dir = plot_root / "A"
        a_dir.mkdir(parents=True, exist_ok=True)
        write_csv(
            a_dir / "A_metrics_vs_scale.csv",
            rows,
            fieldnames=[
                "scale",
                "scale_label",
                "method",
                "return_mean",
                "return_std",
                "cpsat_relative_done_mean",
                "cpsat_relative_done_std",
                "recognized_done_tasks_mean",
                "recognized_done_tasks_std",
                "recognized_done_rate_mean",
                "recognized_done_rate_std",
                "runtime_mean",
                "runtime_std",
                "violation_rate_mean",
                "violation_rate_std",
                "throughput_mean",
                "infer_time_mean",
            ],
        )
        plot_a_from_rows(rows=rows, out_dir=a_dir, title="A. Cross-Scale Generalization Metrics")
        print(f"saved_A_dir={a_dir.as_posix()}", flush=True)

    # Save suite summary for traceability.
    summary = {
        "scales": scales,
        "num_envs": int(args.num_envs),
        "fixed_root": fixed_root.as_posix(),
        "save_root": save_root.as_posix(),
        "eval_root": eval_root.as_posix(),
        "plot_root": plot_root.as_posix(),
        "oracle_root": oracle_root.as_posix(),
        "chunk_size": int(args.chunk_size),
        "mappo_wo_mask": bool(args.mappo_wo_mask),
        "mappo_update_epochs": int(args.mappo_update_epochs),
        "ours_train_ckpt": resolve_path(workspace, args.ours_train_ckpt).as_posix(),
        "mappo_ckpt": resolve_path(workspace, args.mappo_ckpt).as_posix(),
        "ours_policy_decode": str(args.ours_policy_decode),
        "ours_policy_explore_eps": float(args.ours_policy_explore_eps),
        "preview_only": bool(args.preview_only),
        "preview_num_envs": int(args.preview_num_envs),
        "only_phase": str(args.only_phase),
        "train_seeds": parse_csv_list(args.train_seeds),
        "strict_train_protocol": bool(args.strict_train_protocol),
        "scale_difficulty_min": args.scale_difficulty_min,
        "scale_difficulty_max": args.scale_difficulty_max,
    }
    summary_path = plot_root / "suite_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("done", flush=True)
    print(f"suite_summary={summary_path.as_posix()}", flush=True)


if __name__ == "__main__":
    main()
