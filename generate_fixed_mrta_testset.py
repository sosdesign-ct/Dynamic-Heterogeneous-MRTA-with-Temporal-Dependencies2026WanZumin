import argparse
import csv
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from env.mrta_env import MRTAEnv
from env.mrta_config import MRTAConfig


def build_config(args: argparse.Namespace) -> MRTAConfig:
    cfg = MRTAConfig()

    num_agents = int(args.num_agents)
    cap_dim = int(args.cap_dim)
    max_tasks = int(args.max_tasks)
    min_init_tasks = int(args.min_init_tasks)
    max_init_tasks = int(args.max_init_tasks)
    total_tasks_floor = int(args.total_tasks_floor)
    total_tasks_cap = int(args.total_tasks_cap)
    dynamic_arrival_every = int(args.dynamic_arrival_every)
    dynamic_arrival_min = int(args.dynamic_arrival_min)
    dynamic_arrival_max = int(args.dynamic_arrival_max)

    if num_agents <= 0:
        raise ValueError(f"num_agents must be > 0, got {num_agents}")
    if cap_dim <= 0:
        raise ValueError(f"cap_dim must be > 0, got {cap_dim}")
    if max_tasks <= 0:
        raise ValueError(f"max_tasks must be > 0, got {max_tasks}")
    if min_init_tasks < 0 or max_init_tasks < 0:
        raise ValueError(f"init task range must be >= 0, got ({min_init_tasks}, {max_init_tasks})")
    if min_init_tasks > max_init_tasks:
        raise ValueError(f"min_init_tasks > max_init_tasks: {min_init_tasks} > {max_init_tasks}")
    if total_tasks_floor < 0:
        raise ValueError(f"total_tasks_floor must be >= 0, got {total_tasks_floor}")
    if total_tasks_cap <= 0:
        raise ValueError(f"total_tasks_cap must be > 0, got {total_tasks_cap}")
    if dynamic_arrival_min > dynamic_arrival_max:
        raise ValueError(
            f"dynamic_arrival_min > dynamic_arrival_max: {dynamic_arrival_min} > {dynamic_arrival_max}"
        )

    cfg.num_agents = num_agents
    cfg.cap_dim = cap_dim
    cfg.max_tasks = max_tasks
    cfg.total_tasks_cap = min(total_tasks_cap, cfg.max_tasks)
    cfg.total_tasks_floor = min(total_tasks_floor, cfg.total_tasks_cap)
    cfg.init_tasks_range = (
        min(min_init_tasks, cfg.total_tasks_cap),
        min(max_init_tasks, cfg.total_tasks_cap),
    )
    cfg.edge_density = float(args.edge_density)
    if getattr(args, "lag_max", None) is not None:
        lo = int(cfg.lag_range[0])
        cfg.lag_range = (lo, max(lo, int(args.lag_max)))
    cfg.dynamic_arrival_every = dynamic_arrival_every
    cfg.dynamic_arrival_range = (dynamic_arrival_min, dynamic_arrival_max)
    if args.difficulty_min is not None or args.difficulty_max is not None:
        cfg.enforce_difficulty_range = True
        if args.difficulty_min is not None:
            cfg.difficulty_min = float(args.difficulty_min)
        if args.difficulty_max is not None:
            cfg.difficulty_max = float(args.difficulty_max)
    cfg.difficulty_max_reset_attempts = int(args.difficulty_max_reset_attempts)
    return cfg


def _round_list(values: List[float], ndigits: int = 6) -> List[float]:
    return [round(float(v), ndigits) for v in values]


def _attempts_used(base_seed: int, used_seed: int, seed_stride: int) -> int:
    if seed_stride <= 0:
        return 1
    return int((used_seed - base_seed) // seed_stride + 1)


def serialize_env(env: MRTAEnv, seed: int, cfg: MRTAConfig) -> Dict:
    robots = {}
    for rid, r in env.robots.items():
        robots[str(rid)] = {
            "id": int(rid),
            "type": int(r.agent_type),
            "pos": {"x": round(float(r.pos[0]), 6), "y": round(float(r.pos[1]), 6)},
            "v_max": round(float(r.v_max), 6),
            "cap": _round_list(r.cap.tolist()),
        }

    tasks = {}
    for tid, t in env.tasks.items():
        dst = None
        if t.dst_loc is not None:
            dst = {"x": round(float(t.dst_loc[0]), 6), "y": round(float(t.dst_loc[1]), 6)}
        tasks[str(tid)] = {
            "id": int(tid),
            "task_type": int(t.task_type),
            "exec_mode": str(t.exec_mode),
            "pos": {"x": round(float(t.loc[0]), 6), "y": round(float(t.loc[1]), 6)},
            "dst_pos": dst,
            "req": _round_list(t.req.tolist()),
            "workload": round(float(t.workload), 6),
            "demand_count": int(t.demand_count),
            "k": int(t.k),
            "release_time": int(t.release_time),
            "type_time_scale": round(float(t.type_time_scale), 6),
        }

    edges = []
    for e in env.edges:
        edges.append({
            "u": int(e.u),
            "v": int(e.v),
            "rel_type": str(e.rel_type),
            "lag": int(e.lag),
        })

    return {
        "setup": {
            "seed": int(seed),
            "num_agents": int(cfg.num_agents),
            "cap_dim": int(cfg.cap_dim),
            "num_tasks": int(len(tasks)),
            "num_edges": int(len(edges)),
            "max_tasks": int(cfg.max_tasks),
            "dynamic_arrival_every": int(cfg.dynamic_arrival_every),
            "dynamic_arrival_range": [int(cfg.dynamic_arrival_range[0]), int(cfg.dynamic_arrival_range[1])],
            "distance_task_types": [int(x) for x in cfg.distance_task_types],
        },
        "config": asdict(cfg),
        "type_speeds": _round_list(env.type_speeds.tolist()),
        "robots": robots,
        "tasks": tasks,
        "edges": edges,
    }


def generate_dataset(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overview_rows = []

    for i in range(args.num_envs):
        seed = args.base_seed + i
        cfg = build_config(args)
        env = None
        used_seed = seed
        gen_summary = {
            "feasible": 0,
            "horizon_violations": 0,
            "window_violations": 0,
        }
        for attempt in range(int(max(1, args.max_reset_attempts))):
            try_seed = int(seed + attempt * args.attempt_seed_stride)
            trial_env = MRTAEnv(cfg, seed=try_seed)
            trial_env.reset(seed=try_seed)
            summary = trial_env.generation_feasibility_summary()
            if int(summary.get("feasible", 0)) == 1:
                env = trial_env
                used_seed = try_seed
                gen_summary = summary
                break
            gen_summary = summary

        if env is None:
            raise RuntimeError(
                f"Failed to generate a feasible env_{i} after {args.max_reset_attempts} attempts "
                f"(base_seed={seed}, last_horizon_violations={gen_summary.get('horizon_violations', -1)}, "
                f"last_window_violations={gen_summary.get('window_violations', -1)})."
            )

        env_name = f"env_{i}"
        env_dir = out_dir / env_name
        env_dir.mkdir(parents=True, exist_ok=True)

        env_json = serialize_env(env, seed=used_seed, cfg=cfg)
        env_json["generation_check"] = {
            "attempts_used": _attempts_used(seed, used_seed, int(args.attempt_seed_stride)),
            "base_seed": int(seed),
            "used_seed": int(used_seed),
            "feasible": bool(int(gen_summary.get("feasible", 0))),
            "horizon_violations": int(gen_summary.get("horizon_violations", 0)),
            "window_violations": int(gen_summary.get("window_violations", 0)),
            "difficulty": float(env.current_scenario_difficulty),
            "difficulty_components": env.scenario_difficulty_summary(),
        }
        config_path = env_dir / "config.json"
        config_path.write_text(json.dumps(env_json, ensure_ascii=False, indent=2), encoding="utf-8")

        difficulty = env.scenario_difficulty_summary()
        overview_rows.append({
            "env_id": i,
            "env_name": env_name,
            "seed": used_seed,
            "robots": cfg.num_agents,
            "tasks": len(env.tasks),
            "cap_dim": cfg.cap_dim,
            "edge_density": cfg.edge_density,
            "difficulty": float(difficulty["score"]),
            "task_ratio": float(difficulty["task_ratio"]),
            "edge_ratio": float(difficulty["edge_ratio"]),
            "collab_ratio": float(difficulty["collab_ratio"]),
            "horizon_ratio": float(difficulty["horizon_ratio"]),
            "window_tightness": float(difficulty["window_tightness"]),
            "dynamic_ratio": float(difficulty["dynamic_ratio"]),
            "num_edges": len(env.edges),
            "attempts_used": _attempts_used(seed, used_seed, int(args.attempt_seed_stride)),
            "config_path": f"{env_name}/config.json",
        })

    overview_path = out_dir / "overview.csv"
    with overview_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "env_id",
                "env_name",
                "seed",
                "robots",
                "tasks",
                "cap_dim",
                "edge_density",
                "difficulty",
                "task_ratio",
                "edge_ratio",
                "collab_ratio",
                "horizon_ratio",
                "window_tightness",
                "dynamic_ratio",
                "num_edges",
                "attempts_used",
                "config_path",
            ],
        )
        writer.writeheader()
        writer.writerows(overview_rows)

    print(f"Generated fixed MRTA test set: {out_dir}")
    print(f"Total environments: {args.num_envs}")
    print(f"Overview file: {overview_path}")


def parse_args() -> argparse.Namespace:
    base_cfg = MRTAConfig()
    p = argparse.ArgumentParser(description="Generate a fixed MRTA test set with random but reproducible environments.")
    p.add_argument("--output_dir", type=str, default="MRTAFixedTestSet")
    p.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help="Overwrite output_dir by deleting old generated files before writing new test set.",
    )
    p.add_argument(
        "--no_overwrite",
        action="store_false",
        dest="overwrite",
        help="Keep existing files and only update generated file names.",
    )
    p.add_argument("--num_envs", type=int, default=35)
    p.add_argument("--base_seed", type=int, default=20260323)

    # 关键默认值直接继承 MRTAConfig，避免训练/固定集维度漂移导致不兼容。
    p.add_argument("--num_agents", type=int, default=int(base_cfg.num_agents))
    p.add_argument("--cap_dim", type=int, default=int(base_cfg.cap_dim))
    p.add_argument("--max_tasks", type=int, default=int(base_cfg.max_tasks))
    p.add_argument("--min_init_tasks", type=int, default=int(base_cfg.init_tasks_range[0]))
    p.add_argument("--max_init_tasks", type=int, default=int(base_cfg.init_tasks_range[1]))
    p.add_argument("--total_tasks_floor", type=int, default=int(base_cfg.total_tasks_floor))
    p.add_argument("--total_tasks_cap", type=int, default=int(base_cfg.total_tasks_cap))
    p.add_argument("--edge_density", type=float, default=float(base_cfg.edge_density))
    p.add_argument("--lag_max", type=int, default=None,
                   help="Optional override for dependency lag upper bound (default keeps config lag_range). "
                        "Raise it (e.g. 12) to deepen temporal coupling for graph-complexity test sets.")

    p.add_argument("--dynamic_arrival_every", type=int, default=int(base_cfg.dynamic_arrival_every),
                   help="Use 0 to disable dynamic arrivals and keep static test environments.")
    p.add_argument("--dynamic_arrival_min", type=int, default=int(base_cfg.dynamic_arrival_range[0]))
    p.add_argument("--dynamic_arrival_max", type=int, default=int(base_cfg.dynamic_arrival_range[1]))
    p.add_argument("--difficulty_min", type=float, default=None,
                   help="Optional scenario difficulty lower bound in [0,1].")
    p.add_argument("--difficulty_max", type=float, default=None,
                   help="Optional scenario difficulty upper bound in [0,1].")
    p.add_argument("--difficulty_max_reset_attempts", type=int, default=int(base_cfg.difficulty_max_reset_attempts),
                   help="Reset retries used by difficulty-constrained generation.")
    p.add_argument("--max_reset_attempts", type=int, default=8,
                   help="Maximum reset retries per environment until a feasible graph is generated.")
    p.add_argument("--attempt_seed_stride", type=int, default=9973,
                   help="Seed increment per retry to avoid repeated infeasible samples.")
    return p.parse_args()


if __name__ == "__main__":
    generate_dataset(parse_args())
