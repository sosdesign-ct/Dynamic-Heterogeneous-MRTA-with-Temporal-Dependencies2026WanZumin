from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from ortools.sat.python import cp_model

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from env.fixed_set_loader import load_fixed_env_from_path, materialize_fixed_dynamic_arrivals, resolve_fixed_scene_paths


Team = Tuple[int, ...]


def _safe_mean(values: Iterable[float]) -> float:
    vals = list(values)
    return float(np.mean(np.asarray(vals, dtype=np.float64))) if vals else 0.0


def _safe_std(values: Iterable[float]) -> float:
    vals = list(values)
    return float(np.std(np.asarray(vals, dtype=np.float64))) if vals else 0.0


def _ceil_travel_steps(a: np.ndarray, b: np.ndarray, speed: float, dt_seconds: float) -> int:
    del dt_seconds  # Movement ETA in MRTAEnv is already expressed in environment steps.
    return int(math.ceil(float(np.linalg.norm(a - b)) / max(float(speed), 1e-6)))


def _env_travel_steps(env, a: np.ndarray, b: np.ndarray, speed: float) -> int:
    """Mirror MRTAEnv pre-gathering ETA exactly for oracle route constraints."""
    distance = float(np.linalg.norm(a - b))
    if bool(getattr(env.config, "allow_pre_gathering", True)):
        distance = max(0.0, distance - float(max(getattr(env.config, "pre_gather_radius", 0.0), 0.0)))
    return int(math.ceil(distance / max(float(speed), 1e-6)))


def _extend_scene_to_dynamic_cap(env, max_steps: int) -> None:
    """Replay deterministic fixed-scene dynamic arrivals so the oracle matches eval total_tasks."""
    materialize_fixed_dynamic_arrivals(env, max_steps)


def _duration_for_team(env, task, team: Team) -> int:
    if task.exec_mode == "DISTANCE":
        dst = task.dst_loc if task.dst_loc is not None else task.loc
        distance = float(np.linalg.norm(task.loc - dst))
        speed = float(max(env.type_speeds[int(task.task_type)], 1e-6))
        physical_time = distance / speed
    else:
        demand = float(task.req[int(task.task_type)])
        workload_ref = max((env.config.task_workload_range[0] + env.config.task_workload_range[1]) * 0.5, 1e-6)
        workload_factor = float(task.workload) / workload_ref
        team_cap = float(sum(env.robots[int(rid)].cap[int(task.task_type)] for rid in team))
        scale = float(task.type_time_scale)
        physical_time = demand * workload_factor / max(team_cap, 1e-6) * scale
        t_min, t_max = env.non_distance_time_bounds
        physical_time = float(np.clip(physical_time, t_min, t_max))
    physical_time = max(physical_time, float(env.config.dt_seconds))
    return max(1, int(math.ceil(physical_time / max(float(env.config.dt_seconds), 1e-6))))


def _feasible_teams(env, task, max_teams_per_task: int) -> List[Tuple[Team, int]]:
    typed_robots = [int(rid) for rid, r in sorted(env.robots.items()) if int(r.agent_type) == int(task.task_type)]
    k = int(task.k)
    if k <= 0 or len(typed_robots) < k:
        return []
    req = float(task.req[int(task.task_type)])
    rows: List[Tuple[float, int, Team, int]] = []
    for team in combinations(typed_robots, k):
        cap = float(sum(env.robots[rid].cap[int(task.task_type)] for rid in team))
        if cap + 1e-9 < req:
            continue
        dur = _duration_for_team(env, task, tuple(int(x) for x in team))
        # Prefer shorter and stronger teams when limiting candidate count.
        rows.append((float(dur), -cap, tuple(int(x) for x in team), int(dur)))
    rows.sort(key=lambda x: (x[0], x[1], x[2]))
    if max_teams_per_task > 0:
        rows = rows[: int(max_teams_per_task)]
    return [(team, dur) for _, _, team, dur in rows]


def solve_scene(
    scene_path: Path,
    max_steps: int = 800,
    time_limit_sec: float = 60.0,
    max_teams_per_task: int = 80,
    workers: int = 8,
    include_deterministic_arrivals: bool = False,
    omniscient_prepositioning: bool = False,
    target_min_done: int = 0,
    feasibility_only: bool = False,
    random_seed: int = 0,
) -> Dict[str, object]:
    env = load_fixed_env_from_path(scene_path, wo_commitment=False)
    env.config.max_steps = int(max_steps)
    env.config.use_policy_commitment_logic = False
    env.config.capacity_topk_enabled = False
    if include_deterministic_arrivals:
        _extend_scene_to_dynamic_cap(env, max_steps=max_steps)
    env.config.dynamic_arrival_every = 0
    env.config.dynamic_arrival_range = (0, 0)
    env._refresh_task_statuses()
    env._quick_generation_global_check()

    task_ids = sorted(int(tid) for tid in env.tasks.keys())
    robot_ids = sorted(int(rid) for rid in env.robots.keys())
    horizon = int(max_steps)
    big_m = horizon * 4 + 1000

    team_options: Dict[int, List[Tuple[Team, int]]] = {}
    for tid in task_ids:
        team_options[tid] = _feasible_teams(env, env.tasks[tid], max_teams_per_task=max_teams_per_task)

    model = cp_model.CpModel()
    selected = {tid: model.NewBoolVar(f"selected_{tid}") for tid in task_ids}
    start = {tid: model.NewIntVar(0, horizon, f"start_{tid}") for tid in task_ids}
    finish = {tid: model.NewIntVar(0, horizon, f"finish_{tid}") for tid in task_ids}
    duration = {tid: model.NewIntVar(0, horizon, f"duration_{tid}") for tid in task_ids}

    team_vars: Dict[Tuple[int, int], cp_model.IntVar] = {}
    robot_task_vars: Dict[Tuple[int, int], cp_model.IntVar] = {}
    for tid in task_ids:
        opts = team_options[tid]
        ys = []
        for j, (_, dur) in enumerate(opts):
            y = model.NewBoolVar(f"team_{tid}_{j}")
            team_vars[(tid, j)] = y
            ys.append(y)
        if ys:
            model.Add(sum(ys) == selected[tid])
            model.Add(duration[tid] == sum(int(opts[j][1]) * team_vars[(tid, j)] for j in range(len(opts))))
        else:
            model.Add(selected[tid] == 0)
            model.Add(duration[tid] == 0)
        model.Add(start[tid] >= int(env.tasks[tid].release_time)).OnlyEnforceIf(selected[tid])
        model.Add(finish[tid] == start[tid] + duration[tid]).OnlyEnforceIf(selected[tid])
        model.Add(finish[tid] <= horizon).OnlyEnforceIf(selected[tid])

    for rid in robot_ids:
        for tid in task_ids:
            opts = team_options[tid]
            contains = [team_vars[(tid, j)] for j, (team, _) in enumerate(opts) if int(rid) in team]
            x = model.NewBoolVar(f"x_r{rid}_t{tid}")
            robot_task_vars[(rid, tid)] = x
            if contains:
                model.Add(x == sum(contains))
            else:
                model.Add(x == 0)

    rel_window = {str(k): int(v) for k, v in env.rel_window_steps.items()}
    for edge in env.edges:
        u = int(edge.u)
        v = int(edge.v)
        lag = int(edge.lag)
        w = int(rel_window.get(str(edge.rel_type), 30))
        model.Add(selected[v] <= selected[u])
        if edge.rel_type == "FS":
            model.Add(start[v] >= finish[u] + lag).OnlyEnforceIf(selected[v])
            model.Add(start[v] <= finish[u] + lag + w).OnlyEnforceIf(selected[v])
        elif edge.rel_type == "SS":
            model.Add(start[v] >= start[u] + lag).OnlyEnforceIf(selected[v])
            model.Add(start[v] <= start[u] + lag + w).OnlyEnforceIf(selected[v])
        elif edge.rel_type == "FF":
            model.Add(finish[v] >= finish[u] + lag).OnlyEnforceIf(selected[v])
            model.Add(finish[v] <= finish[u] + lag + w).OnlyEnforceIf(selected[v])
        elif edge.rel_type == "SF":
            model.Add(finish[v] >= start[u] + lag).OnlyEnforceIf(selected[v])
            model.Add(finish[v] <= start[u] + lag + w).OnlyEnforceIf(selected[v])

    # Per-robot route circuits add exact travel/setup times between the tasks that a robot serves.
    for rid in robot_ids:
        candidates = [tid for tid in task_ids if team_options[tid] and int(env.tasks[tid].task_type) == int(env.robots[rid].agent_type)]
        if not candidates:
            continue
        idx = {tid: i + 1 for i, tid in enumerate(candidates)}
        arcs = []
        for tid in candidates:
            x = robot_task_vars[(rid, tid)]
            self_loop = model.NewBoolVar(f"self_r{rid}_t{tid}")
            model.Add(self_loop + x == 1)
            arcs.append((idx[tid], idx[tid], self_loop))

            lit_from_depot = model.NewBoolVar(f"arc_r{rid}_0_{tid}")
            arcs.append((0, idx[tid], lit_from_depot))
            travel0 = _env_travel_steps(env, env.robots[rid].pos, env.tasks[tid].loc, env.robots[rid].v_max)
            if bool(omniscient_prepositioning):
                # The oracle knows future tasks at t=0 and may travel before release,
                # but execution itself still cannot begin before release_time.
                model.Add(start[tid] >= travel0).OnlyEnforceIf(lit_from_depot)
            else:
                model.Add(start[tid] >= int(env.tasks[tid].release_time) + travel0).OnlyEnforceIf(lit_from_depot)

            lit_to_depot = model.NewBoolVar(f"arc_r{rid}_{tid}_0")
            arcs.append((idx[tid], 0, lit_to_depot))

        if not candidates:
            arcs.append((0, 0, model.NewConstant(1)))
        else:
            depot_self = model.NewBoolVar(f"arc_r{rid}_0_0")
            served_count = sum(robot_task_vars[(rid, tid)] for tid in candidates)
            model.Add(served_count == 0).OnlyEnforceIf(depot_self)
            model.Add(served_count >= 1).OnlyEnforceIf(depot_self.Not())
            arcs.append((0, 0, depot_self))

        for i_tid in candidates:
            for j_tid in candidates:
                if i_tid == j_tid:
                    continue
                lit = model.NewBoolVar(f"arc_r{rid}_{i_tid}_{j_tid}")
                arcs.append((idx[i_tid], idx[j_tid], lit))
                travel = _env_travel_steps(
                    env,
                    env.tasks[i_tid].loc,
                    env.tasks[j_tid].loc,
                    env.robots[rid].v_max,
                )
                model.Add(start[j_tid] >= finish[i_tid] + travel - big_m * (1 - lit))
        model.AddCircuit(arcs)

    # Maximize compliant completions. Keep the objective pure so CP-SAT bounds are task-count bounds.
    selected_count = sum(selected[tid] for tid in task_ids)
    if int(target_min_done) > 0:
        model.Add(selected_count >= int(target_min_done))
    if not bool(feasibility_only):
        model.Maximize(selected_count)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_sec)
    solver.parameters.num_search_workers = max(1, int(workers))
    solver.parameters.random_seed = int(random_seed)

    t0 = time.perf_counter()
    status = solver.Solve(model)
    runtime = time.perf_counter() - t0
    status_name = solver.StatusName(status)

    objective_count = 0
    best_bound_count = 0.0
    schedule: List[Dict[str, object]] = []
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        objective_count = int(round(sum(int(solver.Value(selected[tid])) for tid in task_ids)))
        for tid in task_ids:
            if int(solver.Value(selected[tid])) <= 0:
                continue
            chosen_team: Optional[Team] = None
            chosen_dur = 0
            for j, (team, dur) in enumerate(team_options[tid]):
                if int(solver.Value(team_vars[(tid, j)])) > 0:
                    chosen_team = team
                    chosen_dur = int(dur)
                    break
            schedule.append(
                {
                    "task_id": int(tid),
                    "team": list(chosen_team or []),
                    "start": int(solver.Value(start[tid])),
                    "finish": int(solver.Value(finish[tid])),
                    "duration": int(chosen_dur),
                    "release_time": int(env.tasks[tid].release_time),
                    "task_type": int(env.tasks[tid].task_type),
                }
            )
        best_bound_count = float(solver.BestObjectiveBound())
    else:
        best_bound_count = 0.0

    total_tasks = len(task_ids)
    return {
        "scene_path": scene_path.as_posix(),
        "status": status_name,
        "is_optimal": int(status == cp_model.OPTIMAL),
        "max_recognized_done_tasks": int(objective_count),
        "upper_bound_tasks": float(best_bound_count),
        "total_tasks": int(total_tasks),
        "max_recognized_done_rate": float(objective_count) / max(1, total_tasks),
        "team_options_total": int(sum(len(v) for v in team_options.values())),
        "runtime_sec": float(runtime),
        "omniscient_prepositioning": int(bool(omniscient_prepositioning)),
        "schedule": schedule,
    }


def aggregate(records: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    keys = [
        "max_recognized_done_tasks",
        "upper_bound_tasks",
        "total_tasks",
        "max_recognized_done_rate",
        "is_optimal",
        "team_options_total",
        "runtime_sec",
    ]
    return {k: {"mean": _safe_mean(float(r[k]) for r in records), "std": _safe_std(float(r[k]) for r in records)} for k in keys}


def _write_records_atomic(path: Path, records: Sequence[Dict[str, object]]) -> None:
    if not records:
        return
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    tmp_path.replace(path)


def _load_cached_records(path: Path) -> Dict[str, Dict[str, object]]:
    if not path.is_file():
        return {}
    with path.open("r", newline="", encoding="utf-8") as f:
        rows: Dict[str, Dict[str, object]] = {}
        numeric_fields = {
            "scene_index", "max_recognized_done_tasks", "total_tasks", "upper_bound_tasks",
            "team_options_total", "runtime_sec", "max_recognized_done_rate", "is_optimal",
        }
        for row in csv.DictReader(f):
            parsed: Dict[str, object] = dict(row)
            for field in numeric_fields:
                if field in parsed and parsed[field] != "":
                    parsed[field] = float(parsed[field]) if field in {"upper_bound_tasks", "runtime_sec", "max_recognized_done_rate"} else int(float(parsed[field]))
            rows[str(parsed["scene_path"])] = parsed
        return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CP-SAT oracle for fixed MRTA scenes: maximizes compliant task completion.")
    p.add_argument("--fixed_set_path", type=str, default="MRTAFixedTestSet")
    p.add_argument("--output_dir", type=str, default="results/cpsat_oracle")
    p.add_argument("--output_tag", type=str, default="fullset")
    p.add_argument("--max_scenes", type=int, default=0, help="0 means all scenes; use 35 for the 35-scene protocol.")
    p.add_argument("--scene_start", type=int, default=0, help="start index in the resolved fixed-scene list")
    p.add_argument("--fixed_set_max_steps", type=int, default=800)
    p.add_argument("--time_limit_sec", type=float, default=60.0)
    p.add_argument("--max_teams_per_task", type=int, default=80)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--random_seed", type=int, default=0, help="CP-SAT search seed")
    p.add_argument(
        "--iterative_proof_rounds",
        type=int,
        default=0,
        help="after a non-optimal maximize run, try this many incumbent+1 feasibility/proof rounds",
    )
    p.add_argument(
        "--proof_time_limit_sec",
        type=float,
        default=0.0,
        help="time limit per incumbent+1 proof round; <=0 reuses --time_limit_sec",
    )
    p.add_argument(
        "--prove_cached_incumbent",
        action="store_true",
        help="for cached non-optimal rows, directly test whether cached incumbent+1 is feasible",
    )
    p.add_argument(
        "--max_cached_gap",
        type=float,
        default=0.0,
        help="with --prove_cached_incumbent, only retry rows whose cached upper-bound gap is at most this value; <=0 disables filtering",
    )
    p.add_argument(
        "--min_cached_gap",
        type=float,
        default=0.0,
        help="with --prove_cached_incumbent, skip rows whose cached upper-bound gap is below this value",
    )
    p.add_argument(
        "--target_min_done",
        type=int,
        default=0,
        help="optional feasibility constraint: require at least this many compliant completions",
    )
    p.add_argument(
        "--feasibility_only",
        action="store_true",
        help="solve only feasibility after applying --target_min_done; useful for proving incumbent+1 impossible",
    )
    p.add_argument(
        "--include_deterministic_arrivals",
        action="store_true",
        help="include fixed RNG dynamic arrivals up to total_tasks_cap before solving; matches fixed-set eval total_tasks.",
    )
    p.add_argument(
        "--omniscient_prepositioning",
        action="store_true",
        help=(
            "give the oracle all future task/release information at t=0 and allow travel before release; "
            "task execution still respects release_time"
        ),
    )
    p.add_argument("--write_schedules", action="store_true")
    p.add_argument("--resume", action="store_true", help="resume from the per-scene partial CSV")
    p.add_argument(
        "--resume_recompute_nonoptimal",
        action="store_true",
        help="when resuming, keep proven-optimal rows but recompute cached FEASIBLE/UNKNOWN rows",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    scene_paths = resolve_fixed_scene_paths(args.fixed_set_path)
    scene_start = max(0, int(args.scene_start))
    scene_paths = scene_paths[scene_start:]
    if int(args.max_scenes) > 0:
        scene_paths = scene_paths[: int(args.max_scenes)]

    out_dir = Path(args.output_dir) / str(args.output_tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    schedule_dir = out_dir / "schedules"
    if bool(args.write_schedules):
        schedule_dir.mkdir(parents=True, exist_ok=True)

    partial_csv = out_dir / "episode_metrics.partial.csv"
    cached = _load_cached_records(partial_csv) if bool(args.resume) else {}
    partial_records: Dict[str, Dict[str, object]] = dict(cached)
    records: List[Dict[str, object]] = []
    for i, scene_path in enumerate(scene_paths):
        cached_row = cached.get(scene_path.as_posix())
        cached_gap = (
            float(cached_row.get("upper_bound_tasks", 0.0))
            - float(cached_row.get("max_recognized_done_tasks", 0))
            if cached_row is not None
            else float("inf")
        )
        keep_cached = cached_row is not None and (
            not bool(args.resume_recompute_nonoptimal) or int(cached_row.get("is_optimal", 0)) == 1
        )
        if (
            cached_row is not None
            and bool(args.prove_cached_incumbent)
            and (
                cached_gap < float(args.min_cached_gap)
                or (
                    float(args.max_cached_gap) > 0.0
                    and cached_gap > float(args.max_cached_gap)
                )
            )
        ):
            keep_cached = True
        if keep_cached:
            records.append(cached_row)
            print(f"scene={scene_start + i} resumed from partial cache", flush=True)
            continue
        if bool(args.prove_cached_incumbent) and cached_row is not None:
            incumbent = int(cached_row["max_recognized_done_tasks"])
            probe = solve_scene(
                scene_path=scene_path,
                max_steps=int(args.fixed_set_max_steps),
                time_limit_sec=(
                    float(args.proof_time_limit_sec)
                    if float(args.proof_time_limit_sec) > 0.0
                    else float(args.time_limit_sec)
                ),
                max_teams_per_task=int(args.max_teams_per_task),
                workers=int(args.workers),
                include_deterministic_arrivals=bool(args.include_deterministic_arrivals),
                omniscient_prepositioning=bool(args.omniscient_prepositioning),
                target_min_done=incumbent + 1,
                feasibility_only=True,
                random_seed=int(args.random_seed),
            )
            result = dict(cached_row)
            result["schedule"] = []
            result["runtime_sec"] = float(cached_row["runtime_sec"]) + float(probe["runtime_sec"])
            if str(probe["status"]) == "INFEASIBLE":
                result["status"] = "OPTIMAL_PROVEN"
                result["is_optimal"] = 1
                result["upper_bound_tasks"] = float(incumbent)
            elif str(probe["status"]) in {"FEASIBLE", "OPTIMAL"}:
                probe_obj = int(probe["max_recognized_done_tasks"])
                if probe_obj > incumbent:
                    result["status"] = "FEASIBLE"
                    result["max_recognized_done_tasks"] = probe_obj
                    result["max_recognized_done_rate"] = probe_obj / max(1, int(result["total_tasks"]))
                    if probe_obj == int(result["total_tasks"]):
                        result["status"] = "OPTIMAL_PROVEN"
                        result["is_optimal"] = 1
                        result["upper_bound_tasks"] = float(probe_obj)
        else:
            result = solve_scene(
                scene_path=scene_path,
                max_steps=int(args.fixed_set_max_steps),
                time_limit_sec=float(args.time_limit_sec),
                max_teams_per_task=int(args.max_teams_per_task),
                workers=int(args.workers),
                include_deterministic_arrivals=bool(args.include_deterministic_arrivals),
                omniscient_prepositioning=bool(args.omniscient_prepositioning),
                target_min_done=int(args.target_min_done),
                feasibility_only=bool(args.feasibility_only),
                random_seed=int(args.random_seed),
            )
        if int(result["is_optimal"]) == 0 and int(args.iterative_proof_rounds) > 0:
            incumbent = int(result["max_recognized_done_tasks"])
            proof_limit = (
                float(args.proof_time_limit_sec)
                if float(args.proof_time_limit_sec) > 0.0
                else float(args.time_limit_sec)
            )
            for proof_round in range(int(args.iterative_proof_rounds)):
                probe = solve_scene(
                    scene_path=scene_path,
                    max_steps=int(args.fixed_set_max_steps),
                    time_limit_sec=proof_limit,
                    max_teams_per_task=int(args.max_teams_per_task),
                    workers=int(args.workers),
                    include_deterministic_arrivals=bool(args.include_deterministic_arrivals),
                    omniscient_prepositioning=bool(args.omniscient_prepositioning),
                    target_min_done=incumbent + 1,
                    feasibility_only=True,
                    random_seed=int(args.random_seed) + proof_round + 1,
                )
                probe_status = str(probe["status"])
                if probe_status == "INFEASIBLE":
                    result["status"] = "OPTIMAL_PROVEN"
                    result["is_optimal"] = 1
                    result["upper_bound_tasks"] = float(incumbent)
                    result["runtime_sec"] = float(result["runtime_sec"]) + float(probe["runtime_sec"])
                    break
                probe_obj = int(probe["max_recognized_done_tasks"])
                if probe_status in {"FEASIBLE", "OPTIMAL"} and probe_obj > incumbent:
                    prior_runtime = float(result["runtime_sec"])
                    prior_bound = float(result["upper_bound_tasks"])
                    result = probe
                    result["status"] = "FEASIBLE"
                    result["is_optimal"] = 0
                    result["upper_bound_tasks"] = min(prior_bound, float(result["total_tasks"]))
                    result["runtime_sec"] = prior_runtime + float(probe["runtime_sec"])
                    incumbent = probe_obj
                    if incumbent == int(result["total_tasks"]):
                        result["status"] = "OPTIMAL_PROVEN"
                        result["is_optimal"] = 1
                        result["upper_bound_tasks"] = float(incumbent)
                        break
                    continue
                result["runtime_sec"] = float(result["runtime_sec"]) + float(probe["runtime_sec"])
                break
        schedule = result.pop("schedule")
        row = {"scene_index": int(scene_start + i), **result}
        if cached_row is not None and int(row["is_optimal"]) == 0:
            old_obj = int(cached_row.get("max_recognized_done_tasks", 0))
            new_obj = int(row["max_recognized_done_tasks"])
            old_bound = float(cached_row.get("upper_bound_tasks", float("inf")))
            new_bound = float(row["upper_bound_tasks"])
            if old_obj > new_obj or (old_obj == new_obj and old_bound <= new_bound):
                row = cached_row
                schedule = []
                print(
                    f"scene={scene_start + i} kept stronger cached result "
                    f"incumbent={old_obj} bound={old_bound:.1f}",
                    flush=True,
                )
        records.append(row)
        partial_records[str(row["scene_path"])] = row
        ordered_partial = sorted(
            partial_records.values(), key=lambda item: int(item.get("scene_index", 0))
        )
        _write_records_atomic(partial_csv, ordered_partial)
        if bool(args.write_schedules):
            (schedule_dir / f"scene_{scene_start + i:04d}_schedule.json").write_text(
                json.dumps(schedule, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        print(
            f"scene={scene_start + i} status={row['status']} optimal={row['is_optimal']} "
            f"max_done={row['max_recognized_done_tasks']}/{row['total_tasks']} "
            f"rate={row['max_recognized_done_rate']:.3f} bound={row['upper_bound_tasks']:.1f} "
            f"runtime={row['runtime_sec']:.2f}s",
            flush=True,
        )

    records_csv = out_dir / "episode_metrics.csv"
    _write_records_atomic(records_csv, records)

    agg = aggregate(records)
    aggregate_csv = out_dir / "aggregate_metrics.csv"
    with aggregate_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "mean", "std"])
        writer.writeheader()
        for metric, vals in agg.items():
            writer.writerow({"metric": metric, "mean": vals["mean"], "std": vals["std"]})

    summary = {"records": records, "aggregate": agg}
    summary_json = out_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nAggregate:")
    for metric, vals in agg.items():
        print(f"{metric}: mean={vals['mean']:.4f} std={vals['std']:.4f}")
    print(f"\nepisode_metrics={records_csv.as_posix()}")
    print(f"aggregate_metrics={aggregate_csv.as_posix()}")
    print(f"summary_json={summary_json.as_posix()}")


if __name__ == "__main__":
    main()
