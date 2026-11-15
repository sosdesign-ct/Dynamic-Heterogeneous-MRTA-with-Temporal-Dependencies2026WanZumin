import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from env.mrta_config import MRTAConfig
from env.mrta_env import DependencyEdge, MRTAEnv, Robot, Task


def _coerce_tuple2(value, cast_fn):
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (cast_fn(value[0]), cast_fn(value[1]))
    raise ValueError(f"expected length-2 sequence, got: {value}")


def resolve_fixed_scene_paths(source_path: str) -> List[Path]:
    src = Path(source_path)
    if not src.exists():
        raise FileNotFoundError(f"Fixed set path does not exist: {src}")

    if src.is_file():
        if src.name.lower() != "config.json":
            raise ValueError(f"Expected fixed scene file name to be config.json, got: {src.name}")
        return [src]

    def env_index(path: Path) -> int:
        try:
            return int(path.parent.name.split("_", 1)[1])
        except Exception:
            return 10**9

    paths = sorted(src.glob("env_*/config.json"), key=lambda p: (env_index(p), p.as_posix()))
    if not paths:
        raise FileNotFoundError(
            f"No fixed scenario config found under: {src}. Expected files like env_0/config.json"
        )
    return paths


def load_fixed_env_from_path(path: Path, wo_commitment: bool, strict_start_in_mask: Optional[bool] = None) -> MRTAEnv:
    try:
        raw_text = path.read_text(encoding="utf-8")
    except Exception as exc:
        raise ValueError(f"Failed to read fixed scene config: {path.as_posix()} ({exc})") from exc

    if not raw_text.strip():
        raise ValueError(f"Fixed scene config is empty: {path.as_posix()}")

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in fixed scene config: {path.as_posix()} ({exc})") from exc

    cfg_map = data.get("config", {})

    default_cfg = MRTAConfig()
    cfg = MRTAConfig()
    if isinstance(cfg_map, dict):
        for k, v in cfg_map.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    cfg.area_size = _coerce_tuple2(cfg.area_size, float)
    cfg.v_max_range = _coerce_tuple2(cfg.v_max_range, float)
    cfg.init_tasks_range = _coerce_tuple2(cfg.init_tasks_range, int)
    cfg.dynamic_arrival_range = _coerce_tuple2(cfg.dynamic_arrival_range, int)
    cfg.lag_range = _coerce_tuple2(cfg.lag_range, int)
    cfg.agent_type_cap_range = _coerce_tuple2(cfg.agent_type_cap_range, float)
    cfg.task_demand_range = _coerce_tuple2(cfg.task_demand_range, float)
    cfg.task_workload_range = _coerce_tuple2(cfg.task_workload_range, float)
    cfg.task_k_range = _coerce_tuple2(cfg.task_k_range, int)
    cfg.non_distance_time_bounds = _coerce_tuple2(cfg.non_distance_time_bounds, float)
    cfg.distance_task_types = tuple(int(x) for x in cfg.distance_task_types)
    cfg.non_distance_type_scales = tuple(float(x) for x in cfg.non_distance_type_scales)
    cfg.ordered_commitment = not bool(wo_commitment)
    cfg.use_policy_commitment_logic = not bool(wo_commitment)
    cfg.strict_start_in_mask = (
        bool(default_cfg.strict_start_in_mask)
        if strict_start_in_mask is None
        else bool(strict_start_in_mask)
    )

    seed = int(data.get("setup", {}).get("seed", 0))
    env = MRTAEnv(cfg, seed=seed)
    env.reset(seed=seed)

    env.current_time = 0
    env.next_task_id = 0
    env.rejected_commitments = 0
    env.missed_ss_window = 0
    env.infeasible_team = 0
    env.waiting_cost = 0
    env.logical_wait_cost = 0
    env.total_done = 0

    type_speeds = np.array(data.get("type_speeds", []), dtype=np.float64)
    if type_speeds.size == cfg.cap_dim:
        env.type_speeds = type_speeds

    env.robots = {}
    for rid_str, rob in data.get("robots", {}).items():
        rid = int(rid_str)
        env.robots[rid] = Robot(
            robot_id=rid,
            pos=np.array([float(rob["pos"]["x"]), float(rob["pos"]["y"])], dtype=np.float64),
            v_max=float(rob["v_max"]),
            cap=np.array(rob["cap"], dtype=np.float64),
            state="IDLE",
            task_id=None,
            eta=0,
            work_left=0,
            agent_type=int(rob["type"]),
        )

    env.tasks = {}
    max_tid = -1
    for tid_str, task_dict in data.get("tasks", {}).items():
        tid = int(tid_str)
        max_tid = max(max_tid, tid)
        dst = task_dict.get("dst_pos")
        dst_loc = None
        if dst is not None:
            dst_loc = np.array([float(dst["x"]), float(dst["y"])], dtype=np.float64)
        env.tasks[tid] = Task(
            task_id=tid,
            loc=np.array([float(task_dict["pos"]["x"]), float(task_dict["pos"]["y"])], dtype=np.float64),
            dst_loc=dst_loc,
            req=np.array(task_dict["req"], dtype=np.float64),
            workload=float(task_dict["workload"]),
            demand_count=int(task_dict["demand_count"]),
            k=int(task_dict["k"]),
            release_time=int(task_dict["release_time"]),
            task_type=int(task_dict["task_type"]),
            exec_mode=str(task_dict.get("exec_mode", "NON_DISTANCE")),
            type_time_scale=float(task_dict.get("type_time_scale", 180.0)),
            status="UNRELEASED",
        )

    env.next_task_id = max_tid + 1
    env.edges = [
        DependencyEdge(
            u=int(edge["u"]),
            v=int(edge["v"]),
            rel_type=str(edge["rel_type"]),
            lag=int(edge["lag"]),
        )
        for edge in data.get("edges", [])
    ]
    env._refresh_task_statuses()
    feasible = env._quick_generation_global_check()
    diff = env.scenario_difficulty_summary()
    env.current_scenario_difficulty = float(diff.get("score", 0.0))
    env.current_scenario_difficulty_components = diff
    env.last_reset_attempts = 1
    env.last_reset_range_ok = True
    env.last_generation_feasible = bool(feasible)
    return env


def materialize_fixed_dynamic_arrivals(env: MRTAEnv, max_steps: int) -> MRTAEnv:
    """Pre-generate deterministic arrivals while preserving online visibility."""
    # Arrival generation does not need policy-side candidate filtering, but those
    # flags are part of the evaluated algorithm and must survive materialization.
    use_policy_commitment_logic = bool(env.config.use_policy_commitment_logic)
    capacity_topk_enabled = bool(env.config.capacity_topk_enabled)
    try:
        env.config.use_policy_commitment_logic = False
        env.config.capacity_topk_enabled = False
        env.current_time = 0
        for t in range(0, int(max_steps)):
            env.current_time = int(t)
            before = len(env.tasks)
            env._dynamic_arrival()
            if len(env.tasks) > before:
                for tid in range(before, len(env.tasks)):
                    if tid in env.tasks:
                        env.tasks[tid].status = "UNRELEASED"
        env.current_time = 0
        env._refresh_task_statuses()
        env.config.dynamic_arrival_every = 0
        env.config.dynamic_arrival_range = (0, 0)
    finally:
        env.config.use_policy_commitment_logic = use_policy_commitment_logic
        env.config.capacity_topk_enabled = capacity_topk_enabled
    return env


def apply_config_overrides_to_env(env: MRTAEnv, cfg_overrides: Optional[Dict], strict_start_in_mask: Optional[bool] = None) -> MRTAEnv:
    if not isinstance(cfg_overrides, dict):
        return env
    default_cfg = MRTAConfig()
    ordered_commitment = bool(getattr(env.config, "ordered_commitment", True))
    use_policy_commitment_logic = bool(getattr(env.config, "use_policy_commitment_logic", True))
    for k, v in cfg_overrides.items():
        if hasattr(env.config, k):
            setattr(env.config, k, v)
    env.config.area_size = _coerce_tuple2(env.config.area_size, float)
    env.config.v_max_range = _coerce_tuple2(env.config.v_max_range, float)
    env.config.init_tasks_range = _coerce_tuple2(env.config.init_tasks_range, int)
    env.config.dynamic_arrival_range = _coerce_tuple2(env.config.dynamic_arrival_range, int)
    env.config.lag_range = _coerce_tuple2(env.config.lag_range, int)
    env.config.agent_type_cap_range = _coerce_tuple2(env.config.agent_type_cap_range, float)
    env.config.task_demand_range = _coerce_tuple2(env.config.task_demand_range, float)
    env.config.task_workload_range = _coerce_tuple2(env.config.task_workload_range, float)
    env.config.task_k_range = _coerce_tuple2(env.config.task_k_range, int)
    env.config.non_distance_time_bounds = _coerce_tuple2(env.config.non_distance_time_bounds, float)
    env.config.distance_task_types = tuple(int(x) for x in env.config.distance_task_types)
    env.config.non_distance_type_scales = tuple(float(x) for x in env.config.non_distance_type_scales)
    env.config.ordered_commitment = ordered_commitment
    env.config.use_policy_commitment_logic = use_policy_commitment_logic
    env.config.strict_start_in_mask = (
        bool(default_cfg.strict_start_in_mask)
        if strict_start_in_mask is None
        else bool(strict_start_in_mask)
    )
    env._area_diag = float(np.linalg.norm(np.array(env.config.area_size, dtype=np.float64)))
    env._refresh_task_statuses()
    return env


# 只允许恢复策略执行相关的配置键，避免覆盖场景生成参数
_POLICY_EXECUTION_KEYS = frozenset({
    # 承诺/释放逻辑
    "use_policy_commitment_logic",
    "capacity_topk_enabled",
    "ordered_commitment",
    "strict_start_in_mask",
    # 任务吸引力/候选筛选
    "task_attractiveness_relative_margin",
    "task_attractiveness_mask_threshold",
    "task_attractiveness_min_keep",
    "dependency_open_horizon_steps",
    "dependency_open_min_ready_ratio",
    "dependency_open_min_criticality",
    "dependency_anchor_min_keep",
    # 容量 Top-K
    "capacity_topk_per_agent",
    "capacity_topk_min_keep",
    "capacity_topk_remaining_slots_weight",
    "capacity_topk_fullness_weight",
    "capacity_topk_shared_candidate_weight",
    "capacity_topk_dependency_protect_threshold",
    "capacity_topk_ready_protect_threshold",
    "capacity_topk_ready_bucket_keep",
    "capacity_topk_dependency_bucket_keep",
    "capacity_topk_imminent_horizon_steps",
    # 依赖相关
    "dependency_criticality_bonus_weight",
    "dependency_window_urgency_bonus_weight",
    "dependency_ready_bonus_weight",
    # 预集结
    "allow_pre_gathering",
    "pre_gather_radius",
    "pre_gather_start_horizon_steps",
    "pre_gather_max_wait_steps",
    "pre_gather_team_cap",
    "pre_gather_team_cap_when_ready_exists",
    "pre_gather_team_cap_when_imminent_exists",
    "far_pre_gather_team_cap",
    "pre_gather_reserve_ratio",
    "pre_gather_reserve_min",
    "pre_gather_type_reserve_min",
    # stale 重分配
    "allow_stale_wait_reassignment",
    "stale_wait_steps",
    "stale_reassign_score_margin",
    # 承诺仲裁
    "capacity_aware_commitment_arbitration",
    # 方案C：依赖特征稀疏化与RGAT时间传播（评估时必须与训练一致）
    "sparse_dependency_features",
    "rgat_temporal_propagation",
    # 其他执行期参数
    "dt_seconds",
    "max_steps",
    "local_noop_feasible_penalty",
    "reward_noop_feasible_weight",
})


def apply_policy_execution_overrides_to_env(env: MRTAEnv, cfg_overrides: Optional[Dict], strict_start_in_mask: Optional[bool] = None) -> MRTAEnv:
    """只恢复策略执行相关的配置，不恢复场景生成参数（避免与固定测试集场景不一致）。"""
    if not isinstance(cfg_overrides, dict):
        return env
    default_cfg = MRTAConfig()
    ordered_commitment = bool(getattr(env.config, "ordered_commitment", True))
    use_policy_commitment_logic = bool(getattr(env.config, "use_policy_commitment_logic", True))
    # 只应用白名单中的键
    for k, v in cfg_overrides.items():
        if k in _POLICY_EXECUTION_KEYS and hasattr(env.config, k):
            setattr(env.config, k, v)
    # 恢复这些键的原值（它们由场景生成参数决定，不应被覆盖）
    env.config.ordered_commitment = ordered_commitment
    env.config.use_policy_commitment_logic = use_policy_commitment_logic
    env.config.strict_start_in_mask = (
        bool(default_cfg.strict_start_in_mask)
        if strict_start_in_mask is None
        else bool(strict_start_in_mask)
    )
    env._refresh_task_statuses()
    return env