import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from env.mrta_config import MRTAConfig

# ----------------------------------------------------------------------
# 奖励函数冻结常量（从原 15 个超参数降维后固定，不再暴露为可调旋钮）
# 物理含义见 mrta_config.py 奖励段注释；调整需重新做灵敏度分析。
# ----------------------------------------------------------------------
_REWARD_EFFICIENCY_WEIGHT = 0.02      # 效率项权重（原 reward_efficiency_weight）
_REWARD_NOOP_FEASIBLE_WEIGHT = 0.06   # 可行却 NOOP 的全局惩罚（原 reward_noop_feasible_weight）
_REWARD_LOGICAL_WAIT_WEIGHT = 0.03    # 逻辑等待惩罚（原 reward_logical_wait_weight）
_LOCAL_CREDIT_WEIGHT = 0.1            # 局部信用分配权重（原 local_credit_weight）
_LOCAL_REJECT_PENALTY = 0.4           # rejected commitment 局部惩罚（原 local_reject_penalty）
_LOCAL_INFEASIBLE_PENALTY = 0.5       # infeasible team 局部惩罚（原 local_infeasible_penalty）
_LOCAL_NOOP_FEASIBLE_PENALTY = 0.08   # 可行却 NOOP 局部惩罚（原 local_noop_feasible_penalty）


REL_TYPES = {"FS": 0, "SS": 1, "FF": 2, "SF": 3}
REL_NAMES = ["FS", "SS", "FF", "SF"]

CreditEventMap = Dict[str, List[int]]
RuntimeCacheValue = Any
RuntimeCache = Dict[str, RuntimeCacheValue]


@dataclass
class Robot:
    """机器人实体状态。

    作用：
    - 表达单个机器人在环境中的物理与任务状态。
    - 在 step 推进中被状态机更新（IDLE/MOVING/WAITING/EXECUTING/LOGICAL_WAIT）。

    关键字段：
    - pos/v_max/cap: 位置、速度上限、能力向量。
    - task_id/eta/work_left: 当前绑定任务、到达剩余步、执行剩余步。
    """
    robot_id: int
    pos: np.ndarray
    v_max: float
    cap: np.ndarray
    state: str = "IDLE"  # IDLE/MOVING/WAITING/EXECUTING/LOGICAL_WAIT
    task_id: Optional[int] = None
    eta: int = 0
    work_left: int = 0
    agent_type: int = 0
    waiting_since: int = -1
    moving_since: int = -1
    last_progress_time: int = -1


@dataclass
class DependencyEdge:
    """任务依赖边。

    作用：
    - 描述前驱任务 u 与后继任务 v 的时序关系与时间滞后 lag。
    - 被开始约束与结束约束检查函数消费。

    语义：
    - rel_type 取值为 FS/SS/FF/SF。
    """
    u: int
    v: int
    rel_type: str
    lag: int


@dataclass
class Task:
    """任务实体状态。

    作用：
    - 承载任务需求、协作规模、时间戳与执行状态。
    - 在承诺、激活、执行、完成过程中持续更新。

    关键字段：
    - req/workload/k: 能力需求、工作量、协作人数。
    - assigned_robots: 当前已承诺成员。
    - t_start/t_finish_phys/t_finish: 开始、物理完成、约束完成时刻。
    """
    task_id: int
    loc: np.ndarray
    dst_loc: Optional[np.ndarray]
    req: np.ndarray
    workload: float
    demand_count: int
    k: int
    release_time: int
    task_type: int = 0
    exec_mode: str = "NON_DISTANCE"  # DISTANCE/NON_DISTANCE
    type_time_scale: float = 180.0
    status: str = "UNRELEASED"  # UNRELEASED/AVAILABLE/ASSIGNED/READY/EXECUTING/PHYS_DONE/LOGICAL_WAIT/DONE
    assigned_robots: List[int] = field(default_factory=list)
    t_start: int = -1
    t_finish_phys: int = -1
    t_finish: int = -1
    logical_wait_steps: int = 0
    last_logical_wait_reason: str = ""
    duration_steps: int = 0
    remain_phys: float = 0.0
    ss_window_missed: bool = False
    window_missed_edges: Set[str] = field(default_factory=set)
    has_rejected_commitment: bool = False
    has_infeasible_team: bool = False
    has_dependency_violation: bool = False
    has_start_violation: bool = False
    has_finish_violation: bool = False
    has_pseudo_completion: bool = False
    earliest_start: float = 0.0
    earliest_finish: float = 0.0
    latest_start: float = 0.0
    latest_finish: float = 0.0
    slack: float = 0.0
    is_temporally_feasible: bool = True
    num_unsatisfied_predecessors: int = 0
    num_ready_predecessors: int = 0
    predecessor_ready_ratio: float = 0.0
    window_urgency: float = 0.0
    start_feasible_now: bool = False
    finish_feasible_now: bool = False
    criticality: float = 0.0


class MRTAEnv:
    """新版动态异构多机器人任务分配（MRTA）仿真环境。

    功能定位：
    - 提供 RL 所需的 reset/step 交互接口。
    - 模拟动态任务到达、任务依赖约束、多人协作与能力匹配。
    - 输出图结构观测（task graph + agent feats + pair feats + action mask）。

    设计意义：
    - 让策略在“可行性约束 + 时序约束 + 动态变化”下学习调度行为。
    - 统一承诺机制、奖励统计与评估指标，支持训练与测试复用。
    """
    def __init__(self, config: Optional[MRTAConfig] = None, seed: Optional[int] = None):
        """新版动态异构 MRTA 环境初始化。

        输入：
        - config: 环境超参数对象（任务规模、约束、奖励系数等）。
        - seed: 随机种子。

        输出：
        - 无显式返回；完成环境内部状态与统计量初始化。
        """
        self.config = config or MRTAConfig()
        self.rng = np.random.default_rng(seed)
        self.current_time = 0
        self.robots: Dict[int, Robot] = {}
        self.tasks: Dict[int, Task] = {}
        self.edges: List[DependencyEdge] = []
        self.next_task_id = 0

        self.rejected_commitments = 0
        self.missed_ss_window = 0
        self.infeasible_team = 0
        self.waiting_cost = 0
        self.logical_wait_cost = 0
        self.total_done = 0
        self.total_recognized_done = 0
        self.last_local_credit_rewards: Dict[int, float] = {}
        self.last_credit_events: CreditEventMap = {}
        self.commit_reject_reason_counts: Counter = Counter()
        self.last_commit_reject_reason_counts: Dict[str, int] = {}

        self.rel_window_steps = self._build_relation_window_steps()
        self.task_type_dim = self.config.cap_dim
        self.distance_task_types = {int(t) for t in getattr(self.config, "distance_task_types", ()) if 0 <= int(t) < self.task_type_dim}
        self.non_distance_time_bounds = getattr(self.config, "non_distance_time_bounds", (60.0, 600.0))
        self.non_distance_type_scales = self._build_non_distance_type_scales()
        self.type_speeds = np.zeros((self.task_type_dim,), dtype=np.float64)
        self.gen_planned_start_steps: Dict[int, int] = {}
        self.gen_planned_finish_steps: Dict[int, int] = {}
        self._accepted_edges_since_global_check = 0
        self._last_generation_check: Dict[str, int] = {
            "horizon_violations": 0,
            "window_violations": 0,
            "checked_tasks": 0,
            "checked_edges": 0,
        }
        self.current_scenario_difficulty = 0.0
        self.current_scenario_difficulty_components: Dict[str, float] = {}
        self.last_reset_attempts = 1
        self.last_reset_range_ok = True
        self.last_generation_feasible = True
        self._area_diag = float(np.linalg.norm(self.config.area_size))
        self._perf_stats: Dict[str, float] = {
            "step_total_sec": 0.0,
            "dynamic_arrival_sec": 0.0,
            "commitment_sec": 0.0,
            "progress_sec": 0.0,
            "refresh_status_sec": 0.0,
            "obs_total_sec": 0.0,
            "obs_temporal_sec": 0.0,
            "obs_node_sec": 0.0,
            "obs_edge_sec": 0.0,
            "obs_agent_sec": 0.0,
            "obs_pair_sec": 0.0,
            "obs_action_mask_sec": 0.0,
            "can_commit_calls": 0.0,
            "can_commit_dry_run_calls": 0.0,
            "can_commit_wet_run_calls": 0.0,
            "get_obs_calls": 0.0,
            "step_calls": 0.0,
        }
        self._obs_runtime_cache: Optional[RuntimeCache] = None

    def _incoming_edges_map(self) -> Dict[int, List[DependencyEdge]]:
        runtime_cache = self._obs_runtime_cache
        if runtime_cache is not None:
            cached = runtime_cache.get("incoming_edges_map")
            if isinstance(cached, dict):
                return cached
        incoming: Dict[int, List[DependencyEdge]] = {int(tid): [] for tid in self.tasks.keys()}
        for edge in self.edges:
            incoming.setdefault(int(edge.v), []).append(edge)
        if runtime_cache is not None:
            runtime_cache["incoming_edges_map"] = incoming
        return incoming

    def _outgoing_edges_map(self) -> Dict[int, List[DependencyEdge]]:
        runtime_cache = self._obs_runtime_cache
        if runtime_cache is not None:
            cached = runtime_cache.get("outgoing_edges_map")
            if isinstance(cached, dict):
                return cached
        outgoing: Dict[int, List[DependencyEdge]] = {int(tid): [] for tid in self.tasks.keys()}
        for edge in self.edges:
            outgoing.setdefault(int(edge.u), []).append(edge)
        if runtime_cache is not None:
            runtime_cache["outgoing_edges_map"] = outgoing
        return outgoing

    def _build_action_mask_and_commit_scores(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        task_count = int(self.config.max_tasks)
        robot_ids = sorted(self.robots.keys())
        action_mask = np.zeros((len(robot_ids), task_count + 1), dtype=np.float32)
        can_commit_matrix = np.zeros((len(robot_ids), task_count), dtype=np.float32)
        score_matrix = np.full((len(robot_ids), task_count), -1e9, dtype=np.float32)
        action_mask[:, 0] = 1.0

        use_policy_logic = bool(getattr(self.config, "use_policy_commitment_logic", True))
        rel_margin = float(getattr(self.config, "task_attractiveness_relative_margin", 2.2))
        abs_threshold = float(getattr(self.config, "task_attractiveness_mask_threshold", -0.25))
        min_keep = max(1, int(getattr(self.config, "task_attractiveness_min_keep", 1)))
        dep_open_horizon = max(0.0, float(getattr(self.config, "dependency_open_horizon_steps", 18)))
        dep_open_ready_ratio = float(getattr(self.config, "dependency_open_min_ready_ratio", 0.5))
        dep_open_min_criticality = float(getattr(self.config, "dependency_open_min_criticality", 1.6))
        dep_anchor_min_keep = max(0, int(getattr(self.config, "dependency_anchor_min_keep", 1)))
        capacity_topk_enabled = bool(getattr(self.config, "capacity_topk_enabled", False))
        capacity_topk_per_agent = max(1, int(getattr(self.config, "capacity_topk_per_agent", 2)))
        capacity_topk_min_keep = max(1, int(getattr(self.config, "capacity_topk_min_keep", 1)))

        candidate_popularity: Counter[int] = Counter()
        candidate_rows: Dict[int, List[Tuple[int, float]]] = {}
        for row_i, rid in enumerate(robot_ids):
            robot = self.robots[int(rid)]
            if robot.state != "IDLE":
                continue
            candidate_scores: List[Tuple[int, float]] = []
            for tid in range(task_count):
                if tid not in self.tasks:
                    continue
                can_commit = self._can_commit(int(rid), int(tid), dry_run=True)
                if not can_commit:
                    continue
                can_commit_matrix[row_i, tid] = 1.0
                if use_policy_logic:
                    score = float(self._task_candidate_attractiveness(int(rid), self.tasks[tid]))
                    score_matrix[row_i, tid] = score
                    candidate_scores.append((tid, score))
                    candidate_popularity[int(tid)] += 1
                else:
                    action_mask[row_i, tid + 1] = 1.0
            if use_policy_logic and candidate_scores:
                candidate_rows[row_i] = candidate_scores

        if not use_policy_logic:
            return action_mask, can_commit_matrix, score_matrix

        for row_i, candidate_scores in candidate_rows.items():
            if not candidate_scores:
                continue
            best_score = max(score for _, score in candidate_scores)
            kept_count = 0
            anchor_candidates: List[Tuple[int, float]] = []
            for tid, score in candidate_scores:
                task = self.tasks.get(int(tid))
                if task is None:
                    continue
                open_by_score = score >= abs_threshold and score >= best_score - rel_margin
                has_preds = int(getattr(task, "num_unsatisfied_predecessors", 0)) > 0
                est_start = float(getattr(task, "earliest_start", self.current_time))
                ready_ratio = float(np.clip(getattr(task, "predecessor_ready_ratio", 0.0), 0.0, 1.0))
                criticality = float(getattr(task, "criticality", 0.0))
                imminent_dep = est_start <= float(self.current_time) + max(1.0, dep_open_horizon * 0.6)
                highly_ready_dep = ready_ratio >= min(0.95, dep_open_ready_ratio + 0.1)
                open_by_dependency = (
                    has_preds
                    and not bool(getattr(task, "start_feasible_now", False))
                    and est_start <= float(self.current_time) + dep_open_horizon
                    and imminent_dep
                    and ready_ratio >= dep_open_ready_ratio
                    and highly_ready_dep
                    and criticality >= dep_open_min_criticality
                )
                if open_by_score or open_by_dependency:
                    action_mask[row_i, tid + 1] = 1.0
                    kept_count += 1
                if not has_preds:
                    anchor_candidates.append((tid, score))
            if kept_count < min_keep:
                for tid, _ in sorted(candidate_scores, key=lambda item: item[1], reverse=True):
                    if action_mask[row_i, tid + 1] > 0.0:
                        continue
                    action_mask[row_i, tid + 1] = 1.0
                    kept_count += 1
                    if kept_count >= min_keep:
                        break
            if capacity_topk_enabled:
                self._apply_capacity_aware_topk(
                    row_i=row_i,
                    candidate_scores=candidate_scores,
                    action_mask=action_mask,
                    score_matrix=score_matrix,
                    candidate_popularity=candidate_popularity,
                    per_agent_keep=capacity_topk_per_agent,
                    min_keep=max(capacity_topk_min_keep, min_keep),
                )
            anchor_kept = sum(1 for tid, _ in anchor_candidates if action_mask[row_i, tid + 1] > 0.0)
            if anchor_kept >= dep_anchor_min_keep:
                continue
            for tid, _ in sorted(anchor_candidates, key=lambda item: item[1], reverse=True):
                if action_mask[row_i, tid + 1] > 0.0:
                    continue
                action_mask[row_i, tid + 1] = 1.0
                anchor_kept += 1
                if anchor_kept >= dep_anchor_min_keep:
                    break

        return action_mask, can_commit_matrix, score_matrix

    def _capacity_aware_candidate_score(self, task: Task, base_score: float, candidate_popularity: int = 1) -> float:
        remaining_slots = max(0, int(task.k) - len(task.assigned_robots))
        remaining_ratio = float(remaining_slots) / float(max(1, int(task.k)))
        fullness = float(len(task.assigned_robots)) / float(max(1, int(task.k)))
        slot_weight = float(getattr(self.config, "capacity_topk_remaining_slots_weight", 1.1))
        fullness_weight = float(getattr(self.config, "capacity_topk_fullness_weight", 1.4))
        congestion_penalty = slot_weight * (1.0 - remaining_ratio) + fullness_weight * fullness
        popularity_weight = float(getattr(self.config, "capacity_topk_shared_candidate_weight", 0.0))
        shared_penalty = popularity_weight * max(0.0, float(candidate_popularity) - 1.0)
        return float(base_score - congestion_penalty - shared_penalty)

    def _capacity_topk_task_protected(self, task: Task) -> bool:
        if bool(getattr(task, "start_feasible_now", False)):
            return True
        criticality_threshold = float(getattr(self.config, "capacity_topk_dependency_protect_threshold", 1.8))
        ready_ratio_threshold = float(getattr(self.config, "capacity_topk_ready_protect_threshold", 0.85))
        has_preds = int(getattr(task, "num_unsatisfied_predecessors", 0)) > 0
        return bool(
            has_preds
            and (
                float(getattr(task, "criticality", 0.0)) >= criticality_threshold
                or float(getattr(task, "predecessor_ready_ratio", 0.0)) >= ready_ratio_threshold
            )
        )

    def _capacity_topk_task_bucket(self, task: Task) -> str:
        imminent_horizon = max(0.0, float(getattr(self.config, "capacity_topk_imminent_horizon_steps", 8)))
        est_start = float(getattr(task, "earliest_start", self.current_time))
        ready_now = bool(getattr(task, "start_feasible_now", False))
        near_ready = est_start <= float(self.current_time) + imminent_horizon
        if ready_now or near_ready:
            return "ready"
        if self._capacity_topk_task_protected(task):
            return "dependency"
        return "general"

    def _apply_capacity_aware_topk(
        self,
        row_i: int,
        candidate_scores: List[Tuple[int, float]],
        action_mask: np.ndarray,
        score_matrix: np.ndarray,
        candidate_popularity: Counter[int],
        per_agent_keep: int,
        min_keep: int,
    ):
        kept_candidates: List[Tuple[int, float, float, bool, str]] = []
        for tid, base_score in candidate_scores:
            if action_mask[row_i, tid + 1] <= 0.0:
                continue
            task = self.tasks.get(int(tid))
            if task is None:
                continue
            adjusted_score = self._capacity_aware_candidate_score(
                task,
                float(base_score),
                int(candidate_popularity.get(int(tid), 1)),
            )
            protected = self._capacity_topk_task_protected(task)
            bucket = self._capacity_topk_task_bucket(task)
            kept_candidates.append((int(tid), float(base_score), float(adjusted_score), bool(protected), bucket))
            score_matrix[row_i, tid] = float(adjusted_score)
        if not kept_candidates:
            return
        chosen_ids = set()
        ready_keep = max(0, int(getattr(self.config, "capacity_topk_ready_bucket_keep", 1)))
        dep_keep = max(0, int(getattr(self.config, "capacity_topk_dependency_bucket_keep", 1)))

        ready_bucket = [(tid, adjusted) for tid, _, adjusted, _, bucket in kept_candidates if bucket == "ready"]
        dep_bucket = [(tid, adjusted) for tid, _, adjusted, _, bucket in kept_candidates if bucket == "dependency"]
        general_bucket = [(tid, adjusted) for tid, _, adjusted, _, bucket in kept_candidates if bucket == "general"]

        for tid, _ in sorted(ready_bucket, key=lambda item: item[1], reverse=True)[: min(ready_keep, per_agent_keep)]:
            chosen_ids.add(int(tid))
        dep_chosen = 0
        for tid, _ in sorted(dep_bucket, key=lambda item: item[1], reverse=True):
            if len(chosen_ids) >= per_agent_keep or dep_chosen >= dep_keep:
                break
            chosen_ids.add(int(tid))
            dep_chosen += 1

        protected_ids = [tid for tid, _, _, protected, _ in kept_candidates if protected]
        for tid in protected_ids:
            if len(chosen_ids) >= per_agent_keep:
                break
            chosen_ids.add(int(tid))

        remaining_budget = max(per_agent_keep - len(chosen_ids), 0)
        fallback_pool = sorted(
            ready_bucket + dep_bucket + general_bucket,
            key=lambda item: item[1],
            reverse=True,
        )
        for tid, _ in fallback_pool:
            if int(tid) in chosen_ids:
                continue
            chosen_ids.add(int(tid))
            if len(chosen_ids) >= per_agent_keep:
                break
        if len(chosen_ids) < min_keep:
            for tid, _, adjusted, _, _ in sorted(kept_candidates, key=lambda item: item[2], reverse=True):
                chosen_ids.add(int(tid))
                if len(chosen_ids) >= min_keep:
                    break
        for tid, _, _, _, _ in kept_candidates:
            action_mask[row_i, tid + 1] = 1.0 if int(tid) in chosen_ids else 0.0

    def _get_step_action_snapshot(self) -> Tuple[np.ndarray, np.ndarray]:
        runtime_cache: RuntimeCache = {"can_commit": {}}
        prev_cache = self._obs_runtime_cache
        self._obs_runtime_cache = runtime_cache
        try:
            self._update_task_temporal_stats()
            action_mask, can_commit_matrix, _ = self._build_action_mask_and_commit_scores()
            return action_mask, can_commit_matrix
        finally:
            self._obs_runtime_cache = prev_cache

    def _task_has_hard_violation(self, task: Task) -> bool:
        return bool(
            task.has_pseudo_completion
            or task.has_dependency_violation
            or task.has_infeasible_team
        )

    def _task_commitment_gate_ok(self, task: Task) -> bool:
        if task.status not in ["AVAILABLE", "ASSIGNED", "READY"]:
            return False
        if self._task_has_hard_violation(task):
            return False
        if len(task.assigned_robots) >= task.k:
            return False
        if not bool(getattr(self.config, "use_policy_commitment_logic", True)):
            preds = self._incoming_edges_map().get(int(task.task_id), [])
            if not preds:
                return True
            if not bool(getattr(self.config, "allow_pre_gathering", True)):
                return False
            idle_count = sum(1 for r in self.robots.values() if r.state == "IDLE")
            type_idle_count = sum(1 for r in self.robots.values() if r.state == "IDLE" and int(r.agent_type) == int(task.task_type))
            free_after_commit = idle_count - max(0, task.k - len(task.assigned_robots))
            type_free_after_commit = type_idle_count - 1
            reserve_ratio = float(getattr(self.config, "pre_gather_reserve_ratio", 0.15))
            reserve_floor = int(getattr(self.config, "pre_gather_reserve_min", 1))
            reserve = max(reserve_floor, int(math.ceil(reserve_ratio * max(1, self.config.num_agents))))
            type_reserve = int(max(1, getattr(self.config, "pre_gather_type_reserve_min", 1)))
            return free_after_commit >= reserve and type_free_after_commit >= type_reserve
        preds = self._incoming_edges_map().get(int(task.task_id), [])
        unresolved_preds = 0
        predictable_preds = 0
        known_ready_preds = 0
        earliest_ready_time = int(self.current_time)
        for e in preds:
            u = self.tasks[e.u]
            if e.rel_type in ["FS", "FF"]:
                ref_time = self._predicted_finish_time(u) if e.rel_type == "FF" else self._constraint_reference_time(u, e.rel_type)
                if ref_time < 0:
                    unresolved_preds += 1
                else:
                    predictable_preds += 1
                    known_ready_preds += 1
                    if e.rel_type == "FS":
                        earliest_ready_time = max(earliest_ready_time, int(ref_time + e.lag))
                    else:
                        succ_duration = self._effective_task_duration_steps(task)
                        earliest_ready_time = max(earliest_ready_time, int(ref_time + e.lag - succ_duration))
            else:
                ref_time = self._predicted_start_time(u) if e.rel_type == "SF" else self._constraint_reference_time(u, e.rel_type)
                if ref_time < 0:
                    unresolved_preds += 1
                else:
                    predictable_preds += 1
                    known_ready_preds += 1
                    if e.rel_type == "SS":
                        earliest_ready_time = max(earliest_ready_time, int(ref_time + e.lag))
                    else:
                        succ_duration = self._effective_task_duration_steps(task)
                        earliest_ready_time = max(earliest_ready_time, int(ref_time + e.lag - succ_duration))
        total_preds = max(0, len(preds))
        known_pred_ratio = float(known_ready_preds) / float(max(1, total_preds))
        if unresolved_preds <= 0:
            return True
        if not bool(getattr(self.config, "allow_pre_gathering", True)):
            return False
        start_horizon = int(getattr(self.config, "pre_gather_start_horizon_steps", 12))
        pre_gather_wait = int(max(0, earliest_ready_time - int(self.current_time)))
        max_wait = int(max(start_horizon, getattr(self.config, "pre_gather_max_wait_steps", start_horizon * 2)))
        if pre_gather_wait > max_wait:
            return False
        if total_preds > 0 and known_pred_ratio < 0.5:
            return False

        hard_ready_tasks = [
            t for t in self.tasks.values()
            if t.task_id != task.task_id
            and t.status in ["AVAILABLE", "ASSIGNED", "READY"]
            and not self._task_has_hard_violation(t)
            and int(getattr(t, "task_type", -1)) == int(task.task_type)
            and bool(getattr(t, "start_feasible_now", False))
            and bool(getattr(t, "is_temporally_feasible", True))
        ]
        imminent_ready_tasks = [
            t for t in self.tasks.values()
            if t.task_id != task.task_id
            and t.status in ["AVAILABLE", "ASSIGNED", "READY"]
            and not self._task_has_hard_violation(t)
            and int(getattr(t, "task_type", -1)) == int(task.task_type)
            and float(getattr(t, "earliest_start", self.current_time)) <= float(self.current_time + start_horizon)
        ]

        max_pre_gather_team = int(getattr(self.config, "pre_gather_team_cap", 1))
        if predictable_preds > 0 and earliest_ready_time <= self.current_time + start_horizon:
            if hard_ready_tasks:
                max_pre_gather_team = int(getattr(self.config, "pre_gather_team_cap_when_ready_exists", 0))
            elif imminent_ready_tasks:
                max_pre_gather_team = min(max_pre_gather_team, int(getattr(self.config, "pre_gather_team_cap_when_imminent_exists", 1)))
        else:
            if hard_ready_tasks or imminent_ready_tasks:
                return False
            max_pre_gather_team = int(getattr(self.config, "far_pre_gather_team_cap", 0))

        if len(task.assigned_robots) >= max(0, max_pre_gather_team):
            return False

        idle_count = sum(1 for r in self.robots.values() if r.state == "IDLE")
        type_idle_count = sum(1 for r in self.robots.values() if r.state == "IDLE" and int(r.agent_type) == int(task.task_type))
        free_after_commit = idle_count - max(0, task.k - len(task.assigned_robots))
        type_free_after_commit = type_idle_count - 1
        reserve_ratio = float(getattr(self.config, "pre_gather_reserve_ratio", 0.15))
        reserve_floor = int(getattr(self.config, "pre_gather_reserve_min", 1))
        reserve = max(reserve_floor, int(math.ceil(reserve_ratio * max(1, self.config.num_agents))))
        extra_type_reserve = 1 if hard_ready_tasks else 0
        type_reserve_floor = int(max(0, getattr(self.config, "pre_gather_type_reserve_min", 0)))
        type_reserve = int(type_reserve_floor + extra_type_reserve)
        return free_after_commit >= reserve and type_free_after_commit >= type_reserve

    def _task_candidate_attractiveness(self, rid: int, task: Task) -> float:
        if not bool(getattr(self.config, "use_policy_commitment_logic", True)):
            return 0.0
        robot = self.robots[int(rid)]
        if robot.state != "IDLE":
            return -1e9
        if task.status not in ["AVAILABLE", "ASSIGNED", "READY"]:
            return -1e9
        if int(robot.agent_type) != int(task.task_type):
            return -1e9

        dist = float(np.linalg.norm(robot.pos - task.loc))
        dist_score = 1.0 - float(np.clip(dist / max(self._area_diag, 1e-6), 0.0, 1.0))
        preds = self._incoming_edges_map().get(int(task.task_id), [])
        no_pred_bonus = 1.0 if len(preds) == 0 else 0.0
        ready_ratio = float(np.clip(getattr(task, "predecessor_ready_ratio", 0.0), 0.0, 1.0))
        feasible_now = 1.0 if bool(getattr(task, "start_feasible_now", False)) else 0.0
        criticality = float(max(0.0, getattr(task, "criticality", 0.0)))
        criticality_norm = float(np.clip(criticality / 4.0, 0.0, 1.5))
        window_urgency = float(np.clip(getattr(task, "window_urgency", 0.0), 0.0, 1.0))
        assigned_ratio = float(len(task.assigned_robots)) / float(max(1, int(task.k)))
        near_full_bonus = assigned_ratio
        est_start = float(max(self.current_time, int(math.ceil(getattr(task, "earliest_start", self.current_time)))))
        start_horizon = max(1.0, float(getattr(self.config, "pre_gather_start_horizon_steps", 12)))
        imminence = 1.0 - float(np.clip((est_start - float(self.current_time)) / start_horizon, 0.0, 1.0))
        far_lock_penalty = 0.0
        if int(getattr(task, "num_unsatisfied_predecessors", 0)) > 0 and not bool(getattr(task, "start_feasible_now", False)):
            far_wait = max(0.0, est_start - float(self.current_time + start_horizon))
            far_lock_penalty = float(np.clip(far_wait / max(start_horizon, 1.0), 0.0, 1.5))
        stuck_penalty = 0.0
        if task.status == "ASSIGNED" and len(task.assigned_robots) > 0:
            anchors = []
            for arid in task.assigned_robots:
                rr = self.robots.get(int(arid))
                if rr is None:
                    continue
                for val in [rr.last_progress_time, rr.moving_since, rr.waiting_since]:
                    if int(val) >= 0:
                        anchors.append(int(val))
                        break
            if anchors:
                stale_steps = self.current_time - max(anchors)
                stale_horizon = max(1.0, float(getattr(self.config, "stale_task_penalty_horizon_steps", 16)))
                stuck_penalty = float(np.clip(stale_steps / stale_horizon, 0.0, 1.0))
        dep_criticality_bonus_w = float(getattr(self.config, "dependency_criticality_bonus_weight", 1.2))
        dep_window_bonus_w = float(getattr(self.config, "dependency_window_urgency_bonus_weight", 1.4))
        dep_ready_bonus_w = float(getattr(self.config, "dependency_ready_bonus_weight", 0.8))
        return float(
            4.0 * feasible_now
            + 2.4 * no_pred_bonus
            + 1.8 * ready_ratio
            + dep_ready_bonus_w * ready_ratio
            + 0.9 * near_full_bonus
            + 2.2 * imminence
            + 0.8 * dist_score
            + dep_criticality_bonus_w * criticality_norm
            + dep_window_bonus_w * window_urgency
            - 2.8 * far_lock_penalty
            - 1.5 * stuck_penalty
        )

    def _maybe_release_stale_commitments(self):
        if not bool(getattr(self.config, "use_policy_commitment_logic", True)):
            return
        if not bool(getattr(self.config, "allow_stale_wait_reassignment", True)):
            return
        stale_wait_steps = int(max(1, getattr(self.config, "stale_wait_steps", 18)))
        score_margin = float(getattr(self.config, "stale_reassign_score_margin", 0.35))
        for rid, robot in self.robots.items():
            if robot.task_id is None or robot.state not in ["WAITING", "MOVING"]:
                continue
            task = self.tasks.get(int(robot.task_id))
            if task is None or task.status not in ["AVAILABLE", "ASSIGNED"]:
                continue
            anchor = int(robot.waiting_since if robot.state == "WAITING" else robot.moving_since)
            if anchor < 0 or self.current_time - anchor < stale_wait_steps:
                continue
            unresolved = int(getattr(task, "num_unsatisfied_predecessors", 0))
            est_start = float(getattr(task, "earliest_start", self.current_time))
            horizon = float(max(1, getattr(self.config, "pre_gather_start_horizon_steps", 12)))
            hopeless_locked = unresolved > 0 and est_start > float(self.current_time) + horizon
            cur_score = self._task_candidate_attractiveness(int(rid), task)
            better_exists = False
            for cand in self.tasks.values():
                if cand.task_id == task.task_id:
                    continue
                cand_score = self._task_candidate_attractiveness(int(rid), cand)
                if cand_score > cur_score + score_margin:
                    better_exists = True
                    break
            if not better_exists and not hopeless_locked:
                continue
            if rid in task.assigned_robots:
                task.assigned_robots.remove(int(rid))
            if task.status in ["ASSIGNED", "READY"] and len(task.assigned_robots) == 0:
                task.status = "AVAILABLE"
            robot.state = "IDLE"
            robot.task_id = None
            robot.eta = 0
            robot.work_left = 0
            robot.waiting_since = -1
            robot.moving_since = -1

    def _forced_noop_stats(self, actions_dict: Dict[int, int], action_mask_snapshot: np.ndarray) -> Tuple[int, int, int]:
        forced_noop = 0
        noop_when_feasible = 0
        feasible_agents = 0
        for rid in sorted(self.robots.keys()):
            feasible = bool(np.sum(action_mask_snapshot[rid, 1:]) > 0)
            is_noop = int(actions_dict.get(rid, 0)) == 0
            if feasible:
                feasible_agents += 1
                if is_noop:
                    noop_when_feasible += 1
            elif is_noop:
                forced_noop += 1
        return forced_noop, noop_when_feasible, feasible_agents

    def _constraint_reference_time(self, pred_task: Task, rel_type: str) -> int:
        if rel_type in ["FS", "FF"]:
            return int(pred_task.t_finish)
        return int(pred_task.t_start)

    def _window_status(self, ref_time: int, ready_t: int, window_t: int) -> str:
        if ref_time < 0:
            return "blocked"
        if self.current_time < ready_t:
            return "early"
        if self.current_time > window_t:
            return "missed"
        return "open"

    def _build_relation_window_steps(self) -> Dict[str, int]:
        """将四类关系的窗口秒数转换为离散步数。"""
        cfg_windows = getattr(self.config, "relation_window_seconds", {})
        out: Dict[str, int] = {}
        for rel in REL_NAMES:
            sec = float(cfg_windows.get(rel, 60.0)) if isinstance(cfg_windows, dict) else 60.0
            out[rel] = max(1, int(math.ceil(sec / max(self.config.dt_seconds, 1e-6))))
        return out

    def reset(self, seed: Optional[int] = None):
        """重置 episode。

        输入：
        - seed: 可选随机种子。

        输出：
        - obs: 初始观测字典，供策略网络输入。
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        max_attempts = int(max(1, getattr(self.config, "difficulty_max_reset_attempts", 16)))
        enforce_range = bool(getattr(self.config, "enforce_difficulty_range", False))
        diff_min = float(np.clip(getattr(self.config, "difficulty_min", 0.0), 0.0, 1.0))
        diff_max = float(np.clip(getattr(self.config, "difficulty_max", 1.0), 0.0, 1.0))
        if diff_min > diff_max:
            diff_min, diff_max = diff_max, diff_min

        fallback_state = None
        fallback_diff = None
        fallback_summary = None

        for attempt in range(1, max_attempts + 1):
            self.current_time = 0
            self.type_speeds = self.rng.uniform(self.config.v_max_range[0], self.config.v_max_range[1], size=(self.task_type_dim,))
            self.robots = self._generate_robots()
            self.tasks = {}
            self.edges = []
            self.next_task_id = 0
            self.gen_planned_start_steps = {}
            self.gen_planned_finish_steps = {}
            self._accepted_edges_since_global_check = 0
            self._last_generation_check = {
                "horizon_violations": 0,
                "window_violations": 0,
                "checked_tasks": 0,
                "checked_edges": 0,
            }

            self.rejected_commitments = 0
            self.missed_ss_window = 0
            self.infeasible_team = 0
            self.waiting_cost = 0
            self.logical_wait_cost = 0
            self.total_done = 0
            self.total_recognized_done = 0
            self.last_local_credit_rewards = {rid: 0.0 for rid in self.robots.keys()}
            self.last_credit_events = {
                "reject_agents": [],
                "infeasible_agents": [],
                "noop_feasible_agents": [],
            }
            self.commit_reject_reason_counts = Counter()
            self.last_commit_reject_reason_counts = {}

            total_tasks_floor = int(getattr(self.config, "total_tasks_floor", self.config.init_tasks_range[0]))
            total_tasks_cap = int(getattr(self.config, "total_tasks_cap", self.config.max_tasks))
            total_tasks_cap = max(1, min(total_tasks_cap, int(self.config.max_tasks)))
            total_tasks_floor = max(0, min(total_tasks_floor, total_tasks_cap))

            n0 = int(self.rng.integers(self.config.init_tasks_range[0], self.config.init_tasks_range[1] + 1))
            self._spawn_tasks(n0)
            # 强制保证单回合任务总数下限（初始+动态到达）。
            if len(self.tasks) < total_tasks_floor:
                self._spawn_tasks(total_tasks_floor - len(self.tasks))
            self._refresh_task_statuses()

            gen_summary = self.generation_feasibility_summary()
            diff_summary = self.scenario_difficulty_summary()
            score = float(diff_summary.get("score", 0.0))
            feasible = int(gen_summary.get("feasible", 0)) == 1
            in_range = (score >= diff_min) and (score <= diff_max)
            accepted = feasible and ((not enforce_range) or in_range)

            if accepted:
                self.current_scenario_difficulty = score
                self.current_scenario_difficulty_components = diff_summary
                self.last_generation_feasible = True
                self.last_reset_attempts = int(attempt)
                self.last_reset_range_ok = bool(in_range)
                return self._get_obs()

            if feasible and fallback_state is None:
                fallback_state = self.get_state()
                fallback_diff = diff_summary
                fallback_summary = gen_summary

        if fallback_state is None:
            raise RuntimeError(
                f"reset failed: no temporally feasible scenario generated after {max_attempts} attempts"
            )

        if enforce_range:
            raise RuntimeError(
                "reset failed: unable to sample a feasible scenario inside difficulty range "
                f"[{diff_min:.3f}, {diff_max:.3f}] after {max_attempts} attempts"
            )

        self.set_state(fallback_state)
        self.current_scenario_difficulty = float((fallback_diff or {}).get("score", 0.0))
        self.current_scenario_difficulty_components = fallback_diff or {}
        self.last_generation_feasible = int((fallback_summary or {}).get("feasible", 0)) == 1
        self.last_reset_attempts = int(max_attempts)
        self.last_reset_range_ok = bool(
            diff_min <= self.current_scenario_difficulty <= diff_max
        )
        return self._get_obs()

    def get_state(self) -> Dict:
        """导出环境完整状态，用于无缝恢复训练。"""
        return {
            "current_time": int(self.current_time),
            "next_task_id": int(self.next_task_id),
            "rejected_commitments": int(self.rejected_commitments),
            "missed_ss_window": int(self.missed_ss_window),
            "infeasible_team": int(self.infeasible_team),
            "waiting_cost": int(self.waiting_cost),
            "logical_wait_cost": int(self.logical_wait_cost),
            "total_done": int(self.total_done),
            "total_recognized_done": int(self.total_recognized_done),
            "last_local_credit_rewards": {int(k): float(v) for k, v in self.last_local_credit_rewards.items()},
            "last_credit_events": {
                "reject_agents": [int(x) for x in self.last_credit_events.get("reject_agents", [])],
                "infeasible_agents": [int(x) for x in self.last_credit_events.get("infeasible_agents", [])],
                "noop_feasible_agents": [int(x) for x in self.last_credit_events.get("noop_feasible_agents", [])],
            },
            "commit_reject_reason_counts": {str(k): int(v) for k, v in self.commit_reject_reason_counts.items()},
            "last_commit_reject_reason_counts": {str(k): int(v) for k, v in self.last_commit_reject_reason_counts.items()},
            "type_speeds": self.type_speeds.astype(np.float64).tolist(),
            "rng_state": self.rng.bit_generator.state,
            "robots": {
                int(rid): {
                    "robot_id": int(r.robot_id),
                    "pos": r.pos.astype(np.float64).tolist(),
                    "v_max": float(r.v_max),
                    "cap": r.cap.astype(np.float64).tolist(),
                    "state": str(r.state),
                    "task_id": None if r.task_id is None else int(r.task_id),
                    "eta": int(r.eta),
                    "work_left": int(r.work_left),
                    "agent_type": int(r.agent_type),
                }
                for rid, r in self.robots.items()
            },
            "tasks": {
                int(tid): {
                    "task_id": int(t.task_id),
                    "loc": t.loc.astype(np.float64).tolist(),
                    "dst_loc": None if t.dst_loc is None else t.dst_loc.astype(np.float64).tolist(),
                    "req": t.req.astype(np.float64).tolist(),
                    "workload": float(t.workload),
                    "demand_count": int(t.demand_count),
                    "k": int(t.k),
                    "release_time": int(t.release_time),
                    "task_type": int(t.task_type),
                    "exec_mode": str(t.exec_mode),
                    "type_time_scale": float(t.type_time_scale),
                    "status": str(t.status),
                    "assigned_robots": [int(x) for x in t.assigned_robots],
                    "t_start": int(t.t_start),
                    "t_finish_phys": int(t.t_finish_phys),
                    "t_finish": int(t.t_finish),
                    "logical_wait_steps": int(t.logical_wait_steps),
                    "last_logical_wait_reason": str(t.last_logical_wait_reason),
                    "duration_steps": int(t.duration_steps),
                    "remain_phys": float(t.remain_phys),
                    "ss_window_missed": bool(t.ss_window_missed),
                    "window_missed_edges": sorted(list(t.window_missed_edges)),
                    "has_rejected_commitment": bool(t.has_rejected_commitment),
                    "has_infeasible_team": bool(t.has_infeasible_team),
                    "has_dependency_violation": bool(t.has_dependency_violation),
                    "has_start_violation": bool(t.has_start_violation),
                    "has_finish_violation": bool(t.has_finish_violation),
                    "has_pseudo_completion": bool(t.has_pseudo_completion),
                    "earliest_start": float(t.earliest_start),
                    "earliest_finish": float(t.earliest_finish),
                    "latest_start": float(t.latest_start),
                    "latest_finish": float(t.latest_finish),
                    "slack": float(t.slack),
                    "is_temporally_feasible": bool(t.is_temporally_feasible),
                    "num_unsatisfied_predecessors": int(t.num_unsatisfied_predecessors),
                    "criticality": float(t.criticality),
                }
                for tid, t in self.tasks.items()
            },
            "edges": [
                {
                    "u": int(e.u),
                    "v": int(e.v),
                    "rel_type": str(e.rel_type),
                    "lag": int(e.lag),
                }
                for e in self.edges
            ],
            "gen_planned_start_steps": {int(k): int(v) for k, v in self.gen_planned_start_steps.items()},
            "gen_planned_finish_steps": {int(k): int(v) for k, v in self.gen_planned_finish_steps.items()},
            "accepted_edges_since_global_check": int(self._accepted_edges_since_global_check),
            "last_generation_check": {
                "horizon_violations": int(self._last_generation_check.get("horizon_violations", 0)),
                "window_violations": int(self._last_generation_check.get("window_violations", 0)),
                "checked_tasks": int(self._last_generation_check.get("checked_tasks", len(self.tasks))),
                "checked_edges": int(self._last_generation_check.get("checked_edges", len(self.edges))),
            },
            "scenario_difficulty": float(self.current_scenario_difficulty),
            "scenario_difficulty_components": {
                str(k): float(v) for k, v in self.current_scenario_difficulty_components.items()
            },
            "last_reset_attempts": int(self.last_reset_attempts),
            "last_reset_range_ok": bool(self.last_reset_range_ok),
            "last_generation_feasible": bool(self.last_generation_feasible),
        }

    def set_state(self, state: Dict):
        """恢复环境完整状态。"""
        self.current_time = int(state["current_time"])
        self.next_task_id = int(state["next_task_id"])
        self.rejected_commitments = int(state["rejected_commitments"])
        self.missed_ss_window = int(state["missed_ss_window"])
        self.infeasible_team = int(state["infeasible_team"])
        self.waiting_cost = int(state.get("waiting_cost", state.get("hold_cost", 0)))
        self.logical_wait_cost = int(state.get("logical_wait_cost", 0))
        self.total_done = int(state["total_done"])
        self.total_recognized_done = int(state.get("total_recognized_done", state.get("total_done", 0)))
        self.last_local_credit_rewards = {
            int(k): float(v) for k, v in state.get("last_local_credit_rewards", {}).items()
        }
        raw_credit_events = state.get("last_credit_events", {})
        self.last_credit_events = {
            "reject_agents": [int(x) for x in raw_credit_events.get("reject_agents", [])],
            "infeasible_agents": [int(x) for x in raw_credit_events.get("infeasible_agents", [])],
            "noop_feasible_agents": [int(x) for x in raw_credit_events.get("noop_feasible_agents", [])],
        }
        self.commit_reject_reason_counts = Counter({
            str(k): int(v) for k, v in state.get("commit_reject_reason_counts", {}).items()
        })
        self.last_commit_reject_reason_counts = {
            str(k): int(v) for k, v in state.get("last_commit_reject_reason_counts", {}).items()
        }
        self.type_speeds = np.array(state["type_speeds"], dtype=np.float64)

        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = state["rng_state"]

        robots = {}
        for rid_raw, rd in state["robots"].items():
            rid = int(rid_raw)
            robots[rid] = Robot(
                robot_id=int(rd["robot_id"]),
                pos=np.array(rd["pos"], dtype=np.float64),
                v_max=float(rd["v_max"]),
                cap=np.array(rd["cap"], dtype=np.float64),
                state=str(rd["state"]),
                task_id=None if rd["task_id"] is None else int(rd["task_id"]),
                eta=int(rd["eta"]),
                work_left=int(rd["work_left"]),
                agent_type=int(rd["agent_type"]),
            )
        self.robots = robots

        tasks = {}
        for tid_raw, td in state["tasks"].items():
            tid = int(tid_raw)
            tasks[tid] = Task(
                task_id=int(td["task_id"]),
                loc=np.array(td["loc"], dtype=np.float64),
                dst_loc=None if td["dst_loc"] is None else np.array(td["dst_loc"], dtype=np.float64),
                req=np.array(td["req"], dtype=np.float64),
                workload=float(td["workload"]),
                demand_count=int(td["demand_count"]),
                k=int(td["k"]),
                release_time=int(td["release_time"]),
                task_type=int(td["task_type"]),
                exec_mode=str(td["exec_mode"]),
                type_time_scale=float(td["type_time_scale"]),
                status=str(td["status"]),
                assigned_robots=[int(x) for x in td["assigned_robots"]],
                t_start=int(td["t_start"]),
                t_finish_phys=int(td["t_finish_phys"]),
                t_finish=int(td["t_finish"]),
                logical_wait_steps=int(td.get("logical_wait_steps", 0)),
                last_logical_wait_reason=str(td.get("last_logical_wait_reason", "")),
                duration_steps=int(td["duration_steps"]),
                remain_phys=float(td["remain_phys"]),
                ss_window_missed=bool(td["ss_window_missed"]),
                window_missed_edges=set(str(x) for x in td.get("window_missed_edges", [])),
                has_rejected_commitment=bool(td.get("has_rejected_commitment", False)),
                has_infeasible_team=bool(td.get("has_infeasible_team", False)),
                has_dependency_violation=bool(td.get("has_dependency_violation", False)),
                has_start_violation=bool(td.get("has_start_violation", False)),
                has_finish_violation=bool(td.get("has_finish_violation", False)),
                has_pseudo_completion=bool(td.get("has_pseudo_completion", False)),
                earliest_start=float(td.get("earliest_start", 0.0)),
                earliest_finish=float(td.get("earliest_finish", 0.0)),
                latest_start=float(td.get("latest_start", 0.0)),
                latest_finish=float(td.get("latest_finish", 0.0)),
                slack=float(td.get("slack", 0.0)),
                is_temporally_feasible=bool(td.get("is_temporally_feasible", True)),
                num_unsatisfied_predecessors=int(td.get("num_unsatisfied_predecessors", 0)),
                criticality=float(td.get("criticality", 0.0)),
            )
        self.tasks = tasks

        self.edges = [
            DependencyEdge(u=int(e["u"]), v=int(e["v"]), rel_type=str(e["rel_type"]), lag=int(e["lag"]))
            for e in state["edges"]
        ]
        self.gen_planned_start_steps = {int(k): int(v) for k, v in state.get("gen_planned_start_steps", {}).items()}
        self.gen_planned_finish_steps = {int(k): int(v) for k, v in state.get("gen_planned_finish_steps", {}).items()}
        self._accepted_edges_since_global_check = int(state.get("accepted_edges_since_global_check", 0))
        self._last_generation_check = {
            "horizon_violations": int(state.get("last_generation_check", {}).get("horizon_violations", 0)),
            "window_violations": int(state.get("last_generation_check", {}).get("window_violations", 0)),
            "checked_tasks": int(state.get("last_generation_check", {}).get("checked_tasks", len(self.tasks))),
            "checked_edges": int(state.get("last_generation_check", {}).get("checked_edges", len(self.edges))),
        }
        self.current_scenario_difficulty = float(state.get("scenario_difficulty", 0.0))
        raw_components = state.get("scenario_difficulty_components", {})
        self.current_scenario_difficulty_components = {
            str(k): float(v) for k, v in raw_components.items()
        } if isinstance(raw_components, dict) else {}
        self.last_reset_attempts = int(state.get("last_reset_attempts", 1))
        self.last_reset_range_ok = bool(state.get("last_reset_range_ok", True))
        self.last_generation_feasible = bool(state.get("last_generation_feasible", True))
        if not self.gen_planned_start_steps or not self.gen_planned_finish_steps:
            self._quick_generation_global_check()

    def step(self, actions_dict: Dict[int, int]):
        """执行一个环境时间步。

        输入：
        - actions_dict: {robot_id: action_id}，其中 0 为 NOOP，a>0 对应任务 a-1。

        输出：
        - obs: 下一时刻观测。
        - rewards_dict: 每个机器人同一共享奖励。
        - terminated: 是否成功终止。
        - truncated: 是否达到最大步数截断。
        - info: 评估统计（makespan、success、约束违例计数等）。
        """
        step_t0 = time.perf_counter()
        prev_cache = self._obs_runtime_cache
        self._obs_runtime_cache = {"can_commit": {}}
        prev_missed_ss = self.missed_ss_window
        prev_infeasible = self.infeasible_team
        prev_rejected = self.rejected_commitments
        prev_pred_ready = self._count_temporal_ready_edges()
        prev_dependency_violation = sum(
            1
            for t in self.tasks.values()
            if t.has_dependency_violation or t.has_start_violation
        )
        local_credit = {rid: 0.0 for rid in self.robots.keys()}
        credit_events = {
            "reject_agents": [],
            "infeasible_agents": [],
            "noop_feasible_agents": [],
            "reject_reason_counts": {},
        }

        t0 = time.perf_counter()
        self._dynamic_arrival()
        self._perf_stats["dynamic_arrival_sec"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        if self.config.ordered_commitment:
            self._ordered_commitment(actions_dict, local_credit, credit_events)
        else:
            self._simultaneous_commitment(actions_dict, local_credit, credit_events)
        self._perf_stats["commitment_sec"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        self._progress_robots_and_tasks()
        self._perf_stats["progress_sec"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        self._refresh_task_statuses()
        self._perf_stats["refresh_status_sec"] += time.perf_counter() - t0
        pred_ready_now = self._count_temporal_ready_edges()
        pred_progress = max(0, pred_ready_now - prev_pred_ready)

        current_done_total = sum(1 for t in self.tasks.values() if t.status == "DONE")
        done_now = max(0, current_done_total - self.total_done)
        self.total_done = current_done_total
        current_recognized_done_total = sum(1 for t in self.tasks.values() if self._task_is_recognized_done(t, truncated=False))
        recognized_done_now = max(0, current_recognized_done_total - self.total_recognized_done)
        self.total_recognized_done = current_recognized_done_total

        waiting_count = self._count_waiting_robots()
        logical_wait_count = self._count_logical_wait_robots()
        idle_count = sum(1 for r in self.robots.values() if r.state == "IDLE")
        self.waiting_cost += waiting_count
        self.logical_wait_cost += logical_wait_count

        missed_now = self.missed_ss_window - prev_missed_ss
        infeasible_now = self.infeasible_team - prev_infeasible
        reject_now = self.rejected_commitments - prev_rejected
        dependency_violation_now = max(
            0,
            sum(
                1
                for t in self.tasks.values()
                if t.has_dependency_violation or t.has_start_violation
            ) - prev_dependency_violation,
        )
        total_tasks = max(1.0, float(len(self.tasks)))
        total_edges = max(1.0, float(len(self.edges)))
        total_robots = max(1.0, float(len(self.robots)))
        horizon_steps = max(1.0, float(self.config.max_steps))
        action_mask_snapshot, _ = self._get_step_action_snapshot()
        feasible_action_count = float(np.sum(action_mask_snapshot[:, 1:]))
        forced_noop_count, noop_when_feasible_count, feasible_agent_count = self._forced_noop_stats(actions_dict, action_mask_snapshot)
        noop_when_feasible = float(noop_when_feasible_count) / max(1.0, float(feasible_agent_count))
        forced_noop_ratio = float(forced_noop_count) / total_robots
        noop_agents = [
            rid for rid in sorted(self.robots.keys())
            if int(actions_dict.get(rid, 0)) == 0 and bool(np.sum(action_mask_snapshot[rid, 1:]) > 0)
        ]
        if noop_agents:
            for rid in noop_agents:
                local_credit[rid] -= _LOCAL_NOOP_FEASIBLE_PENALTY
            credit_events["noop_feasible_agents"] = [int(x) for x in noop_agents]

        recognized_done_norm = float(recognized_done_now) / total_tasks
        total_completion_norm = float(self.total_recognized_done) / total_tasks
        pred_norm = float(pred_progress) / total_edges
        wait_norm = float(waiting_count) / total_robots
        logical_wait_norm = float(logical_wait_count) / total_robots
        time_norm = float(self.current_time) / horizon_steps
        miss_norm = float(missed_now) / total_edges
        infeasible_norm = float(infeasible_now) / total_robots
        reject_norm = float(reject_now) / total_robots
        dependency_violation_norm = float(dependency_violation_now) / total_tasks
        completion_gap = max(0.0, 1.0 - total_completion_norm)

        # 奖励超参数（已降维：仅 4 个语义旋钮，其余为冻结常量）
        w_done = float(getattr(self.config, "reward_recognized_immediate_weight", 36.0))
        w_prog = float(getattr(self.config, "reward_progress_weight", 0.12))
        w_cum = float(getattr(self.config, "reward_total_completion_weight", 3.0))
        w_vio = float(getattr(self.config, "reward_violation_weight", 1.2))
        violation_term = (
            0.40 * miss_norm
            + 0.35 * dependency_violation_norm
            + 0.15 * reject_norm
            + 0.10 * infeasible_norm
        )
        efficiency_term = 0.35 * time_norm + 0.15 * wait_norm + 0.25 * logical_wait_norm + 0.25 * noop_when_feasible
        gated_efficiency_term = completion_gap * efficiency_term
        reward = (
            w_done * recognized_done_norm
            + w_prog * pred_norm
            + w_cum * total_completion_norm
            - w_vio * violation_term
            - _REWARD_EFFICIENCY_WEIGHT * gated_efficiency_term
            - _REWARD_NOOP_FEASIBLE_WEIGHT * completion_gap * noop_when_feasible
            - _REWARD_LOGICAL_WAIT_WEIGHT * logical_wait_norm
        )
        local_credit_weight = _LOCAL_CREDIT_WEIGHT

        terminated = self._all_done() and all(r.state == "IDLE" for r in self.robots.values())
        truncated = self.current_time >= self.config.max_steps
        strict_success = bool(self._all_done() and not truncated and self.missed_ss_window == 0)
        recognized_done_tasks = sum(1 for t in self.tasks.values() if self._task_is_recognized_done(t, truncated=truncated))
        success_rate = float(recognized_done_tasks) / max(1.0, float(len(self.tasks)))
        recognized_done_rate = float(recognized_done_tasks) / max(1.0, float(len(self.tasks)))

        task_status_counts = self._task_status_counts()
        logical_wait_reason_counts = self._logical_wait_reason_summary()
        logical_wait_bucket_counts = self._logical_wait_bucket_summary()
        logical_wait_tasks = int(task_status_counts.get("LOGICAL_WAIT", 0))
        phys_done_tasks = int(task_status_counts.get("PHYS_DONE", 0))
        done_tasks = int(task_status_counts.get("DONE", 0))
        logical_wait_reason_top = sorted(
            logical_wait_reason_counts.items(),
            key=lambda kv: (-int(kv[1]), kv[0]),
        )[:5]
        logical_wait_bucket_top = sorted(
            logical_wait_bucket_counts.items(),
            key=lambda kv: (-int(kv[1]), kv[0]),
        )[:5]

        info = {
            "time": self.current_time,
            "done_now": done_now,
            "total_done": int(self.total_done),
            "recognized_done_now": int(recognized_done_now),
            "total_recognized_done": int(self.total_recognized_done),
            "total_tasks": int(len(self.tasks)),
            "waiting_count": waiting_count,
            "logical_wait_count": logical_wait_count,
            "idle_count": idle_count,
            "rejected_commitments": self.rejected_commitments,
            "missed_ss_window": self.missed_ss_window,
            "missed_window": self.missed_ss_window,
            "infeasible_team": self.infeasible_team,
            "makespan": self._makespan(),
            "success": success_rate,
            "strict_success": float(strict_success),
            "recognized_done_tasks": int(recognized_done_tasks),
            "recognized_done_rate": float(recognized_done_rate),
            "phys_done_tasks": int(phys_done_tasks),
            "phys_done_rate": float(phys_done_tasks) / total_tasks,
            "done_tasks": int(done_tasks),
            "done_rate": float(done_tasks) / total_tasks,
            "logical_wait_tasks": int(logical_wait_tasks),
            "logical_wait_rate": float(logical_wait_tasks) / total_tasks,
            "waiting_cost": self.waiting_cost,
            "logical_wait_cost": self.logical_wait_cost,
            "avg_logical_wait_steps": float(
                np.mean([float(t.logical_wait_steps) for t in self.tasks.values()]) if self.tasks else 0.0
            ),
            "avg_logical_wait_steps_active": float(
                np.mean([float(t.logical_wait_steps) for t in self.tasks.values() if t.logical_wait_steps > 0])
                if any(t.logical_wait_steps > 0 for t in self.tasks.values())
                else 0.0
            ),
            "task_status_counts": task_status_counts,
            "logical_wait_reason_counts": logical_wait_reason_counts,
            "logical_wait_bucket_counts": logical_wait_bucket_counts,
            "logical_wait_reason_top": [
                {"reason": str(reason), "count": int(count)} for reason, count in logical_wait_reason_top
            ],
            "logical_wait_bucket_top": [
                {"bucket": str(bucket), "count": int(count)} for bucket, count in logical_wait_bucket_top
            ],
            "reward": reward,
            "reward_progress_weight": w_prog,
            "reward_total_completion_weight": w_cum,
            "reward_efficiency_weight": _REWARD_EFFICIENCY_WEIGHT,
            "reward_noop_feasible_weight": _REWARD_NOOP_FEASIBLE_WEIGHT,
            "reward_violation_weight": w_vio,
            "reward_recognized_immediate_weight": w_done,
            "pred_progress": int(pred_progress),
            "reward_recognized_done_norm": recognized_done_norm,
            "reward_total_completion_norm": total_completion_norm,
            "reward_noop_when_feasible": noop_when_feasible,
            "feasible_action_count": float(feasible_action_count),
            "forced_noop_count": int(forced_noop_count),
            "forced_noop_ratio": float(forced_noop_ratio),
            "feasible_agent_count": int(feasible_agent_count),
            "noop_when_feasible_count": int(noop_when_feasible_count),
            "reward_wait_norm": wait_norm,
            "reward_logical_wait_norm": logical_wait_norm,
            "reward_time_norm": time_norm,
            "reward_completion_gap": completion_gap,
            "reward_miss_norm": miss_norm,
            "reward_infeasible_norm": infeasible_norm,
            "reward_reject_norm": reject_norm,
            "reward_dependency_violation_norm": dependency_violation_norm,
            "reward_pred_norm": pred_norm,
            "reward_violation_term": violation_term,
            "reward_efficiency_term": efficiency_term,
            "reward_gated_efficiency_term": gated_efficiency_term,
            "reward_logical_wait_weight": _REWARD_LOGICAL_WAIT_WEIGHT,
            "base_reward": reward,
            "local_credit_weight": _LOCAL_CREDIT_WEIGHT,
            "local_credit_mean": float(np.mean(list(local_credit.values()))) if local_credit else 0.0,
            "local_credit_min": float(np.min(list(local_credit.values()))) if local_credit else 0.0,
            "local_credit_max": float(np.max(list(local_credit.values()))) if local_credit else 0.0,
            "credit_reject_agents": [int(x) for x in credit_events.get("reject_agents", [])],
            "credit_infeasible_agents": [int(x) for x in credit_events.get("infeasible_agents", [])],
            "credit_noop_feasible_agents": [int(x) for x in credit_events.get("noop_feasible_agents", [])],
            "reject_reason_counts": {str(k): int(v) for k, v in credit_events.get("reject_reason_counts", {}).items()},
            "dependency_violation_count": int(dependency_violation_now),
            "violation_count": int(self.missed_ss_window + self.infeasible_team + dependency_violation_now),
            "violation_rate": float(self.missed_ss_window + self.infeasible_team + dependency_violation_now) / max(1.0, float(len(self.edges))),
            "generalized_makespan": float(self._makespan()) / max(1.0, float(recognized_done_tasks)),
            "scenario_difficulty": float(self.current_scenario_difficulty),
            "scenario_difficulty_reset_attempts": int(self.last_reset_attempts),
            "scenario_difficulty_in_range": float(bool(self.last_reset_range_ok)),
        }
        obs = self._get_obs()
        rewards_dict = {
            rid: float(reward + local_credit_weight * float(local_credit.get(rid, 0.0)))
            for rid in self.robots.keys()
        }
        self.last_local_credit_rewards = {int(k): float(v) for k, v in local_credit.items()}
        self.last_credit_events = {
            "reject_agents": [int(x) for x in credit_events.get("reject_agents", [])],
            "infeasible_agents": [int(x) for x in credit_events.get("infeasible_agents", [])],
            "noop_feasible_agents": [int(x) for x in credit_events.get("noop_feasible_agents", [])],
        }
        self.last_commit_reject_reason_counts = {
            str(k): int(v) for k, v in credit_events.get("reject_reason_counts", {}).items()
        }
        self._perf_stats["step_calls"] += 1.0
        self._perf_stats["step_total_sec"] += time.perf_counter() - step_t0
        self._obs_runtime_cache = prev_cache
        return obs, rewards_dict, terminated, truncated, info

    def get_avail_actions(self, agent_id: int) -> np.ndarray:
        """为单机器人生成动作掩码。

        输入：
        - agent_id: 机器人编号。

        输出：
        - mask: shape=(max_tasks+1,)；第 0 维是 NOOP，1 表示可选。
        """
        task_count = self.config.max_tasks
        mask = np.zeros(task_count + 1, dtype=np.float32)
        mask[0] = 1.0  # NOOP
        robot = self.robots[agent_id]
        if robot.state != "IDLE":
            return mask

        runtime_cache = self._obs_runtime_cache
        if runtime_cache is not None:
            cached_action_mask = runtime_cache.get("action_mask")
            if isinstance(cached_action_mask, np.ndarray) and 0 <= int(agent_id) < cached_action_mask.shape[0]:
                return cached_action_mask[int(agent_id)].copy()
        action_mask, _, _ = self._build_action_mask_and_commit_scores()
        if 0 <= int(agent_id) < action_mask.shape[0]:
            return action_mask[int(agent_id)].copy()
        return mask

    def _generate_robots(self) -> Dict[int, Robot]:
        """随机生成机器人集合。输出 {robot_id: Robot}。"""
        robots = {}
        for i in range(self.config.num_agents):
            pos = self.rng.uniform([0, 0], self.config.area_size)
            cap = np.zeros((self.config.cap_dim,), dtype=np.float64)
            agent_type = int(self.rng.integers(0, self.task_type_dim))
            cap[agent_type] = float(self.rng.uniform(*self.config.agent_type_cap_range))
            v_max = float(self.type_speeds[agent_type])
            robots[i] = Robot(i, pos, v_max, cap, agent_type=agent_type)
        return robots

    def _build_non_distance_type_scales(self) -> np.ndarray:
        """构建各任务类型的非距离型时间参数（秒尺度）。"""
        d_min = max(float(self.config.task_demand_range[0]), 1e-6)
        d_max = max(float(self.config.task_demand_range[1]), d_min)
        c_min = max(float(self.config.agent_type_cap_range[0]), 1e-6)
        c_max = max(float(self.config.agent_type_cap_range[1]), c_min)

        # 依据单机器人上下界做参数标定：尽量使最短时长 >= 60s、最长时长 <= 600s。
        scale_low = 60.0 * c_max / d_min
        scale_high = 600.0 * c_min / d_max
        default_scale = (scale_low + scale_high) * 0.5 if scale_low <= scale_high else scale_low

        cfg_scales = getattr(self.config, "non_distance_type_scales", ())
        if len(cfg_scales) == self.task_type_dim:
            out = np.array(cfg_scales, dtype=np.float64)
            out = np.where(out > 0.0, out, default_scale)
            return out
        return np.full((self.task_type_dim,), default_scale, dtype=np.float64)

    def _spawn_tasks(self, n_new: int):
        """生成新任务并与既有任务随机连边（依赖类型 + lag）。输入新增任务数，输出 None。"""
        if n_new <= 0:
            return
        total_tasks_cap = int(getattr(self.config, "total_tasks_cap", self.config.max_tasks))
        total_tasks_cap = max(0, min(total_tasks_cap, int(self.config.max_tasks)))
        type_robot_count = np.zeros((self.task_type_dim,), dtype=np.int64)
        for r in self.robots.values():
            if 0 <= r.agent_type < self.task_type_dim:
                type_robot_count[r.agent_type] += 1
        available_types = [i for i in range(self.task_type_dim) if type_robot_count[i] > 0]
        if not available_types:
            available_types = list(range(self.task_type_dim))

        existing = list(self.tasks.keys())
        for _ in range(n_new):
            if len(self.tasks) >= total_tasks_cap:
                break
            tid = self.next_task_id
            self.next_task_id += 1
            loc = self.rng.uniform([0, 0], self.config.area_size)
            task_type = int(self.rng.choice(available_types))
            type_count = int(type_robot_count[task_type]) if task_type < len(type_robot_count) else 0
            demand = float(self.rng.uniform(*self.config.task_demand_range))
            req = np.zeros((self.config.cap_dim,), dtype=np.float64)
            req[task_type] = demand
            workload = float(self.rng.uniform(*self.config.task_workload_range))
            demand_count = max(1, int(math.ceil(demand)))
            if type_count > 0:
                demand_count = min(demand_count, type_count)
            is_distance = task_type in self.distance_task_types
            if is_distance:
                k = demand_count
                dst_loc = self.rng.uniform([0, 0], self.config.area_size)
                exec_mode = "DISTANCE"
            else:
                k_hi = max(1, 2 * demand_count)
                if type_count > 0:
                    k_hi = min(k_hi, type_count)
                k = int(self.rng.integers(1, k_hi + 1))
                dst_loc = None
                exec_mode = "NON_DISTANCE"
            self.tasks[tid] = Task(
                task_id=tid,
                loc=loc,
                dst_loc=dst_loc,
                req=req,
                workload=workload,
                demand_count=demand_count,
                k=k,
                release_time=self.current_time,
                task_type=task_type,
                exec_mode=exec_mode,
                type_time_scale=float(self.non_distance_type_scales[task_type]),
                status="AVAILABLE",
            )
            if bool(getattr(self.config, "use_structured_graph_generator", True)):
                self._connect_structured_edges(existing, tid)
            else:
                for u in existing:
                    if self.rng.random() < self.config.edge_density:
                        self._try_add_edge(u=u, v=tid, structured=False)

            incoming = [e for e in self.edges if e.v == tid]
            est_start, est_finish, _ = self._estimate_task_bounds(
                self.tasks[tid], incoming, self.gen_planned_start_steps, self.gen_planned_finish_steps
            )
            self.gen_planned_start_steps[tid] = int(est_start)
            self.gen_planned_finish_steps[tid] = int(est_finish)
            existing.append(tid)

    def _estimate_task_duration_steps(self, task: Task) -> int:
        if task.exec_mode == "DISTANCE":
            dst = task.dst_loc if task.dst_loc is not None else task.loc
            distance = float(np.linalg.norm(task.loc - dst))
            speed = float(max(self.type_speeds[task.task_type], 1e-6))
            t_phys = distance / speed
        else:
            demand = float(task.req[task.task_type])
            workload_ref = max((self.config.task_workload_range[0] + self.config.task_workload_range[1]) * 0.5, 1e-6)
            workload_factor = float(task.workload) / workload_ref
            avg_cap = (self.config.agent_type_cap_range[0] + self.config.agent_type_cap_range[1]) * 0.5
            t_phys = demand * workload_factor / max(avg_cap, 1e-6) * float(task.type_time_scale)
            t_min, t_max = self.non_distance_time_bounds
            t_phys = float(np.clip(t_phys, t_min, t_max))
        return max(1, int(math.ceil(max(t_phys, self.config.dt_seconds) / max(self.config.dt_seconds, 1e-6))))

    def _sample_structured_lag(self, pred_task: Task, rel_type: str) -> int:
        base_steps = max(1, self._estimate_task_duration_steps(pred_task))
        hi = int(max(1, round(0.6 * base_steps)))
        if rel_type in ["SS", "SF"]:
            hi = int(max(1, round(0.3 * base_steps)))
        hi = min(hi, int(self.config.lag_range[1]))
        lo = int(self.config.lag_range[0])
        hi = max(hi, lo)
        return int(self.rng.integers(lo, hi + 1))

    def _sample_edge_lag(self, pred_task: Task, succ_task: Task, rel_type: str, structured: bool) -> int:
        lo = int(self.config.lag_range[0])
        hi = int(self.config.lag_range[1])
        hi = max(hi, lo)
        if structured:
            lag = int(self._sample_structured_lag(pred_task, rel_type))
        else:
            lag = int(self.rng.integers(lo, hi + 1))
        return int(np.clip(lag, lo, hi))

    def _relation_lag_upper_bound(self, pred_task: Task, succ_task: Task, rel_type: str) -> int:
        lo = int(self.config.lag_range[0])
        hi = int(self.config.lag_range[1])
        hi = max(hi, lo)

        pred_steps = max(1, self._estimate_task_duration_steps(pred_task))
        vmax = float(max(self.config.v_max_range[1], 1e-6))
        dist_uv = float(np.linalg.norm(pred_task.loc - succ_task.loc))
        travel_steps = int(math.ceil(dist_uv / vmax / max(self.config.dt_seconds, 1e-6)))

        if rel_type == "FS":
            bound = int(max(1, round(0.4 * pred_steps + 0.6 * travel_steps)))
        elif rel_type == "SS":
            bound = int(max(1, round(0.25 * pred_steps)))
        elif rel_type == "FF":
            bound = int(max(1, round(0.3 * pred_steps)))
        else:  # SF
            bound = int(max(1, round(0.2 * pred_steps)))

        return int(min(hi, max(lo, bound)))

    def _max_reposition_steps_to_task(self, task: Task) -> int:
        w = float(self.config.area_size[0])
        h = float(self.config.area_size[1])
        corners = [
            np.array([0.0, 0.0], dtype=np.float64),
            np.array([w, 0.0], dtype=np.float64),
            np.array([0.0, h], dtype=np.float64),
            np.array([w, h], dtype=np.float64),
        ]
        anchors = corners + [t.loc for tid, t in self.tasks.items() if tid != task.task_id]
        if not anchors:
            return 0
        vmax = float(max(self.config.v_max_range[1], 1e-6))
        max_dist = max(float(np.linalg.norm(p - task.loc)) for p in anchors)
        return int(math.ceil(max_dist / vmax / max(self.config.dt_seconds, 1e-6)))

    def _min_window_steps_for_edge(self, succ_task: Task, rel_type: str) -> int:
        margin = int(max(0, getattr(self.config, "window_slack_steps", 2)))
        arrival_steps = self._max_reposition_steps_to_task(succ_task)
        if rel_type in ["FS", "SS"]:
            return int(arrival_steps + margin)
        return int(arrival_steps + self._estimate_task_duration_steps(succ_task) + margin)

    def _estimate_task_bounds(
        self,
        task: Task,
        incoming: List[DependencyEdge],
        start_cache: Dict[int, int],
        finish_cache: Dict[int, int],
    ) -> Tuple[int, int, bool]:
        duration = max(1, self._estimate_task_duration_steps(task))
        start_lb = int(task.release_time)
        finish_lb = int(task.release_time + duration)

        for e in incoming:
            pred = self.tasks[e.u]
            pred_start = int(start_cache.get(e.u, pred.release_time))
            pred_finish = int(
                finish_cache.get(e.u, pred_start + max(1, self._estimate_task_duration_steps(pred)))
            )
            if e.rel_type == "FS":
                start_lb = max(start_lb, pred_finish + int(e.lag))
            elif e.rel_type == "SS":
                start_lb = max(start_lb, pred_start + int(e.lag))
            elif e.rel_type == "FF":
                finish_lb = max(finish_lb, pred_finish + int(e.lag))
            else:  # SF
                finish_lb = max(finish_lb, pred_start + int(e.lag))

        start_est = max(start_lb, finish_lb - duration)
        finish_est = max(finish_lb, start_est + duration)
        feasible = bool(finish_est <= int(self.config.max_steps))
        return int(start_est), int(finish_est), feasible

    def _passes_local_edge_guard(self, u: int, v: int, rel: str, lag: int) -> bool:
        if not bool(getattr(self.config, "edge_feasibility_guard", True)):
            return True
        succ = self.tasks[v]
        pred = self.tasks[u]

        max_in_degree = int(max(1, getattr(self.config, "edge_max_in_degree", getattr(self.config, "structured_max_parents", 3))))
        in_degree = sum(1 for e in self.edges if e.v == v)
        if in_degree + 1 > max_in_degree:
            return False

        if rel == "SF":
            sf_accept_prob = float(np.clip(getattr(self.config, "sf_edge_accept_prob", 0.25), 0.0, 1.0))
            if self.rng.random() > sf_accept_prob:
                return False

        rel_lag_hi = self._relation_lag_upper_bound(pred, succ, rel)
        if int(lag) > int(rel_lag_hi):
            return False

        min_window = self._min_window_steps_for_edge(succ, rel)
        if int(self.rel_window_steps.get(rel, 1)) < int(min_window):
            return False

        return True

    def _passes_incremental_temporal_guard(self, u: int, v: int, rel: str, lag: int) -> bool:
        if not bool(getattr(self.config, "edge_feasibility_guard", True)):
            return True
        trial_incoming = [e for e in self.edges if e.v == v]
        trial_incoming.append(DependencyEdge(u=u, v=v, rel_type=rel, lag=int(lag)))
        _, finish_est, feasible = self._estimate_task_bounds(
            self.tasks[v], trial_incoming, self.gen_planned_start_steps, self.gen_planned_finish_steps
        )
        if not feasible:
            return False
        return bool(finish_est <= int(self.config.max_steps))

    def _quick_generation_global_check(self) -> bool:
        incoming_map: Dict[int, List[DependencyEdge]] = {tid: [] for tid in self.tasks.keys()}
        for e in self.edges:
            incoming_map.setdefault(e.v, []).append(e)

        plan_start: Dict[int, int] = {}
        plan_finish: Dict[int, int] = {}
        horizon_violations = 0
        window_violations = 0

        for tid in sorted(self.tasks.keys()):
            task = self.tasks[tid]
            incoming = incoming_map.get(tid, [])
            est_start, est_finish, feasible = self._estimate_task_bounds(task, incoming, plan_start, plan_finish)
            if not feasible:
                horizon_violations += 1
            for e in incoming:
                if int(self.rel_window_steps.get(e.rel_type, 1)) < self._min_window_steps_for_edge(task, e.rel_type):
                    window_violations += 1
            plan_start[tid] = int(est_start)
            plan_finish[tid] = int(est_finish)

        self._last_generation_check = {
            "horizon_violations": int(horizon_violations),
            "window_violations": int(window_violations),
            "checked_tasks": int(len(self.tasks)),
            "checked_edges": int(len(self.edges)),
        }

        if horizon_violations == 0 and window_violations == 0:
            self.gen_planned_start_steps = plan_start
            self.gen_planned_finish_steps = plan_finish
            return True
        return False

    def generation_feasibility_summary(self) -> Dict[str, int]:
        feasible = self._quick_generation_global_check()
        return {
            "feasible": int(bool(feasible)),
            "num_tasks": int(len(self.tasks)),
            "num_edges": int(len(self.edges)),
            "horizon_violations": int(self._last_generation_check.get("horizon_violations", 0)),
            "window_violations": int(self._last_generation_check.get("window_violations", 0)),
            "checked_tasks": int(self._last_generation_check.get("checked_tasks", len(self.tasks))),
            "checked_edges": int(self._last_generation_check.get("checked_edges", len(self.edges))),
        }

    def scenario_difficulty_summary(self) -> Dict[str, float]:
        n_tasks = int(len(self.tasks))
        n_edges = int(len(self.edges))
        if n_tasks <= 0:
            return {
                "score": 0.0,
                "task_ratio": 0.0,
                "edge_ratio": 0.0,
                "collab_ratio": 0.0,
                "horizon_ratio": 0.0,
                "window_tightness": 0.0,
                "dynamic_ratio": 0.0,
            }

        total_tasks_cap = max(1, int(getattr(self.config, "total_tasks_cap", self.config.max_tasks)))
        task_ratio = float(np.clip(n_tasks / float(total_tasks_cap), 0.0, 1.0))

        max_in_degree = int(max(1, getattr(self.config, "edge_max_in_degree", 3)))
        edge_capacity = max(1, n_tasks * max_in_degree)
        edge_ratio = float(np.clip(n_edges / float(edge_capacity), 0.0, 1.0))

        avg_k = float(np.mean([float(t.k) for t in self.tasks.values()]))
        max_k = int(max(1, self.config.task_k_range[1]))
        collab_ratio = float(np.clip((avg_k - 1.0) / float(max(1, max_k - 1)), 0.0, 1.0))

        if self.gen_planned_finish_steps:
            mean_finish = float(np.mean(list(self.gen_planned_finish_steps.values())))
        else:
            mean_finish = float(np.mean([self._estimate_task_duration_steps(t) for t in self.tasks.values()]))
        horizon_ratio = float(np.clip(mean_finish / float(max(1, self.config.max_steps)), 0.0, 1.0))

        window_pressures = []
        for e in self.edges:
            succ = self.tasks.get(e.v)
            if succ is None:
                continue
            min_w = float(self._min_window_steps_for_edge(succ, e.rel_type))
            rel_w = float(max(1, self.rel_window_steps.get(e.rel_type, 1)))
            window_pressures.append(min_w / rel_w)
        window_tightness = float(np.clip(np.mean(window_pressures) if window_pressures else 0.0, 0.0, 1.0))

        dyn_every = int(self.config.dynamic_arrival_every)
        if dyn_every <= 0:
            dynamic_ratio = 0.0
        else:
            events = float(max(1, self.config.max_steps)) / float(dyn_every)
            arr_avg = 0.5 * float(self.config.dynamic_arrival_range[0] + self.config.dynamic_arrival_range[1])
            init_avg = 0.5 * float(self.config.init_tasks_range[0] + self.config.init_tasks_range[1])
            extra_budget = max(1.0, float(total_tasks_cap) - init_avg)
            dynamic_ratio = float(np.clip((events * arr_avg) / extra_budget, 0.0, 1.0))

        score = (
            0.25 * task_ratio
            + 0.20 * edge_ratio
            + 0.15 * collab_ratio
            + 0.20 * horizon_ratio
            + 0.10 * window_tightness
            + 0.10 * dynamic_ratio
        )
        return {
            "score": float(np.clip(score, 0.0, 1.0)),
            "task_ratio": task_ratio,
            "edge_ratio": edge_ratio,
            "collab_ratio": collab_ratio,
            "horizon_ratio": horizon_ratio,
            "window_tightness": window_tightness,
            "dynamic_ratio": dynamic_ratio,
        }

    def _try_add_edge(self, u: int, v: int, structured: bool) -> bool:
        if u not in self.tasks or v not in self.tasks or u == v:
            return False

        if not bool(getattr(self.config, "edge_feasibility_guard", True)):
            rel = self._sample_rel_type()
            lag = self._sample_edge_lag(self.tasks[u], self.tasks[v], rel, structured)
            self.edges.append(DependencyEdge(u=u, v=v, rel_type=rel, lag=lag))
            return True

        attempts = int(max(1, getattr(self.config, "edge_candidate_resamples", 3)))
        global_check_interval = int(max(1, getattr(self.config, "edge_global_check_interval", 5)))
        for _ in range(attempts):
            rel = self._sample_rel_type()
            lag = self._sample_edge_lag(self.tasks[u], self.tasks[v], rel, structured)
            if not self._passes_local_edge_guard(u, v, rel, lag):
                continue
            if not self._passes_incremental_temporal_guard(u, v, rel, lag):
                continue

            self.edges.append(DependencyEdge(u=u, v=v, rel_type=rel, lag=lag))
            self._accepted_edges_since_global_check += 1

            if self._accepted_edges_since_global_check >= global_check_interval:
                if not self._quick_generation_global_check():
                    self.edges.pop()
                    self._accepted_edges_since_global_check = max(0, self._accepted_edges_since_global_check - 1)
                    continue
                self._accepted_edges_since_global_check = 0
            return True
        return False

    def _connect_structured_edges(self, existing: List[int], tid: int):
        if not existing:
            return
        span = int(max(1, getattr(self.config, "structured_parent_span", 12)))
        cand = existing[-span:]
        if not cand:
            return
        max_parents = int(max(0, min(getattr(self.config, "structured_max_parents", 3), len(cand))))
        min_parents = int(max(0, min(getattr(self.config, "structured_min_parents", 0), max_parents)))

        exp_parents = int(round(self.config.edge_density * len(cand)))
        parent_count = int(np.clip(exp_parents, min_parents, max_parents))
        if parent_count <= 0:
            return

        parents = list(self.rng.choice(np.array(cand, dtype=np.int64), size=parent_count, replace=False))
        for u in sorted(int(x) for x in parents):
            self._try_add_edge(u=u, v=tid, structured=True)

    def _sample_rel_type(self) -> str:
        """按 relation_ratio 采样关系类型（FS/SS/FF/SF）。输出关系名字符串。"""
        probs = np.array([self.config.relation_ratio[k] for k in REL_NAMES], dtype=np.float64)
        probs = probs / probs.sum()
        return str(self.rng.choice(REL_NAMES, p=probs))

    def _dynamic_arrival(self):
        """按 dynamic_arrival_every 触发动态任务到达。输出 None。"""
        if self.config.dynamic_arrival_every <= 0:
            return
        if self.current_time > 0 and self.current_time % self.config.dynamic_arrival_every == 0:
            n_new = int(self.rng.integers(self.config.dynamic_arrival_range[0], self.config.dynamic_arrival_range[1] + 1))
            self._spawn_tasks(n_new)

    def _ordered_commitment(self, actions_dict: Dict[int, int], local_credit: Dict[int, float], credit_events: CreditEventMap):
        """按机器人 id 顺序处理承诺动作（ordered commitment 机制）。输出 None。"""
        if bool(getattr(self.config, "capacity_aware_commitment_arbitration", True)):
            proposals: Dict[int, List[int]] = {}
            for rid in sorted(self.robots.keys()):
                action = int(actions_dict.get(rid, 0))
                if action <= 0:
                    continue
                proposals.setdefault(action - 1, []).append(int(rid))

            selected_by_rid: Dict[int, int] = {}
            for tid, rids in proposals.items():
                if tid not in self.tasks:
                    for rid in rids:
                        self._can_commit(rid, tid, dry_run=False, local_credit=local_credit, credit_events=credit_events)
                    continue
                task = self.tasks[int(tid)]
                remaining_slots = max(0, int(task.k) - len(task.assigned_robots))
                feasible_rids: List[int] = []
                for rid in rids:
                    if self._can_commit(int(rid), int(tid), dry_run=True):
                        feasible_rids.append(int(rid))
                    else:
                        self._can_commit(int(rid), int(tid), dry_run=False, local_credit=local_credit, credit_events=credit_events)
                if remaining_slots <= 0 or not feasible_rids:
                    continue

                def _proposal_priority(rid: int) -> Tuple[float, float, int]:
                    robot = self.robots[int(rid)]
                    score = float(self._task_candidate_attractiveness(int(rid), task))
                    dist = float(np.linalg.norm(robot.pos - task.loc))
                    return (-score, dist, int(rid))

                for rid in sorted(feasible_rids, key=_proposal_priority)[:remaining_slots]:
                    selected_by_rid[int(rid)] = int(tid)

            for rid in sorted(selected_by_rid.keys()):
                self._can_commit(
                    int(rid),
                    int(selected_by_rid[rid]),
                    dry_run=False,
                    local_credit=local_credit,
                    credit_events=credit_events,
                )
            return

        for rid in sorted(self.robots.keys()):
            action = int(actions_dict.get(rid, 0))
            if action <= 0:
                continue
            tid = action - 1
            self._can_commit(rid, tid, dry_run=False, local_credit=local_credit, credit_events=credit_events)

    def _simultaneous_commitment(self, actions_dict: Dict[int, int], local_credit: Dict[int, float], credit_events: CreditEventMap):
        """并行提案后随机仲裁处理承诺动作（simultaneous commitment）。输出 None。"""
        proposals: Dict[int, List[int]] = {}
        for rid, action in actions_dict.items():
            action = int(action)
            if action <= 0:
                continue
            tid = action - 1
            if tid not in self.tasks:
                continue
            proposals.setdefault(tid, []).append(rid)

        for tid, rids in proposals.items():
            self.rng.shuffle(rids)
            for rid in rids:
                self._can_commit(rid, tid, dry_run=False, local_credit=local_credit, credit_events=credit_events)

    def _record_reject_credit(self, rid: int, tid: int, local_credit: Optional[Dict[int, float]], credit_events: Optional[Dict[str, Any]], reason: str):
        self.rejected_commitments += 1
        reason_key = str(reason) if reason else "unknown"
        self.commit_reject_reason_counts[reason_key] += 1
        if local_credit is not None:
            local_credit[rid] = float(local_credit.get(rid, 0.0)) - _LOCAL_REJECT_PENALTY
        if credit_events is not None:
            reject_agents = credit_events.setdefault("reject_agents", [])
            if rid not in reject_agents:
                reject_agents.append(int(rid))
            reason_counts = credit_events.setdefault("reject_reason_counts", {})
            reason_counts[reason_key] = int(reason_counts.get(reason_key, 0)) + 1

    def _record_infeasible_credit(self, team: List[int], local_credit: Optional[Dict[int, float]], credit_events: Optional[Dict[str, Any]]):
        if local_credit is not None:
            for rid in team:
                local_credit[rid] = float(local_credit.get(rid, 0.0)) - _LOCAL_INFEASIBLE_PENALTY
        if credit_events is not None:
            infeasible_agents = credit_events.setdefault("infeasible_agents", [])
            for rid in team:
                if rid not in infeasible_agents:
                    infeasible_agents.append(int(rid))

    def _can_commit(self, rid: int, tid: int, dry_run: bool, local_credit: Optional[Dict[int, float]] = None, credit_events: Optional[Dict[str, Any]] = None) -> bool:
        """检查并可选执行 robot->task 承诺。

        输入：
        - rid: 机器人 id。
        - tid: 任务 id。
        - dry_run: True 时只判定不落地；False 时会更新任务与机器人状态。

        输出：
        - 是否可承诺。
        """
        self._perf_stats["can_commit_calls"] += 1.0
        if dry_run:
            self._perf_stats["can_commit_dry_run_calls"] += 1.0
            runtime_cache = self._obs_runtime_cache
            if runtime_cache is not None:
                cache_map = runtime_cache.setdefault("can_commit", {})
                key = (int(rid), int(tid))
                if key in cache_map:
                    return bool(cache_map[key])
        else:
            self._perf_stats["can_commit_wet_run_calls"] += 1.0
        if tid not in self.tasks:
            if not dry_run:
                self._record_reject_credit(rid, tid, local_credit, credit_events, reason="task_missing")
            result = False
            if dry_run and self._obs_runtime_cache is not None:
                self._obs_runtime_cache.setdefault("can_commit", {})[(int(rid), int(tid))] = result
            return result
        robot = self.robots[rid]
        task = self.tasks[tid]
        if robot.state != "IDLE":
            if not dry_run:
                self._record_reject_credit(rid, tid, local_credit, credit_events, reason="robot_not_idle")
            result = False
            if dry_run and self._obs_runtime_cache is not None:
                self._obs_runtime_cache.setdefault("can_commit", {})[(int(rid), int(tid))] = result
            return result
        if task.status in ["DONE", "EXECUTING", "PHYS_DONE"]:
            if not dry_run:
                self._record_reject_credit(rid, tid, local_credit, credit_events, reason="task_unavailable_status")
            result = False
            if dry_run and self._obs_runtime_cache is not None:
                self._obs_runtime_cache.setdefault("can_commit", {})[(int(rid), int(tid))] = result
            return result
        if rid in task.assigned_robots:
            if not dry_run:
                self._record_reject_credit(rid, tid, local_credit, credit_events, reason="already_assigned")
            result = False
            if dry_run and self._obs_runtime_cache is not None:
                self._obs_runtime_cache.setdefault("can_commit", {})[(int(rid), int(tid))] = result
            return result
        if len(task.assigned_robots) >= task.k:
            if not dry_run:
                self._record_reject_credit(rid, tid, local_credit, credit_events, reason="team_full")
            result = False
            if dry_run and self._obs_runtime_cache is not None:
                self._obs_runtime_cache.setdefault("can_commit", {})[(int(rid), int(tid))] = result
            return result
        if robot.agent_type != task.task_type:
            if not dry_run:
                self._record_reject_credit(rid, tid, local_credit, credit_events, reason="agent_type_mismatch")
            result = False
            if dry_run and self._obs_runtime_cache is not None:
                self._obs_runtime_cache.setdefault("can_commit", {})[(int(rid), int(tid))] = result
            return result

        if not self._task_commitment_gate_ok(task):
            if not dry_run:
                self._record_reject_credit(rid, tid, local_credit, credit_events, reason="task_commitment_gate")
            result = False
            if dry_run and self._obs_runtime_cache is not None:
                self._obs_runtime_cache.setdefault("can_commit", {})[(int(rid), int(tid))] = result
            return result

        if self.config.strict_start_in_mask and not self._start_constraints_satisfied(task):
            if not dry_run:
                self._record_reject_credit(rid, tid, local_credit, credit_events, reason="strict_start_constraint")
            result = False
            if dry_run and self._obs_runtime_cache is not None:
                self._obs_runtime_cache.setdefault("can_commit", {})[(int(rid), int(tid))] = result
            return result

        if not dry_run:
            new_team = task.assigned_robots + [rid]
            if len(new_team) == task.k:
                if not self._team_capability_ok(task, new_team):
                    task.has_infeasible_team = True
                    self.infeasible_team += 1
                    self._record_infeasible_credit(new_team, local_credit, credit_events)
                    result = False
                    return result
            task.assigned_robots.append(rid)
            task.status = "ASSIGNED"
            robot.task_id = tid
            if bool(getattr(self.config, "allow_pre_gathering", True)):
                dist = float(np.linalg.norm(robot.pos - task.loc))
                gather_radius = float(max(getattr(self.config, "pre_gather_radius", 0.0), 0.0))
                dist_eff = max(0.0, dist - gather_radius)
                robot.eta = int(math.ceil(dist_eff / max(robot.v_max, 1e-6)))
                robot.state = "MOVING" if robot.eta > 0 else "WAITING"
                robot.moving_since = int(self.current_time) if robot.eta > 0 else -1
                robot.waiting_since = int(self.current_time) if robot.eta == 0 else -1
                robot.last_progress_time = int(self.current_time)
                if robot.eta == 0:
                    robot.pos = task.loc.copy()
            else:
                robot.state = "WAITING"
                robot.waiting_since = int(self.current_time)
                robot.moving_since = -1
                robot.last_progress_time = int(self.current_time)
            if len(task.assigned_robots) == task.k:
                if self._start_constraints_satisfied(task):
                    self._activate_task(task)
        result = True
        if dry_run and self._obs_runtime_cache is not None:
            self._obs_runtime_cache.setdefault("can_commit", {})[(int(rid), int(tid))] = result
        return result

    def _activate_task(self, task: Task):
        """激活已满编且满足起始约束的任务，设置队员 MOVING/ETA/work_left。输出 None。"""
        if not self._team_capability_ok(task, task.assigned_robots):
            return
        task.status = "READY"
        team = task.assigned_robots
        self._set_task_duration(task, team)
        for rid in team:
            r = self.robots[rid]
            dist = float(np.linalg.norm(r.pos - task.loc))
            gather_radius = float(max(getattr(self.config, "pre_gather_radius", 0.0), 0.0))
            dist_eff = max(0.0, dist - gather_radius)
            eta = int(math.ceil(dist_eff / max(r.v_max, 1e-6)))
            r.eta = eta
            r.state = "MOVING" if eta > 0 else "WAITING"
            r.task_id = task.task_id
            r.work_left = task.duration_steps
            r.moving_since = int(self.current_time) if eta > 0 else -1
            r.waiting_since = int(self.current_time) if eta == 0 else -1
            r.last_progress_time = int(self.current_time)
            if eta == 0:
                r.pos = task.loc.copy()

    def _set_task_duration(self, task: Task, team: List[int]):
        """按距离型/非距离型规则计算任务物理时长（秒）并离散化为步数。"""
        if not team:
            task.duration_steps = 1
            task.remain_phys = self.config.dt_seconds
            return

        if task.exec_mode == "DISTANCE":
            dst = task.dst_loc if task.dst_loc is not None else task.loc
            distance = float(np.linalg.norm(task.loc - dst))
            speed = float(max(self.type_speeds[task.task_type], 1e-6))
            physical_time = distance / speed
        else:
            demand = float(task.req[task.task_type])
            workload_ref = max((self.config.task_workload_range[0] + self.config.task_workload_range[1]) * 0.5, 1e-6)
            workload_factor = float(task.workload) / workload_ref
            team_cap = float(sum(self.robots[rid].cap[task.task_type] for rid in team))
            scale = float(task.type_time_scale)
            physical_time = demand * workload_factor / max(team_cap, 1e-6) * scale
            t_min, t_max = self.non_distance_time_bounds
            physical_time = float(np.clip(physical_time, t_min, t_max))

        physical_time = max(physical_time, self.config.dt_seconds)
        task.remain_phys = physical_time
        task.duration_steps = int(math.ceil(physical_time / self.config.dt_seconds))
        task.duration_steps = max(1, task.duration_steps)

    def _team_capability_ok(self, task: Task, team: List[int]) -> bool:
        """检查团队能力是否逐维满足任务需求阈值。输出 bool。"""
        runtime_cache = self._obs_runtime_cache
        if runtime_cache is not None:
            team_capability_ok = runtime_cache.get("team_capability_ok")
            if isinstance(team_capability_ok, dict):
                key = tuple(sorted(int(x) for x in team))
                cache_key = (int(task.task_id), key)
                cached = team_capability_ok.get(cache_key)
                if cached is not None:
                    return bool(cached)
        if not team:
            return False
        team_cap = 0.0
        for rid in team:
            robot = self.robots[rid]
            if robot.agent_type != task.task_type:
                if runtime_cache is not None:
                    runtime_cache.setdefault("team_capability_ok", {})[(int(task.task_id), tuple(sorted(int(x) for x in team)))] = False
                return False
            team_cap += float(robot.cap[task.task_type])
        result = bool(team_cap >= float(task.req[task.task_type]))
        if runtime_cache is not None:
            runtime_cache.setdefault("team_capability_ok", {})[(int(task.task_id), tuple(sorted(int(x) for x in team)))] = result
        return result

    def _effective_task_duration_steps(self, task: Task) -> int:
        """返回用于时序判定的任务持续步数；未激活任务使用规划估计时长。"""
        if int(getattr(task, "duration_steps", 0)) > 0:
            return max(1, int(task.duration_steps))
        return max(1, int(self._estimate_task_duration_steps(task)))

    def _predicted_start_time(self, task: Task) -> int:
        """返回任务开始约束可使用的已知或规划开始时刻。"""
        if int(task.t_start) >= 0:
            return int(task.t_start)
        planned = self.gen_planned_start_steps.get(int(task.task_id))
        if planned is not None:
            return int(planned)
        return -1

    def _predicted_finish_time(self, task: Task) -> int:
        """返回任务完成约束可使用的已知或由开始时刻推断的完成时刻。"""
        if int(task.t_finish) >= 0:
            return int(task.t_finish)
        if int(task.t_finish_phys) >= 0:
            return int(task.t_finish_phys)
        if int(task.t_start) >= 0:
            return int(task.t_start) + self._effective_task_duration_steps(task)
        planned = self.gen_planned_finish_steps.get(int(task.task_id))
        if planned is not None:
            return int(planned)
        return -1

    def _start_constraints_satisfied(self, task: Task) -> bool:
        """检查任务开始约束（统一处理 FS/SS/FF/SF+窗口）是否满足。输出 bool。"""
        runtime_cache = self._obs_runtime_cache
        if runtime_cache is not None:
            start_constraints_cache = runtime_cache.get("start_constraints")
            if isinstance(start_constraints_cache, dict) and int(task.task_id) in start_constraints_cache:
                return bool(start_constraints_cache[int(task.task_id)])
            preds = self._incoming_edges_map().get(int(task.task_id), [])
        else:
            preds = self._incoming_edges_map().get(int(task.task_id), [])
        succ_duration = self._effective_task_duration_steps(task)
        for e in preds:
            u = self.tasks[e.u]
            if self._task_has_hard_violation(u):
                self._mark_window_miss(task, e)
                if runtime_cache is not None:
                    runtime_cache.setdefault("start_constraints", {})[int(task.task_id)] = False
                return False
            if e.rel_type == "FS":
                ref_time = self._constraint_reference_time(u, e.rel_type)
                if ref_time < 0:
                    if runtime_cache is not None:
                        runtime_cache.setdefault("start_constraints", {})[int(task.task_id)] = False
                    return False
                ready_t = ref_time + e.lag
                window_t = ready_t + self.rel_window_steps["FS"]
                status = self._window_status(ref_time, ready_t, window_t)
                if status in ["blocked", "early"]:
                    if runtime_cache is not None:
                        runtime_cache.setdefault("start_constraints", {})[int(task.task_id)] = False
                    return False
                if status == "missed":
                    self._mark_window_miss(task, e)
                    if runtime_cache is not None:
                        runtime_cache.setdefault("start_constraints", {})[int(task.task_id)] = False
                    return False
            elif e.rel_type == "SS":
                ref_time = self._constraint_reference_time(u, e.rel_type)
                if ref_time < 0:
                    if runtime_cache is not None:
                        runtime_cache.setdefault("start_constraints", {})[int(task.task_id)] = False
                    return False
                ready_t = ref_time + e.lag
                window_t = ready_t + self.rel_window_steps["SS"]
                status = self._window_status(ref_time, ready_t, window_t)
                if status in ["blocked", "early"]:
                    if runtime_cache is not None:
                        runtime_cache.setdefault("start_constraints", {})[int(task.task_id)] = False
                    return False
                if status == "missed":
                    self._mark_window_miss(task, e)
                    if runtime_cache is not None:
                        runtime_cache.setdefault("start_constraints", {})[int(task.task_id)] = False
                    return False
            elif e.rel_type == "FF":
                pred_finish = self._predicted_finish_time(u)
                if pred_finish < 0:
                    if runtime_cache is not None:
                        runtime_cache.setdefault("start_constraints", {})[int(task.task_id)] = False
                    return False
                ready_t = pred_finish + e.lag - succ_duration
                window_t = ready_t + self.rel_window_steps["FF"]
                status = self._window_status(pred_finish, ready_t, window_t)
                if status in ["blocked", "early"]:
                    if runtime_cache is not None:
                        runtime_cache.setdefault("start_constraints", {})[int(task.task_id)] = False
                    return False
                if status == "missed":
                    self._mark_window_miss(task, e)
                    if runtime_cache is not None:
                        runtime_cache.setdefault("start_constraints", {})[int(task.task_id)] = False
                    return False
            elif e.rel_type == "SF":
                pred_start = self._predicted_start_time(u)
                if pred_start < 0:
                    if runtime_cache is not None:
                        runtime_cache.setdefault("start_constraints", {})[int(task.task_id)] = False
                    return False
                ready_t = pred_start + e.lag - succ_duration
                window_t = ready_t + self.rel_window_steps["SF"]
                status = self._window_status(pred_start, ready_t, window_t)
                if status in ["blocked", "early"]:
                    if runtime_cache is not None:
                        runtime_cache.setdefault("start_constraints", {})[int(task.task_id)] = False
                    return False
                if status == "missed":
                    self._mark_window_miss(task, e)
                    if runtime_cache is not None:
                        runtime_cache.setdefault("start_constraints", {})[int(task.task_id)] = False
                    return False
        if runtime_cache is not None:
            runtime_cache.setdefault("start_constraints", {})[int(task.task_id)] = True
        return True

    def _finish_constraints_satisfied(self, task: Task) -> bool:
        """检查任务结束约束（FF/SF+窗口）是否满足。输出 bool。"""
        preds = [e for e in self._incoming_edges_map().get(int(task.task_id), []) if e.rel_type in ["FF", "SF"]]
        for e in preds:
            u = self.tasks[e.u]
            if self._task_has_hard_violation(u):
                self._mark_window_miss(task, e)
                return False
            if e.rel_type == "FF":
                ref_time = self._constraint_reference_time(u, e.rel_type)
                if ref_time < 0:
                    return False
                ready_t = ref_time + e.lag
                window_t = ready_t + self.rel_window_steps["FF"]
                status = self._window_status(ref_time, ready_t, window_t)
                if status in ["blocked", "early"]:
                    return False
                if status == "missed":
                    self._mark_window_miss(task, e)
                    return False
            elif e.rel_type == "SF":
                ref_time = self._constraint_reference_time(u, e.rel_type)
                if ref_time < 0:
                    return False
                ready_t = ref_time + e.lag
                window_t = ready_t + self.rel_window_steps["SF"]
                status = self._window_status(ref_time, ready_t, window_t)
                if status in ["blocked", "early"]:
                    return False
                if status == "missed":
                    self._mark_window_miss(task, e)
                    return False
        return True

    def _edge_key(self, e: DependencyEdge) -> str:
        return f"{e.u}->{e.v}:{e.rel_type}"

    def _mark_window_miss(self, task: Task, e: DependencyEdge):
        """记录窗口违规：适用于 FS/SS/FF/SF 全部关系。"""
        key = self._edge_key(e)
        if key in task.window_missed_edges:
            return
        task.window_missed_edges.add(key)
        self.missed_ss_window += 1
        task.ss_window_missed = True
        task.has_dependency_violation = True
        if e.rel_type in ["FS", "SS"]:
            task.has_start_violation = True
        else:
            task.has_finish_violation = True

    def _task_is_recognized_done(self, task: Task, truncated: bool) -> bool:
        if task.status != "DONE":
            return False
        if self._task_has_hard_violation(task):
            return False
        return True

    def _invalidate_task_team(self, task: Task):
        """Release a team whose task has become permanently non-compliant."""
        for rid in list(task.assigned_robots):
            robot = self.robots[rid]
            robot.state = "IDLE"
            robot.task_id = None
            robot.eta = 0
            robot.work_left = 0
            robot.waiting_since = -1
            robot.moving_since = -1
            robot.last_progress_time = int(self.current_time)
        task.assigned_robots.clear()
        if task.status != "DONE":
            task.status = "AVAILABLE"

    def _task_status_counts(self) -> Dict[str, int]:
        counts: Counter[str] = Counter()
        for task in self.tasks.values():
            counts[str(task.status)] += 1
        return {k: int(v) for k, v in counts.items()}

    def _logical_wait_reason_summary(self) -> Dict[str, int]:
        counts: Counter[str] = Counter()
        for task in self.tasks.values():
            if task.status != "LOGICAL_WAIT":
                continue
            reason = str(task.last_logical_wait_reason or self._logical_wait_reason(task) or "unknown")
            counts[reason] += 1
        return {k: int(v) for k, v in counts.items()}

    def _logical_wait_reason_bucket(self, reason: str) -> str:
        reason = str(reason or "unknown")
        if reason.startswith("wait_pred_finish:"):
            return "pred_finish_pending"
        if reason.startswith("wait_pred_start:"):
            return "pred_start_pending"
        if reason.startswith("wait_fs_lag:"):
            return "fs_lag_pending"
        if reason.startswith("wait_ss_lag:"):
            return "ss_lag_pending"
        if reason.startswith("wait_ff_lag:"):
            return "ff_lag_pending"
        if reason.startswith("wait_sf_lag:"):
            return "sf_lag_pending"
        if reason == "wait_teammates":
            return "team_formation_pending"
        if reason == "wait_start_constraint":
            return "start_constraint_pending"
        if reason == "wait_finish_constraint":
            return "finish_constraint_pending"
        return "other"

    def _logical_wait_bucket_summary(self) -> Dict[str, int]:
        counts: Counter[str] = Counter()
        for task in self.tasks.values():
            if task.status != "LOGICAL_WAIT":
                continue
            reason = str(task.last_logical_wait_reason or self._logical_wait_reason(task) or "unknown")
            counts[self._logical_wait_reason_bucket(reason)] += 1
        return {k: int(v) for k, v in counts.items()}

    def _count_temporal_ready_edges(self) -> int:
        runtime_cache = self._obs_runtime_cache
        if runtime_cache is not None:
            cached = runtime_cache.get("temporal_ready_edges")
            if cached is not None:
                return int(cached)
        cnt = 0
        for e in self.edges:
            u = self.tasks.get(e.u)
            if u is None:
                continue
            if e.rel_type in ["FS", "FF"]:
                if u.t_finish >= 0:
                    cnt += 1
            else:
                if u.t_start >= 0:
                    cnt += 1
        if runtime_cache is not None:
            runtime_cache["temporal_ready_edges"] = int(cnt)
        return cnt

    def _progress_robots_and_tasks(self):
        """推进机器人运动/执行与任务状态机（ASSIGNED->EXECUTING->PHYS_DONE/LOGICAL_WAIT->DONE）。输出 None。"""
        self._maybe_release_stale_commitments()
        for task in self.tasks.values():
            if (
                task.status == "ASSIGNED"
                and len(task.assigned_robots) == task.k
                and self._team_capability_ok(task, task.assigned_robots)
                and self._start_constraints_satisfied(task)
            ):
                self._activate_task(task)

        executing_tasks = set()
        for rid, robot in self.robots.items():
            if robot.task_id is None:
                continue
            task = self.tasks[robot.task_id]
            if robot.state == "MOVING":
                robot.eta = max(0, robot.eta - 1)
                if robot.eta == 0:
                    robot.pos = task.loc.copy()
                    robot.last_progress_time = int(self.current_time)
                    robot.moving_since = -1
                    if task.status in ["READY", "EXECUTING"]:
                        robot.state = "EXECUTING"
                        robot.waiting_since = -1
                    else:
                        robot.state = "WAITING"
                        robot.waiting_since = int(self.current_time)
                        task.last_logical_wait_reason = self._logical_wait_reason(task)
            if robot.state in ["WAITING", "LOGICAL_WAIT"] and robot.task_id is not None:
                task = self.tasks[robot.task_id]
                if task.status in ["READY", "EXECUTING"]:
                    robot.state = "EXECUTING"
                    robot.waiting_since = -1
                    robot.last_progress_time = int(self.current_time)
                else:
                    robot.state = "WAITING" if task.status in ["AVAILABLE", "ASSIGNED"] else "LOGICAL_WAIT"
                    if robot.state == "WAITING" and robot.waiting_since < 0:
                        robot.waiting_since = int(self.current_time)
                    task.last_logical_wait_reason = self._logical_wait_reason(task)
            elif robot.state == "LOGICAL_WAIT" and robot.task_id is not None:
                task = self.tasks[robot.task_id]
                if task.status == "DONE":
                    robot.state = "IDLE"
                    robot.task_id = None
                    robot.eta = 0
                    robot.work_left = 0
                    robot.waiting_since = -1
                    robot.moving_since = -1
            elif robot.state == "EXECUTING":
                if task.status == "EXECUTING":
                    executing_tasks.add(task.task_id)

            if robot.task_id is not None:
                task = self.tasks[robot.task_id]
                if task.t_start < 0 and all(self.robots[r].state == "EXECUTING" for r in task.assigned_robots):
                    # A pre-gathered team may arrive after its dependency window closes.
                    # Recheck at the physical start, rather than only at activation time.
                    if self._start_constraints_satisfied(task):
                        task.t_start = self.current_time
                        task.status = "EXECUTING"
                    elif self._task_has_hard_violation(task):
                        self._invalidate_task_team(task)

        for tid in executing_tasks:
            task = self.tasks[tid]
            task.remain_phys = max(0.0, task.remain_phys - self.config.dt_seconds)
            remain_steps = int(math.ceil(task.remain_phys / max(self.config.dt_seconds, 1e-6)))
            remain_steps = max(0, remain_steps)
            for rid in task.assigned_robots:
                robot = self.robots[rid]
                if robot.state == "EXECUTING":
                    robot.work_left = remain_steps

        for task in self.tasks.values():
            if task.status == "EXECUTING":
                if task.remain_phys <= 1e-6:
                    task.status = "PHYS_DONE"
                    if task.t_finish_phys < 0:
                        task.t_finish_phys = self.current_time
            if task.status == "PHYS_DONE":
                if self._finish_constraints_satisfied(task):
                    task.status = "DONE"
                    task.t_finish = self.current_time
                    task.last_logical_wait_reason = ""
                    for rid in list(task.assigned_robots):
                        r = self.robots[rid]
                        r.state = "IDLE"
                        r.task_id = None
                        r.eta = 0
                        r.work_left = 0
                        r.waiting_since = -1
                        r.moving_since = -1
                        r.last_progress_time = int(self.current_time)
                    task.assigned_robots.clear()
                else:
                    task.status = "LOGICAL_WAIT"
                    task.logical_wait_steps += 1
                    task.last_logical_wait_reason = self._logical_wait_reason(task)
                    for rid in list(task.assigned_robots):
                        r = self.robots[rid]
                        r.state = "LOGICAL_WAIT"
                        r.task_id = task.task_id
                        r.eta = 0
                        r.work_left = 0
            elif task.status == "LOGICAL_WAIT":
                task.logical_wait_steps += 1
                task.last_logical_wait_reason = self._logical_wait_reason(task)
                if self._finish_constraints_satisfied(task):
                    task.status = "DONE"
                    task.t_finish = self.current_time
                    task.last_logical_wait_reason = ""
                    for rid in list(task.assigned_robots):
                        r = self.robots[rid]
                        r.state = "IDLE"
                        r.task_id = None
                        r.eta = 0
                        r.work_left = 0
                        r.waiting_since = -1
                        r.moving_since = -1
                        r.last_progress_time = int(self.current_time)
                    task.assigned_robots.clear()

        if self.current_time + 1 >= self.config.max_steps:
            for task in self.tasks.values():
                if task.status in ["PHYS_DONE", "LOGICAL_WAIT", "DONE"] and task.t_finish < 0:
                    task.has_pseudo_completion = True

        self.current_time += 1

    def _refresh_task_statuses(self):
        """根据 release 与分配成员数量同步任务状态标签。输出 None。"""
        for t in self.tasks.values():
            if t.status == "UNRELEASED" and self.current_time >= t.release_time:
                t.status = "AVAILABLE"
            if t.status in ["AVAILABLE", "ASSIGNED"]:
                t.status = "ASSIGNED" if len(t.assigned_robots) > 0 else "AVAILABLE"

    def _all_done(self) -> bool:
        """判断是否所有任务均 DONE。输出 bool。"""
        if not self.tasks:
            return False
        return all(t.status == "DONE" for t in self.tasks.values())

    def _count_waiting_robots(self) -> int:
        """统计普通等待机器人数量，不包含逻辑约束等待。"""
        waiting = 0
        for r in self.robots.values():
            if r.state == "WAITING":
                waiting += 1
        return waiting

    def _count_logical_wait_robots(self) -> int:
        """统计逻辑等待机器人数量（开始前或完成后因约束未满足而等待）。"""
        waiting = 0
        for r in self.robots.values():
            if r.state == "LOGICAL_WAIT":
                waiting += 1
        return waiting

    def _logical_wait_reason(self, task: Task) -> str:
        if task.t_start < 0:
            preds = [e for e in self.edges if e.v == task.task_id]
            for e in preds:
                u = self.tasks[e.u]
                if e.rel_type == "FS" and u.t_finish < 0:
                    return f"wait_pred_finish:{e.u}"
                if e.rel_type == "FS" and self.current_time < u.t_finish + e.lag:
                    return f"wait_fs_lag:{e.u}"
                if e.rel_type == "SS" and u.t_start < 0:
                    return f"wait_pred_start:{e.u}"
                if e.rel_type == "SS" and self.current_time < u.t_start + e.lag:
                    return f"wait_ss_lag:{e.u}"
                if e.rel_type == "FF" and u.t_finish < 0:
                    return f"wait_pred_finish:{e.u}"
                if e.rel_type == "SF" and u.t_start < 0:
                    return f"wait_pred_start:{e.u}"
            if len(task.assigned_robots) < task.k:
                return "wait_teammates"
            return "wait_start_constraint"
        preds = [e for e in self.edges if e.v == task.task_id and e.rel_type in ["FF", "SF"]]
        for e in preds:
            u = self.tasks[e.u]
            if e.rel_type == "FF" and u.t_finish < 0:
                return f"wait_pred_finish:{e.u}"
            if e.rel_type == "FF" and self.current_time < u.t_finish + e.lag:
                return f"wait_ff_lag:{e.u}"
            if e.rel_type == "SF" and u.t_start < 0:
                return f"wait_pred_start:{e.u}"
            if e.rel_type == "SF" and self.current_time < u.t_start + e.lag:
                return f"wait_sf_lag:{e.u}"
        return "wait_finish_constraint"

    def _makespan(self) -> int:
        """返回当前已完成任务的最大完成时刻。输出 int。"""
        finishes = [t.t_finish for t in self.tasks.values() if t.t_finish >= 0]
        return max(finishes) if finishes else 0

    def _build_node_features(self) -> np.ndarray:
        """构建任务节点特征矩阵。输出 shape=(max_tasks, cap_dim+base_dim+8+task_type_dim)。

        base_dim 在完整模式下为 21（含依赖派生量），在稀疏模式（方案 C 信息瓶颈）下为 7
        （仅保留非依赖派生的物理/进度量），此时依赖推理只能经 RGAT 沿边获得。"""
        max_t = self.config.max_tasks
        sparse = bool(getattr(self.config, "sparse_dependency_features", False))
        # 结构：2(位置) + cap_dim(需求) + base_dim + 8(status) + task_type_dim(type onehot)
        base_dim = 7 if sparse else 21
        feat_dim = self.config.cap_dim + base_dim + 10 + self.task_type_dim
        x = np.zeros((max_t, feat_dim), dtype=np.float32)

        for tid, t in self.tasks.items():
            if tid >= max_t:
                continue
            status_onehot = np.zeros(8, dtype=np.float32)
            status_map = {
                "UNRELEASED": 0,
                "AVAILABLE": 1,
                "ASSIGNED": 2,
                "READY": 3,
                "EXECUTING": 4,
                "PHYS_DONE": 5,
                "LOGICAL_WAIT": 6,
                "DONE": 7,
            }
            status_onehot[status_map[t.status]] = 1.0
            type_onehot = np.zeros(self.task_type_dim, dtype=np.float32)
            if 0 <= t.task_type < self.task_type_dim:
                type_onehot[t.task_type] = 1.0
            # 非依赖派生的物理/进度量（两种模式都保留）
            base = [
                t.workload / max(self.config.task_workload_range[1], 1e-6),
                t.k / max(self.config.num_agents, 1),
                len(t.assigned_robots) / max(t.k, 1),
                float(self.current_time - t.release_time) / max(self.config.max_steps, 1),
                float(t.t_start >= 0),
                float(t.t_finish_phys >= 0),
                float(t.t_finish >= 0),
            ]
            if not sparse:
                # 依赖派生量：仅完整模式提供。稀疏模式下这些只能由 RGAT 沿边传播获得。
                base += [
                    float(t.logical_wait_steps) / max(self.config.max_steps, 1),
                    float(t.earliest_start) / max(self.config.max_steps, 1),
                    float(t.earliest_finish) / max(self.config.max_steps, 1),
                    float(t.latest_start) / max(self.config.max_steps, 1),
                    float(t.latest_finish) / max(self.config.max_steps, 1),
                    float(np.clip(t.slack, -self.config.max_steps, self.config.max_steps)) / max(self.config.max_steps, 1),
                    float(t.is_temporally_feasible),
                    float(t.num_unsatisfied_predecessors) / max(len(self.edges), 1),
                    float(t.num_ready_predecessors) / max(len(self.edges), 1),
                    float(np.clip(t.predecessor_ready_ratio, 0.0, 1.0)),
                    float(np.clip(t.window_urgency, 0.0, 1.0)),
                    float(t.start_feasible_now),
                    float(t.finish_feasible_now),
                    float(np.tanh(t.criticality)),
                ]
            x[tid] = np.concatenate([
                np.array([
                    t.loc[0] / self.config.area_size[0],
                    t.loc[1] / self.config.area_size[1],
                ], dtype=np.float32),
                t.req.astype(np.float32),
                np.array(base, dtype=np.float32),
                status_onehot,
                type_onehot,
            ])
        return x

    def _build_graph_edges(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """构建图边张量。输出 (edge_index[2,E], edge_type[E], edge_lag[E], edge_mask[E])。"""
        edge_count = len(self.edges)
        edge_index = np.zeros((2, edge_count), dtype=np.int64)
        edge_type = np.zeros((edge_count,), dtype=np.int64)
        edge_lag = np.zeros((edge_count,), dtype=np.float32)
        edge_mask = np.ones((edge_count,), dtype=np.float32)
        lag_den = max(abs(int(self.config.lag_range[0])), abs(int(self.config.lag_range[1])), 1)
        for i, e in enumerate(self.edges):
            edge_index[0, i] = e.u
            edge_index[1, i] = e.v
            edge_type[i] = REL_TYPES[e.rel_type]
            edge_lag[i] = float(e.lag) / float(lag_den)
        return edge_index, edge_type, edge_lag, edge_mask

    def _build_agent_features(self) -> np.ndarray:
        """构建机器人特征矩阵。输出 shape=(num_agents, 5+cap_dim+5)。"""
        feats = []
        for rid in sorted(self.robots.keys()):
            r = self.robots[rid]
            state_onehot = np.zeros(5, dtype=np.float32)
            state_map = {"IDLE": 0, "MOVING": 1, "WAITING": 2, "EXECUTING": 3, "LOGICAL_WAIT": 4}
            state_onehot[state_map[r.state]] = 1.0
            feat = np.concatenate([
                np.array([
                    r.pos[0] / self.config.area_size[0],
                    r.pos[1] / self.config.area_size[1],
                    r.v_max / max(self.config.v_max_range[1], 1e-6),
                    r.eta / max(self.config.max_steps, 1),
                    r.work_left / max(self.config.max_steps, 1),
                ], dtype=np.float32),
                r.cap.astype(np.float32),
                state_onehot,
            ])
            feats.append(feat)
        return np.stack(feats, axis=0)

    def _build_pair_features(self) -> np.ndarray:
        """构建机器人-任务配对特征。

        完整模式输出 13 维（含依赖派生的时间窗/可行性特征）；稀疏模式（方案 C）输出 6 维，
        仅保留空间-能力-组队类特征，依赖时间可行性只能经 RGAT 传播到任务嵌入后间接获得。"""
        sparse = bool(getattr(self.config, "sparse_dependency_features", False))
        pair_dim = 6 if sparse else 13
        out = np.zeros((self.config.num_agents, self.config.max_tasks, pair_dim), dtype=np.float32)
        runtime_cache = self._obs_runtime_cache
        can_commit_cache = runtime_cache.get("can_commit") if isinstance(runtime_cache, dict) else None
        can_commit_matrix = runtime_cache.get("can_commit_matrix") if isinstance(runtime_cache, dict) else None
        task_window_urgency = runtime_cache.get("task_window_urgency") if isinstance(runtime_cache, dict) else None
        task_req_sum = runtime_cache.get("task_req_sum") if isinstance(runtime_cache, dict) else None
        task_req_norm = runtime_cache.get("task_req_norm") if isinstance(runtime_cache, dict) else None
        assigned_team_cap = runtime_cache.get("assigned_team_cap") if isinstance(runtime_cache, dict) else None
        area_diag = max(self._area_diag, 1e-6)
        for rid, r in self.robots.items():
            r_cap_norm = float(np.linalg.norm(r.cap))
            for tid, t in self.tasks.items():
                if tid >= self.config.max_tasks:
                    continue
                dist = float(np.linalg.norm(r.pos - t.loc))
                eta = dist / max(r.v_max, 1e-6)
                remain_slots = max(0, t.k - len(t.assigned_robots))
                if isinstance(assigned_team_cap, dict):
                    base_team_cap = assigned_team_cap.get(int(tid))
                else:
                    base_team_cap = None
                if base_team_cap is None:
                    if t.assigned_robots:
                        base_team_cap = np.sum([self.robots[x].cap for x in t.assigned_robots], axis=0)
                    else:
                        base_team_cap = np.zeros_like(t.req, dtype=np.float64)
                team_cap = base_team_cap + r.cap
                gap = np.maximum(t.req - team_cap, 0.0)
                if isinstance(task_req_sum, dict):
                    task_req_total = float(task_req_sum.get(int(tid), 0.0))
                else:
                    task_req_total = float(np.sum(t.req))
                if isinstance(task_req_norm, dict):
                    t_req_norm = float(task_req_norm.get(int(tid), 0.0))
                else:
                    t_req_norm = float(np.linalg.norm(t.req))
                est_arrive = self.current_time + int(math.ceil(eta / max(self.config.dt_seconds, 1e-6)))
                est_start_if_join = max(est_arrive, int(math.ceil(t.earliest_start)))
                remain_after_join = max(0, t.k - len(t.assigned_robots) - (0 if rid in t.assigned_robots else 1))
                start_slack = (t.latest_start - self.current_time) / max(self.config.max_steps, 1)
                est_finish_if_join = est_start_if_join + max(t.duration_steps, 1)
                finish_slack = (t.latest_finish - est_finish_if_join) / max(self.config.max_steps, 1)
                if isinstance(can_commit_matrix, np.ndarray) and 0 <= int(rid) < can_commit_matrix.shape[0] and 0 <= int(tid) < can_commit_matrix.shape[1]:
                    can_commit_now = float(can_commit_matrix[int(rid), int(tid)])
                elif isinstance(can_commit_cache, dict) and (int(rid), int(tid)) in can_commit_cache:
                    can_commit_now = 1.0 if bool(can_commit_cache[(int(rid), int(tid))]) else 0.0
                else:
                    can_commit_now = 1.0 if self._can_commit(rid, tid, dry_run=True) else 0.0
                start_gap = (t.earliest_start - est_arrive) / max(self.config.max_steps, 1)
                feasible_margin = min(start_slack, finish_slack)
                if isinstance(task_window_urgency, dict):
                    window_urgency = float(task_window_urgency.get(int(tid), 0.0))
                else:
                    window_urgency = float(t.window_urgency)
                if sparse:
                    # 稀疏模式：仅空间-能力-组队特征（5 个基础 + 保留 can_commit 硬可行性）。
                    out[rid, tid] = np.array([
                        dist / area_diag,
                        eta / max(self.config.max_steps, 1),
                        remain_slots / max(t.k, 1),
                        float(np.dot(r.cap, t.req)) / (r_cap_norm * t_req_norm + 1e-6),
                        float(np.sum(gap)) / (task_req_total + 1e-6),
                        float(can_commit_now),
                    ], dtype=np.float32)
                else:
                    out[rid, tid] = np.array([
                        dist / area_diag,
                        eta / max(self.config.max_steps, 1),
                        remain_slots / max(t.k, 1),
                        float(np.dot(r.cap, t.req)) / (r_cap_norm * t_req_norm + 1e-6),
                        float(np.sum(gap)) / (task_req_total + 1e-6),
                        float(np.clip(start_slack, -1.0, 1.0)),
                        float(np.clip(finish_slack, -1.0, 1.0)),
                        float(np.clip(window_urgency, 0.0, 1.0)),
                        float(est_start_if_join) / max(self.config.max_steps, 1),
                        remain_after_join / max(t.k, 1),
                        float(np.clip(start_gap, -1.0, 1.0)),
                        float(np.clip(feasible_margin, -1.0, 1.0)),
                        float(can_commit_now),
                    ], dtype=np.float32)
        return out

    def _update_task_temporal_stats(self):
        incoming_map = self._incoming_edges_map()
        succ_map = self._outgoing_edges_map()
        temporal_ready_edges = 0
        for t in self.tasks.values():
            preds = incoming_map.get(int(t.task_id), [])
            succs = succ_map.get(int(t.task_id), [])

            unsatisfied = 0
            ready_preds = 0
            earliest_start = float(self.current_time)
            for e in preds:
                u = self.tasks[e.u]
                if e.rel_type == "FS":
                    if u.t_finish < 0:
                        unsatisfied += 1
                        continue
                    temporal_ready_edges += 1
                    ready_preds += 1
                    earliest_start = max(earliest_start, float(u.t_finish + e.lag))
                elif e.rel_type == "SS":
                    if u.t_start < 0:
                        unsatisfied += 1
                        continue
                    temporal_ready_edges += 1
                    ready_preds += 1
                    earliest_start = max(earliest_start, float(u.t_start + e.lag))
                elif e.rel_type in ["FF", "SF"]:
                    if e.rel_type == "FF":
                        if u.t_finish < 0:
                            unsatisfied += 1
                        else:
                            temporal_ready_edges += 1
                            ready_preds += 1
                    if e.rel_type == "SF":
                        if u.t_start < 0:
                            unsatisfied += 1
                        else:
                            temporal_ready_edges += 1
                            ready_preds += 1

            duration = float(max(t.duration_steps, 1))
            earliest_finish = earliest_start + duration

            latest_finish = float(self.config.max_steps)
            for e in succs:
                v = self.tasks[e.v]
                if e.rel_type == "FS" and v.t_start >= 0:
                    latest_finish = min(latest_finish, float(v.t_start - e.lag))
                elif e.rel_type == "FF" and v.t_finish >= 0:
                    latest_finish = min(latest_finish, float(v.t_finish - e.lag))

            latest_start = latest_finish - duration
            slack = latest_start - float(self.current_time)
            feasible = earliest_start <= latest_start + 1e-6
            downstream = len(succs)
            criticality = float(downstream) + 1.0 / max(slack + 1.0, 1e-6)
            pred_total = max(len(preds), 1)
            ready_ratio = float(ready_preds) / float(pred_total)
            start_feasible_now = bool(feasible and earliest_start <= float(self.current_time) + 1e-6)
            finish_feasible_now = bool(latest_finish >= float(self.current_time) + duration - 1e-6)
            window_urgency = 0.0
            for e in preds:
                if e.rel_type not in ["FS", "SS"]:
                    continue
                u = self.tasks[e.u]
                base_t = u.t_finish if e.rel_type == "FS" else u.t_start
                if base_t < 0:
                    continue
                lower = base_t + e.lag
                upper = lower + self.rel_window_steps[e.rel_type]
                rem = upper - self.current_time
                rel_urgency = 1.0 - np.clip(rem / max(self.rel_window_steps[e.rel_type], 1), 0.0, 1.0)
                window_urgency = max(window_urgency, float(rel_urgency))

            t.earliest_start = earliest_start
            t.earliest_finish = earliest_finish
            t.latest_start = latest_start
            t.latest_finish = latest_finish
            t.slack = slack
            t.is_temporally_feasible = bool(feasible)
            t.num_unsatisfied_predecessors = int(unsatisfied)
            t.num_ready_predecessors = int(ready_preds)
            t.predecessor_ready_ratio = float(np.clip(ready_ratio, 0.0, 1.0))
            t.window_urgency = float(np.clip(window_urgency, 0.0, 1.0))
            t.start_feasible_now = bool(start_feasible_now)
            t.finish_feasible_now = bool(finish_feasible_now)
            t.criticality = criticality
        if isinstance(self._obs_runtime_cache, dict):
            self._obs_runtime_cache["temporal_ready_edges"] = int(temporal_ready_edges)

    def _count_ready_tasks(self) -> int:
        return sum(
            1
            for t in self.tasks.values()
            if t.status in ["AVAILABLE", "ASSIGNED", "READY"] and t.start_feasible_now and t.is_temporally_feasible
        )

    def _build_task_mask(self) -> np.ndarray:
        """构建任务存在掩码（用于屏蔽未生成任务位）。输出 shape=(max_tasks,) 的 0/1 向量。"""
        mask = np.zeros((self.config.max_tasks,), dtype=np.float32)
        for tid in self.tasks.keys():
            if tid < self.config.max_tasks:
                mask[tid] = 1.0
        return mask

    def _get_obs(self):
        """汇总环境观测。

        输出字段：
        - task_graph: node_features/edge_index/edge_type/edge_lag/node_mask
        - agent_feats
        - pair_feats
        - action_mask
        - global_time
        """
        self._obs_runtime_cache = {"can_commit": {}}
        obs_t0 = time.perf_counter()
        t0 = time.perf_counter()
        self._update_task_temporal_stats()
        self._perf_stats["obs_temporal_sec"] += time.perf_counter() - t0
        if self._obs_runtime_cache is not None:
            self._obs_runtime_cache["task_window_urgency"] = {
                int(tid): float(task.window_urgency) for tid, task in self.tasks.items()
            }
            self._obs_runtime_cache["task_req_sum"] = {
                int(tid): float(np.sum(task.req)) for tid, task in self.tasks.items()
            }
            self._obs_runtime_cache["task_req_norm"] = {
                int(tid): float(np.linalg.norm(task.req)) for tid, task in self.tasks.items()
            }
            self._obs_runtime_cache["assigned_team_cap"] = {
                int(tid): (
                    np.sum([self.robots[rid].cap for rid in task.assigned_robots], axis=0)
                    if task.assigned_robots else np.zeros_like(task.req, dtype=np.float64)
                )
                for tid, task in self.tasks.items()
            }
        t0 = time.perf_counter()
        node_features = self._build_node_features()
        node_mask = self._build_task_mask()
        self._perf_stats["obs_node_sec"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        edge_index, edge_type, edge_lag, edge_mask = self._build_graph_edges()
        self._perf_stats["obs_edge_sec"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        agent_feats = self._build_agent_features()
        self._perf_stats["obs_agent_sec"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        action_mask, can_commit_matrix, commit_score_matrix = self._build_action_mask_and_commit_scores()
        self._perf_stats["obs_action_mask_sec"] += time.perf_counter() - t0
        self._obs_runtime_cache["action_mask"] = action_mask
        self._obs_runtime_cache["can_commit_matrix"] = can_commit_matrix
        self._obs_runtime_cache["commit_score_matrix"] = commit_score_matrix
        t0 = time.perf_counter()
        pair_feats = self._build_pair_features()
        self._perf_stats["obs_pair_sec"] += time.perf_counter() - t0
        obs = {
            "task_graph": {
                "node_features": node_features,
                "edge_index": edge_index,
                "edge_type": edge_type,
                "edge_lag": edge_lag,
                "edge_mask": edge_mask,
                "node_mask": node_mask,
            },
            "agent_feats": agent_feats,
            "pair_feats": pair_feats,
            "action_mask": action_mask,
            "global_time": float(self.current_time),
        }
        # 方案 C：供 RGAT 沿边时间传播使用的环境真值开始/物理完成时间（归一化，未发生为 -1）。
        # 与依赖派生特征不同，这两个原始时间戳不直接给出松弛/可行性，须由 RGAT 沿边聚合
        # （源时间 + lag 的 max-plus 传播）才能转化为对目标节点的时间下界，因此不破坏信息瓶颈。
        if bool(getattr(self.config, "rgat_temporal_propagation", False)):
            tstart = np.full((self.config.max_tasks,), -1.0, dtype=np.float32)
            tfin = np.full((self.config.max_tasks,), -1.0, dtype=np.float32)
            for tid, t in self.tasks.items():
                if tid >= self.config.max_tasks:
                    continue
                if t.t_start >= 0:
                    tstart[tid] = float(t.t_start) / max(self.config.max_steps, 1)
                if t.t_finish_phys >= 0:
                    tfin[tid] = float(t.t_finish_phys) / max(self.config.max_steps, 1)
            obs["task_graph"]["node_t_start"] = tstart
            obs["task_graph"]["node_t_finish_phys"] = tfin
        self._perf_stats["get_obs_calls"] += 1.0
        self._perf_stats["obs_total_sec"] += time.perf_counter() - obs_t0
        self._obs_runtime_cache = None
        return obs

    def get_perf_stats(self) -> Dict[str, float]:
        out = {str(k): float(v) for k, v in self._perf_stats.items()}
        step_calls = max(1.0, out.get("step_calls", 0.0))
        obs_calls = max(1.0, out.get("get_obs_calls", 0.0))
        out["step_avg_sec"] = out.get("step_total_sec", 0.0) / step_calls
        out["obs_avg_sec"] = out.get("obs_total_sec", 0.0) / obs_calls
        out["can_commit_calls_per_step"] = out.get("can_commit_calls", 0.0) / step_calls
        out["dry_run_calls_per_step"] = out.get("can_commit_dry_run_calls", 0.0) / step_calls
        return out
