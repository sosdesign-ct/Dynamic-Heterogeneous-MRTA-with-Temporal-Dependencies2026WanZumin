from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from baseline.cbs_ta_ptc_assignment import cbs_ta_ptc_actions
from baseline.cbs_ta_ptc_2024 import cbs_ta_ptc_actions_2024
from baseline.ga_assignment import ga_actions


def _candidate_tasks(env) -> List[int]:
    out: List[int] = []
    for tid, task in env.tasks.items():
        if int(tid) >= int(env.config.max_tasks):
            continue
        if task.status in {"AVAILABLE", "ASSIGNED", "READY"}:
            out.append(int(tid))
    return out


def _dominant_task_type(task) -> int:
    return int(getattr(task, "task_type", int(np.argmax(task.req))))


def _dominant_agent_type(robot) -> int:
    return int(getattr(robot, "agent_type", int(np.argmax(robot.cap))))


@dataclass
class BaselinePlan:
    task_sequence_by_agent: Dict[int, List[int]] = field(default_factory=dict)
    path_targets_by_agent: Dict[int, List[int]] = field(default_factory=dict)
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class OriginalBaselineAdapter:
    method: str = "cbs_ta_ptc"
    timeout_sec: float = 10.0
    _cached_signature: Optional[Tuple] = None
    _cached_plan: Optional[BaselinePlan] = None
    _cached_action_signature: Optional[Tuple] = None
    _cached_actions: Optional[Dict[int, int]] = None
    _cached_action_uses: int = 0
    _planning_calls: int = 0
    _cache_hits: int = 0
    _zero_action_plans: int = 0

    def __post_init__(self):
        self.method = self.method.lower()
        if self.method not in {"cbs_ta_ptc", "cbs_ta_ptc_2018", "ga", "ctas_d"}:
            raise ValueError(f"Unsupported original baseline method: {self.method}")

    def supports_planned_execution(self) -> bool:
        return True

    def reset_episode(self) -> None:
        self._cached_signature = None
        self._cached_plan = None
        self._cached_action_signature = None
        self._cached_actions = None
        self._cached_action_uses = 0
        self._planning_calls = 0
        self._cache_hits = 0
        self._zero_action_plans = 0

    def get_episode_stats(self) -> Dict[str, float]:
        return {
            "baseline_planning_calls": float(self._planning_calls),
            "baseline_cache_hits": float(self._cache_hits),
            "baseline_zero_action_plans": float(self._zero_action_plans),
        }

    def plan(self, env) -> BaselinePlan:
        signature = self._scene_signature(env)
        if self._cached_signature == signature and self._cached_plan is not None:
            return self._cached_plan

        if self.method == "cbs_ta_ptc":
            plan = self._plan_cbs_ta_ptc(env)
        elif self.method == "ga":
            plan = self._plan_ga(env)
        else:
            plan = self._plan_ctas_d(env)

        self._cached_signature = signature
        self._cached_plan = plan
        return plan

    def act(self, env) -> Dict[int, int]:
        obs = env._get_obs()
        action_mask = obs["action_mask"]
        signature = self._decision_signature(env, action_mask)
        if self.method != "ga" and self._cached_action_signature == signature and self._cached_actions is not None:
            cached_has_nonnoop = any(int(action_id) > 0 for action_id in self._cached_actions.values())
            should_retry_cbs = self.method == "cbs_ta_ptc" and not cached_has_nonnoop and self._cached_action_uses >= 4
            if not should_retry_cbs:
                self._cached_action_uses += 1
                self._cache_hits += 1
                return dict(self._cached_actions)

        if self.method == "cbs_ta_ptc":
            actions = cbs_ta_ptc_actions_2024(env, action_mask, timeout_sec=float(self.timeout_sec))
        elif self.method == "cbs_ta_ptc_2018":
            actions = cbs_ta_ptc_actions(env, action_mask, timeout_sec=float(self.timeout_sec))
        elif self.method == "ga":
            actions = ga_actions(env, action_mask)
        else:
            plan = self.plan(env)
            actions = self.act_from_plan(env, plan)

        self._planning_calls += 1
        has_nonnoop_action = any(int(action_id) > 0 for action_id in actions.values())
        if not has_nonnoop_action:
            self._zero_action_plans += 1
        if self.method != "ga":
            self._cached_action_signature = signature
            self._cached_actions = dict(actions)
            self._cached_action_uses = 0
        return actions

    def _decision_signature(self, env, action_mask: np.ndarray) -> Tuple:
        robot_sig = tuple(
            (
                int(rid),
                tuple(np.round(robot.pos.astype(np.float64), 3)),
                str(robot.state),
                int(robot.task_id) if robot.task_id is not None else -1,
                float(getattr(robot, "eta", 0.0)),
                float(getattr(robot, "work_left", 0.0)),
            )
            for rid, robot in sorted(env.robots.items())
        )
        task_sig = tuple(
            (
                int(tid),
                str(task.status),
                tuple(np.round(task.loc.astype(np.float64), 3)),
                int(task.k),
                tuple(sorted(int(rid) for rid in task.assigned_robots)),
                int(task.release_time),
            )
            for tid, task in sorted(env.tasks.items())
            if int(tid) < int(env.config.max_tasks)
        )
        mask = np.ascontiguousarray(action_mask > 0.5, dtype=np.uint8)
        return (
            int(getattr(env, "current_time", 0)),
            robot_sig,
            task_sig,
            tuple(mask.shape),
            mask.tobytes(),
        )

    def act_from_plan(self, env, plan: Optional[BaselinePlan] = None) -> Dict[int, int]:
        if plan is None:
            plan = self.plan(env)
        sequences = {
            int(rid): list(sequence)
            for rid, sequence in plan.task_sequence_by_agent.items()
        }
        actions = {rid: 0 for rid in range(int(env.config.num_agents))}
        for rid, seq in sequences.items():
            robot = env.robots[int(rid)]
            if robot.state != "IDLE":
                continue
            selected_index = None
            for index, tid in enumerate(seq):
                if tid not in env.tasks:
                    continue
                task = env.tasks[int(tid)]
                if task.status in {"DONE", "EXECUTING", "PHYS_DONE"}:
                    continue
                if len(task.assigned_robots) >= int(task.k):
                    continue
                if not env._can_commit(int(rid), int(tid), dry_run=True):
                    continue
                actions[int(rid)] = int(tid) + 1
                selected_index = index
                break
            if selected_index is not None:
                del seq[selected_index]
        return actions

    def _scene_signature(self, env) -> Tuple:
        robot_sig = tuple(
            (
                int(rid),
                tuple(np.round(env.robots[rid].pos.astype(np.float64), 3)),
                int(env.robots[rid].agent_type),
                str(env.robots[rid].state),
                int(env.robots[rid].task_id) if env.robots[rid].task_id is not None else -1,
            )
            for rid in sorted(env.robots.keys())
        )
        task_sig = tuple(
            (
                int(tid),
                tuple(np.round(task.loc.astype(np.float64), 3)),
                int(task.k),
                int(task.release_time),
                int(_dominant_task_type(task)),
                str(task.status),
                tuple(sorted(int(rid) for rid in task.assigned_robots)),
            )
            for tid, task in sorted(env.tasks.items())
        )
        return robot_sig, task_sig

    def _plan_cbs_ta_ptc(self, env) -> BaselinePlan:
        obs = env._get_obs()
        action_mask = obs["action_mask"]
        seq: Dict[int, List[int]] = {int(rid): [] for rid in range(int(env.config.num_agents))}
        actions = cbs_ta_ptc_actions(env, action_mask, timeout_sec=float(self.timeout_sec))
        for rid, action_id in actions.items():
            tid = int(action_id) - 1
            if tid >= 0:
                seq[int(rid)].append(tid)

        planned_tasks = float(sum(len(v) for v in seq.values()))
        return BaselinePlan(
            task_sequence_by_agent=seq,
            metadata={"planned": float(planned_tasks > 0.0), "planned_tasks": planned_tasks},
        )

    def _plan_ga(self, env) -> BaselinePlan:
        obs = env._get_obs()
        actions = ga_actions(env, obs["action_mask"])
        seq: Dict[int, List[int]] = {int(rid): [] for rid in range(int(env.config.num_agents))}
        for rid, action_id in actions.items():
            tid = int(action_id) - 1
            if tid >= 0:
                seq[int(rid)].append(tid)
        planned_tasks = float(sum(len(v) for v in seq.values()))
        return BaselinePlan(
            task_sequence_by_agent=seq,
            metadata={"planned": float(planned_tasks > 0.0), "planned_tasks": planned_tasks},
        )

    def _plan_ctas_d(self, env) -> BaselinePlan:
        from baseline.ctasd_compat import TSPSolver

        solver = TSPSolver()
        seq: Dict[int, List[int]] = {int(rid): [] for rid in range(int(env.config.num_agents))}
        task_ids = _candidate_tasks(env)
        if not task_ids:
            return BaselinePlan(task_sequence_by_agent=seq, metadata={"planned": 0.0})

        type_to_agents: Dict[int, List[int]] = {}
        for rid, robot in env.robots.items():
            type_to_agents.setdefault(_dominant_agent_type(robot), []).append(int(rid))

        type_to_tasks: Dict[int, List[int]] = {}
        for tid in task_ids:
            type_to_tasks.setdefault(_dominant_task_type(env.tasks[int(tid)]), []).append(int(tid))

        for task_type, tids in type_to_tasks.items():
            agent_group = type_to_agents.get(int(task_type), [])
            if not agent_group:
                continue
            coords = [[0.0, 0.0, 0.0]]
            for tid in tids:
                task = env.tasks[int(tid)]
                coords.append([float(task.loc[0]), float(task.loc[1]), float(task.workload)])
            result = solver.run_solver(coords, num_vehicles=len(agent_group), depot=0)
            if result is None:
                continue
            routes, _ = result
            for local_idx, rid in enumerate(agent_group):
                route = routes.get(local_idx, [])
                for stop in route:
                    if int(stop) <= 0:
                        continue
                    mapped_tid = int(tids[int(stop) - 1])
                    seq[int(rid)].append(mapped_tid)
        return BaselinePlan(task_sequence_by_agent=seq, metadata={"planned": 1.0})


__all__ = ["BaselinePlan", "OriginalBaselineAdapter"]
