import argparse
import atexit
import json
import os
import platform
import random
import re
import subprocess
import sys
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, TypedDict, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_

from env.fixed_set_loader import load_fixed_env_from_path, resolve_fixed_scene_paths
from env.subproc_vec_env import make_mrta_env_pool
from mrta_policy import MRTAObs, MRTAPolicy, TaskGraphObs
from torch_runtime import resolve_torch_device
from train_config import TrainConfig
from env.mrta_config import MRTAConfig


class BenchResult(TypedDict):
    num_envs: float
    backend: str
    threads: float
    interop: float
    updates: float
    steps: float
    total_sec: float
    sps: float


class EnvPoolProtocol(Protocol):
    def close(self) -> None: ...
    def reset_all(self, seeds: Optional[Sequence[int]] = None, cfg_overrides: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]: ...
    def reset_one(
        self,
        env_i: int,
        cfg_overrides: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]: ...
    def step(self, action_dicts: Sequence[Dict[int, int]]) -> List[Tuple[Any, Any, Any, Any, Any]]: ...
    def get_states(self) -> List[Dict[str, Any]]: ...
    def set_states(self, states: Sequence[Dict[str, Any]]) -> None: ...
    def get_obs_all(self) -> List[Dict[str, Any]]: ...


def _build_autotune_candidates(cpu_count: int, base_num_envs: int) -> List[Tuple[int, int, int]]:
    env_points = [1, 2, 4, 8, 12]
    th_points = [1, 2, 4, 8, 12]
    env_points = sorted(set(max(1, min(int(x), max(1, cpu_count))) for x in env_points + [int(base_num_envs)]))
    th_points = sorted(set(max(1, min(int(x), max(1, cpu_count))) for x in th_points + [int(base_num_envs)]))

    # 优先测试对角线组合，再补少量非对角组合。
    cases = []
    for x in env_points:
        cases.append((x, x, 1))
    if 12 in env_points and 8 in th_points:
        cases.append((12, 8, 1))
    if 8 in env_points and 12 in th_points:
        cases.append((8, 12, 1))
    if 4 in env_points and 12 in th_points:
        cases.append((4, 12, 1))

    uniq = []
    seen = set()
    for c in cases:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def _parse_bench_result(stdout: str) -> Optional[BenchResult]:
    lines = [ln.strip() for ln in stdout.splitlines() if "RESULT" in ln]
    if not lines:
        return None
    line = lines[-1]
    pattern = (
        r"RESULT\s+num_envs=(\d+)\s+backend=([a-z]+)\s+threads=(\d+)\s+interop=(\d+)"
        r"\s+updates=(\d+)\s+steps=(\d+)\s+total_sec=([0-9.]+)\s+sps=([0-9.]+)"
    )
    m = re.search(pattern, line)
    if m is None:
        return None
    return {
        "num_envs": float(m.group(1)),
        "backend": m.group(2),
        "threads": float(m.group(3)),
        "interop": float(m.group(4)),
        "updates": float(m.group(5)),
        "steps": float(m.group(6)),
        "total_sec": float(m.group(7)),
        "sps": float(m.group(8)),
    }


def select_greedy_actions(
    policy: MRTAPolicy,
    torch_obs: MRTAObs,
    use_mask: bool = True,
    greedy_nonnoop_if_feasible: bool = False,
) -> torch.Tensor:
    forward_out = policy.forward(torch_obs, use_mask=use_mask)
    logits = forward_out[0]
    actions = torch.argmax(logits, dim=-1)
    if bool(greedy_nonnoop_if_feasible) and bool(use_mask) and "action_mask" in torch_obs:
        mask = torch_obs["action_mask"]
        if logits.dim() == 2:
            feasible_agent = (mask[:, 1:] > 0.5).any(dim=-1)
            noop_when_feasible = (actions == 0) & feasible_agent
            if torch.any(noop_when_feasible):
                task_logits = logits[:, 1:].masked_fill(mask[:, 1:] < 0.5, -1e9)
                best_task_act = torch.argmax(task_logits, dim=-1) + 1
                actions = torch.where(noop_when_feasible, best_task_act, actions)
        else:
            feasible_agent = (mask[:, :, 1:] > 0.5).any(dim=-1)
            noop_when_feasible = (actions == 0) & feasible_agent
            if torch.any(noop_when_feasible):
                task_logits = logits[:, :, 1:].masked_fill(mask[:, :, 1:] < 0.5, -1e9)
                best_task_act = torch.argmax(task_logits, dim=-1) + 1
                actions = torch.where(noop_when_feasible, best_task_act, actions)
    return actions


def evaluate_policy_on_fixed_set(
    policy: MRTAPolicy,
    fixed_set_dir: str,
    max_scenes: int,
    device: torch.device,
    wo_mask: bool = False,
    wo_commitment: bool = False,
    greedy_nonnoop_if_feasible: bool = True,
) -> Dict[str, float]:
    scene_paths = resolve_fixed_scene_paths(fixed_set_dir)
    if int(max_scenes) > 0:
        scene_paths = scene_paths[: int(max_scenes)]
    if not scene_paths:
        raise RuntimeError(f"fixed-set eval has no scenes: {fixed_set_dir}")

    records: List[Dict[str, float]] = []
    was_training = bool(policy.training)
    policy.eval()
    try:
        for scene_path in scene_paths:
            env = load_fixed_env_from_path(scene_path, wo_commitment=bool(wo_commitment))
            obs = env._get_obs()
            total_reward = 0.0
            while True:
                with torch.no_grad():
                    torch_obs = to_torch_obs(obs, device)
                    actions_t = select_greedy_actions(
                        policy,
                        torch_obs,
                        use_mask=not bool(wo_mask),
                        greedy_nonnoop_if_feasible=bool(greedy_nonnoop_if_feasible),
                    )
                actions = {i: int(actions_t[i].item()) for i in range(actions_t.shape[0])}
                obs, rewards, terminated, truncated, info = env.step(actions)
                total_reward += float(np.mean(list(rewards.values())))
                if terminated or truncated:
                    recognized_done_tasks = int(info.get("recognized_done_tasks", 0))
                    total_tasks = max(len(env.tasks), 1)
                    recognized_done_rate = float(info.get("recognized_done_rate", recognized_done_tasks / total_tasks))
                    records.append(
                        {
                            "return": float(total_reward),
                            "recognized_done_tasks": float(recognized_done_tasks),
                            "recognized_done_rate": float(recognized_done_rate),
                            "violation_rate": float(np.clip(float(info.get("violation_rate", 0.0)), 0.0, 1.0)),
                            "truncated": float(truncated),
                        }
                    )
                    break
    finally:
        if was_training:
            policy.train()

    keys = records[0].keys()
    return {k: float(np.mean([r[k] for r in records])) for k in keys}


def auto_tune_runtime_params(args, base_num_envs: int):
    total_budget_sec = max(10.0, float(args.auto_tune_seconds))
    cpu_count = int(os.cpu_count() or 1)
    bench_file = Path(__file__).with_name("bench_parallel_speed.py")
    if not bench_file.is_file():
        print("auto_tune disabled: benchmark script not found", flush=True)
        return None

    candidates = _build_autotune_candidates(cpu_count=cpu_count, base_num_envs=base_num_envs)
    if int(args.auto_tune_max_cases) > 0:
        candidates = candidates[: int(args.auto_tune_max_cases)]
    if not candidates:
        return None

    per_case_sec = max(3.0, total_budget_sec / float(len(candidates)))
    print(
        f"auto_tune start: budget={total_budget_sec:.1f}s cases={len(candidates)} "
        f"per_case={per_case_sec:.1f}s backend={args.env_backend}",
        flush=True,
    )

    results = []
    for i, (nenv, nth, nith) in enumerate(candidates, start=1):
        cmd = [
            sys.executable,
            str(bench_file),
            "--backend", str(args.env_backend),
            "--start_method", str(args.env_start_method),
            "--num_envs", str(int(nenv)),
            "--threads", str(int(nth)),
            "--interop", str(int(nith)),
            "--rollout_steps", str(int(args.auto_tune_rollout_steps)),
            "--target_seconds", f"{per_case_sec:.3f}",
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=max(30, int(per_case_sec * 4)),
            )
        except Exception as exc:
            print(f"auto_tune case {i}/{len(candidates)} failed: envs={nenv} threads={nth} err={exc}", flush=True)
            continue

        parsed = _parse_bench_result(proc.stdout)
        if parsed is None:
            print(
                f"auto_tune case {i}/{len(candidates)} invalid result: envs={nenv} threads={nth} "
                f"returncode={proc.returncode}",
                flush=True,
            )
            continue
        results.append(parsed)
        print(
            f"auto_tune case {i}/{len(candidates)}: envs={int(parsed['num_envs'])} "
            f"threads={int(parsed['threads'])} interop={int(parsed['interop'])} "
            f"backend={parsed['backend']} sps={parsed['sps']:.1f}",
            flush=True,
        )

    if not results:
        print("auto_tune failed: no valid benchmark case, using input runtime params", flush=True)
        return None

    best = max(results, key=lambda x: float(x["sps"]))
    print(
        f"auto_tune selected: envs={int(best['num_envs'])} threads={int(best['threads'])} "
        f"interop={int(best['interop'])} backend={best['backend']} sps={best['sps']:.1f}",
        flush=True,
    )
    return best


def configure_torch_cpu_threads(
    num_threads: int,
    num_interop_threads: int,
    device: Optional[torch.device] = None,
    env_backend: str = "sync",
    num_envs: int = 1,
):
    cpu_count = int(os.cpu_count() or 1)
    if int(num_threads) > 0:
        chosen_threads = int(num_threads)
    else:
        # 经验上，GPU 训练时环境与特征构建是 CPU 瓶颈，线程过多会引发争抢；
        # 这里默认压到较小值，避免出现 inter-op/OMP 过度并发导致吞吐下降。
        if device is not None and device.type == "cuda":
            if env_backend == "subproc" and int(num_envs) > 1:
                chosen_threads = min(cpu_count, max(4, min(12, int(num_envs))))
            else:
                chosen_threads = min(4, cpu_count)
        else:
            chosen_threads = cpu_count

    if int(num_interop_threads) > 0:
        chosen_interop = int(num_interop_threads)
    else:
        chosen_interop = 1

    try:
        torch.set_num_threads(max(1, chosen_threads))
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(max(1, chosen_interop))
    except Exception:
        pass
    return int(torch.get_num_threads()), int(torch.get_num_interop_threads())


def aggregate_metrics(metrics: List[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
    if not metrics:
        return {}
    out: Dict[str, Tuple[float, float]] = {}
    for k in metrics[0].keys():
        vals = np.asarray([m[k] for m in metrics], dtype=np.float64)
        out[k] = (float(np.mean(vals)), float(np.std(vals)))
    return out


def _recognized_done_desc() -> str:
    return "合法完成且无时序/依赖/组队违规的任务占比；任务可在物理完成后继续处于逻辑等待，直到满足完成侧约束才记为最终合规完成"


def _violation_rate_desc() -> str:
    return "启动侧时序或依赖违规率，越低越好"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_torch_obs(obs, device) -> MRTAObs:
    edge_type_np = obs["task_graph"]["edge_type"]
    edge_mask_np = obs["task_graph"].get("edge_mask", np.ones((edge_type_np.shape[0],), dtype=np.float32))
    task_graph = {
        "node_features": torch.as_tensor(obs["task_graph"]["node_features"], dtype=torch.float32, device=device),
        "edge_index": torch.as_tensor(obs["task_graph"]["edge_index"], dtype=torch.long, device=device),
        "edge_type": torch.as_tensor(obs["task_graph"]["edge_type"], dtype=torch.long, device=device),
        "edge_lag": torch.as_tensor(obs["task_graph"]["edge_lag"], dtype=torch.float32, device=device),
        "edge_mask": torch.as_tensor(edge_mask_np, dtype=torch.float32, device=device),
        "node_mask": torch.as_tensor(obs["task_graph"]["node_mask"], dtype=torch.float32, device=device),
    }
    if "node_t_start" in obs["task_graph"]:
        task_graph["node_t_start"] = torch.as_tensor(obs["task_graph"]["node_t_start"], dtype=torch.float32, device=device)
        task_graph["node_t_finish_phys"] = torch.as_tensor(obs["task_graph"]["node_t_finish_phys"], dtype=torch.float32, device=device)
    return {
        "task_graph": task_graph,
        "agent_feats": torch.as_tensor(obs["agent_feats"], dtype=torch.float32, device=device),
        "pair_feats": torch.as_tensor(obs["pair_feats"], dtype=torch.float32, device=device),
        "action_mask": torch.as_tensor(obs["action_mask"], dtype=torch.float32, device=device),
        "global_time": torch.as_tensor(obs["global_time"], dtype=torch.float32, device=device),
    }


def to_torch_obs_batch(obs_list, device) -> MRTAObs:
    """把一组观测打包成批量张量，支持可变边数量的 padding。"""
    bsz = len(obs_list)
    if bsz == 0:
        raise ValueError("obs_list must not be empty")

    node_features_np = np.stack([o["task_graph"]["node_features"] for o in obs_list], axis=0)
    node_mask_np = np.stack([o["task_graph"]["node_mask"] for o in obs_list], axis=0)
    agent_feats_np = np.stack([o["agent_feats"] for o in obs_list], axis=0)
    pair_feats_np = np.stack([o["pair_feats"] for o in obs_list], axis=0)
    action_mask_np = np.stack([o["action_mask"] for o in obs_list], axis=0)
    global_time_np = np.asarray([o["global_time"] for o in obs_list], dtype=np.float32)

    node_features = torch.as_tensor(node_features_np, dtype=torch.float32, device=device)
    node_mask = torch.as_tensor(node_mask_np, dtype=torch.float32, device=device)
    agent_feats = torch.as_tensor(agent_feats_np, dtype=torch.float32, device=device)
    pair_feats = torch.as_tensor(pair_feats_np, dtype=torch.float32, device=device)
    action_mask = torch.as_tensor(action_mask_np, dtype=torch.float32, device=device)
    global_time = torch.as_tensor(global_time_np, dtype=torch.float32, device=device)

    max_e = max(int(o["task_graph"]["edge_type"].shape[0]) for o in obs_list)
    edge_index = torch.zeros((bsz, 2, max_e), dtype=torch.long, device=device)
    edge_type = torch.zeros((bsz, max_e), dtype=torch.long, device=device)
    edge_lag = torch.zeros((bsz, max_e), dtype=torch.float32, device=device)
    edge_mask = torch.zeros((bsz, max_e), dtype=torch.float32, device=device)

    for i, o in enumerate(obs_list):
        edge_type_np_i = o["task_graph"]["edge_type"]
        e = int(edge_type_np_i.shape[0])
        edge_index[i, :, :e] = torch.as_tensor(o["task_graph"]["edge_index"], dtype=torch.long, device=device)
        edge_type[i, :e] = torch.as_tensor(edge_type_np_i, dtype=torch.long, device=device)
        edge_lag[i, :e] = torch.as_tensor(o["task_graph"]["edge_lag"], dtype=torch.float32, device=device)
        edge_mask[i, :e] = torch.as_tensor(
            o["task_graph"].get("edge_mask", np.ones((e,), dtype=np.float32)),
            dtype=torch.float32,
            device=device,
        )

    task_graph = {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_type": edge_type,
        "edge_lag": edge_lag,
        "node_mask": node_mask,
        "edge_mask": edge_mask,
    }
    if "node_t_start" in obs_list[0]["task_graph"]:
        task_graph["node_t_start"] = torch.as_tensor(
            np.stack([o["task_graph"]["node_t_start"] for o in obs_list], axis=0), dtype=torch.float32, device=device)
        task_graph["node_t_finish_phys"] = torch.as_tensor(
            np.stack([o["task_graph"]["node_t_finish_phys"] for o in obs_list], axis=0), dtype=torch.float32, device=device)
    return {
        "task_graph": task_graph,
        "agent_feats": agent_feats,
        "pair_feats": pair_feats,
        "action_mask": action_mask,
        "global_time": global_time,
    }


def index_torch_obs_batch(obs_batch: MRTAObs, idx_tensor: torch.Tensor) -> MRTAObs:
    task_graph = obs_batch["task_graph"]
    out_task_graph: TaskGraphObs = {
        "node_features": task_graph["node_features"].index_select(0, idx_tensor),
        "edge_index": task_graph["edge_index"].index_select(0, idx_tensor),
        "edge_type": task_graph["edge_type"].index_select(0, idx_tensor),
        "edge_lag": task_graph["edge_lag"].index_select(0, idx_tensor),
        "node_mask": task_graph["node_mask"].index_select(0, idx_tensor),
    }
    edge_mask = task_graph.get("edge_mask")
    if edge_mask is not None:
        out_task_graph["edge_mask"] = edge_mask.index_select(0, idx_tensor)
    if "node_t_start" in task_graph:
        out_task_graph["node_t_start"] = task_graph["node_t_start"].index_select(0, idx_tensor)
        out_task_graph["node_t_finish_phys"] = task_graph["node_t_finish_phys"].index_select(0, idx_tensor)
    return {
        "task_graph": out_task_graph,
        "agent_feats": obs_batch["agent_feats"].index_select(0, idx_tensor),
        "pair_feats": obs_batch["pair_feats"].index_select(0, idx_tensor),
        "action_mask": obs_batch["action_mask"].index_select(0, idx_tensor),
        "global_time": obs_batch["global_time"].index_select(0, idx_tensor),
    }


def build_aux_targets(obs_list, device):
    node_mask = torch.as_tensor(
        np.stack([o["task_graph"]["node_mask"] for o in obs_list], axis=0),
        dtype=torch.float32,
        device=device,
    )
    node_features = torch.as_tensor(
        np.stack([o["task_graph"]["node_features"] for o in obs_list], axis=0),
        dtype=torch.float32,
        device=device,
    )
    action_mask = torch.as_tensor(
        np.stack([o["action_mask"] for o in obs_list], axis=0),
        dtype=torch.float32,
        device=device,
    )
    task_start_target = node_features[..., -2]
    task_ready_target = node_features[..., -4]
    pair_feasible_target = action_mask[..., 1:]
    pair_mask = node_mask.unsqueeze(1).expand(-1, action_mask.shape[1], -1)
    return {
        "task_start_target": task_start_target,
        "task_ready_target": task_ready_target,
        "pair_feasible_target": pair_feasible_target,
        "task_mask": node_mask,
        "pair_mask": pair_mask,
    }


def slice_aux_targets(aux_targets, idx_tensor: torch.Tensor):
    return {
        "task_start_target": aux_targets["task_start_target"].index_select(0, idx_tensor),
        "task_ready_target": aux_targets["task_ready_target"].index_select(0, idx_tensor),
        "pair_feasible_target": aux_targets["pair_feasible_target"].index_select(0, idx_tensor),
        "task_mask": aux_targets["task_mask"].index_select(0, idx_tensor),
        "pair_mask": aux_targets["pair_mask"].index_select(0, idx_tensor),
    }


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """对张量做掩码均值，常用于 MAPPO 的 active-agent 聚合。"""
    w = mask.to(dtype=x.dtype)
    return (x * w).sum() / (w.sum() + eps)


def _parse_difficulty_bins(text: str) -> List[Tuple[float, float]]:
    bins: List[Tuple[float, float]] = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        lo_text, hi_text = token.split(":", 1)
        lo, hi = float(lo_text), float(hi_text)
        if not 0.0 <= lo < hi <= 1.0:
            raise ValueError(f"invalid difficulty bin: {token}")
        bins.append((lo, hi))
    return bins


def _training_reset_override(
    step_count: int,
    reset_index: int,
    difficulty_bins: Sequence[Tuple[float, float]],
    curriculum_steps: int,
    start_diff_max: float,
    target_diff_min: float,
    target_diff_max: float,
    hard_focus_prob: float,
    hard_focus_min: Optional[float],
    hard_focus_max: Optional[float],
) -> Tuple[Optional[Dict[str, Any]], float, bool]:
    curriculum_override, progress = _curriculum_difficulty_overrides(
        step_count, curriculum_steps, start_diff_max, target_diff_min, target_diff_max
    )
    if difficulty_bins:
        curriculum_max = float(
            curriculum_override.get("difficulty_max", target_diff_max)
            if isinstance(curriculum_override, dict)
            else target_diff_max
        )
        curriculum_max = max(curriculum_max, min(item[1] for item in difficulty_bins))
        eligible = [item for item in difficulty_bins if item[0] < curriculum_max]
        if not eligible:
            eligible = [min(difficulty_bins, key=lambda item: item[0])]
        lo, hi = eligible[int(reset_index) % len(eligible)]
        curriculum_override = {
            "enforce_difficulty_range": True,
            "difficulty_min": max(float(target_diff_min), float(lo)),
            "difficulty_max": min(float(curriculum_max), float(hi)),
            "strict_difficulty_bin": True,
        }
        curriculum_override.update(_difficulty_bin_generation_profile(float(lo), float(hi)))
    focused_override, focused = _maybe_apply_hard_focus_override(
        curriculum_override, hard_focus_prob, hard_focus_min, hard_focus_max
    )
    return focused_override, progress, focused


def _difficulty_bin_generation_profile(lo: float, hi: float) -> Dict[str, Any]:
    """Choose structural parameters whose natural score distribution overlaps a boundary bin."""
    profiles = {
        (0.20, 0.25): ((10, 25), 15, 0.03, 0, (0, 0)),
        (0.25, 0.30): ((12, 28), 18, 0.04, 0, (0, 0)),
        (0.30, 0.35): ((12, 28), 18, 0.04, 80, (1, 1)),
        (0.35, 0.40): ((14, 30), 20, 0.04, 35, (1, 1)),
        (0.40, 0.45): ((16, 32), 22, 0.05, 0, (0, 0)),
        (0.45, 0.50): ((18, 34), 24, 0.06, 0, (0, 0)),
        (0.75, 0.80): ((40, 50), 45, 0.35, 8, (1, 2)),
    }
    key = (round(float(lo), 2), round(float(hi), 2))
    profile = profiles.get(key)
    if profile is None:
        return {}
    init_range, total_floor, edge_density, arrival_every, arrival_range = profile
    return {
        "init_tasks_range": init_range,
        "total_tasks_floor": total_floor,
        "edge_density": edge_density,
        "dynamic_arrival_every": arrival_every,
        "dynamic_arrival_range": arrival_range,
    }



def _curriculum_difficulty_overrides(
    step_count: int,
    curriculum_steps: int,
    start_diff_max: float,
    target_diff_min: float,
    target_diff_max: float,
):
    if int(curriculum_steps) <= 0:
        return None, 1.0
    prog = float(np.clip(float(step_count) / float(max(1, int(curriculum_steps))), 0.0, 1.0))
    cur_max = float(start_diff_max + prog * (target_diff_max - start_diff_max))
    cur_max = float(np.clip(cur_max, 0.0, 1.0))
    cur_min = float(min(target_diff_min, cur_max))
    return {
        "enforce_difficulty_range": True,
        "difficulty_min": cur_min,
        "difficulty_max": cur_max,
    }, prog


def _curriculum_relaxed_override_candidates(
    cur_override: Optional[Dict[str, Any]],
    target_diff_min: float,
    target_diff_max: float,
) -> List[Optional[Dict[str, Any]]]:
    if not isinstance(cur_override, dict):
        return [cur_override]

    # Explicit difficulty bins are strict sampling strata. Retrying a failed
    # reset must preserve both the bin bounds and its generation profile.
    if bool(cur_override.get("strict_difficulty_bin", False)):
        return [dict(cur_override)]

    base_max = float(np.clip(cur_override.get("difficulty_max", target_diff_max), 0.0, 1.0))
    relax_bumps = [0.0, 0.05, 0.10, 0.20, 0.35, 1.00]
    out: List[Optional[Dict[str, Any]]] = []
    seen = set()
    for bump in relax_bumps:
        cand_max = float(np.clip(min(target_diff_max, base_max + float(bump)), 0.0, 1.0))
        cand_min = float(min(target_diff_min, cand_max))
        key = (round(cand_min, 6), round(cand_max, 6))
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "enforce_difficulty_range": True,
                "difficulty_min": cand_min,
                "difficulty_max": cand_max,
            }
        )
    return out


def _maybe_apply_hard_focus_override(
    base_override: Optional[Dict[str, float]],
    focus_prob: float,
    focus_min: Optional[float],
    focus_max: Optional[float],
) -> Tuple[Optional[Dict[str, float]], bool]:
    prob = float(np.clip(float(focus_prob), 0.0, 1.0))
    if prob <= 0.0 or focus_min is None or focus_max is None:
        return base_override, False
    if float(np.random.rand()) > prob:
        return base_override, False

    hard_min = float(np.clip(float(focus_min), 0.0, 1.0))
    hard_max = float(np.clip(float(focus_max), 0.0, 1.0))
    if hard_min > hard_max:
        hard_min, hard_max = hard_max, hard_min

    if isinstance(base_override, dict):
        base_min = float(np.clip(float(base_override.get("difficulty_min", 0.0)), 0.0, 1.0))
        base_max = float(np.clip(float(base_override.get("difficulty_max", 1.0)), 0.0, 1.0))
        if base_min > base_max:
            base_min, base_max = base_max, base_min
        cand_min = max(base_min, hard_min)
        cand_max = min(base_max, hard_max)
        if cand_min > cand_max:
            cand_min, cand_max = hard_min, hard_max
    else:
        cand_min, cand_max = hard_min, hard_max

    return {
        "enforce_difficulty_range": True,
        "difficulty_min": float(cand_min),
        "difficulty_max": float(cand_max),
    }, True


def parse_args():
    base_tcfg = TrainConfig()
    base_cfg = MRTAConfig()
    parser = argparse.ArgumentParser(description="Train MRTA with CTDE MAPPO")
    parser.add_argument(
        "--total_steps",
        type=int,
        default=None,
        help="override total environment interaction steps; default reads TrainConfig.total_steps",
    )
    parser.add_argument(
        "--additional_steps",
        type=int,
        default=0,
        help="when resuming, train for this many more env steps beyond the checkpoint step",
    )
    parser.add_argument("--rollout_steps", type=int, default=base_tcfg.rollout_steps)
    parser.add_argument("--update_epochs", type=int, default=base_tcfg.update_epochs)
    parser.add_argument("--minibatch_size", type=int, default=base_tcfg.minibatch_size)
    parser.add_argument("--lr", type=float, default=base_tcfg.lr)
    parser.add_argument("--entropy_coef", type=float, default=base_tcfg.entropy_coef)
    parser.add_argument("--entropy_coef_final", type=float, default=base_tcfg.entropy_coef_final,
                        help="final entropy coefficient for linear annealing")
    parser.add_argument("--entropy_anneal_steps", type=int, default=base_tcfg.entropy_anneal_steps,
                        help="entropy annealing steps; <=0 disables annealing")
    parser.add_argument("--value_coef", type=float, default=base_tcfg.value_coef)
    parser.add_argument("--value_clip_ratio", type=float, default=base_tcfg.value_clip_ratio)
    parser.add_argument("--huber_delta", type=float, default=base_tcfg.huber_delta,
                        help="delta for Huber value loss")
    parser.add_argument("--use_huber_value_loss", action="store_true",
                        help="enable Huber loss for critic value regression")
    parser.add_argument("--no_use_huber_value_loss", action="store_false", dest="use_huber_value_loss",
                        help="disable Huber value loss and use MSE")
    parser.set_defaults(use_huber_value_loss=base_tcfg.use_huber_value_loss)
    parser.add_argument("--target_kl", type=float, default=base_tcfg.target_kl,
                        help="approximate KL threshold for PPO early-stop; <=0 disables")
    parser.add_argument("--aux_loss_coef", type=float, default=base_tcfg.aux_loss_coef,
                        help="coefficient for task-start/task-ready/pair-feasible auxiliary losses")
    parser.add_argument("--noop_feasible_loss_coef", type=float, default=base_tcfg.noop_feasible_loss_coef,
                        help="auxiliary loss coefficient that discourages NOOP when task actions are feasible")
    parser.add_argument("--noop_feasible_adv_bonus", type=float, default=base_tcfg.noop_feasible_adv_bonus,
                        help="extra negative advantage added to sampled NOOP actions when feasible tasks exist")
    parser.add_argument("--noop_feasible_loss_anneal_steps", type=int, default=base_tcfg.noop_feasible_loss_anneal_steps,
                        help="linear anneal horizon for noop-feasible auxiliary loss; <=0 disables annealing")
    parser.add_argument("--noop_feasible_loss_floor_frac", type=float, default=base_tcfg.noop_feasible_loss_floor_frac,
                        help="fraction of noop-feasible loss coefficient kept after annealing")
    parser.add_argument("--normalize_value_loss", action="store_true",
                        help="normalize critic value loss using rollout return statistics")
    parser.add_argument("--no_normalize_value_loss", action="store_false", dest="normalize_value_loss",
                        help="disable value loss normalization")
    parser.set_defaults(normalize_value_loss=base_tcfg.normalize_value_loss)
    parser.add_argument("--log_every", type=int, default=base_tcfg.log_every)
    parser.add_argument("--tb", action="store_true",
                        help="enable TensorBoard logging")
    parser.add_argument("--no_tb", action="store_false", dest="tb",
                        help="disable TensorBoard logging")
    parser.set_defaults(tb=bool(base_tcfg.tb))
    parser.add_argument("--tb_logdir", type=str, default=base_tcfg.tb_logdir,
                        help="TensorBoard root log directory")
    parser.add_argument("--tb_flush_secs", type=int, default=base_tcfg.tb_flush_secs,
                        help="TensorBoard flush interval in seconds")
    parser.add_argument("--verbose_episode_log", action="store_true",
                        help="print per-episode detailed logs")
    parser.add_argument("--no_verbose_episode_log", action="store_false", dest="verbose_episode_log",
                        help="disable per-episode detailed logs for higher throughput")
    parser.set_defaults(verbose_episode_log=bool(base_tcfg.verbose_episode_log))
    parser.add_argument("--save_path", type=str, default=base_tcfg.save_path)
    parser.add_argument("--save_every_updates", type=int, default=base_tcfg.save_every_updates,
                        help="save checkpoint every N PPO updates; 0 disables periodic saving")
    parser.add_argument("--resume_path", type=str, default=base_tcfg.resume_path,
                        help="resume training from an existing checkpoint")
    parser.add_argument("--seed", type=int, default=base_tcfg.seed)
    parser.add_argument("--num_envs", type=int, default=base_tcfg.num_envs)
    parser.add_argument("--hidden_dim", type=int, default=base_tcfg.hidden_dim,
                        help="policy hidden dimension")
    parser.add_argument("--rgat_layers", type=int, default=base_tcfg.rgat_layers,
                        help="number of RGAT layers")
    parser.add_argument("--graph_transformer_heads", type=int, default=base_tcfg.graph_transformer_heads,
                        help="number of attention heads in the relation-aware graph transformer encoder")
    parser.add_argument("--env_backend", type=str, default=base_tcfg.env_backend, choices=["auto", "sync", "subproc"],
                        help="environment execution backend")
    parser.add_argument("--env_start_method", type=str, default=base_tcfg.env_start_method, choices=["spawn", "fork", "forkserver"],
                        help="multiprocessing start method when env_backend=subproc")
    parser.add_argument("--auto_tune_runtime", action="store_true",
                        help="run pre-training runtime auto-tuning for num_envs/threads")
    parser.add_argument("--no_auto_tune_runtime", action="store_false", dest="auto_tune_runtime",
                        help="disable pre-training runtime auto-tuning")
    parser.set_defaults(auto_tune_runtime=bool(base_tcfg.auto_tune_runtime))
    parser.add_argument("--auto_tune_seconds", type=float, default=base_tcfg.auto_tune_seconds,
                        help="time budget (seconds) for pre-training runtime auto-tuning")
    parser.add_argument("--auto_tune_rollout_steps", type=int, default=base_tcfg.auto_tune_rollout_steps,
                        help="rollout steps per benchmark iteration during auto-tuning")
    parser.add_argument("--auto_tune_max_cases", type=int, default=base_tcfg.auto_tune_max_cases,
                        help="max candidate configurations to benchmark during auto-tuning")
    parser.add_argument("--torch_num_threads", type=int, default=base_tcfg.torch_num_threads,
                        help="set torch intra-op CPU threads; <=0 uses auto heuristic")
    parser.add_argument("--torch_num_interop_threads", type=int, default=base_tcfg.torch_num_interop_threads,
                        help="set torch inter-op CPU threads; <=0 uses 1")
    parser.add_argument("--max_episodes", type=int, default=base_tcfg.max_episodes,
                        help="maximum number of finished episodes; <=0 disables this stop condition")
    parser.add_argument("--eval_fixed_every_steps", type=int, default=base_tcfg.eval_fixed_every_steps,
                        help="evaluate current policy on fixed test set every N env steps; 0 disables")
    parser.add_argument("--eval_fixed_set_dir", type=str, default=base_tcfg.eval_fixed_set_dir,
                        help="fixed test set directory used during training evaluation")
    parser.add_argument("--eval_fixed_max_scenes", type=int, default=base_tcfg.eval_fixed_max_scenes,
                        help="max fixed scenes per in-training evaluation")
    parser.add_argument("--save_best_fixed_eval", action="store_true",
                        help="save best checkpoint according to fixed-set evaluation metric")
    parser.add_argument("--no_save_best_fixed_eval", action="store_false", dest="save_best_fixed_eval",
                        help="disable best checkpoint saving from fixed-set evaluation")
    parser.set_defaults(save_best_fixed_eval=bool(base_tcfg.save_best_fixed_eval))
    parser.add_argument("--best_fixed_metric", type=str, default=base_tcfg.best_fixed_metric,
                        choices=["return", "recognized_done_tasks", "recognized_done_rate", "neg_violation"])
    parser.add_argument("--best_fixed_min_delta", type=float, default=base_tcfg.best_fixed_min_delta)
    parser.add_argument("--best_ckpt_path", type=str, default=base_tcfg.best_ckpt_path,
                        help="path for best fixed-eval checkpoint; empty derives from save_path")
    parser.add_argument("--eval_greedy_nonnoop_if_feasible", action="store_true",
                        help="for fixed-set eval, force greedy NOOP to best feasible task when any task is feasible")
    parser.add_argument("--no_eval_greedy_nonnoop_if_feasible", action="store_false", dest="eval_greedy_nonnoop_if_feasible")
    parser.set_defaults(eval_greedy_nonnoop_if_feasible=True)

    parser.add_argument("--wo_rgat", action="store_true")
    parser.add_argument(
        "--self_only_graph",
        action="store_true",
        help="retain the full graph encoder but mask all inter-task edges for a parameter-matched ablation",
    )
    parser.add_argument("--wo_graph_transformer", action="store_true",
                        help="fall back to the previous relation-specific GAT encoder when graph encoding is enabled")
    parser.add_argument("--wo_mask", action="store_true")
    parser.add_argument("--wo_commitment", action="store_true")
    parser.add_argument(
        "--sparse_dependency_features",
        action="store_true",
        help="方案C信息瓶颈：节点/配对特征移除依赖派生量，依赖推理仅经RGAT沿边获得（默认开启）",
    )
    parser.add_argument(
        "--no_sparse_dependency_features",
        action="store_false",
        dest="sparse_dependency_features",
        help="禁用信息瓶颈（恢复手工依赖派生特征）",
    )
    parser.add_argument(
        "--no_rgat_temporal_propagation",
        action="store_false",
        dest="rgat_temporal_propagation",
        help="禁用RGAT沿边时间松弛传播（默认主模型开启）",
    )
    parser.set_defaults(
        sparse_dependency_features=True,
        rgat_temporal_propagation=True,
    )
    parser.add_argument(
        "--classic_mappo",
        action="store_true",
        help=(
            "enable classic MAPPO profile: disable RGAT/dual decoder and disable "
            "extra optimization tricks (huber/value-norm/KL-stop/entropy-anneal/step-convergence)"
        ),
    )
    parser.add_argument(
        "--paper_fair_mappo",
        action="store_true",
        help=(
            "paper baseline MAPPO profile: disable graph encoder/dual decoder and "
            "disable MRTA-specific auxiliary/noop losses while keeping the same env budget"
        ),
    )
    parser.set_defaults(
        wo_rgat=bool(base_tcfg.wo_rgat),
        self_only_graph=False,
        wo_graph_transformer=not bool(base_tcfg.use_graph_transformer),
        wo_mask=bool(base_tcfg.wo_mask),
        wo_commitment=bool(base_tcfg.wo_commitment),
    )

    # 可选：环境难度/奖励覆盖参数，便于做课程学习与稳定性调参。
    parser.add_argument("--env_max_steps", type=int, default=base_cfg.max_steps)
    parser.add_argument("--env_num_agents", type=int, default=base_cfg.num_agents)
    parser.add_argument("--env_cap_dim", type=int, default=base_cfg.cap_dim)
    parser.add_argument("--env_max_tasks", type=int, default=base_cfg.max_tasks)
    parser.add_argument("--env_init_tasks_min", type=int, default=int(base_cfg.init_tasks_range[0]))
    parser.add_argument("--env_init_tasks_max", type=int, default=int(base_cfg.init_tasks_range[1]))
    parser.add_argument("--env_total_tasks_floor", type=int, default=base_cfg.total_tasks_floor)
    parser.add_argument("--env_total_tasks_cap", type=int, default=base_cfg.total_tasks_cap)
    parser.add_argument("--env_dynamic_arrival_every", type=int, default=base_cfg.dynamic_arrival_every)
    parser.add_argument("--env_dynamic_arrival_min", type=int, default=int(base_cfg.dynamic_arrival_range[0]))
    parser.add_argument("--env_dynamic_arrival_max", type=int, default=int(base_cfg.dynamic_arrival_range[1]))
    parser.add_argument("--env_edge_density", type=float, default=base_cfg.edge_density)
    parser.add_argument("--env_reward_progress_weight", type=float, default=base_cfg.reward_progress_weight)
    parser.add_argument("--env_reward_total_completion_weight", type=float, default=base_cfg.reward_total_completion_weight)
    parser.add_argument("--env_reward_violation_weight", type=float, default=base_cfg.reward_violation_weight)
    parser.add_argument("--env_reward_recognized_immediate_weight", type=float, default=base_cfg.reward_recognized_immediate_weight)
    parser.add_argument(
        "--env_strict_start_in_mask",
        dest="env_strict_start_in_mask",
        action="store_true",
        help="override env strict_start_in_mask=True",
    )
    parser.add_argument(
        "--env_no_strict_start_in_mask",
        dest="env_strict_start_in_mask",
        action="store_false",
        help="override env strict_start_in_mask=False",
    )
    parser.set_defaults(env_strict_start_in_mask=bool(base_cfg.strict_start_in_mask))
    parser.add_argument(
        "--env_edge_feasibility_guard",
        dest="env_edge_feasibility_guard",
        action="store_true",
        help="override env edge_feasibility_guard=True",
    )
    parser.add_argument(
        "--env_no_edge_feasibility_guard",
        dest="env_edge_feasibility_guard",
        action="store_false",
        help="override env edge_feasibility_guard=False",
    )
    parser.set_defaults(env_edge_feasibility_guard=bool(base_cfg.edge_feasibility_guard))
    parser.add_argument("--env_edge_global_check_interval", type=int, default=base_cfg.edge_global_check_interval)
    parser.add_argument("--env_edge_candidate_resamples", type=int, default=base_cfg.edge_candidate_resamples)
    parser.add_argument("--env_edge_max_in_degree", type=int, default=base_cfg.edge_max_in_degree)
    parser.add_argument("--env_sf_edge_accept_prob", type=float, default=base_cfg.sf_edge_accept_prob)
    parser.add_argument("--env_difficulty_min", type=float, default=base_cfg.difficulty_min,
                        help="scenario difficulty lower bound in [0,1]")
    parser.add_argument("--env_difficulty_max", type=float, default=base_cfg.difficulty_max,
                        help="scenario difficulty upper bound in [0,1]")
    parser.add_argument("--env_difficulty_max_attempts", type=int, default=base_cfg.difficulty_max_reset_attempts,
                        help="max reset attempts when sampling difficulty-constrained scenarios")
    parser.add_argument(
        "--env_difficulty_bins",
        type=str,
        default=base_tcfg.env_difficulty_bins,
        help="comma-separated reset bins such as 0.50:0.55,0.55:0.60; empty disables balanced bin sampling",
    )
    parser.add_argument("--curriculum_steps", type=int, default=base_tcfg.curriculum_steps)
    parser.add_argument(
        "--curriculum_start_difficulty_max",
        type=float,
        default=base_tcfg.curriculum_start_difficulty_max,
    )
    parser.add_argument("--env_hard_focus_prob", type=float, default=base_tcfg.env_hard_focus_prob)
    parser.add_argument("--env_hard_focus_min", type=float, default=base_tcfg.env_hard_focus_min)
    parser.add_argument("--env_hard_focus_max", type=float, default=base_tcfg.env_hard_focus_max)
    parser.add_argument(
        "--enforce_env_difficulty",
        dest="enforce_env_difficulty",
        action="store_true",
        help="strictly enforce random training scenarios to stay inside [env_difficulty_min, env_difficulty_max]",
    )
    parser.add_argument(
        "--no_enforce_env_difficulty",
        dest="enforce_env_difficulty",
        action="store_false",
        help="disable strict difficulty-range enforcement",
    )
    parser.set_defaults(enforce_env_difficulty=bool(base_cfg.enforce_difficulty_range))
    args = parser.parse_args()
    if bool(args.self_only_graph) and bool(args.wo_rgat):
        parser.error("--self_only_graph requires the graph encoder and cannot be combined with --wo_rgat")
    if str(getattr(args, "env_start_method", "spawn")) == "spawn" and platform.system().lower() == "linux":
        args.env_start_method = "fork"
    if not os.environ.get("MRTA_DEVICE"):
        os.environ["MRTA_DEVICE"] = "cuda"
    args.verbose_episode_log = bool(getattr(args, "verbose_episode_log", False))
    return args


def init_tb_writer(args):
    if not bool(args.tb):
        return None
    try:
        from torch.utils.tensorboard.writer import SummaryWriter
    except Exception:
        print(
            "tensorboard is not available, fallback to stdout logging only. "
            "Install with: pip install tensorboard",
            flush=True,
        )
        return None

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.tb_logdir, run_name)
    writer = SummaryWriter(log_dir=log_dir, flush_secs=max(1, int(args.tb_flush_secs)))
    print(f"tensorboard enabled: log_dir={log_dir}", flush=True)
    return writer


def save_checkpoint(path, policy, optimizer, cfg, tcfg, args, step_count, update_idx, env_states=None, obs_list=None, runtime_state=None):
    save_dir = os.path.dirname(path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    ckpt = {
        "model": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "mrta_config": asdict(cfg),
        "train_config": asdict(tcfg),
        "step_count": int(step_count),
        "update_idx": int(update_idx),
        "np_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "ablations": {
            "wo_rgat": args.wo_rgat,
            "self_only_graph": bool(getattr(args, "self_only_graph", False)),
            "wo_graph_transformer": args.wo_graph_transformer,
            "wo_mask": args.wo_mask,
            "wo_commitment": args.wo_commitment,
            "sparse_dependency_features": bool(getattr(args, "sparse_dependency_features", True)),
            "rgat_temporal_propagation": bool(getattr(args, "rgat_temporal_propagation", True)),
        },
        "algo_profile": {
            "classic_mappo": bool(getattr(args, "classic_mappo", False)),
            "use_graph_transformer": not bool(getattr(args, "wo_graph_transformer", False)),
            "graph_transformer_heads": int(getattr(args, "graph_transformer_heads", 4)),
        },
    }
    if torch.cuda.is_available():
        ckpt["torch_cuda_random_state_all"] = torch.cuda.get_rng_state_all()
    if env_states is not None:
        ckpt["env_states"] = env_states
    if obs_list is not None:
        ckpt["obs_list"] = obs_list
    if runtime_state is not None:
        ckpt["runtime_state"] = runtime_state
    torch.save({
        **ckpt,
    }, path)


def main():
    args = parse_args()
    cfg = MRTAConfig()
    base_tcfg = TrainConfig()
    total_steps = max(1, int(args.total_steps)) if args.total_steps is not None else max(1, int(base_tcfg.total_steps))

    if bool(getattr(args, "paper_fair_mappo", False)):
        args.classic_mappo = False
        args.wo_rgat = True
        args.wo_graph_transformer = True
        args.aux_loss_coef = 0.0
        args.noop_feasible_loss_coef = 0.0
        args.noop_feasible_adv_bonus = 0.0
        print(
            "paper_fair_mappo profile enabled: wo_rgat=1 wo_graph_transformer=1 "
            "aux_loss=0 noop_loss=0 noop_adv_bonus=0",
            flush=True,
        )

    if bool(args.classic_mappo):
        # Classic MAPPO profile for ablation: keep algorithm plain and comparable.
        args.wo_rgat = True
        args.use_huber_value_loss = False
        args.normalize_value_loss = False
        args.target_kl = 0.0
        args.entropy_coef_final = float(args.entropy_coef)
        args.entropy_anneal_steps = 0
        print(
            "classic_mappo profile enabled: wo_rgat=1 "
            "huber=0 value_norm=0 target_kl=0 entropy_anneal=0",
            flush=True,
        )

    tcfg = TrainConfig(
        total_steps=total_steps,
        rollout_steps=max(1, int(args.rollout_steps)),
        num_envs=max(1, int(args.num_envs)),
        seed=int(args.seed),
        update_epochs=max(1, int(args.update_epochs)),
        minibatch_size=max(1, int(args.minibatch_size)),
        max_episodes=max(0, int(args.max_episodes)),
        lr=float(args.lr),
        entropy_coef=float(args.entropy_coef),
        entropy_coef_final=float(args.entropy_coef_final),
        entropy_anneal_steps=max(0, int(args.entropy_anneal_steps)),
        value_coef=float(args.value_coef),
        value_clip_ratio=max(0.0, float(args.value_clip_ratio)),
        use_huber_value_loss=bool(args.use_huber_value_loss),
        huber_delta=max(1e-6, float(args.huber_delta)),
        normalize_value_loss=bool(args.normalize_value_loss),
        target_kl=float(args.target_kl),
        aux_loss_coef=max(0.0, float(args.aux_loss_coef)),
        noop_feasible_loss_coef=max(0.0, float(args.noop_feasible_loss_coef)),
        noop_feasible_adv_bonus=max(0.0, float(args.noop_feasible_adv_bonus)),
        noop_feasible_loss_anneal_steps=max(0, int(args.noop_feasible_loss_anneal_steps)),
        noop_feasible_loss_floor_frac=float(np.clip(float(args.noop_feasible_loss_floor_frac), 0.0, 1.0)),
        hidden_dim=max(32, int(args.hidden_dim)),
        rgat_layers=max(1, int(args.rgat_layers)),
        use_graph_transformer=not bool(args.wo_graph_transformer),
        graph_transformer_heads=max(1, int(args.graph_transformer_heads)),
        curriculum_steps=max(0, int(args.curriculum_steps)),
        curriculum_start_difficulty_max=float(np.clip(args.curriculum_start_difficulty_max, 0.0, 1.0)),
        env_difficulty_bins=str(args.env_difficulty_bins),
        env_hard_focus_prob=float(np.clip(args.env_hard_focus_prob, 0.0, 1.0)),
        env_hard_focus_min=args.env_hard_focus_min,
        env_hard_focus_max=args.env_hard_focus_max,
        eval_fixed_every_steps=max(0, int(args.eval_fixed_every_steps)),
        eval_fixed_set_dir=str(args.eval_fixed_set_dir),
        eval_fixed_max_scenes=max(0, int(args.eval_fixed_max_scenes)),
        save_best_fixed_eval=bool(args.save_best_fixed_eval),
        best_fixed_metric=str(args.best_fixed_metric),
        best_fixed_min_delta=max(0.0, float(args.best_fixed_min_delta)),
        best_ckpt_path=str(args.best_ckpt_path),
    )

    if bool(args.auto_tune_runtime):
        best_runtime = auto_tune_runtime_params(args=args, base_num_envs=tcfg.num_envs)
        if best_runtime is not None:
            tcfg.num_envs = max(1, int(best_runtime["num_envs"]))
            args.num_envs = int(tcfg.num_envs)
            args.torch_num_threads = int(best_runtime["threads"])
            args.torch_num_interop_threads = int(best_runtime["interop"])
            if str(args.env_backend) == "auto":
                # auto 模式下根据最优 num_envs 自动映射即可；
                # 但若 benchmark 明确给出 sync，也允许强制对齐。
                if str(best_runtime.get("backend", "")) in {"sync", "subproc"}:
                    args.env_backend = str(best_runtime["backend"])

    if args.env_max_steps is not None:
        cfg.max_steps = int(args.env_max_steps)
    if args.env_num_agents is not None:
        cfg.num_agents = max(1, int(args.env_num_agents))
    if args.env_cap_dim is not None:
        cfg.cap_dim = max(1, int(args.env_cap_dim))
    if args.env_max_tasks is not None:
        cfg.max_tasks = max(1, int(args.env_max_tasks))
    if args.env_init_tasks_min is not None or args.env_init_tasks_max is not None:
        init_min = int(args.env_init_tasks_min) if args.env_init_tasks_min is not None else int(cfg.init_tasks_range[0])
        init_max = int(args.env_init_tasks_max) if args.env_init_tasks_max is not None else int(cfg.init_tasks_range[1])
        if init_min > init_max:
            raise ValueError(f"invalid init task range: min={init_min} > max={init_max}")
        cfg.init_tasks_range = (init_min, init_max)
    if args.env_total_tasks_floor is not None:
        cfg.total_tasks_floor = int(args.env_total_tasks_floor)
    elif args.env_init_tasks_min is not None:
        # 若用户只改了初始任务下限，默认将其视为总任务下限。
        cfg.total_tasks_floor = int(cfg.init_tasks_range[0])
    if args.env_total_tasks_cap is not None:
        cfg.total_tasks_cap = int(args.env_total_tasks_cap)
    elif args.env_init_tasks_max is not None:
        # 若用户只改了初始任务上限，默认将其视为总任务上限。
        cfg.total_tasks_cap = int(cfg.init_tasks_range[1])
    if cfg.total_tasks_floor < 0:
        raise ValueError(f"invalid total tasks floor: {cfg.total_tasks_floor}")
    if cfg.total_tasks_cap <= 0:
        raise ValueError(f"invalid total tasks cap: {cfg.total_tasks_cap}")
    cfg.total_tasks_cap = min(int(cfg.total_tasks_cap), int(cfg.max_tasks))
    cfg.total_tasks_floor = min(int(cfg.total_tasks_floor), int(cfg.total_tasks_cap))
    if args.env_dynamic_arrival_every is not None:
        cfg.dynamic_arrival_every = int(args.env_dynamic_arrival_every)
    if args.env_dynamic_arrival_min is not None or args.env_dynamic_arrival_max is not None:
        arr_min = int(args.env_dynamic_arrival_min) if args.env_dynamic_arrival_min is not None else int(cfg.dynamic_arrival_range[0])
        arr_max = int(args.env_dynamic_arrival_max) if args.env_dynamic_arrival_max is not None else int(cfg.dynamic_arrival_range[1])
        if arr_min > arr_max:
            raise ValueError(f"invalid dynamic arrival range: min={arr_min} > max={arr_max}")
        cfg.dynamic_arrival_range = (arr_min, arr_max)
    if args.env_edge_density is not None:
        cfg.edge_density = float(args.env_edge_density)
    if args.env_reward_progress_weight is not None:
        cfg.reward_progress_weight = float(args.env_reward_progress_weight)
    if args.env_reward_total_completion_weight is not None:
        cfg.reward_total_completion_weight = float(args.env_reward_total_completion_weight)
    if args.env_reward_violation_weight is not None:
        cfg.reward_violation_weight = float(args.env_reward_violation_weight)
    if args.env_reward_recognized_immediate_weight is not None:
        cfg.reward_recognized_immediate_weight = float(args.env_reward_recognized_immediate_weight)
    if args.env_strict_start_in_mask is not None:
        cfg.strict_start_in_mask = bool(args.env_strict_start_in_mask)
    if args.env_edge_feasibility_guard is not None:
        cfg.edge_feasibility_guard = bool(args.env_edge_feasibility_guard)
    if args.env_edge_global_check_interval is not None:
        cfg.edge_global_check_interval = max(1, int(args.env_edge_global_check_interval))
    if args.env_edge_candidate_resamples is not None:
        cfg.edge_candidate_resamples = max(1, int(args.env_edge_candidate_resamples))
    if args.env_edge_max_in_degree is not None:
        cfg.edge_max_in_degree = max(1, int(args.env_edge_max_in_degree))
    if args.env_sf_edge_accept_prob is not None:
        cfg.sf_edge_accept_prob = float(np.clip(args.env_sf_edge_accept_prob, 0.0, 1.0))
    if args.env_difficulty_min is not None:
        cfg.difficulty_min = float(args.env_difficulty_min)
    if args.env_difficulty_max is not None:
        cfg.difficulty_max = float(args.env_difficulty_max)
    cfg.difficulty_min = float(np.clip(cfg.difficulty_min, 0.0, 1.0))
    cfg.difficulty_max = float(np.clip(cfg.difficulty_max, 0.0, 1.0))
    if cfg.difficulty_min > cfg.difficulty_max:
        raise ValueError(
            f"invalid difficulty range: min={cfg.difficulty_min} > max={cfg.difficulty_max}"
        )
    if args.env_difficulty_max_attempts is not None:
        cfg.difficulty_max_reset_attempts = max(1, int(args.env_difficulty_max_attempts))
    if args.enforce_env_difficulty is not None:
        cfg.enforce_difficulty_range = bool(args.enforce_env_difficulty)
    elif args.env_difficulty_min is not None or args.env_difficulty_max is not None:
        cfg.enforce_difficulty_range = True

    base_difficulty_overrides = {
        "enforce_difficulty_range": bool(cfg.enforce_difficulty_range),
        "difficulty_min": float(cfg.difficulty_min),
        "difficulty_max": float(cfg.difficulty_max),
    }
    difficulty_bins = _parse_difficulty_bins(args.env_difficulty_bins)
    # 仅在强制难度采样(enforce_env_difficulty)时启用难度分箱。非强制模式
    # (--no_enforce_env_difficulty，如大规模扩展训练)意图在 [difficulty_min, difficulty_max]
    # 全范围可行采样；若仍套用为小规模调参的默认窄分箱，其结构 profile(如 edge_density、
    # total_floor)会与大规模配置不兼容，导致 reset 永远无法采到目标难度而失败。
    if not bool(getattr(cfg, "enforce_difficulty_range", True)):
        difficulty_bins = []
    for lo, hi in difficulty_bins:
        if lo < cfg.difficulty_min or hi > cfg.difficulty_max:
            raise ValueError(
                f"difficulty bin [{lo:.3f}, {hi:.3f}] lies outside configured range "
                f"[{cfg.difficulty_min:.3f}, {cfg.difficulty_max:.3f}]"
            )

    cfg.ordered_commitment = not args.wo_commitment
    cfg.use_policy_commitment_logic = True
    cfg.sparse_dependency_features = bool(args.sparse_dependency_features)
    cfg.rgat_temporal_propagation = bool(args.rgat_temporal_propagation)
    log_every = max(1, int(args.log_every))

    predicted_backend = args.env_backend if args.env_backend != "auto" else ("subproc" if tcfg.num_envs > 1 else "sync")
    device = resolve_torch_device()
    configure_torch_cpu_threads(
        args.torch_num_threads,
        args.torch_num_interop_threads,
        device=device,
        env_backend=predicted_backend,
        num_envs=tcfg.num_envs,
    )
    set_seed(tcfg.seed)
    tb_writer = init_tb_writer(args)

    env_pool_raw, env_backend_used = make_mrta_env_pool(
        cfg=cfg,
        num_envs=tcfg.num_envs,
        seed=tcfg.seed,
        backend=args.env_backend,
        start_method=args.env_start_method,
    )
    env_pool = cast(EnvPoolProtocol, env_pool_raw)
    atexit.register(env_pool.close)
    init_last_exc = None
    obs_list = []
    for env_i in range(tcfg.num_envs):
        init_reset_overrides, _, _ = _training_reset_override(
            step_count=0,
            reset_index=env_i,
            difficulty_bins=difficulty_bins,
            curriculum_steps=tcfg.curriculum_steps,
            start_diff_max=tcfg.curriculum_start_difficulty_max,
            target_diff_min=cfg.difficulty_min,
            target_diff_max=cfg.difficulty_max,
            hard_focus_prob=tcfg.env_hard_focus_prob,
            hard_focus_min=tcfg.env_hard_focus_min,
            hard_focus_max=tcfg.env_hard_focus_max,
        )
        reset_obs = None
        for cand_override in _curriculum_relaxed_override_candidates(
            init_reset_overrides, cfg.difficulty_min, cfg.difficulty_max
        ):
            try:
                reset_obs = env_pool.reset_one(
                    env_i, cfg_overrides=cand_override, seed=tcfg.seed + env_i
                )
                break
            except RuntimeError as exc:
                init_last_exc = exc
        if reset_obs is None:
            raise RuntimeError(
                f"strict difficulty initialization failed for env={env_i}; "
                f"refusing to fall back to unconditioned generation: {init_last_exc}"
            )
        obs_list.append(reset_obs)
    node_dim = obs_list[0]["task_graph"]["node_features"].shape[-1]
    agent_dim = obs_list[0]["agent_feats"].shape[-1]
    pair_dim = obs_list[0]["pair_feats"].shape[-1]

    resume_ckpt = None
    if args.resume_path:
        if not os.path.isfile(args.resume_path):
            raise FileNotFoundError(f"resume checkpoint not found: {args.resume_path}")
        resume_ckpt = torch.load(args.resume_path, map_location=device)
        if isinstance(resume_ckpt, dict) and isinstance(resume_ckpt.get("model"), dict):
            resume_ablations = resume_ckpt.get("ablations", {}) if isinstance(resume_ckpt.get("ablations", {}), dict) else {}
            if "wo_graph_transformer" in resume_ablations:
                args.wo_graph_transformer = bool(resume_ablations.get("wo_graph_transformer", False))
            else:
                args.wo_graph_transformer = not any(
                    "task_encoder.layers.0.rel_bias" in k for k in resume_ckpt["model"].keys()
                )
            args.self_only_graph = bool(resume_ablations.get("self_only_graph", False))

    policy = MRTAPolicy(
        node_dim=node_dim,
        agent_dim=agent_dim,
        pair_dim=pair_dim,
        hidden_dim=tcfg.hidden_dim,
        rgat_layers=tcfg.rgat_layers,
        use_rgat=not args.wo_rgat,
        use_graph_transformer=not args.wo_graph_transformer,
        graph_transformer_heads=tcfg.graph_transformer_heads,
        self_only_graph=bool(args.self_only_graph),
        rgat_temporal_propagation=bool(args.rgat_temporal_propagation),
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=tcfg.lr)

    step_count = 0
    update_idx = 0
    cur_returns = [0.0 for _ in range(tcfg.num_envs)]
    cur_commit_attempts = [0 for _ in range(tcfg.num_envs)]
    cur_noop_feasible = [0 for _ in range(tcfg.num_envs)]
    cur_feasible_agents = [0 for _ in range(tcfg.num_envs)]
    episode_idx = 0
    noop_anneal_start_step = 0
    if args.resume_path:
        ckpt = resume_ckpt if resume_ckpt is not None else torch.load(args.resume_path, map_location=device)
        if "model" not in ckpt:
            raise RuntimeError(f"invalid resume checkpoint (missing 'model'): {args.resume_path}")
        try:
            policy.load_state_dict(ckpt["model"])
        except RuntimeError as exc:
            raise RuntimeError(
                "resume checkpoint is incompatible with current MRTAPolicy architecture. "
                "Please start training from scratch or provide a checkpoint trained by this code version. "
                f"path={args.resume_path}"
            ) from exc
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "np_random_state" in ckpt:
            np.random.set_state(ckpt["np_random_state"])
        if "torch_random_state" in ckpt:
            try:
                torch.set_rng_state(ckpt["torch_random_state"].cpu())
            except (TypeError, RuntimeError) as exc:
                print(f"warning: skip incompatible torch_random_state in resume checkpoint: {exc}", flush=True)
        if torch.cuda.is_available() and "torch_cuda_random_state_all" in ckpt:
            try:
                cuda_rng_states = [state.cpu() for state in ckpt["torch_cuda_random_state_all"]]
                torch.cuda.set_rng_state_all(cuda_rng_states)
            except (TypeError, RuntimeError) as exc:
                print(f"warning: skip incompatible torch_cuda_random_state_all in resume checkpoint: {exc}", flush=True)
        if "env_states" in ckpt and isinstance(ckpt["env_states"], list):
            env_states = ckpt["env_states"]
            if len(env_states) == tcfg.num_envs:
                env_pool.set_states(env_states)
        if "obs_list" in ckpt and isinstance(ckpt["obs_list"], list) and len(ckpt["obs_list"]) == tcfg.num_envs:
            obs_list = ckpt["obs_list"]
        elif "env_states" in ckpt and isinstance(ckpt["env_states"], list) and len(ckpt["env_states"]) == tcfg.num_envs:
            obs_list = env_pool.get_obs_all()
        rt = ckpt.get("runtime_state", {})
        if isinstance(rt, dict):
            if "cur_returns" in rt and len(rt["cur_returns"]) == tcfg.num_envs:
                cur_returns = [float(x) for x in rt["cur_returns"]]
            if "cur_commit_attempts" in rt and len(rt["cur_commit_attempts"]) == tcfg.num_envs:
                cur_commit_attempts = [int(x) for x in rt["cur_commit_attempts"]]
            if "cur_noop_feasible" in rt and len(rt["cur_noop_feasible"]) == tcfg.num_envs:
                cur_noop_feasible = [int(x) for x in rt["cur_noop_feasible"]]
            if "cur_feasible_agents" in rt and len(rt["cur_feasible_agents"]) == tcfg.num_envs:
                cur_feasible_agents = [int(x) for x in rt["cur_feasible_agents"]]
            episode_idx = int(rt.get("episode_idx", 0))
        step_count = int(ckpt.get("step_count", 0))
        update_idx = int(ckpt.get("update_idx", 0))
        additional_steps = max(0, int(getattr(args, "additional_steps", 0)))
        if additional_steps > 0:
            noop_anneal_start_step = step_count
            tcfg.total_steps = max(int(tcfg.total_steps), step_count + additional_steps)
            print(
                f"resume additional_steps enabled: current_step={step_count} "
                f"additional_steps={additional_steps} target_total_steps={tcfg.total_steps}",
                flush=True,
            )
        print(f"resumed from {args.resume_path}: step={step_count}, update={update_idx}", flush=True)

    print(
        f"start MAPPO training: total_steps={tcfg.total_steps} rollout_steps={tcfg.rollout_steps} "
        f"update_epochs={tcfg.update_epochs} minibatch_size={tcfg.minibatch_size} log_every={log_every}",
        flush=True,
    )
    print(
        f"env_cfg: max_steps={cfg.max_steps} num_agents={cfg.num_agents} init_tasks={cfg.init_tasks_range} "
        f"total_tasks_floor={cfg.total_tasks_floor} total_tasks_cap={cfg.total_tasks_cap} "
        f"dynamic_arrival_every={cfg.dynamic_arrival_every} dynamic_arrival_range={cfg.dynamic_arrival_range} "
        f"edge_density={cfg.edge_density:.3f} strict_start_in_mask={int(cfg.strict_start_in_mask)} "
        f"edge_guard={int(cfg.edge_feasibility_guard)} edge_check_interval={int(cfg.edge_global_check_interval)} "
        f"difficulty=[{cfg.difficulty_min:.2f},{cfg.difficulty_max:.2f}] "
        f"enforce={int(cfg.enforce_difficulty_range)} diff_attempts={int(cfg.difficulty_max_reset_attempts)}",
        flush=True,
    )
    print(
        f"reward_cfg: recognized_immediate={cfg.reward_recognized_immediate_weight:.4f} progress={cfg.reward_progress_weight:.4f} "
        f"total_completion={cfg.reward_total_completion_weight:.4f} violation={cfg.reward_violation_weight:.4f}",
        flush=True,
    )
    print(
        f"optim_cfg: lr={tcfg.lr:.6f} entropy_coef={tcfg.entropy_coef:.4f} clip_ratio={tcfg.clip_ratio:.3f} "
        f"value_coef={tcfg.value_coef:.3f} value_clip={tcfg.value_clip_ratio:.3f} "
        f"value_norm={int(tcfg.normalize_value_loss)} huber_value={int(tcfg.use_huber_value_loss)} "
        f"huber_delta={tcfg.huber_delta:.3f} target_kl={tcfg.target_kl:.4f} "
        f"entropy_final={tcfg.entropy_coef_final:.4f} entropy_anneal_steps={tcfg.entropy_anneal_steps} "
        f"max_grad_norm={tcfg.max_grad_norm:.3f} noop_loss_coef={tcfg.noop_feasible_loss_coef:.4f} "
        f"noop_adv_bonus={tcfg.noop_feasible_adv_bonus:.3f}",
        flush=True,
    )
    print(
        f"arch_cfg: hidden_dim={tcfg.hidden_dim} rgat_layers={tcfg.rgat_layers} "
        f"use_rgat={int(not args.wo_rgat)} "
        f"self_only_graph={int(getattr(args, 'self_only_graph', False))}",
        flush=True,
    )
    print(
        f"stop_cfg: max_episodes={tcfg.max_episodes} step_conv={int(tcfg.enable_step_convergence)} "
        f"step_window={tcfg.step_convergence_window} post_conv_steps={tcfg.post_convergence_steps}",
        flush=True,
    )
    print(
        f"cpu_cfg: torch_num_threads={torch.get_num_threads()} "
        f"torch_num_interop_threads={torch.get_num_interop_threads()} num_envs={tcfg.num_envs} "
        f"env_backend={env_backend_used} start_method={args.env_start_method}",
        flush=True,
    )
    ep_return_window = deque(maxlen=tcfg.convergence_window)
    ep_reject_window = deque(maxlen=tcfg.convergence_window)
    ep_truncated_window = deque(maxlen=tcfg.convergence_window)
    ep_violation_window = deque(maxlen=tcfg.convergence_window)
    ep_noop_feasible_window = deque(maxlen=tcfg.convergence_window)
    best_window_return = float("-inf")
    stale_updates = 0
    converged = False
    step_reward_window = deque(maxlen=tcfg.step_convergence_window * 2)
    step_converged = False
    pending_stop_step = None
    stop_reason = ""
    train_wall_t0 = time.perf_counter()
    perf_collect_sec_total = 0.0
    perf_update_sec_total = 0.0
    perf_obs_pack_sec_total = 0.0
    perf_act_sec_total = 0.0
    perf_env_step_sec_total = 0.0
    perf_bootstrap_sec_total = 0.0
    perf_gae_sec_total = 0.0
    perf_rollout_pack_sec_total = 0.0
    perf_minibatch_obs_sec_total = 0.0
    perf_eval_actions_sec_total = 0.0
    perf_optim_sec_total = 0.0
    best_fixed_score = -float("inf")
    fixed_eval_interval = int(tcfg.eval_fixed_every_steps)
    if fixed_eval_interval > 0:
        if step_count > 0:
            next_fixed_eval_step = ((int(step_count) // fixed_eval_interval) + 1) * fixed_eval_interval
        else:
            next_fixed_eval_step = fixed_eval_interval
    else:
        next_fixed_eval_step = 0
    best_ckpt_path = str(tcfg.best_ckpt_path).strip()
    if not best_ckpt_path:
        save_path_obj = Path(str(args.save_path))
        best_ckpt_path = str(save_path_obj.with_name(f"{save_path_obj.stem}_best_fixed_eval{save_path_obj.suffix}"))

    while step_count < tcfg.total_steps and not converged:
        collect_t0 = time.perf_counter()
        rollout = []
        ep_returns = []
        rewards_mat = np.zeros((tcfg.rollout_steps, tcfg.num_envs), dtype=np.float32)
        agent_rewards_mat = np.zeros((tcfg.rollout_steps, tcfg.num_envs, cfg.num_agents), dtype=np.float32)
        values_mat = np.zeros((tcfg.rollout_steps, tcfg.num_envs), dtype=np.float32)
        dones_mat = np.zeros((tcfg.rollout_steps, tcfg.num_envs), dtype=np.float32)

        for t in range(tcfg.rollout_steps):
            t_obs_pack = time.perf_counter()
            torch_obs = to_torch_obs_batch(obs_list, device)
            perf_obs_pack_sec_total += time.perf_counter() - t_obs_pack
            t_act = time.perf_counter()
            with torch.no_grad():
                actions, logp, value, _ = policy.act(torch_obs, deterministic=False, use_mask=not args.wo_mask)
            perf_act_sec_total += time.perf_counter() - t_act

            actions_np = actions.detach().cpu().numpy()
            logp_np = logp.detach().cpu().numpy()
            value_np = value.detach().cpu().numpy()

            action_dicts = []
            for env_i in range(tcfg.num_envs):
                act_i = actions_np[env_i]
                action_dict = {i: int(act_i[i]) for i in range(act_i.shape[0])}
                mask_now = obs_list[env_i]["action_mask"]
                feasible_now = (mask_now[:, 1:].sum(axis=1) > 0)
                feasible_agents = int(np.sum(feasible_now))
                noop_feasible = int(sum(1 for i in range(act_i.shape[0]) if int(act_i[i]) == 0 and bool(feasible_now[i])))
                cur_commit_attempts[env_i] += sum(1 for a in action_dict.values() if a > 0)
                cur_noop_feasible[env_i] += noop_feasible
                cur_feasible_agents[env_i] += feasible_agents
                action_dicts.append(action_dict)

            t_env = time.perf_counter()
            step_outs = env_pool.step(action_dicts)
            perf_env_step_sec_total += time.perf_counter() - t_env

            for env_i, step_out in enumerate(step_outs):
                act_i = actions_np[env_i]
                next_obs, rewards_dict, terminated, truncated, info = step_out
                reward = float(np.mean(list(rewards_dict.values())))
                agent_reward_vec = np.array(
                    [float(rewards_dict.get(rid, reward)) for rid in range(cfg.num_agents)],
                    dtype=np.float32,
                )
                done = terminated or truncated

                rollout.append({
                    "obs": obs_list[env_i],
                    "actions": act_i,
                    "logp": logp_np[env_i],
                    "active": (obs_list[env_i]["action_mask"][:, 1:].sum(axis=1) > 0).astype(np.float32),
                    "env_i": env_i,
                    "t": t,
                })
                rewards_mat[t, env_i] = reward
                agent_rewards_mat[t, env_i, :] = agent_reward_vec
                values_mat[t, env_i] = float(value_np[env_i])
                dones_mat[t, env_i] = 1.0 if done else 0.0

                step_count += 1
                cur_returns[env_i] += reward
                obs_list[env_i] = next_obs
                step_reward_window.append(reward)

                if tcfg.enable_step_convergence and (not step_converged) and len(step_reward_window) == (tcfg.step_convergence_window * 2):
                    rewards_2w = np.asarray(step_reward_window, dtype=np.float32)
                    prev_w = rewards_2w[:tcfg.step_convergence_window]
                    next_w = rewards_2w[tcfg.step_convergence_window:]
                    prev_mean = float(np.mean(prev_w))
                    next_mean = float(np.mean(next_w))
                    next_std = float(np.std(next_w))
                    mean_gap = abs(next_mean - prev_mean)
                    if mean_gap <= tcfg.step_convergence_min_delta and next_std <= tcfg.step_convergence_std_max:
                        step_converged = True
                        target_stop_step = min(tcfg.total_steps, step_count + tcfg.post_convergence_steps)
                        pending_stop_step = target_stop_step if pending_stop_step is None else min(pending_stop_step, target_stop_step)
                        if not stop_reason:
                            stop_reason = (
                                f"step-convergence(mean_gap={mean_gap:.4f}, std={next_std:.4f})"
                            )
                        print(
                            f"step convergence detected at step={step_count}: "
                            f"mean_gap={mean_gap:.4f} std={next_std:.4f} stop_at={pending_stop_step}",
                            flush=True,
                        )

                if done:
                    ep_returns.append(cur_returns[env_i])
                    episode_idx += 1
                    total_tasks = max(int(info.get("total_tasks", 1)), 1)
                    reject_rate = float(info.get("rejected_commitments", 0.0)) / float(max(cur_commit_attempts[env_i], 1))
                    truncated_flag = 1.0 if truncated else 0.0
                    recognized_done_tasks = int(info.get("recognized_done_tasks", 0))
                    recognized_done_rate = float(info.get("recognized_done_rate", 0.0))
                    gen_makespan = float(info.get("generalized_makespan", np.nan))
                    violation_rate = float(info.get("violation_rate", 0.0))
                    waiting_cost = float(info.get("waiting_cost", 0.0))
                    logical_wait_cost = float(info.get("logical_wait_cost", 0.0))
                    missed_window_cnt = int(info.get("missed_ss_window", 0))
                    infeasible_team_cnt = int(info.get("infeasible_team", 0))
                    scene_difficulty = float(info.get("scenario_difficulty", 0.0))
                    noop_feasible_rate = float(cur_noop_feasible[env_i]) / float(max(cur_feasible_agents[env_i], 1))
                    base_reward = float(info.get("base_reward", float("nan")))
                    local_credit_mean = float(info.get("local_credit_mean", 0.0))
                    local_credit_min = float(info.get("local_credit_min", 0.0))
                    local_credit_max = float(info.get("local_credit_max", 0.0))
                    credit_reject_agents = int(len(info.get("credit_reject_agents", [])))
                    credit_infeasible_agents = int(len(info.get("credit_infeasible_agents", [])))
                    credit_noop_agents = int(len(info.get("credit_noop_feasible_agents", [])))

                    ep_return_window.append(cur_returns[env_i])
                    ep_reject_window.append(reject_rate)
                    ep_truncated_window.append(truncated_flag)
                    ep_violation_window.append(violation_rate)
                    ep_noop_feasible_window.append(noop_feasible_rate)

                    if args.verbose_episode_log:
                        print(
                            f"episode={episode_idx} step={step_count} env={env_i} episode_return={cur_returns[env_i]:.3f} "
                            f"compliant_done_tasks={recognized_done_tasks}/{total_tasks} "
                            f"compliant_done_rate={recognized_done_rate:.3f} reject_rate={reject_rate:.3f} "
                            f"start_violation_rate={violation_rate:.3f} missed_window={missed_window_cnt} "
                            f"waiting_cost={waiting_cost:.1f} logical_wait_cost={logical_wait_cost:.1f} "
                            f"infeasible_team={infeasible_team_cnt} gen_makespan={gen_makespan:.3f} "
                            f"difficulty={scene_difficulty:.3f} "
                            f"noop_when_feasible={noop_feasible_rate:.3f} "
                            f"truncated={int(truncated)} makespan={int(info.get('makespan', 0))} "
                            f"total_tasks={total_tasks}",
                            flush=True,
                        )

                    if tb_writer is not None:
                        tb_writer.add_scalar("episode/return", float(cur_returns[env_i]), episode_idx)
                        tb_writer.add_scalar("episode/recognized_done_tasks", float(recognized_done_tasks), episode_idx)
                        tb_writer.add_scalar("episode/recognized_done_rate", recognized_done_rate, episode_idx)
                        tb_writer.add_scalar("episode/reject_rate", reject_rate, episode_idx)
                        tb_writer.add_scalar("episode/violation_rate", violation_rate, episode_idx)
                        tb_writer.add_scalar("episode/waiting_cost", waiting_cost, episode_idx)
                        tb_writer.add_scalar("episode/logical_wait_cost", logical_wait_cost, episode_idx)
                        tb_writer.add_scalar("episode/missed_window", float(missed_window_cnt), episode_idx)
                        tb_writer.add_scalar("episode/infeasible_team", float(infeasible_team_cnt), episode_idx)
                        tb_writer.add_scalar("episode/gen_makespan", gen_makespan, episode_idx)
                        tb_writer.add_scalar("episode/scene_difficulty", scene_difficulty, episode_idx)
                        tb_writer.add_scalar("episode/noop_when_feasible", noop_feasible_rate, episode_idx)
                        tb_writer.add_scalar("episode/base_reward", base_reward, episode_idx)
                        tb_writer.add_scalar("episode/local_credit_mean", local_credit_mean, episode_idx)
                        tb_writer.add_scalar("episode/local_credit_min", local_credit_min, episode_idx)
                        tb_writer.add_scalar("episode/local_credit_max", local_credit_max, episode_idx)
                        tb_writer.add_scalar("episode/credit_reject_agents", float(credit_reject_agents), episode_idx)
                        tb_writer.add_scalar("episode/credit_infeasible_agents", float(credit_infeasible_agents), episode_idx)
                        tb_writer.add_scalar("episode/credit_noop_feasible_agents", float(credit_noop_agents), episode_idx)
                        tb_writer.add_scalar("episode/truncated", truncated_flag, episode_idx)
                        tb_writer.add_scalar("episode/step", float(step_count), episode_idx)

                    cur_returns[env_i] = 0.0
                    cur_commit_attempts[env_i] = 0
                    cur_noop_feasible[env_i] = 0
                    cur_feasible_agents[env_i] = 0

                    reset_overrides, curriculum_progress, hard_focused = _training_reset_override(
                        step_count=step_count,
                        reset_index=episode_idx,
                        difficulty_bins=difficulty_bins,
                        curriculum_steps=tcfg.curriculum_steps,
                        start_diff_max=tcfg.curriculum_start_difficulty_max,
                        target_diff_min=cfg.difficulty_min,
                        target_diff_max=cfg.difficulty_max,
                        hard_focus_prob=tcfg.env_hard_focus_prob,
                        hard_focus_min=tcfg.env_hard_focus_min,
                        hard_focus_max=tcfg.env_hard_focus_max,
                    )
                    reset_done = False
                    last_reset_exc = None
                    cand_overrides = _curriculum_relaxed_override_candidates(
                        reset_overrides, cfg.difficulty_min, cfg.difficulty_max
                    )

                    if tb_writer is not None:
                        tb_writer.add_scalar("curriculum/progress", curriculum_progress, episode_idx)
                        tb_writer.add_scalar("curriculum/hard_focus", float(hard_focused), episode_idx)

                    for cand_idx, cand_override in enumerate(cand_overrides):
                        try:
                            obs_list[env_i] = env_pool.reset_one(
                                env_i,
                                cfg_overrides=cand_override,
                                seed=tcfg.seed + episode_idx * tcfg.num_envs + env_i,
                            )
                            reset_done = True
                            break
                        except RuntimeError as exc:
                            last_reset_exc = exc

                    if not reset_done:
                        print(
                            f"reset fallback env={env_i} step={step_count}: {last_reset_exc}",
                            flush=True,
                        )
                        obs_list[env_i] = env_pool.reset_one(
                            env_i,
                            cfg_overrides=base_difficulty_overrides,
                            seed=tcfg.seed + episode_idx * tcfg.num_envs + env_i,
                        )

                    if tcfg.max_episodes > 0 and episode_idx >= tcfg.max_episodes:
                        converged = True
                        stop_reason = f"max_episodes reached ({episode_idx}/{tcfg.max_episodes})"

                if step_count >= tcfg.total_steps or converged:
                    break

            if step_count >= tcfg.total_steps or converged:
                break

        perf_collect_sec_total += time.perf_counter() - collect_t0

        update_t0 = time.perf_counter()
        with torch.no_grad():
            t_boot = time.perf_counter()
            last_val = policy.forward(to_torch_obs_batch(obs_list, device), use_mask=not args.wo_mask)[1].detach().cpu().numpy()
            perf_bootstrap_sec_total += time.perf_counter() - t_boot

        env_steps = np.zeros((tcfg.num_envs,), dtype=np.int32)
        for item in rollout:
            env_steps[int(item["env_i"])] += 1
        t_eff = max(1, min(env_steps))
        rewards_cut = rewards_mat[:t_eff, :]
        agent_rewards_cut = agent_rewards_mat[:t_eff, :, :]
        values_cut = values_mat[:t_eff, :]
        dones_cut = dones_mat[:t_eff, :]

        t_gae = time.perf_counter()
        adv_mat = np.zeros_like(rewards_cut)
        agent_adv_mat = np.zeros_like(agent_rewards_cut)
        ret_mat = np.zeros_like(rewards_cut)
        for env_i in range(tcfg.num_envs):
            vals = np.concatenate([values_cut[:, env_i], np.array([last_val[env_i]], dtype=np.float32)], axis=0)
            gae = 0.0
            for t in reversed(range(t_eff)):
                delta = rewards_cut[t, env_i] + tcfg.gamma * vals[t + 1] * (1.0 - dones_cut[t, env_i]) - vals[t]
                gae = delta + tcfg.gamma * tcfg.gae_lambda * (1.0 - dones_cut[t, env_i]) * gae
                adv_mat[t, env_i] = gae
                ret_mat[t, env_i] = gae + vals[t]
            for agent_i in range(cfg.num_agents):
                agent_gae = 0.0
                for t in reversed(range(t_eff)):
                    agent_delta = (
                        agent_rewards_cut[t, env_i, agent_i]
                        + tcfg.gamma * vals[t + 1] * (1.0 - dones_cut[t, env_i])
                        - vals[t]
                    )
                    agent_gae = agent_delta + tcfg.gamma * tcfg.gae_lambda * (1.0 - dones_cut[t, env_i]) * agent_gae
                    agent_adv_mat[t, env_i, agent_i] = agent_gae
            perf_gae_sec_total += time.perf_counter() - t_gae

        filtered_rollout = []
        flat_adv_agent = []
        flat_ret = []
        flat_old_value = []
        flat_noop_feasible = []
        for item in rollout:
            t_i = int(item["t"])
            if t_i >= t_eff:
                continue
            e_i = int(item["env_i"])
            filtered_rollout.append(item)
            flat_adv_agent.append(agent_adv_mat[t_i, e_i].astype(np.float32, copy=False))
            flat_ret.append(float(ret_mat[t_i, e_i]))
            flat_old_value.append(float(values_cut[t_i, e_i]))
            active_i = np.asarray(item["active"], dtype=np.float32)
            actions_i = np.asarray(item["actions"], dtype=np.int64)
            flat_noop_feasible.append(((actions_i == 0) & (active_i > 0.5)).astype(np.float32, copy=False))
        rollout = filtered_rollout

        adv_agent = np.stack(flat_adv_agent, axis=0).astype(np.float32, copy=False)
        noop_feasible_np = np.stack(flat_noop_feasible, axis=0).astype(np.float32, copy=False)
        ret = np.array(flat_ret, dtype=np.float32)
        old_val = np.array(flat_old_value, dtype=np.float32)

        t_rollout_pack = time.perf_counter()
        rollout_obs_list = [r["obs"] for r in rollout]
        rollout_obs_batch = to_torch_obs_batch(rollout_obs_list, device)
        perf_rollout_pack_sec_total += time.perf_counter() - t_rollout_pack
        actions_tensor = torch.as_tensor(np.stack([r["actions"] for r in rollout], axis=0), dtype=torch.long, device=device)
        old_logp_tensor = torch.as_tensor(np.stack([r["logp"] for r in rollout], axis=0), dtype=torch.float32, device=device)
        active_tensor = torch.as_tensor(np.stack([r["active"] for r in rollout], axis=0), dtype=torch.float32, device=device)
        noop_feasible_tensor = torch.as_tensor(noop_feasible_np, dtype=torch.float32, device=device)
        active_np = active_tensor.detach().cpu().numpy()
        active_adv = adv_agent[active_np > 0.5]
        if active_adv.size > 0:
            adv_agent = (adv_agent - float(active_adv.mean())) / (float(active_adv.std()) + 1e-8)
        else:
            adv_agent = (adv_agent - float(adv_agent.mean())) / (float(adv_agent.std()) + 1e-8)
        # Apply the feasible-NOOP penalty after normalization so the configured
        # coefficient is expressed in advantage-standard-deviation units rather
        # than being diluted by the raw reward/GAE scale.
        if tcfg.noop_feasible_adv_bonus > 0.0:
            adv_agent = adv_agent - float(tcfg.noop_feasible_adv_bonus) * noop_feasible_np
        aux_targets_all = build_aux_targets(rollout_obs_list, device)
        old_value_tensor = torch.as_tensor(old_val, dtype=torch.float32, device=device)
        adv_agent_tensor = torch.as_tensor(adv_agent, dtype=torch.float32, device=device)
        ret_tensor = torch.as_tensor(ret, dtype=torch.float32, device=device)
        ret_center = ret_tensor.mean()
        ret_scale = ret_tensor.std(unbiased=False).clamp_min(float(tcfg.value_norm_eps))

        mb_pol_losses = []
        mb_val_losses = []
        mb_ent_losses = []
        mb_aux_losses = []
        mb_noop_losses = []
        mb_total_losses = []
        mb_approx_kls = []
        mb_clip_fracs = []
        kl_early_stop = False

        if tcfg.entropy_anneal_steps > 0:
            anneal_p = min(1.0, float(step_count) / float(tcfg.entropy_anneal_steps))
            entropy_coef_now = float(
                tcfg.entropy_coef + anneal_p * (tcfg.entropy_coef_final - tcfg.entropy_coef)
            )
        else:
            entropy_coef_now = float(tcfg.entropy_coef)
        if tcfg.noop_feasible_loss_anneal_steps > 0:
            noop_anneal_elapsed = max(0.0, float(step_count - noop_anneal_start_step))
            noop_anneal_p = min(1.0, noop_anneal_elapsed / float(tcfg.noop_feasible_loss_anneal_steps))
            floor_frac = float(np.clip(float(tcfg.noop_feasible_loss_floor_frac), 0.0, 1.0))
            noop_loss_coef_now = float(tcfg.noop_feasible_loss_coef) * (floor_frac + (1.0 - floor_frac) * (1.0 - noop_anneal_p))
        else:
            noop_loss_coef_now = float(tcfg.noop_feasible_loss_coef)

        idxs = np.arange(len(rollout))
        for epoch_i in range(tcfg.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, len(idxs), tcfg.minibatch_size):
                mb_idx = idxs[start:start + tcfg.minibatch_size]
                if len(mb_idx) == 0:
                    continue
                mb_idx_t = torch.as_tensor(mb_idx, dtype=torch.long, device=device)

                t_mb_obs = time.perf_counter()
                mb_obs = index_torch_obs_batch(rollout_obs_batch, mb_idx_t)
                perf_minibatch_obs_sec_total += time.perf_counter() - t_mb_obs
                mb_actions = actions_tensor[mb_idx_t]

                t_eval = time.perf_counter()
                new_logp, entropy, value, aux = policy.evaluate_actions(mb_obs, mb_actions, use_mask=not args.wo_mask)
                perf_eval_actions_sec_total += time.perf_counter() - t_eval
                aux_targets = slice_aux_targets(aux_targets_all, mb_idx_t)

                mb_old_logp = old_logp_tensor[mb_idx_t]
                ratio = torch.exp(new_logp - mb_old_logp)
                mb_adv_agent = adv_agent_tensor[mb_idx_t]
                mb_ret = ret_tensor[mb_idx_t]
                mb_old_value = old_value_tensor[mb_idx_t]
                mb_active = active_tensor[mb_idx_t]
                mb_noop_feasible = noop_feasible_tensor[mb_idx_t]

                surr1 = ratio * mb_adv_agent
                surr2 = torch.clamp(ratio, 1.0 - tcfg.clip_ratio, 1.0 + tcfg.clip_ratio) * mb_adv_agent
                pol_loss = -masked_mean(torch.min(surr1, surr2), mb_active)

                value_clipped = mb_old_value + torch.clamp(
                    value - mb_old_value,
                    -tcfg.value_clip_ratio,
                    tcfg.value_clip_ratio,
                )

                # 回报量纲较大时先归一化再计算 value loss，避免 critic 梯度压制 actor。
                if tcfg.normalize_value_loss:
                    value_for_loss = (value - ret_center) / ret_scale
                    value_clipped_for_loss = (value_clipped - ret_center) / ret_scale
                    ret_for_loss = (mb_ret - ret_center) / ret_scale
                else:
                    value_for_loss = value
                    value_clipped_for_loss = value_clipped
                    ret_for_loss = mb_ret

                if tcfg.use_huber_value_loss:
                    value_loss_unclipped = F.huber_loss(
                        value_for_loss,
                        ret_for_loss,
                        reduction="none",
                        delta=tcfg.huber_delta,
                    )
                    value_loss_clipped = F.huber_loss(
                        value_clipped_for_loss,
                        ret_for_loss,
                        reduction="none",
                        delta=tcfg.huber_delta,
                    )
                else:
                    value_loss_unclipped = torch.square(value_for_loss - ret_for_loss)
                    value_loss_clipped = torch.square(value_clipped_for_loss - ret_for_loss)
                val_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                if entropy.dim() == mb_active.dim():
                    ent_loss = masked_mean(entropy, mb_active)
                elif entropy.dim() > 0 and mb_active.dim() == entropy.dim() + 1:
                    # 兼容按样本聚合的 entropy：[B] 与 active:[B, N_agent]
                    sample_w = (mb_active.sum(dim=-1) > 0).to(dtype=entropy.dtype)
                    ent_loss = masked_mean(entropy, sample_w)
                else:
                    ent_loss = entropy.mean() if entropy.dim() > 0 else entropy

                aux_task_start_loss = F.binary_cross_entropy_with_logits(
                    aux["task_start_logits"],
                    aux_targets["task_start_target"],
                    reduction="none",
                )
                aux_task_start_loss = masked_mean(aux_task_start_loss, aux_targets["task_mask"])

                aux_task_ready_loss = F.binary_cross_entropy_with_logits(
                    aux["task_ready_logits"],
                    aux_targets["task_ready_target"],
                    reduction="none",
                )
                aux_task_ready_loss = masked_mean(aux_task_ready_loss, aux_targets["task_mask"])

                aux_pair_loss = F.binary_cross_entropy_with_logits(
                    aux["pair_feasible_logits"],
                    aux_targets["pair_feasible_target"],
                    reduction="none",
                )
                aux_pair_loss = masked_mean(aux_pair_loss, aux_targets["pair_mask"])
                aux_loss = (
                    float(tcfg.aux_task_start_coef) * aux_task_start_loss
                    + float(tcfg.aux_task_ready_coef) * aux_task_ready_loss
                    + float(tcfg.aux_pair_feasible_coef) * aux_pair_loss
                )
                if noop_loss_coef_now > 0.0:
                    action_logprob = F.log_softmax(aux["action_logits"], dim=-1)
                    noop_loss = masked_mean(action_logprob[..., 0].exp(), mb_noop_feasible)
                else:
                    noop_loss = torch.zeros((), dtype=pol_loss.dtype, device=device)

                loss = (
                    pol_loss
                    + tcfg.value_coef * val_loss
                    - entropy_coef_now * ent_loss
                    + float(tcfg.aux_loss_coef) * aux_loss
                    + noop_loss_coef_now * noop_loss
                )

                mb_pol_losses.append(float(pol_loss.detach().cpu().item()))
                mb_val_losses.append(float(val_loss.detach().cpu().item()))
                mb_ent_losses.append(float(ent_loss.detach().cpu().item()))
                mb_aux_losses.append(float(aux_loss.detach().cpu().item()))
                mb_noop_losses.append(float(noop_loss.detach().cpu().item()))
                mb_total_losses.append(float(loss.detach().cpu().item()))
                approx_kl_tensor = mb_old_logp - new_logp
                clip_mask = (torch.abs(ratio - 1.0) > tcfg.clip_ratio).to(dtype=ratio.dtype)
                if approx_kl_tensor.dim() == mb_active.dim():
                    approx_kl_t = masked_mean(approx_kl_tensor, mb_active)
                    clip_frac_t = masked_mean(clip_mask, mb_active)
                else:
                    approx_kl_t = approx_kl_tensor.mean()
                    clip_frac_t = clip_mask.mean()
                approx_kl = float(approx_kl_t.detach().cpu().item())
                clip_frac = float(clip_frac_t.detach().cpu().item())
                mb_approx_kls.append(approx_kl)
                mb_clip_fracs.append(clip_frac)

                t_optim = time.perf_counter()
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(policy.parameters(), tcfg.max_grad_norm)
                optimizer.step()
                perf_optim_sec_total += time.perf_counter() - t_optim

                if tcfg.target_kl > 0.0 and approx_kl > 1.5 * tcfg.target_kl:
                    kl_early_stop = True
                    break
            if kl_early_stop:
                print(
                    f"early stop PPO epochs at update={update_idx + 1}, epoch={epoch_i + 1}, approx_kl={approx_kl:.5f}",
                    flush=True,
                )
                break

            perf_update_sec_total += time.perf_counter() - update_t0

        update_idx += 1
        mean_pol_loss = float(np.mean(mb_pol_losses)) if mb_pol_losses else float("nan")
        mean_val_loss = float(np.mean(mb_val_losses)) if mb_val_losses else float("nan")
        mean_ent_loss = float(np.mean(mb_ent_losses)) if mb_ent_losses else float("nan")
        mean_aux_loss = float(np.mean(mb_aux_losses)) if mb_aux_losses else float("nan")
        mean_noop_loss = float(np.mean(mb_noop_losses)) if mb_noop_losses else float("nan")
        mean_total_loss = float(np.mean(mb_total_losses)) if mb_total_losses else float("nan")
        mean_approx_kl = float(np.mean(mb_approx_kls)) if mb_approx_kls else float("nan")
        mean_clip_frac = float(np.mean(mb_clip_fracs)) if mb_clip_fracs else float("nan")

        if update_idx % log_every == 0:
            mean_ret = np.mean(ep_returns) if ep_returns else float(np.mean(cur_returns))
            recent_reject = float(np.mean(ep_reject_window)) if len(ep_reject_window) > 0 else float("nan")
            recent_truncated = float(np.mean(ep_truncated_window)) if len(ep_truncated_window) > 0 else float("nan")
            recent_violation = float(np.mean(ep_violation_window)) if len(ep_violation_window) > 0 else float("nan")
            recent_noop_feasible = float(np.mean(ep_noop_feasible_window)) if len(ep_noop_feasible_window) > 0 else float("nan")
            train_wall_elapsed = max(time.perf_counter() - train_wall_t0, 1e-9)
            overall_sps = float(step_count) / train_wall_elapsed
            collect_share = perf_collect_sec_total / train_wall_elapsed
            update_share = perf_update_sec_total / train_wall_elapsed
            print(
                f"update={update_idx} step={step_count} mean_return={mean_ret:.3f} "
                f"recent_start_violation_rate={recent_violation:.3f} recent_reject_rate={recent_reject:.3f} "
                f"recent_truncated={recent_truncated:.3f} recent_noop_when_feasible={recent_noop_feasible:.3f} "
                f"noop_loss={mean_noop_loss:.4f} noop_loss_coef={noop_loss_coef_now:.4f} "
                f"goal=compliance_first_then_efficiency "
                f"difficulty_max={cfg.difficulty_max:.3f}",
                flush=True,
            )
            print(
                f"perf update={update_idx} step={step_count} sps={overall_sps:.1f} "
                f"wall={train_wall_elapsed:.1f}s collect={perf_collect_sec_total:.1f}s({collect_share:.0%}) "
                f"update={perf_update_sec_total:.1f}s({update_share:.0%}) "
                f"obs_pack={perf_obs_pack_sec_total:.1f}s act={perf_act_sec_total:.1f}s env_step={perf_env_step_sec_total:.1f}s "
                f"bootstrap={perf_bootstrap_sec_total:.1f}s gae={perf_gae_sec_total:.1f}s rollout_pack={perf_rollout_pack_sec_total:.1f}s "
                f"mb_obs={perf_minibatch_obs_sec_total:.1f}s eval={perf_eval_actions_sec_total:.1f}s optim={perf_optim_sec_total:.1f}s",
                flush=True,
            )

        if tb_writer is not None:
            mean_ret = np.mean(ep_returns) if ep_returns else float(np.mean(cur_returns))
            recent_reject = float(np.mean(ep_reject_window)) if len(ep_reject_window) > 0 else float("nan")
            recent_truncated = float(np.mean(ep_truncated_window)) if len(ep_truncated_window) > 0 else float("nan")
            recent_violation = float(np.mean(ep_violation_window)) if len(ep_violation_window) > 0 else float("nan")
            recent_noop_feasible = float(np.mean(ep_noop_feasible_window)) if len(ep_noop_feasible_window) > 0 else float("nan")
            train_wall_elapsed = max(time.perf_counter() - train_wall_t0, 1e-9)
            overall_sps = float(step_count) / train_wall_elapsed
            recent_base_reward = float(np.mean([
                float(out[4].get("base_reward", float("nan")))
                for out in step_outs if isinstance(out, tuple) and len(out) >= 5
            ])) if len(step_outs) > 0 else float("nan")
            recent_local_credit_mean = float(np.mean([
                float(out[4].get("local_credit_mean", 0.0))
                for out in step_outs if isinstance(out, tuple) and len(out) >= 5
            ])) if len(step_outs) > 0 else 0.0
            recent_local_credit_min = float(np.mean([
                float(out[4].get("local_credit_min", 0.0))
                for out in step_outs if isinstance(out, tuple) and len(out) >= 5
            ])) if len(step_outs) > 0 else 0.0
            recent_local_credit_max = float(np.mean([
                float(out[4].get("local_credit_max", 0.0))
                for out in step_outs if isinstance(out, tuple) and len(out) >= 5
            ])) if len(step_outs) > 0 else 0.0
            recent_credit_reject = float(np.mean([
                float(len(out[4].get("credit_reject_agents", [])))
                for out in step_outs if isinstance(out, tuple) and len(out) >= 5
            ])) if len(step_outs) > 0 else 0.0
            recent_credit_infeasible = float(np.mean([
                float(len(out[4].get("credit_infeasible_agents", [])))
                for out in step_outs if isinstance(out, tuple) and len(out) >= 5
            ])) if len(step_outs) > 0 else 0.0
            recent_credit_noop = float(np.mean([
                float(len(out[4].get("credit_noop_feasible_agents", [])))
                for out in step_outs if isinstance(out, tuple) and len(out) >= 5
            ])) if len(step_outs) > 0 else 0.0

            tb_writer.add_scalar("optimization/policy_loss", mean_pol_loss, update_idx)
            tb_writer.add_scalar("optimization/value_loss", mean_val_loss, update_idx)
            tb_writer.add_scalar("optimization/entropy", mean_ent_loss, update_idx)
            tb_writer.add_scalar("optimization/aux_loss", mean_aux_loss, update_idx)
            tb_writer.add_scalar("optimization/noop_feasible_loss", mean_noop_loss, update_idx)
            tb_writer.add_scalar("optimization/total_loss", mean_total_loss, update_idx)
            tb_writer.add_scalar("optimization/approx_kl", mean_approx_kl, update_idx)
            tb_writer.add_scalar("optimization/clip_fraction", mean_clip_frac, update_idx)
            tb_writer.add_scalar("optimization/entropy_coef", float(entropy_coef_now), update_idx)
            tb_writer.add_scalar("optimization/noop_feasible_loss_coef", float(noop_loss_coef_now), update_idx)
            tb_writer.add_scalar("optimization/value_target_mean", float(ret_center.detach().cpu().item()), update_idx)
            tb_writer.add_scalar("optimization/value_target_std", float(ret_scale.detach().cpu().item()), update_idx)
            tb_writer.add_scalar("optimization/lr", float(optimizer.param_groups[0]["lr"]), update_idx)

            tb_writer.add_scalar("performance/mean_return", float(mean_ret), update_idx)
            tb_writer.add_scalar("performance/recent_reject_rate", recent_reject, update_idx)
            tb_writer.add_scalar("performance/recent_violation_rate", recent_violation, update_idx)
            tb_writer.add_scalar("performance/recent_noop_when_feasible", recent_noop_feasible, update_idx)
            tb_writer.add_text("metric_semantics/episode_recognized_done_rate", _recognized_done_desc(), update_idx)
            tb_writer.add_text("metric_semantics/episode_violation_rate", _violation_rate_desc(), update_idx)
            tb_writer.add_scalar("performance/recent_base_reward", recent_base_reward, update_idx)
            tb_writer.add_scalar("performance/recent_local_credit_mean", recent_local_credit_mean, update_idx)
            tb_writer.add_scalar("performance/recent_local_credit_min", recent_local_credit_min, update_idx)
            tb_writer.add_scalar("performance/recent_local_credit_max", recent_local_credit_max, update_idx)
            tb_writer.add_scalar("performance/recent_credit_reject_agents", recent_credit_reject, update_idx)
            tb_writer.add_scalar("performance/recent_credit_infeasible_agents", recent_credit_infeasible, update_idx)
            tb_writer.add_scalar("performance/recent_credit_noop_feasible_agents", recent_credit_noop, update_idx)
            tb_writer.add_scalar("performance/recent_truncated", recent_truncated, update_idx)
            tb_writer.add_scalar("performance/sps", overall_sps, update_idx)
            tb_writer.add_scalar("performance/wall_time_sec", train_wall_elapsed, update_idx)
            tb_writer.add_scalar("performance_time/collect_sec_total", perf_collect_sec_total, update_idx)
            tb_writer.add_scalar("performance_time/update_sec_total", perf_update_sec_total, update_idx)
            tb_writer.add_scalar("performance_time/obs_pack_sec_total", perf_obs_pack_sec_total, update_idx)
            tb_writer.add_scalar("performance_time/act_sec_total", perf_act_sec_total, update_idx)
            tb_writer.add_scalar("performance_time/env_step_sec_total", perf_env_step_sec_total, update_idx)
            tb_writer.add_scalar("performance_time/bootstrap_sec_total", perf_bootstrap_sec_total, update_idx)
            tb_writer.add_scalar("performance_time/gae_sec_total", perf_gae_sec_total, update_idx)
            tb_writer.add_scalar("performance_time/rollout_pack_sec_total", perf_rollout_pack_sec_total, update_idx)
            tb_writer.add_scalar("performance_time/minibatch_obs_sec_total", perf_minibatch_obs_sec_total, update_idx)
            tb_writer.add_scalar("performance_time/eval_actions_sec_total", perf_eval_actions_sec_total, update_idx)
            tb_writer.add_scalar("performance_time/optim_sec_total", perf_optim_sec_total, update_idx)
            tb_writer.add_scalar("progress/step", float(step_count), update_idx)

        if len(ep_return_window) == tcfg.convergence_window:
            window_return = float(np.mean(ep_return_window))
            window_return_std = float(np.std(ep_return_window))
            window_reject = float(np.mean(ep_reject_window))
            window_violation = float(np.mean(ep_violation_window)) if len(ep_violation_window) > 0 else float("inf")

            if window_return > best_window_return + tcfg.convergence_min_delta:
                best_window_return = window_return
                stale_updates = 0
            else:
                stale_updates += 1

            if (
                stale_updates >= tcfg.convergence_patience
                and window_return_std <= tcfg.convergence_return_std_max
                and window_reject <= tcfg.convergence_max_reject_rate
                and window_violation <= tcfg.convergence_max_violation_rate
            ):
                target_stop_step = min(tcfg.total_steps, step_count + tcfg.post_convergence_steps)
                pending_stop_step = target_stop_step if pending_stop_step is None else min(pending_stop_step, target_stop_step)
                if not stop_reason:
                    stop_reason = "episode-window convergence"
                print(
                    "converged: "
                    f"update={update_idx} step={step_count} "
                    f"window_return={window_return:.3f} window_return_std={window_return_std:.3f} "
                    f"window_reject={window_reject:.3f} "
                    f"window_violation={window_violation:.3f} stop_at={pending_stop_step}",
                    flush=True,
                )

        if pending_stop_step is not None and step_count >= pending_stop_step:
            converged = True
            if not stop_reason:
                stop_reason = f"post-convergence steps reached ({pending_stop_step})"

        if args.save_every_updates > 0 and (update_idx % args.save_every_updates == 0):
            runtime_state = {
                "cur_returns": cur_returns,
                "cur_commit_attempts": cur_commit_attempts,
                "cur_noop_feasible": cur_noop_feasible,
                "cur_feasible_agents": cur_feasible_agents,
                "episode_idx": episode_idx,
            }
            save_checkpoint(
                args.save_path,
                policy,
                optimizer,
                cfg,
                tcfg,
                args,
                step_count,
                update_idx,
                env_states=env_pool.get_states(),
                obs_list=obs_list,
                runtime_state=runtime_state,
            )
            print(f"saved periodic checkpoint: {args.save_path}", flush=True)

        if int(tcfg.eval_fixed_every_steps) > 0 and step_count >= next_fixed_eval_step:
            try:
                fixed_metrics = evaluate_policy_on_fixed_set(
                    policy=policy,
                    fixed_set_dir=str(tcfg.eval_fixed_set_dir),
                    max_scenes=int(tcfg.eval_fixed_max_scenes),
                    device=device,
                    wo_mask=bool(args.wo_mask),
                    wo_commitment=bool(args.wo_commitment),
                    greedy_nonnoop_if_feasible=bool(getattr(args, "eval_greedy_nonnoop_if_feasible", True)),
                )
                metric_name = str(tcfg.best_fixed_metric)
                fixed_score = -float(fixed_metrics["violation_rate"]) if metric_name == "neg_violation" else float(fixed_metrics[metric_name])
                print(
                    "fixed_eval: "
                    f"step={step_count} update={update_idx} score[{metric_name}]={fixed_score:.4f} "
                    f"return={fixed_metrics['return']:.3f} "
                    f"recognized_done_rate={fixed_metrics['recognized_done_rate']:.3f} "
                    f"recognized_done_tasks={fixed_metrics['recognized_done_tasks']:.3f} "
                    f"violation_rate={fixed_metrics['violation_rate']:.3f} "
                    f"truncated={fixed_metrics['truncated']:.3f}",
                    flush=True,
                )
                if tb_writer is not None:
                    for k, v in fixed_metrics.items():
                        tb_writer.add_scalar(f"fixed_eval/{k}", float(v), update_idx)
                    tb_writer.add_scalar(f"fixed_eval/score_{metric_name}", float(fixed_score), update_idx)
                if bool(tcfg.save_best_fixed_eval) and fixed_score > best_fixed_score + float(tcfg.best_fixed_min_delta):
                    best_fixed_score = fixed_score
                    save_checkpoint(
                        best_ckpt_path,
                        policy,
                        optimizer,
                        cfg,
                        tcfg,
                        args,
                        step_count,
                        update_idx,
                        env_states=env_pool.get_states(),
                        obs_list=obs_list,
                    )
                    sidecar_path = f"{best_ckpt_path}.best.json"
                    with open(sidecar_path, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "step_count": int(step_count),
                                "update_idx": int(update_idx),
                                "metric": metric_name,
                                "score": float(fixed_score),
                                "metrics": fixed_metrics,
                            },
                            f,
                            ensure_ascii=False,
                            indent=2,
                        )
                    print(f"saved best fixed-eval checkpoint: {best_ckpt_path}", flush=True)
            except Exception as exc:
                print(f"warning: fixed_eval failed at step={step_count}: {exc}", flush=True)
            while next_fixed_eval_step <= step_count:
                next_fixed_eval_step += int(tcfg.eval_fixed_every_steps)

        if tb_writer is not None and episode_idx > 0:
            tb_writer.flush()

    runtime_state = {
        "cur_returns": cur_returns,
        "cur_commit_attempts": cur_commit_attempts,
        "cur_noop_feasible": cur_noop_feasible,
        "cur_feasible_agents": cur_feasible_agents,
        "episode_idx": episode_idx,
    }
    if converged:
        print(f"training stopped early at step={step_count}, update={update_idx}, reason={stop_reason}", flush=True)
    else:
        print(f"training stopped at total_steps limit: step={step_count}, update={update_idx}", flush=True)
    if tb_writer is not None:
        tb_writer.add_text("run/stop_reason", stop_reason if stop_reason else "total_steps_limit", update_idx)
        tb_writer.add_scalar("run/final_step", float(step_count), update_idx)
        tb_writer.add_scalar("run/final_update", float(update_idx), update_idx)
    save_checkpoint(
        args.save_path,
        policy,
        optimizer,
        cfg,
        tcfg,
        args,
        step_count,
        update_idx,
        env_states=env_pool.get_states(),
        obs_list=obs_list,
        runtime_state=runtime_state,
    )
    print(f"saved checkpoint: {args.save_path}")

    env_pool.close()

    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()


if __name__ == "__main__":
    main()
