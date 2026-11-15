from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from env.fixed_set_loader import (
    apply_config_overrides_to_env,
    apply_policy_execution_overrides_to_env,
    load_fixed_env_from_path,
    materialize_fixed_dynamic_arrivals,
    resolve_fixed_scene_paths,
)
from env.mrta_config import MRTAConfig
from env.mrta_env import MRTAEnv
from mrta_policy import MRTAObs, MRTAPolicy
from torch_runtime import resolve_torch_device
from train_config import EvalConfig


def make_legacy_baseline_adapter(method: str, timeout_sec: float = 10.0):
    try:
        from baseline.legacy_adapter import LegacyBaselineAdapter
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "baseline evaluation requires baseline/legacy_adapter.py and its dependencies"
        ) from exc
    return LegacyBaselineAdapter(method=method, timeout_sec=float(timeout_sec))


def configure_torch_cpu_threads(num_threads: int, num_interop_threads: int):
    cpu_count = int(os.cpu_count() or 1)
    chosen_threads = int(num_threads) if int(num_threads) > 0 else min(4, cpu_count)
    chosen_interop = int(num_interop_threads) if int(num_interop_threads) > 0 else 1
    try:
        torch.set_num_threads(max(1, chosen_threads))
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(max(1, chosen_interop))
    except Exception:
        pass


def infer_hidden_dim(ckpt: Dict, default: int = 128) -> int:
    tcfg = ckpt.get("train_config", {}) if isinstance(ckpt, dict) else {}
    if isinstance(tcfg, dict) and "hidden_dim" in tcfg:
        return int(tcfg["hidden_dim"])
    return int(default)


def infer_node_dim(ckpt: Dict) -> Optional[int]:
    model_state = ckpt.get("model", {}) if isinstance(ckpt, dict) else {}
    if not isinstance(model_state, dict):
        return None
    for key in ("task_encoder.input_proj.weight", "task_encoder.net.0.weight"):
        weight = model_state.get(key)
        if weight is not None and getattr(weight, "ndim", 0) == 2:
            return int(weight.shape[1])
    return None


def infer_agent_dim(ckpt: Dict) -> Optional[int]:
    if not isinstance(ckpt, dict):
        return None
    model_state = ckpt.get("model", {})
    if not isinstance(model_state, dict):
        return None
    hidden_dim = infer_hidden_dim(ckpt)
    for key in ("actor_agent_shallow.0.weight", "critic_agent_encoder.0.weight"):
        weight = model_state.get(key)
        if weight is not None and getattr(weight, "ndim", 0) == 2:
            return max(0, int(weight.shape[1]) - int(hidden_dim))
    return None


def infer_pair_dim(ckpt: Dict) -> Optional[int]:
    model_state = ckpt.get("model", {}) if isinstance(ckpt, dict) else {}
    if not isinstance(model_state, dict):
        return None
    weight = model_state.get("critic_pair_encoder.0.weight")
    if weight is not None and getattr(weight, "ndim", 0) == 2:
        return int(weight.shape[1])
    weight = model_state.get("actor_head.0.weight")
    if weight is not None and getattr(weight, "ndim", 0) == 2:
        return max(0, int(weight.shape[1]) - int(infer_hidden_dim(ckpt)) * 3)
    return None


def _adapt_last_dim(array: np.ndarray, target_dim: Optional[int]) -> np.ndarray:
    if target_dim is None:
        return array
    current_dim = int(array.shape[-1])
    target_dim = int(target_dim)
    if current_dim == target_dim:
        return array
    if current_dim > target_dim:
        return array[..., :target_dim]
    pad_shape = (*array.shape[:-1], target_dim - current_dim)
    pad = np.zeros(pad_shape, dtype=array.dtype)
    return np.concatenate([array, pad], axis=-1)


def adapt_obs_feature_dims(
    obs: Dict,
    target_node_dim: Optional[int] = None,
    target_agent_dim: Optional[int] = None,
    target_pair_dim: Optional[int] = None,
) -> Dict:
    adapted = dict(obs)
    if target_agent_dim is not None:
        adapted["agent_feats"] = _adapt_last_dim(obs["agent_feats"], target_agent_dim)
    if target_pair_dim is not None:
        adapted["pair_feats"] = _adapt_last_dim(obs["pair_feats"], target_pair_dim)
    if target_node_dim is not None:
        task_graph = dict(obs["task_graph"])
        task_graph["node_features"] = _adapt_last_dim(obs["task_graph"]["node_features"], target_node_dim)
        adapted["task_graph"] = task_graph
    return adapted


def adapt_obs_agent_dim(obs: Dict, target_agent_dim: Optional[int]) -> Dict:
    return adapt_obs_feature_dims(obs, target_agent_dim=target_agent_dim)


def to_torch_obs(obs, device) -> MRTAObs:
    edge_type_np = obs["task_graph"]["edge_type"]
    edge_mask_np = obs["task_graph"].get("edge_mask", np.ones((edge_type_np.shape[0],), dtype=np.float32))
    task_graph = {
        "node_features": torch.tensor(obs["task_graph"]["node_features"], dtype=torch.float32, device=device),
        "edge_index": torch.tensor(obs["task_graph"]["edge_index"], dtype=torch.long, device=device),
        "edge_type": torch.tensor(obs["task_graph"]["edge_type"], dtype=torch.long, device=device),
        "edge_lag": torch.tensor(obs["task_graph"]["edge_lag"], dtype=torch.float32, device=device),
        "edge_mask": torch.tensor(edge_mask_np, dtype=torch.float32, device=device),
        "node_mask": torch.tensor(obs["task_graph"]["node_mask"], dtype=torch.float32, device=device),
    }
    if "node_t_start" in obs["task_graph"]:
        task_graph["node_t_start"] = torch.tensor(obs["task_graph"]["node_t_start"], dtype=torch.float32, device=device)
        task_graph["node_t_finish_phys"] = torch.tensor(obs["task_graph"]["node_t_finish_phys"], dtype=torch.float32, device=device)
    return {
        "task_graph": task_graph,
        "agent_feats": torch.tensor(obs["agent_feats"], dtype=torch.float32, device=device),
        "pair_feats": torch.tensor(obs["pair_feats"], dtype=torch.float32, device=device),
        "action_mask": torch.tensor(obs["action_mask"], dtype=torch.float32, device=device),
        "global_time": torch.tensor(obs["global_time"], dtype=torch.float32, device=device),
    }


def perturb_policy_graph(obs: Dict, mode: str, rng: np.random.Generator) -> Dict:
    """Perturb only the policy graph input while leaving the environment unchanged."""
    if mode == "original":
        return obs
    out = dict(obs)
    graph = {key: np.array(value, copy=True) for key, value in obs["task_graph"].items()}
    out["task_graph"] = graph
    edge_mask = graph.get("edge_mask", np.ones_like(graph["edge_type"], dtype=np.float32))
    valid = edge_mask > 0.5
    if mode == "drop_edges":
        graph["edge_mask"] = np.zeros_like(edge_mask, dtype=np.float32)
    elif mode == "shuffle_edges":
        destinations = graph["edge_index"][1, valid].copy()
        rng.shuffle(destinations)
        graph["edge_index"][1, valid] = destinations
    elif mode == "shuffle_relations":
        relations = graph["edge_type"][valid].copy()
        rng.shuffle(relations)
        graph["edge_type"][valid] = relations
    else:
        raise ValueError(f"unknown policy graph perturbation: {mode}")
    return out


def run_episode(
    env: MRTAEnv,
    policy: Optional[MRTAPolicy] = None,
    device=None,
    deterministic: bool = True,
    policy_decode: str = "greedy",
    policy_explore_eps: float = 0.0,
    greedy_nonnoop_if_feasible: bool = False,
    wo_mask: bool = False,
    baseline_adapter: Optional[LegacyBaselineAdapter] = None,
    initial_obs=None,
    graph_perturbation: str = "original",
    graph_perturbation_seed: int = 0,
    episode_timeout_sec: float = 0.0,
):
    episode_t0 = time.perf_counter()
    obs = env.reset() if initial_obs is None else initial_obs
    total_reward = 0.0
    infer_time = 0.0
    decision_calls = 0
    feasible_action_count_sum = 0.0
    feasible_agent_count_sum = 0.0
    forced_noop_count_sum = 0.0
    noop_when_feasible_count_sum = 0.0
    graph_rng = np.random.default_rng(int(graph_perturbation_seed))
    if policy is None and baseline_adapter is not None:
        baseline_adapter.reset_episode()

    while True:
        if policy is None:
            if baseline_adapter is None:
                baseline_adapter = make_legacy_baseline_adapter(method="cbs_ta_ptc")
            t0 = time.perf_counter()
            actions = baseline_adapter.act(env)
            infer_time += time.perf_counter() - t0
            decision_calls += 1
        else:
            step_deterministic = bool(deterministic)
            if policy_decode == "sample":
                step_deterministic = False
            elif policy_decode == "greedy" and policy_explore_eps > 0.0:
                step_deterministic = not (float(np.random.random()) < float(policy_explore_eps))
            if device is not None and getattr(device, "type", None) == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                eval_dims = getattr(policy, "_eval_obs_dims", {})
                obs_for_policy = adapt_obs_feature_dims(
                    obs,
                    target_node_dim=eval_dims.get("node_dim"),
                    target_agent_dim=eval_dims.get("agent_dim"),
                    target_pair_dim=eval_dims.get("pair_dim"),
                )
                obs_for_policy = perturb_policy_graph(obs_for_policy, graph_perturbation, graph_rng)
                torch_obs = to_torch_obs(obs_for_policy, device)
                if step_deterministic and bool(greedy_nonnoop_if_feasible) and not bool(wo_mask):
                    logits, _ = policy.forward(torch_obs, use_mask=True)
                    act = torch.argmax(logits, dim=-1)
                    mask = torch_obs["action_mask"]
                    if logits.dim() == 2:
                        feasible_agent = (mask[:, 1:] > 0.5).any(dim=-1)
                        noop_when_feasible = (act == 0) & feasible_agent
                        if torch.any(noop_when_feasible):
                            task_logits = logits[:, 1:].masked_fill(mask[:, 1:] < 0.5, -1e9)
                            best_task_act = torch.argmax(task_logits, dim=-1) + 1
                            act = torch.where(noop_when_feasible, best_task_act, act)
                    else:
                        feasible_agent = (mask[:, :, 1:] > 0.5).any(dim=-1)
                        noop_when_feasible = (act == 0) & feasible_agent
                        if torch.any(noop_when_feasible):
                            task_logits = logits[:, :, 1:].masked_fill(mask[:, :, 1:] < 0.5, -1e9)
                            best_task_act = torch.argmax(task_logits, dim=-1) + 1
                            act = torch.where(noop_when_feasible, best_task_act, act)
                else:
                    act, _, _, _ = policy.act(torch_obs, deterministic=step_deterministic, use_mask=not wo_mask)
            if device is not None and getattr(device, "type", None) == "cuda":
                torch.cuda.synchronize(device)
            infer_time += time.perf_counter() - t0
            decision_calls += 1
            actions = {i: int(act[i].item()) for i in range(act.shape[0])}

        obs, rewards, terminated, truncated, info = env.step(actions)
        total_reward += float(np.mean(list(rewards.values())))
        feasible_action_count_sum += float(info.get("feasible_action_count", 0.0))
        feasible_agent_count_sum += float(info.get("feasible_agent_count", 0.0))
        forced_noop_count_sum += float(info.get("forced_noop_count", 0.0))
        noop_when_feasible_count_sum += float(info.get("noop_when_feasible_count", 0.0))
        # Per-episode wall-clock cutoff for slow traditional baselines on large-scale
        # scenes. When the budget is exceeded we force truncation so the episode ends
        # here; metrics below are still computed from the current env info, so the
        # partial progress (recognized_done_rate) is recorded rather than dropped.
        wallclock_timeout = False
        if (
            policy is None
            and float(episode_timeout_sec) > 0.0
            and not (terminated or truncated)
            and (time.perf_counter() - episode_t0) >= float(episode_timeout_sec)
        ):
            truncated = True
            wallclock_timeout = True
        if terminated or truncated:
            recognized_done_tasks = int(info.get("recognized_done_tasks", 0))
            total_tasks = max(len(env.tasks), 1)
            recognized_done_rate_fallback = float(recognized_done_tasks) / float(total_tasks)
            violation_rate = float(np.clip(float(info.get("violation_rate", 0.0)), 0.0, 1.0))
            recognized_done_rate = float(info.get("recognized_done_rate", recognized_done_rate_fallback))
            throughput = float(recognized_done_tasks) / max(float(info["makespan"]), 1.0)
            metrics = {
                "makespan": info["makespan"],
                "generalized_makespan": float(info.get("generalized_makespan", 0.0)),
                "success_rate": recognized_done_rate,
                # Keep key for backward compatibility, but avoid legacy 0/1 strict gate.
                "success": recognized_done_rate,
                "strict_success": recognized_done_rate,
                "return": total_reward,
                "recognized_done_tasks": recognized_done_tasks,
                "recognized_done_rate": recognized_done_rate,
                "waiting_cost": info["waiting_cost"],
                "logical_wait_cost": float(info.get("logical_wait_cost", 0.0)),
                "rejected_commitments": info["rejected_commitments"],
                "missed_ss_window": info["missed_ss_window"],
                "violation_rate": violation_rate,
                "truncated": float(truncated),
                "completed_tasks": recognized_done_tasks,
                "total_tasks": total_tasks,
                "throughput": throughput,
                "infer_time": infer_time,
                "decision_time": infer_time,
                "decision_calls": float(decision_calls),
                "mean_decision_time": infer_time / max(decision_calls, 1),
                "mean_feasible_action_count": feasible_action_count_sum / max(decision_calls, 1),
                "mean_feasible_agent_count": feasible_agent_count_sum / max(decision_calls, 1),
                "mean_forced_noop_count": forced_noop_count_sum / max(decision_calls, 1),
                "mean_noop_when_feasible_count": noop_when_feasible_count_sum / max(decision_calls, 1),
                "noop_when_feasible_ratio": noop_when_feasible_count_sum / max(feasible_agent_count_sum, 1.0),
                "episode_runtime": time.perf_counter() - episode_t0,
                "scenario_difficulty": float(info.get("scenario_difficulty", env.current_scenario_difficulty)),
                "time_horizon": float(getattr(env.config, "max_steps", 0)),
                "config_use_policy_commitment_logic": float(bool(env.config.use_policy_commitment_logic)),
                "config_capacity_topk_enabled": float(bool(env.config.capacity_topk_enabled)),
                "config_ordered_commitment": float(bool(env.config.ordered_commitment)),
                "config_dynamic_arrival_every": float(env.config.dynamic_arrival_every),
                "wallclock_timeout": float(wallclock_timeout),
            }
            if policy is None and baseline_adapter is not None:
                metrics.update(baseline_adapter.get_episode_stats())
            return metrics


def run_episode_planned_baseline(
    env: MRTAEnv,
    baseline_adapter: LegacyBaselineAdapter,
    reset_env: bool = True,
):
    episode_t0 = time.perf_counter()
    # Fixed-set loaders return an initialized scene. Resetting here would replace
    # that scene with a newly generated one and invalidate paired comparisons.
    if reset_env:
        env.reset()
    t0 = time.perf_counter()
    plan = baseline_adapter.plan(env)
    infer_time = time.perf_counter() - t0
    baseline_plan_tasks = float(sum(len(v) for v in plan.task_sequence_by_agent.values()))
    total_reward = 0.0
    decision_calls = 0
    feasible_action_count_sum = 0.0
    feasible_agent_count_sum = 0.0
    forced_noop_count_sum = 0.0
    noop_when_feasible_count_sum = 0.0

    while True:
        actions = baseline_adapter.act_from_plan(env, plan)
        _, rewards, terminated, truncated, info = env.step(actions)
        total_reward += float(np.mean(list(rewards.values())))
        decision_calls += 1
        feasible_action_count_sum += float(info.get("feasible_action_count", 0.0))
        feasible_agent_count_sum += float(info.get("feasible_agent_count", 0.0))
        forced_noop_count_sum += float(info.get("forced_noop_count", 0.0))
        noop_when_feasible_count_sum += float(info.get("noop_when_feasible_count", 0.0))
        if terminated or truncated:
            recognized_done_tasks = int(info.get("recognized_done_tasks", 0))
            total_tasks = max(len(env.tasks), 1)
            recognized_done_rate_fallback = float(recognized_done_tasks) / float(total_tasks)
            violation_rate = float(np.clip(float(info.get("violation_rate", 0.0)), 0.0, 1.0))
            recognized_done_rate = float(info.get("recognized_done_rate", recognized_done_rate_fallback))
            throughput = float(recognized_done_tasks) / max(float(info["makespan"]), 1.0)
            return {
                "makespan": info["makespan"],
                "generalized_makespan": float(info.get("generalized_makespan", 0.0)),
                "success_rate": recognized_done_rate,
                "success": recognized_done_rate,
                "strict_success": recognized_done_rate,
                "return": total_reward,
                "recognized_done_tasks": recognized_done_tasks,
                "recognized_done_rate": recognized_done_rate,
                "waiting_cost": info["waiting_cost"],
                "logical_wait_cost": float(info.get("logical_wait_cost", 0.0)),
                "rejected_commitments": info["rejected_commitments"],
                "missed_ss_window": info["missed_ss_window"],
                "violation_rate": violation_rate,
                "truncated": float(truncated),
                "completed_tasks": recognized_done_tasks,
                "total_tasks": total_tasks,
                "throughput": throughput,
                "infer_time": infer_time,
                "decision_time": infer_time,
                "decision_calls": 1.0,
                "mean_decision_time": infer_time,
                "mean_feasible_action_count": feasible_action_count_sum / max(decision_calls, 1),
                "mean_feasible_agent_count": feasible_agent_count_sum / max(decision_calls, 1),
                "mean_forced_noop_count": forced_noop_count_sum / max(decision_calls, 1),
                "mean_noop_when_feasible_count": noop_when_feasible_count_sum / max(decision_calls, 1),
                "noop_when_feasible_ratio": noop_when_feasible_count_sum / max(feasible_agent_count_sum, 1.0),
                "episode_runtime": time.perf_counter() - episode_t0,
                "scenario_difficulty": float(info.get("scenario_difficulty", env.current_scenario_difficulty)),
                "time_horizon": float(getattr(env.config, "max_steps", 0)),
                "config_use_policy_commitment_logic": float(bool(env.config.use_policy_commitment_logic)),
                "config_capacity_topk_enabled": float(bool(env.config.capacity_topk_enabled)),
                "config_ordered_commitment": float(bool(env.config.ordered_commitment)),
                "config_dynamic_arrival_every": float(env.config.dynamic_arrival_every),
                "baseline_plan_tasks": baseline_plan_tasks,
            }


def aggregate_metrics(metrics: List[Dict]):
    out = {}
    for k in metrics[0].keys():
        vals = np.array([m[k] for m in metrics], dtype=np.float64)
        out[k] = (float(np.mean(vals)), float(np.std(vals)))
    return out


def _safe_run_name(name: str) -> str:
    valid = []
    for ch in str(name):
        if ch.isalnum() or ch in {"-", "_", "."}:
            valid.append(ch)
        else:
            valid.append("_")
    cleaned = "".join(valid).strip("._")
    return cleaned if cleaned else "eval_run"


def _default_run_name(args) -> str:
    eval_mode = "fixed" if args.fixed_set_path else str(args.setting)
    method = resolve_method_alias(args)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{eval_mode}_{method}"


def resolve_method_alias(args) -> str:
    user_alias = str(getattr(args, "method_alias", "")).strip()
    if user_alias:
        return _safe_run_name(user_alias)
    if bool(getattr(args, "baseline", False)):
        return _safe_run_name(str(getattr(args, "baseline_method", "baseline")))
    return "ours"


def _write_records_csv(path: Path, records: List[Dict]):
    if not records:
        return
    keys = sorted({k for r in records for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in records:
            row = {}
            for k in keys:
                v = r.get(k, "")
                if isinstance(v, Path):
                    row[k] = v.as_posix()
                else:
                    row[k] = v
            writer.writerow(row)


def _write_records_csv_atomic(path: Path, records: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    _write_records_csv(tmp_path, records)
    os.replace(tmp_path, path)


def _artifact_output_dir(args) -> Path:
    run_name = _safe_run_name(args.output_tag) if str(args.output_tag).strip() else _default_run_name(args)
    return Path(args.output_dir) / run_name


def _scene_resume_key(path_value) -> str:
    path = Path(str(path_value))
    try:
        return path.resolve().as_posix()
    except OSError:
        return path.as_posix()


def _load_incremental_records(path: Path, method_alias: str) -> List[Dict]:
    if not path.is_file():
        return []
    records: List[Dict] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            scene_path = str(row.get("scene_path", "")).strip()
            if not scene_path:
                continue
            recorded_method = str(row.get("method", method_alias)).strip()
            if recorded_method and recorded_method != method_alias:
                raise ValueError(
                    f"resume artifact method mismatch: expected={method_alias} found={recorded_method} path={path}"
                )
            records.append(dict(row))
    return records


def _write_aggregate_csv(path: Path, agg: Dict[str, Tuple[float, float]]):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std"])
        for k, (m, st) in agg.items():
            writer.writerow([k, f"{m:.10f}", f"{st:.10f}"])


def load_oracle_metrics(path: str) -> Dict[str, Dict[str, float]]:
    oracle_path = Path(str(path).strip())
    if not oracle_path.is_file():
        raise FileNotFoundError(f"oracle metrics csv not found: {oracle_path}")
    out: Dict[str, Dict[str, float]] = {}
    with oracle_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            scene_path = str(row.get("scene_path", "")).strip()
            if not scene_path:
                continue
            max_done = float(row.get("max_recognized_done_tasks", 0.0) or 0.0)
            total_tasks = float(row.get("total_tasks", 0.0) or 0.0)
            oracle_row = {
                "oracle_max_recognized_done_tasks": max_done,
                "oracle_upper_bound_tasks": float(row.get("upper_bound_tasks", max_done) or max_done),
                "oracle_total_tasks": total_tasks,
                "oracle_max_recognized_done_rate": float(row.get("max_recognized_done_rate", 0.0) or 0.0),
                "oracle_is_optimal": float(row.get("is_optimal", 0.0) or 0.0),
            }
            out[scene_path] = oracle_row
            try:
                out[Path(scene_path).resolve().as_posix()] = oracle_row
            except Exception:
                pass
    if not out:
        raise ValueError(f"oracle metrics csv contains no usable scene_path rows: {oracle_path}")
    return out


def attach_oracle_metrics(metrics: Dict, scene_path: Path, oracle_by_scene: Optional[Dict[str, Dict[str, float]]]) -> Dict:
    if not oracle_by_scene:
        return metrics
    key = scene_path.as_posix()
    oracle = oracle_by_scene.get(key)
    if oracle is None:
        oracle = oracle_by_scene.get(scene_path.resolve().as_posix())
    if oracle is None:
        raise KeyError(f"scene path not found in oracle metrics: {key}")
    out = dict(metrics)
    out.update(oracle)
    denom = float(oracle.get("oracle_max_recognized_done_tasks", 0.0))
    done = float(out.get("recognized_done_tasks", out.get("completed_tasks", 0.0)))
    eval_total_tasks = float(out.get("total_tasks", 0.0))
    oracle_total_tasks = float(oracle.get("oracle_total_tasks", 0.0))
    if abs(eval_total_tasks - oracle_total_tasks) > 1e-9:
        raise ValueError(
            "oracle and evaluation task graphs differ; refusing invalid normalized metric: "
            f"scene={key} eval_total_tasks={eval_total_tasks:.0f} oracle_total_tasks={oracle_total_tasks:.0f}"
        )
    is_optimal = float(oracle.get("oracle_is_optimal", 0.0)) >= 0.5
    certified_upper = float(oracle.get("oracle_upper_bound_tasks", denom))
    if is_optimal and done > denom + 1e-9:
        raise ValueError(
            "oracle lower than environment-recognized completion; refusing invalid normalized metric: "
            f"scene={key} recognized_done={done:.0f} oracle_max_done={denom:.0f}"
        )
    out["oracle_metric_valid"] = float(is_optimal)
    # Exact oracle ratios are defined only when CP-SAT proves optimality.
    out["cpsat_relative_done"] = done / denom if is_optimal and denom > 0.0 else float("nan")
    out["oracle_done_gap"] = denom - done if is_optimal else float("nan")
    out["cpsat_upper_bound_tasks"] = certified_upper
    out["cpsat_upper_bound_relative_done"] = done / certified_upper if certified_upper > 0.0 else 0.0
    return out


def has_oracle_metrics_for_scene(scene_path: Path, oracle_by_scene: Optional[Dict[str, Dict[str, float]]]) -> bool:
    if not oracle_by_scene:
        return True
    if scene_path.as_posix() in oracle_by_scene:
        return True
    return scene_path.resolve().as_posix() in oracle_by_scene


def _save_visualizations(records: List[Dict], out_dir: Path) -> List[Path]:
    if not records:
        return []

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"visualization skipped: matplotlib unavailable ({exc})", flush=True)
        return []

    def arr(key: str):
        return np.array([float(r.get(key, np.nan)) for r in records], dtype=np.float64)

    saved: List[Path] = []
    idx = np.arange(len(records), dtype=np.int64)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    line_metrics = [
        ("return", "Return"),
        ("recognized_done_tasks", "Recognized Done Tasks"),
        ("recognized_done_rate", "Recognized Done Rate"),
        ("violation_rate", "Violation Rate"),
    ]
    for ax, (k, title) in zip(axes.ravel(), line_metrics):
        y = arr(k)
        ax.plot(idx, y, linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel("Eval Sample Index")
        ax.grid(alpha=0.25)
    curve_path = out_dir / "curves_over_samples.png"
    fig.savefig(curve_path, dpi=140)
    plt.close(fig)
    saved.append(curve_path)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    hist_metrics = [
        ("makespan", "Makespan"),
        ("rejected_commitments", "Rejected Commitments"),
        ("scenario_difficulty", "Scenario Difficulty"),
    ]
    for ax, (k, title) in zip(axes.ravel(), hist_metrics):
        y = arr(k)
        y = y[np.isfinite(y)]
        if y.size > 0:
            ax.hist(y, bins=min(25, max(8, int(np.sqrt(y.size) + 1))), alpha=0.85)
        ax.set_title(title)
        ax.grid(alpha=0.25)
    hist_path = out_dir / "histograms.png"
    fig.savefig(hist_path, dpi=140)
    plt.close(fig)
    saved.append(hist_path)

    has_seed = any("seed" in r for r in records)
    if has_seed:
        seed_to_ret: Dict[int, List[float]] = {}
        seed_to_rec_done_rate: Dict[int, List[float]] = {}
        for r in records:
            if "seed" not in r:
                continue
            sid = int(r["seed"])
            seed_to_ret.setdefault(sid, []).append(float(r.get("return", np.nan)))
            seed_to_rec_done_rate.setdefault(sid, []).append(float(r.get("recognized_done_rate", np.nan)))

        if seed_to_ret:
            seeds = sorted(seed_to_ret.keys())
            mean_ret = [float(np.nanmean(np.array(seed_to_ret[s], dtype=np.float64))) for s in seeds]
            mean_rec_done_rate = [float(np.nanmean(np.array(seed_to_rec_done_rate[s], dtype=np.float64))) for s in seeds]

            fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
            axes[0].bar([str(s) for s in seeds], mean_ret)
            axes[0].set_title("Mean Return by Seed")
            axes[0].set_xlabel("Seed")
            axes[0].grid(alpha=0.25, axis="y")

            axes[1].bar([str(s) for s in seeds], mean_rec_done_rate)
            axes[1].set_title("Mean Recognized Done Rate by Seed")
            axes[1].set_xlabel("Seed")
            axes[1].grid(alpha=0.25, axis="y")

            by_seed_path = out_dir / "by_seed_summary.png"
            fig.savefig(by_seed_path, dpi=140)
            plt.close(fig)
            saved.append(by_seed_path)

    return saved


def save_eval_artifacts(records: List[Dict], agg: Dict[str, Tuple[float, float]], args, ckpt_path: str):
    if not bool(args.save_artifacts):
        return None

    out_dir = _artifact_output_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    records_csv = out_dir / "episode_metrics.csv"
    agg_csv = out_dir / "aggregate_metrics.csv"
    summary_json = out_dir / "summary.json"

    _write_records_csv(records_csv, records)
    _write_aggregate_csv(agg_csv, agg)

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(ckpt_path),
        "method_alias": resolve_method_alias(args),
        "setting": str(args.setting),
        "fixed_set_path": str(args.fixed_set_path),
        "baseline": bool(args.baseline),
        "baseline_method": str(args.baseline_method),
        "episodes": int(args.episodes),
        "seeds": int(args.seeds),
        "num_records": int(len(records)),
        "aggregate": {k: {"mean": float(v[0]), "std": float(v[1])} for k, v in agg.items()},
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    figure_paths: List[Path] = []
    if bool(args.plot):
        figure_paths = _save_visualizations(records=records, out_dir=out_dir)

    return {
        "out_dir": out_dir,
        "records_csv": records_csv,
        "agg_csv": agg_csv,
        "summary_json": summary_json,
        "figures": figure_paths,
    }


def make_env_config(setting: str):
    cfg = MRTAConfig()
    if setting == "iid":
        return cfg
    if setting == "ood_big":
        cfg.init_tasks_range = (40, 60)
        cfg.max_tasks = 120
        cfg.num_agents = 20
    elif setting == "ood_dense":
        cfg.edge_density = 0.3
        cfg.relation_ratio = {"FS": 0.3, "SS": 0.2, "FF": 0.25, "SF": 0.25}
    elif setting == "ood_dynamic":
        cfg.dynamic_arrival_every = 5
        cfg.dynamic_arrival_range = (1, 3)
    elif setting == "ood_hetero":
        cfg.num_agents = 16
        cfg.task_k_range = (2, 5)
    return cfg


def list_available_checkpoints(save_dir: Path) -> List[Path]:
    if not save_dir.is_dir():
        return []
    return sorted([p for p in list(save_dir.glob("*.pt")) + list(save_dir.glob("*.pth")) if p.is_file()])


def checkpoint_rank(path: Path) -> Tuple[int, int, float]:
    # Prefer checkpoint with largest step/update; fall back to mtime when unavailable.
    step_count = -1
    update_idx = -1
    try:
        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict):
            step_count = int(ckpt.get("step_count", -1))
            update_idx = int(ckpt.get("update_idx", -1))
    except Exception:
        pass
    return (step_count, update_idx, path.stat().st_mtime)


def resolve_checkpoint_path(requested_path: str) -> str:
    save_dir = Path("model/save")
    available_paths = list_available_checkpoints(save_dir)

    req = (requested_path or "").strip()
    if req.lower() in {"", "latest", "auto"}:
        if not available_paths:
            raise FileNotFoundError(
                "No checkpoint found in model/save. "
                "Train first (e.g. --save_path model/save/mrta_checkpoint.pt) "
                "or pass --checkpoint <path_to_model>."
            )
        best = max(available_paths, key=checkpoint_rank)
        return str(best)

    explicit_path = Path(req)
    if explicit_path.is_file():
        return str(explicit_path)

    # Convenience: allow passing a bare filename that exists under model/save.
    save_dir_candidate = save_dir / req
    if save_dir_candidate.is_file():
        return str(save_dir_candidate)

    existing = [p.name for p in available_paths]

    extra = f" Available in model/save: {', '.join(existing)}." if existing else ""
    raise FileNotFoundError(
        f"Checkpoint not found: {req}."
        " Train first (e.g. --save_path model/save/mrta_checkpoint.pt) "
        "or pass --checkpoint <path_to_model>."
        f"{extra}"
    )


def checkpoint_train_state(ckpt: Dict) -> str:
    step_count = int(ckpt.get("step_count", -1)) if isinstance(ckpt, dict) else -1
    update_idx = int(ckpt.get("update_idx", -1)) if isinstance(ckpt, dict) else -1
    return f"step={step_count} update={update_idx}"


def checkpoint_format(ckpt: Dict) -> str:
    model_sd = ckpt.get("model", None)
    if not isinstance(model_sd, dict):
        return "unknown"

    keys = model_sd.keys()
    if any(k.startswith("task_encoder.") for k in keys):
        return "mrta_new"
    return "unknown"


def build_policy_from_ckpt(ckpt: Dict, obs0: Dict, device: torch.device) -> MRTAPolicy:
    ckpt_fmt = checkpoint_format(ckpt)
    if ckpt_fmt != "mrta_new":
        raise RuntimeError(
            "Incompatible checkpoint format for MRTAPolicy evaluation. "
            f"Detected: {ckpt_fmt}. "
            "This evaluator expects a checkpoint saved by train_mrta.py "
            "(default: model/save/mrta_checkpoint.pt)."
        )

    node_ckpt_dim = infer_node_dim(ckpt)
    agent_ckpt_dim = infer_agent_dim(ckpt)
    pair_ckpt_dim = infer_pair_dim(ckpt)
    obs0 = adapt_obs_feature_dims(obs0, node_ckpt_dim, agent_ckpt_dim, pair_ckpt_dim)
    node_dim = int(obs0["task_graph"]["node_features"].shape[-1])
    agent_dim = int(obs0["agent_feats"].shape[-1])
    pair_dim = int(obs0["pair_feats"].shape[-1])
    ablations = ckpt.get("ablations", {})
    model_sd = ckpt.get("model", {})
    uses_graph_transformer = any("task_encoder.layers.0.rel_bias" in k for k in model_sd.keys())
    if "wo_graph_transformer" in ablations:
        uses_graph_transformer = not bool(ablations.get("wo_graph_transformer", False))
    hidden_dim = infer_hidden_dim(ckpt)
    policy = MRTAPolicy(
        node_dim=node_dim,
        agent_dim=agent_dim,
        pair_dim=pair_dim,
        hidden_dim=hidden_dim,
        use_rgat=not ablations.get("wo_rgat", False),
        use_graph_transformer=bool(uses_graph_transformer),
        self_only_graph=bool(ablations.get("self_only_graph", False)),
        rgat_temporal_propagation=bool(ablations.get("rgat_temporal_propagation", False)),
    ).to(device)
    missing, unexpected = policy.load_state_dict(ckpt["model"], strict=False)
    allowed_missing_prefixes = (
        "task_start_head.",
        "task_ready_head.",
        "pair_feasible_head.",
    )
    disallowed_missing = [k for k in missing if not k.startswith(allowed_missing_prefixes)]
    if disallowed_missing or unexpected:
        raise RuntimeError(
            "checkpoint is incompatible with current MRTAPolicy architecture. "
            f"missing={disallowed_missing[:8]} unexpected={list(unexpected)[:8]}"
        )
    if missing:
        print(
            "warning: checkpoint lacks auxiliary prediction heads; loading policy with randomly initialized auxiliary heads. "
            "These heads are not used during deterministic evaluation.",
            flush=True,
        )
    policy._eval_obs_dims = {"node_dim": node_dim, "agent_dim": agent_dim, "pair_dim": pair_dim}
    policy.eval()
    return policy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        type=str,
        default="latest",
        help="checkpoint path, or 'latest'/'auto' to pick newest training checkpoint in model/save",
    )
    p.add_argument("--setting", type=str, default="iid", choices=["iid", "ood_big", "ood_dense", "ood_dynamic", "ood_hetero"])
    p.add_argument("--episodes", type=int, default=35)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument(
        "--fixed_set_path",
        type=str,
        default="",
        help="Read-only fixed test set path: a directory (env_*/config.json) or a single config.json",
    )
    p.add_argument(
        "--fixed_set_max_steps",
        type=int,
        default=0,
        help="override max_steps for fixed-set evaluation; <=0 keeps scene config",
    )
    p.add_argument("--baseline", action="store_true", help="use non-policy baseline")
    p.add_argument(
        "--baseline_method",
        type=str,
        default="cbs_ta_ptc",
        choices=["cbs_ta_ptc", "ga", "ctas_d"],
    )
    p.add_argument(
        "--baseline_timeout_sec",
        type=float,
        default=5.0,
        help="per-decision timeout budget for compatible baselines such as cbs_ta_ptc",
    )
    p.add_argument(
        "--baseline_episode_timeout_sec",
        type=float,
        default=120.0,
        help="per-episode wall-clock cutoff for slow traditional baselines; 0 disables. "
        "On timeout the episode is truncated and partial metrics are recorded. "
        "Default 120s = 4x the common 30s MAPF benchmark convention.",
    )
    p.add_argument(
        "--baseline_planned_once",
        action="store_true",
        help="run baselines from a single initial plan; default is online replanning at each decision step",
    )
    p.add_argument(
        "--method_alias",
        type=str,
        default="",
        help="method name recorded in saved artifacts (default: ours / baseline_method)",
    )
    p.add_argument("--wo_mask", action="store_true")
    p.add_argument(
        "--graph_perturbation",
        type=str,
        default="original",
        choices=["original", "drop_edges", "shuffle_edges", "shuffle_relations"],
        help="perturb only the policy graph input; environment constraints and action masks remain unchanged",
    )
    p.add_argument(
        "--policy_decode",
        type=str,
        default="greedy",
        choices=["greedy", "sample"],
        help="decode mode for policy evaluation",
    )
    p.add_argument(
        "--policy_explore_eps",
        type=float,
        default=0.0,
        help="epsilon exploration probability under greedy decode",
    )
    p.add_argument(
        "--greedy_nonnoop_if_feasible",
        action="store_true",
        help="under greedy decode, replace NOOP by the best feasible task for agents with any feasible task",
    )
    p.add_argument("--wo_commitment", action="store_true")
    p.add_argument("--use_checkpoint_env_config", action="store_true", help="restore fixed-set env config from checkpoint['mrta_config'] for fair checkpoint evaluation")
    p.add_argument("--torch_num_threads", type=int, default=0,
                   help="set torch intra-op CPU threads; <=0 uses auto heuristic")
    p.add_argument("--torch_num_interop_threads", type=int, default=0,
                   help="set torch inter-op CPU threads; <=0 uses 1")
    p.add_argument("--output_dir", type=str, default="results/eval_vis",
                   help="directory to save evaluation artifacts")
    p.add_argument("--output_tag", type=str, default="",
                   help="optional subfolder name under output_dir; default is auto timestamped")
    p.add_argument("--oracle_metrics_csv", type=str, default="",
                   help="optional CP-SAT episode_metrics.csv used to add cpsat_relative_done")
    p.add_argument("--save_artifacts", action="store_true",
                   help="save CSV/JSON artifacts after evaluation")
    p.add_argument("--no_save_artifacts", action="store_false", dest="save_artifacts",
                   help="disable artifact saving")
    p.set_defaults(save_artifacts=True)
    p.add_argument(
        "--resume",
        action="store_true",
        help="resume fixed-set evaluation from output_tag/episode_metrics.csv",
    )
    p.add_argument("--plot", action="store_true",
                   help="save evaluation plots (.png) when matplotlib is available")
    p.add_argument("--no_plot", action="store_false", dest="plot",
                   help="disable plot generation")
    p.set_defaults(plot=True)
    return p.parse_args()


def main():
    args = parse_args()
    configure_torch_cpu_threads(args.torch_num_threads, args.torch_num_interop_threads)
    ecfg = EvalConfig(episodes=args.episodes, seeds=args.seeds)
    policy_explore_eps = float(np.clip(float(args.policy_explore_eps), 0.0, 1.0))
    device = resolve_torch_device()
    ckpt_path = None if args.baseline else resolve_checkpoint_path(args.checkpoint)

    ckpt = None
    if not args.baseline:
        if ckpt_path is None:
            raise RuntimeError("checkpoint is required when baseline=False")
        ckpt = torch.load(ckpt_path, map_location=device)
        print(
            f"loaded_checkpoint={Path(ckpt_path).as_posix()} format={checkpoint_format(ckpt)} {checkpoint_train_state(ckpt)}",
            flush=True,
        )

        try:
            latest_path = resolve_checkpoint_path("latest")
        except FileNotFoundError:
            latest_path = None
        if latest_path is not None and Path(latest_path).resolve() != Path(ckpt_path).resolve():
            print(
                f"warning: selected checkpoint is not latest. latest_available={Path(latest_path).as_posix()}",
                flush=True,
            )

    all_metrics = []
    eval_records = []
    method_alias = resolve_method_alias(args)
    baseline_adapter = (
        make_legacy_baseline_adapter(method=args.baseline_method, timeout_sec=float(args.baseline_timeout_sec))
        if args.baseline
        else None
    )
    oracle_by_scene = load_oracle_metrics(args.oracle_metrics_csv) if str(args.oracle_metrics_csv).strip() else None

    if args.fixed_set_path:
        fixed_paths = resolve_fixed_scene_paths(args.fixed_set_path)
        n_eval = len(fixed_paths) if int(ecfg.episodes) <= 0 else min(int(ecfg.episodes), len(fixed_paths))
        fixed_paths = fixed_paths[:n_eval]
        if oracle_by_scene:
            missing_oracle = [p for p in fixed_paths if not has_oracle_metrics_for_scene(p, oracle_by_scene)]
            if missing_oracle:
                covered_paths = [p for p in fixed_paths if has_oracle_metrics_for_scene(p, oracle_by_scene)]
                if not covered_paths:
                    sample_missing = ", ".join(p.as_posix() for p in missing_oracle[:5])
                    raise KeyError(
                        "oracle metrics do not match any selected fixed-set scene. "
                        f"sample_missing={sample_missing}"
                    )
                print(
                    "warning: oracle metrics cover only a subset of selected fixed-set scenes; "
                    f"evaluating covered scenes only ({len(covered_paths)}/{len(fixed_paths)}). "
                    f"first_missing={missing_oracle[0].as_posix()}",
                    flush=True,
                )
                fixed_paths = covered_paths
        print(f"fixed_set_eval: source={args.fixed_set_path} scenes={len(fixed_paths)}", flush=True)
        skipped_scenes = 0

        incremental_csv = None
        completed_scene_keys = set()
        if bool(args.save_artifacts):
            incremental_csv = _artifact_output_dir(args) / "episode_metrics.csv"
            if bool(args.resume):
                eval_records = _load_incremental_records(incremental_csv, method_alias=method_alias)
                all_metrics = [
                    {
                        key: float(value)
                        for key, value in row.items()
                        if key not in {"eval_index", "method", "scene_idx", "scene_path"} and str(value).strip() != ""
                    }
                    for row in eval_records
                ]
                completed_scene_keys = {
                    _scene_resume_key(row["scene_path"])
                    for row in eval_records
                    if str(row.get("scene_path", "")).strip()
                }
                print(
                    f"resume_eval: records={len(eval_records)} remaining={sum(_scene_resume_key(p) not in completed_scene_keys for p in fixed_paths)} "
                    f"path={incremental_csv.as_posix()}",
                    flush=True,
                )

        policy = None
        if not args.baseline:
            if ckpt is None:
                raise RuntimeError("checkpoint is required when baseline=False")
            env0 = None
            for scene_path in fixed_paths:
                try:
                    env0 = load_fixed_env_from_path(scene_path, wo_commitment=args.wo_commitment)
                    if args.use_checkpoint_env_config and ckpt is not None:
                        apply_policy_execution_overrides_to_env(env0, ckpt.get("mrta_config", {}))
                    break
                except Exception as exc:
                    skipped_scenes += 1
                    print(f"warn: skip invalid scene for policy bootstrap: {scene_path.as_posix()} ({exc})", flush=True)
            if env0 is None:
                raise RuntimeError("No valid fixed scenes available to bootstrap policy.")
            policy = build_policy_from_ckpt(ckpt, env0._get_obs(), device)

        for i, scene_path in enumerate(fixed_paths):
            if _scene_resume_key(scene_path) in completed_scene_keys:
                continue
            try:
                env = load_fixed_env_from_path(scene_path, wo_commitment=args.wo_commitment)
                if args.use_checkpoint_env_config and ckpt is not None:
                    apply_policy_execution_overrides_to_env(env, ckpt.get("mrta_config", {}))
                if int(args.fixed_set_max_steps) > 0:
                    env.config.max_steps = int(args.fixed_set_max_steps)
                materialize_fixed_dynamic_arrivals(env, int(env.config.max_steps))
                if args.baseline:
                    # These filters are learned-policy components, not neutral
                    # environment constraints shared by classical baselines.
                    env.config.use_policy_commitment_logic = False
                    env.config.capacity_topk_enabled = False
                if not args.baseline and not args.wo_commitment:
                    if not bool(env.config.use_policy_commitment_logic):
                        raise RuntimeError(
                            "fixed-set protocol error: Full Ours requires use_policy_commitment_logic=True"
                        )
                    if ckpt is not None and bool(ckpt.get("mrta_config", {}).get("capacity_topk_enabled", True)):
                        if not bool(env.config.capacity_topk_enabled):
                            raise RuntimeError(
                                "fixed-set protocol error: Full Ours requires capacity_topk_enabled=True"
                            )
                if args.baseline and bool(args.baseline_planned_once) and baseline_adapter is not None and baseline_adapter.supports_planned_execution():
                    m = run_episode_planned_baseline(
                        env,
                        baseline_adapter=baseline_adapter,
                        reset_env=False,
                    )
                else:
                    m = run_episode(
                        env,
                        policy=policy,
                        device=device,
                        deterministic=True,
                        policy_decode=str(args.policy_decode),
                        policy_explore_eps=policy_explore_eps,
                        greedy_nonnoop_if_feasible=bool(args.greedy_nonnoop_if_feasible),
                        wo_mask=args.wo_mask,
                        baseline_adapter=baseline_adapter,
                        initial_obs=env._get_obs(),
                        graph_perturbation=str(args.graph_perturbation),
                        graph_perturbation_seed=20260816 + int(i),
                        episode_timeout_sec=float(getattr(args, "baseline_episode_timeout_sec", 0.0)),
                    )
            except Exception as exc:
                skipped_scenes += 1
                print(f"warn: skip scene={i} path={scene_path.as_posix()} reason={exc}", flush=True)
                continue

            m = attach_oracle_metrics(m, scene_path=scene_path, oracle_by_scene=oracle_by_scene)
            all_metrics.append(m)
            record = {
                "eval_index": int(len(eval_records)),
                "method": method_alias,
                "scene_idx": int(i),
                "scene_path": scene_path.as_posix(),
                **m,
            }
            eval_records.append(record)
            if incremental_csv is not None:
                _write_records_csv_atomic(incremental_csv, eval_records)
            print(
                f"scene={i} path={scene_path.as_posix()} return={m['return']:.3f} "
                f"recognized_done_rate={m['recognized_done_rate']:.3f} success_rate={m['success_rate']:.3f} "
                f"truncated={int(m['truncated'])} makespan={m['makespan']} "
                f"reject={m['rejected_commitments']} viol_rate={m['violation_rate']:.3f} "
                f"done={m['completed_tasks']}/{m['total_tasks']}",
                flush=True,
            )

        if skipped_scenes > 0:
            print(f"warning: skipped_invalid_scenes={skipped_scenes}", flush=True)
    else:
        for s in range(ecfg.seeds):
            cfg = make_env_config(args.setting)
            cfg.ordered_commitment = not args.wo_commitment
            env = MRTAEnv(cfg, seed=1000 + s)

            policy = None
            if not args.baseline:
                if ckpt is None:
                    raise RuntimeError("checkpoint is required when baseline=False")
                obs0 = env.reset(seed=1000 + s)
                policy = build_policy_from_ckpt(ckpt, obs0, device)

            for ep in range(ecfg.episodes):
                env.rng = np.random.default_rng(1000 + s * 10000 + ep)
                if args.baseline and bool(args.baseline_planned_once) and baseline_adapter is not None and baseline_adapter.supports_planned_execution():
                    m = run_episode_planned_baseline(env, baseline_adapter=baseline_adapter)
                else:
                    m = run_episode(
                        env,
                        policy=policy,
                        device=device,
                        deterministic=True,
                        policy_decode=str(args.policy_decode),
                        policy_explore_eps=policy_explore_eps,
                        greedy_nonnoop_if_feasible=bool(args.greedy_nonnoop_if_feasible),
                        wo_mask=args.wo_mask,
                        baseline_adapter=baseline_adapter,
                        graph_perturbation=str(args.graph_perturbation),
                        graph_perturbation_seed=20260816 + int(s) * 10000 + int(ep),
                        episode_timeout_sec=float(getattr(args, "baseline_episode_timeout_sec", 0.0)),
                    )
                all_metrics.append(m)
                eval_records.append(
                    {
                        "eval_index": int(len(eval_records)),
                        "method": method_alias,
                        "seed": int(s),
                        "episode": int(ep),
                        **m,
                    }
                )
                print(
                    f"seed={s} episode={ep} return={m['return']:.3f} recognized_done_rate={m['recognized_done_rate']:.3f} "
                    f"success_rate={m['success_rate']:.3f} truncated={int(m['truncated'])} makespan={m['makespan']} "
                    f"reject={m['rejected_commitments']} viol_rate={m['violation_rate']:.3f} "
                    f"done={m['completed_tasks']}/{m['total_tasks']}",
                    flush=True,
                )

    if not all_metrics:
        raise RuntimeError("No evaluation samples were produced; check episodes/seeds/fixed_set_path.")

    agg = aggregate_metrics(all_metrics)
    print(
        f"setting={args.setting}, fixed_set_path={args.fixed_set_path if args.fixed_set_path else 'none'}, "
        f"baseline={args.baseline}, baseline_method={args.baseline_method if args.baseline else 'none'}"
    )
    for k, (m, st) in agg.items():
        print(f"{k}: {m:.4f} ± {st:.4f}")

    artifacts = save_eval_artifacts(
        records=eval_records,
        agg=agg,
        args=args,
        ckpt_path=ckpt_path,
    )
    if artifacts is not None:
        print(f"saved_eval_artifacts_dir={artifacts['out_dir'].as_posix()}", flush=True)
        print(f"saved_eval_records_csv={artifacts['records_csv'].as_posix()}", flush=True)
        print(f"saved_eval_aggregate_csv={artifacts['agg_csv'].as_posix()}", flush=True)
        print(f"saved_eval_summary_json={artifacts['summary_json'].as_posix()}", flush=True)
        for fig in artifacts["figures"]:
            print(f"saved_eval_figure={fig.as_posix()}", flush=True)


if __name__ == "__main__":
    main()
