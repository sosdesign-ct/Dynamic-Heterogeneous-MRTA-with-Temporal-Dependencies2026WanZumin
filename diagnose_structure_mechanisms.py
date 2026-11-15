from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from env.fixed_set_loader import apply_config_overrides_to_env, load_fixed_env_from_path, resolve_fixed_scene_paths
from evaluate_mrta import build_policy_from_ckpt
from torch_runtime import resolve_torch_device


def clone_obs(obs: Dict) -> Dict:
    out = {}
    for key, value in obs.items():
        if isinstance(value, dict):
            out[key] = {k: np.array(v, copy=True) for k, v in value.items()}
        else:
            out[key] = np.array(value, copy=True)
    return out


def perturb_graph(obs: Dict, mode: str, rng: np.random.Generator) -> Dict:
    out = clone_obs(obs)
    graph = out["task_graph"]
    edge_mask = graph.get("edge_mask", np.ones_like(graph["edge_type"], dtype=np.float32))
    if mode == "drop_edges":
        graph["edge_mask"] = np.zeros_like(edge_mask, dtype=np.float32)
    elif mode == "shuffle_edges":
        valid = edge_mask > 0.5
        destinations = graph["edge_index"][1, valid].copy()
        rng.shuffle(destinations)
        graph["edge_index"][1, valid] = destinations
    elif mode == "shuffle_relations":
        valid = edge_mask > 0.5
        values = graph["edge_type"][valid].copy()
        rng.shuffle(values)
        graph["edge_type"][valid] = values
    elif mode != "original":
        raise ValueError(f"unknown graph perturbation: {mode}")
    return out


def cosine_gap(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    if a.dim() == 3:
        a = a.reshape(-1, a.shape[-1])
        b = b.reshape(-1, b.shape[-1])
    if mask is not None:
        mask = mask.reshape(-1) > 0.5
        a = a[mask]
        b = b[mask]
    if a.numel() == 0:
        return 0.0
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    return float((1.0 - cos).mean().detach().cpu())


def distribution_metrics(logits: torch.Tensor, reference: torch.Tensor) -> tuple[float, float]:
    p = torch.softmax(logits, dim=-1)
    q = torch.softmax(reference, dim=-1)
    kl = torch.sum(p * (torch.log(p.clamp_min(1e-8)) - torch.log(q.clamp_min(1e-8))), dim=-1)
    tv = 0.5 * torch.sum((p - q).abs(), dim=-1)
    return float(kl.mean().cpu()), float(tv.mean().cpu())


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose graph and decoder mechanism sensitivity.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fixed_set_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenes", type=int, default=20)
    parser.add_argument("--steps_per_scene", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()

    device = resolve_torch_device()
    ckpt = torch.load(args.checkpoint, map_location=device)
    paths = resolve_fixed_scene_paths(args.fixed_set_path)[: max(1, int(args.scenes))]
    if not paths:
        raise RuntimeError("no fixed scenes found")

    env0 = load_fixed_env_from_path(paths[0], wo_commitment=False)
    if isinstance(ckpt.get("mrta_config"), dict):
        apply_config_overrides_to_env(env0, ckpt["mrta_config"])
    policy = build_policy_from_ckpt(ckpt, env0._get_obs(), device)
    rng = np.random.default_rng(int(args.seed))
    rows: List[Dict[str, object]] = []

    for scene_idx, path in enumerate(paths):
        env = load_fixed_env_from_path(path, wo_commitment=False)
        if isinstance(ckpt.get("mrta_config"), dict):
            apply_config_overrides_to_env(env, ckpt["mrta_config"])
        obs = env.reset()
        for step in range(max(1, int(args.steps_per_scene))):
            variants = {
                mode: perturb_graph(obs, mode, rng)
                for mode in ("original", "drop_edges", "shuffle_edges", "shuffle_relations")
            }
            outputs = {}
            with torch.no_grad():
                for mode, variant_obs in variants.items():
                    torch_obs = __import__("evaluate_mrta").to_torch_obs(variant_obs, device)
                    _, _, aux = policy.forward(torch_obs, use_mask=True, return_aux=True)
                    outputs[mode] = aux

            base = outputs["original"]
            base_logits = base["action_logits"]
            task_mask = variants["original"]["task_graph"]["node_mask"]
            for mode in ("drop_edges", "shuffle_edges", "shuffle_relations"):
                logits = outputs[mode]["action_logits"]
                rows.append({
                    "scene_idx": scene_idx,
                    "scene_path": str(path),
                    "step": step,
                    "comparison": mode,
                    "logit_mae": float((logits - base_logits).abs().mean().cpu()),
                    "action_flip_rate": float((torch.argmax(logits, dim=-1) != torch.argmax(base_logits, dim=-1)).float().mean().cpu()),
                    "policy_kl": distribution_metrics(logits, base_logits)[0],
                    "policy_total_variation": distribution_metrics(logits, base_logits)[1],
                    "task_representation_cosine_gap": cosine_gap(outputs[mode]["task_h"], base["task_h"], torch.as_tensor(task_mask, device=device)),
                    "graph_representation_cosine_gap": cosine_gap(outputs[mode]["graph_h"], base["graph_h"]),
                    "valid_edge_count": int(np.sum(variants["original"]["task_graph"].get("edge_mask", 1.0) > 0.5)),
                    "relation_type_count": int(np.unique(variants["original"]["task_graph"]["edge_type"]).size),
                })

            rows.append({
                "scene_idx": scene_idx,
                "scene_path": str(path),
                "step": step,
                "comparison": "decoder",
                "logit_mae": 0.0,
                "action_flip_rate": 0.0,
                "policy_kl": 0.0,
                "policy_total_variation": 0.0,
                "task_representation_cosine_gap": cosine_gap(base["task_h"], base["task_z"], torch.as_tensor(task_mask, device=device)),
                "graph_representation_cosine_gap": 0.0,
                "valid_edge_count": int(np.sum(variants["original"]["task_graph"].get("edge_mask", 1.0) > 0.5)),
                "relation_type_count": int(np.unique(variants["original"]["task_graph"]["edge_type"]).size),
            })

            act = torch.argmax(base_logits, dim=-1)
            actions = {i: int(act[i].item()) for i in range(act.shape[0])}
            obs, _, terminated, truncated, _ = env.step(actions)
            if terminated or truncated:
                break

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "checkpoint": str(args.checkpoint),
        "fixed_set_path": str(args.fixed_set_path),
        "scenes": len(paths),
        "rows": len(rows),
        "protocol": "same initial trajectory; graph perturbations affect policy input only",
    }
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()