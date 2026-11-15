from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np


def _stable_ga_seed(env) -> int:
    # Keep baseline search stochastic without consuming env.rng, which drives dynamic arrivals.
    task_sig = 0
    for tid, task in sorted(env.tasks.items()):
        loc = getattr(task, "loc", np.zeros((2,), dtype=np.float32))
        task_sig = (task_sig * 1315423911 + int(tid) * 2654435761) & 0xFFFFFFFF
        task_sig = (task_sig + int(round(float(loc[0]) * 1000.0)) * 374761393) & 0xFFFFFFFF
        task_sig = (task_sig + int(round(float(loc[1]) * 1000.0)) * 668265263) & 0xFFFFFFFF
    return int((task_sig + int(getattr(env, "current_time", 0)) * 2246822519) & 0xFFFFFFFF)


def _euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _candidate_tasks(env) -> List[int]:
    out: List[int] = []
    for tid, task in env.tasks.items():
        if int(tid) >= int(env.config.max_tasks):
            continue
        if task.status in {"AVAILABLE", "ASSIGNED", "READY"}:
            out.append(int(tid))
    return out


def _build_utility(
    env,
    active_agents: Sequence[int],
    task_ids: Sequence[int],
    action_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, int]]:
    n_agents = len(active_agents)
    n_tasks = len(task_ids)
    util = np.full((n_agents, n_tasks), -1e9, dtype=np.float64)
    cap_contrib = np.zeros((n_agents, n_tasks), dtype=np.float64)
    reqs = np.zeros((n_tasks,), dtype=np.float64)
    rem_slots = np.zeros((n_tasks,), dtype=np.int64)
    existing_caps = np.zeros((n_tasks,), dtype=np.float64)
    tid_to_col = {int(tid): j for j, tid in enumerate(task_ids)}

    dt = max(float(getattr(env.config, "dt_seconds", 1.0)), 1e-6)

    for j, tid in enumerate(task_ids):
        task = env.tasks[int(tid)]
        ttype = int(getattr(task, "task_type", 0))
        req = float(task.req[ttype]) if 0 <= ttype < len(task.req) else float(np.max(task.req))
        reqs[j] = max(req, 1e-6)
        rem_slots[j] = max(0, int(task.k) - len(task.assigned_robots))
        existing_caps[j] = float(sum(env.robots[rid].cap[ttype] for rid in task.assigned_robots))

    for ai, rid in enumerate(active_agents):
        rob = env.robots[int(rid)]
        rpos = (float(rob.pos[0]), float(rob.pos[1]))
        rv = max(float(getattr(rob, "v_max", 1.0)), 1e-6)

        for j, tid in enumerate(task_ids):
            action_id = int(tid) + 1
            if action_id >= action_mask.shape[1] or action_mask[int(rid), action_id] <= 0.5:
                continue

            task = env.tasks[int(tid)]
            ttype = int(getattr(task, "task_type", 0))
            if int(getattr(rob, "agent_type", -1)) != ttype:
                continue

            tpos = (float(task.loc[0]), float(task.loc[1]))
            dist = _euclid(rpos, tpos)
            eta_steps = dist / max(rv * dt, 1e-6)

            cap = float(rob.cap[ttype]) if 0 <= ttype < len(rob.cap) else float(np.max(rob.cap))
            cap_contrib[ai, j] = cap

            criticality = float(getattr(task, "criticality", 0.0))
            slack = float(getattr(task, "slack", 0.0))
            release_wait = max(float(task.release_time) - float(env.current_time), 0.0)

            cap_ratio = cap / reqs[j]
            urgent_bonus = max(-slack, 0.0)

            # Higher is better.
            util[ai, j] = (
                1.6 * criticality
                + 1.1 * cap_ratio
                + 0.9 * urgent_bonus
                - 0.22 * eta_steps
                - 0.04 * dist
                - 0.3 * release_wait
            )

    return util, cap_contrib, reqs, rem_slots, existing_caps, tid_to_col


def _decode_with_capacity(
    chrom: np.ndarray,
    task_ids: Sequence[int],
    tid_to_col: Dict[int, int],
    rem_slots: np.ndarray,
    util: np.ndarray,
) -> np.ndarray:
    decoded = np.zeros_like(chrom)
    if chrom.size == 0:
        return decoded

    selected_by_task: List[List[int]] = [[] for _ in range(len(task_ids))]
    for ai, act in enumerate(chrom):
        if int(act) <= 0:
            continue
        tid = int(act) - 1
        col = tid_to_col.get(tid, -1)
        if col < 0:
            continue
        selected_by_task[col].append(ai)

    for col, selected_agents in enumerate(selected_by_task):
        if not selected_agents:
            continue
        cap = int(rem_slots[col])
        if cap <= 0:
            continue

        if len(selected_agents) > cap:
            selected_agents = sorted(selected_agents, key=lambda x: float(util[x, col]), reverse=True)[:cap]

        action_id = int(task_ids[col]) + 1
        for ai in selected_agents:
            decoded[ai] = action_id

    return decoded


def _fitness(
    decoded: np.ndarray,
    has_non_noop: np.ndarray,
    util: np.ndarray,
    cap_contrib: np.ndarray,
    reqs: np.ndarray,
    rem_slots: np.ndarray,
    existing_caps: np.ndarray,
    tid_to_col: Dict[int, int],
) -> float:
    score = 0.0
    n_tasks = reqs.shape[0]
    fill = np.zeros((n_tasks,), dtype=np.int64)
    cap_sum = np.zeros((n_tasks,), dtype=np.float64)

    for ai, act in enumerate(decoded):
        act_i = int(act)
        if act_i <= 0:
            if bool(has_non_noop[ai]):
                score -= 0.08
            continue

        tid = act_i - 1
        col = tid_to_col.get(tid, -1)
        if col < 0:
            score -= 1.0
            continue
        score += float(util[ai, col])
        fill[col] += 1
        cap_sum[col] += float(cap_contrib[ai, col])

    for col in range(n_tasks):
        slots = int(rem_slots[col])
        if slots <= 0:
            continue

        assigned = int(fill[col])
        if assigned <= 0:
            continue

        score += 0.35 * float(assigned) / float(slots)

        if assigned >= slots:
            req = float(reqs[col])
            team_cap = float(existing_caps[col] + cap_sum[col])
            if team_cap + 1e-9 < req:
                shortage = (req - team_cap) / max(req, 1e-6)
                score -= 4.0 + 6.0 * float(shortage)
            else:
                margin = min((team_cap - req) / max(req, 1e-6), 1.0)
                score += 0.6 * float(margin)

    return score


def _greedy_seed(
    util: np.ndarray,
    rem_slots: np.ndarray,
    task_ids: Sequence[int],
) -> np.ndarray:
    n_agents, n_tasks = util.shape
    chrom = np.zeros((n_agents,), dtype=np.int64)
    if n_agents == 0 or n_tasks == 0:
        return chrom

    cap_left = rem_slots.astype(np.int64).copy()
    cand: List[Tuple[float, int, int]] = []
    for ai in range(n_agents):
        for col in range(n_tasks):
            u = float(util[ai, col])
            if u <= -1e8:
                continue
            cand.append((u, ai, col))
    cand.sort(key=lambda x: x[0], reverse=True)

    for _, ai, col in cand:
        if chrom[ai] > 0:
            continue
        if cap_left[col] <= 0:
            continue
        chrom[ai] = int(task_ids[col]) + 1
        cap_left[col] -= 1

    return chrom


def ga_actions(env, action_mask: np.ndarray) -> Dict[int, int]:
    num_agents = int(env.config.num_agents)
    actions = {rid: 0 for rid in range(num_agents)}

    task_ids = _candidate_tasks(env)
    if not task_ids:
        return actions

    active_agents = [rid for rid in range(num_agents) if float(np.sum(action_mask[rid, 1:])) > 0.5]
    if not active_agents:
        return actions

    util, cap_contrib, reqs, rem_slots, existing_caps, tid_to_col = _build_utility(
        env=env,
        active_agents=active_agents,
        task_ids=task_ids,
        action_mask=action_mask,
    )
    n_active, n_tasks = util.shape
    if n_active == 0 or n_tasks == 0:
        return actions

    # Allowed actions per active agent: always include NOOP.
    allowed_actions: List[np.ndarray] = []
    has_non_noop = np.zeros((n_active,), dtype=bool)
    for ai, rid in enumerate(active_agents):
        opts = [0]
        for tid in task_ids:
            action_id = int(tid) + 1
            if action_id < action_mask.shape[1] and action_mask[int(rid), action_id] > 0.5:
                opts.append(action_id)
        arr = np.array(opts, dtype=np.int64)
        allowed_actions.append(arr)
        has_non_noop[ai] = arr.size > 1

    rng = np.random.default_rng(_stable_ga_seed(env))

    pop_size = int(np.clip(2 * n_active + 4, 10, 24))
    generations = int(np.clip(3 + n_active // 4 + n_tasks // 6, 4, 10))
    elite = max(1, pop_size // 16)
    mut_prob = 0.35
    cross_prob = 0.55
    tour_k = 2

    assign_prob = min(0.45, 0.2 + 0.25 * min(1.0, float(n_tasks) / float(max(n_active, 1))))

    population = np.zeros((pop_size, n_active), dtype=np.int64)
    population[0] = _greedy_seed(util=util, rem_slots=rem_slots, task_ids=task_ids)

    for pi in range(1, pop_size):
        for ai in range(n_active):
            opts = allowed_actions[ai]
            if opts.size <= 1:
                population[pi, ai] = 0
                continue
            if float(rng.random()) < assign_prob:
                # Prefer non-NOOP at initialization to reduce wasted generations.
                choice = int(rng.choice(opts[1:]))
            else:
                choice = 0
            population[pi, ai] = choice

    def decode(chrom: np.ndarray) -> np.ndarray:
        return _decode_with_capacity(
            chrom=chrom,
            task_ids=task_ids,
            tid_to_col=tid_to_col,
            rem_slots=rem_slots,
            util=util,
        )

    def fit(decoded: np.ndarray) -> float:
        return _fitness(
            decoded=decoded,
            has_non_noop=has_non_noop,
            util=util,
            cap_contrib=cap_contrib,
            reqs=reqs,
            rem_slots=rem_slots,
            existing_caps=existing_caps,
            tid_to_col=tid_to_col,
        )

    decoded_cache = [decode(population[i]) for i in range(pop_size)]
    fitness = np.array([fit(decoded_cache[i]) for i in range(pop_size)], dtype=np.float64)

    best_idx = int(np.argmax(fitness))
    best_decoded = decoded_cache[best_idx].copy()
    best_fit = float(fitness[best_idx])

    def tournament_select() -> np.ndarray:
        idxs = rng.integers(0, pop_size, size=(tour_k,))
        j = int(idxs[np.argmax(fitness[idxs])])
        return population[j]

    for _ in range(generations):
        order = np.argsort(-fitness)
        new_population = np.zeros_like(population)
        for ei in range(elite):
            new_population[ei] = population[int(order[ei])]

        fill_at = elite
        while fill_at < pop_size:
            p1 = tournament_select()
            p2 = tournament_select()

            if float(rng.random()) < cross_prob:
                mask = rng.random(n_active) < 0.5
                child = np.where(mask, p1, p2).astype(np.int64, copy=False)
            else:
                child = p1.copy()

            for ai in range(n_active):
                if float(rng.random()) < mut_prob:
                    child[ai] = int(rng.choice(allowed_actions[ai]))

            # Safety repair: force gene to be in allowed set.
            for ai in range(n_active):
                if child[ai] not in allowed_actions[ai]:
                    child[ai] = int(rng.choice(allowed_actions[ai]))

            new_population[fill_at] = child
            fill_at += 1

        population = new_population
        decoded_cache = [decode(population[i]) for i in range(pop_size)]
        fitness = np.array([fit(decoded_cache[i]) for i in range(pop_size)], dtype=np.float64)

        cur_idx = int(np.argmax(fitness))
        cur_fit = float(fitness[cur_idx])
        if cur_fit > best_fit:
            best_fit = cur_fit
            best_decoded = decoded_cache[cur_idx].copy()

    for ai, rid in enumerate(active_agents):
        action_id = int(best_decoded[ai])
        if action_id <= 0:
            continue
        if action_id >= action_mask.shape[1] or action_mask[int(rid), action_id] <= 0.5:
            continue
        actions[int(rid)] = action_id

    return actions


__all__ = ["ga_actions"]
