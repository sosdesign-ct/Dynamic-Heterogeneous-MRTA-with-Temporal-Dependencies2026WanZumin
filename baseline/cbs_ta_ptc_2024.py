# -*- coding: utf-8 -*-
"""CBS-TA-PTC (AAMAS 2024, Chong et al.) — high level + MRTA integration.

Implements:
  * Algorithm 2  : CBS-TA-PTC high-level CT-forest search ordered by return.
  * Algorithm 1  : task decomposition into subtasks via a user heuristic
                   (we use the earliest-absolute-deadline / release heuristic,
                   the paper's recommended criterion tied to T_abs).
  * Integration  : an online, incremental ``cbs_ta_ptc_actions_2024`` entry
                   point that mirrors the existing 2018 baseline's interface so
                   the two remain directly comparable.

The implementation is faithful to the paper while adapting only the
problem-snapshot source (online MRTA env) and the oracle (lightweight analytic
return), both decided with the user.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import itertools
import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from baseline.cbs_ta_ptc_assignment import (
    FORBIDDEN_COST,
    GridPath,
    GridPoint,
    _back_dijkstra,
    _candidate_tasks,
    _build_task_slots,
    _build_assignment_cost_matrix,
    _grid_dims,
)
from baseline.cbs_ta_ptc_constraints import (
    ConstraintKind,
    InterGoalConstraint,
    PrecedenceConstraint,
    TemporalConstraint,
)
from baseline.cbs_ta_ptc_lowlevel import (
    NodeConstraints,
    analytic_return,
    infer_completion_times,
    mla_star_search,
)


# ---------------------------------------------------------------------------
# Module-level heuristic-map cache.
#
# ``_back_dijkstra`` runs a full-map Dijkstra *per goal* in pure Python.  On the
# large-scale scenes (e.g. s7/s8, 100x100 grids with dozens of agents) this is
# the dominant per-decision cost, and the goals are fixed for the lifetime of a
# subtask, so we memoise the heuristic maps keyed by (goals, grid size).  This
# does not change any planning semantics -- it only avoids recomputing identical
# shortest-path maps across the repeated CBS calls of one episode.
# ---------------------------------------------------------------------------
_HMAP_CACHE: Dict[Tuple[Tuple[Tuple[int, int], ...], int, int], list] = {}
_HMAP_CACHE_CAP = 256


def _cached_back_dijkstra(goals, map_grid, x_size, y_size, collision_thresh):
    key = (tuple((int(g.x_pos), int(g.y_pos)) for g in goals), int(x_size), int(y_size))
    hit = _HMAP_CACHE.get(key)
    if hit is not None:
        return hit
    h = _back_dijkstra(goals, map_grid, x_size, y_size, collision_thresh)
    if len(_HMAP_CACHE) >= _HMAP_CACHE_CAP:
        _HMAP_CACHE.clear()
    _HMAP_CACHE[key] = h
    return h


# ---------------------------------------------------------------------------
# CT node
# ---------------------------------------------------------------------------


@dataclass
class CTNode:
    constraints: NodeConstraints
    assignment: List[int]              # agent local idx -> goal local idx
    solution: List[GridPath]
    completion_times: List[int]
    cost: float
    ret: float
    n_conflicts: int
    order: int = field(default=0, compare=True)


# ---------------------------------------------------------------------------
# Conflict detection (vertex / edge) + temporal violations
# ---------------------------------------------------------------------------


def _lengthen(paths: List[GridPath]) -> None:
    if not paths:
        return
    maxlen = max((len(p.path_vect) for p in paths if p.path_vect), default=0)
    for p in paths:
        if not p.path_vect:
            continue
        last = p.path_vect[-1]
        while len(p.path_vect) < maxlen:
            p.path_vect.append(last)


def _count_motion_conflicts(paths: Sequence[GridPath]) -> int:
    n = 0
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            pi, pj = paths[i].path_vect, paths[j].path_vect
            L = min(len(pi), len(pj))
            for k in range(L):
                if pi[k] == pj[k]:
                    n += 1
                    break
            for k in range(1, L):
                if pi[k] == pj[k - 1] and pj[k] == pi[k - 1]:
                    n += 1
                    break
    return n


def _temporal_violations(
    completion_times: Sequence[int],
    constraints: NodeConstraints,
) -> List[Tuple[ConstraintKind, object]]:
    """Return violated temporal constraints in priority order."""
    out: List[Tuple[ConstraintKind, object]] = []
    for c in constraints.temporal:
        tau = completion_times[c.goal] if 0 <= c.goal < len(completion_times) else -1
        if tau < 0 or not c.is_satisfied(tau):
            out.append((ConstraintKind.ABSOLUTE, c))
    for c in constraints.precedence:
        ta = completion_times[c.goal_a] if 0 <= c.goal_a < len(completion_times) else -1
        tb = completion_times[c.goal_b] if 0 <= c.goal_b < len(completion_times) else -1
        if ta < 0 or tb < 0 or (tb - ta) < c.min_gap:
            out.append((ConstraintKind.PRECEDENCE, c))
    for c in constraints.intergoal:
        ta = completion_times[c.goal_a] if 0 <= c.goal_a < len(completion_times) else -1
        tb = completion_times[c.goal_b] if 0 <= c.goal_b < len(completion_times) else -1
        if ta < 0 or tb < 0 or (tb - ta) > c.t_inter:
            out.append((ConstraintKind.INTERGOAL, c))
    out.sort(key=lambda x: x[0])
    return out


# ---------------------------------------------------------------------------
# Low-level re-plan of one agent under current constraints
# ---------------------------------------------------------------------------


def _plan_agent(
    agent_idx: int,
    start: GridPoint,
    goal: GridPoint,
    h_map,
    map_grid,
    x_size: int,
    y_size: int,
    constraints: NodeConstraints,
    goal_lower: int,
    goal_upper: Optional[int],
    deadline_ts: float,
) -> GridPath:
    # Cap the low-level search effort so a single re-plan cannot blow the
    # per-decision budget on large maps.  max_time_steps scales with the map
    # diagonal (longest useful path) plus a waiting buffer; max_expanded bounds
    # the total node expansions as a hard safety net alongside the wall-clock
    # deadline.
    diag = int(math.ceil(math.hypot(x_size, y_size)))
    mts = max(64, min(256, diag * 2 + 32))
    return mla_star_search(
        start=start,
        goal=goal,
        h_map=h_map,
        map_grid=map_grid,
        x_size=x_size,
        y_size=y_size,
        collision_thresh=1.0,
        agent_index=agent_idx,
        vertex_constraints=constraints.vertex,
        goal_lower_bound=goal_lower,
        goal_upper_bound=goal_upper,
        max_time_steps=mts,
        deadline_ts=deadline_ts,
        max_expanded=40_000,
    )


# ---------------------------------------------------------------------------
# Algorithm 2 : CBS-TA-PTC high-level search over one subtask
# ---------------------------------------------------------------------------


def cbs_ta_ptc_plan(
    starts: List[GridPoint],
    goals: List[GridPoint],
    map_grid,
    x_size: int,
    y_size: int,
    base_precedence: Sequence[PrecedenceConstraint],
    base_intergoal: Sequence[InterGoalConstraint],
    base_absolute: Sequence[TemporalConstraint],
    goals_meta: Sequence[dict],
    dt_seconds: float,
    cost_matrix: np.ndarray,
    timeout_sec: float = 10.0,
    max_assignments: int = 8,
    max_ct_expansions: int = 60,
) -> Optional[CTNode]:
    """High-level CT-forest search ordered by return (Algorithm 2).

    Enumerates assignments (root CT nodes), pushes them to OPEN, then expands
    by best return, resolving conflicts in the paper's priority order.
    """
    t0 = time.time()
    deadline = t0 + max(0.0, timeout_sec)
    n_agents = len(starts)
    n_goals = len(goals)
    if n_agents == 0 or n_goals == 0:
        return None

    h_maps = _cached_back_dijkstra(goals, map_grid, x_size, y_size, collision_thresh=1.0)

    # Goal-arrival bounds per goal.  The lower bound combines:
    #   * absolute temporal range constraints (release / latest bounds), and
    #   * the precedence transitive closure: a successor goal cannot complete
    #     before (max predecessor lower bound) + min_gap, recursively.
    # This is what lets the low level enforce precedence *during* planning
    # rather than only penalising it in the return, which is essential for
    # CBS-TA-PTC to return constraint-satisfying solutions (paper Section 4.1).
    abs_lo: List[int] = [0] * n_goals
    abs_hi: List[Optional[int]] = [None] * n_goals
    for c in base_absolute:
        if 0 <= c.goal < n_goals:
            if c.lower is not None:
                abs_lo[c.goal] = max(abs_lo[c.goal], int(c.lower))
            if c.upper is not None:
                abs_hi[c.goal] = int(c.upper) if abs_hi[c.goal] is None else min(abs_hi[c.goal], int(c.upper))

    # propagate precedence lower bounds (DAG assumed; guard against cycles)
    prec_lo = list(abs_lo)
    for _ in range(n_goals + 1):
        changed = False
        for c in base_precedence:
            if 0 <= c.goal_a < n_goals and 0 <= c.goal_b < n_goals:
                need = prec_lo[c.goal_a] + int(c.min_gap)
                if need > prec_lo[c.goal_b]:
                    prec_lo[c.goal_b] = need
                    changed = True
        if not changed:
            break

    def goal_bounds(gi: int, constraints: Optional[NodeConstraints] = None) -> Tuple[int, Optional[int]]:
        """Lower/upper arrival bounds for goal ``gi``.

        Combines the static precedence/absolute bounds with any *dynamic*
        temporal constraints added on the current CT node (``constraints.temporal``).
        This is what lets the low level enforce CBS-added temporal bounds during
        re-planning (paper Section 4.1, lines 29 of Algorithm 2).
        """
        lo, hi = prec_lo[gi], abs_hi[gi]
        if constraints is not None:
            for c in constraints.temporal:
                if c.goal != gi:
                    continue
                if c.lower is not None:
                    lo = max(lo, int(c.lower))
                if c.upper is not None:
                    hi = int(c.upper) if hi is None else min(hi, int(c.upper))
        return lo, hi

    def full_solution(assignment: Sequence[int], constraints: NodeConstraints) -> Optional[List[GridPath]]:
        paths: List[GridPath] = []
        for ai in range(n_agents):
            gi = assignment[ai]
            lo, hi = goal_bounds(gi, constraints)
            p = _plan_agent(
                ai, starts[ai], goals[gi], h_maps[gi], map_grid, x_size, y_size,
                constraints, lo, hi, deadline - time.time(),
            )
            if not p.path_vect or not math.isfinite(p.cost):
                return None
            paths.append(p)
        return paths

    def evaluate(paths: List[GridPath], constraints: NodeConstraints) -> Tuple[List[int], int, float]:
        # Completion times must be derived from the *raw* arrival time of each
        # path (before any goal-wait padding), otherwise _lengthen would push a
        # fast predecessor up to the successor's arrival time and destroy the
        # precedence gap we just enforced.  We therefore infer completion times
        # first, then lengthen only for the motion-conflict check.
        ct = infer_completion_times(paths, goals_meta, dt_seconds)
        motion_paths = [GridPath(list(p.path_vect), p.cost) for p in paths]
        _lengthen(motion_paths)
        nconf = _count_motion_conflicts(motion_paths)
        ret = analytic_return(
            paths, ct, goals_meta,
            constraints.precedence, constraints.intergoal, constraints.temporal,
            n_conflicts=nconf,
        )
        return ct, nconf, ret

    # --- Enumerate candidate assignments (root CT nodes), best-first by cost.
    open_heap: List[Tuple[float, int, CTNode]] = []
    counter = itertools.count()
    max_ret = float(len(goals))  # every goal satisfied => upper bound on return

    # candidate goal subsets per agent (finite-cost columns)
    per_agent_cols: List[List[int]] = []
    for ai in range(n_agents):
        cols = [g for g in range(n_goals) if cost_matrix[ai, g] < FORBIDDEN_COST * 0.5]
        if not cols:
            cols = list(range(n_goals))
        per_agent_cols.append(cols)

    # rank assignments by assignment cost, take the cheapest max_assignments.
    # NOTE: on large scenes the full cartesian product of per-agent goal columns
    # is astronomically large (len(col)^n_agents), so we must NOT materialise it.
    # When the product is small we enumerate it exactly; otherwise we fall back to
    # a greedy per-agent-cheapest construction plus a bounded DFS over the
    # cheapest column alternatives, which keeps the same optimal-first ordering
    # for the cases that matter while guaranteeing termination.
    n_cols = [len(c) for c in per_agent_cols]
    total_combos = 1
    for c in n_cols:
        total_combos *= max(1, c)
        if total_combos > 200_000:  # early guard; exact value irrelevant beyond cap
            break

    assignment_candidates: List[Tuple[float, List[int]]] = []
    if total_combos <= 200_000:
        for combo in itertools.product(*[range(len(c)) for c in per_agent_cols]):
            assign = [per_agent_cols[ai][combo[ai]] for ai in range(n_agents)]
            if len(set(assign)) < len(assign):
                continue  # one goal per agent
            acost = sum(cost_matrix[ai, assign[ai]] for ai in range(n_agents))
            assignment_candidates.append((acost, assign))
        assignment_candidates.sort(key=lambda x: x[0])
        assignment_candidates = assignment_candidates[:max_assignments]
    else:
        # Greedy / bounded branch-and-bound over cheapest columns.  We explore a
        # depth-limited DFS ordered by per-column cost, keeping the best
        # max_assignments distinct-goal assignments seen.  A global deadline guard
        # ensures this can never blow the per-decision budget.
        sorted_cols = [
            sorted(range(len(c)), key=lambda j: cost_matrix[ai, c[j]])
            for ai, c in enumerate(per_agent_cols)
        ]
        best: List[Tuple[float, List[int]]] = []
        col_limit = 4  # only the 4 cheapest columns per agent
        dfs_deadline = time.time() + max(0.05, 0.4 * max(0.0, deadline - time.time()))

        def _dfs(ai: int, used: set, cur: List[int], cur_cost: float) -> None:
            if len(best) >= max_assignments * 4 or time.time() > dfs_deadline:
                return
            if ai == n_agents:
                best.append((cur_cost, list(cur)))
                return
            for j in sorted_cols[ai][:col_limit]:
                g = per_agent_cols[ai][j]
                if g in used:
                    continue
                cur.append(g)
                used.add(g)
                _dfs(ai + 1, used, cur, cur_cost + cost_matrix[ai, g])
                used.discard(g)
                cur.pop()

        _dfs(0, set(), [], 0.0)
        best.sort(key=lambda x: x[0])
        # dedupe identical assignments, keep cheapest max_assignments
        seen = set()
        for acost, assign in best:
            t = tuple(assign)
            if t in seen:
                continue
            seen.add(t)
            assignment_candidates.append((acost, assign))
            if len(assignment_candidates) >= max_assignments:
                break
        if not assignment_candidates:  # fallback: per-agent cheapest distinct
            used = set()
            assign = []
            ok = True
            for ai in range(n_agents):
                pick = None
                for j in sorted_cols[ai]:
                    g = per_agent_cols[ai][j]
                    if g not in used:
                        pick = g
                        break
                if pick is None:
                    ok = False
                    break
                used.add(pick)
                assign.append(pick)
            if ok:
                acost = sum(cost_matrix[ai, assign[ai]] for ai in range(n_agents))
                assignment_candidates.append((acost, assign))

    for _, assign in assignment_candidates:
        if time.time() > deadline:
            break
        nc = NodeConstraints(
            temporal=list(base_absolute),
            precedence=list(base_precedence),
            intergoal=list(base_intergoal),
        )
        paths = full_solution(assign, nc)
        if paths is None:
            continue
        ct, nconf, ret = evaluate(paths, nc)
        node = CTNode(nc, list(assign), paths, ct, sum(p.cost for p in paths), ret, nconf)
        if ret >= max_ret and nconf == 0 and not _temporal_violations(ct, nc):
            return node
        heapq.heappush(open_heap, (-ret, next(counter), node))

    # --- Best-first CT expansion by return.
    expansions = 0
    best_node: Optional[CTNode] = None
    while open_heap and time.time() < deadline and expansions < max_ct_expansions:
        _, _, P = heapq.heappop(open_heap)
        if best_node is None or P.ret > best_node.ret:
            best_node = P
        viol = _temporal_violations(P.completion_times, P.constraints)
        if not viol and P.n_conflicts == 0:
            return P
        expansions += 1

        # Resolve highest-priority conflict (temporal first, then motion).
        if viol:
            kind, c = viol[0]
            children = _resolve_temporal(P, kind, c)
        else:
            children = _resolve_motion(P)

        for child_constraints, replan_agent in children:
            if time.time() > deadline:
                break
            Qc = child_constraints
            paths = [GridPath(list(p.path_vect), p.cost) for p in P.solution]
            ai = replan_agent
            gi = P.assignment[ai]
            lo, hi = goal_bounds(gi, Qc)
            p = _plan_agent(ai, starts[ai], goals[gi], h_maps[gi], map_grid,
                             x_size, y_size, Qc, lo, hi, deadline - time.time())
            if not p.path_vect or not math.isfinite(p.cost):
                continue
            paths[ai] = p
            ct, nconf, ret = evaluate(paths, Qc)
            node = CTNode(Qc, list(P.assignment), paths, ct, sum(x.cost for x in paths), ret, nconf)
            if ret >= max_ret and nconf == 0 and not _temporal_violations(ct, Qc):
                return node
            heapq.heappush(open_heap, (-ret, next(counter), node))

    return best_node


def _resolve_temporal(P: CTNode, kind: ConstraintKind, c) -> List[Tuple[NodeConstraints, int]]:
    """Generate child constraint-sets for a temporal violation (paper 4.1)."""
    out: List[Tuple[NodeConstraints, int]] = []
    if kind == ConstraintKind.ABSOLUTE:
        # tighten the offending goal's lower bound -> replan its agent
        newc = TemporalConstraint(goal=c.goal, kind=ConstraintKind.ABSOLUTE,
                                  lower=(c.lower if c.lower is not None else 0),
                                  upper=c.upper)
        nc = P.constraints.copy()
        nc.temporal.append(newc)
        agent = _agent_of_goal(P, c.goal)
        if agent is not None:
            out.append((nc, agent))
    elif kind == ConstraintKind.PRECEDENCE:
        # push successor later: tau(b) >= tau(a)+min_gap  -> lower bound on b
        ta = P.completion_times[c.goal_a] if 0 <= c.goal_a < len(P.completion_times) else 0
        nc = P.constraints.copy()
        nc.temporal.append(TemporalConstraint(goal=c.goal_b, kind=ConstraintKind.PRECEDENCE,
                                              lower=int(ta) + c.min_gap))
        agent = _agent_of_goal(P, c.goal_b)
        if agent is not None:
            out.append((nc, agent))
    elif kind == ConstraintKind.INTERGOAL:
        # two children (paper): (1) a earlier, (2) b earlier by >=1
        tb = P.completion_times[c.goal_b] if 0 <= c.goal_b < len(P.completion_times) else 0
        nc1 = P.constraints.copy()
        nc1.temporal.append(TemporalConstraint(goal=c.goal_a, kind=ConstraintKind.INTERGOAL,
                                               upper=int(tb) - c.t_inter))
        a1 = _agent_of_goal(P, c.goal_a)
        if a1 is not None:
            out.append((nc1, a1))
        nc2 = P.constraints.copy()
        nc2.temporal.append(TemporalConstraint(goal=c.goal_b, kind=ConstraintKind.INTERGOAL,
                                               upper=max(0, int(tb) - 1)))
        a2 = _agent_of_goal(P, c.goal_b)
        if a2 is not None:
            out.append((nc2, a2))
    return out


def _resolve_motion(P: CTNode) -> List[Tuple[NodeConstraints, int]]:
    """First vertex/edge conflict -> two children with symmetric constraints."""
    paths = P.solution
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            pi, pj = paths[i].path_vect, paths[j].path_vect
            L = min(len(pi), len(pj))
            for k in range(L):
                if pi[k] == pj[k]:
                    nci = P.constraints.copy(); nci.vertex.append((i, pi[k], k))
                    ncj = P.constraints.copy(); ncj.vertex.append((j, pi[k], k))
                    return [(nci, i), (ncj, j)]
            for k in range(1, L):
                if pi[k] == pj[k - 1] and pj[k] == pi[k - 1]:
                    nci = P.constraints.copy(); nci.vertex.append((i, pi[k], k))
                    ncj = P.constraints.copy(); ncj.vertex.append((j, pj[k], k))
                    return [(nci, i), (ncj, j)]
    return []


def _agent_of_goal(P: CTNode, goal: int) -> Optional[int]:
    for ai, g in enumerate(P.assignment):
        if g == goal:
            return ai
    return None


# ---------------------------------------------------------------------------
# Algorithm 1 : decompose into subtasks + sequential composition (online step)
# ---------------------------------------------------------------------------


def cbs_ta_ptc_actions_2024(env, action_mask: np.ndarray, timeout_sec: float = 10.0,
                            goals_per_subtask: int = 3) -> Dict[int, int]:
    """Online incremental entry point mirroring the 2018 baseline interface.

    We take the current snapshot of committable tasks, order them by the
    paper-recommended absolute-deadline heuristic, form one subtask (the goals
    we can commit this step), and run Algorithm 2 on it.  Returns env actions.
    """
    num_agents = int(env.config.num_agents)
    actions = {rid: 0 for rid in range(num_agents)}

    task_ids = _candidate_tasks(env)
    if not task_ids:
        return actions
    active_agents = [rid for rid in range(num_agents) if float(np.sum(action_mask[rid, 1:])) > 0.5]
    if not active_agents:
        return actions

    # HeuristicSort: ascending absolute deadline (earliest_finish), then release.
    def sort_key(tid: int):
        t = env.tasks[int(tid)]
        hi = getattr(t, "latest_finish", 0.0) or getattr(t, "earliest_finish", 0.0)
        return (float(hi) if hi else float("inf"), int(getattr(t, "release_time", 0)))

    task_ids = sorted(task_ids, key=sort_key)[: max(1, goals_per_subtask)]

    task_slots = _build_task_slots(env, task_ids, action_mask, active_agents)
    if not task_slots:
        return actions
    task_slots = [s for s in task_slots
                  if any((int(s[0]) + 1) < action_mask.shape[1] and action_mask[r, int(s[0]) + 1] > 0.5
                         for r in active_agents)]
    if not task_slots:
        return actions

    cost_matrix, starts, goals = _build_assignment_cost_matrix(env, active_agents, task_slots, action_mask)
    if cost_matrix.size == 0 or not np.any(cost_matrix < FORBIDDEN_COST * 0.5):
        return actions

    gx, gy = _grid_dims(env)
    grid_map = [[0.0 for _ in range(gy)] for _ in range(gx)]

    # Keep a solvable agent subset (one goal per agent), cheapest first.
    # NOTE(2026-09): the original 2018 adapter retries with progressively
    # smaller agent subsets when the current subset admits no distinct-goal
    # assignment (sparse cost matrix / more robots than solvable slots).  The
    # 2024 port previously kept exactly ``n_goals`` agents once and gave up if
    # that single subset had no feasible assignment, which made some scenes
    # (e.g. oracle229 env_2) return zero actions every step.  We restore the
    # shrink-and-retry behaviour here: try the cheapest ``n_goals`` agents
    # first, then keep_n-1, ... until ``cbs_ta_ptc_plan`` finds a solution.
    finite_row = np.where(np.any(cost_matrix < FORBIDDEN_COST * 0.5, axis=1))[0]
    order = np.argsort(np.min(cost_matrix[finite_row, :], axis=1))
    ranked_rows = finite_row[order]  # all solvable agents, cheapest first
    max_keep = int(min(len(ranked_rows), cost_matrix.shape[1]))

    # ---- Build PTC constraints from env (precedence/inter-goal/absolute).
    slot_index = {tid: gi for gi, (tid, _) in enumerate(task_slots)}
    precedence: List[PrecedenceConstraint] = []
    intergoal: List[InterGoalConstraint] = []
    absolute: List[TemporalConstraint] = []
    cur_t = int(getattr(env, "current_time", 0))
    for e in getattr(env, "edges", []):
        if e.u in slot_index and e.v in slot_index:
            ga, gb = slot_index[e.u], slot_index[e.v]
            lag = int(max(0, getattr(e, "lag", 0)))
            precedence.append(PrecedenceConstraint(goal_a=ga, goal_b=gb, min_gap=lag))
    for gi, (tid, _) in enumerate(task_slots):
        t = env.tasks[int(tid)]
        ls = getattr(t, "latest_start", 0.0)
        ef = getattr(t, "earliest_finish", 0.0)
        lo = int(max(0, getattr(t, "release_time", 0))) - cur_t
        hi = int(ls) - cur_t if ls else None
        if hi is not None and hi < 0:
            continue
        absolute.append(TemporalConstraint(goal=gi, kind=ConstraintKind.ABSOLUTE,
                                           lower=max(0, lo), upper=hi))

    goals_meta = []
    dt = float(getattr(env.config, "dt_seconds", 1.0))
    for tid, _ in task_slots:
        t = env.tasks[int(tid)]
        dur = float(getattr(t, "duration_steps", 0.0) or 0.0)
        goals_meta.append({"exec_steps": int(max(0, round(dur))), "reward": 1.0})

    # Shrink-and-retry over the cheapest agent subsets until one is solvable.
    # The PTC constraints above depend only on the task slots / goals, not on
    # the agent subset, so we can reuse them across retries.
    node: Optional[CTNode] = None
    agents: List[int] = []
    decision_deadline = time.time() + max(0.0, timeout_sec)
    for keep_n in range(max_keep, 0, -1):
        keep = ranked_rows[:keep_n]
        cand_agents = [int(active_agents[int(i)]) for i in keep.tolist()]
        cand_starts = [starts[int(i)] for i in keep.tolist()]
        cand_cost = cost_matrix[keep, :]
        remaining = decision_deadline - time.time()
        if remaining <= 0.0:
            break
        node = cbs_ta_ptc_plan(
            starts=cand_starts, goals=goals, map_grid=grid_map, x_size=gx, y_size=gy,
            base_precedence=precedence, base_intergoal=intergoal, base_absolute=absolute,
            goals_meta=goals_meta, dt_seconds=dt, cost_matrix=cand_cost,
            timeout_sec=remaining,
        )
        if node is not None:
            agents = cand_agents
            break
    if node is None:
        return actions

    for ai, rid in enumerate(agents):
        gi = int(node.assignment[ai]) if ai < len(node.assignment) else -1
        if gi < 0 or gi >= len(task_slots):
            continue
        tid = int(task_slots[gi][0])
        aid = tid + 1
        if aid < action_mask.shape[1] and action_mask[int(rid), aid] > 0.5:
            actions[int(rid)] = aid
    return actions


__all__ = ["cbs_ta_ptc_plan", "cbs_ta_ptc_actions_2024", "CTNode"]
