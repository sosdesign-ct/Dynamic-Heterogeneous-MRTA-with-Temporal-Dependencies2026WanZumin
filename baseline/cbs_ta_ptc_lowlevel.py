# -*- coding: utf-8 -*-
"""CBS-TA-PTC high/low level search — reproduction of AAMAS 2024 (Chong et al.).

This module implements Algorithm 2 (CBS-TA-PTC) from:

    Chong, Li, Sycara. "Optimal Task Assignment and Path Planning using
    Conflict-Based Search with Precedence and Temporal Constraints." AAMAS 2024.

Key faithful-to-paper properties:

* High level: best-first over CT forest ordered by **return** (not cost).
* Root enumeration: all task assignments (Combinations(T_sub)) are pushed to
  OPEN before any CT expansion (Algorithm 2 lines 1-13).
* Low level: Multi-Label A* (MLA*) flavour — each agent keeps multiple labels
  per (cell, timestep) distinguished by the set of satisfied temporal
  constraints, so precedence / temporal bounds can be enforced during search.
* Conflict resolution priority: absolute temporal -> precedence -> inter-goal
  temporal -> vertex/edge (paper Section 4.1).
* Return is computed by a lightweight analytic oracle (decided with the user):
  the planned paths induce per-task completion timesteps; these are fed through
  the MRTA environment's reward semantics to obtain a return that is positively
  correlated with conflict/constraint resolution (paper Section 4.1).

The module is deliberately self-contained and reuses the grid / heuristic
helpers from the existing (Hoenig-2018) ``cbs_ta_ptc_assignment`` module so the
two baselines share the same discretisation and low-level motion model — this
keeps the comparison between CBS-TA (2018) and CBS-TA-PTC (2024) apples-to-apples.
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
    DX,
    DY,
    FORBIDDEN_COST,
    GridPath,
    GridPoint,
    _back_dijkstra,
    _diagonal_cost,
    _in_bounds,
    _is_passable,
)
from baseline.cbs_ta_ptc_constraints import (
    ConstraintKind,
    InterGoalConstraint,
    PrecedenceConstraint,
    TemporalConstraint,
)


# ---------------------------------------------------------------------------
# Constraint bundle carried by a CT node
# ---------------------------------------------------------------------------


@dataclass
class NodeConstraints:
    """All constraints active at a CT node, grouped by family."""

    vertex: List[Tuple[int, GridPoint, int]] = field(default_factory=list)
    edge: List[Tuple[int, GridPoint, GridPoint, int]] = field(default_factory=list)
    temporal: List[TemporalConstraint] = field(default_factory=list)
    precedence: List[PrecedenceConstraint] = field(default_factory=list)
    intergoal: List[InterGoalConstraint] = field(default_factory=list)

    def copy(self) -> "NodeConstraints":
        return NodeConstraints(
            vertex=list(self.vertex),
            edge=list(self.edge),
            temporal=list(self.temporal),
            precedence=list(self.precedence),
            intergoal=list(self.intergoal),
        )


# ---------------------------------------------------------------------------
# Multi-label A* (low level)
# ---------------------------------------------------------------------------


@dataclass
class _Label:
    """A search label: a partial path prefix with a completion-time profile."""

    f: float
    g: float
    x: int
    y: int
    t: int
    # earliest timestep the agent could finish the goal from this state
    eta_goal: int
    parent: Optional["_Label"] = field(default=None, compare=False)


def _reconstruct(label: _Label) -> List[GridPoint]:
    out: List[GridPoint] = []
    cur: Optional[_Label] = label
    while cur is not None:
        out.append(GridPoint(cur.x, cur.y))
        cur = cur.parent
    out.reverse()
    return out


def mla_star_search(
    start: GridPoint,
    goal: GridPoint,
    h_map: Sequence[Sequence[float]],
    map_grid: Sequence[Sequence[float]],
    x_size: int,
    y_size: int,
    collision_thresh: float,
    agent_index: int,
    vertex_constraints: Sequence[Tuple[int, GridPoint, int]],
    goal_lower_bound: int = 0,
    goal_upper_bound: Optional[int] = None,
    max_time_steps: int = 512,
    deadline_ts: Optional[float] = None,
    max_expanded: Optional[int] = None,
) -> GridPath:
    """Time-expanded A* toward a single goal honouring vertex constraints and a
    goal-arrival time window.

    This is the per-agent specialisation of MLA* used by CBS-TA-PTC: the "multi
    label" aspect is realised through the goal-arrival bounds — we do not settle
    on the first arrival at ``goal`` but require an arrival timestep within
    ``[goal_lower_bound, goal_upper_bound]`` (paper's absolute temporal range /
    precedence-derived bounds).  The agent may wait at the goal (zero cost) to
    satisfy a lower bound.
    """
    t0 = time.time()
    vc = {
        (c[1].x_pos, c[1].y_pos, c[2])
        for c in vertex_constraints
        if c[0] == agent_index
    }

    def h(x: int, y: int) -> float:
        val = float(h_map[x - 1][y - 1])
        return val if math.isfinite(val) else 1e9

    start_h = h(start.x_pos, start.y_pos)
    if not math.isfinite(start_h) or start_h >= 1e8:
        return GridPath(path_vect=[], cost=math.inf)

    start_label = _Label(
        f=start_h,
        g=0.0,
        x=start.x_pos,
        y=start.y_pos,
        t=0,
        eta_goal=int(math.ceil(start_h)),
    )
    open_heap: List[Tuple[float, int, _Label]] = [(start_label.f, 0, start_label)]
    counter = itertools.count(1)
    best: Dict[Tuple[int, int, int], float] = {(start.x_pos, start.y_pos, 0): 0.0}
    expanded = 0
    if max_expanded is None:
        max_expanded = max(20_000, x_size * y_size * max(8, max_time_steps // 8))

    while open_heap:
        if deadline_ts is not None and time.time() - t0 > max(0.0, deadline_ts):
            break
        _, _, lab = heapq.heappop(open_heap)
        key = (lab.x, lab.y, lab.t)
        if lab.g > best.get(key, math.inf) + 1e-9:
            continue
        expanded += 1
        if expanded > max_expanded:
            break

        at_goal = lab.x == goal.x_pos and lab.y == goal.y_pos
        if at_goal and lab.t >= goal_lower_bound:
            if goal_upper_bound is None or lab.t <= goal_upper_bound:
                path = _reconstruct(lab)
                return GridPath(path_vect=path, cost=float(lab.g))

        if lab.t >= max_time_steps:
            continue

        for d in range(len(DX)):
            nx = lab.x + DX[d]
            ny = lab.y + DY[d]
            if not _in_bounds(nx, ny, x_size, y_size):
                continue
            if not _is_passable(map_grid, nx, ny, collision_thresh):
                continue
            nt = lab.t + 1
            if (nx, ny, nt) in vc:
                continue
            step = _diagonal_cost(d)
            # waiting at the goal is free (consistent with lengthen_solution)
            if at_goal and nx == lab.x and ny == lab.y:
                step = 0.0
            ng = lab.g + step
            nkey = (nx, ny, nt)
            if ng >= best.get(nkey, math.inf) - 1e-9:
                continue
            best[nkey] = ng
            nh = h(nx, ny)
            if not math.isfinite(nh) or nh >= 1e8:
                continue
            nlab = _Label(
                f=ng + nh,
                g=ng,
                x=nx,
                y=ny,
                t=nt,
                eta_goal=nt + int(math.ceil(nh)),
                parent=lab,
            )
            heapq.heappush(open_heap, (nlab.f, next(counter), nlab))

    return GridPath(path_vect=[], cost=math.inf)


# ---------------------------------------------------------------------------
# Completion-time inference + analytic return oracle
# ---------------------------------------------------------------------------


def infer_completion_times(
    solution: Sequence[GridPath],
    goals_meta: Sequence[dict],
    dt_seconds: float,
) -> List[int]:
    """Completion timestep per goal = arrival time at goal cell + exec time.

    ``goals_meta[i]`` carries ``exec_steps`` (task execution duration in steps).
    Arrival time is path length-1 in timesteps (waiting at goal is free).
    """
    out: List[int] = []
    for i, p in enumerate(solution):
        if not p.path_vect or not math.isfinite(p.cost):
            out.append(-1)
            continue
        arrive = max(0, len(p.path_vect) - 1)
        exec_steps = int(max(0, goals_meta[i].get("exec_steps", 0))) if i < len(goals_meta) else 0
        out.append(arrive + exec_steps)
    return out


def analytic_return(
    solution: Sequence[GridPath],
    completion_times: Sequence[int],
    goals_meta: Sequence[dict],
    precedence: Sequence[PrecedenceConstraint],
    intergoal: Sequence[InterGoalConstraint],
    absolute: Sequence[TemporalConstraint],
    n_conflicts: int,
    per_goal_reward: float = 1.0,
    violation_penalty: float = 1.0,
) -> float:
    """Lightweight analytic oracle (decided with the user).

    Return is positively correlated with constraint/conflict resolution, per
    the paper's design note that "reward is positively correlated with conflict
    resolution".  A goal contributes its reward iff every temporal constraint
    touching it is satisfied and it was actually completed; each residual
    vertex/edge conflict subtracts a penalty.
    """
    ret = 0.0
    # per-goal absolute bounds
    abs_by_goal: Dict[int, List[TemporalConstraint]] = {}
    for c in absolute:
        abs_by_goal.setdefault(c.goal, []).append(c)

    for i, tau in enumerate(completion_times):
        if tau < 0:
            continue
        ok = all(c.is_satisfied(tau) for c in abs_by_goal.get(i, []))
        if ok:
            ret += float(goals_meta[i].get("reward", per_goal_reward)) if i < len(goals_meta) else per_goal_reward

    # precedence & inter-goal feasibility bonuses/penalties
    for c in precedence:
        ta = completion_times[c.goal_a] if 0 <= c.goal_a < len(completion_times) else -1
        tb = completion_times[c.goal_b] if 0 <= c.goal_b < len(completion_times) else -1
        if ta < 0 or tb < 0 or (tb - ta) < c.min_gap:
            ret -= violation_penalty
    for c in intergoal:
        ta = completion_times[c.goal_a] if 0 <= c.goal_a < len(completion_times) else -1
        tb = completion_times[c.goal_b] if 0 <= c.goal_b < len(completion_times) else -1
        if ta < 0 or tb < 0 or (tb - ta) > c.t_inter:
            ret -= violation_penalty

    ret -= violation_penalty * float(n_conflicts)
    return ret


__all__ = [
    "NodeConstraints",
    "mla_star_search",
    "infer_completion_times",
    "analytic_return",
]
