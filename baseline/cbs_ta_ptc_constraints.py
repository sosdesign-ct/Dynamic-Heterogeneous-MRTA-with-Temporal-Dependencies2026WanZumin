# -*- coding: utf-8 -*-
"""Precedence / Temporal Constraint (PTC) definitions for CBS-TA-PTC.

Faithful reproduction of the constraint layer from:

    Yu Quan Chong, Jiaoyang Li, Katia Sycara.
    "Optimal Task Assignment and Path Planning using Conflict-Based Search
    with Precedence and Temporal Constraints." AAMAS 2024 (arXiv:2402.08772).

The paper augments classic CBS vertex/edge constraints with three additional
constraint families, resolved in a fixed priority order:

    1) Absolute temporal range constraints  (T_abs)
    2) Precedence constraints               (T_prec, from CBS-PC)
    3) Inter-goal temporal range constraints(T_inter)
    4) Vertex / Edge constraints            (classic CBS)

We map these onto our MRTA environment as follows (decided with the user):

* Precedence constraints        <- env.edges : DependencyEdge(u, v, rel_type, lag)
                                   rel_type in {FS, SS, FF, SF}.
* Inter-goal temporal ranges    <- DependencyEdge lag bounds (relative spacing).
* Absolute temporal ranges      <- Task.{earliest_start, latest_start,
                                   earliest_finish, latest_finish} per task.

Every constraint is expressed over *goal completion timesteps* tau(g), which
in our adaptation is the timestep at which the agent finishes the task slot
``g`` (a (task_id, GridPoint) goal).  A path plus the environment's temporal
rules induces a completion time per goal; constraints bound these times.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


class ConstraintKind(IntEnum):
    """Resolution priority order (lower value = higher priority)."""

    ABSOLUTE = 0
    PRECEDENCE = 1
    INTERGOAL = 2
    VERTEX = 3
    EDGE = 4


@dataclass(frozen=True)
class TemporalConstraint:
    """A bound on a goal's completion timestep tau(goal).

    ``goal`` identifies a task slot (its index within the subtask).
    ``kind`` gives the constraint family (priority).
    ``lower`` / ``upper`` are inclusive timestep bounds; ``None`` = unbounded.
    """

    goal: int
    kind: ConstraintKind
    lower: Optional[int] = None
    upper: Optional[int] = None

    def is_satisfied(self, tau: int) -> bool:
        if self.lower is not None and tau < self.lower:
            return False
        if self.upper is not None and tau > self.upper:
            return False
        return True


@dataclass(frozen=True)
class PrecedenceConstraint:
    """tau(goal_b) - tau(goal_a) >= min_gap  (a must precede b by >= min_gap).

    Derived from DependencyEdge(u, v, rel_type, lag): the successor goal's
    relevant timestamp must be at least ``lag`` after the predecessor's.
    For planning we reduce every relation to a minimum separation between the
    two goals' completion times.
    """

    goal_a: int
    goal_b: int
    min_gap: int
    kind: ConstraintKind = ConstraintKind.PRECEDENCE


@dataclass(frozen=True)
class InterGoalConstraint:
    """tau(goal_b) - tau(goal_a) <= t_inter  (b at most t_inter after a)."""

    goal_a: int
    goal_b: int
    t_inter: int
    kind: ConstraintKind = ConstraintKind.INTERGOAL


__all__ = [
    "ConstraintKind",
    "TemporalConstraint",
    "PrecedenceConstraint",
    "InterGoalConstraint",
]
