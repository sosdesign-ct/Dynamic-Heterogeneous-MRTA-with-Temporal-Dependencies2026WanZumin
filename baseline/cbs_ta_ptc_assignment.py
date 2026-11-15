from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


DX = (-1, -1, -1, 0, 0, 1, 1, 1, 0)
DY = (-1, 0, 1, -1, 1, -1, 0, 1, 0)
FORBIDDEN_COST = 1e12
DEFAULT_CELL_SIZE = 5.0


@dataclass(frozen=True)
class GridPoint:
    x_pos: int
    y_pos: int


@dataclass
class GridPath:
    path_vect: List[GridPoint] = field(default_factory=list)
    cost: float = math.inf


@dataclass
class AssignmentPartition:
    assignment: List[int] = field(default_factory=list)
    constraints_i: List[Tuple[int, int]] = field(default_factory=list)
    constraints_o: List[Tuple[int, int]] = field(default_factory=list)
    cost: float = math.inf


@dataclass
class CBSNode:
    root: bool
    constraints: List[Tuple[int, GridPoint, int]]
    assignment: List[int]
    solution: List[GridPath]
    cost: float


def _euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _in_bounds(x: int, y: int, x_size: int, y_size: int) -> bool:
    return 1 <= x <= x_size and 1 <= y <= y_size


def _cell_cost(map_grid: Sequence[Sequence[float]], x: int, y: int) -> float:
    return float(map_grid[x - 1][y - 1])


def _is_passable(map_grid: Sequence[Sequence[float]], x: int, y: int, collision_thresh: float) -> bool:
    c = _cell_cost(map_grid, x, y)
    return 0.0 <= c < collision_thresh


def _diagonal_cost(dir_idx: int) -> float:
    if dir_idx in (0, 2, 5, 7):
        return 1.4
    if dir_idx in (1, 3, 4, 6):
        return 1.0
    return 0.0


def _rectangular_hungarian(cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
    if cost_matrix.ndim != 2:
        raise ValueError("cost_matrix must be 2D")

    n_rows, n_cols = cost_matrix.shape
    n = max(n_rows, n_cols)
    padded = np.full((n, n), FORBIDDEN_COST, dtype=np.float64)
    padded[:n_rows, :n_cols] = cost_matrix

    mat = padded.copy()
    mask = np.zeros((n, n), dtype=np.int8)
    row_cover = np.zeros((n,), dtype=np.int8)
    col_cover = np.zeros((n,), dtype=np.int8)

    mat -= mat.min(axis=1, keepdims=True)

    for i in range(n):
        for j in range(n):
            if abs(mat[i, j]) <= 1e-12 and row_cover[i] == 0 and col_cover[j] == 0:
                mask[i, j] = 1
                row_cover[i] = 1
                col_cover[j] = 1

    row_cover[:] = 0
    col_cover[:] = 0

    def cover_star_cols() -> None:
        col_cover[:] = 0
        for j in range(n):
            if np.any(mask[:, j] == 1):
                col_cover[j] = 1

    def find_uncovered_zero() -> Tuple[int, int]:
        for i in range(n):
            if row_cover[i] == 1:
                continue
            for j in range(n):
                if col_cover[j] == 0 and abs(mat[i, j]) <= 1e-12:
                    return i, j
        return -1, -1

    def find_star_in_row(r: int) -> int:
        idx = np.where(mask[r] == 1)[0]
        return int(idx[0]) if idx.size > 0 else -1

    def find_star_in_col(c: int) -> int:
        idx = np.where(mask[:, c] == 1)[0]
        return int(idx[0]) if idx.size > 0 else -1

    def find_prime_in_row(r: int) -> int:
        idx = np.where(mask[r] == 2)[0]
        return int(idx[0]) if idx.size > 0 else -1

    def clear_primes() -> None:
        mask[mask == 2] = 0

    cover_star_cols()

    while int(col_cover.sum()) < n:
        z_r, z_c = find_uncovered_zero()

        while z_r == -1:
            uncovered = mat[row_cover == 0][:, col_cover == 0]
            delta = float(np.min(uncovered)) if uncovered.size > 0 else 0.0
            mat[row_cover == 1, :] += delta
            mat[:, col_cover == 0] -= delta
            z_r, z_c = find_uncovered_zero()

        mask[z_r, z_c] = 2
        star_col = find_star_in_row(z_r)

        if star_col != -1:
            row_cover[z_r] = 1
            col_cover[star_col] = 0
            continue

        path: List[Tuple[int, int]] = [(z_r, z_c)]
        done = False
        while not done:
            r_star = find_star_in_col(path[-1][1])
            if r_star == -1:
                done = True
            else:
                path.append((r_star, path[-1][1]))
                c_prime = find_prime_in_row(r_star)
                path.append((r_star, c_prime))

        for r, c in path:
            if mask[r, c] == 1:
                mask[r, c] = 0
            else:
                mask[r, c] = 1

        row_cover[:] = 0
        col_cover[:] = 0
        clear_primes()
        cover_star_cols()

    pairs: List[Tuple[int, int]] = []
    for i in range(n_rows):
        js = np.where(mask[i] == 1)[0]
        if js.size == 0:
            continue
        j = int(js[0])
        if j < n_cols:
            pairs.append((i, j))
    return pairs


def _solve_rectangular_assignment(
    cost_matrix: np.ndarray,
    include_pairs: Sequence[Tuple[int, int]],
    exclude_pairs: Sequence[Tuple[int, int]],
) -> Optional[List[int]]:
    if cost_matrix.ndim != 2:
        raise ValueError("cost_matrix must be 2D")

    n_rows, n_cols = cost_matrix.shape
    if n_rows <= 0 or n_cols <= 0 or n_rows > n_cols:
        return None

    assignment = [-1 for _ in range(n_rows)]
    used_rows = set()
    used_cols = set()

    for r, c in include_pairs:
        if not (0 <= r < n_rows and 0 <= c < n_cols):
            return None
        if r in used_rows or c in used_cols:
            return None
        if float(cost_matrix[r, c]) >= FORBIDDEN_COST * 0.5:
            return None
        assignment[r] = c
        used_rows.add(r)
        used_cols.add(c)

    rem_rows = [r for r in range(n_rows) if r not in used_rows]
    rem_cols = [c for c in range(n_cols) if c not in used_cols]
    if len(rem_rows) > len(rem_cols):
        return None
    if not rem_rows:
        return assignment

    reduced = np.full((len(rem_rows), len(rem_cols)), FORBIDDEN_COST, dtype=np.float64)
    excluded = {(int(r), int(c)) for r, c in exclude_pairs}
    for i, r in enumerate(rem_rows):
        for j, c in enumerate(rem_cols):
            if (r, c) in excluded:
                continue
            reduced[i, j] = float(cost_matrix[r, c])

    pairs = _rectangular_hungarian(reduced)
    if len(pairs) < len(rem_rows):
        return None

    for i, j in pairs:
        r = rem_rows[i]
        c = rem_cols[j]
        if float(cost_matrix[r, c]) >= FORBIDDEN_COST * 0.5:
            return None
        assignment[r] = c

    if any(v < 0 for v in assignment):
        return None
    return assignment


def _assignment_cost(cost_matrix: np.ndarray, assignment: Sequence[int]) -> float:
    if len(assignment) != int(cost_matrix.shape[0]):
        return math.inf
    total = 0.0
    for i, j in enumerate(assignment):
        if j < 0 or j >= int(cost_matrix.shape[1]):
            return math.inf
        total += float(cost_matrix[i, j])
    return total


class AssignmentSearchGeneratorRect:
    def __init__(self):
        self._open: List[Tuple[float, int, AssignmentPartition]] = []
        self._counter = 0

    def _push(self, node: AssignmentPartition) -> None:
        heapq.heappush(self._open, (float(node.cost), int(self._counter), node))
        self._counter += 1

    def first_assignment(self, cost_matrix: np.ndarray) -> Optional[List[int]]:
        root = AssignmentPartition()
        assign = _solve_rectangular_assignment(cost_matrix, root.constraints_i, root.constraints_o)
        if assign is None:
            return None
        root.assignment = assign
        root.cost = _assignment_cost(cost_matrix, assign)
        self._push(root)
        return list(assign)

    def next_assignment(self, cost_matrix: np.ndarray) -> Optional[List[int]]:
        if not self._open:
            return None

        _, _, best = heapq.heappop(self._open)
        n_rows = int(cost_matrix.shape[0])
        included_rows = {int(r) for r, _ in best.constraints_i}

        for row_idx in range(n_rows):
            if row_idx in included_rows:
                continue
            new_include = list(best.constraints_i)
            new_exclude = list(best.constraints_o)
            new_exclude.append((row_idx, int(best.assignment[row_idx])))
            for j in range(row_idx):
                new_include.append((j, int(best.assignment[j])))
            assign = _solve_rectangular_assignment(cost_matrix, new_include, new_exclude)
            if assign is None:
                continue
            node = AssignmentPartition(
                assignment=list(assign),
                constraints_i=new_include,
                constraints_o=new_exclude,
                cost=_assignment_cost(cost_matrix, assign),
            )
            self._push(node)

        return list(best.assignment)


def _back_dijkstra(
    goals: Sequence[GridPoint],
    map_grid: Sequence[Sequence[float]],
    x_size: int,
    y_size: int,
    collision_thresh: float,
) -> List[List[List[float]]]:
    h_maps: List[List[List[float]]] = []
    for goal in goals:
        h = [[math.inf for _ in range(y_size)] for _ in range(x_size)]
        visited = [[False for _ in range(y_size)] for _ in range(x_size)]
        gx, gy = goal.x_pos, goal.y_pos
        pq: List[Tuple[float, int, int]] = []
        if _in_bounds(gx, gy, x_size, y_size):
            h[gx - 1][gy - 1] = 0.0
            pq.append((0.0, gx, gy))

        while pq:
            cur_h, x_cur, y_cur = heapq.heappop(pq)
            if visited[x_cur - 1][y_cur - 1]:
                continue
            visited[x_cur - 1][y_cur - 1] = True
            if cur_h > h[x_cur - 1][y_cur - 1]:
                continue

            for dir_idx in range(len(DX)):
                nx = x_cur + DX[dir_idx]
                ny = y_cur + DY[dir_idx]
                if not _in_bounds(nx, ny, x_size, y_size):
                    continue
                if not _is_passable(map_grid, nx, ny, collision_thresh):
                    continue
                if visited[nx - 1][ny - 1]:
                    continue
                cand = cur_h + _cell_cost(map_grid, nx, ny) + _diagonal_cost(dir_idx)
                if cand < h[nx - 1][ny - 1]:
                    h[nx - 1][ny - 1] = cand
                    heapq.heappush(pq, (cand, nx, ny))
        h_maps.append(h)
    return h_maps


def _unconstrained_search(
    h_maps: Sequence[Sequence[Sequence[float]]],
    robot_posns: Sequence[GridPoint],
    assignment: Sequence[int],
    goals: Sequence[GridPoint],
    x_size: int,
    y_size: int,
) -> List[GridPath]:
    out_paths: List[GridPath] = []
    for i, start in enumerate(robot_posns):
        goal_idx = int(assignment[i]) if i < len(assignment) else -1
        if goal_idx < 0 or goal_idx >= len(goals):
            out_paths.append(GridPath(path_vect=[start], cost=math.inf))
            continue

        current = start
        target = goals[goal_idx]
        path = [current]
        step_cap = max(10, x_size * y_size * 4)
        for _ in range(step_cap):
            if current == target:
                break
            best_h = math.inf
            best_next: Optional[GridPoint] = None
            for dir_idx in range(len(DX)):
                nx = current.x_pos + DX[dir_idx]
                ny = current.y_pos + DY[dir_idx]
                if not _in_bounds(nx, ny, x_size, y_size):
                    continue
                cand_h = h_maps[goal_idx][nx - 1][ny - 1] + _diagonal_cost(dir_idx)
                if cand_h < best_h:
                    best_h = cand_h
                    best_next = GridPoint(nx, ny)
            if best_next is None:
                break
            current = best_next
            path.append(current)
        cost = h_maps[goal_idx][start.x_pos - 1][start.y_pos - 1]
        out_paths.append(GridPath(path_vect=path, cost=cost))
    return out_paths


def _cbs_okay(constraints: Sequence[Tuple[int, GridPoint, int]], newx: int, newy: int, newt: int, agent_idx: int) -> bool:
    for c_agent, c_pos, c_time in constraints:
        if c_agent == agent_idx and c_time == newt and c_pos.x_pos == newx and c_pos.y_pos == newy:
            return False
    return True


def _constrained_search(
    h_maps: Sequence[Sequence[Sequence[float]]],
    robot_posn: GridPoint,
    robot_index: int,
    goal_index: int,
    goal_point: GridPoint,
    temp_constr: Sequence[Tuple[int, GridPoint, int]],
    x_size: int,
    y_size: int,
    map_grid: Sequence[Sequence[float]],
    collision_thresh: float,
    max_time_steps: int,
    max_expanded_states: int,
    deadline_ts: Optional[float],
) -> GridPath:
    constraints = [c for c in temp_constr if c[0] == robot_index]
    time_max = max((c[2] for c in constraints), default=0)

    start_key = (robot_posn.x_pos, robot_posn.y_pos, 0)
    open_heap: List[Tuple[float, float, int, int, int]] = []
    parent: Dict[Tuple[int, int, int], Optional[Tuple[int, int, int]]] = {start_key: None}
    best_g: Dict[Tuple[int, int, int], float] = {start_key: 0.0}

    start_h = h_maps[goal_index][robot_posn.x_pos - 1][robot_posn.y_pos - 1]
    heapq.heappush(open_heap, (start_h, 0.0, robot_posn.x_pos, robot_posn.y_pos, 0))

    expanded_states = 0
    while open_heap:
        if deadline_ts is not None and time.time() >= float(deadline_ts):
            break
        if expanded_states >= int(max_expanded_states):
            break

        _, g_val, x_cur, y_cur, t_cur = heapq.heappop(open_heap)
        expanded_states += 1
        cur_key = (x_cur, y_cur, t_cur)
        if g_val > best_g.get(cur_key, math.inf):
            continue

        if x_cur == goal_point.x_pos and y_cur == goal_point.y_pos and t_cur >= time_max:
            path_points: List[GridPoint] = []
            trace: Optional[Tuple[int, int, int]] = cur_key
            while trace is not None:
                path_points.append(GridPoint(trace[0], trace[1]))
                trace = parent[trace]
            path_points.reverse()
            return GridPath(path_vect=path_points, cost=g_val)

        if t_cur >= int(max_time_steps):
            continue

        for dir_idx in range(len(DX)):
            nx = x_cur + DX[dir_idx]
            ny = y_cur + DY[dir_idx]
            nt = t_cur + 1
            if nt > int(max_time_steps):
                continue
            if not _in_bounds(nx, ny, x_size, y_size):
                continue
            if not _is_passable(map_grid, nx, ny, collision_thresh):
                continue
            if not _cbs_okay(constraints, nx, ny, nt, robot_index):
                continue

            ng = g_val + _cell_cost(map_grid, nx, ny) + _diagonal_cost(dir_idx)
            nkey = (nx, ny, nt)
            if ng >= best_g.get(nkey, math.inf):
                continue

            best_g[nkey] = ng
            parent[nkey] = cur_key
            nh = h_maps[goal_index][nx - 1][ny - 1]
            heapq.heappush(open_heap, (ng + nh, ng, nx, ny, nt))

    return GridPath(path_vect=[robot_posn], cost=math.inf)


def _check_conflict(node: CBSNode) -> Tuple[bool, Optional[Tuple[int, GridPoint, int]], Optional[Tuple[int, GridPoint, int]]]:
    sol = node.solution
    num_agents = len(sol)
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            agent1 = sol[i].path_vect
            agent2 = sol[j].path_vect
            m = min(len(agent1), len(agent2))
            for k in range(m):
                if agent1[k] == agent2[k]:
                    return False, (i, agent1[k], k), (j, agent1[k], k)
                if k > 0:
                    if agent1[k] == agent2[k - 1] and agent2[k] == agent1[k - 1]:
                        return False, (i, agent1[k], k), (j, agent2[k], k)
                    mid1x = (agent1[k].x_pos + agent1[k - 1].x_pos) / 2.0
                    mid1y = (agent1[k].y_pos + agent1[k - 1].y_pos) / 2.0
                    mid2x = (agent2[k].x_pos + agent2[k - 1].x_pos) / 2.0
                    mid2y = (agent2[k].y_pos + agent2[k - 1].y_pos) / 2.0
                    if mid1x == mid2x and mid1y == mid2y:
                        return False, (i, agent1[k], k), (j, agent2[k], k)
    return True, None, None


def _get_sic(node: CBSNode) -> float:
    return float(sum(path.cost for path in node.solution))


def _lengthen_solution(paths: List[GridPath]) -> None:
    if not paths:
        return
    max_len = max(len(p.path_vect) for p in paths)
    for p in paths:
        if not p.path_vect:
            continue
        last = p.path_vect[-1]
        while len(p.path_vect) < max_len:
            p.path_vect.append(last)


def cbs_ota_plan_rect(
    starts: Sequence[GridPoint],
    goals: Sequence[GridPoint],
    map_grid: Sequence[Sequence[float]],
    collision_thresh: float,
    assignment_cost_matrix: Optional[np.ndarray] = None,
    timeout_sec: float = 5.0,
) -> Optional[CBSNode]:
    num_agents = len(starts)
    num_goals = len(goals)
    if num_agents == 0 or num_goals == 0 or num_agents > num_goals:
        return None

    x_size = len(map_grid)
    y_size = len(map_grid[0]) if x_size > 0 else 0
    h_maps = _back_dijkstra(goals, map_grid, x_size, y_size, collision_thresh)

    if assignment_cost_matrix is None:
        cost_matrix = np.zeros((num_agents, num_goals), dtype=np.float64)
        for i, start in enumerate(starts):
            for j in range(num_goals):
                cost_matrix[i, j] = float(h_maps[j][start.x_pos - 1][start.y_pos - 1])
    else:
        cost_matrix = np.asarray(assignment_cost_matrix, dtype=np.float64)
        if cost_matrix.shape != (num_agents, num_goals):
            raise ValueError("assignment_cost_matrix shape mismatch")

    asg_open = AssignmentSearchGeneratorRect()
    assignment0 = asg_open.first_assignment(cost_matrix)
    if assignment0 is None:
        return None

    root_solutions = _unconstrained_search(h_maps, starts, assignment0, goals, x_size, y_size)
    _lengthen_solution(root_solutions)
    root = CBSNode(root=True, constraints=[], assignment=list(assignment0), solution=root_solutions, cost=0.0)
    root.cost = _get_sic(root)

    open_heap: List[Tuple[float, int, CBSNode]] = []
    order = 0
    heapq.heappush(open_heap, (root.cost, order, root))
    order += 1

    seen_assignments = {tuple(assignment0)}
    t0 = time.time()
    deadline_ts = t0 + float(timeout_sec)
    while open_heap and (time.time() - t0) <= timeout_sec:
        _, _, curr = heapq.heappop(open_heap)
        no_conflict, conflict1, conflict2 = _check_conflict(curr)

        if no_conflict:
            return curr

        if curr.root:
            # Match the original planner_unlinked.cpp structure: when a root
            # assignment still has path conflicts, generate the next assignment
            # from the ASG queue after replacing the selected assignment costs
            # by the realized path costs of the current CBS root solution.
            cost_matrix_next = cost_matrix.copy()
            for agent_idx, goal_idx in enumerate(curr.assignment):
                if 0 <= int(goal_idx) < int(cost_matrix_next.shape[1]):
                    cost_matrix_next[agent_idx, int(goal_idx)] = float(curr.solution[agent_idx].cost)
            next_assignment = asg_open.next_assignment(cost_matrix_next)
            if next_assignment is not None and tuple(next_assignment) not in seen_assignments:
                seen_assignments.add(tuple(next_assignment))
                nsol = _unconstrained_search(h_maps, starts, next_assignment, goals, x_size, y_size)
                _lengthen_solution(nsol)
                nnode = CBSNode(root=True, constraints=[], assignment=list(next_assignment), solution=nsol, cost=0.0)
                nnode.cost = _get_sic(nnode)
                heapq.heappush(open_heap, (nnode.cost, order, nnode))
                order += 1

        if conflict1 is None or conflict2 is None:
            continue

        goal_points = [goals[g] for g in curr.assignment]
        for conflict in (conflict1, conflict2):
            child_constraints = list(curr.constraints)
            child_constraints.append(conflict)

            child_paths: List[GridPath] = []
            infeasible_child = False
            for agent_idx in range(num_agents):
                constraints_per_agent = [c for c in child_constraints if c[0] == agent_idx]
                if not constraints_per_agent:
                    child_paths.append(curr.solution[agent_idx])
                    continue

                goal_idx = int(curr.assignment[agent_idx])
                max_time_steps = int(
                    np.clip(
                        x_size + y_size + 2 * len(constraints_per_agent) + 24,
                        32,
                        x_size * y_size + 64,
                    )
                )
                max_expanded_states = int(
                    np.clip(
                        max_time_steps * max(1, x_size * y_size // 2),
                        20_000,
                        200_000,
                    )
                )
                p = _constrained_search(
                    h_maps=h_maps,
                    robot_posn=starts[agent_idx],
                    robot_index=agent_idx,
                    goal_index=goal_idx,
                    goal_point=goal_points[agent_idx],
                    temp_constr=constraints_per_agent,
                    x_size=x_size,
                    y_size=y_size,
                    map_grid=map_grid,
                    collision_thresh=collision_thresh,
                    max_time_steps=max_time_steps,
                    max_expanded_states=max_expanded_states,
                    deadline_ts=deadline_ts,
                )
                if not math.isfinite(float(p.cost)):
                    infeasible_child = True
                    break
                child_paths.append(p)

            if infeasible_child:
                continue

            _lengthen_solution(child_paths)
            child = CBSNode(
                root=False,
                constraints=child_constraints,
                assignment=list(curr.assignment),
                solution=child_paths,
                cost=0.0,
            )
            child.cost = _get_sic(child)
            heapq.heappush(open_heap, (child.cost, order, child))
            order += 1

    return None


def _candidate_tasks(env) -> List[int]:
    out: List[int] = []
    for tid, t in env.tasks.items():
        if tid >= env.config.max_tasks:
            continue
        if t.status in {"AVAILABLE", "ASSIGNED", "READY"}:
            out.append(int(tid))
    return out


def _grid_dims(env) -> Tuple[int, int]:
    area_x = float(env.config.area_size[0])
    area_y = float(env.config.area_size[1])
    gx = max(8, int(math.ceil(area_x / DEFAULT_CELL_SIZE)) + 1)
    gy = max(8, int(math.ceil(area_y / DEFAULT_CELL_SIZE)) + 1)
    return gx, gy


def _continuous_to_grid(pos: np.ndarray, env, gx: int, gy: int) -> GridPoint:
    area_x = max(float(env.config.area_size[0]), 1e-6)
    area_y = max(float(env.config.area_size[1]), 1e-6)
    x = int(round(float(pos[0]) / area_x * float(gx - 1))) + 1
    y = int(round(float(pos[1]) / area_y * float(gy - 1))) + 1
    x = min(max(1, x), gx)
    y = min(max(1, y), gy)
    return GridPoint(x, y)


def _assign_unique_cell(target: GridPoint, gx: int, gy: int, occupied: set) -> GridPoint:
    if (target.x_pos, target.y_pos) not in occupied:
        occupied.add((target.x_pos, target.y_pos))
        return target

    for radius in range(1, max(gx, gy) + 1):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx = target.x_pos + dx
                ny = target.y_pos + dy
                if not _in_bounds(nx, ny, gx, gy):
                    continue
                if (nx, ny) in occupied:
                    continue
                occupied.add((nx, ny))
                return GridPoint(nx, ny)

    occupied.add((target.x_pos, target.y_pos))
    return target


def _build_task_slots(env, task_ids: Sequence[int], action_mask: np.ndarray, active_agents: Sequence[int]) -> List[Tuple[int, GridPoint]]:
    gx, gy = _grid_dims(env)
    occupied = set()
    slots: List[Tuple[int, GridPoint]] = []
    for tid in task_ids:
        task = env.tasks.get(int(tid))
        if task is None:
            continue
        remain = max(int(task.k) - len(task.assigned_robots), 1)
        goal_base = _continuous_to_grid(task.loc, env, gx, gy)
        for _ in range(remain):
            goal = _assign_unique_cell(goal_base, gx, gy, occupied)
            slots.append((int(tid), goal))
    return slots


def _build_assignment_cost_matrix(
    env,
    active_agents: Sequence[int],
    task_slots: Sequence[Tuple[int, GridPoint]],
    action_mask: np.ndarray,
) -> Tuple[np.ndarray, List[GridPoint], List[GridPoint]]:
    """Build the CBS-OTA assignment matrix in the same spirit as the original code.

    The MATLAB/C++ implementation computes each entry from the reverse-Dijkstra
    grid heuristic between a robot start and a candidate goal, then masks
    infeasible task-agent pairs through the assignment constraints.  For fair
    MRTA evaluation we only use the environment action mask to forbid pairs that
    violate capability, capacity, release/dependency, or commitment constraints;
    we intentionally do not add reward-shaped terms such as criticality bonuses.
    """
    gx, gy = _grid_dims(env)
    starts: List[GridPoint] = []
    goals: List[GridPoint] = [goal for _, goal in task_slots]

    start_occupied = set()
    for rid in active_agents:
        rob = env.robots[int(rid)]
        base = _continuous_to_grid(rob.pos, env, gx, gy)
        starts.append(_assign_unique_cell(base, gx, gy, start_occupied))

    grid_map = [[0.0 for _ in range(gy)] for _ in range(gx)]
    h_maps = _back_dijkstra(goals, grid_map, gx, gy, collision_thresh=1.0)

    cost = np.full((len(active_agents), len(task_slots)), FORBIDDEN_COST, dtype=np.float64)
    for ai, rid in enumerate(active_agents):
        rob = env.robots.get(int(rid))
        if rob is None:
            continue
        rpos = (float(rob.pos[0]), float(rob.pos[1]))
        vmax = max(float(getattr(rob, "v_max", 1.0)), 1e-6)

        for sj, (tid, _) in enumerate(task_slots):
            action_id = int(tid) + 1
            if action_id >= action_mask.shape[1] or action_mask[int(rid), action_id] <= 0.5:
                continue

            task = env.tasks.get(int(tid))
            if task is None:
                continue

            grid_cost = float(h_maps[sj][starts[ai].x_pos - 1][starts[ai].y_pos - 1])
            if not math.isfinite(grid_cost):
                continue

            cost[ai, sj] = grid_cost

    return cost, starts, goals


def _fallback_actions(env, action_mask: np.ndarray, active_agents: Sequence[int], task_slots: Sequence[Tuple[int, GridPoint]]) -> Dict[int, int]:
    actions = {rid: 0 for rid in range(int(env.config.num_agents))}
    if not active_agents or not task_slots:
        return actions

    cost = np.full((len(active_agents), len(task_slots)), FORBIDDEN_COST, dtype=np.float64)
    for ai, rid in enumerate(active_agents):
        rob = env.robots[int(rid)]
        rpos = (float(rob.pos[0]), float(rob.pos[1]))
        vmax = max(float(getattr(rob, "v_max", 1.0)), 1e-6)
        for sj, (tid, _) in enumerate(task_slots):
            action_id = int(tid) + 1
            if action_id >= action_mask.shape[1] or action_mask[int(rid), action_id] <= 0.5:
                continue
            task = env.tasks[int(tid)]
            tpos = (float(task.loc[0]), float(task.loc[1]))
            dist = _euclid(rpos, tpos)
            eta_steps = dist / max(vmax * max(float(env.config.dt_seconds), 1e-6), 1e-6)
            cost[ai, sj] = dist + 0.25 * eta_steps

    pairs = _rectangular_hungarian(cost)
    for ai, sj in pairs:
        if ai < 0 or ai >= len(active_agents) or sj < 0 or sj >= len(task_slots):
            continue
        if cost[ai, sj] >= FORBIDDEN_COST * 0.5:
            continue
        rid = int(active_agents[ai])
        tid = int(task_slots[sj][0])
        actions[rid] = tid + 1
    return actions


def cbs_ta_ptc_actions(env, action_mask: np.ndarray, timeout_sec: float = 10.0) -> Dict[int, int]:
    num_agents = int(env.config.num_agents)
    actions = {rid: 0 for rid in range(num_agents)}

    task_ids = _candidate_tasks(env)
    if not task_ids:
        return actions

    active_agents = [rid for rid in range(num_agents) if float(np.sum(action_mask[rid, 1:])) > 0.5]
    if not active_agents:
        return actions

    task_slots = _build_task_slots(env, task_ids, action_mask, active_agents)
    if not task_slots:
        return actions

    feasible_slots = []
    for tid, goal in task_slots:
        if any((int(tid) + 1) < action_mask.shape[1] and action_mask[rid, int(tid) + 1] > 0.5 for rid in active_agents):
            feasible_slots.append((tid, goal))
    task_slots = feasible_slots
    if not task_slots:
        return actions

    cost_matrix, starts, goals = _build_assignment_cost_matrix(env, active_agents, task_slots, action_mask)
    if cost_matrix.shape[0] <= 0 or cost_matrix.shape[1] <= 0:
        return actions

    finite_thresh = FORBIDDEN_COST * 0.5
    finite_row_mask = np.any(cost_matrix < finite_thresh, axis=1)
    if not bool(np.any(finite_row_mask)):
        return actions

    row_idx = np.where(finite_row_mask)[0]
    if row_idx.size > cost_matrix.shape[1]:
        # CBS-TA-PTC here solves a one-goal-per-agent assignment; when agents exceed
        # task slots, keep the most promising agents for this planning round.
        best_cost = np.min(cost_matrix[row_idx, :], axis=1)
        keep_order = np.argsort(best_cost)[: cost_matrix.shape[1]]
        row_idx = row_idx[keep_order]

    active_agents = [int(active_agents[int(i)]) for i in row_idx.tolist()]
    starts = [starts[int(i)] for i in row_idx.tolist()]
    cost_matrix = cost_matrix[row_idx, :]

    # The original CBS-OTA planner assumes a one-goal-per-agent planning
    # instance.  In the online MRTA snapshot, the number of active robots can be
    # larger than the number of immediately committable task slots and some
    # robot-task pairs are forbidden by the common action mask.  The adapter
    # therefore selects the largest currently solvable active-robot subset and
    # applies the original CBS-OTA core to that subset.  This is an interface
    # adaptation only: the assignment cost remains reverse-Dijkstra grid cost,
    # and no reward/criticality information from our learned method is used.
    base_row_order = np.argsort(np.min(cost_matrix, axis=1))
    gx, gy = _grid_dims(env)
    grid_map = [[0.0 for _ in range(gy)] for _ in range(gx)]
    node = None
    plan_agents: List[int] = []
    max_rows = int(min(cost_matrix.shape[0], cost_matrix.shape[1]))
    planning_deadline = time.time() + max(float(timeout_sec), 0.0)
    for keep_n in range(max_rows, 0, -1):
        remaining_timeout = planning_deadline - time.time()
        if remaining_timeout <= 0.0:
            break
        sel = base_row_order[:keep_n]
        cm_try = cost_matrix[sel, :]
        starts_try = [starts[int(i)] for i in sel.tolist()]
        node_try = cbs_ota_plan_rect(
            starts=starts_try,
            goals=goals,
            map_grid=grid_map,
            collision_thresh=1.0,
            assignment_cost_matrix=cm_try,
            timeout_sec=remaining_timeout,
        )
        if node_try is not None:
            node = node_try
            plan_agents = [int(active_agents[int(i)]) for i in sel.tolist()]
            break

    if node is None:
        # Strict CBS-TA-PTC mode: do not fall back to extra heuristics.
        # If planning fails within timeout, keep NOOP actions for this step.
        return actions

    for ai, rid in enumerate(plan_agents):
        goal_idx = int(node.assignment[ai]) if ai < len(node.assignment) else -1
        if goal_idx < 0 or goal_idx >= len(task_slots):
            continue
        tid = int(task_slots[goal_idx][0])
        action_id = tid + 1
        if action_id < action_mask.shape[1] and action_mask[int(rid), action_id] > 0.5:
            actions[int(rid)] = action_id

    return actions


__all__ = ["cbs_ta_ptc_actions", "cbs_ota_plan_rect", "CBSNode", "GridPoint", "GridPath"]