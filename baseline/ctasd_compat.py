from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


class _FallbackTSPSolver:
    """Small mTSP-compatible fallback used when the original CTAS-D dependencies are unavailable."""

    def run_solver(self, coords: Sequence[Sequence[float]], num_vehicles: int = 1, depot: int = 0):
        pts = np.asarray(coords, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] <= 1:
            return {i: [int(depot)] for i in range(max(1, int(num_vehicles)))}, 0.0

        n_veh = max(1, int(num_vehicles))
        depot_idx = int(depot)
        routes: Dict[int, List[int]] = {i: [] for i in range(n_veh)}
        route_cost = np.zeros((n_veh,), dtype=np.float64)
        route_pos = np.repeat(pts[depot_idx : depot_idx + 1, :2], n_veh, axis=0)
        remaining = [i for i in range(pts.shape[0]) if i != depot_idx]

        while remaining:
            best: Tuple[float, int, int] | None = None
            for vi in range(n_veh):
                for node in remaining:
                    dist = float(np.linalg.norm(route_pos[vi] - pts[node, :2]))
                    workload = float(pts[node, 2]) if pts.shape[1] > 2 else 0.0
                    score = float(route_cost[vi] + dist + max(0.0, workload))
                    if best is None or score < best[0]:
                        best = (score, vi, node)
            if best is None:
                break
            _, vi, node = best
            route_cost[vi] += float(np.linalg.norm(route_pos[vi] - pts[node, :2]))
            if pts.shape[1] > 2:
                route_cost[vi] += max(0.0, float(pts[node, 2]))
            route_pos[vi] = pts[node, :2]
            routes[vi].append(int(node))
            remaining.remove(node)

        for vi in range(n_veh):
            routes[vi].append(depot_idx)
        return routes, float(np.max(route_cost) if route_cost.size else 0.0)


_SRC = Path(__file__).resolve().parent / "CTAS-D.py"
_SPEC = spec_from_file_location("baseline_ctasd_module", _SRC)
try:
    if _SPEC is None or _SPEC.loader is None:
        raise ImportError(f"Unable to load CTAS-D module from {_SRC}")
    _MODULE = module_from_spec(_SPEC)
    _SPEC.loader.exec_module(_MODULE)
    TSPSolver = _MODULE.TSPSolver
except Exception:
    TSPSolver = _FallbackTSPSolver

__all__ = ["TSPSolver"]
