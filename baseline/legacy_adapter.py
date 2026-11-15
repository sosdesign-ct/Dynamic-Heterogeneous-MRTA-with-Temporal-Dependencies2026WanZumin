from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from baseline.original_baseline_adapter import OriginalBaselineAdapter


@dataclass
class LegacyBaselineAdapter:
    """Adapter that maps baseline names to minimally adapted original implementations.

    Supported methods:
    - cbs_ta_ptc: original-style CBS-TA-PTC planning adapter
    - ga: original-style planning baseline without reward-shaped heuristics
    - ctas_d: original CTAS-D route planner adapter
    """

    method: str = "cbs_ta_ptc"
    timeout_sec: float = 10.0
    _impl: OriginalBaselineAdapter | None = None

    def __post_init__(self):
        self.method = self.method.lower()
        if self.method not in {"cbs_ta_ptc", "ga", "ctas_d"}:
            raise ValueError(f"Unsupported legacy baseline method: {self.method}")
        self._impl = OriginalBaselineAdapter(method=self.method, timeout_sec=float(self.timeout_sec))

    def act(self, env) -> Dict[int, int]:
        if self._impl is None:
            raise ValueError(f"Unsupported legacy baseline method: {self.method}")
        return self._impl.act(env)

    def reset_episode(self) -> None:
        if self._impl is not None:
            self._impl.reset_episode()

    def get_episode_stats(self) -> Dict[str, float]:
        if self._impl is None:
            return {}
        return self._impl.get_episode_stats()

    def plan(self, env):
        if self._impl is None:
            raise ValueError(f"Unsupported legacy baseline method: {self.method}")
        return self._impl.plan(env)

    def act_from_plan(self, env, plan=None) -> Dict[int, int]:
        if self._impl is None:
            raise ValueError(f"Unsupported legacy baseline method: {self.method}")
        return self._impl.act_from_plan(env, plan)

    def supports_planned_execution(self) -> bool:
        return bool(self._impl is not None and self._impl.supports_planned_execution())
