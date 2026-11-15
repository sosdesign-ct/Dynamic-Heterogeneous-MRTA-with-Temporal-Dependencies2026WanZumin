from __future__ import annotations

import multiprocessing as mp
import traceback
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from env.mrta_config import MRTAConfig
from env.mrta_env import MRTAEnv


def _clone_cfg(base_cfg: MRTAConfig, cfg_overrides: Optional[Dict]) -> MRTAConfig:
	data = asdict(base_cfg)
	if cfg_overrides:
		for k, v in cfg_overrides.items():
			if k in data:
				data[k] = v
	return MRTAConfig(**data)


def _worker_loop(conn, base_cfg_data: Dict[str, Any], seed: int):
	base_cfg = MRTAConfig(**base_cfg_data)
	env = MRTAEnv(_clone_cfg(base_cfg, None), seed=int(seed))
	try:
		while True:
			msg = conn.recv()
			cmd = msg.get("cmd")
			data = msg.get("data", {})
			if cmd == "close":
				conn.send({"ok": True, "result": None})
				break
			if cmd == "reset":
				seed_i = int(data.get("seed", seed))
				cfg_overrides = data.get("cfg_overrides")
				cfg_i = _clone_cfg(base_cfg, cfg_overrides)
				env = MRTAEnv(cfg_i, seed=seed_i)
				result = env.reset(seed=seed_i)
				conn.send({"ok": True, "result": result})
			elif cmd == "step":
				result = env.step(data["actions"])
				conn.send({"ok": True, "result": result})
			elif cmd == "get_state":
				conn.send({"ok": True, "result": env.get_state()})
			elif cmd == "set_state":
				env.set_state(data["state"])
				conn.send({"ok": True, "result": None})
			elif cmd == "get_obs":
				conn.send({"ok": True, "result": env._get_obs()})
			else:
				raise RuntimeError(f"unknown worker cmd: {cmd}")
	except EOFError:
		pass
	except KeyboardInterrupt:
		pass
	except Exception as exc:
		conn.send(
			{
				"ok": False,
				"error": repr(exc),
				"traceback": traceback.format_exc(),
			}
		)
	finally:
		conn.close()


class SyncMRTAEnvPool:
	"""Lightweight vectorized env pool with the interface expected by train_mrta.py."""

	def __init__(self, cfg: MRTAConfig, num_envs: int, seed: int):
		self.base_cfg = _clone_cfg(cfg, None)
		self.num_envs = int(max(1, num_envs))
		self.base_seed = int(seed)
		self.envs: List[MRTAEnv] = []
		for i in range(self.num_envs):
			self.envs.append(MRTAEnv(_clone_cfg(self.base_cfg, None), seed=self.base_seed + i))

	def close(self):
		# Sync pool owns no subprocess resources.
		return None

	def reset_all(self, seeds: Optional[Sequence[int]] = None, cfg_overrides: Optional[Dict] = None):
		obs_list = []
		for i in range(self.num_envs):
			seed_i = int(seeds[i]) if seeds is not None else (self.base_seed + i)
			cfg_i = _clone_cfg(self.base_cfg, cfg_overrides)
			self.envs[i] = MRTAEnv(cfg_i, seed=seed_i)
			obs_list.append(self.envs[i].reset(seed=seed_i))
		return obs_list

	def reset_one(self, env_i: int, cfg_overrides: Optional[Dict] = None, seed: Optional[int] = None):
		idx = int(env_i)
		if idx < 0 or idx >= self.num_envs:
			raise IndexError(f"env index out of range: {env_i}")
		seed_i = int(seed) if seed is not None else self.base_seed + idx
		cfg_i = _clone_cfg(self.base_cfg, cfg_overrides)
		self.envs[idx] = MRTAEnv(cfg_i, seed=seed_i)
		return self.envs[idx].reset(seed=seed_i)

	def step(self, action_dicts: Sequence[Dict[int, int]]):
		if len(action_dicts) != self.num_envs:
			raise ValueError(f"action_dicts length mismatch: got={len(action_dicts)} expected={self.num_envs}")
		outs = []
		for env, acts in zip(self.envs, action_dicts):
			outs.append(env.step(acts))
		return outs

	def get_states(self):
		return [env.get_state() for env in self.envs]

	def set_states(self, states: Sequence[Dict]):
		if len(states) != self.num_envs:
			raise ValueError(f"states length mismatch: got={len(states)} expected={self.num_envs}")
		for env, s in zip(self.envs, states):
			env.set_state(s)

	def get_obs_all(self):
		return [env._get_obs() for env in self.envs]


class SubprocMRTAEnvPool:
	"""Multi-process env pool with the same interface as SyncMRTAEnvPool."""

	def __init__(self, cfg: MRTAConfig, num_envs: int, seed: int, start_method: str = "spawn"):
		self.base_cfg = _clone_cfg(cfg, None)
		self.base_cfg_data = asdict(self.base_cfg)
		self.num_envs = int(max(1, num_envs))
		self.base_seed = int(seed)
		ctx = mp.get_context(str(start_method))
		self._ctx = ctx
		self._closed = False
		self._conns = []
		self._procs = []
		for i in range(self.num_envs):
			parent_conn, child_conn = ctx.Pipe()
			proc = ctx.Process(
				target=_worker_loop,
				args=(child_conn, self.base_cfg_data, self.base_seed + i),
				daemon=True,
			)
			proc.start()
			child_conn.close()
			self._conns.append(parent_conn)
			self._procs.append(proc)

	def _request_many(self, payloads: Sequence[Dict[str, Any]]):
		if self._closed:
			raise RuntimeError("env pool already closed")
		if len(payloads) != self.num_envs:
			raise ValueError(f"payload length mismatch: got={len(payloads)} expected={self.num_envs}")
		for conn, payload in zip(self._conns, payloads):
			conn.send(payload)
		results = []
		for idx, conn in enumerate(self._conns):
			msg = conn.recv()
			if not msg.get("ok", False):
				err = msg.get("error", "unknown worker error")
				tb = msg.get("traceback", "")
				raise RuntimeError(f"subproc env worker {idx} failed: {err}\n{tb}")
			results.append(msg.get("result"))
		return results

	def close(self):
		if self._closed:
			return None
		for conn in self._conns:
			try:
				conn.send({"cmd": "close", "data": {}})
			except Exception:
				pass
		for conn in self._conns:
			try:
				conn.recv()
			except Exception:
				pass
			try:
				conn.close()
			except Exception:
				pass
		for proc in self._procs:
			proc.join(timeout=2.0)
			if proc.is_alive():
				proc.terminate()
				proc.join(timeout=1.0)
		self._closed = True
		return None

	def reset_all(self, seeds: Optional[Sequence[int]] = None, cfg_overrides: Optional[Dict] = None):
		payloads = []
		for i in range(self.num_envs):
			seed_i = int(seeds[i]) if seeds is not None else (self.base_seed + i)
			payloads.append({"cmd": "reset", "data": {"seed": seed_i, "cfg_overrides": cfg_overrides}})
		return self._request_many(payloads)

	def reset_one(self, env_i: int, cfg_overrides: Optional[Dict] = None, seed: Optional[int] = None):
		idx = int(env_i)
		if idx < 0 or idx >= self.num_envs:
			raise IndexError(f"env index out of range: {env_i}")
		seed_i = int(seed) if seed is not None else self.base_seed + idx
		conn = self._conns[idx]
		conn.send({"cmd": "reset", "data": {"seed": seed_i, "cfg_overrides": cfg_overrides}})
		msg = conn.recv()
		if not msg.get("ok", False):
			err = msg.get("error", "unknown worker error")
			tb = msg.get("traceback", "")
			raise RuntimeError(f"subproc env worker {idx} failed: {err}\n{tb}")
		return msg.get("result")

	def step(self, action_dicts: Sequence[Dict[int, int]]):
		payloads = []
		for acts in action_dicts:
			payloads.append({"cmd": "step", "data": {"actions": acts}})
		return self._request_many(payloads)

	def get_states(self):
		payloads = [{"cmd": "get_state", "data": {}} for _ in range(self.num_envs)]
		return self._request_many(payloads)

	def set_states(self, states: Sequence[Dict]):
		payloads = []
		for s in states:
			payloads.append({"cmd": "set_state", "data": {"state": s}})
		self._request_many(payloads)

	def get_obs_all(self):
		payloads = [{"cmd": "get_obs", "data": {}} for _ in range(self.num_envs)]
		return self._request_many(payloads)


def make_mrta_env_pool(
	cfg: MRTAConfig,
	num_envs: int,
	seed: int,
	backend: str = "auto",
	start_method: str = "spawn",
	) -> Tuple[object, str]:
	"""Factory used by training code.

	Current implementation provides a robust sync backend. If subproc/auto is
	requested, it gracefully falls back to sync to keep training runnable.
	"""

	backend_req = str(backend).lower().strip()
	if backend_req not in {"auto", "sync", "subproc"}:
		raise ValueError(f"unsupported backend: {backend}")

	backend_used = "sync"
	pool: object
	if backend_req == "sync" or int(num_envs) <= 1:
		pool = SyncMRTAEnvPool(cfg=cfg, num_envs=int(num_envs), seed=int(seed))
		backend_used = "sync"
		return pool, backend_used

	try:
		pool = SubprocMRTAEnvPool(
			cfg=cfg,
			num_envs=int(num_envs),
			seed=int(seed),
			start_method=str(start_method),
		)
		backend_used = "subproc"
		return pool, backend_used
	except Exception:
		if backend_req == "subproc":
			raise
		pool = SyncMRTAEnvPool(cfg=cfg, num_envs=int(num_envs), seed=int(seed))
		backend_used = "sync"
		return pool, backend_used


__all__ = ["SyncMRTAEnvPool", "SubprocMRTAEnvPool", "make_mrta_env_pool"]
