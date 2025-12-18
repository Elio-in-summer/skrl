from __future__ import annotations

import json
import os
import threading
import time
from typing import Optional, cast

import torch
from skrl.envs.wrappers.torch import Wrapper

try:
    import zmq  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    zmq = None


DEFAULT_ZMQ_ENDPOINT = os.environ.get("ALLEGRO_ZMQ_ENDPOINT", "tcp://127.0.0.1:5556")
ISAAC_JOINT_SEQUENCE = [
    "index_joint_0",
    "middle_joint_0",
    "ring_joint_0",
    "thumb_joint_0",
    "index_joint_1",
    "middle_joint_1",
    "ring_joint_1",
    "thumb_joint_1",
    "index_joint_2",
    "middle_joint_2",
    "ring_joint_2",
    "thumb_joint_2",
    "index_joint_3",
    "middle_joint_3",
    "ring_joint_3",
    "thumb_joint_3",
]
URDF_TO_ISAAC = {
    "joint_0.0": "index_joint_0",
    "joint_1.0": "index_joint_1",
    "joint_2.0": "index_joint_2",
    "joint_3.0": "index_joint_3",
    "joint_4.0": "middle_joint_0",
    "joint_5.0": "middle_joint_1",
    "joint_6.0": "middle_joint_2",
    "joint_7.0": "middle_joint_3",
    "joint_8.0": "ring_joint_0",
    "joint_9.0": "ring_joint_1",
    "joint_10.0": "ring_joint_2",
    "joint_11.0": "ring_joint_3",
    "joint_12.0": "thumb_joint_0",
    "joint_13.0": "thumb_joint_1",
    "joint_14.0": "thumb_joint_2",
    "joint_15.0": "thumb_joint_3",
}


class ZMQCommandListener:
    """Subscribe to Allegro ZMQ joint commands."""

    def __init__(self, endpoint: str = DEFAULT_ZMQ_ENDPOINT, side: str = "right") -> None:
        self.endpoint = endpoint
        self.side = side.lower()
        self.ready = zmq is not None and bool(self.endpoint)
        self._latest: tuple[list[str], list[float], float] | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        if self.ready:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()
        else:
            print("[WARN]: ZMQ unavailable or endpoint missing; external takeover disabled")

    def _worker(self) -> None:
        assert zmq is not None
        ctx = zmq.Context.instance()
        socket = ctx.socket(zmq.SUB)
        socket.setsockopt_string(zmq.SUBSCRIBE, "")
        try:
            socket.connect(self.endpoint)
        except Exception as exc:
            print(f"[WARN]: Failed to connect to ZMQ endpoint {self.endpoint}: {exc}")
            self.ready = False
            socket.close(0)
            return

        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        while not self._stop.is_set():
            events = dict(poller.poll(100))
            if socket in events:
                try:
                    raw = socket.recv_string(zmq.NOBLOCK)
                except zmq.Again:
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if payload.get("side", "").lower() != self.side:
                    continue
                names = payload.get("name", [])
                values = payload.get("normalized_position", [])
                if len(names) != len(values):
                    continue
                with self._lock:
                    self._latest = (list(names), [float(v) for v in values], float(payload.get("timestamp", time.time())))
        socket.close(0)

    def get_latest(self) -> tuple[list[str], list[float], float] | None:
        with self._lock:
            if self._latest is None:
                return None
            names, values, ts = self._latest
            return (list(names), list(values), ts)

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)


class ExternalActionProcessor:
    """Map incoming Allegro commands to Isaac Lab action tensors."""

    def __init__(self, env: Wrapper) -> None:
        action_shape = cast(tuple, env.action_space.shape)
        self.action_dim = action_shape[0]
        self.num_envs = env.num_envs
        self.device = env.device
        self.template = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self.index_map = {name: idx for idx, name in enumerate(ISAAC_JOINT_SEQUENCE)}

    def convert(self, payload: tuple[list[str], list[float], float] | None) -> Optional[torch.Tensor]:
        if payload is None:
            return None
        names, values, _ = payload
        result = self.template.clone()
        filled = False
        for name, value in zip(names, values):
            isaac_name = URDF_TO_ISAAC.get(name, name)
            idx = self.index_map.get(isaac_name)
            if idx is None or idx >= self.action_dim:
                continue
            norm_val = max(-1.0, min(1.0, float(value)))
            result[:, idx] = norm_val
            filled = True
        return result if filled else None

