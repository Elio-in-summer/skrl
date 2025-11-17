from __future__ import annotations

from typing import Dict, List, Sequence

import os

import torch

from skrl import config, logger


def _to_device(data, device):
    if data is None:
        return None
    if isinstance(data, dict):
        return {k: _to_device(v, device) for k, v in data.items()}
    if isinstance(data, torch.Tensor):
        return data.to(device=device)
    raise TypeError(f"Unsupported offline dataset field type: {type(data)}")


def _gather(data, indexes: torch.Tensor):
    if data is None:
        return None
    if isinstance(data, dict):
        return {k: _gather(v, indexes) for k, v in data.items()}
    return data.index_select(0, indexes)


class OfflineDataset:
    """Simple in-memory offline buffer stored as PyTorch tensors."""

    def __init__(
        self,
        data: dict | None = None,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        self.device = config.torch.parse_device(device)
        self._data: dict[str, torch.Tensor] = {}
        self._length: int = 0
        if data:
            self.set_data(data)

    @property
    def length(self) -> int:
        return self._length

    def set_data(self, data: dict) -> None:
        required = [
            "observations",
            "next_observations",
            "actions",
            "rewards",
            "terminated",
            "truncated",
        ]
        for key in required:
            if key not in data:
                raise KeyError(f"Offline dataset missing required field '{key}'")

        self._length = data["rewards"].shape[0]
        for key, value in data.items():
            self._data[key] = _to_device(value, self.device)

    def load(self, path: str, *, map_location: str | torch.device | None = None) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location=map_location or "cpu")
        self.set_data(payload)
        logger.info(f"[OfflineDataset] Loaded {self._length} samples from '{path}'")

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cpu_data = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self._data.items()}
        torch.save(cpu_data, path)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor | None]:
        if self._length == 0:
            raise RuntimeError("Offline dataset is empty")
        batch_size = min(batch_size, self._length)
        indexes = torch.randint(0, self._length, (batch_size,), device=self.device)
        return {k: _gather(v, indexes) for k, v in self._data.items()}

    def append(self, new_data: dict[str, torch.Tensor], *, inplace: bool = True) -> None:
        if not inplace:
            raise NotImplementedError("Non-inplace append is not supported yet")
        if not self._data:
            self.set_data(new_data)
            return
        for key, value in new_data.items():
            if value is None:
                continue
            existing = self._data.get(key)
            if existing is None:
                self._data[key] = _to_device(value, self.device)
            else:
                self._data[key] = torch.cat([existing, _to_device(value, self.device)], dim=0)
        self._length = self._data["rewards"].shape[0]

    def keys(self) -> List[str]:
        return list(self._data.keys())

    def as_dict(self) -> dict[str, torch.Tensor | None]:
        return self._data

