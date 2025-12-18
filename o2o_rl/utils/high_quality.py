from __future__ import annotations

import os
from types import MethodType
from typing import Any, Callable, Optional

import torch

from skrl import logger
from skrl.datasets import OfflineDataset

#! The entry for any hq picking is not achieved yet, now this is specil for Allegro Cube Task
class HighQualityTrajectoryTargetReached(RuntimeError):
    """Raised to stop rollout once enough high-quality trajectories are collected."""


class HighQualityTrajectoryCollector:
    def __init__(
        self,
        *,
        base_env: Any,
        num_envs: int,
        threshold: int,
        target_episodes: Optional[int],
        command_name: str,
        metric_name: str,
        store_episodes: bool = True,
        episode_callback: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self._base_env = base_env
        self._num_envs = num_envs
        self._threshold = int(threshold)
        self._target = target_episodes
        self._metric_name = metric_name
        self._command_term = base_env.command_manager.get_term(command_name)
        metric_tensor = self._command_term.metrics[metric_name]
        self._success_flags = torch.zeros_like(metric_tensor, dtype=torch.bool)
        self._trajectories = [[] for _ in range(num_envs)]
        self._episodes = [] if store_episodes else None
        self._collected = 0
        self._episode_callback = episode_callback

    @property
    def collected(self) -> int:
        return self._collected

    @property
    def target(self) -> Optional[int]:
        return self._target

    @property
    def threshold(self) -> int:
        return int(self._threshold)

    def set_threshold(self, value: int) -> int:
        new_value = max(0, int(value))
        if new_value == self._threshold:
            return self._threshold
        self._threshold = new_value
        self._success_flags.zero_()
        return self._threshold

    def increment_threshold(self, delta: int = 1) -> int:
        return self.set_threshold(self._threshold + int(delta))

    def process_transition(
        self,
        *,
        observations: torch.Tensor,
        states: Optional[torch.Tensor],
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        next_states: Optional[torch.Tensor],
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> bool:
        with torch.no_grad():
            metrics = self._command_term.metrics[self._metric_name]
            self._success_flags |= metrics >= self._threshold

        reached_target = False
        for env_id in range(self._num_envs):
            step = {
                "observations": observations[env_id].detach().clone(),
                "next_observations": next_observations[env_id].detach().clone(),
                "actions": actions[env_id].detach().clone(),
                "rewards": rewards[env_id].detach().clone(),
                "terminated": terminated[env_id].detach().clone(),
                "truncated": truncated[env_id].detach().clone(),
            }
            if states is not None:
                step["states"] = states[env_id].detach().clone()
            if next_states is not None:
                step["next_states"] = next_states[env_id].detach().clone()
            self._trajectories[env_id].append(step)

            env_terminated = bool(terminated[env_id].bool().any().item())
            env_truncated = bool(truncated[env_id].bool().any().item())
            if env_terminated or env_truncated:
                reached_target |= self._finalize_env_episode(env_id)
        return reached_target

    def build_dataset(self) -> dict:
        if not self._episodes:
            return {}
        aggregated = {}
        for episode in self._episodes:
            for key, value in episode.items():
                aggregated.setdefault(key, []).append(value)
        return {key: torch.cat(value_list, dim=0) for key, value_list in aggregated.items()}

    def _finalize_env_episode(self, env_id: int) -> bool:
        reached_target = False
        episode_steps = self._trajectories[env_id]
        if self._success_flags[env_id].item() and episode_steps:
            packed = self._pack_episode(episode_steps)
            if self._episode_callback is not None:
                self._episode_callback(packed)
            if self._episodes is not None:
                self._episodes.append(packed)
            self._collected += 1
            if self._target is not None:
                reached_target = self._collected >= self._target
        self._trajectories[env_id] = []
        self._success_flags[env_id] = False
        return reached_target

    @staticmethod
    def _pack_episode(steps):
        episode = {}
        for step in steps:
            for key, value in step.items():
                episode.setdefault(key, []).append(value)
        return {key: torch.stack(values, dim=0) for key, values in episode.items()}


class BoundedOfflineDataset(OfflineDataset):
    def __init__(self, *args, max_transitions: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_transitions = max_transitions

    def set_data(self, data: dict) -> None:
        super().set_data(data)
        self._enforce_limit()

    def append(self, new_data: dict[str, torch.Tensor], *, inplace: bool = True) -> None:
        super().append(new_data, inplace=inplace)
        self._enforce_limit()

    def _enforce_limit(self) -> None:
        excess = self.length - self.max_transitions
        if excess <= 0 or not self._data:
            return
        for key, value in self._data.items():
            if value is not None:
                self._data[key] = value[excess:]
        self._length = self._data["rewards"].shape[0]


class HighQualityTrajectoryManager:
    """Helper that wires HQ collection, dataset streaming, and agent patching."""

    def __init__(
        self,
        *,
        agent,
        base_env: Any,
        num_envs: int,
        threshold: int,
        target_episodes: Optional[int],
        command_name: str,
        metric_name: str,
        rollout_mode: bool,
        offline_dataset: Optional[BoundedOfflineDataset],
        device,
        max_transitions: int,
        threshold_increment_every: int,
        offline_ratio: float,
    ) -> None:
        self.agent = agent
        self._base_env = base_env
        self._num_envs = num_envs
        self._threshold = int(threshold)
        self._target_episodes = target_episodes
        self._command_name = command_name
        self._metric_name = metric_name
        self._rollout_mode = rollout_mode
        self.offline_dataset = offline_dataset
        self._device = device
        self._max_transitions = max_transitions
        self._threshold_increment_every = max(0, int(threshold_increment_every))
        self._offline_ratio = offline_ratio
        self.collector: Optional[HighQualityTrajectoryCollector] = None
        self.mode = "rollout" if rollout_mode else "train"
        self.stats = {"episodes": 0, "transitions": 0}
        self._original_record_transition = None

        self._setup_collector()
        self._patch_agent()

    def _setup_collector(self) -> None:
        if self._rollout_mode:
            self.collector = HighQualityTrajectoryCollector(
                base_env=self._base_env,
                num_envs=self._num_envs,
                threshold=self._threshold,
                target_episodes=self._target_episodes,
                command_name=self._command_name,
                metric_name=self._metric_name,
                store_episodes=True,
            )
            logger.info(
                "High-quality rollout mode enabled (threshold=%d, target=%s trajectories)",
                self._threshold,
                self._target_episodes,
            )
            return

        if self.offline_dataset is None:
            self.offline_dataset = BoundedOfflineDataset(
                device=self._device,
                max_transitions=self._max_transitions,
            )
        collector_ref: dict[str, HighQualityTrajectoryCollector] = {}

        def _maybe_raise_threshold() -> None:
            collector = collector_ref.get("collector")
            if self._threshold_increment_every <= 0 or collector is None:
                return
            if self.stats["episodes"] == 0 or (self.stats["episodes"] % self._threshold_increment_every) != 0:
                return
            new_threshold = collector.increment_threshold()
            logger.info(
                "Raised HQ trajectory threshold to %d after collecting %d high-quality episodes",
                new_threshold,
                self.stats["episodes"],
            )
            if hasattr(self.agent, "track_data"):
                self.agent.track_data("Data / HQ threshold", float(new_threshold))

        def _consume_episode(episode: dict) -> None:
            if self.offline_dataset is None:
                return
            flat_episode = {key: tensor for key, tensor in episode.items() if tensor is not None}
            self.offline_dataset.append(flat_episode)
            self.stats["episodes"] += 1
            step_count = flat_episode["rewards"].shape[0]
            self.stats["transitions"] += step_count
            if hasattr(self.agent, "track_data"):
                self.agent.track_data("Data / HQ offline trajectories", float(self.stats["episodes"]))
                self.agent.track_data("Data / HQ offline transitions", float(self.stats["transitions"]))
            _maybe_raise_threshold()

        self.collector = HighQualityTrajectoryCollector(
            base_env=self._base_env,
            num_envs=self._num_envs,
            threshold=self._threshold,
            target_episodes=None,
            command_name=self._command_name,
            metric_name=self._metric_name,
            store_episodes=False,
            episode_callback=_consume_episode,
        )
        collector_ref["collector"] = self.collector

        logger.info(
            "High-quality training mode enabled (threshold=%d, offline budget=%d transitions)",
            self._threshold,
            self.offline_dataset.max_transitions,
        )
        if self._threshold_increment_every > 0:
            logger.info(
                "HQ trajectory threshold will increase by 1 every %d collected high-quality episodes",
                self._threshold_increment_every,
            )
        if self._offline_ratio <= 0:
            logger.warning("Offline ratio is 0; high-quality trajectories will not be sampled during updates")

        if self.offline_dataset is not None:
            self.agent.offline_dataset = self.offline_dataset

    def _patch_agent(self) -> None:
        if self.collector is None:
            return

        original_method = self.agent.record_transition
        self._original_record_transition = original_method
        collector = self.collector
        manager = self

        def record_transition_with_hq(
            self_agent,
            *,
            observations,
            states,
            actions,
            rewards,
            next_observations,
            next_states,
            terminated,
            truncated,
            infos,
            timestep,
            timesteps,
        ):
            reached_target = collector.process_transition(
                observations=observations,
                states=states,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
            )
            original_method(
                observations=observations,
                states=states,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
                infos=infos,
                timestep=timestep,
                timesteps=timesteps,
            )
            if reached_target and manager._rollout_mode:
                raise HighQualityTrajectoryTargetReached

        self.agent.record_transition = MethodType(record_transition_with_hq, self.agent)

    @property
    def collected(self) -> int:
        return self.collector.collected if self.collector else 0

    @property
    def target(self) -> Optional[int]:
        return self.collector.target if self.collector else None

    def build_dataset(self) -> dict:
        if not self.collector:
            return {}
        return self.collector.build_dataset()

    def save_rollout_dataset(self, output_path: str) -> bool:
        if not self._rollout_mode or not self.collector:
            return False
        dataset = self.collector.build_dataset()
        if not dataset:
            logger.warning("No high-quality trajectories collected; rollout dataset not saved")
            return False
        cpu_dataset = {key: value.clone().cpu() for key, value in dataset.items()}
        sample_count = next(iter(cpu_dataset.values())).shape[0]
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        torch.save(cpu_dataset, output_path)
        logger.info(
            "Saved %d high-quality transitions (%d trajectories) to '%s'",
            sample_count,
            self.collector.collected,
            output_path,
        )
        return True

