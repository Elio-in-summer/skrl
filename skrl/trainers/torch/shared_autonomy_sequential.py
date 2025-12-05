from __future__ import annotations

import sys
import threading
import time
from typing import Callable, Optional

import torch
import tqdm

from skrl import logger
from skrl.agents.torch import Agent
from skrl.envs.wrappers.torch import MultiAgentEnvWrapper, Wrapper
from skrl.multi_agents.torch import MultiAgent
from skrl.trainers.torch.sequential import SequentialTrainer, SequentialTrainerCfg
from skrl.utils import ScopedTimer


class SharedAutonomySequentialTrainer(SequentialTrainer):
    """Sequential trainer that allows external actions to override the policy during eval."""

    def __init__(
        self,
        *,
        env: Wrapper | MultiAgentEnvWrapper,
        agents: Agent | MultiAgent | list[Agent] | list[MultiAgent],
        scopes: list[int] | None = None,
        cfg: SequentialTrainerCfg | dict = {},
        real_time: bool = False,
        real_time_dt: float | None = None,
    ) -> None:
        super().__init__(env=env, agents=agents, scopes=scopes, cfg=cfg)
        self._external_action: Optional[torch.Tensor] = None
        self._external_action_callback: Optional[Callable[[], Optional[torch.Tensor]]] = None
        self._human_intervene: bool = False
        self._shared_lock = threading.Lock()
        self._real_time = bool(real_time)
        step_dt = real_time_dt
        if step_dt is None:
            step_dt = getattr(env, "step_dt", None)
        if step_dt is None:
            inner = getattr(env, "unwrapped", None)
            step_dt = getattr(inner, "step_dt", None)
        self._real_time_dt = float(step_dt) if step_dt is not None else None
        if self._real_time and self._real_time_dt is None:
            logger.warning("Real-time mode requested but no step_dt detected; disabling real-time")
            self._real_time = False

    # --- Shared-autonomy control -------------------------------------------------
    def update_external_action(self, action: torch.Tensor | list | tuple | float | int) -> None:
        """Update the external action that can override policy outputs."""

        if not isinstance(action, torch.Tensor):
            tensor = torch.as_tensor(action)
        else:
            tensor = action
        tensor = tensor.detach().clone()
        with self._shared_lock:
            self._external_action = tensor

    def set_external_action_callback(self, callback: Optional[Callable[[], Optional[torch.Tensor]]]) -> None:
        """Register a callback that returns an action each time an override is requested."""

        with self._shared_lock:
            self._external_action_callback = callback

    def activate_human_intervene(self) -> None:
        with self._shared_lock:
            self._human_intervene = True

    def end_human_intervene(self) -> None:
        with self._shared_lock:
            self._human_intervene = False

    # --- Eval loop ---------------------------------------------------------------
    def eval(self) -> None:
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.enable_training_mode(False)
        else:
            self.agents.enable_training_mode(False)

        if self.num_simultaneous_agents == 1:
            self._eval_single_agent()
        else:
            self._eval_multi_agent()

    # --- Internal helpers -------------------------------------------------------
    def _eval_single_agent(self) -> None:
        observations, infos = self.env.reset()
        states = self.env.state()

        for timestep in tqdm.tqdm(range(self.cfg.timesteps), disable=self.cfg.disable_progressbar, file=sys.stdout):
            loop_start = time.time()
            self.agents.pre_interaction(timestep=timestep, timesteps=self.cfg.timesteps)

            with torch.no_grad():
                with ScopedTimer() as timer:
                    actions, outputs = self.agents.act(
                        observations, states, timestep=timestep, timesteps=self.cfg.timesteps
                    )
                    self.agents.track_data("Stats / Inference time (ms)", timer.elapsed_time_ms)

                policy_actions = actions if self.cfg.stochastic_evaluation else outputs.get("mean_actions", actions)
                chosen_actions = self._maybe_override_actions(policy_actions)

                with ScopedTimer() as timer:
                    next_observations, rewards, terminated, truncated, infos = self.env.step(chosen_actions)
                    next_states = self.env.state()
                    self.agents.track_data("Stats / Env stepping time (ms)", timer.elapsed_time_ms)

                if not self.cfg.headless:
                    self.env.render()

                self.agents.record_transition(
                    observations=observations,
                    states=states,
                    actions=chosen_actions,
                    rewards=rewards,
                    next_observations=next_observations,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.cfg.timesteps,
                )

                if self.cfg.environment_info in infos:
                    for k, v in infos[self.cfg.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            self.agents.track_data(k if "/" in k else f"Info / {k}", v.item())

            super(self.agents.__class__, self.agents).post_interaction(timestep=timestep, timesteps=self.cfg.timesteps)

            if self.env.num_envs > 1:
                observations = next_observations
                states = next_states
            else:
                if self.env.num_agents > 1:
                    should_reset = not self.env.agents
                else:
                    should_reset = terminated.any() or truncated.any()
                if should_reset:
                    with torch.no_grad():
                        observations, infos = self.env.reset()
                        states = self.env.state()
                else:
                    observations = next_observations
                    states = next_states

            self._enforce_real_time(loop_start)

    def _eval_multi_agent(self) -> None:
        observations, infos = self.env.reset()
        states = self.env.state()

        for timestep in tqdm.tqdm(range(self.cfg.timesteps), disable=self.cfg.disable_progressbar, file=sys.stdout):
            loop_start = time.time()
            for agent in self.agents:
                agent.pre_interaction(timestep=timestep, timesteps=self.cfg.timesteps)

            with torch.no_grad():
                _actions, _outputs = [], []
                for agent, scope in zip(self.agents, self.scopes):
                    with ScopedTimer() as timer:
                        actions, outputs = agent.act(
                            observations[scope[0] : scope[1]],
                            states[scope[0] : scope[1]] if states is not None else None,
                            timestep=timestep,
                            timesteps=self.cfg.timesteps,
                        )
                        agent.track_data("Stats / Inference time (ms)", timer.elapsed_time_ms)
                    policy_actions = actions if self.cfg.stochastic_evaluation else outputs.get("mean_actions", actions)
                    _actions.append(policy_actions)
                    _outputs.append(outputs)

                actions = torch.vstack(_actions)
                actions = self._maybe_override_actions(actions)

                with ScopedTimer() as timer:
                    next_observations, rewards, terminated, truncated, infos = self.env.step(actions)
                    next_states = self.env.state()
                    elapsed = timer.elapsed_time_ms
                    for agent in self.agents:
                        agent.track_data("Stats / Env stepping time (ms)", elapsed)

                if not self.cfg.headless:
                    self.env.render()

                for agent, scope in zip(self.agents, self.scopes):
                    agent.record_transition(
                        observations=observations[scope[0] : scope[1]],
                        states=states[scope[0] : scope[1]] if states is not None else None,
                        actions=actions[scope[0] : scope[1]],
                        rewards=rewards[scope[0] : scope[1]],
                        next_observations=next_observations[scope[0] : scope[1]],
                        next_states=next_states[scope[0] : scope[1]] if next_states is not None else None,
                        terminated=terminated[scope[0] : scope[1]],
                        truncated=truncated[scope[0] : scope[1]],
                        infos=infos,
                        timestep=timestep,
                        timesteps=self.cfg.timesteps,
                    )

                if self.cfg.environment_info in infos:
                    for k, v in infos[self.cfg.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            for agent in self.agents:
                                agent.track_data(k if "/" in k else f"Info / {k}", v.item())

            for agent in self.agents:
                super(agent.__class__, agent).post_interaction(timestep=timestep, timesteps=self.cfg.timesteps)

            if self.env.num_envs > 1:
                observations = next_observations
                states = next_states
            else:
                raise RuntimeError("Sequential trainer is not supported for single environment")

            self._enforce_real_time(loop_start)

    def _maybe_override_actions(self, policy_actions: torch.Tensor) -> torch.Tensor:
        try:
            override = self._resolve_override_action(policy_actions)
        except ValueError as exc:
            logger.warning(f"Shared autonomy override ignored: {exc}")
            override = None
        if override is None:
            return policy_actions
        return override

    def _enforce_real_time(self, loop_start: float) -> None:
        if not self._real_time or self._real_time_dt is None:
            return
        elapsed = time.time() - loop_start
        delay = self._real_time_dt - elapsed
        if delay > 0:
            time.sleep(delay)

    def _resolve_override_action(self, reference: torch.Tensor) -> Optional[torch.Tensor]:
        with self._shared_lock:
            intervene_active = self._human_intervene
            callback = self._external_action_callback
            stored_action = self._external_action

        if not intervene_active:
            return None

        candidate = None
        if callback is not None:
            candidate = callback()
        if candidate is None:
            candidate = stored_action
        if candidate is None:
            return None

        if not isinstance(candidate, torch.Tensor):
            candidate = torch.as_tensor(candidate)
        candidate = candidate.to(device=reference.device, dtype=reference.dtype)

        return self._align_action_shape(candidate, reference)

    def _align_action_shape(self, action: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if action.shape == reference.shape:
            return action

        if action.ndim == reference.ndim and action.shape[0] == 1:
            return action.expand_as(reference)

        if action.ndim == reference.ndim - 1 and action.shape == reference.shape[1:]:
            return action.unsqueeze(0).expand(reference.shape[0], *action.shape)

        if action.ndim == reference.ndim and reference.shape[0] == 1 and action.shape[1:] == reference.shape[1:]:
            return action

        raise ValueError(
            f"External action shape {tuple(action.shape)} incompatible with policy action shape {tuple(reference.shape)}"
        )
