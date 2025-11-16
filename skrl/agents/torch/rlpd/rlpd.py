from __future__ import annotations

from typing import Any, List

import itertools
import gymnasium
from packaging import version

import numpy as np
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.utils import ScopedTimer

from .rlpd_cfg import RLPD_CFG


class RLPD(Agent):
    def __init__(
        self,
        *,
        models: dict[str, Model],
        memory: Memory | None = None,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
        cfg: RLPD_CFG | dict = {},
    ) -> None:
        """RLPD (initial version: identical to SAC behavior).

        This class duplicates SAC's functionality under a new agent name so that
        we can evolve it towards true RLPD in later steps without touching SAC.
        :param models: Agent's models.
        :param memory: Memory to storage agent's data and environment transitions.
        :param observation_space: Observation space.
        :param state_space: State space.
        :param action_space: Action space.
        :param device: Data allocation and computation device. If not specified, the default device will be used.
        :param cfg: Agent's configuration.

        :raises KeyError: If a configuration key is missing.
        """
        self.cfg: RLPD_CFG
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            cfg=RLPD_CFG(**cfg) if isinstance(cfg, dict) else cfg,
        )

        # models
        self.policy = self.models.get("policy", None)

        # collect ensemble critics by numeric suffix ordering: critic_1, critic_2, ...
        def _collect_models(prefix: str) -> List[Model]:
            pairs = []
            for k, v in self.models.items():
                m = re.match(rf"^{prefix}_(\d+)$", k)
                if m:
                    pairs.append((int(m.group(1)), v))
            pairs.sort(key=lambda x: x[0])
            return [v for _, v in pairs]

        self.critics: List[Model] = _collect_models("critic")
        self.target_critics: List[Model] = _collect_models("target_critic")

        # Backward compatibility: if only critic_1/_2 provided, lists already populated.
        # Map first two for legacy attribute access (used by some utilities)
        self.critic_1 = self.critics[0] if len(self.critics) >= 1 else None
        self.critic_2 = self.critics[1] if len(self.critics) >= 2 else None
        self.target_critic_1 = self.target_critics[0] if len(self.target_critics) >= 1 else None
        self.target_critic_2 = self.target_critics[1] if len(self.target_critics) >= 2 else None

        # checkpoint modules
        self.checkpoint_modules["policy"] = self.policy
        for i, c in enumerate(self.critics, start=1):
            self.checkpoint_modules[f"critic_{i}"] = c
        for i, c in enumerate(self.target_critics, start=1):
            self.checkpoint_modules[f"target_critic_{i}"] = c

        # debug / validation logs
        logger.info(
            f"[RLPD] Detected critics: E={len(self.critics)}; targets: T={len(self.target_critics)}; "
            f"cfg: num_qs={self.cfg.num_qs}, num_min_qs={self.cfg.num_min_qs}, utd_ratio={self.cfg.utd_ratio}"
        )
        if len(self.critics) == 0:
            raise ValueError(
                "No critics found in models. Ensure you add keys 'critic_1'..'critic_E'. "
                f"Available model keys: {sorted(list(self.models.keys()))}"
            )
        if len(self.target_critics) == 0:
            raise ValueError(
                "No target critics found in models. Ensure you add keys 'target_critic_1'..'target_critic_E'. "
                f"Available model keys: {sorted(list(self.models.keys()))}"
            )
        if len(self.target_critics) != len(self.critics):
            logger.info(
                f"[RLPD][WARNING] Number of target critics ({len(self.target_critics)}) != critics ({len(self.critics)}). "
                "Training may fail; please check model assembly in the example script."
            )

        # broadcast models' parameters in distributed runs
        if config.torch.is_distributed:
            logger.info(f"Broadcasting models' parameters")
            if self.policy is not None:
                self.policy.broadcast_parameters()
            for c in self.critics:
                c.broadcast_parameters()

        # set up automatic mixed precision
        self._device_type = torch.device(self.device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self.cfg.mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.mixed_precision)

        # entropy
        self._entropy_coefficient = self.cfg.initial_entropy_value
        if self.cfg.learn_entropy:
            self._target_entropy = self.cfg.target_entropy
            if self._target_entropy is None:
                if issubclass(type(self.action_space), gymnasium.spaces.Box):
                    self._target_entropy = -np.prod(self.action_space.shape).astype(np.float32)
                elif issubclass(type(self.action_space), gymnasium.spaces.Discrete):
                    self._target_entropy = -self.action_space.n
                else:
                    self._target_entropy = 0

            self.log_entropy_coefficient = torch.log(
                torch.ones(1, device=self.device) * self._entropy_coefficient
            ).requires_grad_(True)
            self.entropy_optimizer = torch.optim.Adam([self.log_entropy_coefficient], lr=self.cfg.learning_rate[2])

            self.checkpoint_modules["entropy_optimizer"] = self.entropy_optimizer

        # set up optimizers and learning rate schedulers
        if self.policy is not None and len(self.critics) > 0:
            # - optimizers
            self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.cfg.learning_rate[0])
            self.critic_optimizer = torch.optim.Adam(
                itertools.chain(*[c.parameters() for c in self.critics]), lr=self.cfg.learning_rate[1]
            )
            self.checkpoint_modules["policy_optimizer"] = self.policy_optimizer
            self.checkpoint_modules["critic_optimizer"] = self.critic_optimizer
            # - learning rate schedulers
            self.policy_scheduler = self.cfg.learning_rate_scheduler[0]
            self.critic_scheduler = self.cfg.learning_rate_scheduler[1]
            if self.policy_scheduler is not None:
                self.policy_scheduler = self.cfg.learning_rate_scheduler[0](
                    self.policy_optimizer, **self.cfg.learning_rate_scheduler_kwargs[0]
                )
            if self.critic_scheduler is not None:
                self.critic_scheduler = self.cfg.learning_rate_scheduler[1](
                    self.critic_optimizer, **self.cfg.learning_rate_scheduler_kwargs[1]
                )

        # set up target networks
        if len(self.target_critics) == len(self.critics) and len(self.critics) > 0:
            # - freeze target networks with respect to optimizers (update via .update_parameters())
            for tc in self.target_critics:
                tc.freeze_parameters(True)
            # - update target networks (hard update)
            for c, tc in zip(self.critics, self.target_critics):
                tc.update_parameters(c, polyak=1)

        # set up preprocessors
        # - observations
        if self.cfg.observation_preprocessor:
            self._observation_preprocessor = self.cfg.observation_preprocessor(
                **self.cfg.observation_preprocessor_kwargs
            )
            self.checkpoint_modules["observation_preprocessor"] = self._observation_preprocessor
        else:
            self._observation_preprocessor = self._empty_preprocessor
        # - states
        if self.cfg.state_preprocessor:
            self._state_preprocessor = self.cfg.state_preprocessor(**self.cfg.state_preprocessor_kwargs)
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor

    def init(self, *, trainer_cfg: dict[str, Any] | None = None) -> None:
        """Initialize the agent.

        :param trainer_cfg: Trainer configuration.
        """
        super().init(trainer_cfg=trainer_cfg)
        self.enable_models_training_mode(False)

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="observations", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="next_observations", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="states", size=self.state_space, dtype=torch.float32)
            self.memory.create_tensor(name="next_states", size=self.state_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)

            self._tensors_names = [
                "observations",
                "states",
                "actions",
                "rewards",
                "next_observations",
                "next_states",
                "terminated",
                "truncated",
            ]

    def act(
        self, observations: torch.Tensor, states: torch.Tensor | None, *, timestep: int, timesteps: int
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Process the environment's observations/states to make a decision (actions) using the main policy.

        :param observations: Environment observations.
        :param states: Environment states.
        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.

        :return: Agent output. The first component is the expected action/value returned by the agent.
            The second component is a dictionary containing extra output values according to the model.
        """
        inputs = {
            "observations": self._observation_preprocessor(observations),
            "states": self._state_preprocessor(states),
        }
        # sample random actions
        # TODO, check for stochasticity
        if timestep < self.cfg.random_timesteps:
            return self.policy.random_act(inputs, role="policy")

        # sample stochastic actions
        with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            actions, outputs = self.policy.act(inputs, role="policy")

        return actions, outputs

    def record_transition(
        self,
        *,
        observations: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        """Record an environment transition in memory.

        :param observations: Environment observations.
        :param states: Environment states.
        :param actions: Actions taken by the agent.
        :param rewards: Instant rewards achieved by the current actions.
        :param next_observations: Next environment observations.
        :param next_states: Next environment states.
        :param terminated: Signals that indicate episodes have terminated.
        :param truncated: Signals that indicate episodes have been truncated.
        :param infos: Additional information about the environment.
        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        super().record_transition(
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

        if self.memory is not None:
            # reward shaping
            if self.cfg.rewards_shaper is not None:
                rewards = self.cfg.rewards_shaper(rewards, timestep, timesteps)

            # storage transition in memory
            self.memory.add_samples(
                observations=observations,
                states=states,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
            )

    def pre_interaction(self, *, timestep: int, timesteps: int) -> None:
        """Method called before the interaction with the environment.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        pass

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        """Method called after the interaction with the environment.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        if timestep >= self.cfg.learning_starts:
            with ScopedTimer() as timer:
                self.enable_models_training_mode(True)
                self.update(timestep=timestep, timesteps=timesteps)
                self.enable_models_training_mode(False)
                self.track_data("Stats / Algorithm update time (ms)", timer.elapsed_time_ms)

        # write tracking data and checkpoints
        super().post_interaction(timestep=timestep, timesteps=timesteps)

    def update(self, *, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        # RLPD-style UTD: sample a large batch and slice into 'utd_ratio' mini-batches;
        # update critics 'utd_ratio' times, then update actor/entropy once using the last mini-batch.
        utd = max(1, int(self.cfg.utd_ratio))

        # sample a batch from memory
        (
            sampled_observations,
            sampled_states,
            sampled_actions,
            sampled_rewards,
            sampled_next_observations,
            sampled_next_states,
            sampled_terminated,
            sampled_truncated,
        ) = self.memory.sample(names=self._tensors_names, batch_size=self.cfg.batch_size * utd)[0]

        # split tensors into mini-batches
        def split_opt(x):
            if x is None:
                return [None] * utd
            return torch.chunk(x, chunks=utd, dim=0)

        obs_splits = split_opt(sampled_observations)
        states_splits = split_opt(sampled_states)
        acts_splits = split_opt(sampled_actions)
        rews_splits = split_opt(sampled_rewards)
        next_obs_splits = split_opt(sampled_next_observations)
        next_states_splits = split_opt(sampled_next_states)
        terminated_splits = split_opt(sampled_terminated)
        truncated_splits = split_opt(sampled_truncated)

        last_cache = None

        for i in range(utd):
            obs = obs_splits[i]
            states = states_splits[i]
            acts = acts_splits[i]
            rews = rews_splits[i]
            next_obs = next_obs_splits[i]
            next_states = next_states_splits[i]
            terminated = terminated_splits[i]
            truncated = truncated_splits[i]

            with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                inputs = {
                    "observations": self._observation_preprocessor(obs, train=True),
                    "states": self._state_preprocessor(states, train=True),
                }
                next_inputs = {
                    "observations": self._observation_preprocessor(next_obs, train=True),
                    "states": self._state_preprocessor(next_states, train=True),
                }

                with torch.no_grad():
                    # compute target values with REDQ subset min
                    next_actions, out_next = self.policy.act(next_inputs, role="policy")
                    next_log_prob = out_next["log_prob"]

                    E = len(self.target_critics)
                    M = min(self.cfg.num_min_qs, E)
                    if E <= 0 or M <= 0:
                        raise ValueError(
                            f"Empty target ensemble (E={E}, M={M}). Check that models contain 'target_critic_1..{self.cfg.num_qs}'. "
                            f"Models: {sorted(list(self.models.keys()))}"
                        )
                    if M < E:
                        idx = torch.randperm(E, device=next_actions.device)[:M]
                        chosen_targets = [self.target_critics[j] for j in idx.tolist()]
                    else:
                        chosen_targets = self.target_critics

                    target_q_list = []
                    for j, tc in enumerate(chosen_targets):
                        qv, _ = tc.act({**next_inputs, "taken_actions": next_actions}, role=f"target_critic_{j}")
                        target_q_list.append(qv)
                    target_q_stack = torch.stack(target_q_list, dim=0)  # [M, B, 1]
                    target_q_min, _ = torch.min(target_q_stack, dim=0)  # [B, 1]
                    target_q_values = target_q_min - self._entropy_coefficient * next_log_prob
                    target_values = (
                        rews + self.cfg.discount_factor * (terminated | truncated).logical_not() * target_q_values
                    )

                # compute critic loss over ensemble
                q_values = []
                for j, c in enumerate(self.critics):
                    qj, _ = c.act({**inputs, "taken_actions": acts}, role=f"critic_{j}")
                    q_values.append(qj)
                q_stack = torch.stack(q_values, dim=0)  # [E, B, 1]
                critic_loss = F.mse_loss(q_stack, target_values.expand_as(q_stack))

            # optimization step (critic)
            self.critic_optimizer.zero_grad()
            self.scaler.scale(critic_loss).backward()

            if config.torch.is_distributed:
                for c in self.critics:
                    c.reduce_parameters()

            if self.cfg.grad_norm_clip > 0:
                self.scaler.unscale_(self.critic_optimizer)
                nn.utils.clip_grad_norm_(itertools.chain(*[c.parameters() for c in self.critics]), self.cfg.grad_norm_clip)

            self.scaler.step(self.critic_optimizer)

            # update target networks (polyak)
            for c, tc in zip(self.critics, self.target_critics):
                tc.update_parameters(c, polyak=self.cfg.polyak)

            # cache for actor/entropy step
            last_cache = (inputs, q_stack, target_values, obs, states)

        # actor and entropy update (once, on last mini-batch)
        if last_cache is not None:
            inputs, q_stack, target_values, _, _ = last_cache
            with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                actions, outputs = self.policy.act(inputs, role="policy")
                log_prob = outputs["log_prob"]
                # compute Q across ensemble and average
                q_values_pi = []
                for j, c in enumerate(self.critics):
                    qj, _ = c.act({**inputs, "taken_actions": actions}, role=f"critic_{j}")
                    q_values_pi.append(qj)
                q_pi_stack = torch.stack(q_values_pi, dim=0)  # [E, B, 1]
                q_pi_mean = torch.mean(q_pi_stack, dim=0)
                # q_pi_min = torch.min(q_pi_stack, dim=0).values
                policy_loss = (self._entropy_coefficient * log_prob - q_pi_mean).mean()
                # policy_loss = (self._entropy_coefficient * log_prob - q_pi_min).mean()

            # optimization step (policy)
            self.policy_optimizer.zero_grad()
            self.scaler.scale(policy_loss).backward()

            if config.torch.is_distributed:
                self.policy.reduce_parameters()

            if self.cfg.grad_norm_clip > 0:
                self.scaler.unscale_(self.policy_optimizer)
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_norm_clip)

            self.scaler.step(self.policy_optimizer)

            # entropy learning
            if self.cfg.learn_entropy:
                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    entropy_loss = -(self.log_entropy_coefficient * (log_prob + self._target_entropy).detach()).mean()
                self.entropy_optimizer.zero_grad()
                self.scaler.scale(entropy_loss).backward()
                self.scaler.step(self.entropy_optimizer)
                self._entropy_coefficient = torch.exp(self.log_entropy_coefficient.detach())

            self.scaler.update()  # after optimizers have been stepped

            # update learning rate schedulers once per update
            if self.policy_scheduler:
                self.policy_scheduler.step()
            if self.critic_scheduler:
                self.critic_scheduler.step()

            # record data
            if self.write_interval > 0:
                # generic losses
                self.track_data("Loss / Policy loss", policy_loss.item())
                self.track_data("Loss / Critic loss", critic_loss.item())

                # ensemble statistics on last mini-batch
                # reuse q_stack from critic step (on data) and q_pi_stack (on policy actions)
                # Here log stats for q_stack (data batch)
                q_vals = q_stack  # [E, B, 1]
                q_vals_mean = torch.mean(q_vals)
                q_vals_max = torch.max(q_vals)
                q_vals_min = torch.min(q_vals)
                q_ens_std = torch.std(q_vals.squeeze(-1), dim=0).mean()  # mean std across batch

                self.track_data("Q-ensemble / Q (max)", q_vals_max.item())
                self.track_data("Q-ensemble / Q (min)", q_vals_min.item())
                self.track_data("Q-ensemble / Q (mean)", q_vals_mean.item())
                self.track_data("Q-ensemble / Ensemble std", q_ens_std.item())

                self.track_data("Target / Target (max)", torch.max(target_values).item())
                self.track_data("Target / Target (min)", torch.min(target_values).item())
                self.track_data("Target / Target (mean)", torch.mean(target_values).item())

                if self.cfg.learn_entropy:
                    self.track_data("Loss / Entropy loss", entropy_loss.item())
                    self.track_data("Coefficient / Entropy coefficient", self._entropy_coefficient.item())

                if self.policy_scheduler:
                    self.track_data("Learning / Policy learning rate", self.policy_scheduler.get_last_lr()[0])
                if self.critic_scheduler:
                    self.track_data("Learning / Critic learning rate", self.critic_scheduler.get_last_lr()[0])

                # log hparams as scalars (cheap) for traceability
                self.track_data("Ensemble / num_qs", float(len(self.critics)))
                self.track_data("Ensemble / num_min_qs", float(min(self.cfg.num_min_qs, len(self.critics))))
                self.track_data("Learning / UTD ratio", float(utd))
