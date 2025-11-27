from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Type

import torch
import torch.nn as nn
from torch.distributions import Normal

from skrl.models.torch import Model
from skrl.utils.spaces.torch import compute_space_limits
from skrl import logger


@dataclass
class _MLPConfig:
    hidden_dims: Sequence[int]
    activation: Type[nn.Module]
    activate_final: bool
    use_layer_norm: bool
    layer_norm_affine: bool
    dropout_rate: Optional[float]
    use_pnorm: bool


class _RLPDActorBackbone(nn.Module):
    """MLP block mirroring the JAX RLPD actor defaults."""

    def __init__(self, in_dim: int, cfg: _MLPConfig) -> None:
        super().__init__()
        self._cfg = cfg
        self._activation = cfg.activation()
        self._layers = nn.ModuleList()
        self._layer_norms = (
            nn.ModuleList(
                [
                    nn.LayerNorm(size, elementwise_affine=cfg.layer_norm_affine)
                    for size in cfg.hidden_dims
                ]
            )
            if cfg.use_layer_norm
            else None
        )
        prev_dim = in_dim
        for size in cfg.hidden_dims:
            self._layers.append(nn.Linear(prev_dim, size))
            prev_dim = size
        self._dropout = (
            nn.Dropout(p=cfg.dropout_rate) if cfg.dropout_rate and cfg.dropout_rate > 0 else None
        )
        self.output_dim = prev_dim if cfg.hidden_dims else in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self._layers):
            x = layer(x)
            apply_activation = (i + 1) < len(self._layers) or self._cfg.activate_final
            if apply_activation:
                if self._dropout is not None:
                    x = self._dropout(x)
                if self._layer_norms is not None:
                    x = self._layer_norms[i](x)
                x = self._activation(x)
        if self._cfg.use_pnorm:
            eps = 1e-6
            norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(eps)
            x = x / norm
        return x


class RLPDTanhGaussianActor(Model):
    """Torch actor that mirrors the JAX RLPD Tanh-normal policy."""

    def __init__(
        self,
        observation_space,
        state_space,
        action_space,
        device,
        *,
        hidden_dims: Sequence[int] = (256, 256),
        activation: Type[nn.Module] = nn.ReLU,
        activate_final: bool = True,
        use_layer_norm: bool = False,
        layer_norm_affine: bool = True,
        dropout_rate: Optional[float] = None,
        use_pnorm: bool = False,
        state_dependent_std: bool = True,
        log_std_bounds: Tuple[float, float] = (-5.0, 2.0),
        init_log_std: float = -0.5,
        rescale_actions: bool = False,
        tanh_epsilon: float = 1e-6,
    ) -> None:
        if use_layer_norm:
            logger.info("Using layer normalization")
        if use_pnorm:
            logger.info("Using p-norm")
        if rescale_actions:
            logger.info("Using action rescaling")
        if state_dependent_std:
            logger.info("Using state-dependent std")
        if dropout_rate:
            logger.info("Using dropout rate: {dropout_rate}")
        if not hidden_dims:
            hidden_dims = tuple()
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        if self.num_actions <= 0:
            raise ValueError("Action space must be continuous with non-zero size for RLPDTanhGaussianActor.")

        cfg = _MLPConfig(
            hidden_dims=hidden_dims,
            activation=activation,
            activate_final=activate_final,
            use_layer_norm=use_layer_norm,
            layer_norm_affine=layer_norm_affine,
            dropout_rate=dropout_rate,
            use_pnorm=use_pnorm,
        )
        self.backbone = _RLPDActorBackbone(self.num_observations, cfg)
        feature_dim = self.backbone.output_dim
        self.mean_head = nn.Linear(feature_dim, self.num_actions)
        self.state_dependent_std = state_dependent_std
        if self.state_dependent_std:
            self.log_std_head = nn.Linear(feature_dim, self.num_actions)
        else:
            init_value = torch.full((self.num_actions,), init_log_std, dtype=torch.float32)
            self.log_std_parameter = nn.Parameter(init_value)

        self.log_std_bounds = tuple(log_std_bounds)
        if self.log_std_bounds[0] > self.log_std_bounds[1]:
            raise ValueError("log_std_bounds must be in the form (min, max).")
        self.register_buffer("_tanh_epsilon", torch.tensor(tanh_epsilon, dtype=torch.float32), persistent=False)
        self._setup_action_rescaling(rescale_actions)

    def _setup_action_rescaling(self, rescale_actions: bool) -> None:
        self._rescale_actions = False
        self.register_buffer("_action_mid", None, persistent=False)
        self.register_buffer("_action_half_range", None, persistent=False)
        if not rescale_actions or self.action_space is None:
            return
        low, high = compute_space_limits(self.action_space, device=self.device)
        if low is None or high is None:
            return
        if torch.isinf(low).any() or torch.isinf(high).any():
            return
        self._rescale_actions = True
        self._action_mid = (high + low) / 2
        self._action_half_range = (high - low) / 2
        logger.info(f"Action rescaling enabled with mid: {self._action_mid} and half range: {self._action_half_range}")

    def compute(self, inputs, role: str = "") -> tuple[torch.Tensor, dict]:
        observations = inputs["observations"]
        features = self.backbone(observations)
        mean = self.mean_head(features)
        if self.state_dependent_std:
            log_std = self.log_std_head(features)
        else:
            log_std = self.log_std_parameter.unsqueeze(0).expand_as(mean)
        log_std = torch.clamp(log_std, self.log_std_bounds[0], self.log_std_bounds[1])
        return mean, {"log_std": log_std}

    def act(self, inputs, *, role: str = "") -> tuple[torch.Tensor, dict]:
        """Sample or score actions, depending on whether ``taken_actions`` is provided.

        Args:
            inputs: Dict that must contain ``"observations"`` (batch, obs_dim). Two modes:

                1. **Sampling** – call with only observations:

                    >>> actions, info = policy.act({"observations": obs})

                   This draws actions from the policy.

                2. **Evaluation** – include ``"taken_actions"`` (batch, act_dim):

                    >>> actions, info = policy.act({"observations": obs,
                    ...                              "taken_actions": replay_actions})

                   In this mode no sampling happens; the provided actions are used to
                   compute their log-probabilities and returned unchanged.

            role: Optional label propagated by the agent framework.

        Returns:
            actions: Either freshly sampled (mode 1) or the provided ``taken_actions`` (mode 2).
            outputs: Dict with ``log_prob``, ``log_std``, ``mean_actions`` and ``pre_tanh_actions``.
        """
        mean, extras = self.compute(inputs, role=role)
        log_std = extras["log_std"]
        std = torch.exp(log_std)
        base_dist = Normal(mean, std)

        # If caller passes taken_actions, we only evaluate that batch; otherwise we sample.
        taken_actions = inputs.get("taken_actions")

        if taken_actions is None:
            pre_tanh = base_dist.rsample()
            squashed = torch.tanh(pre_tanh)
            log_prob = self._log_prob(base_dist, pre_tanh, squashed)
            actions = self._scale_action(squashed)
        else:
            normalized = self._normalize_action(taken_actions)
            normalized = normalized.clamp(-1 + self._tanh_epsilon.item(), 1 - self._tanh_epsilon.item())
            pre_tanh = self._atanh(normalized)
            log_prob = self._log_prob(base_dist, pre_tanh, normalized)
            actions = taken_actions

        mean_actions = self._scale_action(torch.tanh(mean))

        outputs = {
            "log_prob": log_prob,
            "log_std": log_std,
            "mean_actions": mean_actions,
            "pre_tanh_actions": pre_tanh,
        }
        return actions, outputs

    def _scale_action(self, normalized: torch.Tensor) -> torch.Tensor:
        if not self._rescale_actions or self._action_mid is None or self._action_half_range is None:
            return normalized
        return self._action_mid + normalized * self._action_half_range

    def _normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        if not self._rescale_actions or self._action_mid is None or self._action_half_range is None:
            return action
        denom = torch.where(
            self._action_half_range.abs() > self._tanh_epsilon,
            self._action_half_range,
            torch.ones_like(self._action_half_range),
        )
        return (action - self._action_mid) / denom

    def _atanh(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def _log_prob(self, dist: Normal, pre_tanh: torch.Tensor, squashed: torch.Tensor) -> torch.Tensor:
        log_prob = dist.log_prob(pre_tanh)
        correction = torch.log(1 - squashed.pow(2) + self._tanh_epsilon)
        log_prob = log_prob - correction
        return log_prob.sum(dim=-1, keepdim=True)
