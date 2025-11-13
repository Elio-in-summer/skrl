from __future__ import annotations

import torch
import torch.nn as nn

from skrl.models.torch import Model
from skrl.models.torch import DeterministicMixin


class MLPWithLayerNorm(nn.Module):
    """MLP block with LayerNorm on hidden layers (no LN on output layer).

    Structure per hidden layer: Linear -> LayerNorm -> Activation.
    The final output layer is a plain Linear without LayerNorm or activation.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int] | tuple[int, ...] = (512, 256),
        out_dim: int = 1,
        activation: type[nn.Module] = nn.ReLU,
        layer_norm_affine: bool = True,
    ) -> None:
        super().__init__()
        self._act = activation()

        dims = [in_dim, *hidden_dims]
        self.fcs = nn.ModuleList()
        self.lns = nn.ModuleList()

        for i in range(len(hidden_dims)):
            self.fcs.append(nn.Linear(dims[i], dims[i + 1]))
            self.lns.append(nn.LayerNorm(dims[i + 1], elementwise_affine=layer_norm_affine))

        # scalar head (no LN, no activation)
        self.out = nn.Linear(dims[-1], out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for fc, ln in zip(self.fcs, self.lns):
            x = fc(x)
            x = ln(x)
            x = self._act(x)
        return self.out(x)


class RLPDStateActionCritic(DeterministicMixin, Model):
    """State-action value network with LayerNorm on hidden layers.

    - Concatenates observations and actions along the last dim.
    - Passes through an MLP with LayerNorm on each hidden layer.
    - Final output is a single Q-value (no LayerNorm on this scalar).
    """

    def __init__(
        self,
        observation_space,
        state_space,
        action_space,
        device,
        *,
        hidden_dims: list[int] | tuple[int, ...] = (512, 256),
        activation: type[nn.Module] = nn.ReLU,
        layer_norm_affine: bool = True,
    ) -> None:
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self)

        in_dim = self.num_observations + self.num_actions
        self.net = MLPWithLayerNorm(
            in_dim=in_dim,
            hidden_dims=hidden_dims,
            out_dim=1,
            activation=activation,
            layer_norm_affine=layer_norm_affine,
        )

    def compute(self, inputs, role):  # noqa: ARG002 (role is required by interface)
        x = torch.cat([inputs["observations"], inputs["taken_actions"]], dim=1)
        return self.net(x), {}

