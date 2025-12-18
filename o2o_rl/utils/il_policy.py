from __future__ import annotations

import torch
import torch.nn as nn

from skrl.models.torch.rlpd_actor import RLPDTanhGaussianActor


def load_il_policy(
    observation_space,
    state_space,
    action_space,
    device,
    checkpoint_path: str,
    *,
    hidden_dims=(512, 256, 128),
    activation: type[nn.Module] = nn.ELU,
    log_std_bounds=(-20.0, 2.0),
) -> RLPDTanhGaussianActor:
    """Instantiate an RLPD actor and load weights from a checkpoint."""
    policy = RLPDTanhGaussianActor(
        observation_space,
        state_space,
        action_space,
        device,
        hidden_dims=hidden_dims,
        activation=activation,
        log_std_bounds=log_std_bounds,
    )
    policy.eval()
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model_state_dict" in state:
            state = state["model_state_dict"]
    policy.load_state_dict(state, strict=False)
    for param in policy.parameters():
        param.requires_grad_(False)
    return policy

