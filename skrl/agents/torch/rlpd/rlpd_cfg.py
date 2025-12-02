from __future__ import annotations

from typing import Callable

import dataclasses

from skrl.agents.torch import AgentCfg


@dataclasses.dataclass(kw_only=True)
class RLPD_CFG(AgentCfg):
    """Configuration for the RLPD agent.

    Note: This initial version mirrors SAC_CFG one-to-one so it can be swapped
    in without behavioral changes. We keep the same fields for compatibility.
    """

    gradient_steps: int = 1
    """Number of gradient steps to perform for each update."""

    batch_size: int = 64
    """Batch size for sampling transitions from memory during training."""

    discount_factor: float = 0.99
    """Parameter that balances the importance of future rewards (close to 1.0) versus immediate rewards (close to 0.0).

    Range: ``[0.0, 1.0]``.
    """

    polyak: float = 0.005
    """Parameter to control the update of the target networks by polyak averaging.

    Range: ``[0.0, 1.0]``. See :py:meth:`~skrl.models.torch.base.Model.update_parameters` for more details.
    """

    learning_rate: float | tuple[float, float, float] = 1e-3
    """Learning rate for the actor and critic networks, and entropy coefficient.

    * If a float is provided, the same learning rate will be used for the networks/coefficient.
    * If a tuple is provided, its elements will be used for each network/coefficient in order.
    """

    learning_rate_scheduler: type | tuple[type | None, type | None, type | None] | None = None
    """Learning rate scheduler class for the actor and critic networks, and entropy coefficient.

    See :ref:`learning_rate_schedulers` for more details.

    * If a class is provided, the same learning rate scheduler will be used for the networks/coefficient.
    * If a tuple is provided, its elements will be used for each network/coefficient in order.
    """

    learning_rate_scheduler_kwargs: dict | tuple[dict, dict, dict] = dataclasses.field(default_factory=dict)
    """Keyword arguments for the learning rate scheduler's constructor.

    See :ref:`learning_rate_schedulers` for more details.

    .. warning::

        The ``optimizer`` argument is automatically passed to the learning rate scheduler's constructor.
        Therefore, it must not be provided in the keyword arguments.

    * If a dictionary is provided, the same keyword arguments will be used for the networks/coefficient.
    * If a tuple is provided, its elements will be used for each network/coefficient in order.
    """

    observation_preprocessor: type | None = None
    """Preprocessor class to process the environment's observations.

    See :ref:`preprocessors` for more details.
    """

    observation_preprocessor_kwargs: dict = dataclasses.field(default_factory=dict)
    """Keyword arguments for the observation preprocessor's constructor.

    See :ref:`preprocessors` for more details.
    """

    state_preprocessor: type | None = None
    """Preprocessor class to process the environment's states.

    See :ref:`preprocessors` for more details.
    """

    state_preprocessor_kwargs: dict = dataclasses.field(default_factory=dict)
    """Keyword arguments for the state preprocessor's constructor.

    See :ref:`preprocessors` for more details.
    """

    random_timesteps: int = 0
    """Number of random exploration (sampling random actions) steps to perform before sampling actions from the policy."""

    learning_starts: int = 0
    """Number of steps to perform before calling the algorithm update function."""

    grad_norm_clip: float = 0
    """Clipping coefficient for the gradients by their global norm.

    If less than or equal to 0, the gradients will not be clipped.
    """

    learn_entropy: bool = True
    """Whether to learn the entropy coefficient."""

    initial_entropy_value: float = 0.2
    """Initial value for the entropy coefficient."""

    target_entropy: float | None = None
    """Target value for computing the entropy loss."""

    rewards_shaper: Callable | None = None
    """Rewards shaping function."""

    mixed_precision: bool = False
    """Whether to enable automatic mixed precision for higher performance."""

    # RLPD-specific model hints (read by examples when building networks)
    critic_layer_norm: bool = False
    """Enable LayerNorm on critic hidden layers (final scalar head stays without LN)."""

    layer_norm_affine: bool = True
    """Whether LayerNorm uses learnable affine parameters (gamma/beta). Defaults to True (matches Flax/Linen)."""

    # Ensemble and update-to-data controls (RLPD / REDQ style)
    num_qs: int = 5
    """Number of critic networks in the ensemble (E)."""

    num_min_qs: int | None = 2
    """Number of target critics to subsample for min aggregation (M). If None, use ``num_qs``."""

    utd_ratio: int = 1
    """Update-to-data ratio (slice a large batch into this many mini-batches; update critics UTD times, actor once)."""

    env_steps_per_update: int = 1
    """Number of environment steps to collect before triggering an update (>= 1)."""

    offline_ratio: float = 0.0
    """Fraction of each training batch sourced from offline data (D buffer)."""

    offline_pretrain_steps: int = 0
    """Number of offline-only update iterations to run before interacting with the environment."""

    offline_mix_mode: str = "shuffle"
    """Strategy to merge offline and online samples. Options: ``shuffle`` (default), ``interleave``, ``sequential``."""

    def expand(self) -> None:
        """Expand the configuration (mirrors SAC_CFG.expand)."""
        super().expand()
        # learning rate
        if not isinstance(self.learning_rate, (tuple, list)):
            self.learning_rate = (self.learning_rate, self.learning_rate, self.learning_rate)
        # learning rate scheduler
        if self.learning_rate_scheduler is None:
            self.learning_rate_scheduler = (None, None, None)
        elif not isinstance(self.learning_rate_scheduler, (tuple, list)):
            self.learning_rate_scheduler = (
                self.learning_rate_scheduler,
                self.learning_rate_scheduler,
                self.learning_rate_scheduler,
            )
        # learning rate scheduler kwargs
        if not isinstance(self.learning_rate_scheduler_kwargs, (tuple, list)):
            self.learning_rate_scheduler_kwargs = (
                self.learning_rate_scheduler_kwargs,
                self.learning_rate_scheduler_kwargs,
                self.learning_rate_scheduler_kwargs,
            )

        # validate ensemble-related fields
        if self.num_qs <= 0:
            raise ValueError("num_qs must be >= 1")
        if self.num_min_qs is None:
            self.num_min_qs = self.num_qs
        if self.num_min_qs <= 0 or self.num_min_qs > self.num_qs:
            raise ValueError("num_min_qs must be in [1, num_qs]")
        if self.utd_ratio <= 0:
            raise ValueError("utd_ratio must be >= 1")

        if not 0.0 <= self.offline_ratio <= 1.0:
            raise ValueError("offline_ratio must be in [0, 1]")
        if self.offline_pretrain_steps < 0:
            raise ValueError("offline_pretrain_steps must be >= 0")
        allowed_mix_modes = {"shuffle", "interleave", "sequential"}
        if self.offline_mix_mode.lower() not in allowed_mix_modes:
            raise ValueError(
                f"offline_mix_mode must be one of {allowed_mix_modes}, got '{self.offline_mix_mode}'"
            )
