import argparse
import os
from datetime import datetime
from typing import Any, Callable, Optional, cast

# IMPORTANT: delay importing torch/skrl heavy modules until after SimulationApp
# is created (via load_isaaclab_env). This avoids GLIBCXX/libstdc++ conflicts
# by letting Isaac Sim's runtime be loaded first.
from skrl import logger
from skrl.envs.loaders.torch import load_isaaclab_env

WANDB_PROJECT_NAME = "Cube-Allegro"
WANDB_ENTITY_NAME = "DexGen_ZBL"
WANDB_CONFIG_KEYS = (
    "gradient_steps",
    "batch_size",
    "learning_rate",
    "critic_layer_norm",
    "ln_affine",
    "num_qs",
    "num_min_qs",
    "utd_ratio",
    "offline_dataset",
    "offline_ratio",
)
BUFFER_TRANSITION_UPLIMIT = 1000000
HQ_TRAJECTORY_TARGET = 200
HQ_COMMAND_NAME = "object_pose"
HQ_METRIC_NAME = "consecutive_success"


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=None, help="Load checkpoint from path")
parser.add_argument("--eval", action="store_true", help="Run in evaluation mode (logging/checkpointing disabled)")
parser.add_argument("--timesteps", type=int, default=100000, help="Number of timesteps to run the trainer")
parser.add_argument("--no_pbar", action="store_true", help="Disable progress bar output")
parser.add_argument("--gradient_steps", type=int, default=1, help="Number of gradient steps per env step")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size for updates")
parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate for the actor and critic networks")
parser.add_argument("--critic_layer_norm", action="store_true", help="Enable LayerNorm on critic hidden layers")
parser.add_argument("--ln_affine", dest="ln_affine", action="store_true", help="LayerNorm with learnable affine (gamma/beta)")
parser.add_argument("--no_ln_affine", dest="ln_affine", action="store_false", help="LayerNorm without learnable affine")
parser.set_defaults(ln_affine=True)
parser.add_argument("--num_qs", type=int, default=10, help="Number of critic networks (ensemble size E)")
parser.add_argument("--num_min_qs", type=int, default=1, help="Number of target critics for min (subset size M)")
parser.add_argument("--utd_ratio", type=int, default=1, help="Update-to-data ratio (UTD)")
parser.add_argument(
    "--steps_per_update",
    type=int,
    default=1,
    help="Number of environment steps to collect before triggering learning updates",
)
parser.add_argument("--offline_dataset", type=str, default=None, help="Path to offline dataset (.pt)")
parser.add_argument("--offline_ratio", type=float, default=0.5, help="Fraction of each batch drawn from offline data")
parser.add_argument("--rollout_dataset", type=str, default=None, help="Save collected transitions to this .pt file (forces eval mode)")
parser.add_argument(
    "--wandb_run_name",
    type=str,
    default=None,
    help="Weights & Biases run name suffix (prefixed with YYYYMMDD)",
)
parser.add_argument(
    "--wandb_tags",
    type=str,
    nargs="*",
    default=None,
    help="Optional list of tags for the W&B run",
)
parser.add_argument("--hq_traj_enable", action="store_true", help="Enable high-quality trajectory filtering")
parser.add_argument(
    "--hq_traj_threshold",
    type=int,
    default=10,
    help="Threshold on consecutive_success used to flag high-quality trajectories",
)

# load the environment FIRST so that SimulationApp initializes and resolves
# runtime libraries before importing torch/skrl heavy modules.
task_name = "Isaac-Repose-Cube-Allegro-v0"
env = load_isaaclab_env(task_name=task_name, parser=parser, num_envs=64)

# Now import torch/skrl heavy modules safely after SimulationApp is alive
import torch
import torch.nn as nn
from types import MethodType

import gymnasium as gym
from skrl.datasets import OfflineDataset
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, Model
from skrl.models.torch.rlpd_actor import RLPDTanhGaussianActor
from skrl.models.torch.mlp_ln import RLPDStateActionCritic
from skrl.agents.torch.rlpd import RLPD, RLPD_CFG
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


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
        command_name: str = HQ_COMMAND_NAME,
        metric_name: str = HQ_METRIC_NAME,
        store_episodes: bool = True,
        episode_callback: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self._base_env = base_env
        self._num_envs = num_envs
        self._threshold = threshold
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
    def target(self) -> int:
        return self._target

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
    def __init__(self, *args, max_transitions: int = BUFFER_TRANSITION_UPLIMIT, **kwargs) -> None:
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


# wrap the environment
env = wrap_env(env)
observation_space = cast(gym.Space, env.observation_space)
state_space = cast(Optional[gym.Space], getattr(env, "state_space", None))
action_space = cast(gym.Space, env.action_space)

device = env.device


# defer parsing of arguments to include loader arguments (run with --help to see all the arguments)
args, _ = parser.parse_known_args()

wandb_run = None

# seed for reproducibility
set_seed(args.seed)  # e.g. `set_seed(42)` for fixed seed

# if args.offline_ratio > 0.0 and not args.offline_dataset:
#     logger.error("--offline_ratio > 0 requires --offline_dataset")
#     exit(1)


# instantiate a replay memory
BASE_MEMORY_SIZE = int(BUFFER_TRANSITION_UPLIMIT / env.num_envs)
memory_device = device
memory_size = BASE_MEMORY_SIZE

if args.rollout_dataset:
    memory_device = "cpu"


logger.info(f"Memory size: {memory_size}, Memory device: {memory_device}")
memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=memory_device)


# instantiate the agent's models (function approximators)
models = {}
models["policy"] = RLPDTanhGaussianActor(
    observation_space,
    state_space,
    action_space,
    device,
    hidden_dims=(512, 256, 128),
    activation=nn.ELU,
    log_std_bounds=(-20.0, 2.0), # smaller log_std_bounds allow policy have bigger entropy, high-dim actor needs more entropy
)

# choose critic implementation and build ensemble
if args.critic_layer_norm:
    logger.info("Using RLPDStateActionCritic with layer normalization")


def critic_factory():
    if args.critic_layer_norm:
        return RLPDStateActionCritic(
            observation_space,
            state_space,
            action_space,
            device,
            hidden_dims=(512, 256, 128),
            activation=nn.ELU,
            layer_norm_affine=args.ln_affine,
        )
    else:
        raise ValueError("Critic Without Layer Normalization is Not Supported！")

E = max(1, int(args.num_qs))
for i in range(1, E + 1):
    models[f"critic_{i}"] = critic_factory()
for i in range(1, E + 1):
    models[f"target_critic_{i}"] = critic_factory()

def _create_offline_dataset() -> BoundedOfflineDataset:
    return BoundedOfflineDataset(device=device, max_transitions=BUFFER_TRANSITION_UPLIMIT)


offline_dataset: Optional[BoundedOfflineDataset] = None
if args.offline_dataset:
    offline_dataset = _create_offline_dataset()
    offline_dataset.load(args.offline_dataset)
elif args.hq_traj_enable and not args.rollout_dataset:
    offline_dataset = _create_offline_dataset()


import wandb

date_prefix = datetime.now().strftime("%Y%m%d")
run_name_suffix = args.wandb_run_name or ""
wandb_run_name = (
    f"{date_prefix}{run_name_suffix}" if run_name_suffix else date_prefix
)

# configure and instantiate the agent (visit其文档查看所有参数)
cfg = RLPD_CFG()
cfg.gradient_steps = args.gradient_steps
cfg.batch_size = args.batch_size
cfg.discount_factor = 0.97
cfg.polyak = 0.005
cfg.learning_rate = args.learning_rate
cfg.random_timesteps = 1000  # better early coverage; can be overridden below
cfg.learning_starts = 1000
cfg.learn_entropy = True
cfg.initial_entropy_value = 1
cfg.critic_layer_norm = args.critic_layer_norm
cfg.layer_norm_affine = args.ln_affine
cfg.num_qs = args.num_qs
cfg.num_min_qs = args.num_min_qs
cfg.utd_ratio = args.utd_ratio
cfg.env_steps_per_update = args.steps_per_update
cfg.offline_ratio = args.offline_ratio

cfg.state_preprocessor = RunningStandardScaler
cfg.state_preprocessor_kwargs = {"size": observation_space, "device": device}

# logging to TensorBoard and write checkpoints (in timesteps)
cfg.experiment.write_interval = "auto" if not args.eval else 0
cfg.experiment.checkpoint_interval = "auto" if not args.eval else 0
cfg.experiment.directory = f"runs/torch/{WANDB_PROJECT_NAME}/{wandb_run_name}"


agent = RLPD(
    models=models,
    memory=memory,
    cfg=cfg,
    observation_space=observation_space,
    state_space=state_space,
    action_space=action_space,
    device=device,
    offline_dataset=offline_dataset,
)

tags = args.wandb_tags or None
if tags:
    if len(tags) == 1 and "," in tags[0]:
        tags = [tag.strip() for tag in tags[0].split(",") if tag.strip()]
    if not tags:
        tags = None

wandb_config = {key: getattr(args, key, None) for key in WANDB_CONFIG_KEYS}
wandb_config["num_envs"] = env.num_envs
state_preprocessor = getattr(
    cfg.state_preprocessor,
    "__name__",
    str(cfg.state_preprocessor),
)
wandb_config["state_preprocessor"] = state_preprocessor

wandb_run = wandb.init(
    project=WANDB_PROJECT_NAME,
    entity=WANDB_ENTITY_NAME,
    name=wandb_run_name,
    config=wandb_config,
    tags=tags,
)

original_write_tracking_data = agent.write_tracking_data


def write_tracking_data_with_wandb(self, *, timestep: int, timesteps: int) -> None:
    if self.tracking_data:
        metrics = {}
        for key, values in self.tracking_data.items():
            if values:
                metrics[key] = sum(values) / len(values)
        if metrics:
            metrics.setdefault("timestep", timestep)
            wandb_run.log(metrics, step=timestep)
    original_write_tracking_data(timestep=timestep, timesteps=timesteps)


agent.write_tracking_data = MethodType(write_tracking_data_with_wandb, agent)

hq_collector = None
hq_mode: Optional[str] = None
hq_stats = {"episodes": 0, "transitions": 0}
if args.hq_traj_enable:
    base_env = getattr(env, "_unwrapped", None) or getattr(env, "unwrapped", None) or env
    try:
        if args.rollout_dataset:
            hq_collector = HighQualityTrajectoryCollector(
                base_env=base_env,
                num_envs=env.num_envs,
                threshold=args.hq_traj_threshold,
                target_episodes=HQ_TRAJECTORY_TARGET,
            )
            hq_mode = "rollout"
            logger.info(
                "High-quality rollout mode enabled (threshold=%d, target=%d trajectories)",
                args.hq_traj_threshold,
                HQ_TRAJECTORY_TARGET,
            )
        else:
            if offline_dataset is None:
                offline_dataset = _create_offline_dataset()
            if offline_dataset is None:
                raise RuntimeError("Failed to initialize offline dataset for HQ collection")

            def _consume_episode(episode: dict) -> None:
                flat_episode = {key: tensor for key, tensor in episode.items() if tensor is not None}
                offline_dataset.append(flat_episode)
                hq_stats["episodes"] += 1
                step_count = flat_episode["rewards"].shape[0]
                hq_stats["transitions"] += step_count
                if hasattr(agent, "track_data"):
                    agent.track_data("Data / HQ offline trajectories", float(hq_stats["episodes"]))
                    agent.track_data("Data / HQ offline transitions", float(hq_stats["transitions"]))

            hq_collector = HighQualityTrajectoryCollector(
                base_env=base_env,
                num_envs=env.num_envs,
                threshold=args.hq_traj_threshold,
                target_episodes=None,
                store_episodes=False,
                episode_callback=_consume_episode,
            )
            hq_mode = "train"
            logger.info(
                "High-quality training mode enabled (threshold=%d, offline budget=%d transitions)",
                args.hq_traj_threshold,
                offline_dataset.max_transitions,
            )
            if cfg.offline_ratio <= 0:
                logger.warning("Offline ratio is 0; high-quality trajectories will not be sampled during updates")
    except Exception as exc:
        logger.error(f"Failed to initialize high-quality trajectory collector: {exc}")
        exit(1)

if hq_collector is not None:
    original_record_transition = agent.record_transition

    def record_transition_with_hq(
        self,
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
        reached_target = hq_collector.process_transition(
            observations=observations,
            states=states,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            next_states=next_states,
            terminated=terminated,
            truncated=truncated,
        )
        original_record_transition(
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
        if reached_target and hq_mode == "rollout":
            raise HighQualityTrajectoryTargetReached

    agent.record_transition = MethodType(record_transition_with_hq, agent)


# configure and instantiate the RL trainer
cfg_trainer = {
    "timesteps": args.timesteps,
    "headless": args.headless,
    "disable_progressbar": (args.eval or args.no_pbar),
}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

if args.checkpoint:
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint file not found: '{args.checkpoint}'")
        exit(1)
    agent.load(args.checkpoint)


run_eval = args.eval or bool(args.rollout_dataset)
if run_eval:
    if hq_collector is not None:
        try:
            trainer.eval()
        except HighQualityTrajectoryTargetReached:
            logger.info(
                "Collected %d high-quality trajectories (target %d); stopping rollout",
                hq_collector.collected,
                hq_collector.target,
            )
    else:
        trainer.eval()
else:
    trainer.train()

if args.rollout_dataset:
    if hq_collector is not None:
        dataset = hq_collector.build_dataset()
        if not dataset:
            logger.warning("No high-quality trajectories collected; rollout dataset not saved")
        else:
            cpu_dataset = {key: value.clone().cpu() for key, value in dataset.items()}
            sample_count = next(iter(cpu_dataset.values())).shape[0]
            output_dir = os.path.dirname(args.rollout_dataset)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            torch.save(cpu_dataset, args.rollout_dataset)
            logger.info(
                "Saved %d high-quality transitions (%d trajectories) to '%s'",
                sample_count,
                hq_collector.collected,
                args.rollout_dataset,
            )
    else:
        num_samples = len(memory)
        if num_samples == 0:
            logger.warning("No samples collected; rollout dataset not saved")
        else:
            dataset = {}
            for name in agent._tensors_names:
                tensor_view = memory.tensors_view.get(name)
                if tensor_view is None:
                    continue
                tensor = tensor_view[:num_samples]
                dataset[name] = tensor.clone().cpu()
            output_dir = os.path.dirname(args.rollout_dataset)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            torch.save(dataset, args.rollout_dataset)
            logger.info(f"Saved {num_samples} transitions to '{args.rollout_dataset}'")

if wandb_run is not None:
    wandb_run.finish()
