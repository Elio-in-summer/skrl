import argparse
import os
import sys
from datetime import datetime
from typing import Optional, cast

# IMPORTANT: delay importing torch/skrl heavy modules until after SimulationApp
# is created (via load_isaaclab_env). This avoids GLIBCXX/libstdc++ conflicts
# by letting Isaac Sim's runtime be loaded first.
from skrl import logger
from skrl.envs.loaders.torch import load_isaaclab_env


WANDB_PROJECT_NAME = "Cube-Allegro"
WANDB_ENTITY_NAME = "DexGen_ZBL"
# Training parameters to track in Weights & Biases
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
    "hq_traj_enable",
    "hq_traj_threshold",
    "hq_traj_increment_every",
    "discount_factor",
    "polyak",
    "initial_entropy_value",
    "random_timesteps",
    "learning_starts",
    "enable_proposal",
    "proposal_beta",
)


BUFFER_TRANSITION_UPLIMIT = 1000000
HQ_TRAJECTORY_TARGET = 200
HQ_COMMAND_NAME = "object_pose"
HQ_METRIC_NAME = "consecutive_success"


# parse arguments
parser = argparse.ArgumentParser()
# eval mode relavent
parser.add_argument("--checkpoint", type=str, default=None, help="Load checkpoint from path")
parser.add_argument("--eval", action="store_true", help="Run in evaluation mode (logging/checkpointing disabled)")
# progeress monitoring
parser.add_argument("--timesteps", type=int, default=100000, help="Number of timesteps to run the trainer")
parser.add_argument("--no_pbar", action="store_true", help="Disable progress bar output")
# training parameters affecting UTD ratio
parser.add_argument("--gradient_steps", type=int, default=1, help="Number of gradient steps per env step")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size for updates")
parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate for the actor and critic networks")
parser.add_argument("--utd_ratio", type=int, default=1, help="Update N times Critic Then Update Actor")
parser.add_argument(
    "--steps_per_update",
    type=int,
    default=1,
    help="Number of environment steps to collect before triggering learning updates",
)
# feature of critic layernorm
parser.add_argument("--critic_layer_norm", action="store_true", help="Enable LayerNorm on critic hidden layers")
parser.add_argument("--ln_affine", dest="ln_affine", action="store_true", help="LayerNorm with learnable affine (gamma/beta)")
parser.add_argument("--no_ln_affine", dest="ln_affine", action="store_false", help="LayerNorm without learnable affine")
parser.set_defaults(ln_affine=True)
# feature of randomlized ensemble Q
parser.add_argument("--num_qs", type=int, default=10, help="Number of critic networks (ensemble size E)")
parser.add_argument("--num_min_qs", type=int, default=1, help="Number of target critics for min (subset size M)")
# feature of mix offline and online data
parser.add_argument("--offline_dataset", type=str, default=None, help="Path to offline dataset (.pt)")
parser.add_argument("--offline_ratio", type=float, default=0.5, help="Fraction of each batch drawn from offline data")
# feature of high-quality trajectory filtering & add to offline dataset or rollout
parser.add_argument("--hq_traj_enable", action="store_true", help="Enable high-quality trajectory filtering")
parser.add_argument(
    "--hq_traj_threshold",
    type=int,
    default=1,
    help="Threshold on consecutive_success used to flag high-quality trajectories",
)
parser.add_argument(
    "--hq_traj_increment_every",
    type=int,
    default=HQ_TRAJECTORY_TARGET,
    help="Number of HQ trajectories to collect before raising the threshold during training (set 0 to disable)",
)
# feature of actor proposal & bootstrap proposal 
parser.add_argument("--enable_proposal", action="store_true", help="Enable IBRL actor/bootstrap proposals")
parser.add_argument("--il_policy_checkpoint", type=str, default=None, help="Checkpoint for IL policy used in proposals")
parser.add_argument("--proposal_beta", type=float, default=10.0, help="Inverse temperature for proposal softmax")
# rollout mode relavent
parser.add_argument("--rollout_dataset", type=str, default=None, help="Save collected transitions to this .pt file (forces eval mode)")
# wandb relavent
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
# if human intervention is enabled, match simulation speed to wall-clock time during eval/rollout
parser.add_argument("--real_time", action="store_true", help="Match simulation speed to wall-clock time during eval/rollout")
# training parameters which not so important yet, usually not changed
parser.add_argument("--discount_factor", type=float, default=0.97, help="Discount factor for the RL agent")
parser.add_argument("--polyak", type=float, default=0.005, help="Polyak factor for the RL agent")
parser.add_argument("--initial_entropy_value", type=float, default=1, help="Initial entropy value for the RL agent")
parser.add_argument("--random_timesteps", type=int, default=1000, help="Number of random timesteps to collect before training")
parser.add_argument("--learning_starts", type=int, default=1000, help="Number of learning starts for the RL agent")


# load the environment FIRST so that SimulationApp initializes and resolves
# runtime libraries before importing torch/skrl heavy modules.
task_name = "Isaac-Repose-Cube-Allegro-v0"
env = load_isaaclab_env(task_name=task_name, parser=parser, num_envs=64)

# Now import torch/skrl heavy modules safely after SimulationApp is alive
import numpy as np
import torch
import torch.nn as nn
from types import MethodType

import gymnasium as gym
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, Model
from skrl.models.torch.rlpd_actor import RLPDTanhGaussianActor
from skrl.models.torch.mlp_ln import RLPDStateActionCritic
from skrl.agents.torch.rlpd import RLPD, RLPD_CFG
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.trainers.torch.shared_autonomy_sequential import SharedAutonomySequentialTrainer
from skrl.utils import set_seed
from o2o_rl.utils.high_quality import (
    BoundedOfflineDataset,
    HighQualityTrajectoryManager,
    HighQualityTrajectoryTargetReached,
)
from o2o_rl.utils.shared_autonomy import SharedAutonomyController
from o2o_rl.utils.allegro_zmq import ZMQCommandListener, ExternalActionProcessor
from o2o_rl.utils.il_policy import load_il_policy

# wrap the environment
env = wrap_env(env)


# Normalize action space to [-1, 1] before exposing it to the agent
act_shape = env.action_space.shape
bounded_space = gym.spaces.Box(
    low=-np.ones(act_shape, dtype=np.float32),
    high=np.ones(act_shape, dtype=np.float32),
    dtype=np.float32,
)
env.unwrapped.single_action_space = bounded_space

observation_space = cast(gym.Space, env.observation_space)
state_space = cast(Optional[gym.Space], getattr(env, "state_space", None))
action_space = cast(gym.Space, env.action_space)
logger.info(f"Action space: {action_space}")
logger.info(f"Observation space: {observation_space}")
logger.info(f"State space: {state_space}")
device = env.device


# defer parsing of arguments to include loader arguments (run with --help to see all the arguments)
args, _ = parser.parse_known_args()
run_eval = args.eval or bool(args.rollout_dataset)

wandb_run = None
enable_wandb = not run_eval

if args.real_time and not run_eval:
    logger.error("--real_time is only supported in evaluation or rollout modes")
    exit(1)

# seed for reproducibility
set_seed(args.seed)  # e.g. `set_seed(42)` for fixed seed

if args.offline_ratio > 0.0 and not args.offline_dataset and not run_eval:
    logger.error("--offline_ratio > 0 requires --offline_dataset")
    exit(1)


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

il_policy = None
if args.enable_proposal:
    if not args.il_policy_checkpoint:
        logger.error("--enable_proposal requires --il_policy_checkpoint")
        exit(1)
    if not os.path.exists(args.il_policy_checkpoint):
        logger.error(f"IL policy checkpoint not found: '{args.il_policy_checkpoint}'")
        exit(1)
    il_policy = load_il_policy(
        observation_space,
        state_space,
        action_space,
        device,
        args.il_policy_checkpoint,
        hidden_dims=(512, 256, 128),
        activation=nn.ELU,
        log_std_bounds=(-20.0, 2.0),
    )
    logger.info("Loaded IL policy from %s", args.il_policy_checkpoint)

date_prefix = datetime.now().strftime("%Y%m%d")
run_name_suffix = args.wandb_run_name or ""
wandb_run_name = (
    f"{date_prefix}{run_name_suffix}" if run_name_suffix else date_prefix
)

# configure and instantiate the agent (visit其文档查看所有参数)
cfg = RLPD_CFG()
cfg.gradient_steps = args.gradient_steps
cfg.batch_size = args.batch_size
cfg.discount_factor = args.discount_factor
cfg.polyak = args.polyak
cfg.learning_rate = args.learning_rate
cfg.random_timesteps = args.random_timesteps if not run_eval else 0
cfg.learning_starts = args.learning_starts if not run_eval else 0
cfg.learn_entropy = True
cfg.initial_entropy_value = args.initial_entropy_value
cfg.critic_layer_norm = args.critic_layer_norm
cfg.layer_norm_affine = args.ln_affine
cfg.num_qs = args.num_qs
cfg.num_min_qs = args.num_min_qs
cfg.utd_ratio = args.utd_ratio
cfg.env_steps_per_update = args.steps_per_update
cfg.offline_ratio = args.offline_ratio
cfg.enable_proposal = bool(args.enable_proposal)
cfg.proposal_softmax_beta = float(args.proposal_beta)

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
    il_policy=il_policy,
)

tags = args.wandb_tags or None
if tags:
    if len(tags) == 1 and "," in tags[0]:
        tags = [tag.strip() for tag in tags[0].split(",") if tag.strip()]
    if not tags:
        tags = None

if enable_wandb:
    import wandb

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
    if wandb_run is None:
        raise RuntimeError("Failed to initialize Weights & Biases run")
    wandb_session = wandb_run

    original_write_tracking_data = agent.write_tracking_data

    def write_tracking_data_with_wandb(self, *, timestep: int, timesteps: int) -> None:
        if self.tracking_data:
            metrics = {}
            for key, values in self.tracking_data.items():
                if values:
                    metrics[key] = sum(values) / len(values)
            if metrics:
                metrics.setdefault("timestep", timestep)
                wandb_session.log(metrics, step=timestep)
        original_write_tracking_data(timestep=timestep, timesteps=timesteps)

    agent.write_tracking_data = MethodType(write_tracking_data_with_wandb, agent)

hq_manager: Optional[HighQualityTrajectoryManager] = None
if args.hq_traj_enable:
    base_env = getattr(env, "_unwrapped", None) or getattr(env, "unwrapped", None) or env
    try:
        hq_manager = HighQualityTrajectoryManager(
            agent=agent,
            base_env=base_env,
            num_envs=env.num_envs,
            threshold=args.hq_traj_threshold,
            target_episodes=HQ_TRAJECTORY_TARGET,
            command_name=HQ_COMMAND_NAME,
            metric_name=HQ_METRIC_NAME,
            rollout_mode=bool(args.rollout_dataset),
            offline_dataset=offline_dataset,
            device=device,
            max_transitions=BUFFER_TRANSITION_UPLIMIT,
            threshold_increment_every=args.hq_traj_increment_every,
            offline_ratio=cfg.offline_ratio,
        )
        if hq_manager.offline_dataset is not None:
            offline_dataset = hq_manager.offline_dataset
    except Exception as exc:
        logger.error(f"Failed to initialize high-quality trajectory collector: {exc}")
        exit(1)


# configure and instantiate the RL trainer
cfg_trainer = {
    "timesteps": args.timesteps,
    "headless": args.headless,
    "disable_progressbar": (args.eval or args.no_pbar),
}
trainer_cls = SharedAutonomySequentialTrainer if run_eval else SequentialTrainer
trainer_kwargs = {}
if run_eval and trainer_cls is SharedAutonomySequentialTrainer:
    trainer_kwargs["real_time"] = bool(args.real_time)
    step_dt = getattr(env, "step_dt", None)
    if step_dt is None:
        step_dt = getattr(getattr(env, "unwrapped", None), "step_dt", None)
    trainer_kwargs["real_time_dt"] = step_dt
trainer = trainer_cls(cfg=cfg_trainer, env=env, agents=agent, **trainer_kwargs)

if args.checkpoint:
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint file not found: '{args.checkpoint}'")
        exit(1)
    agent.load(args.checkpoint)
shared_controller = None
if run_eval:
    listener = ZMQCommandListener()
    processor = ExternalActionProcessor(env)
    shared_controller = SharedAutonomyController(trainer, listener, processor)
    shared_controller.start()
    try:
        if hq_manager is not None:
            try:
                trainer.eval()
            except HighQualityTrajectoryTargetReached:
                logger.info(
                    "Collected %d high-quality trajectories (target %d); stopping rollout",
                    hq_manager.collected,
                    hq_manager.target,
                )
        else:
            trainer.eval()
    finally:
        shared_controller.stop()
else:
    trainer.train()

if args.rollout_dataset:
    if hq_manager is not None:
        hq_manager.save_rollout_dataset(args.rollout_dataset)
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
