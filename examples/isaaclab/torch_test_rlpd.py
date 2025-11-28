import argparse
import os
from datetime import datetime
from typing import Optional, cast

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
    "offline_pretrain_steps",
)
BUFFER_TRANSITION_UPLIMIT = 1000000


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
parser.add_argument("--offline_dataset", type=str, default=None, help="Path to offline dataset (.pt)")
parser.add_argument("--offline_ratio", type=float, default=0.5, help="Fraction of each batch drawn from offline data")
parser.add_argument("--offline_pretrain_steps", type=int, default=0, help="Number of offline-only updates before training")
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


class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations + self.num_actions, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["observations"], inputs["taken_actions"]], dim=1)), {}


# wrap the environment
env = wrap_env(env)
observation_space = cast(gym.Space, env.observation_space)
state_space = cast(Optional[gym.Space], getattr(env, "state_space", None))
action_space = cast(gym.Space, env.action_space)

# Isaac Lab joint targets expect [-1, 1]; override underlying Box so GaussianMixin can clamp
try:
    import numpy as np
    from gymnasium import spaces

    act_shape = action_space.shape
    bounded_space = spaces.Box(
        low=-np.ones(act_shape, dtype=np.float32),
        high=np.ones(act_shape, dtype=np.float32),
        dtype=np.float32,
    )
    # property returns _unwrapped.single_action_space, so patch that object
    env.unwrapped.single_action_space = bounded_space
except Exception as e:
    logger.warning(f"Failed to override action space bounds: {e}")
device = env.device


# defer parsing of arguments to include loader arguments (run with --help to see all the arguments)
args, _ = parser.parse_known_args()

wandb_run = None

# seed for reproducibility
set_seed(args.seed)  # e.g. `set_seed(42)` for fixed seed

if args.offline_ratio > 0.0 and not args.offline_dataset:
    logger.error("--offline_ratio > 0 requires --offline_dataset")
    exit(1)


# instantiate a replay memory
BASE_MEMORY_SIZE = int(BUFFER_TRANSITION_UPLIMIT / env.num_envs)
memory_device = device
memory_size = BASE_MEMORY_SIZE

if args.rollout_dataset:
    requested_samples = env.num_envs * args.timesteps + 1
    if requested_samples > BASE_MEMORY_SIZE:
        logger.error(
            "Rollout request (%d samples) exceeds memory capacity (%d). Reduce timesteps or num_envs.",
            requested_samples,
            BASE_MEMORY_SIZE,
        )
        exit(1)
    memory_size = requested_samples
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
    return Critic(observation_space, state_space, action_space, device)


E = max(1, int(args.num_qs))
for i in range(1, E + 1):
    models[f"critic_{i}"] = critic_factory()
for i in range(1, E + 1):
    models[f"target_critic_{i}"] = critic_factory()

offline_dataset = None
if args.offline_dataset:
    offline_dataset = OfflineDataset(device=device)
    offline_dataset.load(args.offline_dataset)


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
cfg.offline_ratio = args.offline_ratio
cfg.offline_pretrain_steps = args.offline_pretrain_steps

cfg.state_preprocessor = RunningStandardScaler
cfg.state_preprocessor_kwargs = {"size": observation_space, "device": device}

# logging to TensorBoard and write checkpoints (in timesteps)
cfg.experiment.write_interval = "auto" if not args.eval else 0
cfg.experiment.checkpoint_interval = "auto" if not args.eval else 0


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


import wandb

date_prefix = datetime.now().strftime("%Y%m%d")
run_name_suffix = args.wandb_run_name or ""
wandb_run_name = (
    f"{date_prefix}{run_name_suffix}" if run_name_suffix else date_prefix
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

cfg.experiment.directory = f"runs/torch/{WANDB_PROJECT_NAME}/{wandb_run_name}"
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

if args.offline_pretrain_steps > 0:
    if offline_dataset is None:
        logger.error("Offline pretraining requested but no offline dataset provided")
        exit(1)
    logger.info(f"Running {args.offline_pretrain_steps} offline pretrain updates")
    agent.run_offline_updates(args.offline_pretrain_steps)

run_eval = args.eval or bool(args.rollout_dataset)
if run_eval:
    trainer.eval()
else:
    trainer.train()

if args.rollout_dataset:
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
