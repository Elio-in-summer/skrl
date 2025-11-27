import argparse
import os
from typing import Optional, cast

# IMPORTANT: delay importing torch/skrl heavy modules until after SimulationApp
# is created (via load_isaaclab_env). This avoids GLIBCXX/libstdc++ conflicts
# by letting Isaac Sim's runtime be loaded first.
from skrl import logger
from skrl.envs.loaders.torch import load_isaaclab_env


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=None, help="Load checkpoint from path")
parser.add_argument("--eval", action="store_true", help="Run in evaluation mode (logging/checkpointing disabled)")

# load the environment FIRST so that SimulationApp initializes and resolves
# runtime libraries before importing torch/skrl heavy modules.
task_name = "Isaac-Ant-Direct-v0"
env = load_isaaclab_env(task_name=task_name, parser=parser, num_envs=64)

# Now import torch/skrl heavy modules safely after SimulationApp is alive
import torch
import torch.nn as nn
import gymnasium as gym
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, Model
from skrl.models.torch.rlpd_actor import RLPDTanhGaussianActor
from skrl.agents.torch.sac import SAC, SAC_CFG
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
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["observations"], inputs["taken_actions"]], dim=1)), {}


# wrap the environment
env = wrap_env(env)
device = env.device

observation_space = cast(gym.Space, env.observation_space)
state_space = cast(Optional[gym.Space], getattr(env, "state_space", None))
action_space = cast(gym.Space, env.action_space)


# defer parsing of arguments to include loader arguments (run with --help to see all the arguments)
args, _ = parser.parse_known_args()


# seed for reproducibility
set_seed(args.seed)  # e.g. `set_seed(42)` for fixed seed


# instantiate a replay memory
memory = RandomMemory(memory_size=16000, num_envs=env.num_envs, device=device)


# instantiate the agent's models (function approximators).
# SAC requires 5 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#models
models = {}
models["policy"] = RLPDTanhGaussianActor(
    observation_space,
    state_space,
    action_space,
    device,
    hidden_dims=(512, 256),
    activation=nn.ReLU,
    log_std_bounds=(-5.0, 2.0),
)
models["critic_1"] = Critic(observation_space, state_space, action_space, device)
models["critic_2"] = Critic(observation_space, state_space, action_space, device)
models["target_critic_1"] = Critic(observation_space, state_space, action_space, device)
models["target_critic_2"] = Critic(observation_space, state_space, action_space, device)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#configuration-and-hyperparameters
cfg = SAC_CFG()
cfg.gradient_steps = 1
cfg.batch_size = 4096
cfg.discount_factor = 0.99
cfg.polyak = 0.005
cfg.learning_rate = 5e-4
cfg.random_timesteps = 50
cfg.learning_starts = 50
cfg.learn_entropy = True
cfg.initial_entropy_value = 1.0
cfg.state_preprocessor = RunningStandardScaler
cfg.state_preprocessor_kwargs = {"size": env.observation_space, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg.experiment.write_interval = "auto" if not args.eval else 0
cfg.experiment.checkpoint_interval = "auto" if not args.eval else 0
cfg.experiment.directory = f"runs/torch/{task_name}"

agent = SAC(
    models=models,
    memory=memory,
    cfg=cfg,
    observation_space=observation_space,
    state_space=state_space,
    action_space=action_space,
    device=device,
)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 100000, "headless": args.headless}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

if args.checkpoint:
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint file not found: '{args.checkpoint}'")
        exit(1)
    agent.load(args.checkpoint)

trainer.train() if not args.eval else trainer.eval()
