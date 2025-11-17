import argparse
import os

# IMPORTANT: delay importing torch/skrl heavy modules until after SimulationApp
# is created (via load_isaaclab_env). This avoids GLIBCXX/libstdc++ conflicts
# by letting Isaac Sim's runtime be loaded first.
from skrl import logger
from skrl.envs.loaders.torch import load_isaaclab_env


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=None, help="Load checkpoint from path")
parser.add_argument("--eval", action="store_true", help="Run in evaluation mode (logging/checkpointing disabled)")
parser.add_argument("--timesteps", type=int, default=160000, help="Number of timesteps to run the trainer")
parser.add_argument("--no_pbar", action="store_true", help="Disable progress bar output")
parser.add_argument("--gradient_steps", type=int, default=1, help="Number of gradient steps per env step")
parser.add_argument("--batch_size", type=int, default=4096, help="Batch size for updates")
parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate for the actor and critic networks")
parser.add_argument("--critic_layer_norm", action="store_true", help="Enable LayerNorm on critic hidden layers")
parser.add_argument("--ln_affine", dest="ln_affine", action="store_true", help="LayerNorm with learnable affine (gamma/beta)")
parser.add_argument("--no_ln_affine", dest="ln_affine", action="store_false", help="LayerNorm without learnable affine")
parser.set_defaults(ln_affine=True)
parser.add_argument("--num_qs", type=int, default=5, help="Number of critic networks (ensemble size E)")
parser.add_argument("--num_min_qs", type=int, default=2, help="Number of target critics for min (subset size M)")
parser.add_argument("--utd_ratio", type=int, default=1, help="Update-to-data ratio (UTD)")
parser.add_argument("--offline_dataset", type=str, default=None, help="Path to offline dataset (.pt)")
parser.add_argument("--offline_ratio", type=float, default=0.5, help="Fraction of each batch drawn from offline data")
parser.add_argument("--offline_pretrain_steps", type=int, default=0, help="Number of offline-only updates before training")
parser.add_argument("--rollout_dataset", type=str, default=None, help="Save collected transitions to this .pt file (forces eval mode)")

# load the environment FIRST so that SimulationApp initializes and resolves
# runtime libraries before importing torch/skrl heavy modules.
task_name = "Isaac-Ant-Direct-v0"
env = load_isaaclab_env(task_name=task_name, parser=parser, num_envs=64)

# Now import torch/skrl heavy modules safely after SimulationApp is alive
import torch
import torch.nn as nn
from skrl.datasets import OfflineDataset
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.models.torch.mlp_ln import RLPDStateActionCritic
from skrl.agents.torch.rlpd import RLPD, RLPD_CFG
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


# define models (stochastic and deterministic models) using mixins
class StochasticActor(GaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        state_space,
        action_space,
        device,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-5,
        max_log_std=2,
        reduction="sum",
    ):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
            reduction=reduction,
        )

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_actions),
            nn.Tanh(),
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["observations"]), {"log_std": self.log_std_parameter}


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


# defer parsing of arguments to include loader arguments (run with --help to see all the arguments)
args, _ = parser.parse_known_args()


# seed for reproducibility
set_seed(args.seed)  # e.g. `set_seed(42)` for fixed seed

if args.offline_ratio > 0.0 and not args.offline_dataset:
    logger.error("--offline_ratio > 0 requires --offline_dataset")
    exit(1)


# instantiate a replay memory
BASE_MEMORY_SIZE = 60000
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
memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=memory_device)


# instantiate the agent's models (function approximators)
models = {}
models["policy"] = StochasticActor(env.observation_space, env.state_space, env.action_space, device)

# choose critic implementation and build ensemble
if args.critic_layer_norm:
    def critic_factory():
        return RLPDStateActionCritic(
            env.observation_space, env.state_space, env.action_space, device,
            hidden_dims=(512, 256), activation=nn.ReLU, layer_norm_affine=args.ln_affine,
        )
else:
    def critic_factory():
        return Critic(env.observation_space, env.state_space, env.action_space, device)

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
cfg.discount_factor = 0.99
cfg.polyak = 0.005
cfg.learning_rate = args.learning_rate
cfg.random_timesteps = 50
cfg.learning_starts = 50
cfg.learn_entropy = True
cfg.initial_entropy_value = 1.0
cfg.critic_layer_norm = args.critic_layer_norm
cfg.layer_norm_affine = args.ln_affine
cfg.num_qs = args.num_qs
cfg.num_min_qs = args.num_min_qs
cfg.utd_ratio = args.utd_ratio
cfg.offline_ratio = args.offline_ratio
cfg.offline_pretrain_steps = args.offline_pretrain_steps
cfg.state_preprocessor = RunningStandardScaler
cfg.state_preprocessor_kwargs = {"size": env.observation_space, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg.experiment.write_interval = "auto" if not args.eval else 0
cfg.experiment.checkpoint_interval = "auto" if not args.eval else 0
suffix_parts = []
if args.critic_layer_norm:
    suffix_parts.append("LN" if args.ln_affine else "LN_noaff")
suffix_parts.append(f"Q{args.num_qs}M{args.num_min_qs}")
suffix_parts.append(f"UTD{args.utd_ratio}")
if args.offline_dataset:
    suffix_parts.append(f"Off{int(args.offline_ratio * 100):02d}")
suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
cfg.experiment.directory = f"runs/torch/{task_name}{suffix}"

agent = RLPD(
    models=models,
    memory=memory,
    cfg=cfg,
    observation_space=env.observation_space,
    state_space=env.state_space,
    action_space=env.action_space,
    device=device,
    offline_dataset=offline_dataset,
)

if args.offline_pretrain_steps > 0:
    if offline_dataset is None:
        logger.error("Offline pretraining requested but no offline dataset provided")
        exit(1)
    logger.info(f"Running {args.offline_pretrain_steps} offline pretrain updates")
    agent.run_offline_updates(args.offline_pretrain_steps)


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
