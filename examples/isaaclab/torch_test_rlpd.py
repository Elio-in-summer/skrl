import argparse
import json
import os
import queue
import sys
import threading
import time
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
HQ_TRAJECTORY_TARGET = 10
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
    default=1,
    help="Threshold on consecutive_success used to flag high-quality trajectories",
)
parser.add_argument("--real_time", action="store_true", help="Match simulation speed to wall-clock time during eval/rollout")

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
from skrl.datasets import OfflineDataset
from skrl.envs.wrappers.torch import Wrapper, wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, Model
from skrl.models.torch.rlpd_actor import RLPDTanhGaussianActor
from skrl.models.torch.mlp_ln import RLPDStateActionCritic
from skrl.agents.torch.rlpd import RLPD, RLPD_CFG
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.trainers.torch.shared_autonomy_sequential import SharedAutonomySequentialTrainer
from skrl.utils import set_seed

try:
    import zmq
except Exception:
    zmq = None

DEFAULT_ZMQ_ENDPOINT = os.environ.get("ALLEGRO_ZMQ_ENDPOINT", "tcp://127.0.0.1:5556")
ISAAC_JOINT_SEQUENCE = [
    "index_joint_0",
    "middle_joint_0",
    "ring_joint_0",
    "thumb_joint_0",
    "index_joint_1",
    "middle_joint_1",
    "ring_joint_1",
    "thumb_joint_1",
    "index_joint_2",
    "middle_joint_2",
    "ring_joint_2",
    "thumb_joint_2",
    "index_joint_3",
    "middle_joint_3",
    "ring_joint_3",
    "thumb_joint_3",
]
URDF_TO_ISAAC = {
    "joint_0.0": "index_joint_0",
    "joint_1.0": "index_joint_1",
    "joint_2.0": "index_joint_2",
    "joint_3.0": "index_joint_3",
    "joint_4.0": "middle_joint_0",
    "joint_5.0": "middle_joint_1",
    "joint_6.0": "middle_joint_2",
    "joint_7.0": "middle_joint_3",
    "joint_8.0": "ring_joint_0",
    "joint_9.0": "ring_joint_1",
    "joint_10.0": "ring_joint_2",
    "joint_11.0": "ring_joint_3",
    "joint_12.0": "thumb_joint_0",
    "joint_13.0": "thumb_joint_1",
    "joint_14.0": "thumb_joint_2",
    "joint_15.0": "thumb_joint_3",
}


class ModeController:
    def __init__(self) -> None:
        self.mode = "autonomous"
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        while True:
            try:
                ch = sys.stdin.read(1)
            except Exception:
                break
            if not ch:
                break
            self._queue.put(ch.lower())

    def poll(self) -> bool:
        changed = False
        while True:
            try:
                key = self._queue.get_nowait()
            except queue.Empty:
                break
            if key == "w" and self.mode != "external":
                self.mode = "external"
                changed = True
            elif key == "s" and self.mode != "autonomous":
                self.mode = "autonomous"
                changed = True
        return changed


class ZMQCommandListener:
    def __init__(self, endpoint: str = DEFAULT_ZMQ_ENDPOINT, side: str = "right") -> None:
        self.endpoint = endpoint
        self.side = side.lower()
        self.ready = zmq is not None and bool(self.endpoint)
        self._latest: tuple[list[str], list[float], float] | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if self.ready:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()
        else:
            print("[WARN]: ZMQ unavailable or endpoint missing; external takeover disabled")

    def _worker(self) -> None:
        assert zmq is not None
        ctx = zmq.Context.instance()
        socket = ctx.socket(zmq.SUB)
        socket.setsockopt_string(zmq.SUBSCRIBE, "")
        try:
            socket.connect(self.endpoint)
        except Exception as exc:
            print(f"[WARN]: Failed to connect to ZMQ endpoint {self.endpoint}: {exc}")
            self.ready = False
            socket.close(0)
            return

        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        while not self._stop.is_set():
            events = dict(poller.poll(100))
            if socket in events:
                try:
                    raw = socket.recv_string(zmq.NOBLOCK)
                except zmq.Again:
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if payload.get("side", "").lower() != self.side:
                    continue
                names = payload.get("name", [])
                values = payload.get("normalized_position", [])
                if len(names) != len(values):
                    continue
                with self._lock:
                    self._latest = (list(names), [float(v) for v in values], float(payload.get("timestamp", time.time())))
        socket.close(0)

    def get_latest(self) -> tuple[list[str], list[float], float] | None:
        with self._lock:
            if self._latest is None:
                return None
            names, values, ts = self._latest
            return (list(names), list(values), ts)

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)


class ExternalActionProcessor:
    def __init__(self, env: Wrapper) -> None:
        action_shape = cast(tuple, env.action_space.shape)
        self.action_dim = action_shape[0]
        self.num_envs = env.num_envs
        self.device = env.device
        self.template = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self.index_map = {name: idx for idx, name in enumerate(ISAAC_JOINT_SEQUENCE)}

    def convert(self, payload: tuple[list[str], list[float], float] | None) -> torch.Tensor | None:
        if payload is None:
            return None
        names, values, _ = payload
        result = self.template.clone()
        filled = False
        for name, value in zip(names, values):
            isaac_name = URDF_TO_ISAAC.get(name, name)
            idx = self.index_map.get(isaac_name)
            if idx is None or idx >= self.action_dim:
                continue
            norm_val = max(-1.0, min(1.0, float(value)))
            result[:, idx] = norm_val
            filled = True
        return result if filled else None


class SharedAutonomyController:
    def __init__(self, trainer: SharedAutonomySequentialTrainer, env: Wrapper) -> None:
        self.trainer = trainer
        self.listener = ZMQCommandListener()
        self.mode_controller = ModeController()
        self.processor = ExternalActionProcessor(env)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.last_action: torch.Tensor | None = None

    def start(self) -> None:
        self.thread.start()
        self._print_mode()

    def stop(self) -> None:
        self.stop_event.set()
        self.listener.stop()
        if self.thread.is_alive():
            self.thread.join(timeout=0.5)
        print()

    def _print_mode(self) -> None:
        label = "External (ZMQ)" if self.mode_controller.mode == "external" else "Autonomous"
        print(f"\r[MODE] {label:<20}", end="", flush=True)

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            if self.mode_controller.poll():
                self._print_mode()
                if self.mode_controller.mode == "autonomous":
                    self.trainer.end_human_intervene()

            if self.mode_controller.mode == "external" and self.listener.ready:
                payload = self.listener.get_latest()
                tensor = self.processor.convert(payload)
                if tensor is not None:
                    self.last_action = tensor
                    self.trainer.update_external_action(tensor)
                    self.trainer.activate_human_intervene()
                elif self.last_action is not None:
                    self.trainer.update_external_action(self.last_action)
                    self.trainer.activate_human_intervene()
                else:
                    self.trainer.end_human_intervene()
            else:
                self.trainer.end_human_intervene()

            time.sleep(0.05)

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
            logger.info(f"High-quality trajectory collected: {self._collected}")
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
    shared_controller = SharedAutonomyController(trainer, env)
    shared_controller.start()
    try:
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
    finally:
        shared_controller.stop()
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
