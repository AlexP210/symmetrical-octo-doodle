"""Train a visual PPO expert on one of this project's ManiSkill tasks.

`ppo_stages_fast.py` with the staged dataset collection removed and images in place of state. The
agent sees exactly what `custom_maniskill_tasks.make_env` renders for a given `--camera_view`, so
an expert trained here and the datasets, world models and planners in this project all look at the
task through the same camera, at the same resolution, under the same control mode. Recording a
dataset is still `ppo_stages_fast.py`'s job; this script's product is a policy checkpoint.

Observations are ManiSkill's `obs_mode="rgb"` flattened by `FlattenRGBDObservationWrapper` into
`{"rgb", "state"}` and fed to the NatureCNN of `examples/baselines/ppo/ppo_rgb.py`. Under an image
obs mode the tasks withhold their privileged state (object and goal poses), so `--include_state`
adds proprioception and the tcp pose only -- not a state-based shortcut around the camera. Pass
`--no-include_state` for pixels alone.

Three things follow from the observation being an image rather than a state vector:

* The rollout buffer is `num_steps * num_envs * resolution^2 * 3` bytes of uint8, so `--num_envs`
  is bounded by GPU memory rather than by throughput -- at the default 224x224 and a 50-step
  horizon, 256 envs is ~1.8 GiB of buffer alone. That is why the defaults here are ManiSkill's
  RGB-PPO shape (`num_envs=256`, `update_epochs=8`, `num_minibatches=8`) and not the much wider
  state-PPO one that `make_ppo_staged_maniskill_data.sh` uses.
* Frames are copied out of the observation before being stored. ManiSkill hands back its live
  camera buffer, which is overwritten in place on the next step, so a rollout that stored the
  tensor would end up holding `num_steps` aliases of the final frame -- see `clone_observations`.
* The whole rollout is a `TensorDict` whose `obs` entry is itself a `TensorDict` of `rgb`/`state`,
  which is what lets the stacking, flattening and minibatch indexing of `ppo_fast.py` carry a dict
  observation unchanged.

Terminations are ignored on both envs: the supported ids are this project's `-v1.1` variants,
which never terminate early, so every episode runs the full horizon and holding the goal longer is
worth strictly more return. `num_eval_steps` is forced to that horizon, so one eval rollout is
exactly `num_eval_envs` complete episodes and `success_once` means what it says.

Training stops as soon as an eval reaches `--training_done_success_rate`. The best-scoring policy
so far is kept at `best_ckpt.pt` alongside `final_ckpt.pt` -- those two are the whole output unless
`--save_ckpt_freq` asks for intermediate ones -- and `training_summary.json` records which is
which, along with the camera the expert was trained through.
"""

import os

from mani_skill.utils import gym_utils
from mani_skill.utils.io_utils import dump_json
from mani_skill.utils.wrappers.flatten import (
    FlattenActionSpaceWrapper,
    FlattenRGBDObservationWrapper,
)
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

# also registers the -v1.1 task ids, so --env_id can name them
from custom_maniskill_tasks import (
    CAMERA_VIEWS,
    DEFAULT_CAMERA_RESOLUTION,
    FULL_HORIZON_TASKS,
    backend_kwargs,
    clone_observations,
    make_env,
)

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

import math
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import tensordict
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import tyro
from torch.utils.tensorboard import SummaryWriter
import wandb
from tensordict import from_module
from tensordict.nn import CudaGraphModule
from torch.distributions.normal import Normal

SUPPORTED_ENV_IDS = tuple(FULL_HORIZON_TASKS)
"""Only the `-v1.1` ids. The stock `-v1` tasks terminate on success, which would cut episodes
short and break the full-horizon reward convention this script's `ignore_terminations=True` and
horizon-length eval rollouts assume."""


@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    device: str = "cuda:0"
    """Name of the device to use. Also pins the physx sim and sapien renderer to this GPU, so
    give it an explicit index (`cuda:0`, not `cuda`) on multi-GPU hosts."""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    wandb_group: str = "PPO"
    """the group of the run for wandb"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `{save_path}/{run_name}` folder"""
    save_ckpt_freq: int = 0
    """how often, in iterations, to also keep a numbered intermediate checkpoint; 0 keeps only
    best_ckpt.pt and final_ckpt.pt. Off by default because the CNN makes a checkpoint far heavier
    than a state agent's -- 38 MiB at the default 224x224 -- so one per eval over a full run is
    tens of GiB of files nothing reads."""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start training from"""
    save_path: str = "runs"
    """the root folder to save training runs, checkpoints, and video data to"""

    # Environment specific arguments
    env_id: str = "PushCube-v1.1"
    """the id of the environment, one of SUPPORTED_ENV_IDS"""
    camera_view: str = "default"
    """which camera the observations come from -- "default" (the task's own), "focused" (that
    camera re-posed onto the tabletop through a narrow fov) or "wrist" (a hand-mounted fisheye).
    See custom_maniskill_tasks.cameras; a policy is only ever an expert in the view it was
    trained on."""
    camera_resolution: int = DEFAULT_CAMERA_RESOLUTION
    """square resolution to render observations at. The default matches every dataset in this
    project; lowering it is the cheapest way to buy back rollout memory and update time."""
    wrist_only: bool = True
    """under `--camera_view wrist`, drop the task's own camera instead of feeding both to the CNN"""
    include_state: bool = True
    """whether to concatenate the (non-privileged) state vector onto the image features"""
    num_envs: int = 256
    """the number of parallel environments"""
    num_eval_envs: int = 16
    """the number of parallel evaluation environments"""
    num_steps: int = 0
    """the number of steps to run in each environment per policy rollout (0 means the task horizon)"""
    num_eval_steps: int = 0
    """the number of steps to run in each evaluation environment during evaluation (forced to the task horizon)"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 1
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    eval_freq: int = 25
    """evaluation frequency in terms of iterations"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""
    control_mode: Optional[str] = "pd_ee_delta_pos"
    """the control mode to use for the environment"""

    # Expert specific arguments
    training_done_success_rate: float = 0.95
    """mean eval success rate at which the policy is considered an expert and training stops"""
    success_metric: str = "success_once"
    """the eval episode metric compared against `training_done_success_rate`"""

    # Algorithm specific arguments
    total_timesteps: int = 50000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.8
    """the discount factor gamma"""
    gae_lambda: float = 0.9
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8
    """the number of mini-batches"""
    update_epochs: int = 8
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.1
    """the target KL divergence threshold"""
    reward_scale: float = 1.0
    """Scale the reward by this factor"""
    finite_horizon_gae: bool = False

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    # Torch optimizations
    compile: bool = False
    """whether to use torch.compile."""
    cudagraphs: bool = False
    """whether to use cudagraphs on top of compile. Measured not to work with ManiSkill's GPU
    sim on this stack: sapien allocates inside the stream tensordict captures the policy on and
    dies with "operation not permitted when stream is capturing". Left exposed because it costs
    nothing to, but `--compile --no-cudagraphs` is the combination that runs."""


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class NatureCNN(nn.Module):
    """The feature extractor of ManiSkill's `ppo_rgb.py`, kept identical on purpose.

    One CNN over the camera image (all cameras concatenated on the channel axis, which is what
    `FlattenRGBDObservationWrapper` hands over) and, when there is one, a linear embedding of the
    state vector; the two are concatenated into the latent the actor and critic share.
    """

    def __init__(self, sample_obs, feature_size: int = 256, device=None):
        super().__init__()
        extractors = {}
        self.out_features = 0

        in_channels = sample_obs["rgb"].shape[-1]
        cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0, device=device),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0, device=device),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, device=device),
            nn.ReLU(),
            nn.Flatten(),
        )
        # the flattened width depends on --camera_resolution, so read it off a real observation
        # rather than hardcoding it
        with torch.no_grad():
            n_flatten = cnn(sample_obs["rgb"][:1].float().permute(0, 3, 1, 2)).shape[1]
        extractors["rgb"] = nn.Sequential(
            cnn, nn.Linear(n_flatten, feature_size, device=device), nn.ReLU()
        )
        self.out_features += feature_size

        if "state" in sample_obs.keys():
            extractors["state"] = nn.Linear(sample_obs["state"].shape[-1], 256, device=device)
            self.out_features += 256

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            obs = observations[key]
            if key == "rgb":
                # NHWC uint8 as ManiSkill renders it -> NCHW float in [0, 1]
                obs = obs.float().permute(0, 3, 1, 2) / 255.0
            encoded_tensor_list.append(extractor(obs))
        return torch.cat(encoded_tensor_list, dim=1)


class Agent(nn.Module):
    def __init__(self, sample_obs, n_act, device=None):
        super().__init__()
        self.feature_net = NatureCNN(sample_obs, device=device)
        latent_size = self.feature_net.out_features
        self.critic = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512, device=device)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 1, device=device)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512, device=device)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, n_act, device=device), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, n_act, device=device) * -0.5)

    def get_value(self, obs):
        return self.critic(self.feature_net(obs))

    def get_action(self, obs):
        """The deterministic action, which is what an eval rollout and an expert rollout use."""
        return self.actor_mean(self.feature_net(obs))

    def get_action_and_value(self, obs, action=None):
        x = self.feature_net(obs)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        # validate_args=False: the check is `torch._is_all_true(scale > 0)`, read on the host and
        # so a GPU sync on every one of the num_steps policy calls per rollout. The scale is an
        # exp() of a parameter and positive by construction, so there is nothing to catch.
        probs = Normal(action_mean, action_std, validate_args=False)
        if action is None:
            action = action_mean + action_std * torch.randn_like(action_mean)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


class Logger:
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb
    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)
    def close(self):
        self.writer.close()


def as_obs_td(obs, num):
    """A private copy of an observation dict, as a `TensorDict` the rollout can stack.

    The copy is the point: ManiSkill returns its live camera buffer and overwrites it in place on
    the next step, so a rollout that kept the tensor would hold `num_steps` references to the last
    frame rendered (see `custom_maniskill_tasks.clone_observations`). State leaves are freshly
    allocated each step and copying them is only for uniformity.
    """
    return tensordict.TensorDict(clone_observations(obs), batch_size=(num,))


def gae(next_obs, next_done, container, final_values):
    # bootstrap value if not done
    next_value = get_value(next_obs).reshape(-1)
    lastgaelam = 0
    nextnonterminals = (~container["dones"]).float().unbind(0)
    vals = container["vals"]
    vals_unbind = vals.unbind(0)
    rewards = container["rewards"].unbind(0)

    advantages = []
    nextnonterminal = (~next_done).float()
    nextvalues = next_value
    for t in range(args.num_steps - 1, -1, -1):
        cur_val = vals_unbind[t]
        # real_next_values = nextvalues * nextnonterminal
        real_next_values = nextnonterminal * nextvalues + final_values[t] # t instead of t+1
        delta = rewards[t] + args.gamma * real_next_values - cur_val
        advantages.append(delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam)
        lastgaelam = advantages[-1]

        nextnonterminal = nextnonterminals[t]
        nextvalues = cur_val

    advantages = container["advantages"] = torch.stack(list(reversed(advantages)))
    container["returns"] = advantages + vals
    return container


def rollout(obs, done):
    ts = []
    final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
    for step in range(args.num_steps):
        # ALGO LOGIC: action logic
        action, logprob, _, value = policy(obs=obs)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, next_done, infos = step_func(action)

        if "final_info" in infos:
            final_info = infos["final_info"]
            done_mask = infos["_final_info"]
            for k, v in final_info["episode"].items():
                logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)
            with torch.no_grad():
                # already a copy of the pre-reset observation (ManiSkillVectorEnv clones it), so
                # unlike the rollout's own obs this one needs no clone of its own
                final_obs = tensordict.TensorDict(infos["final_observation"], batch_size=(args.num_envs,))
                final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = agent.get_value(final_obs[done_mask]).view(-1)

        ts.append(
            tensordict.TensorDict._new_unsafe(
                obs=obs,
                # cleanrl ppo examples associate the done with the previous obs (not the done resulting from action)
                dones=done,
                vals=value.flatten(),
                actions=action,
                logprobs=logprob,
                rewards=reward,
                batch_size=(args.num_envs,),
            )
        )
        # NOTE (stao): change here for gpu env
        obs = next_obs = next_obs
        done = next_done
    # NOTE (stao): need to do .to(device) i think? otherwise container.device is None, not sure if this affects anything
    container = torch.stack(ts, 0).to(device)
    return next_obs, done, container, final_values


def update(obs, actions, logprobs, advantages, returns, vals):
    optimizer.zero_grad()
    _, newlogprob, entropy, newvalue = agent.get_action_and_value(obs, actions)
    logratio = newlogprob - logprobs
    ratio = logratio.exp()

    with torch.no_grad():
        # calculate approx_kl http://joschu.net/blog/kl-approx.html
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()

    if args.norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Policy loss
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    # Value loss
    newvalue = newvalue.view(-1)
    if args.clip_vloss:
        v_loss_unclipped = (newvalue - returns) ** 2
        v_clipped = vals + torch.clamp(
            newvalue - vals,
            -args.clip_coef,
            args.clip_coef,
        )
        v_loss_clipped = (v_clipped - returns) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()
    else:
        v_loss = 0.5 * ((newvalue - returns) ** 2).mean()

    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

    loss.backward()
    gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
    optimizer.step()

    return approx_kl, v_loss.detach(), pg_loss.detach(), entropy_loss.detach(), old_approx_kl, clipfrac, gn


update = tensordict.nn.TensorDictModule(
    update,
    in_keys=["obs", "actions", "logprobs", "advantages", "returns", "vals"],
    out_keys=["approx_kl", "v_loss", "pg_loss", "entropy_loss", "old_approx_kl", "clipfrac", "gn"],
)

if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.env_id in SUPPORTED_ENV_IDS, f"env_id must be one of {SUPPORTED_ENV_IDS}"
    assert args.camera_view in CAMERA_VIEWS, f"camera_view must be one of {CAMERA_VIEWS}"
    assert 0.0 <= args.training_done_success_rate <= 1.0, "training_done_success_rate must be in [0, 1]"

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.camera_view}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(args.device if torch.cuda.is_available() and args.cuda else "cpu")

    ####### Environment setup #######
    # `backend_kwargs` pins the sim and the renderer to one explicit device index: an index-less
    # `physx_cuda` is resolved by sapien and need not agree with torch's `cuda`, which mixes tensors
    # across GPUs and crashes.
    if device.type == "cuda":
        gpu_idx = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_device(gpu_idx)
        backends = backend_kwargs(f"physx_cuda:{gpu_idx}")
    else:
        backends = backend_kwargs("physx_cpu", render_backend="sapien_cpu")
    env_kwargs = dict(obs_mode="rgb", render_mode="rgb_array", control_mode=args.control_mode, **backends)
    # `ignore_terminations=False`: the vector env below ignores terminations for reset purposes,
    # and the `-v1.1` tasks report none in the first place, so the wrapper would be a no-op.
    # The observation flattening goes in `obs_wrappers` so that it sits inside make_env's stack,
    # where the agent's own observation adapters would also sit.
    flatten_obs = lambda env: FlattenRGBDObservationWrapper(
        env, rgb=True, depth=False, state=args.include_state
    )
    make_kwargs = dict(
        camera_view=args.camera_view,
        camera_resolution=args.camera_resolution,
        wrist_only=args.wrist_only,
        ignore_terminations=False,
        obs_wrappers=[flatten_obs],
        **env_kwargs,
    )
    envs = make_env(args.env_id, num_envs=args.num_envs, reconfiguration_freq=args.reconfiguration_freq, **make_kwargs)
    eval_envs = make_env(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq, human_render_camera_configs=dict(shader_pack="default"), **make_kwargs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)

    # an eval rollout must be exactly one full episode per env, so `success_once` is measured over
    # complete episodes; a training rollout may be shorter, since GAE bootstraps across the seam
    max_episode_steps = gym_utils.find_max_episode_steps_value(envs)
    args.num_eval_steps = max_episode_steps
    args.num_steps = args.num_steps or max_episode_steps
    print(f"{args.env_id} horizon is {max_episode_steps}, using num_steps={args.num_steps}")
    batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = batch_size // args.num_minibatches
    args.batch_size = args.num_minibatches * args.minibatch_size
    args.num_iterations = args.total_timesteps // args.batch_size
    assert args.num_iterations > 0, (
        f"total_timesteps={args.total_timesteps} buys no iterations at all: one iteration costs "
        f"{args.batch_size} env steps"
    )

    if args.capture_video:
        eval_output_dir = f"{args.save_path}/{run_name}/videos"
        print(f"Saving eval videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x : (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(envs, output_dir=f"{args.save_path}/{run_name}/train_videos", save_trajectory=False, save_video_trigger=save_video_trigger, max_steps_per_video=args.num_steps, video_fps=30)
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=False, save_video=True, max_steps_per_video=args.num_eval_steps, video_fps=30)
    # no automatic reset on success: episodes always run the full horizon, so holding the goal
    # early is what earns the extra dense reward
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=True, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=True, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    device = envs.device

    print("Running training")
    if args.track:
        import wandb
        config = vars(args)
        config["env_cfg"] = dict(**env_kwargs, num_envs=args.num_envs, env_id=args.env_id, camera_view=args.camera_view, camera_resolution=args.camera_resolution, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=False)
        config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, camera_view=args.camera_view, camera_resolution=args.camera_resolution, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=False)
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=False,
            config=config,
            name=run_name,
            save_code=True,
            group=args.wandb_group,
            tags=["ppo", "walltime_efficient", "visual_expert", args.camera_view, f"GPU:{torch.cuda.get_device_name()}"]
        )
    writer = SummaryWriter(f"{args.save_path}/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    logger = Logger(log_wandb=args.track, tensorboard=writer)

    n_act = math.prod(envs.single_action_space.shape)

    # Register step as a special op not to graph break
    # @torch.library.custom_op("mylib::step", mutates_args=())
    def step_func(action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # NOTE (stao): change here for gpu env
        next_obs, reward, terminations, truncations, info = envs.step(action)
        next_done = torch.logical_or(terminations, truncations)
        return as_obs_td(next_obs, args.num_envs), reward, next_done, info

    def evaluate():
        """One full-horizon rollout of every eval env under the mean action, as episode metrics."""
        eval_obs, _ = eval_envs.reset()
        eval_metrics = defaultdict(list)
        for _ in range(args.num_eval_steps):
            with torch.no_grad():
                eval_action = agent.get_action(as_obs_td(eval_obs, args.num_eval_envs))
                eval_obs, _, _, _, eval_infos = eval_envs.step(eval_action)
            if "final_info" in eval_infos:
                mask = eval_infos["_final_info"]
                for k, v in eval_infos["final_info"]["episode"].items():
                    eval_metrics[k].append(v[mask].float())
        return {k: torch.cat(v).mean().item() for k, v in eval_metrics.items()}

    ####### Agent #######
    # the CNN's flattened width is read off a real observation, so the agent is built from one
    sample_obs = as_obs_td(eval_envs.reset(seed=args.seed)[0], args.num_eval_envs)
    agent = Agent(sample_obs, n_act, device=device)
    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint))
    # Make a version of agent with detached params
    agent_inference = Agent(sample_obs, n_act, device=device)
    agent_inference_p = from_module(agent).data
    agent_inference_p.to_module(agent_inference)
    del sample_obs

    ####### Optimizer #######
    optimizer = optim.Adam(
        agent.parameters(),
        lr=torch.tensor(args.learning_rate, device=device),
        eps=1e-5,
        capturable=args.cudagraphs and not args.compile,
    )

    ####### Executables #######
    # Define networks: wrapping the policy in a TensorDictModule allows us to use CudaGraphModule
    policy = agent_inference.get_action_and_value
    get_value = agent_inference.get_value

    # Compile policy
    if args.compile:
        policy = torch.compile(policy)
        gae = torch.compile(gae, fullgraph=True)
        update = torch.compile(update)

    if args.cudagraphs:
        policy = CudaGraphModule(policy)
        gae = CudaGraphModule(gae)
        update = CudaGraphModule(update)

    global_step = 0
    start_time = time.time()
    container_local = None
    cumulative_times = defaultdict(float)
    summary = dict(
        env_id=args.env_id,
        camera_view=args.camera_view,
        camera_resolution=args.camera_resolution,
        include_state=args.include_state,
        control_mode=args.control_mode,
        horizon=max_episode_steps,
        num_eval_envs=args.num_eval_envs,
    )
    summary_path = f"{args.save_path}/{run_name}/training_summary.json"

    iteration = 0
    success_rate = 0.0
    best_success_rate = -1.0
    best_iteration = 0
    threshold_reached = False

    next_obs = as_obs_td(envs.reset(seed=args.seed)[0], args.num_envs)
    next_done = torch.zeros(args.num_envs, device=device, dtype=torch.bool)
    pbar = tqdm.tqdm(range(1, args.num_iterations + 1))

    for iteration in pbar:
        agent.eval()
        if (iteration - 1) % args.eval_freq == 0:
            stime = time.perf_counter()
            eval_metrics_mean = evaluate()
            for k, v in eval_metrics_mean.items():
                logger.add_scalar(f"eval/{k}", v, global_step)
            success_rate = eval_metrics_mean[args.success_metric]
            pbar.set_description(f"{args.success_metric}: {success_rate:.3f} (best {max(best_success_rate, success_rate):.3f})")
            eval_time = time.perf_counter() - stime
            cumulative_times["eval_time"] += eval_time
            logger.add_scalar("time/eval_time", eval_time, global_step)
            # keep the best policy seen: the last one is not reliably the strongest, and this
            # script's product is the expert rather than the run
            if success_rate > best_success_rate:
                best_success_rate, best_iteration = success_rate, iteration
                if args.save_model:
                    torch.save(agent.state_dict(), f"{args.save_path}/{run_name}/best_ckpt.pt")
            if success_rate >= args.training_done_success_rate:
                threshold_reached = True
                print(f"reached {args.success_metric}={success_rate:.3f} >= {args.training_done_success_rate} at iteration {iteration}")
                break
        if args.save_model and args.save_ckpt_freq and (iteration - 1) % args.save_ckpt_freq == 0:
            model_path = f"{args.save_path}/{run_name}/ckpt_{iteration}.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"].copy_(lrnow)

        torch.compiler.cudagraph_mark_step_begin()
        rollout_time = time.perf_counter()
        next_obs, next_done, container, final_values = rollout(next_obs, next_done)
        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time
        global_step += container.numel()

        update_time = time.perf_counter()
        container = gae(next_obs, next_done, container, final_values)
        container_flat = container.view(-1)

        # Optimizing the policy and value network
        clipfracs = []
        for epoch in range(args.update_epochs):
            b_inds = torch.randperm(container_flat.shape[0], device=device).split(args.minibatch_size)
            for b in b_inds:
                container_local = container_flat[b]

                out = update(container_local, tensordict_out=tensordict.TensorDict())
                clipfracs.append(out["clipfrac"])
                if args.target_kl is not None and out["approx_kl"] > args.target_kl:
                    break
            else:
                continue
            break
        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time

        logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        logger.add_scalar("losses/value_loss", out["v_loss"].item(), global_step)
        logger.add_scalar("losses/policy_loss", out["pg_loss"].item(), global_step)
        logger.add_scalar("losses/entropy", out["entropy_loss"].item(), global_step)
        logger.add_scalar("losses/old_approx_kl", out["old_approx_kl"].item(), global_step)
        logger.add_scalar("losses/approx_kl", out["approx_kl"].item(), global_step)
        logger.add_scalar("losses/clipfrac", torch.stack(clipfracs).mean().cpu().item(), global_step)
        logger.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        logger.add_scalar("time/step", global_step, global_step)
        logger.add_scalar("time/update_time", update_time, global_step)
        logger.add_scalar("time/rollout_time", rollout_time, global_step)
        logger.add_scalar("time/rollout_fps", args.num_envs * args.num_steps / rollout_time, global_step)
        for k, v in cumulative_times.items():
            logger.add_scalar(f"time/total_{k}", v, global_step)
        logger.add_scalar("time/total_rollout+update_time", cumulative_times["rollout_time"] + cumulative_times["update_time"], global_step)

    if args.save_model:
        model_path = f"{args.save_path}/{run_name}/final_ckpt.pt"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    summary.update(
        iterations=iteration,
        global_step=global_step,
        threshold_reached=threshold_reached,
        training_done_success_rate=args.training_done_success_rate,
        success_metric=args.success_metric,
        final_success_rate=success_rate,
        best_success_rate=best_success_rate,
        best_iteration=best_iteration,
        expert_checkpoint="best_ckpt.pt" if args.save_model else None,
    )
    dump_json(summary_path, summary, indent=2)
    print(f"training summary written to {summary_path}")

    if not threshold_reached:
        print(f"WARNING: training finished at {args.success_metric}={success_rate:.3f} (best "
              f"{best_success_rate:.3f}), below the requested training_done_success_rate="
              f"{args.training_done_success_rate}. This policy is not an expert.")

    logger.close()
    envs.close()
    eval_envs.close()
