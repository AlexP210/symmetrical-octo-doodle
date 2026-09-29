"""Collect a staged ManiSkill dataset: random episodes -> mid-training episodes -> expert episodes.

Structurally `ppo_fast.py` with a collection schedule wrapped around it. Two differences in the
environment itself:

* Terminations are always ignored (`ignore_terminations=True`) on both the training and the eval
  env, so an episode is never cut short by reaching the goal. Every episode runs the full task
  horizon, and because the dense reward keeps accruing while the goal is held, reaching the goal
  sooner is worth strictly more return -- that is the intended incentive.
* `num_steps` / `num_eval_steps` are forced to the task horizon, so one eval rollout is exactly
  `num_eval_envs` complete episodes.

Both envs are built by `custom_maniskill_tasks.make_env`, which is also what the agents build their
online and planning envs with, so a dataset and the policy trained on it cannot drift apart on the
camera view, the control mode or the sim/render device. The `-v1.1` task ids carry the
no-early-termination convention above in the *task*, so it holds for anything that builds them --
including a plain `gym.make`, and including the replay of this dataset. The `terminated` field of a
dataset recorded under a `-v1.1` id is therefore all False; `success` is recorded separately, per
step and per episode, and is unaffected.

Recording happens on the eval env and is toggled per eval, so episode budgets are quantised to
`num_eval_envs`; a budget that is not a multiple of it rounds up. All three stages append to one
`{save_path}/{run_name}/videos/trajectory.h5`, in order (random, then mid-training, then expert);
`collection_summary.json` records each stage's `episode_range` as a [start, end) pair of `traj_N`
indices so the stages can be sliced back apart.

The mid-training budget is spread over the *planned* run length (`total_timesteps`), since the
number of iterations needed to hit `training_done_success_rate` is not knowable up front. If
training hits the threshold early, whatever is left of the budget is recorded at that point.
"""

import os

from mani_skill.utils import gym_utils
from mani_skill.utils.io_utils import dump_json
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

# also registers the -v1.1 task ids, so --env_id can name them
from custom_maniskill_tasks import FULL_HORIZON_TASKS, backend_kwargs, make_env

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

SUPPORTED_ENV_IDS = tuple(FULL_HORIZON_TASKS) + (
    # stock ManiSkill ids, kept so earlier runs remain reproducible. On these the
    # no-early-termination convention comes only from the ManiSkillVectorEnv flag below, so their
    # recorded `terminated` field still carries the raw success flag.
    "PushCube-v1",
    "PegInsertionSide-v1",
    "PlaceSphere-v1",
    "OpenCabinetDrawer-v1",
    "LiftPegUpright-v1",
    "PushT-v1",
)


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
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_trajectory: bool = True
    """whether to save trajectory data into the `videos` folder"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start training from"""
    save_path: str = "runs"
    """the root folder to save training runs, checkpoints, and trajectory/video data to"""

    # Environment specific arguments
    env_id: str = "PushCube-v1.1"
    """the id of the environment, one of SUPPORTED_ENV_IDS"""
    env_vectorization: str = "gpu"
    """the type of environment vectorization to use"""
    num_envs: int = 512
    """the number of parallel environments"""
    num_eval_envs: int = 16
    """the number of parallel evaluation environments, and the granularity of every episode budget"""
    num_steps: int = 0
    """the number of steps to run in each environment per policy rollout (forced to the task horizon)"""
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
    control_mode: Optional[str] = "pd_joint_delta_pos"
    """the control mode to use for the environment"""

    # Collection specific arguments
    number_of_random_episodes: int = 0
    """number of episodes to record under a uniformly random policy before training begins"""
    number_of_training_episodes: int = 0
    """number of episodes to record between the start of training and the eval at which the mean
    success rate first reaches `training_done_success_rate`"""
    training_done_success_rate: float = 0.9
    """mean eval success rate at which training is considered done"""
    number_of_expert_episodes: int = 0
    """number of episodes to record with the final policy once it reaches `training_done_success_rate`"""
    success_metric: str = "success_once"
    """the eval episode metric compared against `training_done_success_rate`"""

    # Algorithm specific arguments
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.8
    """the discount factor gamma"""
    gae_lambda: float = 0.9
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 4
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
    """whether to use cudagraphs on top of compile."""

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, n_obs, n_act, device=None):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(n_obs, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1, device=device)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(n_obs, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, n_act, device=device), std=0.01*np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, n_act, device=device))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, obs, action=None):
        action_mean = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = action_mean + action_std * torch.randn_like(action_mean)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs)

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


class RandomPolicy:
    """Uniform random policy over a bounded continuous action space, sampled on `device`."""
    def __init__(self, action_space: gym.spaces.Box, num_envs: int, device=None):
        low = torch.as_tensor(action_space.low, dtype=torch.float32, device=device)
        high = torch.as_tensor(action_space.high, dtype=torch.float32, device=device)
        assert torch.isfinite(low).all() and torch.isfinite(high).all(), (
            f"unbounded action space {action_space}, cannot sample uniformly"
        )
        self.low = low
        self.span = high - low
        self.shape = (num_envs, *action_space.shape)
        self.device = device

    def __call__(self, obs=None):
        return self.low + self.span * torch.rand(self.shape, device=self.device)


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
                final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = agent.get_value(infos["final_observation"][done_mask]).view(-1)

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
    assert 0.0 <= args.training_done_success_rate <= 1.0, "training_done_success_rate must be in [0, 1]"

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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
    env_kwargs = dict(obs_mode="state", render_mode="rgb_array", control_mode=args.control_mode, **backends)
    # `ignore_terminations=False`: the vector env below already ignores terminations for reset
    # purposes, and leaving the raw flag intact underneath RecordEpisode keeps the `terminated`
    # field of a stock `-v1` recording as it has always been. Under a `-v1.1` id the task itself
    # reports no termination, so there the field is all False either way.
    # camera_resolution=None: obs_mode is "state", so no observation camera is read at all, and
    # videos come from the human render camera, which sensor configs do not touch.
    make_kwargs = dict(camera_view="default", camera_resolution=None, ignore_terminations=False, **env_kwargs)
    envs = make_env(args.env_id, num_envs=args.num_envs, reconfiguration_freq=args.reconfiguration_freq, **make_kwargs)
    eval_envs = make_env(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq, human_render_camera_configs=dict(shader_pack="default"), **make_kwargs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)

    # one rollout must be exactly one full episode per env, so every recorded episode is complete
    # and each eval yields exactly `num_eval_envs` episodes
    max_episode_steps = gym_utils.find_max_episode_steps_value(envs)
    args.num_steps = args.num_eval_steps = max_episode_steps
    print(f"{args.env_id} horizon is {max_episode_steps}, using it for num_steps and num_eval_steps")
    batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = batch_size // args.num_minibatches
    args.batch_size = args.num_minibatches * args.minibatch_size
    args.num_iterations = args.total_timesteps // args.batch_size

    record_env = None
    eval_output_dir = f"{args.save_path}/{run_name}/videos"
    if args.capture_video or args.save_trajectory:
        print(f"Saving eval trajectories/videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x : (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(envs, output_dir=f"{args.save_path}/{run_name}/train_videos", save_trajectory=False, save_video_trigger=save_video_trigger, max_steps_per_video=args.num_steps, video_fps=30)
        eval_envs = record_env = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=args.save_trajectory, save_video=args.capture_video, trajectory_name="trajectory", max_steps_per_video=args.num_eval_steps, video_fps=30)
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
        config["env_cfg"] = dict(**env_kwargs, num_envs=args.num_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=False)
        config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=False)
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=False,
            config=config,
            name=run_name,
            save_code=True,
            group=args.wandb_group,
            tags=["ppo", "walltime_efficient", "staged_collection", f"GPU:{torch.cuda.get_device_name()}"]
        )
    writer = SummaryWriter(f"{args.save_path}/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    logger = Logger(log_wandb=args.track, tensorboard=writer)

    n_act = math.prod(envs.single_action_space.shape)
    n_obs = math.prod(envs.single_observation_space.shape)

    # Register step as a special op not to graph break
    # @torch.library.custom_op("mylib::step", mutates_args=())
    def step_func(action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # NOTE (stao): change here for gpu env
        next_obs, reward, terminations, truncations, info = envs.step(action)
        next_done = torch.logical_or(terminations, truncations)
        return next_obs, reward, next_done, info

    ####### Collection helpers #######
    def episodes_recorded():
        return 0 if record_env is None else record_env._episode_id + 1

    def rollouts_for(num_episodes):
        return math.ceil(num_episodes / args.num_eval_envs)

    def stage_range(start):
        """Episode count and [start, end) traj_N index range this stage wrote to the shared h5."""
        end = episodes_recorded()
        return dict(episodes=end - start, episode_range=[start, end])

    def set_recording(enabled):
        """Gate trajectory/video writes on the eval env without disturbing eval metrics."""
        if record_env is None:
            return
        record_env.save_trajectory = enabled and args.save_trajectory
        record_env._save_video = enabled and args.capture_video
        if not enabled:
            record_env._trajectory_buffer = None

    def evaluate(action_fn):
        """One full-horizon rollout of every eval env, returning mean episode metrics."""
        eval_obs, _ = eval_envs.reset()
        eval_metrics = defaultdict(list)
        for _ in range(args.num_eval_steps):
            with torch.no_grad():
                eval_obs, _, _, _, eval_infos = eval_envs.step(action_fn(eval_obs))
            if "final_info" in eval_infos:
                mask = eval_infos["_final_info"]
                for k, v in eval_infos["final_info"]["episode"].items():
                    eval_metrics[k].append(v[mask].float())
        return {k: torch.cat(v).mean().item() for k, v in eval_metrics.items()}

    def collect_stage(name, num_episodes, action_fn, log_prefix=None):
        """Record at least `num_episodes` episodes, rounded up to whole eval rollouts."""
        stage_metrics = defaultdict(list)
        num_rollouts = rollouts_for(num_episodes)
        set_recording(True)
        for i in range(num_rollouts):
            metrics = evaluate(action_fn)
            for k, v in metrics.items():
                stage_metrics[k].append(v)
            print(f"{name} rollout {i + 1}/{num_rollouts} ({episodes_recorded()} episodes): "
                  + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
        set_recording(False)
        mean = {k: float(np.mean(v)) for k, v in stage_metrics.items()}
        if log_prefix is not None:
            for k, v in mean.items():
                logger.add_scalar(f"{log_prefix}/{k}", v, global_step)
        return mean

    ####### Agent #######
    agent = Agent(n_obs, n_act, device=device)
    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint))
    # Make a version of agent with detached params
    agent_inference = Agent(n_obs, n_act, device=device)
    agent_inference_p = from_module(agent).data
    agent_inference_p.to_module(agent_inference)

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
    summary = dict(env_id=args.env_id, horizon=max_episode_steps, num_eval_envs=args.num_eval_envs, stages={})
    summary_path = f"{args.save_path}/{run_name}/collection_summary.json"

    ####### Stage 1: random policy #######
    set_recording(False)
    if args.number_of_random_episodes > 0:
        random_policy = RandomPolicy(eval_envs.single_action_space, args.num_eval_envs, device=device)
        print(f"Recording {args.number_of_random_episodes} episode(s) with a random action policy before training")
        stage_start = episodes_recorded()
        mean = collect_stage("random", args.number_of_random_episodes, random_policy, log_prefix="random")
        summary["stages"]["random"] = dict(**stage_range(stage_start), **mean)
        dump_json(summary_path, summary, indent=2)

    ####### Stage 2: training #######
    train_stage_start = episodes_recorded()
    train_rollouts_needed = rollouts_for(args.number_of_training_episodes) if args.number_of_training_episodes > 0 else 0
    planned_evals = max(1, args.num_iterations // args.eval_freq)
    record_every = max(1, planned_evals // train_rollouts_needed) if train_rollouts_needed > 0 else 0
    train_rollouts_done = 0
    eval_idx = 0
    iteration = 0
    train_metrics = defaultdict(list)
    success_rate = 0.0
    threshold_reached = False
    if train_rollouts_needed > 0:
        print(f"Recording {args.number_of_training_episodes} training episode(s) over {train_rollouts_needed} "
              f"eval rollout(s), one every {record_every} of ~{planned_evals} planned evals")

    next_obs = envs.reset()[0]
    next_done = torch.zeros(args.num_envs, device=device, dtype=torch.bool)
    pbar = tqdm.tqdm(range(1, args.num_iterations + 1))

    for iteration in pbar:
        agent.eval()
        if (iteration - 1) % args.eval_freq == 0:
            stime = time.perf_counter()
            record_this_eval = train_rollouts_done < train_rollouts_needed and eval_idx % record_every == 0
            set_recording(record_this_eval)
            eval_metrics_mean = evaluate(agent.actor_mean)
            set_recording(False)
            if record_this_eval:
                train_rollouts_done += 1
                for k, v in eval_metrics_mean.items():
                    train_metrics[k].append(v)
            eval_idx += 1
            for k, v in eval_metrics_mean.items():
                logger.add_scalar(f"eval/{k}", v, global_step)
            success_rate = eval_metrics_mean.get(args.success_metric, 0.0)
            pbar.set_description(f"{args.success_metric}: {success_rate:.3f}, recorded: {episodes_recorded()}")
            eval_time = time.perf_counter() - stime
            cumulative_times["eval_time"] += eval_time
            logger.add_scalar("time/eval_time", eval_time, global_step)
            if success_rate >= args.training_done_success_rate:
                threshold_reached = True
                print(f"reached {args.success_metric}={success_rate:.3f} >= {args.training_done_success_rate} at iteration {iteration}")
                break
        if args.save_model and (iteration - 1) % args.eval_freq == 0:
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

    # training ended before the budget was spent -- record what is left at the policy we stopped at
    if train_rollouts_done < train_rollouts_needed:
        remaining = (train_rollouts_needed - train_rollouts_done) * args.num_eval_envs
        print(f"training ended with {remaining} training episode(s) of budget unspent, recording them now")
        mean = collect_stage("training (top-up)", remaining, agent.actor_mean)
        for k, v in mean.items():
            train_metrics[k].append(v)
    summary["stages"]["training"] = dict(
        **stage_range(train_stage_start),
        iterations=iteration,
        threshold_reached=threshold_reached,
        final_success_rate=success_rate,
        **{k: float(np.mean(v)) for k, v in train_metrics.items()},
    )
    dump_json(summary_path, summary, indent=2)

    if not threshold_reached:
        print(f"WARNING: training finished at {args.success_metric}={success_rate:.3f}, below the requested "
              f"training_done_success_rate={args.training_done_success_rate}. The expert stage below is "
              f"recorded with this policy and does NOT meet the requested success rate.")

    ####### Stage 3: expert policy #######
    if args.number_of_expert_episodes > 0:
        print(f"Recording {args.number_of_expert_episodes} episode(s) with the final policy")
        stage_start = episodes_recorded()
        mean = collect_stage("expert", args.number_of_expert_episodes, agent.actor_mean, log_prefix="expert")
        summary["stages"]["expert"] = dict(**stage_range(stage_start), **mean)
        dump_json(summary_path, summary, indent=2)

    print(f"collection summary written to {summary_path}")
    # RecordEpisode.close() only cleans/dumps/closes the h5 when save_trajectory is set, so restore
    # the flags the recording gate turned off before tearing the envs down
    set_recording(True)
    logger.close()
    envs.close()
    eval_envs.close()
