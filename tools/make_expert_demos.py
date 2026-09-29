"""Roll a trained visual PPO expert out under graded action noise and record a ManiSkill dataset.

Takes a run folder produced by `ppo_visual_expert_fast.py`, reads the camera view, resolution,
control mode and state flag back out of its `training_summary.json`, and rebuilds exactly that env
through `custom_maniskill_tasks.make_env`. The policy rolled out is `best_ckpt.pt`, the one that
run's `training_summary.json` names as its expert, rather than whichever policy the last iteration
left behind. The expert cannot be run under a camera it was not
trained on, and the demos cannot drift from the run that produced them, because neither is a
command line argument -- both come from the run itself.

Each episode draws one action-noise std uniformly from `[0, max_action_noise]` and holds it for the
whole episode; every step takes `actor_mean(obs) + N(0, std^2)`. One std per episode rather than
per step is what makes the noise level a usable per-episode label: a demo is cleanly "expert" at
std 0 and degrades smoothly towards random at `max_action_noise`, and the std is written into the
dataset's own metadata (`action_noise_std` on each episode entry, plus `demo_summary.json`).

The recorded dataset is the same shape as the project's existing ManiSkill datasets -- the raw
ManiSkill observation dict (`obs/sensor_data/<camera>/rgb`, `obs/agent/*`, `obs/extra/*`), env
states, rewards, per-step success -- because `RecordEpisode` sits *underneath* the observation
flattening, seeing what the task returns rather than what the CNN eats. The `-v1.1` ids never
terminate early, so every episode is exactly one full horizon and the `terminated` field is all
False; `success` is recorded separately, per step and per episode.

Actions are recorded unclipped, which is the convention the project's existing datasets already
follow: the PPO actor's output is unbounded and ManiSkill clips to [-1, 1] internally when it
applies one, so at large noise the recorded action and the effected action diverge. `--clip_actions`
records the effected action instead; the summary reports what fraction was out of range either way.

Alongside the dataset, `noise_grid.mp4` shows `--video_rows` sample episodes at each of
`--video_cols` noise levels evenly spaced from 0 to `max_action_noise`, one level per labelled
column, so the chosen noise range can be eyeballed before committing to a long collection.
"""

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from mani_skill.utils import gym_utils
from mani_skill.utils.io_utils import dump_json
from mani_skill.utils.visualization import misc as vis_misc
from mani_skill.utils.wrappers.flatten import (
    FlattenActionSpaceWrapper,
    FlattenRGBDObservationWrapper,
)
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

# registers the -v1.1 ids, and is where the camera views and backend pinning live
from custom_maniskill_tasks import backend_kwargs, make_env

import gymnasium as gym
import imageio
import torch
import tqdm
import tyro
from PIL import Image, ImageDraw, ImageFont

# the expert's architecture and the observation copy it expects, from the script that trained it,
# so a checkpoint cannot be loaded into a network that has drifted from the one it was saved from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppo_visual_expert_fast import Agent, as_obs_td

FONT_PATH = os.path.join(os.path.dirname(vis_misc.__file__), "UbuntuSansMono-Regular.ttf")


@dataclass
class Args:
    agent_folder: str
    """the run folder of a ppo_visual_expert_fast.py training run: the folder holding
    training_summary.json and the checkpoints, e.g.
    /data/.../ppo_visual_experts/PushCube-v1.1-ppo-visual-wrist-224-pd_ee_delta_pos"""
    num_demos: str = "1000"
    """how many demos to record; rounded up to a whole multiple of --num_envs. Digit separators
    are accepted, so both 10000 and 10,000 (and 10_000) mean the same thing."""
    max_action_noise: float = 1.0
    """the top of the per-episode action-noise range. Actions are normalized to [-1, 1], so a std
    of 1 is noise as large as the whole action range and the policy is nearly drowned out."""
    checkpoint_name: str = "best_ckpt.pt"
    """which checkpoint in agent_folder to roll out. best_ckpt.pt is the policy from the
    highest-scoring eval, which is what `training_summary.json` calls the expert; final_ckpt.pt is
    whatever the last iteration happened to leave behind, and is only the same policy when the run
    stopped on its success threshold."""
    output_dir: Optional[str] = None
    """where to write the dataset and the video; defaults to `{agent_folder}/demos`"""

    num_envs: int = 128
    """how many episodes to collect in parallel, and the granularity of the demo count.
    RecordEpisode buffers a whole rollout in host memory, so this is bounded by RAM: one env is
    (horizon + 1) * resolution^2 * 3 bytes, ~7.7 MiB at 224x224."""
    seed: int = 1
    """seed of the first reset; every later episode follows from it"""
    device: str = "cuda:0"
    """torch device, which also pins the physx sim and the sapien renderer -- give it an explicit
    index on a multi-GPU host"""
    reconfiguration_freq: Optional[int] = 1
    """how often to reconfigure the scene, in resets. 1 re-randomizes everything the task
    randomizes at reconfiguration, which is what the project's existing datasets were recorded
    with; 0 (or None) is much faster and less diverse."""
    clip_actions: bool = False
    """record the action the sim actually applied (clipped to [-1, 1]) instead of the raw
    actor_mean + noise"""

    skip_demos: bool = False
    """only render the noise grid video, for choosing --max_action_noise before a long collection"""
    skip_video: bool = False
    """only record the dataset"""
    video_rows: int = 5
    """sample episodes per noise level in noise_grid.mp4"""
    video_cols: int = 7
    """noise levels in noise_grid.mp4, evenly spaced from 0 to --max_action_noise"""
    video_tile: int = 160
    """pixel size of one episode's tile in noise_grid.mp4"""
    video_fps: int = 15
    """frame rate of noise_grid.mp4; a 50-step episode is over in a blink at 30"""


def parse_count(text: str) -> int:
    """An integer written with `,` or `_` separators, since demo counts are written that way."""
    cleaned = text.replace(",", "").replace("_", "").strip()
    if not cleaned.isdigit():
        raise ValueError(f"expected a whole number of demos, got {text!r}")
    return int(cleaned)


def load_run_config(agent_folder: Path) -> dict:
    """The env this run was trained on, read back out of its own summary.

    Every key is required rather than defaulted: a demo recorded through a different camera or
    control mode than the policy was trained on is silently useless, so a summary that predates a
    field should fail here instead of guessing.
    """
    summary_path = agent_folder / "training_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"{summary_path} not found -- --agent_folder must be a ppo_visual_expert_fast.py run "
            "folder (the one holding training_summary.json and the checkpoints)"
        )
    summary = json.loads(summary_path.read_text())
    return dict(
        env_id=summary["env_id"],
        camera_view=summary["camera_view"],
        camera_resolution=summary["camera_resolution"],
        include_state=summary["include_state"],
        control_mode=summary["control_mode"],
    )


def resolve_backends(device: torch.device) -> dict:
    """Pin physx and sapien to the same explicit cuda index torch is on."""
    if device.type == "cuda":
        gpu_idx = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_device(gpu_idx)
        return backend_kwargs(f"physx_cuda:{gpu_idx}")
    return backend_kwargs("physx_cpu", render_backend="sapien_cpu")


def build_env(cfg, num_envs, backends, *, record_to=None, reconfiguration_freq=1, render_size=None):
    """The training env rebuilt from `cfg`, optionally with a recorder under the flattening.

    Wrapper order matters and is the whole point: `RecordEpisode` goes *below*
    `FlattenRGBDObservationWrapper`, so the dataset holds the task's own observation dict (the
    shape every other dataset in this project has) while the policy above it still receives the
    flat `{rgb, state}` the CNN was trained on.
    """
    human_render = dict(shader_pack="default")
    if render_size is not None:
        human_render.update(width=render_size, height=render_size)
    env = make_env(
        cfg["env_id"],
        num_envs=num_envs,
        obs_mode="rgb",
        render_mode="rgb_array",
        control_mode=cfg["control_mode"],
        camera_view=cfg["camera_view"],
        camera_resolution=cfg["camera_resolution"],
        # the -v1.1 tasks report no termination of their own, so this wrapper is a no-op; the
        # vector env below is what stops a truncation from being treated as an early end
        ignore_terminations=False,
        reconfiguration_freq=reconfiguration_freq,
        reward_mode="normalized_dense",
        human_render_camera_configs=human_render,
        **backends,
    )
    if record_to is not None:
        env = RecordEpisode(
            env,
            output_dir=str(record_to),
            save_trajectory=True,
            trajectory_name="trajectory",
            save_video=False,
            record_reward=True,
            record_env_state=True,
            max_steps_per_video=None,
        )
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=cfg["include_state"])
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    return env


def load_agent(sample_obs, n_act, ckpt_path: Path, device):
    if not ckpt_path.exists():
        available = sorted(p.name for p in ckpt_path.parent.glob("*.pt"))
        raise FileNotFoundError(
            f"{ckpt_path} not found. {ckpt_path.parent.name} holds "
            + (", ".join(available) if available else "no checkpoints at all")
            + " -- pass one of those as --checkpoint_name"
        )
    agent = Agent(sample_obs, n_act, device=device)
    agent.load_state_dict(torch.load(ckpt_path, map_location=device))
    agent.eval()
    return agent


def noisy_action(agent, obs, num_envs, stds, clip):
    """`actor_mean(obs)` plus per-episode gaussian noise, one std per env."""
    with torch.no_grad():
        action = agent.get_action(as_obs_td(obs, num_envs))
    action = action + stds[:, None] * torch.randn_like(action)
    return action.clamp(-1.0, 1.0) if clip else action


def collect_demos(args, cfg, backends, device, out_dir: Path) -> dict:
    """Record `num_demos` episodes, each at its own action-noise level, into one h5."""
    num_demos = parse_count(args.num_demos)
    env = build_env(
        cfg, args.num_envs, backends,
        record_to=out_dir, reconfiguration_freq=args.reconfiguration_freq,
    )
    horizon = gym_utils.find_max_episode_steps_value(env)
    # no early stop and no partial resets: every env truncates together at the horizon, so one
    # rollout flushes exactly num_envs complete episodes, in env order
    envs = ManiSkillVectorEnv(env, args.num_envs, ignore_terminations=True, record_metrics=True)

    num_rollouts = math.ceil(num_demos / args.num_envs)
    total = num_rollouts * args.num_envs
    if total != num_demos:
        print(f"rounding {num_demos} demos up to {total}, a whole multiple of num_envs={args.num_envs}")
    frame_mib = (horizon + 1) * cfg["camera_resolution"] ** 2 * 3 / 1048576
    print(f"{cfg['env_id']}: {total} demos over {num_rollouts} rollout(s) of {args.num_envs} envs, "
          f"horizon {horizon}, noise std ~ U(0, {args.max_action_noise})")
    print(f"RecordEpisode buffers {frame_mib * args.num_envs:.0f} MiB of frames per rollout")

    obs, _ = envs.reset(seed=args.seed)
    agent = load_agent(
        as_obs_td(obs, args.num_envs),
        math.prod(envs.single_action_space.shape),
        Path(args.agent_folder) / args.checkpoint_name,
        device,
    )
    noise_stds, successes, out_of_range, action_count = [], [], 0, 0
    h5_path = out_dir / "trajectory.h5"
    for rollout in tqdm.tqdm(range(num_rollouts), desc="collecting"):
        stds = torch.rand(args.num_envs, device=device) * args.max_action_noise
        noise_stds.append(stds.cpu().numpy())
        for _ in range(horizon):
            action = noisy_action(agent, obs, args.num_envs, stds, args.clip_actions)
            out_of_range += int((action.abs() > 1.0).sum())
            action_count += action.numel()
            obs, _, _, _, infos = envs.step(action)
        # the horizon-th step truncates every env at once, which auto-resets and flushes
        assert "final_info" in infos, "expected every env to truncate on the horizon-th step"
        successes.append(infos["final_info"]["episode"]["success_once"].cpu().numpy())
        if rollout == 0 and h5_path.exists():
            per_demo = h5_path.stat().st_size / args.num_envs
            print(f"first rollout wrote {per_demo / 1048576:.1f} MiB per demo, so {total} demos "
                  f"is roughly {per_demo * total / 1024 ** 3:.1f} GiB")
    envs.close()

    noise_stds = np.concatenate(noise_stds)
    successes = np.concatenate(successes)
    label_episodes(out_dir / "trajectory.json", noise_stds, total)
    return dict(
        demos=total,
        horizon=horizon,
        num_envs=args.num_envs,
        max_action_noise=args.max_action_noise,
        clip_actions=args.clip_actions,
        checkpoint=args.checkpoint_name,
        success_rate=float(successes.mean()),
        success_rate_by_noise_decile=success_by_noise(noise_stds, successes, args.max_action_noise),
        fraction_actions_out_of_range=out_of_range / max(action_count, 1),
    )


def success_by_noise(stds, successes, max_noise, bins=10):
    """Success rate per noise band -- the table that says whether the noise range is well chosen."""
    edges = np.linspace(0.0, max_noise, bins + 1)
    idx = np.clip(np.digitize(stds, edges[1:-1]), 0, bins - 1)
    return [
        dict(
            noise_range=[float(edges[b]), float(edges[b + 1])],
            episodes=int((idx == b).sum()),
            success_rate=float(successes[idx == b].mean()) if (idx == b).any() else None,
        )
        for b in range(bins)
    ]


def label_episodes(json_path: Path, noise_stds, expected):
    """Write each episode's noise std into the dataset's own json.

    Flushes happen in env-index order and every env truncates together, so `traj_i` is the i-th
    std sampled. Both invariants are asserted rather than assumed, since a silent misalignment
    would mislabel the whole dataset.
    """
    data = json.loads(json_path.read_text())
    episodes = data["episodes"]
    assert len(episodes) == expected, (
        f"{json_path} holds {len(episodes)} episodes but {expected} were collected; the "
        "episode-to-noise mapping cannot be trusted"
    )
    for i, episode in enumerate(episodes):
        assert episode["episode_id"] == i, f"episode {i} is out of order (id {episode['episode_id']})"
        episode["action_noise_std"] = float(noise_stds[i])
    dump_json(str(json_path), data, indent=2)


def render_noise_grid(args, cfg, backends, device, out_path: Path):
    """A tiled video: one labelled column per noise level, `video_rows` sample episodes each."""
    levels = np.linspace(0.0, args.max_action_noise, args.video_cols)
    num_envs = args.video_rows * args.video_cols
    env = build_env(
        cfg, num_envs, backends,
        reconfiguration_freq=args.reconfiguration_freq, render_size=args.video_tile,
    )
    horizon = gym_utils.find_max_episode_steps_value(env)
    obs, _ = env.reset(seed=args.seed)
    agent = load_agent(
        as_obs_td(obs, num_envs),
        math.prod(env.get_wrapper_attr("single_action_space").shape),
        Path(args.agent_folder) / args.checkpoint_name,
        device,
    )

    # env i sits at row i // cols, column i % cols, so its noise level is levels[i % cols]
    stds = torch.as_tensor(
        np.tile(levels, args.video_rows), dtype=torch.float32, device=device
    )
    print(f"rendering {args.video_rows}x{args.video_cols} noise grid at levels "
          + ", ".join(f"{l:.2f}" for l in levels))

    frames = []
    for _ in tqdm.tqdm(range(horizon), desc="rendering"):
        action = noisy_action(agent, obs, num_envs, stds, args.clip_actions)
        obs, _, _, _, _ = env.step(action)
        frames.append(env.unwrapped.render_rgb_array().cpu().numpy().astype(np.uint8))
    env.close()

    banner = column_banner(levels, args.video_tile)
    grid = [
        pad_to_macro_block(
            np.concatenate([banner, tile_grid(f, args.video_rows, args.video_cols)], axis=0)
        )
        for f in frames
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), grid, fps=args.video_fps, quality=8)
    print(f"noise grid video written to {out_path} ({grid[0].shape[1]}x{grid[0].shape[0]})")


def tile_grid(frames: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """(rows*cols, H, W, 3) -> one (rows*H, cols*W, 3) image, filled row-major."""
    n, h, w, c = frames.shape
    assert n == rows * cols, f"{n} frames does not fill a {rows}x{cols} grid"
    return frames.reshape(rows, cols, h, w, c).transpose(0, 2, 1, 3, 4).reshape(rows * h, cols * w, c)


def pad_to_macro_block(image: np.ndarray, block: int = 16) -> np.ndarray:
    """Pad to a multiple of ffmpeg's macro block, which imageio would otherwise rescale to."""
    h, w = image.shape[:2]
    pad_h, pad_w = (-h) % block, (-w) % block
    if not pad_h and not pad_w:
        return image
    return np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)))


def column_banner(levels, tile: int, height: int = 28) -> np.ndarray:
    """A header strip naming each column's noise level, centred over its column."""
    width = tile * len(levels)
    image = Image.new("RGB", (width, height), color=(16, 16, 16))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT_PATH, size=max(11, min(18, tile // 9)))
    for i, level in enumerate(levels):
        text = f"noise {level:.2f}"
        bbox = draw.textbbox((0, 0), text=text, font=font)
        x = i * tile + (tile - (bbox[2] - bbox[0])) // 2
        draw.text((x, (height - (bbox[3] - bbox[1])) // 2 - bbox[1]), text, fill=(255, 255, 255), font=font)
    return np.array(image)


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.max_action_noise >= 0.0, "max_action_noise must be non-negative"
    assert args.video_cols >= 2, "a noise grid needs at least a zero column and a max column"

    agent_folder = Path(args.agent_folder)
    cfg = load_run_config(agent_folder)
    out_dir = Path(args.output_dir) if args.output_dir else agent_folder / "demos"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{agent_folder.name}: {cfg}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    backends = resolve_backends(device)

    summary = dict(agent_folder=str(agent_folder), **cfg)
    if not args.skip_video:
        render_noise_grid(args, cfg, backends, device, out_dir / "noise_grid.mp4")

    if not args.skip_demos:
        summary.update(collect_demos(args, cfg, backends, device, out_dir))
        dump_json(str(out_dir / "demo_summary.json"), summary, indent=2)
        print(f"demo summary written to {out_dir / 'demo_summary.json'}")
