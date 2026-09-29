"""Replay ManiSkill trajectories stored in HDF5 (.h5) format

The replayed trajectory can use different observation modes and control modes.

We support translating actions from certain controllers to a limited number of controllers.

The script is only tested for Panda, and may include some Panda-specific hardcode.
"""

import copy
import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import Annotated, List, Literal, Optional

import gymnasium as gym
import h5py
import numpy as np
import torch
import tyro
from tqdm import tqdm

import mani_skill.envs

# The camera views, the device pinning and the -v1.1 task ids all come from the central module, so a
# dataset converted here shows a policy exactly what its online env will. Its `make_env` is not used
# directly: this script builds the env from the kwargs recorded in the trajectory json and needs
# `RecordEpisode` to see one primitive action per step, which the FrameSkip/FrameStack wrappers
# `make_env` can add would break.
from custom_maniskill_tasks import backend_kwargs, build_sensor_configs, camera_view_applied
from mani_skill.envs.utils.system.backend import CPU_SIM_BACKENDS
from mani_skill.trajectory import utils as trajectory_utils
from mani_skill.trajectory.merge_trajectory import merge_trajectories
from mani_skill.trajectory.utils.actions import conversion as action_conversion
from mani_skill.utils import common, io_utils, wrappers
from mani_skill.utils.logging_utils import logger
from mani_skill.utils.visualization.misc import images_to_video, tile_images
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode


@dataclass
class Args:
    traj_path: str
    """Path to the trajectory .h5 file to replay"""
    sim_backend: Annotated[Optional[str], tyro.conf.arg(aliases=["-b"])] = None
    """Which simulation backend to use. Can be 'physx_cpu', 'physx_gpu'. If not specified the backend used is the same as the one used to collect the trajectory data."""
    obs_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-o"])] = None
    """Target observation mode to record in the trajectory. See
    https://maniskill.readthedocs.io/en/latest/user_guide/concepts/observation.html for a full list of supported observation modes."""
    target_control_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-c"])] = None
    """Target control mode to convert the demonstration actions to.
    Note that not all control modes can be converted to others successfully and not all robots have easy to convert control modes.
    Currently the Panda robots are the best supported when it comes to control mode conversion. Furthermore control mode conversion is not supported in GPU parallelized environments.
    """
    verbose: bool = False
    """Whether to print verbose information during trajectory replays"""
    save_traj: bool = False
    """Whether to save trajectories to disk. This will not override the original trajectory file."""
    save_video: bool = False
    """Whether to save videos"""
    save_example_video: bool = True
    """Whether to decode the image observations of the newly written dataset back into mp4s
    alongside it. Unlike `--save-video` (which re-renders `--render-mode` from the human render
    camera) this shows the exact frames stored in the dataset, so it is a direct check on what
    `--camera-resolution` and the sensor pose actually produced. `--example-video-count`
    episodes are written, spread over the dataset's return distribution. No-op unless
    `--save-traj` is set and the target `--obs-mode` contains image observations."""
    example_video_count: int = 10
    """How many episodes `--save-example-video` renders, picked at equally spaced quantiles of
    episode return (0 = the worst episode, 1 = the best), so the set spans what the dataset
    actually contains rather than showing the same early episodes every time. Return needs
    `--record-rewards`; without it the episodes are spread evenly by index instead and a warning
    says so. Fewer videos than asked for when the dataset has fewer episodes, since the same
    episode is never rendered twice. When replaying with multiple CPU processes the quantiles are
    taken over worker 0's shard, since that is the worker which writes the videos."""
    max_retry: int = 0
    """Maximum number of times to try and replay a trajectory until the task reaches a success state at the end."""
    discard_timeout: bool = False
    """Whether to discard episodes that timeout and are truncated (depends on the max_episode_steps parameter of task)"""
    allow_failure: bool = False
    """Whether to include episodes that fail in saved videos and trajectory data based on the environment's evaluation returned "success" label"""
    vis: bool = False
    """Whether to visualize the trajectory replay via the GUI."""
    use_env_states: bool = False
    """Whether to replay by environment states instead of actions. This guarantees that the environment will look exactly
    the same as the original trajectory at every step."""
    use_first_env_state: bool = False
    """Use the first env state in the trajectory to set initial state. This can be useful for trying to replay
    demonstrations collected in the CPU simulation in the GPU simulation by first starting with the same initial
    state as GPU simulated tasks will randomize initial states differently despite given the same seed compared to CPU sim."""
    count: Optional[int] = None
    """Number of demonstrations to replay before exiting. By default will replay all demonstrations"""
    trajectories_to_replay: Optional[List[int]] = None
    """List of trajectory indices (into the trajectory json's episode list) to replay. If set, overrides `count` and
    only these trajectories are replayed. By default (None) the first `count` trajectories are replayed."""
    reward_mode: Optional[str] = None
    """Specifies the reward type that the env should use. By default it will pick the first supported reward mode. Most environments
    support 'sparse', 'none', and some further support 'normalized_dense' and 'dense' reward modes"""
    record_rewards: bool = False
    """Whether the replayed trajectory should include rewards"""
    shader: Optional[str] = None
    """Change shader used for rendering for all cameras. Default is none meaning it will use whatever was used in the original data collection or the environment default.
    Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer"""
    video_fps: Optional[int] = None
    """The FPS of saved videos. Defaults to the control frequency"""
    render_mode: str = "rgb_array"
    """The render mode used for saving videos. Typically there is also 'sensors' and 'all' render modes which further render all sensor outputs like cameras."""
    camera_resolution: int = 448
    """The resolution to render every observation camera at, whichever `--camera-view` selects."""
    camera_view: Literal["default", "standard", "wrist", "focused"] = "default"
    """Which observation camera the recorded dataset should see the scene through.
    'default' (also spelled 'standard') keeps the task's own registered base_camera pose and fov;
    'wrist' swaps in a Panda carrying the hand-mounted realsense and records only that, dropping
    base_camera; 'focused' points base_camera down at the workspace through a narrow (36 degree)
    fov. These are the views defined in environments/custom_maniskill_tasks, the same ones
    `make_env` builds an online env with -- pick the one the policy will be run with, since a
    dataset recorded through one view does not transfer to an env using another."""
    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    """Number of environments to run to replay trajectories. With CPU backends typically this is parallelized via python multiprocessing.
    For parallelized simulation backends like physx_gpu, this is parallelized within a single python process by leveraging the GPU."""


@dataclass
class ReplayResult:
    num_replays: int
    successful_replays: int


def sanity_check_and_format_seed(episode):
    """sanity checks the trajectory seed aligns with the episode seed. reformats the reset kwargs seed if missing or formatted wrong"""
    if "seed" in episode["reset_kwargs"]:
        if isinstance(episode["reset_kwargs"]["seed"], list):

            assert (
                len(episode["reset_kwargs"]["seed"]) == 1
            ), f"found multiple seeds for one trajectory (id={episode['episode_id']}) in the reset kwargs which means it is ambiguous which seed to use"
            episode["reset_kwargs"]["seed"] = episode["reset_kwargs"]["seed"][0]
        assert (
            episode["reset_kwargs"]["seed"] == episode["episode_seed"]
        ), f"found mismatch between trajectory seed and episode seed (id={episode['episode_id']})"
    else:
        episode["reset_kwargs"]["seed"] = episode["episode_seed"]


def replay_parallelized_sim(
    args: Args, env: RecordEpisode, pbar, episodes, trajectories
):
    pbar.reset(total=len(episodes))
    warned_reset_kwargs_options = False
    # split all episodes into batches of args.num_envs environments and process each batch in parallel, truncating where necessary
    # add fake episode padding to the end of the episodes to make sure all batches are the same size
    episode_pad = (args.num_envs - len(episodes) % args.num_envs) % args.num_envs
    batches = np.pad(
        np.array(episodes),
        (0, episode_pad),
        mode="constant",
        constant_values=episodes[-1],
    ).reshape(-1, args.num_envs)

    successful_replays = 0
    if pbar is not None:
        pbar.reset(total=len(episodes))
    for episode_batch_index, episode_batch in enumerate(batches):
        trajectory_ids = [episode["episode_id"] for episode in episode_batch]
        episode_lens = np.array([episode["elapsed_steps"] for episode in episode_batch])
        ori_control_mode = episode_batch[0]["control_mode"]
        assert all(
            [episode["control_mode"] == ori_control_mode for episode in episode_batch]
        ), "Replay trajectory with parallelized environments is only supported for trajectories with the same control mode"
        episode_batch_max_len = max(episode_lens)
        seeds = torch.tensor(
            [episode["episode_seed"] for episode in episode_batch],
            device=env.base_env.device,
        )
        env.reset(seed=seeds)

        # generate batched env states and actions
        env_states_list = []
        original_actions_batch = []
        env_states_batch = []  # list of batched env states shape (max_steps, D)
        for i, trajectory_id in enumerate(trajectory_ids):

            # sanity check seeds and warn user if reset kwargs includes options (which are not supported in GPU sim replay)
            traj = trajectories[f"traj_{trajectory_id}"]
            episode = episode_batch[i]
            sanity_check_and_format_seed(episode)
            if not warned_reset_kwargs_options and "options" in episode["reset_kwargs"]:
                logger.warning(
                    f"Reset kwargs includes options, which are not supported in GPU sim replay and will be ignored."
                )
                warned_reset_kwargs_options = True

            # note (stao): this code to reformat the trajectories into a list of batched dicts can be optimized
            env_states = trajectory_utils.dict_to_list_of_dicts(traj["env_states"])
            actions = np.array(traj["actions"])

            # padding
            for _ in range(episode_batch_max_len + 1 - len(env_states)):
                env_states.append(env_states[-1])
            if len(actions) < episode_batch_max_len:
                actions = np.concatenate(
                    [
                        actions,
                        np.zeros(
                            (episode_batch_max_len - len(actions), actions.shape[1])
                        ),
                    ],
                    axis=0,
                )
            env_states_list.append(env_states)
            original_actions_batch.append(actions)
        for t in range(episode_batch_max_len + 1):
            env_states_batch.append(
                trajectory_utils.list_of_dicts_to_dict(
                    [env_states_list[i][t] for i in range(len(env_states_list))]
                )
            )

        original_actions_batch = np.stack(original_actions_batch, axis=1)
        if args.use_first_env_state or args.use_env_states:
            # set the first environment state to the first states in the trajectories given
            env.base_env.set_state_dict(env_states_batch[0])
            if args.save_traj:
                # replace the first saved env state
                # since we set state earlier and RecordEpisode will save the reset to state.
                def recursive_replace(x, y):
                    if isinstance(x, np.ndarray):
                        x[-1, :] = y[-1, :]
                    else:
                        for k in x.keys():
                            recursive_replace(x[k], y[k])

                recursive_replace(
                    env._trajectory_buffer.state, common.batch(env_states_batch[0])
                )
                recursive_replace(
                    env._trajectory_buffer.observation,
                    common.to_numpy(common.batch(env.base_env.get_obs())),
                )

        # replay with env states / actions
        if (
            args.target_control_mode is None
            or ori_control_mode == args.target_control_mode
        ):
            flushed_trajectories = np.zeros(len(episode_batch), dtype=bool)
            # mark the fake padding trajectories as flushed
            if episode_batch_index == len(batches) - 1 and episode_pad > 0:
                flushed_trajectories[-episode_pad:] = True
            for t, a in enumerate(original_actions_batch):
                _, _, _, truncated, info = env.step(a)
                if args.use_env_states:
                    # NOTE (stao): due to the high precision nature of some tasks even taking a single step in GPU simulation (in e.g. PushT-v1) can lead
                    # to some non-deterministic behaviors leading to some steps labeled with slightly wrong observations/rewards/success/fail data (1e-4 error).
                    # I unfortunately do not have a good solution for this apart from using the same number of parallel environments to replay demos as the original trajectory collection.
                    env.base_env.set_state_dict(env_states_batch[t])
                if args.vis:
                    env.base_env.render_human()
                # if the elapsed_steps mark saved in the trajectory is reached for any env, flush that trajectory buffer

                if args.save_traj:
                    envs_to_flush = (t >= episode_lens - 1) & (~flushed_trajectories)
                    flushed_trajectories |= envs_to_flush
                    if envs_to_flush.sum() > 0:
                        pbar.update(n=envs_to_flush.sum())
                        if not args.allow_failure:
                            if "success" in info:
                                envs_to_flush &= (info["success"] == True).cpu().numpy()
                        if args.discard_timeout:
                            envs_to_flush &= (truncated == False).cpu().numpy()
                        successful_replays += envs_to_flush.sum()
                        env.flush_trajectory(
                            env_idxs_to_flush=np.where(envs_to_flush)[0]
                        )
        else:
            raise NotImplementedError(
                "Replay with different control modes are not supported when replaying on GPU parallelized environments"
            )
    return ReplayResult(
        num_replays=len(episodes), successful_replays=successful_replays
    )


def replay_cpu_sim(
    args: Args, env: RecordEpisode, ori_env, pbar, episodes, trajectories
):
    successful_replays = 0
    for episode in episodes:
        sanity_check_and_format_seed(episode)
        episode_id = episode["episode_id"]
        traj_id = f"traj_{episode_id}"
        reset_kwargs = episode["reset_kwargs"]
        ori_control_mode = episode["control_mode"]
        if pbar is not None:
            pbar.set_description(f"Replaying {traj_id}")
        if traj_id not in trajectories:
            tqdm.write(f"{traj_id} does not exist in {args.traj_path}")
            continue

        for _ in range(args.max_retry + 1):
            # Each trial for each trajectory to replay, we reset the environment
            # and optionally set the first environment state
            env.reset(**reset_kwargs)
            if ori_env is not None:
                ori_env.reset(**reset_kwargs)

            # set first environment state and update recorded env state
            if args.use_first_env_state or args.use_env_states:
                ori_env_states = trajectory_utils.dict_to_list_of_dicts(
                    trajectories[traj_id]["env_states"]
                )
                if ori_env is not None:
                    ori_env.unwrapped.set_state_dict(ori_env_states[0])
                env.base_env.set_state_dict(ori_env_states[0])
                ori_env_states = ori_env_states[1:]
                if args.save_traj:
                    # replace the first saved env state
                    # since we set state earlier and RecordEpisode will save the reset to state.
                    def recursive_replace(x, y):
                        if isinstance(x, np.ndarray):
                            x[-1, :] = y[-1, :]
                        else:
                            for k in x.keys():
                                recursive_replace(x[k], y[k])

                    recursive_replace(
                        env._trajectory_buffer.state, common.batch(ori_env_states[0])
                    )
                    fixed_obs = env.base_env.get_obs()
                    recursive_replace(
                        env._trajectory_buffer.observation,
                        common.to_numpy(common.batch(fixed_obs)),
                    )
            # Original actions to replay
            ori_actions = trajectories[traj_id]["actions"][:]
            info = {}

            # Without conversion between control modes
            assert (
                args.target_control_mode is None
                or ori_control_mode == args.target_control_mode
                or not args.use_env_states
            ), "Cannot use env states when trying to \
                convert from one control mode to another. This is because control mode conversion causes there to be changes \
                in how many actions are taken to achieve the same states"
            if (
                args.target_control_mode is None
                or ori_control_mode == args.target_control_mode
            ):
                n = len(ori_actions)
                if pbar is not None:
                    pbar.reset(total=n)
                for t, a in enumerate(ori_actions):
                    if pbar is not None:
                        pbar.update()
                    _, _, _, truncated, info = env.step(a)
                    if args.use_env_states:
                        env.base_env.set_state_dict(ori_env_states[t])
                    if args.vis:
                        env.base_env.render_human()

            # From joint position to others
            elif ori_control_mode == "pd_joint_pos":
                info = action_conversion.from_pd_joint_pos(
                    args.target_control_mode,
                    ori_actions,
                    ori_env,
                    env,
                    render=args.vis,
                    pbar=pbar,
                    verbose=args.verbose,
                )

            # From joint delta position to others
            elif ori_control_mode == "pd_joint_delta_pos":
                info = action_conversion.from_pd_joint_delta_pos(
                    args.target_control_mode,
                    ori_actions,
                    ori_env,
                    env,
                    render=args.vis,
                    pbar=pbar,
                    verbose=args.verbose,
                )
            else:
                raise NotImplementedError(
                    f"Script currently does not support converting {ori_control_mode} to {args.target_control_mode}"
                )

            success = info.get("success", False)
            if args.discard_timeout:
                success = success and (not truncated)

            if success or args.allow_failure:
                successful_replays += 1
                if args.save_traj:
                    env.flush_trajectory()
                if args.save_video:
                    env.flush_video(ignore_empty_transition=False)
                break
            else:
                if args.verbose:
                    print("info", info)
        else:
            env.flush_video(save=False)
            tqdm.write(f"Episode {episode_id} is not replayed successfully. Skipping")

    return ReplayResult(
        num_replays=len(episodes), successful_replays=successful_replays
    )


def _sorted_traj_ids(h5_file: h5py.File) -> List[str]:
    """The dataset's trajectory ids in episode order, rather than h5py's lexicographic one."""
    return sorted(h5_file.keys(), key=lambda key: int(key.split("_")[-1]))


def episode_returns(h5_file: h5py.File) -> Optional[dict]:
    """Total reward per trajectory, or None if this dataset stores no rewards.

    `RecordEpisode` only writes a `rewards` dataset under `--record-rewards`, which is off by
    default, so "no rewards" is an ordinary state rather than a corrupt file. A partially
    rewarded dataset is treated as unrewarded: ranking episodes on a quantity only some of them
    have would order them by whether they have it.
    """
    traj_ids = _sorted_traj_ids(h5_file)
    if not traj_ids or any("rewards" not in h5_file[traj_id] for traj_id in traj_ids):
        return None
    return {traj_id: float(np.sum(h5_file[traj_id]["rewards"][:])) for traj_id in traj_ids}


def select_example_episodes(h5_file: h5py.File, count: int) -> List[tuple]:
    """`count` episodes spread over the dataset, as `(traj_id, quantile, return)` triples.

    Sorted by episode return and sampled at `count` equally spaced quantiles, so the set runs
    from the worst episode (q=0) to the best (q=1) with the body of the distribution in between.
    A single video is taken at the median instead, that being the representative episode rather
    than the worst one.

    `return` is None when the dataset holds no rewards (see `episode_returns`); the episodes are
    then spread evenly by index, which still shows the whole dataset rather than its first
    `count` episodes, but says nothing about behaviour.

    Deduplicated, so a dataset with fewer episodes than `count` yields one video per episode
    rather than the same episode several times over.
    """
    traj_ids = _sorted_traj_ids(h5_file)
    if not traj_ids or count < 1:
        return []

    returns = episode_returns(h5_file)
    if returns is None:
        logger.warning(
            f"{h5_file.filename} stores no rewards, so example videos cannot be picked by "
            "episode return -- spreading them evenly over the dataset instead. Pass "
            "--record-rewards to rank them."
        )
        ordered = traj_ids
    else:
        ordered = sorted(traj_ids, key=lambda traj_id: returns[traj_id])

    quantiles = np.linspace(0.0, 1.0, count) if count > 1 else np.array([0.5])
    selected, seen = [], set()
    for quantile in quantiles:
        traj_id = ordered[int(round(float(quantile) * (len(ordered) - 1)))]
        if traj_id in seen:
            continue
        seen.add(traj_id)
        selected.append(
            (traj_id, float(quantile), None if returns is None else returns[traj_id])
        )
    return selected


def save_example_observation_video(
    h5_file: h5py.File,
    traj_id: str,
    output_dir: str,
    dataset_name: str,
    args: Args,
    fps: int,
    label: Optional[str] = None,
):
    """Render one episode of a freshly written dataset to an mp4 next to it.

    This reads the image observations back out of the new .h5 rather than re-rendering the scene,
    so the video is exactly what a policy trained on this dataset would see -- same camera pose,
    same `--camera-resolution`, same shader. All rgb sensors are tiled side by side.

    `label` goes into the filename ahead of `traj_id`, so a set of videos sorts in the order it
    was chosen in (`q000`, `q011`, ...) rather than by episode id.
    """
    obs = h5_file[traj_id].get("obs")
    sensor_data = obs.get("sensor_data") if isinstance(obs, h5py.Group) else None
    if sensor_data is None:
        logger.warning(
            f"{traj_id} of {h5_file.filename} stores no sensor data (obs_mode="
            f"{args.obs_mode}), skipping the example observation video"
        )
        return
    # one (T, H, W, 3) uint8 array per rgb sensor
    camera_frames = [
        sensor_data[cam_name]["rgb"][:]
        for cam_name in sorted(sensor_data.keys())
        if "rgb" in sensor_data[cam_name]
    ]
    if len(camera_frames) == 0:
        logger.warning(
            f"{traj_id} of {h5_file.filename} has sensor data but no rgb images (obs_mode="
            f"{args.obs_mode}), skipping the example observation video"
        )
        return

    if len(camera_frames) == 1:
        frames = list(camera_frames[0])
    else:
        frames = [
            tile_images([cam[t] for cam in camera_frames])
            for t in range(len(camera_frames[0]))
        ]
    # e.g. trajectory.rgb.pd_ee_delta_pos.physx_cuda.example_obs.q000.traj_17.mp4
    name = f"{dataset_name}.example_obs.{traj_id}" if label is None \
        else f"{dataset_name}.example_obs.{label}.{traj_id}"
    images_to_video(
        frames,
        output_dir=output_dir,
        video_name=name,
        fps=fps,
        verbose=True,
    )


def _main_helper(x):
    return _main(*x)


def _main(
    args: Args,
    use_cpu_backend,
    env_id,
    env_kwargs,
    ori_env_kwargs,
    record_episode_kwargs,
    proc_id: int = 0,
    num_procs=1,
):
    pbar = tqdm(position=proc_id, leave=None, unit="step", dynamic_ncols=True)

    # Load HDF5 containing trajectories
    traj_path = args.traj_path
    ori_h5_file = h5py.File(traj_path, "r")

    # Load associated json
    json_path = traj_path.replace(".h5", ".json")
    json_data = io_utils.load_json(json_path)
    # `camera_view_applied` holds the two things `sensor_configs` cannot express -- adding the wrist
    # camera to the robot, and dropping the task's own base_camera so RecordEpisode does not write a
    # second unread view into every episode. Entered per worker process because the CPU path uses
    # mp.Pool with the spawn start method, whose children inherit neither the agent registry nor the
    # task-class swap.
    with camera_view_applied(args.camera_view, env_id):
        env = gym.make(env_id, **env_kwargs)
    if isinstance(env.action_space, gym.spaces.Dict):
        logger.warning(
            "We currently do not track which wrappers are used when recording trajectories but majority of the time in multi-agent envs with dictionary action spaces the actions are stored as flat vectors. We will flatten the action space with the ManiSkill provided FlattenActionSpaceWrapper. If you do not want this behavior you can copy the replay trajectory code yourself and modify it as needed."
        )
        env = FlattenActionSpaceWrapper(env)
    # TODO (support adding wrappers to the recorded data?)

    # if pbar is not None:
    #     pbar.set_postfix(
    #         {
    #             "control_mode": env_kwargs.get("control_mode"),
    #             "obs_mode": env_kwargs.get("obs_mode"),
    #         }
    #     )

    ### Prepare for recording ###

    # note for maniskill trajectory datasets the general naming format is <trajectory_name>.<obs_mode>.<control_mode>.<sim_backend>.h5
    # If it is called <file_name>.h5 then we assume obs_mode=None, control_mode=pd_joint_pos, and sim_backend=physx_cpu
    output_dir = os.path.dirname(traj_path)
    ori_traj_name = os.path.splitext(os.path.basename(traj_path))[0]
    parts = ori_traj_name.split(".")
    if len(parts) > 1:
        ori_traj_name = parts[0]
    suffix = "{}.{}.{}".format(
        env.unwrapped.obs_mode,
        env.unwrapped.control_mode,
        env.unwrapped.backend.sim_backend,
    )
    new_traj_name = ori_traj_name + "." + suffix
    # name the example video after the merged dataset rather than this worker's shard
    example_video_dataset_name = new_traj_name
    if use_cpu_backend:
        if num_procs > 1:
            new_traj_name = new_traj_name + "." + str(proc_id)
        if args.target_control_mode is not None:
            # the source env for control-mode conversion gets the same view, so the two envs differ
            # only in their controller
            with camera_view_applied(args.camera_view, env_id):
                ori_env = gym.make(env_id, **ori_env_kwargs)
        else:
            ori_env = None
    else:
        pass

    video_fps = (
        args.video_fps if args.video_fps is not None else env.unwrapped.control_freq
    )
    env = wrappers.RecordEpisode(
        env,
        output_dir,
        trajectory_name=new_traj_name,
        video_fps=video_fps,
        **record_episode_kwargs,
    )

    if env.save_trajectory:
        output_h5_path = env._h5_file.filename
        assert not os.path.samefile(output_h5_path, traj_path)
    else:
        output_h5_path = None

    if args.trajectories_to_replay is not None:
        episodes = [json_data["episodes"][i] for i in args.trajectories_to_replay]
    else:
        episodes = json_data["episodes"][: args.count]
    if use_cpu_backend:
        inds = np.arange(len(episodes))
        inds = np.array_split(inds, num_procs)[proc_id]
        replay_result = replay_cpu_sim(
            args,
            env,
            ori_env,
            pbar,
            [episodes[index] for index in inds],
            ori_h5_file,
        )
    else:
        replay_result = replay_parallelized_sim(
            args,
            env,
            pbar,
            episodes,
            ori_h5_file,
        )

    # Written here rather than as each episode is flushed: which episodes to show is decided by
    # where their returns fall in the dataset's distribution, and that is not known until the
    # last one has been replayed. One worker writes them, over its own shard.
    if args.save_example_video and env.save_trajectory and proc_id == 0:
        env._h5_file.flush()
        selected = select_example_episodes(env._h5_file, args.example_video_count)
        if not selected:
            logger.warning(
                f"No episodes were saved to {output_h5_path}, skipping the example observation videos"
            )
        elif len(selected) < args.example_video_count:
            logger.warning(
                f"{output_h5_path} holds {len(selected)} distinct episode(s), fewer than the "
                f"--example-video-count of {args.example_video_count}; rendering each once"
            )
        for traj_id, quantile, episode_return in selected:
            described = "" if episode_return is None else f", return {episode_return:.3f}"
            logger.info(
                f"Example observation video: {traj_id} at return quantile "
                f"{quantile:.2f}{described}"
            )
            save_example_observation_video(
                env._h5_file,
                traj_id,
                output_dir,
                example_video_dataset_name,
                args,
                fps=video_fps,
                label=f"q{int(round(quantile * 100)):03d}",
            )

    env.close()
    ori_h5_file.close()
    return output_h5_path, replay_result


def parse_args(args=None):
    return tyro.cli(Args, args=args)


def main(args: Args):
    traj_path = args.traj_path
    # Load trajectory metadata json
    json_path = traj_path.replace(".h5", ".json")
    json_data = io_utils.load_json(json_path)

    env_info = json_data["env_info"]
    env_id = env_info["env_id"]
    ori_env_kwargs = env_info["env_kwargs"]
    env_kwargs = ori_env_kwargs.copy()

    ### Checks and setting up env kwargs ###
    # First we determine how to setup the environment to replay demonstrations and raise relevant warnings to the user
    if (
        "sim_backend" in ori_env_kwargs
        and ori_env_kwargs["sim_backend"] != args.sim_backend
        and args.use_env_states
    ):
        logger.warning(
            f"Warning: Using different backend ({args.sim_backend}) than the original used to collect the trajectory data "
            f"({ori_env_kwargs['sim_backend']}). This may cause replay failures due to "
            f"differences in simulation/physics engine backend. Use the same backend by passing -b {ori_env_kwargs['sim_backend']} "
            f"or replay by environment states by passing --use-env-states instead."
        )
    if args.sim_backend is None:
        # try to guess which sim backend to use
        if "sim_backend" not in ori_env_kwargs:
            args.sim_backend = "physx_cpu"
        else:
            args.sim_backend = ori_env_kwargs["sim_backend"]

    ori_env_kwargs["sim_backend"] = args.sim_backend
    env_kwargs["sim_backend"] = args.sim_backend

    # modify the env kwargs according to the users inputs
    target_obs_mode = args.obs_mode
    target_control_mode = args.target_control_mode
    if target_obs_mode is not None:
        env_kwargs["obs_mode"] = target_obs_mode
    if target_control_mode is not None:
        env_kwargs["control_mode"] = target_control_mode
    if args.shader is not None:
        env_kwargs["shader_dir"] = args.shader  # change all shaders

    # Fail on a bad --camera-view here rather than inside a worker process. Plain dicts and a
    # list-form pose, so these kwargs stay picklable for the mp.Pool spawn path below and
    # json-serializable for the metadata RecordEpisode writes.
    env_kwargs["sensor_configs"] = build_sensor_configs(
        args.camera_view, args.camera_resolution
    )

    env_kwargs["reward_mode"] = args.reward_mode
    env_kwargs[
        "render_mode"
    ] = (
        args.render_mode
    )  # note this only affects the videos saved as RecordEpisode wrapper calls env.render

    if args.save_example_video and not args.save_traj:
        logger.warning(
            "--save-example-video needs a written dataset to read observations back from; "
            "pass --save-traj as well"
        )

    record_episode_kwargs = dict(
        save_on_reset=False,
        save_trajectory=args.save_traj,
        save_video=args.save_video,
        record_reward=args.record_rewards,
    )

    if args.trajectories_to_replay is not None:
        num_episodes = len(json_data["episodes"])
        invalid_indices = [
            i for i in args.trajectories_to_replay if i < 0 or i >= num_episodes
        ]
        assert not invalid_indices, (
            f"Invalid trajectory indices {invalid_indices}: there are only {num_episodes} demos collected"
        )
        args.count = len(args.trajectories_to_replay)
    elif args.count is not None and args.count > len(json_data["episodes"]):
        logger.warning(
            f"Warning: Requested to replay {args.count} demos but there are only {len(json_data['episodes'])} demos collected, replaying all demos now"
        )
        args.count = len(json_data["episodes"])
    elif args.count is None:
        args.count = len(json_data["episodes"])

    pbar = tqdm(total=args.count, unit="step", dynamic_ncols=True)

    # if missing info or auto sim backend is provided, we try to infer which backend is being used
    if "sim_backend" not in env_kwargs or (
        env_kwargs["sim_backend"] == "auto"
        and ("num_envs" not in env_kwargs or env_kwargs["num_envs"] == 1)
    ):
        env_kwargs["sim_backend"] = "physx_cpu"
    # After the inference above, so both see the backend the env is actually built with.
    # `backend_kwargs` puts the sapien renderer on the same GPU as the physx sim: ManiSkill leaves
    # `render_backend` at a bare "gpu", which sapien resolves on its own and which on a multi-GPU
    # host need not agree with the sim, and the env then dies with "cuda pose buffer (cuda:0) and
    # the renderer (cuda:1) are on different cuda devices". Trajectories collected before this
    # pinning existed record no `render_backend` at all, so replaying them is exactly when it bites.
    for kwargs in (env_kwargs, ori_env_kwargs):
        kwargs.update(
            backend_kwargs(
                kwargs.get("sim_backend"), kwargs.get("render_backend"), args.num_envs
            )
        )
    env_kwargs["num_envs"] = args.num_envs
    if env_kwargs["sim_backend"] not in CPU_SIM_BACKENDS:
        record_episode_kwargs["max_steps_per_video"] = env_info["max_episode_steps"]
        output_h5_path, replay_result = _main(
            args,
            use_cpu_backend=False,
            env_id=env_id,
            env_kwargs=env_kwargs,
            ori_env_kwargs=ori_env_kwargs,
            record_episode_kwargs=record_episode_kwargs,
            proc_id=0,
            num_procs=1,
        )

    else:
        env_kwargs["num_envs"] = 1
        ori_env_kwargs["num_envs"] = 1
        if args.num_envs > 1:
            pool = mp.Pool(args.num_envs)
            proc_args = [
                (
                    copy.deepcopy(args),
                    True,
                    env_id,
                    env_kwargs,
                    ori_env_kwargs,
                    record_episode_kwargs,
                    i,
                    args.num_envs,
                )
                for i in range(args.num_envs)
            ]
            # res = pool.starmap(_main, proc_args)
            res = list(tqdm(pool.imap(_main_helper, proc_args), total=args.count))
            replay_results_list = [x[1] for x in res]
            trajectory_paths = [x[0] for x in res]
            output_h5_path = None
            pool.close()
            if args.save_traj:
                # A hack to find the path
                output_path = trajectory_paths[0][: -len("0.h5")] + "h5"
                merge_trajectories(output_path, trajectory_paths)
                output_h5_path = output_path
                for h5_path in trajectory_paths:
                    tqdm.write(f"Remove {h5_path}")
                    os.remove(h5_path)
                    json_path = h5_path.replace(".h5", ".json")
                    tqdm.write(f"Remove {json_path}")
                    os.remove(json_path)
            replay_result = ReplayResult(
                num_replays=sum([x.num_replays for x in replay_results_list]),
                successful_replays=sum(
                    [x.successful_replays for x in replay_results_list]
                ),
            )
        else:
            output_h5_path, replay_result = _main(
                args,
                use_cpu_backend=True,
                env_id=env_id,
                env_kwargs=env_kwargs,
                ori_env_kwargs=ori_env_kwargs,
                record_episode_kwargs=record_episode_kwargs,
                proc_id=0,
                num_procs=1,
            )

    pbar.close()
    print(
        f"Replayed {replay_result.num_replays} episodes, "
        f"{replay_result.successful_replays}/{replay_result.num_replays}={replay_result.successful_replays/replay_result.num_replays*100:.2f}% demos saved"
    )


if __name__ == "__main__":
    # spawn is needed due to warp init issue
    mp.set_start_method("spawn")
    main(parse_args())
