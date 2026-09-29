"""Render one PNG per trajectory stage from a staged ManiSkill collection.

Reads the trajectory .h5 written by `tools/ppo_stages_fast.py` (after it has been
replayed to RGB observations with `tools/convert_trajectories.sh`, since the raw
recording is state-only) and, for each stage in `collection_summary.json`
(random -> training -> expert), saves a `rows x cols` grid of frames: one
trajectory per row, `cols` frames spread evenly over that episode.

Outputs `{out_prefix}-{stage}.png`, e.g. `PushCube-v1-random.png`.

Stage slicing comes from `collection_summary.json`, whose `episode_range` is the
[start, end) pair of `traj_N` indices each stage wrote (replays that dropped
failed episodes leave gaps, so missing indices are skipped). Two layouts are
accepted for the h5 itself:

  * one combined `{run}/videos/trajectory*.rgb.*.h5` sliced by those ranges, or
  * per-stage `{run}/videos/{stage}/trajectory*.rgb.*.h5` files, each taken whole.

Frames whose step is a task success are outlined in green.

Usage:
    python tools/plot_trajectory_grids.py /data/.../PushCube-v1-ppo-staged-pd_ee_delta_pos
    python tools/plot_trajectory_grids.py <run_dir> --select random --seed 0 --rows 8 --cols 12
    python tools/plot_trajectory_grids.py <run_dir>/videos/trajectory.rgb.pd_ee_delta_pos.physx_cuda.h5 \
        --summary <run_dir>/collection_summary.json --out-prefix /tmp/pushcube
    # a file with no collection_summary.json (e.g. a random-only run) is one stage:
    python tools/plot_trajectory_grids.py <run_dir> --stage random
"""

import argparse
import glob
import json
import os

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STAGE_ORDER = ("random", "training", "expert")
# excluded from h5 auto-discovery: not raw RGB recordings
NON_RGB_MARKERS = (".dino.", ".mmap.", ".example_obs.")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "run",
        help="a collection run directory, its videos/ directory, or a trajectory .h5 directly",
    )
    parser.add_argument("--h5", default=None, help="explicit trajectory .h5 (overrides discovery)")
    parser.add_argument("--summary", default=None, help="explicit collection_summary.json")
    parser.add_argument(
        "--stages", nargs="+", default=None,
        help=f"stages to render (default: whichever of {', '.join(STAGE_ORDER)} the summary has)",
    )
    parser.add_argument(
        "--stage", default=None,
        help="name for the single stage a summary-less .h5 is treated as (default: all)",
    )
    parser.add_argument("--rows", type=int, default=5, help="trajectories per grid (default: 5)")
    parser.add_argument("--cols", type=int, default=10, help="frames per trajectory (default: 10)")
    parser.add_argument(
        "--select", choices=("spread", "random", "first"), default="spread",
        help="which trajectories of the stage to show: evenly spread over the stage (default), "
             "a random sample, or the first `rows`",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed for --select random (default: 0)")
    parser.add_argument(
        "--camera", default=None,
        help="sensor to read rgb from (default: the only/first camera in the file)",
    )
    parser.add_argument(
        "--out-prefix", default=None,
        help="output path prefix; '-{stage}.png' is appended (default: ./{run_name})",
    )
    parser.add_argument("--tile-size", type=float, default=1.3, help="inches per tile (default: 1.3)")
    parser.add_argument("--dpi", type=int, default=150, help="output dpi (default: 150)")
    return parser.parse_args()


def find_rgb_h5(directory):
    """The single RGB trajectory .h5 in `directory`, or None if there is no unambiguous one."""
    candidates = [
        path for path in sorted(glob.glob(os.path.join(directory, "*.h5")))
        if not any(marker in os.path.basename(path) for marker in NON_RGB_MARKERS)
    ]
    rgb = [path for path in candidates if ".rgb." in os.path.basename(path)]
    pool = rgb or candidates
    if len(pool) == 1:
        return pool[0]
    if len(pool) > 1:
        raise SystemExit(
            f"several trajectory files in {directory}, pass one with --h5:\n  "
            + "\n  ".join(os.path.basename(p) for p in pool)
        )
    return None


def has_rgb_obs(path):
    """Whether `path` stores image observations at all (a state-only recording does not)."""
    with h5py.File(path, "r") as handle:
        key = next((name for name in handle if name.startswith("traj_")), None)
        if key is None:
            return False
        obs = handle[key].get("obs")
        sensors = obs.get("sensor_data") if isinstance(obs, h5py.Group) else None
        return sensors is not None and any("rgb" in sensors[name] for name in sensors)


def resolve_run(run, h5_arg, summary_arg):
    """Returns (videos_dir, combined_h5 or None, summary dict or None).

    `run` may be a run directory, its videos/ directory, or an .h5 file.
    """
    run = os.path.abspath(run)
    if os.path.isfile(run):
        videos_dir, combined = os.path.dirname(run), run
    else:
        videos_dir = os.path.join(run, "videos") if os.path.isdir(os.path.join(run, "videos")) else run
        combined = None
    if h5_arg is not None:
        combined = os.path.abspath(h5_arg)
        videos_dir = os.path.dirname(combined)

    summary_path = summary_arg
    if summary_path is None:
        for directory in (videos_dir, os.path.dirname(videos_dir)):
            candidate = os.path.join(directory, "collection_summary.json")
            if os.path.isfile(candidate):
                summary_path = candidate
                break
    summary = None
    if summary_path is not None:
        with open(summary_path) as handle:
            summary = json.load(handle)
        print(f"stage ranges from {summary_path}")
    return videos_dir, combined, summary


def resolve_stages(args, videos_dir, combined, summary):
    """Returns [(stage_name, h5_path, episode_range or None)] to render, in order.

    A per-stage subdirectory layout wins over slicing the combined file, since its
    files hold exactly one stage each (episode_range then does not apply).
    """
    stage_dirs = {
        name: path for name in (args.stages or STAGE_ORDER)
        if os.path.isdir(path := os.path.join(videos_dir, name))
    }
    if stage_dirs and combined is None:
        # only usable if those per-stage files were replayed to RGB; ppo_stages_fast.py's own
        # state-only recordings live here too, and the combined file is the one that has images
        sources = [(name, find_rgb_h5(path), None) for name, path in stage_dirs.items()]
        found = [source for source in sources if source[1] is not None and has_rgb_obs(source[1])]
        if found:
            return found
        if sources:
            print(f"per-stage directories {list(stage_dirs)} hold no RGB observations, "
                  "slicing the combined trajectory file instead")

    if combined is None:
        combined = find_rgb_h5(videos_dir)
    if combined is None:
        raise SystemExit(f"no trajectory .h5 found in {videos_dir} (pass one with --h5)")

    if summary is None:
        name = args.stage or "all"
        print(f"no collection_summary.json found; treating the whole file as stage '{name}'")
        return [(name, combined, None)]

    stages = summary.get("stages", {})
    wanted = args.stages or [name for name in STAGE_ORDER if name in stages] or list(stages)
    missing = [name for name in wanted if name not in stages]
    if missing:
        raise SystemExit(f"summary has no stage(s) {missing}; it has {list(stages)}")
    return [(name, combined, stages[name].get("episode_range")) for name in wanted]


def pick_camera(traj_group, requested):
    """Name of the camera to read, validated against what this trajectory actually stores."""
    if "obs" not in traj_group or not isinstance(traj_group["obs"], h5py.Group):
        raise SystemExit(
            "this trajectory has no image observations (state-only recording). Replay it to RGB "
            "first with tools/convert_trajectories.sh, then point this script at the "
            "*.rgb.*.h5 it writes."
        )
    sensors = traj_group["obs"].get("sensor_data")
    cameras = [name for name in (sensors or {}) if "rgb" in sensors[name]]
    if not cameras:
        raise SystemExit(f"no obs/sensor_data/*/rgb in {traj_group.name}")
    if requested is None:
        if len(cameras) > 1:
            print(f"cameras {cameras} available, using '{cameras[0]}' (override with --camera)")
        return cameras[0]
    if requested not in cameras:
        raise SystemExit(f"camera '{requested}' not in {cameras}")
    return requested


def select_trajectories(available, rows, how, seed):
    """`rows` (or fewer) episode indices drawn from `available` by strategy `how`."""
    if len(available) <= rows:
        return available
    if how == "first":
        return available[:rows]
    if how == "random":
        rng = np.random.default_rng(seed)
        return sorted(rng.choice(available, size=rows, replace=False).tolist())
    # spread: evenly over the stage, so early and late episodes are both represented
    positions = np.linspace(0, len(available) - 1, rows).round().astype(int)
    return [available[position] for position in positions]


def read_row(traj_group, camera, cols):
    """`cols` frames spread over the episode.

    Returns (frames, frame indices, success flag per shown frame, success anywhere).
    """
    rgb = traj_group[f"obs/sensor_data/{camera}/rgb"]
    num_frames = rgb.shape[0]
    indices = np.linspace(0, num_frames - 1, cols).round().astype(int)
    # h5py wants a strictly increasing selection; short episodes repeat frames, so read
    # the distinct ones and fan them back out
    unique, inverse = np.unique(indices, return_inverse=True)
    frames = rgb[unique.tolist()][inverse]

    # `success` is per env step, so obs frame f follows step f-1; frame 0 predates any step
    success = np.zeros(num_frames, dtype=bool)
    if "success" in traj_group:
        steps = np.asarray(traj_group["success"])
        success[1:1 + len(steps)] = steps[:num_frames - 1]
    return frames, indices, success[indices], bool(success.any())


def render_stage(h5_path, stage, episode_range, args, env_id):
    """Writes one `rows x cols` grid PNG for `stage`. Returns the output path."""
    with h5py.File(h5_path, "r") as handle:
        keys = set(handle.keys())
        if episode_range is None:
            episodes = sorted(int(key.split("_")[1]) for key in keys if key.startswith("traj_"))
        else:
            start, end = episode_range
            episodes = [index for index in range(start, end) if f"traj_{index}" in keys]
        if not episodes:
            print(f"  {stage}: no episodes in {h5_path}, skipping")
            return None
        num_episodes = len(episodes)
        span = f"traj_{episodes[0]}..traj_{episodes[-1]}"
        print(f"  {stage}: {num_episodes} episodes ({span})")

        chosen = select_trajectories(episodes, args.rows, args.select, args.seed)
        camera = pick_camera(handle[f"traj_{chosen[0]}"], args.camera)
        rows = [(index, *read_row(handle[f"traj_{index}"], camera, args.cols)) for index in chosen]

    # margins in inches, so every tile stays square whatever the grid shape
    left_in, right_in, header_in, footer_in = 0.5, 0.08, 0.95, 0.08
    num_rows, num_cols = len(rows), args.cols
    width = num_cols * args.tile_size + left_in + right_in
    height = num_rows * args.tile_size + header_in + footer_in
    figure, axes = plt.subplots(
        num_rows, num_cols, squeeze=False, figsize=(width, height),
        gridspec_kw=dict(wspace=0.03, hspace=0.03),
    )
    for row, (index, frames, steps, success, succeeded) in enumerate(rows):
        for col in range(num_cols):
            axis = axes[row][col]
            axis.imshow(frames[col])
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color("#2e9e4f" if success[col] else "#cccccc")
                spine.set_linewidth(1.8 if success[col] else 0.5)
            if row == 0:
                axis.set_title(f"t={steps[col]}", fontsize=7, pad=3)
        axes[row][0].set_ylabel(f"traj {index}\n{'success' if succeeded else 'fail'}", fontsize=7)

    title_size = float(np.clip(width, 8.0, 13.0))
    figure.suptitle(f"{env_id} - {stage} ({num_episodes} episodes)", fontsize=title_size, y=1 - 0.25 / height)
    figure.text(
        0.5, 1 - 0.58 / height,
        f"{num_rows} trajectories x {num_cols} frames | {camera} rgb | "
        "green outline = task success at that step",
        ha="center", va="center", fontsize=title_size - 2.5, color="#555555",
    )
    figure.subplots_adjust(
        left=left_in / width, right=1 - right_in / width,
        top=1 - header_in / height, bottom=footer_in / height,
    )

    out_path = f"{args.out_prefix}-{stage}.png"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    figure.savefig(out_path, dpi=args.dpi)
    plt.close(figure)
    print(f"  wrote {out_path}")
    return out_path


def main():
    args = parse_args()
    videos_dir, combined, summary = resolve_run(args.run, args.h5, args.summary)
    stages = resolve_stages(args, videos_dir, combined, summary)

    run_name = os.path.basename(os.path.dirname(videos_dir) if os.path.basename(videos_dir) == "videos" else videos_dir)
    if args.out_prefix is None:
        args.out_prefix = os.path.join(".", run_name)
    env_id = (summary or {}).get("env_id", run_name)

    for stage, h5_path, episode_range in stages:
        print(f"{stage} <- {h5_path}")
        render_stage(h5_path, stage, episode_range, args, env_id)


if __name__ == "__main__":
    main()
