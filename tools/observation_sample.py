"""Save observation views (RGB and/or DINOv3 features) for the offline
ManiSkill push-cube task.

With no arguments, the observation comes from resetting the live env
(task.env.reset()) and always includes "rgb", so all three outputs below are
produced:

  1. observation_sample_rgb.png: the raw RGB frame.
  2. observation_sample_rgb_resized.png: that frame resized to 224x224 with the
     same resize step DINOV3EncoderModel.make_transform applies before encoding
     (normalization is skipped since it isn't meaningfully visualizable).
  3. observation_sample_dino.png: a PCA-to-RGB false-color visualization of the
     DINOv3 patch features for that frame, computed on demand by running it
     through DINOV3EncoderModel (configs/model/frozen_dinov3_vits_patch_encoder.yaml).

With --dataset pointing to an offline ManiSkill trajectory .h5 file, the
observation instead comes from that dataset's first sample. Which outputs get
written then depends on what that file's dataset_structure actually contains
(see tools/preprocess_data_mmap.py, which strips raw rgb to save disk space):

  - If the sample has "rgb", all three outputs above are produced the same way
    (rgb/rgb_resized from the sample, dino computed on demand from it).
  - Otherwise, if it has precomputed "dino_patch_features", only
    observation_sample_dino.png is produced, directly from those (no rgb to
    save, no need to load the DINO backbone).

--dataset requires an mmap-compatible file (every array stored contiguous and
uncompressed), since the task config this script uses has load: False — see
tools/preprocess_data_mmap.py. A raw/gzip-compressed trajectory .h5 (ManiSkill's
default recording output) will raise a clear error rather than silently loading.

Usage:
    python tools/observation_sample.py
    python tools/observation_sample.py --dataset /path/to/trajectory.h5

Requirements:
    - s2p package on PYTHONPATH, e.g.:
        export PYTHONPATH=/path/to/project/agents/squeeze2plan:$PYTHONPATH
"""

import argparse
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Read by s2p.models.dinov3_encoder_model at import time to locate dependencies/dinov3.
os.environ.setdefault("PROJECT_ROOT", REPO_ROOT)

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import v2

from s2p.tasks.maniskill_task import get_or_create
from s2p.models.dinov3_encoder_model import DINOV3EncoderModel

DINO_UPSCALED_SIZE = 224
# Matches cfg.resize_size in configs/model/frozen_dinov3_vits_patch_encoder.yaml
RESIZE_SIZE = 224

RESIZE_TRANSFORM = v2.Compose([
    v2.ToImage(),
    v2.Resize((RESIZE_SIZE, RESIZE_SIZE), antialias=True),
])

TASK_CONFIG_PATH = os.path.join(
    REPO_ROOT, "agents", "s2p", "s2p", "configs", "task", "maniskill_push_cube.yaml",
)
MODEL_CONFIG_PATH = os.path.join(
    REPO_ROOT, "agents", "s2p", "s2p", "configs", "model", "frozen_dinov3_vits_patch_encoder.yaml",
)
RGB_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "observation_sample_rgb.png")
RGB_RESIZED_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "observation_sample_rgb_resized.png")
DINO_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "observation_sample_dino.png")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to an offline ManiSkill trajectory .h5 file. If omitted, the "
             "observation is sampled by resetting the live env instead.",
    )
    return parser.parse_args()


def pca_to_rgb(patches: np.ndarray) -> np.ndarray:
    """Projects (N, D) patch features onto their top-3 principal components,
    normalized per-channel to [0, 255] uint8."""
    centered = patches - patches.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = centered @ vt[:3].T  # (N, 3)
    components -= components.min(axis=0, keepdims=True)
    components /= components.max(axis=0, keepdims=True) + 1e-8
    return (components * 255).astype(np.uint8)


def build_cfgs():
    """Resolves the task's/model's ${data_dir}/${checkpoint_dir}/${device}
    interpolations the same way Hydra would when these configs are composed
    under the top-level config."""
    task_node = OmegaConf.load(TASK_CONFIG_PATH)
    model_node = OmegaConf.load(MODEL_CONFIG_PATH)
    root = OmegaConf.create({
        "data_dir": os.environ.get("DATA_DIR", "/path/to/datasets"),
        "checkpoint_dir": os.environ.get("CHECKPOINT_DIR", "/path/to/pretrained_checkpoints"),
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "task": task_node,
        "model": model_node,
    })
    OmegaConf.resolve(root)
    return root.task.cfg, root.model.cfg


def get_observation(task, dataset_path):
    """Returns a dict with whichever of "rgb" (a single most-recent (3, H, W)
    uint8 frame) and "dino_patch_features" (a single most-recent (N, D) float
    array) are available from the requested source."""
    if dataset_path is not None:
        print("HERE")
        sample_obs = task.training_dataset[0]["obs"]
        # Sample leaves are horizon-windowed (extra leading dim) and
        # frame-stacked; take the first horizon step here, most-recent frame below.
        stacked = {key: sample_obs[key][0] for key in sample_obs.keys()}
    else:
        # task.env.reset() obs are already just frame-stacked (no horizon dim).
        stacked = task.env.reset()

    num_frames = task.cfg.num_frames
    result = {}
    if "rgb" in stacked:
        # Frames are channel-concatenated oldest-first, so the last 3 channels
        # are the most recent one (see custom_maniskill_tasks.FrameStack).
        result["rgb"] = stacked["rgb"][-3:]
    if "dino_patch_features" in stacked:
        # Frames are patch-concatenated oldest-first, so the last N patches
        # are the most recent one (see ManiSkillTrajectoryDataset._stack_obs_window).
        num_patches = stacked["dino_patch_features"].shape[0] // num_frames
        result["dino_patch_features"] = stacked["dino_patch_features"][-num_patches:]
    return result


def save_rgb_pngs(frame):
    image = frame.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    Image.fromarray(image).save(RGB_OUTPUT_PATH)
    print(f"Saved observation sample to {RGB_OUTPUT_PATH}")

    resized = RESIZE_TRANSFORM(frame)  # (3, RESIZE_SIZE, RESIZE_SIZE) uint8
    resized_image = resized.permute(1, 2, 0).cpu().numpy()
    Image.fromarray(resized_image).save(RGB_RESIZED_OUTPUT_PATH)
    print(f"Saved observation sample to {RGB_RESIZED_OUTPUT_PATH}")


def encode_patches(encoder, frame) -> np.ndarray:
    # encoder.encode expects a frame-stacked (S*3, H, W) input; a lone frame
    # (S=1) makes it apply its own resize+normalize and return that single
    # frame's (N, D) patch tokens directly (see DINOV3EncoderModel.encode).
    with torch.no_grad():
        patches = encoder.encode({"rgb": frame.to(encoder.cfg.device)})  # (N, D)
    return patches.cpu().numpy()


def save_dino_png(patches: np.ndarray):
    num_patches = patches.shape[0]
    side = int(round(num_patches ** 0.5))
    assert side * side == num_patches, f"expected a square patch grid, got {num_patches} patches"
    image = pca_to_rgb(patches).reshape(side, side, 3)

    Image.fromarray(image).resize((DINO_UPSCALED_SIZE, DINO_UPSCALED_SIZE), Image.NEAREST).save(DINO_OUTPUT_PATH)
    print(f"Saved observation sample to {DINO_OUTPUT_PATH}")


def main():
    args = parse_args()
    task_cfg, model_cfg = build_cfgs()
    if args.dataset is not None:
        task_cfg.data_path = args.dataset
    task = get_or_create(task_cfg)

    obs = get_observation(task, args.dataset)

    if "rgb" in obs:
        frame = obs["rgb"]
        save_rgb_pngs(frame)
        encoder = DINOV3EncoderModel(model_cfg, task)
        save_dino_png(encode_patches(encoder, frame))
    elif "dino_patch_features" in obs:
        print("No 'rgb' observation available; skipping rgb/rgb_resized outputs.")
        save_dino_png(obs["dino_patch_features"].cpu().numpy())
    else:
        raise RuntimeError(
            "Observation has neither 'rgb' nor 'dino_patch_features' to visualize."
        )


if __name__ == "__main__":
    main()
