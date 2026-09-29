"""Copy a random fraction of ManiSkill trajectories into a new h5 file, applying either,
both, or neither of the two independent training-speed optimizations.

Supersedes preprocess_data_mmap.py and OLD_preprocess_data.py, which each hardcoded one
fixed combination (both, and DINO-only-gzipped, respectively). The two optimizations
address different bottlenecks and have very different disk costs, so which to apply is a
real decision — hence the flags:

--memmap
    Write every dataset *contiguous and uncompressed* (no `compression=`, no `chunks=`).
    Contiguous storage means each dataset lives at one fixed byte offset, which is what
    lets the dataset loaders build a zero-copy np.memmap view onto it; chunked and/or
    compressed datasets have no fixed offset and fall back to reading through h5py.
    Measured on PushCube: ~20.7 ms/sample through the h5py fallback versus ~0.02 ms
    through a warm memmap, for the same strided 4-frame read.

    Costs disk: the source rgb is gzipped, so dropping compression inflates it.

--dino / --dino-fp16
    Precompute DINOv3 patch features so training never runs the backbone. Saves GPU time
    on the critical path (~1.6 ms/sample for ViT-S/16 at batch 32), which prefetching
    cannot hide the way it hides dataloader I/O.

    Costs disk, and this is the expensive one: fp32 features are ~2x the bytes of the rgb
    they describe. --dino-fp16 halves that, which is free at training time because the
    loaders already do `.astype(np.float32)` on read. Patch features are activations, not
    weights, so fp16 is ample precision.

Neither flag changes the output *schema* — the two are orthogonal and compose. With
neither, this only subsets the trajectories.

Usage:
    # both optimizations, features in fp16 (the usual choice)
    python preprocess_data.py <traj.h5> --fraction 0.45 --memmap --dino-fp16 \\
        --camera base_camera --checkpoint /path/to/dinov3_vits16_pretrain.pth

    # memmap only — no GPU or checkpoint needed, ~3x cheaper on disk than adding fp32 DINO
    python preprocess_data.py <traj.h5> --fraction 0.9 --memmap

    # size the job without writing anything
    python preprocess_data.py <traj.h5> --fraction 0.45 --memmap --dino-fp16 --dry-run

Requirements (only when a --dino flag is given):
    - s2p package on PYTHONPATH, e.g.
        export PYTHONPATH=/path/to/project/agents/squeeze2plan:$PYTHONPATH
    - dinov3 dependency available at dependencies/dinov3 relative to cwd, or adjust the
      torch.hub source path in DINOV3EncoderModel accordingly.
"""

import argparse
import json
import os
import random
import shutil

import h5py
import numpy as np
from tqdm import tqdm


class _Task:
    """Minimal task stub — provides observation_dimension for DINOV3EncoderModel."""
    def __init__(self, observation_dimension):
        self.observation_dimension = observation_dimension  # (C, H, W)


def build_encoder(device: str, image_hw: tuple, checkpoint: str = None):
    # Imported here rather than at module scope so that a --memmap-only run needs
    # neither torch, the s2p package, nor the dinov3 dependency.
    import torch  # noqa: F401  (imported for its side effect on DINOV3EncoderModel)
    from omegaconf import OmegaConf
    from s2p.models.dinov3_encoder_model import DINOV3EncoderModel

    cfg = OmegaConf.create({
        "resize_size": 224,
        "model_name": "dinov3_vits16",  # TODO: change if a different backbone is desired
        "token_mode": "patch",
        "freeze": True,
        "checkpoint": checkpoint,
        "device": device,
        "observation_key": None,
        "eval_mode": True
    })
    task = _Task(observation_dimension=(1, 3, *image_hw))
    encoder = DINOV3EncoderModel(cfg, task)
    encoder.eval()
    return encoder


def encode_rgb(encoder, rgb: np.ndarray, device: str, batch_size: int, dtype) -> np.ndarray:
    """Encode a sequence of RGB frames into DINO patch features.

    Args:
        rgb: (T, H, W, 3) uint8 numpy array.
        dtype: numpy dtype the features are stored as (np.float32 or np.float16).

    Returns:
        (T, N, D) array of `dtype`, where N = (resize_size // 16) ** 2 = 196 for vits16.
    """
    import torch

    # HWC -> CHW as uint8; the encoder's internal transform handles resizing, float
    # conversion, and normalisation.
    x = torch.from_numpy(rgb).permute(0, 3, 1, 2).to(device)  # (T, 3, H, W)

    # encoder.encode() expects (..., S*3, H, W); with observation_dimension=(3, H, W)
    # batch_dims=(T,) and S=1, so (T, 3, H, W) is correct.
    feats = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            f = encoder.encode(x[start : start + batch_size])  # (B, N, D)
            feats.append(f.cpu().numpy())
    # Cast after encoding, never during: the backbone runs in its own precision and only
    # the stored copy is narrowed.
    return np.concatenate(feats, axis=0).astype(dtype, copy=False)


def _layout_kwargs(item: h5py.Dataset, memmap: bool) -> dict:
    """create_dataset kwargs reproducing `item`'s storage layout, or a mappable one.

    With --memmap, no compression/chunks kwarg is passed at all, which is what makes h5py
    store the dataset contiguously at a fixed offset. Without it the source's own layout
    is mirrored, so a gzipped input stays gzipped rather than silently inflating.
    """
    if memmap:
        return {}
    kwargs = {}
    if item.compression is not None:
        kwargs["compression"] = item.compression
        if item.compression_opts is not None:
            kwargs["compression_opts"] = item.compression_opts
    if item.chunks is not None:
        kwargs["chunks"] = item.chunks
    return kwargs


def copy_traj(
    src_traj: h5py.Group,
    dst_parent: h5py.Group,
    traj_key: str,
    memmap: bool,
    save_images: bool = True,
) -> h5py.Group:
    """Copy a trajectory group into dst_parent.

    Args:
        memmap: If True, every dataset is written contiguous and uncompressed regardless
            of the source's layout. If False, each dataset's source layout is preserved.
        save_images: If False, raw camera ``rgb`` datasets are skipped (DINO features
            added by the caller are kept regardless).
    """
    def _copy(src_group, dst_group):
        for name, item in src_group.items():
            if isinstance(item, h5py.Group):
                dst_child = dst_group.create_group(name)
                dst_child.attrs.update(item.attrs)
                _copy(item, dst_child)
            else:
                if name == "rgb" and not save_images:
                    continue
                data = item[()]
                # Raw camera images are stored as uint8; enforce it so the output layout
                # is stable regardless of the source's rgb dtype.
                if name == "rgb":
                    data = data.astype(np.uint8, copy=False)
                dst_group.create_dataset(name, data=data, **_layout_kwargs(item, memmap))
                dst_group[name].attrs.update(item.attrs)

    dst_traj = dst_parent.create_group(traj_key)
    dst_traj.attrs.update(src_traj.attrs)
    _copy(src_traj, dst_traj)
    return dst_traj


def derive_output_path(input_path: str, fraction: float, memmap: bool, dino_dtype) -> str:
    """Encode the applied optimizations in the filename, matching the existing convention
    (`....dino.mmap.frac0.7.h5`) so a file's provenance is readable off its name."""
    stem, ext = os.path.splitext(input_path)
    parts = []
    if dino_dtype is not None:
        parts.append("dino16" if dino_dtype == np.float16 else "dino")
    if memmap:
        parts.append("mmap")
    parts.append(f"frac{fraction}")
    return f"{stem}.{'.'.join(parts)}{ext}"


def estimate_output_bytes(
    src: h5py.File,
    sample_key: str,
    n_select: int,
    camera: str,
    memmap: bool,
    save_images: bool,
    dino_dtype,
    src_path: str,
    n_total: int,
) -> int:
    """Rough output size, from one trajectory scaled by the selection count.

    Exact for --memmap (nothing is compressed, so uncompressed nbytes is the size on
    disk). For a compression-preserving run the copied datasets are estimated from the
    source file's actual on-disk size instead, since their compressed size is not
    predictable from shapes.
    """
    traj = src[sample_key]

    copied = 0
    def _walk(group):
        nonlocal copied
        for name, item in group.items():
            if isinstance(item, h5py.Group):
                _walk(item)
            elif not (name == "rgb" and not save_images):
                copied += item.size * item.dtype.itemsize
    _walk(traj)

    if memmap:
        per_traj = copied
    else:
        # Fall back to measured on-disk bytes per episode; if rgb is being dropped its
        # uncompressed share cannot simply be subtracted, so this over-estimates then.
        per_traj = os.path.getsize(src_path) / max(n_total, 1)

    if dino_dtype is not None:
        T, H, W, _ = traj[f"obs/sensor_data/{camera}/rgb"].shape
        n_patches = (224 // 16) ** 2
        per_traj += T * n_patches * 384 * np.dtype(dino_dtype).itemsize

    return int(per_traj * n_select)


def _fmt(nbytes: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(nbytes) < 1024 or unit == "TiB":
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024


def main():
    parser = argparse.ArgumentParser(
        description="Copy a fraction of ManiSkill trajectories, optionally memory-mappable "
                    "and/or augmented with DINO patch features.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("traj_path", help="Path to the source trajectory .h5 file")
    parser.add_argument(
        "--fraction", type=float, required=True,
        help="Fraction of trajectories to select (e.g. 0.1 for 10%%)",
    )

    parser.add_argument(
        "--memmap", action="store_true",
        help="Write every dataset contiguous and uncompressed so it can be memory-mapped "
             "at training time. Without this the source's compression/chunking is preserved.",
    )
    parser.add_argument(
        "--dino", action="store_true",
        help="Add precomputed DINOv3 patch features, stored as float32.",
    )
    parser.add_argument(
        "--dino-fp16", action="store_true",
        help="Add precomputed DINOv3 patch features, stored as float16 (half the disk of "
             "--dino; the loaders cast to float32 on read, so training is unaffected). "
             "Wins if combined with --dino.",
    )

    parser.add_argument("--output", default=None, help="Output .h5 path. Defaults to a name "
                                                      "encoding the applied optimizations.")
    parser.add_argument("--checkpoint", default=None, help="Path to the DINO checkpoint .pth "
                                                          "(required by --dino/--dino-fp16)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=64, dest="batch_size",
                        help="Frames per encoder forward pass")
    parser.add_argument("--camera", default="base_camera",
                        help="Which camera's rgb to encode. Features are written to "
                             "obs/sensor_data/<camera>/dino_patch_features, which must match "
                             "the DINO_KEY the dataset loader expects.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for trajectory selection")
    parser.add_argument(
        "--save-images", action=argparse.BooleanOptionalAction, default=True,
        help="Copy raw rgb alongside the features (--no-save-images stores only features, "
             "which leaves nothing for a decoder to reconstruct)",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Report the selection and estimated output size, then exit")
    parser.add_argument("--allow-overfill", action="store_true",
                        help="Proceed even when the estimate exceeds free space on the "
                             "output filesystem")
    args = parser.parse_args()

    assert 0.0 < args.fraction <= 1.0, "--fraction must be in (0, 1]"

    # --dino-fp16 is --dino at narrower precision, so it simply wins when both are given.
    dino_dtype = np.float16 if args.dino_fp16 else (np.float32 if args.dino else None)
    if dino_dtype is None and not args.memmap:
        print("Neither --memmap nor --dino/--dino-fp16 given: this run only subsets "
              "trajectories and applies no optimization.")
    if dino_dtype is None and not args.save_images:
        parser.error("--no-save-images without a --dino flag would write no visual data at all.")
    if dino_dtype is not None and args.checkpoint is None:
        print("Warning: no --checkpoint given; DINOV3EncoderModel will fall back to whatever "
              "default its config carries.")

    output_path = args.output or derive_output_path(
        args.traj_path, args.fraction, args.memmap, dino_dtype
    )
    output_json_path = os.path.splitext(output_path)[0] + ".json"

    src_json_path = os.path.splitext(args.traj_path)[0] + ".json"
    with open(src_json_path) as f:
        json_data = json.load(f)

    all_episodes = json_data["episodes"]
    n_select = max(1, round(len(all_episodes) * args.fraction))
    rng = random.Random(args.seed)
    selected_episodes = all_episodes if args.fraction == 1.0 else rng.sample(all_episodes, n_select)

    applied = [n for n, on in (("memmap", args.memmap),
                               ("dino", dino_dtype is not None)) if on] or ["none"]
    print(f"Optimizations: {', '.join(applied)}"
          + (f" (features as {np.dtype(dino_dtype).name})" if dino_dtype is not None else ""))
    print(f"Selected {n_select}/{len(all_episodes)} trajectories")

    with h5py.File(args.traj_path, "r") as src:
        sample_key = f"traj_{selected_episodes[0]['episode_id']}"
        rgb_path = f"{sample_key}/obs/sensor_data/{args.camera}/rgb"
        if rgb_path not in src:
            cams = list(src[f"{sample_key}/obs/sensor_data"].keys())
            parser.error(f"No rgb for camera {args.camera!r}; file has: {cams}")
        _, H, W, _ = src[rgb_path].shape

        estimate = estimate_output_bytes(
            src, sample_key, n_select, args.camera, args.memmap, args.save_images,
            dino_dtype, args.traj_path, len(all_episodes),
        )
        free = shutil.disk_usage(os.path.dirname(os.path.abspath(output_path))).free
        print(f"Estimated output: {_fmt(estimate)} "
              f"({_fmt(estimate / n_select)}/episode) | free: {_fmt(free)}")
        if dino_dtype is not None:
            print(f"Features -> obs/sensor_data/{args.camera}/dino_patch_features "
                  f"(the loader's DINO_KEY must match this exactly, or it will silently "
                  f"report no features and re-run the backbone every batch)")
        if estimate > free and not args.allow_overfill:
            parser.error(
                f"Estimate ({_fmt(estimate)}) exceeds free space ({_fmt(free)}). Lower "
                f"--fraction to about {args.fraction * free / estimate:.2f}, use "
                f"--dino-fp16 instead of --dino, drop --dino, or pass --allow-overfill."
            )
        if args.dry_run:
            print("--dry-run: nothing written.")
            return

        encoder = build_encoder(args.device, (H, W), args.checkpoint) if dino_dtype is not None else None

        with h5py.File(output_path, "w") as dst:
            for ep in tqdm(selected_episodes, desc="Copying" + (" & encoding" if encoder else "")):
                traj_key = f"traj_{ep['episode_id']}"
                features = None
                if encoder is not None:
                    rgb = src[f"{traj_key}/obs/sensor_data/{args.camera}/rgb"][:]  # (T,H,W,3)
                    features = encode_rgb(encoder, rgb, args.device, args.batch_size, dino_dtype)

                dst_traj = copy_traj(src[traj_key], dst, traj_key,
                                     memmap=args.memmap, save_images=args.save_images)
                if features is not None:
                    # Mirror the run's layout choice: contiguous under --memmap, else
                    # gzipped like the rgb it was derived from.
                    kwargs = {} if args.memmap else {"compression": "gzip"}
                    dst_traj.create_dataset(
                        f"obs/sensor_data/{args.camera}/dino_patch_features",
                        data=features, **kwargs,
                    )

    with open(output_json_path, "w") as f:
        json.dump({
            "env_info": json_data["env_info"],
            "commit_info": json_data.get("commit_info", {}),
            "episodes": selected_episodes,
        }, f)

    print(f"Wrote {output_path} ({_fmt(os.path.getsize(output_path))})")
    print(f"Wrote {output_json_path}")


if __name__ == "__main__":
    main()
