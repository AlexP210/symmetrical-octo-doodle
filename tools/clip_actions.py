"""
Clip a recording's stored actions to the env's action box, in place.

Why
---
`tools/ppo_stages_fast.py` stores the PPO policy's raw Gaussian samples, and
`replay_trajectory.py` copies them through verbatim. The controller does not: with
`normalize_action=True` (which `pd_ee_delta_pos` has), `BaseController.set_action` runs
`_clip_and_scale_action`, so anything outside the box is clipped before the sim sees it.
Measured on PushCube-v1.1-ppo-visual: the recorded actions span roughly [-7.5, 5.5], 16.8% of
components sit outside [-1, 1], and 45% of transitions have at least one. Every one of those is
a transition labelled with an action that did not produce it -- verified by stepping a restored
state with [5,0,0,0] and [1,0,0,0] and getting bit-identical results.

Online training never sees this, since `OnlineTrainer.generate_subtrajectory` stores the
already-clamped tensor it handed to `env.step`. It is an offline-only defect.

Note what clipping does *not* fix: because clip is a deterministic function, `a -> z'` was
already a well-defined mapping (`z' = env(z, clip(a))`), so a model with enough capacity could
learn the correct in-box dynamics from the unclipped labels. What this buys is a smooth target
over the region the planner actually queries instead of a saturating one spread over a domain 7x
wider, plus a correct `OfflineTaskBase._limits_from_dataset` for env-less tasks.

In place
--------
HDF5 datasets have fixed shape and dtype, and this writes the same shape and dtype back, so only
the `actions` bytes change -- a few MB against a 155 GB file. Nothing is relaid out, no dataset
moves, and the file size is unchanged, so the memmap offsets `lib/maniskill_transition_dataset.py`
relies on (`_view_dataset`) stay valid.

Two consequences of the size staying the same, both worth knowing:

  * `stage_to_slurm_tmpdir` reuses an already-staged copy when its size matches, so **a copy
    staged before this ran will silently keep the unclipped actions**. Delete it, or let the
    run stage fresh.
  * Nothing else records that the file changed, so this writes provenance attributes on the
    root group and refuses to run twice without `--force`.

Do not run this while a training job has the file open. HDF5 without SWMR gives no guarantees
to a concurrent reader.

Usage
-----
Reports what it would do and exits (default -- this mutates a recording, so writing is opt-in):

    python tools/clip_actions.py $DATA_DIR/maniskill/<task>/demos/trajectory.dino16.mmap.frac1.0.h5

Actually write, keeping a sidecar of the originals next to the file:

    python tools/clip_actions.py <path.h5> --apply

Undo:

    python tools/clip_actions.py <path.h5> --restore

Several files at once (the dino-augmented copy and the recording it came from carry the same
actions, and both want clipping if both are read):

    python tools/clip_actions.py <a.h5> <b.h5> --apply
"""

import argparse
import datetime
import os
import sys

import h5py
import numpy as np
from tqdm import tqdm

CLIPPED_LOW_ATTR = "actions_clipped_low"
CLIPPED_HIGH_ATTR = "actions_clipped_high"
CLIPPED_AT_ATTR = "actions_clipped_at"
BACKUP_SUFFIX = ".actions_raw.npz"


def action_dataset_paths(handle: h5py.File) -> list:
    """Every `.../actions` dataset in the file, in a stable order.

    Found by walking rather than by assuming `traj_<episode_id>/actions`, so a recording that
    nests its episodes differently still works. `visititems` skips soft/external links, which
    is what we want -- following one would clip the same data twice.
    """
    found = []

    def visit(name, node):
        if isinstance(node, h5py.Dataset) and name.rsplit("/", 1)[-1] == "actions":
            found.append(name)

    handle.visititems(visit)
    return sorted(found)


def describe(actions: np.ndarray, low: float, high: float) -> dict:
    """Per-dataset tallies, kept as sums so they add straight across datasets."""
    flat = actions.reshape(-1, actions.shape[-1])
    outside = (flat < low) | (flat > high)
    overshoot = np.maximum(flat - high, low - flat)
    return {
        "transitions": flat.shape[0],
        "components": flat.size,
        "outside_components": int(outside.sum()),
        "outside_transitions": int(outside.any(axis=1).sum()),
        "outside_per_dim": outside.sum(axis=0).astype(np.int64),
        "overshoot_sum": float(overshoot[outside].sum()),
        "min": float(flat.min()) if flat.size else float("inf"),
        "max": float(flat.max()) if flat.size else float("-inf"),
    }


def merge(total: dict, part: dict) -> dict:
    if not total:
        return dict(part)
    for key in ("transitions", "components", "outside_components", "outside_transitions",
                "overshoot_sum", "outside_per_dim"):
        total[key] = total[key] + part[key]
    total["min"] = min(total["min"], part["min"])
    total["max"] = max(total["max"], part["max"])
    return total


def report(label: str, total: dict) -> None:
    components = max(total["components"], 1)
    transitions = max(total["transitions"], 1)
    outside = total["outside_components"]
    per_dim = total["outside_per_dim"] / transitions
    print(f"  {label}")
    print(f"    transitions                 {total['transitions']:,}")
    print(f"    range                       [{total['min']:.4f}, {total['max']:.4f}]")
    print(f"    components outside the box  {outside:,}/{components:,} ({outside / components:.2%})")
    print(f"    transitions with >=1        {total['outside_transitions']:,}/{total['transitions']:,} "
          f"({total['outside_transitions'] / transitions:.2%})")
    if outside:
        print(f"    mean overshoot when outside {total['overshoot_sum'] / outside:.4f}")
    print(f"    per-component outside rate  {np.array2string(per_dim, precision=4, suppress_small=True)}")


def backup_path(h5_path: str, backup_dir: str = None) -> str:
    name = os.path.basename(h5_path) + BACKUP_SUFFIX
    return os.path.join(backup_dir or os.path.dirname(os.path.abspath(h5_path)), name)


def write_backup(path: str, names: list, arrays: list) -> None:
    """One concatenated array plus a length index, rather than 10k arrays in a zip.

    A recording here holds ~10k episodes; `np.savez` with an entry each spends most of its time
    on zip member overhead, and the whole thing is only a few MB concatenated.
    """
    np.savez(
        path,
        names=np.array(names),  # fixed-width unicode, so the archive needs no pickle
        lengths=np.array([len(a) for a in arrays], dtype=np.int64),
        data=np.concatenate(arrays, axis=0),
    )


def read_backup(path: str):
    with np.load(path) as loaded:
        names = [str(name) for name in loaded["names"]]
        offsets = np.concatenate([[0], np.cumsum(loaded["lengths"])])
        data = loaded["data"]
    return names, offsets, data


def processes_holding(h5_path: str) -> list:
    """`(pid, command)` for every other process with this file open or mapped.

    An in-place write is only safe when nothing else is reading: HDF5 without SWMR makes no
    promises to a concurrent reader, and a memory-mapped reader (which is what `load=False`
    training and `debug.py` both are) shares the very pages being overwritten. Read straight
    out of /proc rather than shelling out to lsof, which is not installed everywhere. Processes
    whose /proc entries we cannot read (another user's) are invisible here, so this reduces the
    risk rather than eliminating it.
    """
    target = os.path.realpath(h5_path)
    holders = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) == os.getpid():
            continue
        try:
            descriptors = os.listdir(f"/proc/{entry}/fd")
        except OSError:
            continue
        hit = False
        for descriptor in descriptors:
            try:
                if os.path.realpath(f"/proc/{entry}/fd/{descriptor}") == target:
                    hit = True
                    break
            except OSError:
                continue
        if not hit:
            try:
                with open(f"/proc/{entry}/maps") as maps:
                    hit = any(target in line for line in maps)
            except OSError:
                continue
        if hit:
            try:
                with open(f"/proc/{entry}/cmdline", "rb") as cmdline:
                    command = cmdline.read().replace(b"\0", b" ").decode(errors="replace").strip()
            except OSError:
                command = "?"
            holders.append((int(entry), command))
    return holders


def already_clipped(handle: h5py.File):
    if CLIPPED_LOW_ATTR not in handle.attrs:
        return None
    return (
        float(handle.attrs[CLIPPED_LOW_ATTR]),
        float(handle.attrs[CLIPPED_HIGH_ATTR]),
        str(handle.attrs.get(CLIPPED_AT_ATTR, "unknown time")),
    )


def process(h5_path: str, args) -> int:
    print(f"\n{h5_path}")
    if not os.path.exists(h5_path):
        print("  missing; skipped")
        return 1

    if args.apply or args.restore:
        holders = processes_holding(h5_path)
        if holders:
            print(f"  {len(holders)} other process(es) have this file open or mapped:")
            for pid, command in holders:
                print(f"    pid {pid}: {command[:150]}")
            if not args.force:
                print("  refusing to write underneath a live reader. Stop them (or pass --force "
                      "if you are certain they are finished with it).")
                return 1
            print("  --force given; writing anyway.")

    mode = "r+" if (args.apply or args.restore) else "r"
    with h5py.File(h5_path, mode) as handle:
        paths = action_dataset_paths(handle)
        if not paths:
            print("  no `actions` datasets found; skipped")
            return 1
        print(f"  {len(paths)} action datasets")

        if args.restore:
            return restore(handle, h5_path, paths, args)

        stamp = already_clipped(handle)
        if stamp and not args.force:
            print(f"  already clipped to [{stamp[0]}, {stamp[1]}] at {stamp[2]}; "
                  "pass --force to clip again (it is idempotent, so this is normally unnecessary)")
            return 0

        contiguous = sum(1 for p in paths if handle[p].id.get_offset() is not None)
        if contiguous != len(paths):
            print(f"  note: {len(paths) - contiguous} of {len(paths)} are chunked/compressed. "
                  "An in-place write of the same shape is still safe, but may churn storage.")

        before, arrays, names = {}, [], []
        for path in tqdm(paths, desc="  reading", leave=False):
            actions = handle[path][:]
            before = merge(before, describe(actions, args.low, args.high))
            arrays.append(actions)
            names.append(path)
        report("before", before)

        if not args.apply:
            print(f"\n  dry run; nothing written. Re-run with --apply to clip to "
                  f"[{args.low}, {args.high}] in place.")
            return 0

        if not args.no_backup:
            path = backup_path(h5_path, args.backup_dir)
            if os.path.exists(path) and not args.force:
                print(f"  backup already exists at {path}; refusing to overwrite it "
                      "(it may hold the only copy of the original actions). Pass --force to replace.")
                return 1
            write_backup(path, names, arrays)
            print(f"  wrote backup {path} ({os.path.getsize(path) / 1e6:.1f} MB)")

        after = {}
        for path, actions in zip(tqdm(names, desc="  clipping", leave=False), arrays):
            clipped = np.clip(actions, args.low, args.high).astype(actions.dtype, copy=False)
            # Same shape, same dtype: this overwrites the dataset's existing bytes rather than
            # reallocating, which is what keeps the operation cheap and the file's layout intact.
            handle[path][...] = clipped
            after = merge(after, describe(clipped, args.low, args.high))

        handle.attrs[CLIPPED_LOW_ATTR] = args.low
        handle.attrs[CLIPPED_HIGH_ATTR] = args.high
        handle.attrs[CLIPPED_AT_ATTR] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        handle.flush()
        report("after", after)

        if after["outside_components"]:
            print("  ERROR: components still outside the box after clipping")
            return 1

    # Reopened rather than checked through the same handle, so the verification reads what
    # actually landed on disk instead of anything h5py still holds in its cache.
    with h5py.File(h5_path, "r") as handle:
        worst = 0.0
        for path in paths:
            values = handle[path][:]
            if values.size:
                worst = max(worst, abs(float(values.min())), abs(float(values.max())))
        bound = max(abs(args.low), abs(args.high))
        print(f"  verified on reread: max |action| = {worst:.6f} (bound {bound})")
        if worst > bound + 1e-6:
            print("  ERROR: reread found values outside the box")
            return 1
    return 0


def restore(handle: h5py.File, h5_path: str, paths: list, args) -> int:
    path = backup_path(h5_path, args.backup_dir)
    if not os.path.exists(path):
        print(f"  no backup at {path}; cannot restore")
        return 1

    names, offsets, data = read_backup(path)
    known = dict(zip(names, range(len(names))))
    missing = [p for p in paths if p not in known]
    if missing:
        print(f"  backup does not cover {len(missing)} datasets (e.g. {missing[0]}); refusing to restore")
        return 1

    for dataset_path in tqdm(paths, desc="  restoring", leave=False):
        index = known[dataset_path]
        original = data[offsets[index]:offsets[index + 1]]
        if original.shape != handle[dataset_path].shape:
            print(f"  shape mismatch for {dataset_path}; refusing to restore")
            return 1
        handle[dataset_path][...] = original

    for attr in (CLIPPED_LOW_ATTR, CLIPPED_HIGH_ATTR, CLIPPED_AT_ATTR):
        if attr in handle.attrs:
            del handle.attrs[attr]
    handle.flush()
    print(f"  restored {len(paths)} datasets from {path}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("paths", nargs="+", help="trajectory .h5 file(s) to clip")
    parser.add_argument("--low", type=float, default=-1.0,
                        help="lower bound; must match the env's action_space (default -1.0, which "
                             "is what normalize_action=True gives)")
    parser.add_argument("--high", type=float, default=1.0, help="upper bound (default 1.0)")
    parser.add_argument("--apply", action="store_true",
                        help="actually write. Without it this only reports what it would change.")
    parser.add_argument("--restore", action="store_true",
                        help="write the sidecar backup's actions back and drop the provenance attrs")
    parser.add_argument("--no-backup", action="store_true",
                        help="skip the sidecar of the original actions (it is a few MB; keep it)")
    parser.add_argument("--backup-dir", default=None,
                        help="where the sidecar goes (default: alongside the .h5)")
    parser.add_argument("--force", action="store_true",
                        help="re-clip a file already marked as clipped, and overwrite an existing backup")
    args = parser.parse_args()

    if args.restore and args.apply:
        parser.error("--restore and --apply are mutually exclusive")
    if args.low >= args.high:
        parser.error(f"--low ({args.low}) must be below --high ({args.high})")

    if args.apply:
        print("Writing in place. Make sure no training job has these files open -- HDF5 without "
              "SWMR makes no promises to a concurrent reader.\n"
              "The file size does not change, so `stage_to_slurm_tmpdir` will reuse an already-"
              "staged copy: delete any stale $SLURM_TMPDIR copy before the next run.")

    failures = sum(process(path, args) for path in args.paths)
    if failures:
        print(f"\n{failures} file(s) had problems")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
