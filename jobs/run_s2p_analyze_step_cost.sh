#!/bin/bash

# This script launches analysis/analyze_step_cost.py with command line arguments. It takes
# care of providing the dataset, checkpoint, and output dirs based on whichever machine/*.env
# file is sourced.
#
# Measures what a real_task_planning eval costs besides planning: the per-env-step wall-clock
# of RealTaskEvaluator's loop (env step, rendering, encoder update) at the evaluator's
# num_envs / max_episode_steps / save_video, and -- given the same grid as
# run_s2p_analyze_latency_flops_grid.sh plus +batched_latency_repeats=<n> -- plan latency at
# that num_envs for every cell. Pass +timed_episodes=<n> for the step-cost measurement. See
# the module docstring in analyze_step_cost.py. feasible_planner_configs.py reads both back.
#
# Timings, like the latency grid: one GPU, otherwise idle, and the same GPU model the
# latency grid ran on (feasible_planner_configs.py only matches measurements on that GPU).
#
# Step cost goes to ${PROJECT_ROOT}/agents/squeeze2plan/analysis/results/step_cost/<run_name>.json;
# batched latency to the `batched_latency` stage of .../results/<run_name>.json -- so the
# PROJECT_ROOT bind has to stay writable (do not add :ro to it).

DEBUG=false
PASSTHROUGH=()
for arg in "$@"; do
    if [ "$arg" = "--debug" ] || [ "$arg" = "DEBUG=true" ]; then
        DEBUG=true
    else
        PASSTHROUGH+=("$arg")
    fi
done
set -- "${PASSTHROUGH[@]}"

if [ "$DEBUG" = true ]; then
    DEBUG_STRING="-m debugpy --listen 5678 --wait-for-client"
else
    DEBUG_STRING=""
fi

# Specialize the output dir
export OUTPUT_ROOT=${OUTPUT_DIR}          # keep the shared root bindable
export S2P_OUTPUT_DIR=${OUTPUT_DIR}/s2p
export WANDB_DIR=${S2P_OUTPUT_DIR}
export TORCH_HOME=${CHECKPOINT_DIR}

# On a machine whose APPTAINER_COPY_DIR is the project itself (e.g. a local workstation), the container is already
# where it will be read from and this copy is a no-op that only prints "cannot copy a directory into
# itself" -- confusing enough to look like the run failed. A cluster may copy to node-local storage instead.
CONTAINER_DIR="${APPTAINER_COPY_DIR}/containers"
if [ "$(readlink -f "${PROJECT_ROOT}/containers")" != "$(readlink -f "${CONTAINER_DIR}")" ]; then
	# rsync rather than cp, for four properties cp lacks here. The trailing slashes copy the *contents*
	# of containers/, so a rerun onto an existing directory can never nest into dst/containers/. rsync
	# writes to a temporary name and renames only on success, so an interrupted transfer never leaves a
	# truncated s2p.sif under the name we mount -- which previously survived every retry and failed as
	# "Something went wrong trying to read the squashfs image". Size+mtime comparison skips the
	# unchanged 8.9G SIF on reruns while still picking up files added or rebuilt since the last sync.
	# And --delete drops files no longer in the source instead of leaving them to be mounted by accident.
	echo "Syncing ${PROJECT_ROOT}/containers to ${CONTAINER_DIR}"
	mkdir -p "${CONTAINER_DIR}"
	if ! rsync -a --delete "${PROJECT_ROOT}/containers/" "${CONTAINER_DIR}/"; then
		echo "Container sync failed; ${CONTAINER_DIR} may be incomplete" >&2
		exit 1
	fi
fi

# The ManiSkill dataset loader stages the h5 onto node-local disk when it can see $SLURM_TMPDIR.
# Apptainer forwards the variable but not the mount, so bind it when we're in a job -- otherwise
# `maniskill_transition_dataset` warns that it cannot see the path and reads over the network
# instead. Guarded rather than bound unconditionally, as the other *_plan/*_train jobs do: off
# SLURM the variable is empty, and if it is set but the directory is not there, binding it is a
# fatal error rather than a warning.
TMPDIR_BIND=()
if [ -n "${SLURM_TMPDIR}" ] && [ -d "${SLURM_TMPDIR}" ]; then
	TMPDIR_BIND=(--bind "${SLURM_TMPDIR}:${SLURM_TMPDIR}")
fi

# Hosts that keep the driver in /usr/lib64/nvidia (RHEL-family hosts generally) need
# help; hosts that keep it in /usr/lib/x86_64-linux-gnu (Ubuntu generally) have no such
# directory at all, so everything below is conditional -- binding a path that does not exist is a
# fatal error in apptainer, not a warning.
#
# Where it does exist, every file under it is a symlink to ../<name> (i.e. to the real file in
# /usr/lib64 itself). Binding only /usr/lib64/nvidia leaves those symlinks dangling inside the
# container, since ".." resolves to the container's own /usr/lib64, not the host's. --nv separately
# auto-injects a curated set of well-known driver libs (libGLX_nvidia, libEGL_nvidia, ...) that
# mask this for most cases, but less common ones (e.g. libnvidia-glsi, needed by SAPIEN's Vulkan
# renderer) fall through and fail with "cannot open shared object file". Resolve each symlink's real
# target and bind it individually onto the same host path so it's reachable from inside
# /usr/lib64/nvidia too.
NVIDIA_LIB_BINDS=()
NVIDIA_LD_PATH=""
if [ -d /usr/lib64/nvidia ]; then
	NVIDIA_LIB_BINDS+=(--bind /usr/lib64/nvidia)
	NVIDIA_LD_PATH="/usr/lib64/nvidia:"
	mapfile -t NVIDIA_LIB_TARGETS < <(for f in /usr/lib64/nvidia/*; do readlink -f "$f"; done | sort -u)
	for target in "${NVIDIA_LIB_TARGETS[@]}"; do
		if [ -e "$target" ] && [[ "$target" != /usr/lib64/nvidia/* ]]; then
			NVIDIA_LIB_BINDS+=(--bind "${target}:${target}")
		fi
	done
fi

apptainer exec \
	--overlay ${APPTAINER_COPY_DIR}/containers/s2p_custom_dependencies.overlay:ro \
	--nv \
	--bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
	--bind ${DATA_DIR}:${DATA_DIR} \
	--bind ${CHECKPOINT_DIR}:${CHECKPOINT_DIR} \
	--bind ${OUTPUT_ROOT}:${OUTPUT_ROOT} \
	"${TMPDIR_BIND[@]}" \
	"${NVIDIA_LIB_BINDS[@]}" \
	${APPTAINER_COPY_DIR}/containers/s2p.sif \
	bash -c 'export LD_LIBRARY_PATH='"${NVIDIA_LD_PATH}"'$LD_LIBRARY_PATH && \
	python '"${DEBUG_STRING}"' '"${PROJECT_ROOT}"'/agents/squeeze2plan/analysis/analyze_step_cost.py \
		"$@" \
		hydra.run.dir='"${S2P_OUTPUT_DIR}"'/hydra \
		data_dir='"${DATA_DIR}"' \
		checkpoint_dir='"${CHECKPOINT_DIR}"' \
		project_root='"${PROJECT_ROOT}"' \
		output_dir='"${S2P_OUTPUT_DIR}"'' bash "$@"
