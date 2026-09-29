#!/bin/bash

# This script launches s2p/main.py with comamnd line arguments
# It takes care of providing the dataset, checkpoint, and output dirs
# based on whichever machine/*.env file is sourced.

# DEBUG=false
# PASSTHROUGH=()
# for arg in "$@"; do
#     if [ "$arg" = "--debug" ] || [ "$arg" = "DEBUG=true" ]; then
#         DEBUG=true
#     else
#         PASSTHROUGH+=("$arg")
#     fi
# done
# set -- "${PASSTHROUGH[@]}"

# if [ "$DEBUG" = true ]; then
#     DEBUG_STRING="-m debugpy --listen 5678 --wait-for-client"
# else
#     DEBUG_STRING=""
# fi

# Specialize the output dir
export OUTPUT_DIR=${OUTPUT_DIR}/tdmpc2
export WANDB_DIR=${OUTPUT_DIR}
export TORCH_HOME=${CHECKPOINT_DIR}

# cp ${PROJECT_ROOT}/containers/s2p.sif ${SLURM_TMPDIR}/s2p.sif

# Hosts that keep the driver in /usr/lib64/nvidia (RHEL-family hosts
# generally) need help; hosts that keep it in /usr/lib/x86_64-linux-gnu (Ubuntu
# generally) have no such directory at all, so everything below is conditional -- binding a
# path that does not exist is a fatal error in apptainer, not a warning.
#
# Where it does exist, every file under it is a symlink to ../<name> (i.e. to the real file in
# /usr/lib64 itself). Binding only /usr/lib64/nvidia leaves those symlinks dangling inside the
# container, since ".." resolves to the container's own /usr/lib64, not the host's. --nv separately
# auto-injects a curated set of well-known driver libs (libGLX_nvidia, libEGL_nvidia, ...) that
# mask this for most cases, but less common ones (e.g. libnvidia-glsi, needed by SAPIEN's Vulkan
# renderer) fall through and fail with "cannot open shared object file". Resolve each symlink's real
# target and bind it individually onto the same host path so it's reachable from inside
# /usr/lib64/nvidia too.
#
# Without this, the symptom is not a missing-library message but a renderer that enumerates zero
# devices -- `RuntimeError: Failed to find a supported physical device "cuda:0"` out of
# sapien.render.RenderSystem() during the env's first reconfigure, which reads as a GPU allocation
# problem even though torch can see the GPU perfectly well.
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
	--overlay ${PROJECT_ROOT}/containers/s2p_custom_dependencies.overlay:ro \
	--nv \
	--bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
	--bind ${DATA_DIR}:${DATA_DIR} \
	--bind ${CHECKPOINT_DIR}:${CHECKPOINT_DIR} \
	--bind ${OUTPUT_DIR}:${OUTPUT_DIR} \
	"${NVIDIA_LIB_BINDS[@]}" \
	${PROJECT_ROOT}/containers/s2p.sif \
	bash -c 'export LD_LIBRARY_PATH='"${NVIDIA_LD_PATH}"'$LD_LIBRARY_PATH && \
	python '"${PROJECT_ROOT}"'/agents/tdmpc2/tdmpc2/train.py \
		"$@" \
		hydra.run.dir='"${OUTPUT_DIR}"'/hydra \
		data_dir='"${DATA_DIR}"'/tdmpc2 \
		output_dir='"${OUTPUT_DIR}"'' bash "$@"
