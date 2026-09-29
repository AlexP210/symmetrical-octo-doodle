#!/bin/bash

# This script launches s2p/main.py with comamnd line arguments
# It takes care of providing the dataset, checkpoint, and output dirs
# based on whichever machine/*.env file is sourced.

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
export OUTPUT_DIR=${OUTPUT_DIR}/s2p
export WANDB_DIR=${OUTPUT_DIR}
export TORCH_HOME=${CHECKPOINT_DIR}

# cp ${PROJECT_ROOT}/containers/s2p.sif ${SLURM_TMPDIR}/s2p.sif

apptainer exec \
	--overlay ${PROJECT_ROOT}/containers/s2p_custom_dependencies.overlay:ro \
	--nv \
	--bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
	--bind ${DATA_DIR}:${DATA_DIR} \
	--bind ${CHECKPOINT_DIR}:${CHECKPOINT_DIR} \
	--bind ${OUTPUT_DIR}:${OUTPUT_DIR} \
	${PROJECT_ROOT}/containers/s2p.sif \
	python ${DEBUG_STRING} ${PROJECT_ROOT}/agents/squeeze2plan/analysis/analysis.py \
		"$@" \
		hydra.run.dir=${OUTPUT_DIR}/hydra \
		data_dir=${DATA_DIR} \
		checkpoint_dir=${CHECKPOINT_DIR} \
		output_dir=${OUTPUT_DIR}
