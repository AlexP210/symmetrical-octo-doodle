#!/bin/bash

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
export OUTPUT_DIR=${OUTPUT_DIR}/repo
export WANDB_DIR=${OUTPUT_DIR}
apptainer exec \
	--overlay ${PROJECT_ROOT}/containers/s2p_custom_dependencies.overlay:ro \
	--nv \
	--bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
	--bind ${DATA_DIR}:${DATA_DIR} \
	--bind ${CHECKPOINT_DIR}:${CHECKPOINT_DIR} \
	--bind ${OUTPUT_DIR}:${OUTPUT_DIR} \
	${PROJECT_ROOT}/containers/s2p.sif \
	python ${DEBUG_STRING} ${PROJECT_ROOT}/agents/repo/repo/experiments/train_repo.py \
		"$@" \
		--offline_dir=${DATA_DIR} \
		--checkpoint_dir=${CHECKPOINT_DIR} \
		--out_folder=${OUTPUT_DIR} \
