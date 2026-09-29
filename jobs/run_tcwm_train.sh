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


# hydra.run.dir=${OUTPUT_DIR}/hydra \
# Specialize the output dir
export TCWM_OUTPUT_DIR=${OUTPUT_DIR}/tcwm
# Additional DATASET_DIR env var that DINO needs to find the datasets
export DATASET_DIR=${DATA_DIR}/tcwm
export WANDB_DIR=${OUTPUT_DIR}

# The dataset loader stages the h5 onto node-local disk when it can see $SLURM_TMPDIR.
# Apptainer forwards the variable but not the mount, so bind it when we're in a job.
TMPDIR_BIND=()
if [ -n "${SLURM_TMPDIR}" ] && [ -d "${SLURM_TMPDIR}" ]; then
	TMPDIR_BIND=(--bind "${SLURM_TMPDIR}:${SLURM_TMPDIR}")
fi

apptainer exec \
	--overlay ${PROJECT_ROOT}/containers/s2p_custom_dependencies.overlay:ro \
	--nv \
	--bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
	--bind ${DATA_DIR}:${DATA_DIR} \
	--bind ${CHECKPOINT_DIR}:${CHECKPOINT_DIR} \
	--bind ${OUTPUT_DIR}:${OUTPUT_DIR} \
	"${TMPDIR_BIND[@]}" \
	${PROJECT_ROOT}/containers/s2p.sif \
	python ${DEBUG_STRING} ${PROJECT_ROOT}/agents/TC-WM/train.py \
		"$@" \
		data_dir=${DATA_DIR} \
		ckpt_base_path=${TCWM_OUTPUT_DIR}