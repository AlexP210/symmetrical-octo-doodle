#!/usr/bin/env bash
# Train one visual PPO expert per task with ppo_visual_expert_fast.py, which builds every env
# through environments/custom_maniskill_tasks so the expert, the recorded datasets and the agents'
# online/planning envs all see the task through the same camera.
#
# The camera is the point of this script: ${CAMERA_VIEW} picks one of "default", "focused" or
# "wrist" (see custom_maniskill_tasks/cameras.py) and is baked into the run name, so experts for
# different views live side by side and cannot be confused for one another. A policy trained on
# one view is not an expert in any other.
#
# Only the `-v1.1` ids are supported. They never terminate early, so an episode always runs the
# full 50-step horizon, one eval rollout is exactly ${NUM_EVAL_ENVS} complete episodes, and
# `success_once` is measured over whole episodes.
#
# Sizing note -- read before changing NUM_ENVS or RESOLUTION. Unlike the state-PPO recipe in
# make_ppo_staged_maniskill_data.sh (num_envs=1024), visual PPO is bounded by GPU memory: the
# rollout holds num_envs*50 uint8 frames, which is 1.8 GiB at 256 envs and 224x224 and grows with
# the square of the resolution. The defaults below are ManiSkill's own RGB-PPO shape for PushCube
# (num_envs=256, update_epochs=8, num_minibatches=8, examples/baselines/ppo/baselines.sh), and the
# preflight prints the derived iteration count and buffer size and refuses runs that cannot fit or
# cannot learn.
#
# Measured on one RTX 4090, PushCube-v1.1 at camera_view=default and 128x128: eval success_once
# first leaves 0 at iteration ~15, and reaches 0.94 at iteration 46 (576k env steps, ~3.1 s per
# iteration). The budgets below are ceilings well past that -- each run stops early the moment an
# eval reaches its per-task success rate -- and the best-scoring policy is kept at best_ckpt.pt
# whether or not that happened. Expect the 224x224 default to cost noticeably more per iteration
# than the 128x128 those numbers came from, so the harder two tasks are hours, not minutes.
#
# Usage:  ./train_ppo_visual_experts.sh [task ...]     (default: all three tasks)
#   CAMERA_VIEW=focused ./train_ppo_visual_experts.sh
#   GPU=1 ./train_ppo_visual_experts.sh PushCube-v1.1
#   INCLUDE_STATE=0 ./train_ppo_visual_experts.sh PushCube-v1.1     (pixels only)
#   CKPT_FREQ=50 ./train_ppo_visual_experts.sh PushCube-v1.1         (keep every 50th policy)
#   RESOLUTION=128 NUM_ENVS=64 TOTAL_TIMESTEPS=500000 TRACK=0 \
#     ./train_ppo_visual_experts.sh PushCube-v1.1      (fast end-to-end dry run)

set -uo pipefail

: "${PROJECT_ROOT:?set PROJECT_ROOT to the s2p-project root}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to the training output root}"

control_mode=pd_ee_delta_pos
seed=${SEED:-1}
camera_view=${CAMERA_VIEW:-default}
resolution=${RESOLUTION:-224}
# pin sim and torch to one physical GPU so sapien and torch cannot disagree on the device index
gpu=${GPU:-0}
export CUDA_VISIBLE_DEVICES=${gpu}

case ${camera_view} in
default | focused | wrist) ;;
*)
	echo "unknown camera_view ${camera_view}, expected default, focused or wrist" >&2
	exit 1 ;;
esac

out_dir="${OUTPUT_DIR}/ppo_visual_experts"
mkdir -p "${out_dir}"

tasks=("$@")
if [ ${#tasks[@]} -eq 0 ]; then
	tasks=(PushCube-v1.1 LiftPegUpright-v1.1 PlaceSphere-v1.1)
fi

# total memory of the GPU the run will land on, for the rollout-buffer preflight below
gpu_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i ${gpu} 2>/dev/null | head -1)
gpu_mib=${gpu_mib:-0}

failed=()
for env_id in "${tasks[@]}"; do
	gamma=0.8
	gae_lambda=0.9
	num_envs=256
	num_eval_envs=16
	update_epochs=8
	num_minibatches=8
	# every eval is a full-horizon rollout of num_eval_envs plus an mp4, so on the thousand-plus
	# iteration budgets below this is the difference between minutes and an hour of overhead;
	# 10 is still fine-grained enough to stop PushCube (converges at ~46) close to convergence
	eval_freq=10
	horizon=50
	case ${env_id} in
	PushCube-v1.1)
		# the one task with a published RGB-PPO baseline; measured to converge at iteration
		# ~46, so this budget is roughly an order of magnitude past it
		total_timesteps=30_000_000
		success_rate=1.0 ;;
	LiftPegUpright-v1.1)
		# no RGB baseline; its state-PPO entry matches PushCube's, so the gap here is the
		# pixels, not the task
		total_timesteps=30_000_000
		success_rate=1.0 ;;
	PlaceSphere-v1.1)
		# no baseline of either kind, and the hardest of the three (grasp, then place in the
		# bin), so treat a shortfall against success_rate as expected rather than as a bug
		total_timesteps=30_000_000
		success_rate=1.0 ;;
	PickCube-v1.1)
		horizon=50; total_timesteps=30_000_000
		success_rate=0.95 ;;
	*)
		echo "unknown task ${env_id}" >&2
		failed+=("${env_id} (unknown task)")
		continue ;;
	esac

	# overrides, for dry runs that exercise the whole pipeline without the full training budget
	num_envs=${NUM_ENVS:-${num_envs}}
	num_eval_envs=${NUM_EVAL_ENVS:-${num_eval_envs}}
	total_timesteps=${TOTAL_TIMESTEPS:-${total_timesteps}}
	if [ "${TRACK:-1}" = "1" ]; then track_flag="--track"; else track_flag="--no-track"; fi
	# INCLUDE_STATE=0 trains on pixels alone; the default adds proprioception and the tcp pose,
	# which under an image obs mode is all ManiSkill exposes (no object or goal pose)
	if [ "${INCLUDE_STATE:-1}" = "1" ]; then state_flag="--include_state"; else state_flag="--no-include_state"; fi
	# CKPT_FREQ=N also keeps ckpt_<iteration>.pt every N iterations, for a learning-progression
	# sweep rather than just the finished expert. Off by default because a checkpoint is 38 MiB
	# at 224x224, so one per eval over LiftPegUpright's 2344 iterations would be ~9 GiB.
	ckpt_freq=${CKPT_FREQ:-0}

	batch=$((num_envs * horizon))
	iters=$(( ${total_timesteps//_/} / batch ))
	# the rollout's uint8 frames; peak usage is roughly twice this (the per-step tensors are
	# still alive while they are stacked) plus the sim, the renderer and the model
	buffer_mib=$(( batch * resolution * resolution * 3 / 1048576 ))
	echo "=== ${env_id}: visual PPO expert, camera_view=${camera_view} at ${resolution}x${resolution} ==="
	echo "    batch=${batch} (${num_envs} envs x ${horizon} horizon)  iterations=${iters}  evals=$((iters / eval_freq))"
	echo "    rollout buffer=${buffer_mib} MiB of $((gpu_mib)) MiB on GPU ${gpu}"
	if [ ${iters} -lt 50 ]; then
		echo "    SKIP: only ${iters} iterations; PushCube needs ~46 even at 256 envs." >&2
		echo "    Raise TOTAL_TIMESTEPS or lower NUM_ENVS (batch is num_envs*horizon)." >&2
		failed+=("${env_id} (only ${iters} iterations)")
		continue
	fi
	if [ ${gpu_mib} -gt 0 ] && [ $((buffer_mib * 5 / 2)) -gt ${gpu_mib} ]; then
		echo "    SKIP: the rollout buffer alone needs ${buffer_mib} MiB, which leaves no room" >&2
		echo "    for the sim, the renderer and the update on a ${gpu_mib} MiB GPU." >&2
		echo "    Lower NUM_ENVS or RESOLUTION (buffer scales with both, squared in RESOLUTION)." >&2
		failed+=("${env_id} (rollout buffer ${buffer_mib} MiB too large)")
		continue
	fi

	exp_name=${env_id}-ppo-visual-${camera_view}-${resolution}-${control_mode}
	python ${PROJECT_ROOT}/tools/ppo_visual_expert_fast.py --env_id=${env_id} \
		--seed=${seed} \
		--total_timesteps=${total_timesteps} \
		--num_envs=${num_envs} --update_epochs=${update_epochs} --num_minibatches=${num_minibatches} \
		--gamma=${gamma} --gae_lambda=${gae_lambda} \
		--eval_freq=${eval_freq} --num_eval_envs=${num_eval_envs} \
		--save_ckpt_freq=${ckpt_freq} \
		--camera_view=${camera_view} --camera_resolution=${resolution} \
		${state_flag} \
		--training_done_success_rate=${success_rate} \
		--capture_video --save_model \
		--no-cudagraphs --compile \
		--save-path "${out_dir}/" \
		--exp-name=${exp_name} \
		--control-mode ${control_mode} \
		--device=cuda:0 \
		${track_flag} 2>&1 | tee "${out_dir}/${exp_name}.log"
	# tee is last in the pipe, so check python's status rather than tee's
	[ "${PIPESTATUS[0]}" -ne 0 ] && failed+=("${env_id}")
done

for env_id in "${tasks[@]}"; do
	summary="${out_dir}/${env_id}-ppo-visual-${camera_view}-${resolution}-${control_mode}/training_summary.json"
	[ -f "${summary}" ] && echo "--- ${env_id} ---" && cat "${summary}"
done

if [ ${#failed[@]} -ne 0 ]; then
	echo "FAILED: ${failed[*]}" >&2
	exit 1
fi
