#!/usr/bin/env bash
# Record ${EPISODES} random + ${EPISODES} mid-training + ${EPISODES} expert episodes per task with
# ppo_stages_fast.py, which builds every env through environments/custom_maniskill_tasks so the
# recording and the agents' online/planning envs cannot drift apart.
#
# The four tasks this project actually trains on are recorded under their `-v1.1` ids, which carry
# "no automatic reset on success" in the task itself rather than in this script's wrapper stack (see
# environments/custom_maniskill_tasks/custom_maniskill_tasks/tasks.py). Datasets recorded under a
# `-v1.1` id therefore have an all-False `terminated` field -- per-step and per-episode `success` is
# recorded separately and is unaffected. The stock `-v1` ids still work and behave exactly as
# before; both spellings take the same per-task settings below.
#
# Sizing note -- read before changing total_timesteps or num_envs. ppo_stages_fast.py forces
# num_steps to the task horizon, so one iteration costs num_envs*horizon env steps and the run gets
# total_timesteps/(num_envs*horizon) iterations. Learning tracks *iterations*, not env steps, so a
# wide num_envs at a fixed timestep budget buys fewer updates, not faster learning. The baselines.sh
# shape (num_envs=4096, update_epochs=8) assumes its own short num_steps=4 rollouts and does NOT
# transfer here: at 1M timesteps it yields 4 iterations and never leaves 0% success.
#
# These configs instead follow make_ppo_random_maniskill_data.sh, which is measured to reach
# eval/success_once=1.0 on PushCube-v1: num_envs=1024, update_epochs=32, num_minibatches=32.
# On that recipe PushCube first moves off 0% at iteration ~10 and saturates at ~14, so
# total_timesteps is set to roughly twice the expected convergence point. The preflight below
# prints the derived iteration count and refuses to start a run too short to learn anything.
#
# num_eval_envs divides EPISODES exactly, so each stage records exactly EPISODES episodes.
#
# Usage:  ./make_ppo_staged_maniskill_data.sh [task ...]     (default: all four tasks)
#   GPU=1 ./make_ppo_staged_maniskill_data.sh PushCube-v1.1
#   EPISODES=8 NUM_ENVS=64 NUM_EVAL_ENVS=4 TOTAL_TIMESTEPS=200000 TRACK=0 \
#     ./make_ppo_staged_maniskill_data.sh PushCube-v1.1      (fast end-to-end dry run)

set -uo pipefail

: "${PROJECT_ROOT:?set PROJECT_ROOT to the s2p-project root}"
: "${DATA_DIR:?set DATA_DIR to the dataset root}"

control_mode=pd_ee_delta_pos
seed=${SEED:-1}
episodes=${EPISODES:-10000}
# pin sim and torch to one physical GPU so sapien and torch cannot disagree on the device index
export CUDA_VISIBLE_DEVICES=${GPU:-0}

out_dir="${DATA_DIR}/maniskill"
mkdir -p "${out_dir}"

tasks=("$@")
if [ ${#tasks[@]} -eq 0 ]; then
	tasks=(PushCube-v1.1 PlaceSphere-v1.1 LiftPegUpright-v1.1 PokeCube-v1.1)
fi

failed=()
for env_id in "${tasks[@]}"; do
	gamma=0.8
	gae_lambda=0.9
	num_envs=1024
	num_eval_envs=500
	update_epochs=32
	eval_freq=1
	asset_group=""
	# both id spellings of a task take the same settings: `-v1.1` differs from `-v1` only in that it
	# never terminates early, which changes nothing about the horizon or how long it takes to learn
	case ${env_id} in
	PushCube-v1.1)
		horizon=50; total_timesteps=1_000_000
		success_rate=0.95 ;;
	PlaceSphere-v1.1)
		# no baselines.sh entry; mirrors PushCube-v1, which shares its horizon and robot
		horizon=50; total_timesteps=30_000_000
		success_rate=0.85 ;;
	LiftPegUpright-v1.1)
		horizon=50; total_timesteps=30_000_000
		success_rate=0.90 ;;
	PokeCube-v1.1)
		horizon=50; total_timesteps=30_000_000
		success_rate=0.90 ;;
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

	# preflight: too few iterations is the failure mode that looks like "PPO does not learn"
	batch=$((num_envs * horizon))
	iters=$(( ${total_timesteps//_/} / batch ))
	evals=$((iters / eval_freq))
	recordings=$(( (episodes + num_eval_envs - 1) / num_eval_envs ))
	record_every=$(( evals / recordings )); [ ${record_every} -lt 1 ] && record_every=1
	echo "=== ${env_id}: ${episodes} random + ${episodes} training + ${episodes} expert episodes ==="
	echo "    batch=${batch} (${num_envs} envs x ${horizon} horizon)  iterations=${iters}  evals=${evals}"
	echo "    mid-training: ${recordings} recordings of ${num_eval_envs}, one every ${record_every} evals"
	if [ ${iters} -lt 15 ]; then
		echo "    SKIP: only ${iters} iterations; PushCube-v1 needs ~14 just to leave 0% success." >&2
		echo "    Raise total_timesteps or lower num_envs (batch is num_envs*horizon)." >&2
		failed+=("${env_id} (only ${iters} iterations)")
		continue
	fi

	# preflight: a missing asset group otherwise stops the batch on an interactive y/n prompt.
	# Count per-asset the way ManiSkill does -- a partial download leaves the directory non-empty.
	if [ -n "${asset_group:-}" ]; then
		missing=$(python -c "
from mani_skill.utils import assets
print(sum(not assets.is_data_source_downloaded(u) for u in assets.DATA_GROUPS['${asset_group}']))
" 2>/dev/null)
		if [ "${missing:-1}" != "0" ]; then
			echo "    SKIP: ${missing} ${asset_group} asset(s) still missing. Fetch them with:" >&2
			echo "      python -m mani_skill.utils.download_asset ${asset_group}" >&2
			failed+=("${env_id} (missing ${asset_group} assets)")
			continue
		fi
	fi

	exp_name=${env_id}-ppo-staged-${control_mode}
	python ${PROJECT_ROOT}/tools/ppo_stages_fast.py --env_id=${env_id} \
		--seed=${seed} \
		--total_timesteps=${total_timesteps} \
		--num_envs=${num_envs} --update_epochs=${update_epochs} --num_minibatches=32 \
		--gamma=${gamma} --gae_lambda=${gae_lambda} \
		--eval_freq=${eval_freq} --num_eval_envs=${num_eval_envs} \
		--number_of_random_episodes=${episodes} \
		--number_of_training_episodes=${episodes} \
		--number_of_expert_episodes=${episodes} \
		--training_done_success_rate=${success_rate} \
		--save_trajectory --no_capture_video \
		--no_cudagraphs --compile \
		--save-path "${out_dir}/" \
		--exp-name=${exp_name} \
		--control-mode ${control_mode} \
		--device=cuda:0 \
		${track_flag} 2>&1 | tee "${out_dir}/${exp_name}.log"
	# tee is last in the pipe, so check python's status rather than tee's
	[ "${PIPESTATUS[0]}" -ne 0 ] && failed+=("${env_id}")
done

for env_id in "${tasks[@]}"; do
	summary="${out_dir}/${env_id}-ppo-staged-${control_mode}/collection_summary.json"
	[ -f "${summary}" ] && echo "--- ${env_id} ---" && cat "${summary}"
done

if [ ${#failed[@]} -ne 0 ]; then
	echo "FAILED: ${failed[*]}" >&2
	exit 1
fi
