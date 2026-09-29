#!/usr/bin/env bash
# Record one noise-graded demo dataset per task with make_expert_demos.py, rolling out the visual
# PPO expert that train_ppo_visual_experts.sh left in ${OUTPUT_DIR}/ppo_visual_experts.
#
# This is the second half of that script: the run folder is not named here, it is *derived* from
# the same ${CAMERA_VIEW} and ${RESOLUTION} the expert was trained under, so
#
#   CAMERA_VIEW=wrist ./train_ppo_visual_experts.sh PushCube-v1.1
#   CAMERA_VIEW=wrist ./make_expert_demos.sh        PushCube-v1.1
#
# always pairs a policy with its own camera. make_expert_demos.py then reads the rest of the env
# (resolution, control mode, state flag) out of that run's training_summary.json, so nothing about
# the demos can drift from the policy that generated them. best_ckpt.pt is the policy rolled out.
#
# Each episode gets one action-noise std drawn from U(0, max_action_noise) and held for the whole
# episode, so the dataset spans clean expert behaviour through to near-random, labelled per episode
# with `action_noise_std`. The per-task ceilings below were measured (see each case), chosen so the
# span is actually useful: a ceiling that leaves success near 1.0 wastes the range, and one that
# drives it to 0 fills the dataset with flailing.
#
# Expert-quality note. The zero-noise end of each dataset is only as good as the policy, and the
# policies are weaker than their training_summary.json says: measured over 256 clean episodes the
# wrist experts score 1.00 (PushCube), 0.93 (LiftPegUpright) and 0.84 (PlaceSphere), against 1.0
# claimed for all three. That gap is a selection effect, not a bug in either script -- training
# stops the first time an eval of only NUM_EVAL_ENVS=16 episodes all succeed, and over ~80 evals a
# 0.84 policy will hit 16/16 eventually. Judge an expert by demo_summary.json's lowest noise band,
# which is measured over far more episodes, and raise NUM_EVAL_ENVS if you want the training-time
# number to mean something.
#
# A re-run OVERWRITES. RecordEpisode opens the h5 with mode "w", so pointing this script at a task
# whose demos folder is already populated destroys what is there -- including a dry run, which
# writes to the same place a real collection does. Set DEMOS_DIR to send a dry run somewhere
# disposable instead of into ${OUTPUT_DIR}/ppo_visual_experts/<run>/demos.
#
# Sizing note -- read before changing NUM_DEMOS or RESOLUTION. Measured at 224x224 with one wrist
# camera, gzip-5 as RecordEpisode writes it: 3.0 MiB per demo, so the default 10000 demos is ~30
# GiB per task and all three are ~90 GiB. That scales with the square of the resolution and
# linearly with the demo count, and the preflight below refuses a run that will not fit on the
# filesystem ${OUTPUT_DIR} lives on.
#
# Usage:  ./make_expert_demos.sh [task ...]        (default: all three tasks)
#   CAMERA_VIEW=wrist ./make_expert_demos.sh
#   GPU=1 CAMERA_VIEW=wrist ./make_expert_demos.sh PushCube-v1.1
#   SKIP_DEMOS=1 CAMERA_VIEW=wrist ./make_expert_demos.sh PushCube-v1.1   (noise grid video only,
#                                                          for choosing MAX_ACTION_NOISE)
#   NUM_DEMOS=64 NUM_ENVS=32 CAMERA_VIEW=wrist DEMOS_DIR=/tmp/demo_dryrun \
#     ./make_expert_demos.sh PushCube-v1.1         (fast end-to-end dry run, written somewhere
#                                                   disposable so it cannot clobber real demos)

set -uo pipefail

# Every stop below is `return 1 2>/dev/null || exit 1`: `return` ends the script when it is
# sourced and errors harmlessly when it is not, so a failure cannot take an interactive shell (and
# its tmux pane) down with it. It has to be written out at the top level each time -- inside a
# helper function `return` would only leave the function.

: "${PROJECT_ROOT:?set PROJECT_ROOT to the s2p-project root}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to the training output root}"

control_mode=pd_ee_delta_pos
seed=${SEED:-1}
camera_view=${CAMERA_VIEW:-default}
resolution=${RESOLUTION:-224}
checkpoint=${CHECKPOINT:-best_ckpt.pt}
horizon=50
# pin sim and torch to one physical GPU so sapien and torch cannot disagree on the device index
gpu=${GPU:-0}
export CUDA_VISIBLE_DEVICES=${gpu}

case ${camera_view} in
default | focused | wrist) ;;
*)
	echo "unknown camera_view ${camera_view}, expected default, focused or wrist" >&2
	return 1 2>/dev/null || exit 1 ;;
esac

runs_dir="${OUTPUT_DIR}/ppo_visual_experts"
if [ ! -d "${runs_dir}" ]; then
	echo "${runs_dir} does not exist -- train an expert first with train_ppo_visual_experts.sh" >&2
	return 1 2>/dev/null || exit 1
fi

tasks=("$@")
if [ ${#tasks[@]} -eq 0 ]; then
	tasks=(PushCube-v1.1 LiftPegUpright-v1.1 PlaceSphere-v1.1)
fi

# space left where the datasets are about to be written, for the size preflight below
avail_mib=$(df -PBM "${runs_dir}" 2>/dev/null | awk 'NR==2 {gsub("M","",$4); print $4}')
avail_mib=${avail_mib:-0}

failed=()
for env_id in "${tasks[@]}"; do
	num_envs=128
	case ${env_id} in
	PushCube-v1.1)
		# measured on the wrist expert at 224: 1.00 clean, falling to 0.56 by U(0, 1) -- a clean
		# gradient from expert to barely-competent with no dead zone at either end
		num_demos=10_000
		max_action_noise=1.0 ;;
	LiftPegUpright-v1.1)
		# 0.93 clean (over 256 episodes, not the 1.0 its training_summary.json claims -- see the
		# note below), 0.31 at 0.8, and flat ~0.13 past that, so the extra range would buy
		# flailing rather than gradient
		num_demos=10_000
		max_action_noise=0.8 ;;
	PlaceSphere-v1.1)
		# the hardest and most noise-sensitive of the three: 0.84 clean, 0.33 by 0.4 and flat
		# 0.0 from 0.6 up, so a ceiling of 1.0 would spend 40% of the range on failures
		num_demos=10_000
		max_action_noise=0.5 ;;
	PickCube-v1.1)
		# the hardest and most noise-sensitive of the three: 0.84 clean, 0.33 by 0.4 and flat
		# 0.0 from 0.6 up, so a ceiling of 1.0 would spend 40% of the range on failures
		num_demos=10_000
		max_action_noise=1.0 ;;
	*)
		echo "unknown task ${env_id}" >&2
		failed+=("${env_id} (unknown task)")
		continue ;;
	esac

	# overrides, for dry runs that exercise the whole pipeline without the full dataset
	num_demos=${NUM_DEMOS:-${num_demos}}
	num_envs=${NUM_ENVS:-${num_envs}}
	max_action_noise=${MAX_ACTION_NOISE:-${max_action_noise}}
	reconfiguration_freq=${RECONFIGURATION_FREQ:-1}
	skip_flags=""
	[ "${SKIP_VIDEO:-0}" = "1" ] && skip_flags="${skip_flags} --skip_video"
	[ "${SKIP_DEMOS:-0}" = "1" ] && skip_flags="${skip_flags} --skip_demos"
	# DEMOS_DIR redirects the write away from the run folder, so a dry run cannot overwrite a real
	# collection; the per-task suffix keeps several tasks from landing on top of each other
	out_flag=""
	[ -n "${DEMOS_DIR:-}" ] && out_flag="--output_dir=${DEMOS_DIR}/${env_id}-${camera_view}-${resolution}"

	agent_folder="${runs_dir}/${env_id}-ppo-visual-${camera_view}-${resolution}-${control_mode}"
	demos=${num_demos//_/}
	# rounded the way make_expert_demos.py rounds it: up to a whole multiple of num_envs
	demos=$(( (demos + num_envs - 1) / num_envs * num_envs ))
	# 0.4 is the measured gzip-5 ratio on this data, not a guess -- 3.0 MiB stored per 7.7 MiB raw
	size_mib=$(( demos * (horizon + 1) * resolution * resolution * 3 * 2 / 5 / 1048576 ))
	echo "=== ${env_id}: ${demos} demos at noise ~ U(0, ${max_action_noise}), camera_view=${camera_view} ==="
	echo "    expert: ${agent_folder##*/}/${checkpoint}"
	echo "    dataset ~${size_mib} MiB of ${avail_mib} MiB free on $(df -P "${runs_dir}" | awk 'NR==2 {print $6}')"

	if [ ! -f "${agent_folder}/training_summary.json" ]; then
		echo "    SKIP: no expert at ${agent_folder}." >&2
		echo "    Runs that do exist under ${runs_dir}:" >&2
		ls -1 "${runs_dir}" 2>/dev/null | grep -v '\.log$' | sed 's/^/      /' >&2
		echo "    Train one with: CAMERA_VIEW=${camera_view} RESOLUTION=${resolution} ./train_ppo_visual_experts.sh ${env_id}" >&2
		failed+=("${env_id} (no expert for camera_view=${camera_view} at ${resolution})")
		continue
	fi
	if [ ! -f "${agent_folder}/${checkpoint}" ]; then
		echo "    SKIP: ${agent_folder}/${checkpoint} is missing; that run saved no such checkpoint." >&2
		failed+=("${env_id} (no ${checkpoint})")
		continue
	fi
	if [ "${SKIP_DEMOS:-0}" != "1" ] && [ ${avail_mib} -gt 0 ] && [ ${size_mib} -gt ${avail_mib} ]; then
		echo "    SKIP: the dataset needs ~${size_mib} MiB but only ${avail_mib} MiB is free." >&2
		echo "    Lower NUM_DEMOS or RESOLUTION (size scales with the square of the resolution)." >&2
		failed+=("${env_id} (needs ~${size_mib} MiB, ${avail_mib} MiB free)")
		continue
	fi

	python ${PROJECT_ROOT}/tools/make_expert_demos.py \
		--agent_folder="${agent_folder}" \
		--num_demos=${num_demos} \
		--max_action_noise=${max_action_noise} \
		--checkpoint_name=${checkpoint} \
		--num_envs=${num_envs} \
		--reconfiguration_freq=${reconfiguration_freq} \
		--seed=${seed} \
		--device=cuda:0 \
		${out_flag} ${skip_flags} 2>&1 | tee "${runs_dir}/${env_id}-demos-${camera_view}-${resolution}.log"
	# tee is last in the pipe, so check python's status rather than tee's
	[ "${PIPESTATUS[0]}" -ne 0 ] && failed+=("${env_id}")
done

for env_id in "${tasks[@]}"; do
	if [ -n "${DEMOS_DIR:-}" ]; then
		summary="${DEMOS_DIR}/${env_id}-${camera_view}-${resolution}/demo_summary.json"
	else
		summary="${runs_dir}/${env_id}-ppo-visual-${camera_view}-${resolution}-${control_mode}/demos/demo_summary.json"
	fi
	[ -f "${summary}" ] && echo "--- ${env_id} ---" && cat "${summary}"
done

if [ ${#failed[@]} -ne 0 ]; then
	echo "FAILED: ${failed[*]}" >&2
	return 1 2>/dev/null || exit 1
fi
