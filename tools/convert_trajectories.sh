export CUDA_VISIBLE_DEVICES=0
python ${PROJECT_ROOT}/tools/replay_trajectory.py \
  --traj-path $1 \
  --camera-view wrist \
  --use-first-env-state -o rgb \
  --save-traj --num-envs 1024 -b physx_cuda:0 --use-first-env-state --allow-failure \
  --record-rewards --reward-mode=normalized_dense --camera-resolution 224