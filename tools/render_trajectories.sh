export CUDA_VISIBLE_DEVICES=1
python replay_trajectory.py \
  --traj-path $1 \
  --use-first-env-state -o rgb \
  --save-video --num-envs 1 -b physx_cpu --use-first-env-state --allow-failure \
  --record-rewards --reward-mode=dense --camera-resolution 224  --trajectories-to-replay 0 100 200 300 400 500 1000 2000 3000 4000 5000 10000 20000 25000 26000 27000 28000 29000