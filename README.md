# Squeeze-to-Plan

## Note for ICLR Reviewers
This is the top-level repository for the Squeeze-to-Plan (S2P) project. It holds the shared set-up (Apptainer container, job launch scripts, data tools); the code itself is split across the following anonymized repositories:

| Component | Path in this project | Anonymized repository |
|---|---|---|
| S2P (our method), including the configs used for S2P and all baselines | `agents/squeeze2plan` | https://anonymous.4open.science/r/SqueezeToPlan-790B |
| Custom ManiSkill tasks (task definitions, lighting conditions, environment construction) | `environments/custom_maniskill_tasks` | https://anonymous.4open.science/r/CustomManiskillTasks-DBA4 |

The S2P repository's README lists the exact training and evaluation configs used in the paper.

Public dependencies are used unmodified: [DINOv3](https://github.com/facebookresearch/dinov3) at `dependencies/dinov3` and [ManiSkill](https://github.com/mani-skill/ManiSkill) at `dependencies/ManiSkill`.

## Installation
To install S2P on a new machine for training the S2P model or any of the baselines, make sure that `apptainer` is available, place each repository at the path listed in the table above, and run the following:

```bash
cd s2p-project
source machines/<platform-name>.env
source install.sh
```

For a new machine, you need to create a new environment config inside /machines/. This should specify
any machine-specific configurations that are needed for running the project code, including:
- Directory to find any offline datasets
- Directory where to place run outputs
- Directory where to find model weights for pre-trained checkpoints
See /machines/example.env for an example.

Finally, if you are setting up a new baseline, add a new script to /jobs/. This script should do the job of calling the baseline's training script from inside singularity container. See /jobs/run_s2p.sh for an example.

## Project Structure
The project has the following folders:
- agents/: Contains the code for running our approach (at agents/squeeze2plan) and any baselines
- environments/: Implementations for any custom environments
- dependencies/: Additional external dependencies
- containers/: Apptainer-related files for the project environment
- machines/: .env files which define the paths to additional data required for training (e.g. offline datasets, pretrained checkpoints)
- jobs/: Scripts which handle running training jobs for various agents in a platform-agnostic way, using Apptainer and the environment variables set in the machines/<platform-name>.env files.
- tools/: Auxiliary scripts used for generating datasets, visualization/analysis, etc.

## Running S2P Training
To run S2P training, do the following:
```bash
cd path/to/s2p-project
source machines/<your-platform-name>.env
./jobs/run_s2p.sh --config-name=train_visual_online_deterministic_push_cube runner.cfg.run_name=ExampleRun
```

`train_visual_online_deterministic_push_cube` is the S2P training configuration for the PushCube task (use `_lift_peg` or `_pick_cube` for the other tasks). For more details on the training configuration for S2P, and how to modify them to run a different configuration, see the README at agents/squeeze2plan.
