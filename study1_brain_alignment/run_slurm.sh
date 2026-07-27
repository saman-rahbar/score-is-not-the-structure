#!/bin/bash
#SBATCH --job-name=study1
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=11:59:00
#SBATCH --output=%x-%j.out

# Example SLURM launcher. Edit YOUR_ACCOUNT and the module lines for your cluster.
# Compute nodes are assumed offline: prefetch models + BLiMP on a login node first.
module load python cuda        # adjust to your site's module names
source .venv/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PEREIRA_NPZ=$SCRATCH/pereira_cache.npz
export NOISE_CEILING_JSON=$SCRATCH/noise_ceiling.json

# headline model (20 seeds, full protocol) and scale-generality models:
export EXP_MODEL="${EXP_MODEL:-EleutherAI/pythia-160m}"
export EXP_SEEDS="${EXP_SEEDS:-20}"
export EXP_MAIN_ONLY="${EXP_MAIN_ONLY:-0}"

python experiment.py
