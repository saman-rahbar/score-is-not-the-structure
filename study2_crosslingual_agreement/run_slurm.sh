#!/bin/bash
#SBATCH --job-name=study2
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out

# Example SLURM launcher. Edit YOUR_ACCOUNT and the module lines for your cluster.
# Prefetch the multilingual model + MultiBLiMP on a login node first; lang2vec
# bundles its URIEL data and works offline once installed.
module load python cuda        # adjust to your site's module names
source .venv/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

python experiment.py
