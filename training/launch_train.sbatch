#!/bin/bash
#SBATCH --job-name=planning
#SBATCH --output=/data/user_data/mbronars/slurm/output/planning_%j.out
#SBATCH --error=/data/user_data/mbronars/slurm/error/planning_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --partition=general

# 2. Load environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pctrl


cd /data/user_data/mbronars/packages/Planning_wrapper

python training/train.py --config training/config_circle.yaml