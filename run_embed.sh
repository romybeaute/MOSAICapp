#!/bin/bash
#SBATCH --job-name=mosaic_embed
#SBATCH --partition=short
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=embed_%j.log

module load CUDA/12.1.1

source .venv/bin/activate

python run_embed.py
