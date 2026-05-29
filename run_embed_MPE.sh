#!/bin/bash
#SBATCH --job-name=mosaic_embed_MPE
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=embed_MPE_%j.log

module load CUDA/12.1.1

source .venv/bin/activate

python run_embed_MPE.py
