#!/bin/bash
#SBATCH --job-name=mosaic_embed
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=embed_%j.log

module load CUDA/12.1.1

source .venv/bin/activate

python run_embeddings.py
