#!/bin/bash
#SBATCH --job-name=mosaic_embed_5MeO
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=embed_5MeO_%j.log

module load CUDA/12.1.1

source .venv/bin/activate

python run_embed_5MeO.py
