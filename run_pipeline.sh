#!/bin/bash
#SBATCH --job-name=mosaic_pipeline
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=pipeline_%j.log

module load CUDA/12.1.1

source .venv/bin/activate

python run_pipeline.py
