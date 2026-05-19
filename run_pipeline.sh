#!/bin/bash
#SBATCH --job-name=mosaic_pipeline
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=pipeline_%j.log

module load cuda
conda activate .venv  #(to change dependong on one's setup venv name)

python run_pipeline.py
