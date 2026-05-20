#!/bin/bash
#SBATCH --job-name=mosaic_optuna
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=optuna_%j.log

module load CUDA/12.1.1

source .venv/bin/activate

python run_optuna.py
