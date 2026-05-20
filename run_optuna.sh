#!/bin/bash
#SBATCH --job-name=mosaic_optuna
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=optuna_%j.log

module load CUDA/12.1.1

source .venv/bin/activate

python run_optuna.py \
    --dataset NDE \
    --csv data/NDE/preprocessed/NDE_reports_grouped.csv \
    --text-col cleaned_report \
    --sentences \
    --n_trials 100 \
    --min-cluster-size 10 100 \
    --min-samples 5 30 \
    --n-neighbors 5 30 \
    --target-min 40 \
    --target-max 110
