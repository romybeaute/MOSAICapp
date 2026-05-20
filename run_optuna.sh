#!/bin/bash
#SBATCH --job-name=mosaic_optuna
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=optuna_%j.log

module load CUDA/12.1.1

source .venv/bin/activate

python run_optuna.py \
    --dataset NDE \
    --csv data/NDE/preprocessed/NDE_reports_grouped.csv \
    --embedding-model Qwen/Qwen3-Embedding-4B \
    --text-col cleaned_report \
    --sentences \
    --n_trials 100 \
    --min-cluster-size 20 100 \
    --min-samples 10 60 \
    --n-neighbors 5 50 \
    --target-min 40 \
    --target-max 110 \
    --subsample 15000
