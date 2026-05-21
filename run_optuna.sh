#!/bin/bash
#SBATCH --job-name=mosaic_optuna
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
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
    --min-cluster-size 20 70 \
    --min-samples 10 50 \
    --n-neighbors 15 35 \
    --target-min 40 \
    --target-max 110
