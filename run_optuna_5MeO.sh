#!/bin/bash
#SBATCH --job-name=mosaic_optuna_5MeO
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=optuna_5MeO_%j.log

module load CUDA/12.1.1

source .venv/bin/activate

python run_optuna.py \
    --dataset 5MeO \
    --csv data/5-Me0-DMT/5MeO_naturalistic_preprocessed.csv \
    --embedding-model Qwen/Qwen3-Embedding-4B \
    --text-col cleaned_text \
    --sentences \
    --n_trials 100 \
    --min-cluster-size 10 60 \
    --min-samples 5 25 \
    --n-neighbors 5 30 \
    --nc-max 10 \
    --target-min 20 \
    --target-max 60
