#!/bin/bash
#SBATCH --job-name=mosaic_optuna_MPE
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=optuna_MPE_%j.log

module load CUDA/12.1.1

source .venv/bin/activate

python run_optuna.py \
    --dataset MPE \
    --csv data/MPE/preprocessed/MPE_dataset_translated_batched.csv \
    --embedding-model Qwen/Qwen3-Embedding-4B \
    --text-col phen_report_english \
    --sentences \
    --n_trials 100 \
    --min-cluster-size 10 50 \
    --min-samples 5 30 \
    --n-neighbors 5 30 \
    --target-min 15 \
    --target-max 50
