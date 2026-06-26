#!/bin/bash
#SBATCH --job-name=mosaic_pipe_5MeO
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=pipeline_5MeO_%j.log

module load CUDA/12.1.1

source .venv/bin/activate

# Optuna trial 0: mcs=29 ms=24 nn=24 nc=8 → 55 topics, emb_coh=0.583
python run_pipeline_5MeO.py \
    --mcs 29 \
    --ms 24 \
    --nn 24 \
    --nc 8 \
    --tag t0 \
    --nr-repr-docs 7
