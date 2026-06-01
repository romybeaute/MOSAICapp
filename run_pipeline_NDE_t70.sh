#!/bin/bash
#SBATCH --job-name=NDE_t70
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=pipeline_NDE_t70_%j.log

module load CUDA/12.1.1
source .venv/bin/activate

# Trial 70 — 74 topics, 54.3% outliers
PYTHONUNBUFFERED=1 python run_pipeline_NDE_variant.py \
    --mcs 27 --ms 13 --nn 18 --tag t70
