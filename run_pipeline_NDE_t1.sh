#!/bin/bash
#SBATCH --job-name=NDE_t1
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=pipeline_NDE_t1_%j.log

module load CUDA/12.1.1
source .venv/bin/activate

# Trial 1/73 — 78 topics, 49.7% outliers (lowest outlier rate)
PYTHONUNBUFFERED=1 python run_pipeline_NDE_variant.py \
    --mcs 32 --ms 17 --nn 7 --tag t1
