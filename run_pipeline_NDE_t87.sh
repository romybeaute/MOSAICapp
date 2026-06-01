#!/bin/bash
#SBATCH --job-name=NDE_t87
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=pipeline_NDE_t87_%j.log

module load CUDA/12.1.1
source .venv/bin/activate

# Trial 87 — 54 topics, 53.9% outliers (best coherence)
PYTHONUNBUFFERED=1 python run_pipeline_NDE_variant.py \
    --mcs 20 --ms 19 --nn 37 --tag t87
