#!/bin/bash
#SBATCH --job-name=mosaic_pipeline_70b
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=pipeline_70b_%j.log

module load CUDA/12.1.1

source .venv/bin/activate

PYTHONUNBUFFERED=1 python run_pipeline.py \
    --llm-model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --output-dir outputs_mixtral
