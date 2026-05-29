#!/bin/bash
#SBATCH --job-name=mosaic_MPE
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=pipeline_MPE_%j.log

module load CUDA/12.1.1

source .venv/bin/activate

PYTHONUNBUFFERED=1 python run_pipeline_MPE.py
