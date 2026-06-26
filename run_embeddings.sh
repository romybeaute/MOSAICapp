#!/bin/bash
#SBATCH --job-name=mosaic_embed
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=embed_%j.log

module load CUDA/12.1.1

source .venv/bin/activate

# Any args after the script name are forwarded to run_embeddings.py, e.g.:
#   sbatch run_embeddings.sh --csv data-tutorial/DMT.csv --dataset DMT --text-col sentence
python run_embeddings.py "$@"
