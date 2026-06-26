"""
run_embeddings.py — compute and save ONLY the embeddings.

Unlike run_pipeline.py (which also fits BERTopic and calls the LLM), this script
does just the slow part: preprocess the CSV and embed it. It needs no HF_TOKEN and
does not touch the BERTopic config.

It writes the exact two files the Streamlit app looks for:
    precomputed_<...>_docs.json
    precomputed_<...>_<model>_..._embeddings.npy
so you can upload them to the Hugging Face Space (same cache/ path) and the app
will skip embedding entirely.

Usage:
    python run_embeddings.py                  # local (set DEVICE below)
    sbatch run_embeddings.sh                   # on a SLURM cluster (e.g. Artemis)
"""

import json
from pathlib import Path

import numpy as np

from mosaic_core.core_functions import (
    preprocess_and_embed,
    get_cache_dir,
    get_precomputed_filenames,
)

# ── Edit these ────────────────────────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).parent
DATASET_NAME    = "MOSAIC"                          # must match the app sidebar "Project/Dataset name"
CSV_PATH        = "data/preprocessed/MOSAIC/your_file.csv"   # path to your CSV (← change this)
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"         # must match the model picked in the app sidebar
TEXT_COL        = None                              # None = auto-detect, or e.g. "reflection_answer_english"
SPLIT_SENTENCES = True
MIN_WORDS       = 3
DEVICE          = "cuda"                            # "cuda" on Artemis, "mps" on a Mac, "cpu" otherwise
# ──────────────────────────────────────────────────────────────────────────────

CACHE_DIR = get_cache_dir(PROJECT_ROOT, DATASET_NAME)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DOCS_FILE, EMBEDDINGS_FILE = get_precomputed_filenames(
    CACHE_DIR, CSV_PATH, EMBEDDING_MODEL, SPLIT_SENTENCES, TEXT_COL, MIN_WORDS
)

print(f"Dataset : {DATASET_NAME}")
print(f"CSV     : {CSV_PATH}")
print(f"Model   : {EMBEDDING_MODEL}  (device={DEVICE})")
print(f"Docs   -> {DOCS_FILE}")
print(f"Embeds -> {EMBEDDINGS_FILE}")

docs, embeddings = preprocess_and_embed(
    CSV_PATH,
    model_name=EMBEDDING_MODEL,
    text_col=TEXT_COL,
    split_sentences=SPLIT_SENTENCES,
    min_words=MIN_WORDS,
    device=DEVICE,
)

embeddings = np.asarray(embeddings, dtype=np.float32)
np.save(EMBEDDINGS_FILE, embeddings)
with open(DOCS_FILE, "w", encoding="utf-8") as f:
    json.dump(docs, f, ensure_ascii=False)

print(f"\nDone. {len(docs)} docs  ->  embeddings shape {embeddings.shape}")
print("Upload these two files to the Space's cache/ folder to skip embedding there.")
