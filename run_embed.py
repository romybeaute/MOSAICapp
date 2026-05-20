"""
Embed a dataset on HPC and save .npy + .json files compatible with the app and run_optuna.py.

Usage:
    python run_embed.py
    sbatch run_embed.sh
"""

import json
import logging
from pathlib import Path

import numpy as np

from mosaic_core.core_functions import (
    preprocess_and_embed,
    get_cache_dir,
    get_precomputed_filenames,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("embed.log")],
)
log = logging.getLogger(__name__)

# ── Configuration — edit these ────────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).parent
DATASET_NAME    = "NDE"
CSV_PATH        = "data/NDE/preprocessed/NDE_reports_grouped.csv"
TEXT_COL        = "cleaned_report"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"
DEVICE          = "cuda"
SPLIT_SENTENCES = True
MIN_WORDS       = 3
# ─────────────────────────────────────────────────────────────────────────────

CACHE_DIR = get_cache_dir(PROJECT_ROOT, DATASET_NAME)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DOCS_FILE, EMBEDDINGS_FILE = get_precomputed_filenames(
    CACHE_DIR, CSV_PATH, EMBEDDING_MODEL, SPLIT_SENTENCES, TEXT_COL, MIN_WORDS
)

if Path(EMBEDDINGS_FILE).exists() and Path(DOCS_FILE).exists():
    log.info("Embeddings already exist — delete to re-run:")
    log.info("  %s", EMBEDDINGS_FILE)
    log.info("  %s", DOCS_FILE)
else:
    log.info("Embedding model : %s", EMBEDDING_MODEL)
    log.info("CSV             : %s", CSV_PATH)
    log.info("Output docs     : %s", DOCS_FILE)
    log.info("Output embeddings: %s", EMBEDDINGS_FILE)

    docs, embeddings = preprocess_and_embed(
        CSV_PATH,
        model_name=EMBEDDING_MODEL,
        text_col=TEXT_COL,
        split_sentences=SPLIT_SENTENCES,
        min_words=MIN_WORDS,
        device=DEVICE,
    )

    np.save(EMBEDDINGS_FILE, embeddings)
    with open(DOCS_FILE, "w", encoding="utf-8") as f:
        json.dump(docs, f)

    log.info("Done — embedded %d sentences, shape %s", len(docs), embeddings.shape)
    log.info("These files are automatically picked up by the app and run_optuna.py")
