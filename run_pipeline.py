"""
Standalone HPC pipeline for MOSAIC, no Streamlit required.

Outputs are saved using the exact same paths and filenames as the Streamlit app,
so can open app.py afterwards and load precomputed results without re-running.

Usage:
    python run_pipeline.py
    sbatch run_pipeline.sh
"""

import json
import logging
import re
from pathlib import Path

import numpy as np


def _load_secret(key: str) -> str:
    """Read a single key from .streamlit/secrets.toml without extra dependencies."""
    secrets_path = Path(__file__).parent / ".streamlit" / "secrets.toml"
    for line in secrets_path.read_text().splitlines():
        if line.strip().startswith(key):
            _, _, val = line.partition("=")
            return val.strip().strip('"').strip("'")
    raise KeyError(f"{key} not found in {secrets_path}")


from mosaic_core.core_functions import (
    preprocess_and_embed,
    run_topic_model,
    generate_llm_labels,
    get_cache_dir,
    get_precomputed_filenames,
    labels_cache_path,
    get_config_hash,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("pipeline.log")],
)
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).parent
DATASET_NAME   = "NDE"          # must match the sidebar value in app.py
CSV_PATH       = "data/preprocessed/NDE/your_big_dataset.csv"   # ← change this
TEXT_COL       = None              # None = auto-detect; or e.g. "reflection_answer_english"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEVICE         = "cuda"
SPLIT_SENTENCES = True
MIN_WORDS      = 3
HF_TOKEN       = _load_secret("HF_TOKEN")

CONFIG = {
    "umap_params": {
        "n_neighbors": 30,
        "n_components": 5,
        "min_dist": 0.0,
    },
    "hdbscan_params": {
        "min_cluster_size": 100,
        "min_samples": 50,
    },
    "use_vectorizer": True,
    "vectorizer_params": {
        "ngram_range": [1, 2],
        "stop_words": "english",
    },
    "bt_params": {
        "nr_topics": "auto",
        "top_n_words": 10,
    },
}

# ── Paths — identical to what app.py resolves ─────────────────────────────────
CACHE_DIR = get_cache_dir(PROJECT_ROOT, DATASET_NAME)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
log.info(f"Cache dir: {CACHE_DIR}")

DOCS_FILE, EMBEDDINGS_FILE = get_precomputed_filenames(
    CACHE_DIR, CSV_PATH, EMBEDDING_MODEL, SPLIT_SENTENCES, TEXT_COL, MIN_WORDS
)
log.info(f"Docs      → {DOCS_FILE}")
log.info(f"Embeddings → {EMBEDDINGS_FILE}")

# ── Step 1: Preprocess + Embed ────────────────────────────────────────────────
log.info("Step 1 — preprocessing and embedding")

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

log.info(f"Embedded {len(docs)} sentences  →  shape {embeddings.shape}")

# ── Step 2: Topic modelling ───────────────────────────────────────────────────
log.info("Step 2 — topic modelling")

config_hash = get_config_hash(CONFIG)
topic_model, reduced_2d, topics = run_topic_model(docs, embeddings, CONFIG)

topic_model.save(str(CACHE_DIR / "topic_model"))
np.save(CACHE_DIR / "reduced_2d.npy", reduced_2d)
with open(CACHE_DIR / "topics.json", "w") as f:
    json.dump(topics, f)

n_topics   = len(set(t for t in topics if t != -1))
n_outliers = sum(1 for t in topics if t == -1)
log.info(f"Topics: {n_topics}  |  Outliers: {n_outliers} ({100*n_outliers/len(topics):.1f}%)")

# ── Step 3: LLM labelling ─────────────────────────────────────────────────────
log.info("Step 3 — LLM labelling with Qwen3")

LLM_MODEL = "Qwen/Qwen3-8B-Instruct"

labels = generate_llm_labels(
    topic_model,
    hf_token=HF_TOKEN,
    model_id=LLM_MODEL,
    max_topics=100,
    max_docs_per_topic=10,
    doc_char_limit=600,
    temperature=0.2,
)

# Save to the exact path the app's cache lookup expects
llm_cache = labels_cache_path(CACHE_DIR, config_hash, LLM_MODEL)
llm_cache.write_text(
    json.dumps({str(k): v for k, v in labels.items()}, indent=2),
    encoding="utf-8",
)

log.info(f"Labelled {len(labels)} topics  →  {llm_cache}")
log.info("Done. Open app.py and choose 'Use preprocessed CSV on server' to visualise.")
