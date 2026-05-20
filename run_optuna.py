"""
Optuna hyperparameter search for BERTopic — MOSAICapp version.

Reuses precomputed embeddings (no re-embedding per trial).
Searches only: min_cluster_size, min_samples, n_neighbors.

Two objectives:
  1. Embedding coherence (higher = tighter, more meaningful topics)
  2. n_topics (used to filter results in [TARGET_MIN, TARGET_MAX])

Usage:
    python run_optuna.py
    sbatch run_optuna.sh
"""

import json
import logging
import csv
from pathlib import Path

import numpy as np
import optuna
from optuna.samplers import NSGAIISampler
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP
from hdbscan import HDBSCAN

optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("optuna.log")],
)
log = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).parent
DATASET_NAME   = "NDE"
CSV_PATH       = "data/NDE/preprocessed/NDE_reports_grouped.csv"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
TEXT_COL       = None      # None = auto-detect
SPLIT_SENTENCES = True
MIN_WORDS      = 3

# Fixed params (not searched)
UMAP_N_COMPONENTS = 5
UMAP_MIN_DIST     = 0.0
TOP_N_WORDS       = 10

# Target cluster range — used to filter results after the search
TARGET_MIN_TOPICS = 40
TARGET_MAX_TOPICS = 100

N_TRIALS = 50

# ── Paths ──────────────────────────────────────────────────────────────────────
from mosaic_core.core_functions import get_cache_dir, get_precomputed_filenames

CACHE_DIR = get_cache_dir(PROJECT_ROOT, DATASET_NAME)
DOCS_FILE, EMBEDDINGS_FILE = get_precomputed_filenames(
    CACHE_DIR, CSV_PATH, EMBEDDING_MODEL, SPLIT_SENTENCES, TEXT_COL, MIN_WORDS
)

RESULTS_DIR = PROJECT_ROOT / "outputs" / "optuna"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV  = RESULTS_DIR / "optuna_results.csv"
STUDY_DB     = RESULTS_DIR / "optuna_study.db"


# ── Load precomputed data ──────────────────────────────────────────────────────
log.info(f"Loading docs from  {DOCS_FILE}")
log.info(f"Loading embeddings from {EMBEDDINGS_FILE}")

with open(DOCS_FILE, encoding="utf-8") as f:
    docs = json.load(f)

embeddings = np.load(EMBEDDINGS_FILE).astype(np.float32)
log.info(f"Loaded {len(docs)} docs, embeddings shape {embeddings.shape}")


# ── Coherence metric ───────────────────────────────────────────────────────────
def embedding_coherence(topic_model, topics, embeddings):
    """Mean intra-topic cosine similarity to topic centroid (higher = better)."""
    topic_ids = [t for t in set(topics) if t != -1]
    if not topic_ids:
        return 0.0
    scores = []
    for tid in topic_ids:
        idx = [i for i, t in enumerate(topics) if t == tid]
        if len(idx) < 2:
            continue
        vecs = embeddings[idx]
        centroid = vecs.mean(axis=0, keepdims=True)
        sims = cosine_similarity(vecs, centroid).flatten()
        scores.append(sims.mean())
    return float(np.mean(scores)) if scores else 0.0


# ── Optuna objective ───────────────────────────────────────────────────────────
def objective(trial):
    min_cluster_size = trial.suggest_int("min_cluster_size", 10, 80)
    min_samples      = trial.suggest_int("min_samples", 2, 40)
    n_neighbors      = trial.suggest_int("n_neighbors", 5, 30)

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=UMAP_N_COMPONENTS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        prediction_data=True,
    )
    vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words="english")

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        top_n_words=TOP_N_WORDS,
        verbose=False,
    )

    try:
        topics, _ = topic_model.fit_transform(docs, embeddings)
    except Exception as e:
        raise optuna.exceptions.TrialPruned()

    n_topics   = len([t for t in set(topics) if t != -1])
    n_outliers = sum(1 for t in topics if t == -1)
    outlier_pct = 100 * n_outliers / len(topics)
    coh = embedding_coherence(topic_model, topics, embeddings)

    trial.set_user_attr("n_topics", n_topics)
    trial.set_user_attr("n_outliers", n_outliers)
    trial.set_user_attr("outlier_pct", round(outlier_pct, 1))

    log.info(
        f"Trial {trial.number:3d} | "
        f"mcs={min_cluster_size:3d} ms={min_samples:2d} nn={n_neighbors:2d} | "
        f"topics={n_topics:3d} outliers={outlier_pct:.1f}% | "
        f"coherence={coh:.4f}"
    )

    return coh, n_topics


# ── Callback: save each trial to CSV ──────────────────────────────────────────
def save_callback(study, trial):
    if trial.state != optuna.trial.TrialState.COMPLETE:
        return
    row = {
        "trial":            trial.number,
        "coherence":        trial.values[0],
        "n_topics":         trial.values[1],
        "min_cluster_size": trial.params["min_cluster_size"],
        "min_samples":      trial.params["min_samples"],
        "n_neighbors":      trial.params["n_neighbors"],
        "outlier_pct":      trial.user_attrs.get("outlier_pct", ""),
    }
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ── Run study ─────────────────────────────────────────────────────────────────
study_name   = f"mosaic-{DATASET_NAME}-optuna"
storage_name = f"sqlite:///{STUDY_DB}"

try:
    study = optuna.load_study(study_name=study_name, storage=storage_name)
    log.info(f"Resuming study '{study_name}' ({len(study.trials)} existing trials)")
except Exception:
    log.info(f"Starting new study '{study_name}'")
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        sampler=NSGAIISampler(seed=42),
        directions=["maximize", "maximize"],
    )

study.optimize(objective, n_trials=N_TRIALS, callbacks=[save_callback])

# ── Summary ───────────────────────────────────────────────────────────────────
log.info("\n── Results saved to %s", RESULTS_CSV)
log.info("── Pareto front trials in target range [%d, %d] topics:", TARGET_MIN_TOPICS, TARGET_MAX_TOPICS)

in_range = [
    t for t in study.best_trials
    if TARGET_MIN_TOPICS <= t.user_attrs.get("n_topics", 0) <= TARGET_MAX_TOPICS
]
in_range.sort(key=lambda t: t.values[0], reverse=True)

if in_range:
    for t in in_range[:5]:
        log.info(
            "  Trial %d | coherence=%.4f | topics=%d | outliers=%.1f%% | params=%s",
            t.number, t.values[0], t.user_attrs["n_topics"],
            t.user_attrs["outlier_pct"], t.params,
        )
else:
    log.info("  No Pareto-optimal trials found in [%d, %d] range.", TARGET_MIN_TOPICS, TARGET_MAX_TOPICS)
    log.info("  Check outputs/optuna/optuna_results.csv for all trials.")
