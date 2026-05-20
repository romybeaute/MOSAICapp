#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : run_optuna.py
# description     : Optuna multi-objective BERTopic search for MOSAICapp.
#                   Loads precomputed embeddings — no re-embedding per trial.
#                   Two objectives: embedding coherence + CV coherence.
#                   Logs n_topics; filter results for target range [40-100].
# author          : Romy Beauté
# usage           : python run_optuna.py --dataset NDE --text-col cleaned_report --n_trials 50
#                   sbatch run_optuna.sh
# ==============================================================================

import argparse
import csv
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import NSGAIISampler
from bertopic import BERTopic
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
try:
    from cuml.manifold import UMAP
    UMAP_BACKEND = "cuML (GPU)"
except ImportError:
    from umap import UMAP
    UMAP_BACKEND = "umap-learn (CPU)"

from hdbscan import HDBSCAN

from mosaic_core.core_functions import get_cache_dir, get_precomputed_filenames

os.environ["TOKENIZERS_PARALLELISM"] = "true"
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("optuna.log")],
)
log = logging.getLogger(__name__)
log.info("UMAP backend: %s", UMAP_BACKEND)

TEXT_COLUMN_CANDIDATES = [
    "cleaned_report", "reflection_answer_english", "reflection_answer",
    "cleaned_text", "text", "report",
]

UMAP_N_COMPONENTS = 5
UMAP_MIN_DIST     = 0.0
TOP_N_WORDS       = 10
RANDOM_SEED       = 42


# ── Coherence metrics ──────────────────────────────────────────────────────────

def compute_embedding_coherence(topic_model, docs, embeddings):
    """
    Mean pairwise cosine similarity (upper triangle) within each topic.
    More rigorous than centroid-based: every pair of docs is compared.
    """
    documents_df = pd.DataFrame({"Doc": docs, "Topic": topic_model.topics_})
    embeddings = np.array(embeddings)
    documents_df["Embedding"] = list(embeddings)

    scores = []
    for topic_id in documents_df["Topic"].unique():
        if topic_id == -1:
            continue
        topic_df = documents_df[documents_df["Topic"] == topic_id]
        if len(topic_df) < 2:
            continue
        vecs = np.vstack(topic_df["Embedding"].values)
        sim_matrix = cosine_similarity(vecs)
        upper = sim_matrix[np.triu_indices(len(topic_df), k=1)]
        scores.append(float(np.mean(upper)))

    return float(np.mean(scores)) if scores else 0.0


def compute_cv_coherence(topic_model, docs):
    """
    Gensim CV coherence using BERTopic's own vectorizer tokenizer.
    Passes corpus to CoherenceModel for correctness.
    """
    unique_topics = sorted(t for t in set(topic_model.topics_) if t != -1)

    topic_words = []
    for tid in unique_topics:
        words = [w for w, _ in topic_model.get_topic(tid)]
        if any(w.strip() for w in words):
            topic_words.append(words)

    if not topic_words:
        return float("nan")

    vectorizer = topic_model.vectorizer_model
    tokenizer  = vectorizer.build_tokenizer()
    tokens     = [tokenizer(doc) for doc in docs]
    dictionary = Dictionary(tokens)
    corpus     = [dictionary.doc2bow(t) for t in tokens]

    try:
        cm = CoherenceModel(
            topics=topic_words,
            texts=tokens,
            corpus=corpus,
            dictionary=dictionary,
            coherence="c_v",
        )
        return float(cm.get_coherence())
    except Exception as e:
        log.warning("CV coherence failed: %s", e)
        return float("nan")


# ── Main class ─────────────────────────────────────────────────────────────────

class OptunaSearchBERTopic:
    def __init__(self, dataset, csv_path, embedding_model, text_col,
                 split_sentences, min_words, condition, n_trials,
                 target_min, target_max,
                 mcs_min, mcs_max, ms_min, ms_max, nn_min, nn_max,
                 subsample=None):
        self.dataset         = dataset
        self.csv_path        = csv_path
        self.embedding_model = embedding_model
        self.text_col        = text_col        # used to find the right precomputed file
        self.split_sentences = split_sentences  # used to find the right precomputed file
        self.min_words       = min_words
        self.condition       = condition
        self.n_trials        = n_trials
        self.target_min      = target_min
        self.target_max      = target_max
        self.subsample       = subsample       # None = use all docs
        # search ranges (set via CLI — easy to adjust)
        self.mcs_min = mcs_min
        self.mcs_max = mcs_max
        self.ms_min  = ms_min
        self.ms_max  = ms_max
        self.nn_min  = nn_min
        self.nn_max  = nn_max

        self.project_root = Path(__file__).parent
        self.cache_dir    = get_cache_dir(self.project_root, self.dataset)

        self.docs_file, self.emb_file = get_precomputed_filenames(
            self.cache_dir, self.csv_path, self.embedding_model,
            self.split_sentences, self.text_col, self.min_words,
        )

        sanitized_model = self.embedding_model.replace("/", "_")
        tag = f"{self.condition}_" if self.condition else ""
        results_dir = self.project_root / "outputs" / "optuna"
        results_dir.mkdir(parents=True, exist_ok=True)
        self.results_csv = results_dir / f"OPTUNA_{tag}{self.dataset}_{sanitized_model}_results.csv"
        self.study_db    = results_dir / f"OPTUNA_{tag}{self.dataset}_{sanitized_model}.db"

        self.vectorizer = CountVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_df=0.95,
            min_df=2,
            lowercase=True,
        )

    def load_precomputed(self):
        if not Path(self.emb_file).exists():
            raise FileNotFoundError(
                f"Embeddings not found: {self.emb_file}\n"
                "Run run_pipeline.py first."
            )
        if not Path(self.docs_file).exists():
            raise FileNotFoundError(
                f"Docs not found: {self.docs_file}\n"
                "Run run_pipeline.py first."
            )
        log.info("Loading docs       → %s", self.docs_file)
        with open(self.docs_file, encoding="utf-8") as f:
            docs = json.load(f)
        log.info("Loading embeddings → %s", self.emb_file)
        embeddings = np.load(self.emb_file).astype(np.float32)
        log.info("Loaded %d docs, embeddings %s", len(docs), embeddings.shape)

        if self.subsample and self.subsample < len(docs):
            rng = np.random.default_rng(RANDOM_SEED)
            idx = rng.choice(len(docs), size=self.subsample, replace=False)
            idx.sort()
            docs = [docs[i] for i in idx]
            embeddings = embeddings[idx]
            log.info("Subsampled to %d docs for Optuna search (full data used in final pipeline)", len(docs))

        return docs, embeddings

    def _search_space(self, trial):
        return {
            "min_cluster_size": trial.suggest_int("min_cluster_size", self.mcs_min, self.mcs_max),
            "min_samples":      trial.suggest_int("min_samples",      self.ms_min,  self.ms_max),
            "n_neighbors":      trial.suggest_int("n_neighbors",      self.nn_min,  self.nn_max),
        }

    def objective(self, trial):
        params = self._search_space(trial)

        topic_model = BERTopic(
            umap_model=UMAP(
                n_neighbors=params["n_neighbors"],
                n_components=UMAP_N_COMPONENTS,
                min_dist=UMAP_MIN_DIST,
                metric="cosine",
                random_state=RANDOM_SEED,
            ),
            hdbscan_model=HDBSCAN(
                min_cluster_size=params["min_cluster_size"],
                min_samples=params["min_samples"],
                metric="euclidean",
                gen_min_span_tree=False,
                prediction_data=True,
            ),
            vectorizer_model=self.vectorizer,
            top_n_words=TOP_N_WORDS,
            verbose=False,
        )

        try:
            log.info("Trial %d | UMAP+HDBSCAN starting (mcs=%d ms=%d nn=%d)...",
                     trial.number, params["min_cluster_size"], params["min_samples"], params["n_neighbors"])
            topics, _ = topic_model.fit_transform(self.docs, self.embeddings)
            log.info("Trial %d | clustering done, computing coherence...", trial.number)
        except Exception as e:
            log.warning("Trial %d pruned: %s", trial.number, e)
            raise optuna.exceptions.TrialPruned()

        n_topics    = len([t for t in set(topics) if t != -1])
        n_outliers  = sum(1 for t in topics if t == -1)
        outlier_pct = 100 * n_outliers / len(topics)

        emb_coh = compute_embedding_coherence(topic_model, self.docs, self.embeddings)
        cv_coh  = compute_cv_coherence(topic_model, self.docs)

        trial.set_user_attr("n_topics",     n_topics)
        trial.set_user_attr("outlier_pct",  round(outlier_pct, 1))
        trial.set_user_attr("cv_coherence", round(cv_coh, 4))

        log.info(
            "Trial %3d | mcs=%3d ms=%2d nn=%2d | topics=%3d outliers=%.1f%% "
            "| emb_coh=%.4f cv_coh=%.4f",
            trial.number,
            params["min_cluster_size"], params["min_samples"], params["n_neighbors"],
            n_topics, outlier_pct, emb_coh, cv_coh,
        )

        return emb_coh, cv_coh

    def save_callback(self, study, trial):
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        row = {
            "trial":            trial.number,
            "emb_coherence":    trial.values[0],
            "cv_coherence":     trial.values[1],
            "n_topics":         trial.user_attrs.get("n_topics", ""),
            "outlier_pct":      trial.user_attrs.get("outlier_pct", ""),
            "min_cluster_size": trial.params["min_cluster_size"],
            "min_samples":      trial.params["min_samples"],
            "n_neighbors":      trial.params["n_neighbors"],
        }
        write_header = not self.results_csv.exists()
        with open(self.results_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def run(self):
        self.docs, self.embeddings = self.load_precomputed()

        study_name   = f"mosaic-{self.dataset}-{self.condition or 'all'}-optuna"
        storage_name = f"sqlite:///{self.study_db}"

        try:
            study = optuna.load_study(study_name=study_name, storage=storage_name)
            log.info("Resuming study '%s' (%d existing trials)", study_name, len(study.trials))
        except Exception:
            log.info("Starting new study '%s'", study_name)
            study = optuna.create_study(
                study_name=study_name,
                storage=storage_name,
                sampler=NSGAIISampler(seed=RANDOM_SEED),
                directions=["maximize", "maximize"],
            )

        study.optimize(self.objective, n_trials=self.n_trials, callbacks=[self.save_callback])

        log.info("── Results saved to %s", self.results_csv)
        log.info("── Pareto-optimal trials in target range [%d, %d] topics:", self.target_min, self.target_max)

        in_range = [
            t for t in study.best_trials
            if self.target_min <= t.user_attrs.get("n_topics", 0) <= self.target_max
        ]
        in_range.sort(key=lambda t: t.values[0], reverse=True)

        if in_range:
            for t in in_range[:5]:
                log.info(
                    "  Trial %d | emb_coh=%.4f cv_coh=%.4f | topics=%d outliers=%.1f%% | %s",
                    t.number, t.values[0], t.values[1],
                    t.user_attrs["n_topics"], t.user_attrs["outlier_pct"], t.params,
                )
        else:
            log.info("  No Pareto-optimal trials in [%d, %d]. Check %s for all trials.",
                     self.target_min, self.target_max, self.results_csv)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optuna BERTopic search (MOSAICapp)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    parser.add_argument("--dataset",         type=str, default="NDE",
                        help="Dataset name (must match sidebar value in app.py)")
    parser.add_argument("--csv",             type=str,
                        default="data/NDE/preprocessed/NDE_reports_grouped.csv",
                        help="Path to CSV (same as used in run_pipeline.py)")
    parser.add_argument("--embedding-model", type=str, default="Qwen/Qwen3-Embedding-0.6B",
                        help="Embedding model (must match run_pipeline.py)")
    parser.add_argument("--text-col",        type=str, default=None,
                        help="Text column name (None = auto-detect). "
                             "Used to locate the correct precomputed embeddings file.")
    parser.add_argument("--sentences",       action="store_true", default=True,
                        help="Sentence-level granularity (must match run_pipeline.py)")
    parser.add_argument("--min-words",       type=int, default=3,
                        help="Min words per sentence (must match run_pipeline.py)")
    parser.add_argument("--condition",       type=str, default=None,
                        help="Sub-condition label — used in output filenames only")

    # ── Search ────────────────────────────────────────────────────────────────
    parser.add_argument("--n_trials",        type=int, default=100)
    parser.add_argument("--subsample",       type=int, default=15000,
                        help="Number of docs to use per trial (None = all). "
                             "Keeps each trial fast (~2 min) on CPU UMAP. "
                             "Best params are then applied to full data in run_pipeline.py.")

    # Hyperparameter ranges — pass [min max] for each param
    parser.add_argument("--min-cluster-size", type=int, nargs=2, default=[10, 100],
                        metavar=("MIN", "MAX"),
                        help="Search range for HDBSCAN min_cluster_size. "
                             "Lower = more smaller clusters. e.g. --min-cluster-size 10 60")
    parser.add_argument("--min-samples",      type=int, nargs=2, default=[5, 30],
                        metavar=("MIN", "MAX"),
                        help="Search range for HDBSCAN min_samples. "
                             "Controls how conservative clustering is (higher = more outliers). "
                             "Keep [2,20] for 60k sentences. e.g. --min-samples 2 20")
    parser.add_argument("--n-neighbors",      type=int, nargs=2, default=[5, 30],
                        metavar=("MIN", "MAX"),
                        help="Search range for UMAP n_neighbors. "
                             "Lower = more local structure = more clusters. e.g. --n-neighbors 5 25")

    # Target topic range — filter Pareto front in summary
    parser.add_argument("--target-min",      type=int, default=40,
                        help="Min acceptable number of topics")
    parser.add_argument("--target-max",      type=int, default=100,
                        help="Max acceptable number of topics")

    args = parser.parse_args()

    search = OptunaSearchBERTopic(
        dataset         = args.dataset,
        csv_path        = args.csv,
        embedding_model = args.embedding_model,
        text_col        = args.text_col,
        split_sentences = args.sentences,
        min_words       = args.min_words,
        condition       = args.condition,
        n_trials        = args.n_trials,
        target_min      = args.target_min,
        target_max      = args.target_max,
        mcs_min         = args.min_cluster_size[0],
        mcs_max         = args.min_cluster_size[1],
        ms_min          = args.min_samples[0],
        ms_max          = args.min_samples[1],
        nn_min          = args.n_neighbors[0],
        nn_max          = args.n_neighbors[1],
        subsample       = args.subsample,
    )
    search.run()
