"""
Core analysis functions for MOSAIC topic modeling.

This module provides preprocessing, embedding, topic modeling, and LLM labeling
for phenomenological text analysis. No Streamlit dependencies.
"""

import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import nltk
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)


# --- Text column detection ---

TEXT_COLUMN_CANDIDATES = [
    "reflection_answer_english",
    "reflection_answer",
    "text",
    "report",
]


def pick_text_column(df):
    """Return first column matching TEXT_COLUMN_CANDIDATES, or None."""
    for col in TEXT_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    return None


def list_text_columns(df):
    """Return all column names."""
    return list(df.columns)


# --- String utilities ---

def slugify(s):
    """Convert string to filesystem-safe name."""
    s = s.strip()
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s or "DATASET"


def clean_label(raw):
    """
    Normalise LLM-generated topic label.
    
    Takes first line, strips quotes/punctuation, removes wrapper phrases
    like "Experience of". Returns "Unlabelled" if empty.
    """
    text = (raw or "").strip()
    lines = text.splitlines()
    text = lines[0].strip() if lines else ""
    text = text.strip(' "\'`')
    text = re.sub(r"[.:\-–—]+$", "", text).strip()
    text = re.sub(r"[^\w\s]", "", text).strip()
    
    text = re.sub(
        r"^(Experiential(?:\s+Phenomenon)?|Experience of|Subjective Experience of|Phenomenon of)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\s+(experience|experiences|phenomenon|state|states)$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip() or "Unlabelled"


# --- Config and caching utilities ---

def get_config_hash(cfg):
    """Generate a hash string from config dict for caching."""
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def make_run_id(cfg):
    """Generate unique run ID from timestamp and config hash."""
    h = get_config_hash(cfg)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{h}"


def cleanup_old_cache(cache_dir, current_slug):
    """Delete cached .npy files that don't match current dataset slug."""
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return 0
    
    removed = 0
    for p in cache_dir.glob("precomputed_*.npy"):
        if current_slug not in p.name:
            try:
                p.unlink()
                removed += 1
            except Exception as e:
                logger.warning(f"Could not delete {p}: {e}")
    
    if removed:
        logger.info(f"Cleaned up {removed} old cache files")
    return removed


# --- NLTK setup ---

def ensure_nltk_data(data_dir=None):
    """Download NLTK punkt tokenizer if missing."""
    if data_dir and data_dir not in nltk.data.path:
        nltk.data.path.append(data_dir)
    
    for resource in ("punkt_tab", "punkt"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
            return
        except LookupError:
            pass
    
    try:
        nltk.download("punkt", download_dir=data_dir, quiet=True)
    except Exception as e:
        logger.warning(f"Could not download NLTK punkt: {e}")


# --- Embedding ---

def load_embedding_model(model_name):
    """Load a sentence-transformers model."""
    logger.info(f"Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)


def resolve_device(requested):
    """
    Resolve device string to actual device and batch size.
    
    Returns (device, batch_size) where device is 'cpu', 'cuda', or 'mps'.
    """
    if requested.lower() == "cpu":
        return "cpu", 64
    
    import torch
    if torch.cuda.is_available():
        return "cuda", 32
    if torch.backends.mps.is_available():
        return "mps", 32
    
    logger.warning("GPU requested but unavailable, using CPU")
    return "cpu", 64


# --- Preprocessing ---

def preprocess_texts(texts, split_sentences=True, min_words=3):
    """
    Clean and optionally split texts into sentences.
    
    Returns (docs, removed, stats) where stats has keys:
    total_before, total_after, removed_count
    """
    ensure_nltk_data()
    
    if split_sentences:
        units = []
        for text in texts:
            units.extend(nltk.sent_tokenize(str(text)))
    else:
        units = [str(t) for t in texts]
    
    total_before = len(units)
    
    if min_words > 0:
        docs = [u for u in units if len(u.split()) >= min_words]
        removed = [u for u in units if len(u.split()) < min_words]
    else:
        docs = units
        removed = []
    
    stats = {
        "total_before": total_before,
        "total_after": len(docs),
        "removed_count": len(removed),
    }
    return docs, removed, stats


def load_csv_texts(csv_path, text_col=None):
    """
    Load CSV and extract texts from specified or auto-detected column.
    
    Returns list of non-empty text strings.
    Raises ValueError if no valid text column found.
    """
    df = pd.read_csv(csv_path)
    
    if text_col is None:
        text_col = pick_text_column(df)
    
    if text_col is None or text_col not in df.columns:
        raise ValueError(f"No valid text column found in {csv_path}")
    
    df = df.dropna(subset=[text_col])
    df[text_col] = df[text_col].astype(str)
    return [t for t in df[text_col] if t.strip()]


def count_clean_reports(csv_path, text_col=None):
    """Count non-empty reports in CSV."""
    try:
        texts = load_csv_texts(csv_path, text_col)
        return len(texts)
    except Exception:
        return 0


def compute_embeddings(docs, model_name="BAAI/bge-small-en-v1.5", device="cpu"):
    """
    Compute sentence embeddings.
    
    Returns float32 numpy array of shape (n_docs, embedding_dim).
    """
    model = load_embedding_model(model_name)
    encode_device, batch_size = resolve_device(device)
    
    logger.info(f"Encoding {len(docs)} documents on {encode_device}")
    embeddings = model.encode(
        docs,
        show_progress_bar=True,
        batch_size=batch_size,
        device=encode_device,
        convert_to_numpy=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def preprocess_and_embed(csv_path, model_name="BAAI/bge-small-en-v1.5",
                         text_col=None, split_sentences=True, min_words=3,
                         device="cpu"):
    """
    Full pipeline: load CSV, preprocess, compute embeddings.
    
    Returns (docs, embeddings).
    """
    texts = load_csv_texts(csv_path, text_col)
    docs, removed, stats = preprocess_texts(texts, split_sentences, min_words)
    
    logger.info(f"Preprocessed {stats['total_after']} units "
                f"(removed {stats['removed_count']} short)")
    
    embeddings = compute_embeddings(docs, model_name, device)
    return docs, embeddings


# --- Topic modeling ---

def run_topic_model(docs, embeddings, config):
    """
    Fit BERTopic and compute 2D UMAP projection.
    
    Config keys:
        umap_params: dict (default: n_neighbors=15, n_components=5, min_dist=0.0)
        hdbscan_params: dict (default: min_cluster_size=10, min_samples=5)
        vectorizer_params: dict (optional)
        use_vectorizer: bool (default: True)
        bt_params: dict with nr_topics ('auto' or int), top_n_words (default: 10)
    
    Returns (topic_model, reduced_2d, topics).
    """
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    
    umap_params = config.get("umap_params", {
        "n_neighbors": 15, "n_components": 5, "min_dist": 0.0
    })
    hdbscan_params = config.get("hdbscan_params", {
        "min_cluster_size": 10, "min_samples": 5
    })
    vec_params = config.get("vectorizer_params", {}).copy()
    bt_params = config.get("bt_params", {"nr_topics": "auto", "top_n_words": 10})
    
    if "ngram_range" in vec_params and isinstance(vec_params["ngram_range"], list):
        vec_params["ngram_range"] = tuple(vec_params["ngram_range"])
    
    umap_model = UMAP(random_state=42, metric="cosine", **umap_params)
    hdbscan_model = HDBSCAN(metric="euclidean", prediction_data=True, **hdbscan_params)
    
    vectorizer = None
    if config.get("use_vectorizer", True):
        vectorizer = CountVectorizer(**vec_params)
    
    nr_topics = bt_params.get("nr_topics", "auto")
    if nr_topics == "auto":
        nr_topics = None
    else:
        nr_topics = int(nr_topics)
    
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        top_n_words=bt_params.get("top_n_words", 10),
        nr_topics=nr_topics,
        verbose=False,
    )
    
    topics, _ = topic_model.fit_transform(docs, embeddings)
    
    reduced_2d = UMAP(
        n_neighbors=15, n_components=2, min_dist=0.0,
        metric="cosine", random_state=42
    ).fit_transform(embeddings)
    
    return topic_model, reduced_2d, topics


def get_topic_labels(topic_model, topics):
    """Get keyword-based label for each document's assigned topic."""
    info = topic_model.get_topic_info()
    name_map = info.set_index("Topic")["Name"].to_dict()
    return [name_map.get(t, "Unknown") for t in topics]


def get_outlier_stats(topic_model):
    """Return (outlier_count, outlier_percentage)."""
    info = topic_model.get_topic_info()
    total = info["Count"].sum()
    
    if -1 in info["Topic"].values:
        outlier_count = int(info.loc[info["Topic"] == -1, "Count"].iloc[0])
    else:
        outlier_count = 0
    
    pct = (100.0 * outlier_count / total) if total > 0 else 0.0
    return outlier_count, pct


def get_num_topics(topic_model):
    """Return number of topics (excluding outlier topic -1)."""
    info = topic_model.get_topic_info()
    return int((info["Topic"] != -1).sum())


# --- LLM labeling ---

SYSTEM_PROMPT = """You are an expert phenomenologist analysing first-person experiential reports or microphenomenological interviews.

Your task is to assign a concise label to a cluster of similar reports by identifying the
shared lived experiential structure or process they describe.

The label must:
1. Describe what changes in experience itself (e.g. boundaries, temporality, embodiment, agency, affect, meaning).
2. Capture the underlying experiential process or structural transformation, not surface narrative details.
3. Be specific and distinctive, but at the level of experiential structure rather than anecdotal content.
4. Use phenomenological language that describes how cognitive, affective, or perceptual processes are lived, rather than analytic or evaluative abstractions.
5. Be conceptually focused on a single dominant experiential pattern.
6. Be concise and noun-phrase-like.

Constraints:
- Output ONLY the label (no explanation).
- 3–8 words.
- Avoid surface-specific details unless they reflect a recurring experiential structure.
- Avoid meta-level analytic terms (e.g. epistemic, estimation, verification, evaluation) unless they directly describe how the process is experienced.
- Avoid generic wrappers such as "experience of", "state of", or "phenomenon of".
- No punctuation, no quotes, no extra text.
- Do not explain your reasoning.
"""

USER_TEMPLATE = """Here is a cluster of participant reports describing a specific phenomenon:

{documents}

Top keywords associated with this cluster:
{keywords}

Task: Return a single scientifically precise label (3–7 words). Output ONLY the label.
"""


def get_hf_status_code(exc):
    """Extract HTTP status code from HuggingFace exception, if present."""
    resp = getattr(exc, "response", None)
    return getattr(resp, "status_code", None)


def generate_llm_labels(topic_model, hf_token, model_id="meta-llama/Meta-Llama-3-8B-Instruct",
                        max_topics=50, max_docs_per_topic=10, doc_char_limit=400,
                        temperature=0.2):
    """
    Generate topic labels via HuggingFace Inference API.
    
    Returns dict mapping topic_id to label string.
    Raises RuntimeError on 402 (payment required).
    """
    client = InferenceClient(model=model_id, token=hf_token)
    
    info = topic_model.get_topic_info()
    info = info[info["Topic"] != -1].head(max_topics)
    
    labels = {}
    logger.info(f"Generating LLM labels for {len(info)} topics")
    
    for tid in info["Topic"].tolist():
        words = topic_model.get_topic(tid) or []
        keywords = ", ".join([w for w, _ in words[:10]])
        
        try:
            reps = (topic_model.get_representative_docs(tid) or [])[:max_docs_per_topic]
        except Exception:
            reps = []
        
        reps = [r.replace("\n", " ").strip()[:doc_char_limit] for r in reps if str(r).strip()]
        docs_block = "\n".join([f"- {r}" for r in reps]) if reps else "- (No docs)"
        
        prompt = USER_TEMPLATE.format(documents=docs_block, keywords=keywords)
        
        try:
            out = client.chat_completion(
                model=model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=24,
                temperature=temperature,
                stop=["\n"],
            )
            raw = out.choices[0].message.content
            labels[int(tid)] = clean_label(raw)
            
        except Exception as e:
            code = get_hf_status_code(e)
            if code == 402:
                raise RuntimeError(
                    "HuggingFace returned 402 Payment Required. "
                    "Monthly credits exhausted—upgrade or skip LLM labeling."
                ) from e
            logger.warning(f"LLM labeling failed for topic {tid}: {e}")
            labels[int(tid)] = f"Topic {tid}"
    
    return labels


def labels_cache_path(cache_dir, config_hash, model_id):
    """Generate path for cached LLM labels."""
    safe_model = re.sub(r"[^a-zA-Z0-9_.-]", "_", model_id)
    return Path(cache_dir) / f"llm_labels_{safe_model}_{config_hash}.json"


def load_cached_labels(cache_path):
    """Load labels from cache file, returns None if not found or invalid."""
    try:
        data = json.loads(Path(cache_path).read_text(encoding="utf-8"))
        return {int(k): str(v) for k, v in data.items()}
    except Exception:
        return None


def save_labels_cache(cache_path, labels):
    """Save labels dict to cache file."""
    try:
        data = {str(k): v for k, v in labels.items()}
        Path(cache_path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning(f"Could not save labels cache: {e}")
