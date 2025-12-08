"""
File: app.py
Description: Unified MOSAIC App (Lite + Pro).
             Switches between CPU/Lite and GPU/LLM modes automatically based on environment variables.
"""

# =====================================================================
# Imports
# =====================================================================

import os
import sys
import json
import re
import logging
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import nltk

# Standard ML Imports
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
import datamapplot
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. DETECT MODE ---
# We check if the user has enabled the LLM in Hugging Face Secrets/Env Vars
ENABLE_LLM = os.getenv("ENABLE_LLM", "False").lower() in ("true", "1", "yes")

# Try to import LLM libraries only if enabled
LLM_MODULES_AVAILABLE = False
if ENABLE_LLM:
    try:
        from llama_cpp import Llama
        from bertopic.representation import LlamaCPP
        LLM_MODULES_AVAILABLE = True
        logger.info("🟢 LLM Modules imported successfully.")
    except ImportError as e:
        logger.warning(f"🔴 ENABLE_LLM is True, but libraries are missing: {e}. Falling back to Lite mode.")
        ENABLE_LLM = False

# =====================================================================
# NLTK setup
# =====================================================================

NLTK_DATA_DIR = "/usr/local/share/nltk_data"
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_DIR)

for resource in ("punkt_tab", "punkt"):
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        try:
            nltk.download(resource, download_dir=NLTK_DATA_DIR)
        except Exception:
            pass

# =====================================================================
# Path / Cache Utils
# =====================================================================

# Fallback path logic (works without 'mosaic' package)
def _env(key: str, default: str) -> Path:
    val = os.getenv(key, default)
    return Path(val).expanduser().resolve()

_DATA_ROOT = _env("MOSAIC_DATA", str(Path(__file__).parent / "data"))
PROC_DIR = _DATA_ROOT / "preprocessed"
CACHE_DIR = PROC_DIR / "cache"
EVAL_DIR = _env("MOSAIC_EVAL", str(Path(__file__).parent / "eval"))

for p in [PROC_DIR, CACHE_DIR, EVAL_DIR]:
    p.mkdir(parents=True, exist_ok=True)

def _slugify(s: str) -> str:
    s = s.strip()
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s) or "DATASET"

def _cleanup_old_cache(current_slug: str):
    if not CACHE_DIR.exists(): return
    for p in CACHE_DIR.glob("precomputed_*.npy"):
        if current_slug not in p.name:
            try:
                p.unlink()
            except Exception: pass

# =====================================================================
# Streamlit App
# =====================================================================

st.set_page_config(page_title="MOSAIC Dashboard", layout="wide")
st.title("MOSAIC: Topic Modelling Dashboard")

# --- Status Indicator ---
if ENABLE_LLM:
    st.info("🟢 **Pro Mode Active:** LLM Labeling (Llama-3-8B) is ENABLED.")
else:
    st.warning("🟡 **Lite Mode Active:** Running on CPU (Keyword labels only).")

# =====================================================================
# Helper Functions
# =====================================================================

@st.cache_resource
def load_embedding_model(model_name):
    return SentenceTransformer(model_name)

@st.cache_resource
def load_llm_model():
    """Loads LlamaCPP model only if enabled."""
    if not ENABLE_LLM or not LLM_MODULES_AVAILABLE:
        return None
    
    status_container = st.empty()
    status_container.info("⏳ Loading Llama-3-8B (Quantized)... This may take 1-2 minutes.")
    
    try:
        model_repo = "NousResearch/Meta-Llama-3-8B-Instruct-GGUF"
        model_file = "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
        model_path = hf_hub_download(repo_id=model_repo, filename=model_file)
        
        # Offload layers to GPU if available, otherwise CPU
        llm = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=8192, verbose=False)
        status_container.success("✅ LLM Loaded!")
        return llm
    except Exception as e:
        status_container.error(f"Failed to load LLM: {e}")
        return None

@st.cache_data
def load_precomputed_data(docs_file, emb_file):
    return np.load(docs_file, allow_pickle=True).tolist(), np.load(emb_file, allow_pickle=True)

def get_config_hash(cfg):
    return json.dumps(cfg, sort_keys=True)

# =====================================================================
# Topic Modeling Core
# =====================================================================

@st.cache_data
def perform_topic_modeling(_docs, _embeddings, config_hash):
    _docs = list(_docs)
    _embeddings = np.ascontiguousarray(_embeddings, dtype=np.float32)
    config = json.loads(config_hash)

    if "ngram_range" in config["vectorizer_params"]:
        config["vectorizer_params"]["ngram_range"] = tuple(config["vectorizer_params"]["ngram_range"])

    # --- Representation Model Logic (The Switch) ---
    rep_model = None
    if ENABLE_LLM and config.get("use_llm", False):
        llm = load_llm_model()
        if llm:
            prompt = "Q:\nI have a topic described by keywords: '[KEYWORDS]'.\nThe documents are: [DOCUMENTS]\nProvide a short label (5 words max).\nA:"
            rep_model = {"LLM": LlamaCPP(llm, prompt=prompt, nr_docs=10, doc_length=200, tokenizer="whitespace")}

    # --- BERTopic Setup ---
    topic_model = BERTopic(
        umap_model=UMAP(random_state=42, metric="cosine", **config["umap_params"]),
        hdbscan_model=HDBSCAN(metric="euclidean", prediction_data=True, **config["hdbscan_params"]),
        vectorizer_model=CountVectorizer(**config["vectorizer_params"]) if config["use_vectorizer"] else None,
        representation_model=rep_model,
        top_n_words=config["bt_params"]["top_n_words"],
        nr_topics=None if config["bt_params"]["nr_topics"] == "auto" else int(config["bt_params"]["nr_topics"]),
        verbose=False
    )

    topics, _ = topic_model.fit_transform(_docs, _embeddings)
    info = topic_model.get_topic_info()

    # --- Label Extraction ---
    if rep_model and "LLM" in topic_model.get_topics(full=True):
        raw_labels = [label[0][0] for label in topic_model.get_topics(full=True)["LLM"].values()]
        final_labels = [l.split(":")[-1].strip().strip('"') if l else "Unlabelled" for l in raw_labels]
        all_labels = [final_labels[t + topic_model._outliers] if t != -1 else "Unlabelled" for t in topics]
    else:
        name_map = info.set_index("Topic")["Name"].to_dict()
        all_labels = [name_map[t] for t in topics]

    # --- Visualization Data ---
    reduced = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric="cosine", random_state=42).fit_transform(_embeddings)
    
    outlier_pct = 0
    if -1 in info.Topic.values:
        outlier_pct = (info.Count[info.Topic == -1].iloc[0] / info.Count.sum()) * 100

    return topic_model, reduced, all_labels, len(info) - 1, outlier_pct

# =====================================================================
# Main UI Logic
# =====================================================================

st.sidebar.header("Data & Model")
source = st.sidebar.radio("Data Source", ["Server CSV", "Upload CSV"])
CSV_PATH = None

if source == "Server CSV":
    csvs = [str(p) for p in sorted(PROC_DIR.glob("*.csv"))]
    if csvs: CSV_PATH = st.sidebar.selectbox("Select File", csvs)
else:
    up = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if up:
        safe_name = _slugify(os.path.splitext(up.name)[0])
        _cleanup_old_cache(safe_name)
        CSV_PATH = str(PROC_DIR / f"{safe_name}.csv")
        pd.read_csv(up).to_csv(CSV_PATH, index=False)
        st.success(f"Saved: {safe_name}")

if CSV_PATH:
    # --- Data Loading ---
    df = pd.read_csv(CSV_PATH)
    
    # Try to find text column
    text_cols = [c for c in df.columns if df[c].dtype == object]
    if not text_cols:
        st.error("No text columns found.")
        st.stop()
        
    # Auto-pick "reflection_answer_english" if present
    default_idx = 0
    for i, col in enumerate(text_cols):
        if "reflection" in col or "text" in col:
            default_idx = i
            break
            
    selected_text_col = st.sidebar.selectbox("Text Column", text_cols, index=default_idx)
    
    # --- Config ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Params")
    nr_topics = st.sidebar.text_input("Topics (auto or int)", "auto")
    
    # Run Button
    if st.sidebar.button("Run Analysis", type="primary"):
        with st.spinner("Processing..."):
            docs = df[selected_text_col].dropna().astype(str).tolist()
            
            # Simple embedding (In real app, cache this!)
            emb_model = load_embedding_model("BAAI/bge-small-en-v1.5")
            embeddings = emb_model.encode(docs, show_progress_bar=True)
            
            # Config
            config = {
                "umap_params": {"n_neighbors": 15, "n_components": 5, "min_dist": 0.0},
                "hdbscan_params": {"min_cluster_size": 10, "min_samples": 5},
                "