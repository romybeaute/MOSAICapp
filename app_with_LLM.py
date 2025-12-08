"""
File: app.py
Description: Streamlit app for advanced topic modeling on Innerspeech dataset
             with BERTopic, UMAP, HDBSCAN. Supports conditional LLM (LlamaCPP) execution.
Last Modified: 08/12/2025
"""

# =====================================================================
# Imports
# =====================================================================

from pathlib import Path
import sys
import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import nltk
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for LLM enablement via Environment Variable
# set ENABLE_LLM to "True" in Hugging Face Space Settings to activate
ENABLE_LLM = os.getenv("ENABLE_LLM", "False").lower() in ("true", "1", "yes")

# Conditional Imports for Heavy LLM Libraries
LLM_AVAILABLE = False
if ENABLE_LLM:
    try:
        from llama_cpp import Llama
        from bertopic.representation import LlamaCPP
        from huggingface_hub import hf_hub_download
        LLM_AVAILABLE = True
        logger.info("LLM modules imported successfully.")
    except ImportError as e:
        logger.warning(f"ENABLE_LLM is True, but imports failed: {e}. Falling back to Lite mode.")
        ENABLE_LLM = False

# Standard Imports
from mosaic.path_utils import CFG, raw_path, proc_path, eval_path, project_root # type: ignore

# BERTopic stack
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# Clustering/dimensionality reduction
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN

# Visualisation
import datamapplot
import matplotlib.pyplot as plt

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
        except Exception as e:
            print(f"Could not download NLTK resource {resource}: {e}")

# =====================================================================
# Path utils (MOSAIC or fallback)
# =====================================================================

try:
    from mosaic.path_utils import CFG, raw_path, proc_path, eval_path, project_root  # type: ignore
except Exception:
    def _env(key: str, default: str) -> Path:
        val = os.getenv(key, default)
        return Path(val).expanduser().resolve()

    _DATA_ROOT = _env("MOSAIC_DATA", str(Path(__file__).parent / "data"))
    _BOX_ROOT = _env("MOSAIC_BOX", str(Path(__file__).parent / "data" / "raw"))
    _EVAL_ROOT = _env("MOSAIC_EVAL", str(Path(__file__).parent / "eval"))

    CFG = {
        "data_root": str(_DATA_ROOT),
        "box_root": str(_BOX_ROOT),
        "eval_root": str(_EVAL_ROOT),
    }

    def project_root() -> Path:
        return Path(__file__).resolve().parent

    def raw_path(*parts: str) -> Path:
        return _BOX_ROOT.joinpath(*parts)

    def proc_path(*parts: str) -> Path:
        return _DATA_ROOT.joinpath(*parts)

    def eval_path(*parts: str) -> Path:
        return _EVAL_ROOT.joinpath(*parts)

# =====================================================================
# 0. Constants & Helper Functions
# =====================================================================

def _slugify(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s or "DATASET"

def _cleanup_old_cache(current_slug: str):
    if not CACHE_DIR.exists():
        return
    removed_count = 0
    for p in CACHE_DIR.glob("precomputed_*.npy"):
        if current_slug not in p.name:
            try:
                p.unlink()
                removed_count += 1
            except Exception as e:
                print(f"Error deleting {p.name}: {e}")
    if removed_count > 0:
        print(f"Auto-cleanup: Removed {removed_count} old cache files.")

ACCEPTABLE_TEXT_COLUMNS = [
    "reflection_answer_english", "reflection_answer", "text", "report",
]

def _pick_text_column(df: pd.DataFrame) -> str | None:
    for col in ACCEPTABLE_TEXT_COLUMNS:
        if col in df.columns:
            return col
    return None

def _list_text_columns(df: pd.DataFrame) -> list[str]:
    return list(df.columns)

def _set_from_env_or_secrets(key: str):
    if os.getenv(key):
        return
    try:
        val = st.secrets.get(key, None)
    except Exception:
        val = None
    if val:
        os.environ[key] = str(val)

for _k in ("MOSAIC_DATA", "MOSAIC_BOX"):
    _set_from_env_or_secrets(_k)

@st.cache_data
def count_clean_reports(csv_path: str, text_col: str | None = None) -> int:
    df = pd.read_csv(csv_path)
    col = text_col if (text_col and text_col in df.columns) else _pick_text_column(df)
    if col is None:
        return 0
    if col != "reflection_answer_english":
        df = df.rename(columns={col: "reflection_answer_english"})
    df.dropna(subset=["reflection_answer_english"], inplace=True)
    df["reflection_answer_english"] = df["reflection_answer_english"].astype(str)
    df = df[df["reflection_answer_english"].str.strip() != ""]
    return len(df)

# =====================================================================
# 1. Streamlit app setup
# =====================================================================

st.set_page_config(page_title="MOSAIC Dashboard", layout="wide")
st.title("Mapping of Subjective Accounts into Interpreted Clusters (MOSAIC)")

mode_status = "🟢 Pro Mode (LLM Enabled)" if ENABLE_LLM else "🟡 Lite Mode (LLM Disabled)"
st.caption(f"Current Runtime Status: **{mode_status}**")

st.markdown(
    """
    _If you use this tool in your research, please cite the following paper:_\n
    **Beauté, R., et al. (2025).** **Mapping of Subjective Accounts into Interpreted Clusters (MOSAIC)** https://arxiv.org/abs/2502.18318
    """
)

# =====================================================================
# 2. Dataset paths
# =====================================================================

ds_input = st.sidebar.text_input("Project/Dataset name", value="MOSAIC", key="dataset_name_input")
DATASET_DIR = _slugify(ds_input).upper()

RAW_DIR = raw_path(DATASET_DIR)
PROC_DIR = proc_path(DATASET_DIR, "preprocessed")
EVAL_DIR = eval_path(DATASET_DIR)
CACHE_DIR = PROC_DIR / "cache"

PROC_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

def _list_server_csvs(proc_dir: Path) -> list[str]:
    return [str(p) for p in sorted(proc_dir.glob("*.csv"))]

HISTORY_FILE = str(PROC_DIR / "run_history.json")

# =====================================================================
# 3. Embedding & LLM loaders
# =====================================================================

@st.cache_resource
def load_embedding_model(model_name):
    st.info(f"Loading embedding model '{model_name}'...")
    return SentenceTransformer(model_name)

@st.cache_resource
def load_llm_model():
    """Loads LlamaCPP model only if ENABLE_LLM is True."""
    if not ENABLE_LLM or not LLM_AVAILABLE:
        return None
        
    st.info("Loading Llama-3-8B-Instruct (Quantized)...")
    try:
        model_repo = "NousResearch/Meta-Llama-3-8B-Instruct-GGUF"
        model_file = "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
        model_path = hf_hub_download(repo_id=model_repo, filename=model_file)
        # n_gpu_layers=-1 attempts to offload all to GPU
        return Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=8192, stop=["Q:", "\n"], verbose=False)
    except Exception as e:
        st.error(f"Failed to load LLM: {e}")
        return None

@st.cache_data
def load_precomputed_data(docs_file, embeddings_file):
    docs = np.load(docs_file, allow_pickle=True).tolist()
    emb = np.load(embeddings_file, allow_pickle=True)
    return docs, emb

# =====================================================================
# 4. Topic modeling function
# =====================================================================

def get_config_hash(cfg):
    return json.dumps(cfg, sort_keys=True)

@st.cache_data
def perform_topic_modeling(_docs, _embeddings, config_hash, use_llm_flag=False):
    """Fit BERTopic using cached result."""
    _docs = list(_docs)
    _embeddings = np.asarray(_embeddings)
    if _embeddings.dtype == object or _embeddings.ndim != 2:
        try:
            _embeddings = np.vstack(_embeddings)
        except Exception:
            st.error("Embeddings are invalid. Regenerate data.")
            st.stop()
    _embeddings = np.ascontiguousarray(_embeddings, dtype=np.float32)

    if _embeddings.shape[0] != len(_docs):
        st.error("Mismatch between docs and embeddings.")
        st.stop()

    config = json.loads(config_hash)

    if "ngram_range" in config["vectorizer_params"]:
        config["vectorizer_params"]["ngram_range"] = tuple(config["vectorizer_params"]["ngram_range"])

    # --- Representation Model Logic ---
    rep_model = None
    if use_llm_flag and LLM_AVAILABLE:
        llm = load_llm_model()
        if llm:
            prompt = """Q:
You are an expert in micro-phenomenology. The following documents are reflections from participants about their experience.
I have a topic that contains the following documents:
[DOCUMENTS]
The topic is described by the following keywords: '[KEYWORDS]'.
Based on the above information, give a short, informative label (5–10 words).
A:"""
            rep_model = {
                "LLM": LlamaCPP(llm, prompt=prompt, nr_docs=25, doc_length=300, tokenizer="whitespace")
            }
        else:
            print("LLM requested but load failed; falling back to default representation.")

    umap_model = UMAP(random_state=42, metric="cosine", **config["umap_params"])
    hdbscan_model = HDBSCAN(metric="euclidean", prediction_data=True, **config["hdbscan_params"])
    vectorizer_model = CountVectorizer(**config["vectorizer_params"]) if config["use_vectorizer"] else None
    nr_topics_val = None if config["bt_params"]["nr_topics"] == "auto" else int(config["bt_params"]["nr_topics"])

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=rep_model,
        top_n_words=config["bt_params"]["top_n_words"],
        nr_topics=nr_topics_val,
        verbose=False,
    )

    topics, _ = topic_model.fit_transform(_docs, _embeddings)
    info = topic_model.get_topic_info()

    outlier_pct = 0
    if -1 in info.Topic.values:
        outlier_pct = (info.Count[info.Topic == -1].iloc[0] / info.Count.sum()) * 100

    # Label extraction
    if use_llm_flag and rep_model and "LLM" in topic_model.get_topics(full=True):
        # Extract LLM labels if available
        raw_labels = [label[0][0] for label in topic_model.get_topics(full=True)["LLM"].values()]
        cleaned_labels = [lbl.split(":")[-1].strip().strip('"').strip(".") for lbl in raw_labels]
        final_labels = [lbl if lbl else "Unlabelled" for lbl in cleaned_labels]
        # Map back to docs
        all_labels = [final_labels[topic + topic_model._outliers] if topic != -1 else "Unlabelled" for topic in topics]
    else:
        # Default labels
        name_map = info.set_index("Topic")["Name"].to_dict()
        all_labels = [name_map[topic] for topic in topics]

    reduced = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric="cosine", random_state=42).fit_transform(_embeddings)

    return topic_model, reduced, all_labels, len(info) - 1, outlier_pct

# =====================================================================
# 5. CSV → documents → embeddings pipeline
# =====================================================================

def generate_and_save_embeddings(csv_path, docs_file, emb_file, selected_embedding_model, split_sentences, device, text_col=None):
    st.info(f"Reading and preparing CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    col = text_col if (text_col and text_col in df.columns) else _pick_text_column(df)
    if col is None:
        st.error("CSV must contain at least one text column.")
        return

    if col != "reflection_answer_english":
        df = df.rename(columns={col: "reflection_answer_english"})

    df.dropna(subset=["reflection_answer_english"], inplace=True)
    df["reflection_answer_english"] = df["reflection_answer_english"].astype(str)
    df = df[df["reflection_answer_english"].str.strip() != ""]
    reports = df["reflection_answer_english"].tolist()

    if split_sentences:
        try:
            sentences = [s for r in reports for s in nltk.sent_tokenize(r)]
            docs = [s for s in sentences if len(s.split()) > 2]
        except LookupError as e:
            st.error(f"NLTK tokenizer data not found: {e}")
            st.stop()
    else:
        docs = reports

    np.save(docs_file, np.array(docs, dtype=object))
    st.success(f"Prepared {len(docs)} documents")

    st.info(f"Encoding {len(docs)} documents with {selected_embedding_model} on {device}")
    model = load_embedding_model(selected_embedding_model)

    encode_device = "cpu" if device == "CPU" else None
    batch_size = 64 if device == "CPU" else 32

    embeddings = model.encode(docs, show_progress_bar=True, batch_size=batch_size, device=encode_device, convert_to_numpy=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    np.save(emb_file, embeddings)

    st.success("Embedding generation complete!")
    st.balloons()
    st.rerun()

# =====================================================================
# 6. Sidebar — dataset, upload, parameters
# =====================================================================

st.sidebar.header("Data Input Method")

source = st.sidebar.radio("Choose data source", ("Use preprocessed CSV on server", "Upload my own CSV"), index=0, key="data_source")

uploaded_csv_path = None
CSV_PATH = None

if source == "Use preprocessed CSV on server":
    available = _list_server_csvs(PROC_DIR)
    if not available:
        st.info(f"No CSVs found in {PROC_DIR}. Upload a CSV.")
        st.stop()
    selected_csv = st.sidebar.selectbox("Choose a preprocessed CSV", available, key="server_csv_select")
    CSV_PATH = selected_csv
else:
    up = st.sidebar.file_uploader("Upload a CSV", type=["csv"], key="upload_csv")
    if up is not None:
        encodings_to_try = ['utf-8', 'mac_roman', 'cp1252', 'ISO-8859-1']
        tmp_df = None
        for encoding in encodings_to_try:
            try:
                up.seek(0)
                tmp_df = pd.read_csv(up, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        if tmp_df is None or tmp_df.empty:
            st.error("Could not decode file.")
            st.stop()
        
        safe_filename = _slugify(os.path.splitext(up.name)[0])
        _cleanup_old_cache(safe_filename)
        uploaded_csv_path = str((PROC_DIR / f"{safe_filename}.csv").resolve())
        tmp_df.to_csv(uploaded_csv_path, index=False)
        st.success(f"Uploaded CSV saved to {uploaded_csv_path}")
        CSV_PATH = uploaded_csv_path
    else:
        st.info("Upload a CSV to continue.")
        st.stop()

if CSV_PATH is None:
    st.stop()

# Text column selection
@st.cache_data
def get_text_columns(csv_path: str) -> list[str]:
    df_sample = pd.read_csv(csv_path, nrows=2000)
    return _list_text_columns(df_sample)

text_columns = get_text_columns(CSV_PATH)
if not text_columns:
    st.error("No columns found in this CSV.")
    st.stop()

preferred = None
try:
    df_sample = pd.read_csv(CSV_PATH, nrows=2000)
    preferred = _pick_text_column(df_sample)
except Exception: pass

default_idx = text_columns.index(preferred) if preferred in text_columns else 0
selected_text_column = st.sidebar.selectbox("Text column to analyse", text_columns, index=default_idx, key="text_column_select")

# Data Granularity
st.sidebar.subheader("Data Granularity & Subsampling")
selected_granularity = st.sidebar.checkbox("Split reports into sentences", value=True)
granularity_label = "sentences" if selected_granularity else "reports"
subsample_perc = st.sidebar.slider("Data sampling (%)", 10, 100, 100, 5)

st.sidebar.markdown("---")

# Model Selection
st.sidebar.header("Model Selection")
selected_embedding_model = st.sidebar.selectbox("Choose an embedding model", (
    "BAAI/bge-small-en-v1.5",
    "intfloat/multilingual-e5-large-instruct",
    "Qwen/Qwen3-Embedding-0.6B",
    "sentence-transformers/all-mpnet-base-v2",
))
selected_device = st.sidebar.radio("Processing device", ["GPU (MPS)", "CPU"], index=0)

# =====================================================================
# 7. Precompute filenames
# =====================================================================

def get_precomputed_filenames(csv_path, model_name, split_sentences, text_col):
    base = os.path.splitext(os.path.basename(csv_path))[0]
    safe_model = re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)
    suf = "sentences" if split_sentences else "reports"
    col_suffix = f"_{re.sub(r'[^a-zA-Z0-9_-]', '_', text_col)}" if text_col else ""
    return (
        str(CACHE_DIR / f"precomputed_{base}{col_suffix}_{suf}_docs.npy"),
        str(CACHE_DIR / f"precomputed_{base}_{safe_model}{col_suffix}_{suf}_embeddings.npy"),
    )

DOCS_FILE, EMBEDDINGS_FILE = get_precomputed_filenames(CSV_PATH, selected_embedding_model, selected_granularity, selected_text_column)

st.sidebar.markdown("### Cache")
if st.sidebar.button("Clear cached files for this configuration", use_container_width=True):
    for p in (DOCS_FILE, EMBEDDINGS_FILE):
        if os.path.exists(p):
            os.remove(p)
    load_precomputed_data.clear()
    perform_topic_modeling.clear()
    st.success("Cache cleared.")
    st.rerun()

st.sidebar.markdown("---")

# =====================================================================
# 8. Run Analysis
# =====================================================================

if not os.path.exists(EMBEDDINGS_FILE):
    st.warning(f"No precomputed embeddings found for this configuration.")
    if st.button("Prepare Data for This Configuration"):
        generate_and_save_embeddings(CSV_PATH, DOCS_FILE, EMBEDDINGS_FILE, selected_embedding_model, selected_granularity, selected_device, text_col=selected_text_column)
else:
    docs, embeddings = load_precomputed_data(DOCS_FILE, EMBEDDINGS_FILE)
    embeddings = np.asarray(embeddings)
    if embeddings.dtype == object or embeddings.ndim != 2:
        try:
            embeddings = np.vstack(embeddings).astype(np.float32)
        except Exception:
            st.error("Cached embeddings are invalid. Regenerate.")
            st.stop()

    if subsample_perc < 100:
        n = int(len(docs) * (subsample_perc / 100))
        idx = np.random.choice(len(docs), size=n, replace=False)
        docs = [docs[i] for i in idx]
        embeddings = embeddings[idx, :]
        st.warning(f"Running analysis on {subsample_perc}% subsample ({len(docs)} documents)")

    st.subheader("Dataset summary")
    n_reports = count_clean_reports(CSV_PATH, selected_text_column)
    st.metric("Units analysed", len(docs))

    # Parameters
    st.sidebar.header("Model Parameters")
    use_vectorizer = st.sidebar.checkbox("Use CountVectorizer", value=True)
    with st.sidebar.expander("Vectorizer"):
        ng_min, ng_max = st.slider("N-gram Range", 1, 5, (1, 2))
        min_df = st.slider("Min Doc Freq", 1, 50, 1)
        stopwords = st.select_slider("Stopwords", options=[None, "english"], value=None)
    with st.sidebar.expander("UMAP"):
        um_n = st.slider("n_neighbors", 2, 50, 15)
        um_c = st.slider("n_components", 2, 20, 5)
        um_d = st.slider("min_dist", 0.0, 1.0, 0.0)
    with st.sidebar.expander("HDBSCAN"):
        hs = st.slider("min_cluster_size", 5, 100, 10)
        hm = st.slider("min_samples", 2, 100, 5)
    with st.sidebar.expander("BERTopic"):
        nr_topics = st.text_input("nr_topics", value="auto")
        top_n_words = st.slider("top_n_words", 5, 25, 10)

    current_config = {
        "embedding_model": selected_embedding_model,
        "granularity": granularity_label,
        "subsample_percent": subsample_perc,
        "use_vectorizer": use_vectorizer,
        "vectorizer_params": {"ngram_range": (ng_min, ng_max), "min_df": min_df, "stop_words": stopwords},
        "umap_params": {"n_neighbors": um_n, "n_components": um_c, "min_dist": um_d},
        "hdbscan_params": {"min_cluster_size": hs, "min_samples": hm},
        "bt_params": {"nr_topics": nr_topics, "top_n_words": top_n_words},
        "text_column": selected_text_column,
        "llm_enabled": ENABLE_LLM 
    }

    run_button = st.sidebar.button("Run Analysis", type="primary")

    # =================================================================
    # 9. Visualization & History Tabs
    # =================================================================
    main_tab, history_tab = st.tabs(["Main Results", "Run History"])

    def load_history():
        path = HISTORY_FILE
        if not os.path.exists(path): return []
        try:
            data = json.load(open(path))
            for e in data:
                if "outlier_pct" not in e and "outlier_perc" in e: e["outlier_pct"] = e.pop("outlier_perc")
            return data
        except Exception: return []

    def save_history(h):
        json.dump(h, open(HISTORY_FILE, "w"), indent=2)

    if "history" not in st.session_state:
        st.session_state.history = load_history()

    if run_button:
        with st.spinner("Performing topic modeling..."):
            # Pass the global ENABLE_LLM flag to the cached function
            model, reduced, labels, n_topics, outlier_pct = perform_topic_modeling(
                docs, embeddings, get_config_hash(current_config), use_llm_flag=ENABLE_LLM
            )
        st.session_state.latest_results = (model, reduced, labels)

        entry = {
            "timestamp": str(pd.Timestamp.now()),
            "config": current_config,
            "num_topics": n_topics,
            "outlier_pct": f"{outlier_pct:.2f}%",
            "llm_labels": [name for name in model.get_topic_info().Name.values if ("Unlabelled" not in name and "outlier" not in name)],
        }
        st.session_state.history.insert(0, entry)
        save_history(st.session_state.history)
        st.rerun()

    with main_tab:
        if "latest_results" in st.session_state:
            tm, reduced, labs = st.session_state.latest_results
            st.subheader("Experiential Topics Visualisation")
            fig, _ = datamapplot.create_plot(reduced, labs)
            st.pyplot(fig)
            st.subheader("Topic Info")
            st.dataframe(tm.get_topic_info())
            
            # Export Logic
            st.subheader("Export results")
            full_reps = tm.get_topics(full=True)
            llm_reps = full_reps.get("LLM", {})
            
            # Determine how to name topics based on what mode we ran
            llm_names = {}
            if ENABLE_LLM and llm_reps:
                for tid, vals in llm_reps.items():
                    try:
                        llm_names[tid] = (vals[0][0] or "").strip().strip('"').strip(".")
                    except Exception:
                        llm_names[tid] = "Unlabelled"
            else:
                llm_names = tm.get_topic_info().set_index("Topic")["Name"].to_dict()

            doc_info = tm.get_document_info(docs)[["Document", "Topic"]]
            
            # --- Long Format Export (One row per sentence) ---
            long_format_df = doc_info.copy()
            long_format_df["Topic Name"] = long_format_df["Topic"].map(llm_names).fillna("Unlabelled")
            long_format_df = long_format_df[["Topic", "Topic Name", "Document"]]
            
            base = os.path.splitext(os.path.basename(CSV_PATH))[0]
            gran = "sentences" if selected_granularity else "reports"
            long_csv_name = f"all_sentences_{base}_{gran}.csv"

            st.download_button(
                "Download All Sentences (Long Format)",
                data=long_format_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=long_csv_name,
                mime="text/csv",
                use_container_width=True
            )

    with history_tab:
        st.subheader("Run History")
        if not st.session_state.history:
            st.info("No runs yet.")
        else:
            for i, entry in enumerate(st.session_state.history):
                with st.expander(f"Run {i+1} — {entry['timestamp']}"):
                    st.write(f"**Topics:** {entry['num_topics']}")
                    st.write(f"**Outliers:** {entry.get('outlier_pct', 'N/A')}")
                    st.write("**Topic Labels:**")
                    st.write(entry.get("llm_labels", []))
                    st.json(entry["config"])