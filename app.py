"""
File: app.py
Description: Streamlit app for advanced topic modeling on Innerspeech dataset
             with BERTopic, UMAP, HDBSCAN. (LLM features disabled for lite deployment)
Last Modified: 06/11/2025
@author: r.beaut
"""

# =====================================================================
# Imports
# =====================================================================

from pathlib import Path
import sys
# from llama_cpp import Llama  # <-- REMOVED
import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import nltk
import json

# =====================================================================
# NLTK setup
# =====================================================================

NLTK_DATA_DIR = "/usr/local/share/nltk_data"
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_DIR)

# Try to ensure both punkt_tab (new NLTK) and punkt (old NLTK) are available
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
    # Minimal stand-in so the app works anywhere (Streamlit Cloud, local without MOSAIC, etc.)
    def _env(key: str, default: str) -> Path:
        val = os.getenv(key, default)
        return Path(val).expanduser().resolve()

    # Defaults: app-local data/ eval/ that are safe on Cloud
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


# BERTopic stack
from bertopic import BERTopic
# from bertopic.representation import LlamaCPP  # <-- REMOVED
# from llama_cpp import Llama  # <-- REMOVED
from sentence_transformers import SentenceTransformer

# Clustering/dimensionality reduction
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN

# Visualisation
import datamapplot
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

# =====================================================================
# 0. Constants & Helper Functions
# =====================================================================


def _slugify(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s or "DATASET"


# "Nice" default names we know from MOSAIC; NOT a hard constraint anymore
ACCEPTABLE_TEXT_COLUMNS = [
    "reflection_answer_english",
    "reflection_answer",
    "text",
    "report",
]


def _pick_text_column(df: pd.DataFrame) -> str | None:
    """Return the first matching *preferred* text column name if present."""
    for col in ACCEPTABLE_TEXT_COLUMNS:
        if col in df.columns:
            return col
    return None


def _list_text_columns(df: pd.DataFrame) -> list[str]:
    """
    Return all columns; we’ll cast the chosen one to string later.
    This makes the selector work with any column name / dtype.
    """
    return list(df.columns)



def _set_from_env_or_secrets(key: str):
    """Allow hosting: value can come from environment or from Streamlit secrets."""
    if os.getenv(key):
        return
    try:
        val = st.secrets.get(key, None)
    except Exception:
        val = None
    if val:
        os.environ[key] = str(val)


# Enable both MOSAIC_DATA and MOSAIC_BOX automatically
for _k in ("MOSAIC_DATA", "MOSAIC_BOX"):
    _set_from_env_or_secrets(_k)


@st.cache_data
def count_clean_reports(csv_path: str, text_col: str | None = None) -> int:
    """Count non-empty reports in the chosen text column."""
    df = pd.read_csv(csv_path)

    if text_col is not None and text_col in df.columns:
        col = text_col
    else:
        col = _pick_text_column(df)

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
st.title(
    "Mapping of Subjective Accounts into Interpreted Clusters (MOSAIC): "
    "Topic Modelling Dashboard for Phenomenological Reports"
)

st.markdown(
    """
    _If you use this tool in your research, please cite the following paper:_\n
    **Beauté, R., et al. (2025).**  
    **Mapping of Subjective Accounts into Interpreted Clusters (MOSAIC): Topic Modelling and LLM applied to Stroboscopic Phenomenology**  
    https://arxiv.org/abs/2502.18318
    """
)

# =====================================================================
# 2. Dataset paths (using MOSAIC structure)
# =====================================================================

ds_input = st.sidebar.text_input(
    "Project/Dataset name", value="MOSAIC", key="dataset_name_input"
)
DATASET_DIR = _slugify(ds_input).upper()

RAW_DIR = raw_path(DATASET_DIR)
PROC_DIR = proc_path(DATASET_DIR, "preprocessed")
EVAL_DIR = eval_path(DATASET_DIR)
CACHE_DIR = PROC_DIR / "cache"

PROC_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

with st.sidebar.expander("About the dataset name", expanded=False):
    st.markdown(
        f"""
- The name above is converted to **UPPER CASE** and used as a folder name.
- If the folder doesn’t exist, it will be **created**:
  - Preprocessed CSVs: `{PROC_DIR}`
  - Exports (results): `{EVAL_DIR}`
- If you choose **Use preprocessed CSV on server**, I’ll list CSVs in `{PROC_DIR}`.
- If you **upload** a CSV, it will be saved to `{PROC_DIR}/uploaded.csv`.
        """.strip()
    )


def _list_server_csvs(proc_dir: Path) -> list[str]:
    return [str(p) for p in sorted(proc_dir.glob("*.csv"))]


DATASETS = None  # keep name for clarity; we’ll fill it when rendering the sidebar
HISTORY_FILE = str(PROC_DIR / "run_history.json")

# =====================================================================
# 3. Embedding & LLM loaders
# =====================================================================


@st.cache_resource
def load_embedding_model(model_name):
    st.info(f"Loading embedding model '{model_name}'...")
    return SentenceTransformer(model_name)


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
def perform_topic_modeling(_docs, _embeddings, config_hash):
    """Fit BERTopic using cached result."""

    _docs = list(_docs)
    _embeddings = np.asarray(_embeddings)
    if _embeddings.dtype == object or _embeddings.ndim != 2:
        try:
            _embeddings = np.vstack(_embeddings)
        except Exception:
            st.error(
                f"Embeddings are invalid (dtype={_embeddings.dtype}, ndim={_embeddings.ndim}). "
                "Please click **Prepare Data** to regenerate."
            )
            st.stop()
    _embeddings = np.ascontiguousarray(_embeddings, dtype=np.float32)

    if _embeddings.shape[0] != len(_docs):
        st.error(
            f"Mismatch between docs and embeddings: len(docs)={len(_docs)} vs "
            f"embeddings.shape[0]={_embeddings.shape[0]}. "
            "Delete the cached files for this configuration and regenerate."
        )
        st.stop()

    config = json.loads(config_hash)

    if "ngram_range" in config["vectorizer_params"]:
        config["vectorizer_params"]["ngram_range"] = tuple(
            config["vectorizer_params"]["ngram_range"]
        )

    rep_model = None  # <-- Use BERTopic defaults for representation

    umap_model = UMAP(random_state=42, metric="cosine", **config["umap_params"])
    hdbscan_model = HDBSCAN(
        metric="euclidean", prediction_data=True, **config["hdbscan_params"]
    )
    vectorizer_model = (
        CountVectorizer(**config["vectorizer_params"])
        if config["use_vectorizer"]
        else None
    )

    nr_topics_val = (
        None
        if config["bt_params"]["nr_topics"] == "auto"
        else int(config["bt_params"]["nr_topics"])
    )

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
        outlier_pct = (
            info.Count[info.Topic == -1].iloc[0] / info.Count.sum()
        ) * 100

    topic_info = topic_model.get_topic_info()
    name_map = topic_info.set_index("Topic")["Name"].to_dict()
    all_labels = [name_map[topic] for topic in topics]

    reduced = UMAP(
        n_neighbors=15,
        n_components=2,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    ).fit_transform(_embeddings)

    return topic_model, reduced, all_labels, len(info) - 1, outlier_pct


# =====================================================================
# 5. CSV → documents → embeddings pipeline
# =====================================================================


def generate_and_save_embeddings(
    csv_path,
    docs_file,
    emb_file,
    selected_embedding_model,
    split_sentences,
    device,
    text_col=None,
):

    # ---------------------
    # Load & clean CSV
    # ---------------------
    st.info(f"Reading and preparing CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    if text_col is not None and text_col in df.columns:
        col = text_col
    else:
        col = _pick_text_column(df)

    if col is None:
        st.error("CSV must contain at least one text column.")
        return

    if col != "reflection_answer_english":
        df = df.rename(columns={col: "reflection_answer_english"})

    df.dropna(subset=["reflection_answer_english"], inplace=True)
    df["reflection_answer_english"] = df["reflection_answer_english"].astype(str)
    df = df[df["reflection_answer_english"].str.strip() != ""]
    reports = df["reflection_answer_english"].tolist()

    # ---------------------
    # Sentence / report granularity
    # ---------------------
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

    # ---------------------
    # Embeddings
    # ---------------------
    st.info(
        f"Encoding {len(docs)} documents with {selected_embedding_model} on {device}"
    )

    model = load_embedding_model(selected_embedding_model)

    encode_device = None
    batch_size = 32
    if device == "CPU":
        encode_device = "cpu"
        batch_size = 64

    embeddings = model.encode(
        docs,
        show_progress_bar=True,
        batch_size=batch_size,
        device=encode_device,
        convert_to_numpy=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    np.save(emb_file, embeddings)

    st.success("Embedding generation complete!")
    st.balloons()
    st.rerun()


# =====================================================================
# 6. Sidebar — dataset, upload, parameters
# =====================================================================

st.sidebar.header("Data Input Method")

source = st.sidebar.radio(
    "Choose data source",
    ("Use preprocessed CSV on server", "Upload my own CSV"),
    index=0,
    key="data_source",
)

uploaded_csv_path = None
CSV_PATH = None  # will be set in the chosen branch

if source == "Use preprocessed CSV on server":
    available = _list_server_csvs(PROC_DIR)
    if not available:
        st.info(
            f"No CSVs found in {PROC_DIR}. Switch to 'Upload my own CSV' or change the dataset name."
        )
        st.stop()
    selected_csv = st.sidebar.selectbox(
        "Choose a preprocessed CSV", available, key="server_csv_select"
    )
    CSV_PATH = selected_csv
else:
    up = st.sidebar.file_uploader(
        "Upload a CSV", type=["csv"], key="upload_csv"
    )

    st.sidebar.caption(
        "Your CSV should have **one row per report** and at least one text column "
        "(for example `reflection_answer_english`, `reflection_answer`, `text`, `report`, "
        "or any other column containing free text). "
        "Other columns (ID, condition, etc.) are allowed. "
        "After upload, you’ll be able to choose which text column to analyse."
    )

    # if up is not None:
    #     tmp_df = pd.read_csv(up)
    #     if tmp_df.empty:
    #         st.error("Uploaded CSV is empty.")
    #         st.stop()

    # if up is not None:
    #     try:
    #         # Try loading as standard UTF-8
    #         tmp_df = pd.read_csv(up)
    #     except UnicodeDecodeError:
    #         # If that fails (e.g., Excel/Windows CSV), try ISO-8859-1 (Latin-1)
    #         up.seek(0)  # Reset file pointer to the beginning
    #         tmp_df = pd.read_csv(up, encoding='ISO-8859-1')
            
    #     if tmp_df.empty:
    #         st.error("Uploaded CSV is empty.")
    #         st.stop()

    if up is not None:
        # List of encodings to try: 
        # 1. utf-8 (Standard)
        # 2. mac_roman (Fixes the Õ and É issues from Mac Excel)
        # 3. cp1252 (Standard Windows Excel)
        encodings_to_try = ['utf-8', 'mac_roman', 'cp1252', 'ISO-8859-1']
        
        tmp_df = None
        success_encoding = None

        for encoding in encodings_to_try:
            try:
                up.seek(0)  # Always reset to start of file before trying
                tmp_df = pd.read_csv(up, encoding=encoding)
                success_encoding = encoding
                break  # If we get here, it worked, so stop the loop
            except UnicodeDecodeError:
                continue  # If it fails, try the next one

        if tmp_df is None:
            st.error("Could not decode file. Please save your CSV as 'CSV UTF-8' in Excel.")
            st.stop()
            
        if tmp_df.empty:
            st.error("Uploaded CSV is empty.")
            st.stop()
            
        # Optional: Print which encoding worked to the logs (for your info)
        print(f"Successfully loaded CSV using {success_encoding} encoding.")

        # Just save; we’ll choose the text column later
        uploaded_csv_path = str((PROC_DIR / "uploaded.csv").resolve())
        tmp_df.to_csv(uploaded_csv_path, index=False)
        st.success(f"Uploaded CSV saved to {uploaded_csv_path}")
        CSV_PATH = uploaded_csv_path
    else:
        st.info("Upload a CSV to continue.")
        st.stop()

if CSV_PATH is None:
    st.stop()

# ---------------------------------------------------------------------
# Text column selection
# ---------------------------------------------------------------------


@st.cache_data
def get_text_columns(csv_path: str) -> list[str]:
    df_sample = pd.read_csv(csv_path, nrows=2000)
    return _list_text_columns(df_sample)

text_columns = get_text_columns(CSV_PATH)

if not text_columns:
    st.error(
        "No columns found in this CSV. At least one column is required."
    )
    st.stop()


text_columns = get_text_columns(CSV_PATH)
if not text_columns:
    st.error(
        "No text-like columns found in this CSV. At least one column must contain text."
    )
    st.stop()

# Try to pick a nice default (one of the MOSAIC-ish names) if present
try:
    df_sample = pd.read_csv(CSV_PATH, nrows=2000)
    preferred = _pick_text_column(df_sample)
except Exception:
    preferred = None

if preferred in text_columns:
    default_idx = text_columns.index(preferred)
else:
    default_idx = 0

selected_text_column = st.sidebar.selectbox(
    "Text column to analyse",
    text_columns,
    index=default_idx,
    key="text_column_select",
)

# ---------------------------------------------------------------------
# Data granularity & subsampling
# ---------------------------------------------------------------------

st.sidebar.subheader("Data Granularity & Subsampling")

selected_granularity = st.sidebar.checkbox(
    "Split reports into sentences", value=True
)
granularity_label = "sentences" if selected_granularity else "reports"

subsample_perc = st.sidebar.slider("Data sampling (%)", 10, 100, 100, 5)

st.sidebar.markdown("---")

# ---------------------------------------------------------------------
# Embedding model & device
# ---------------------------------------------------------------------

st.sidebar.header("Model Selection")

selected_embedding_model = st.sidebar.selectbox(
    "Choose an embedding model",
    (
        "BAAI/bge-small-en-v1.5",
        "intfloat/multilingual-e5-large-instruct",
        "Qwen/Qwen3-Embedding-0.6B",
        "sentence-transformers/all-mpnet-base-v2",
    ),
)

selected_device = st.sidebar.radio(
    "Processing device",
    ["GPU (MPS)", "CPU"],
    index=0,
)

# =====================================================================
# 7. Precompute filenames and pipeline triggers
# =====================================================================


def get_precomputed_filenames(csv_path, model_name, split_sentences, text_col):
    base = os.path.splitext(os.path.basename(csv_path))[0]
    safe_model = re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)
    suf = "sentences" if split_sentences else "reports"

    col_suffix = ""
    if text_col:
        safe_col = re.sub(r"[^a-zA-Z0-9_-]", "_", text_col)
        col_suffix = f"_{safe_col}"

    return (
        str(CACHE_DIR / f"precomputed_{base}{col_suffix}_{suf}_docs.npy"),
        str(
            CACHE_DIR
            / f"precomputed_{base}_{safe_model}{col_suffix}_{suf}_embeddings.npy"
        ),
    )


DOCS_FILE, EMBEDDINGS_FILE = get_precomputed_filenames(
    CSV_PATH, selected_embedding_model, selected_granularity, selected_text_column
)

# --- Cache management ---
st.sidebar.markdown("### Cache")
if st.sidebar.button(
    "Clear cached files for this configuration", use_container_width=True
):
    try:
        for p in (DOCS_FILE, EMBEDDINGS_FILE):
            if os.path.exists(p):
                os.remove(p)
        try:
            load_precomputed_data.clear()
        except Exception:
            pass
        try:
            perform_topic_modeling.clear()
        except Exception:
            pass

        st.success(
            "Deleted cached docs/embeddings and cleared caches. Click **Prepare Data** again."
        )
        st.rerun()
    except Exception as e:
        st.error(f"Failed to delete cache files: {e}")

st.sidebar.markdown("---")

# =====================================================================
# 8. Prepare Data OR Run Analysis
# =====================================================================

if not os.path.exists(EMBEDDINGS_FILE):
    st.warning(
        f"No precomputed embeddings found for this configuration "
        f"({granularity_label} / {selected_embedding_model} / column '{selected_text_column}')."
    )

    if st.button("Prepare Data for This Configuration"):
        generate_and_save_embeddings(
            CSV_PATH,
            DOCS_FILE,
            EMBEDDINGS_FILE,
            selected_embedding_model,
            selected_granularity,
            selected_device,
            text_col=selected_text_column,
        )

else:
    # Load cached data
    docs, embeddings = load_precomputed_data(DOCS_FILE, EMBEDDINGS_FILE)

    embeddings = np.asarray(embeddings)
    if embeddings.dtype == object or embeddings.ndim != 2:
        try:
            embeddings = np.vstack(embeddings).astype(np.float32)
        except Exception:
            st.error(
                "Cached embeddings are invalid. Please regenerate them for this configuration."
            )
            st.stop()

    if subsample_perc < 100:
        n = int(len(docs) * (subsample_perc / 100))
        idx = np.random.choice(len(docs), size=n, replace=False)
        docs = [docs[i] for i in idx]
        embeddings = np.asarray(embeddings)[idx, :]
        st.warning(
            f"Running analysis on {subsample_perc}% subsample ({len(docs)} documents)"
        )

    # Dataset summary
    st.subheader("Dataset summary")
    n_reports = count_clean_reports(CSV_PATH, selected_text_column)
    unit = "sentences" if selected_granularity else "reports"
    n_units = len(docs)

    c1, c2 = st.columns(2)
    c1.metric("Reports in CSV (cleaned)", n_reports)
    c2.metric(f"Units analysed ({unit})", n_units)

    # --- Parameter controls ---
    st.sidebar.header("Model Parameters")

    use_vectorizer = st.sidebar.checkbox("Use CountVectorizer", value=True)

    with st.sidebar.expander("Vectorizer"):
        ng_min = st.slider("Min N-gram", 1, 5, 1)
        ng_max = st.slider("Max N-gram", 1, 5, 2)
        min_df = st.slider("Min Doc Freq", 1, 50, 1)
        stopwords = st.select_slider(
            "Stopwords", options=[None, "english"], value=None
        )

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
        "vectorizer_params": {
            "ngram_range": (ng_min, ng_max),
            "min_df": min_df,
            "stop_words": stopwords,
        },
        "umap_params": {
            "n_neighbors": um_n,
            "n_components": um_c,
            "min_dist": um_d,
        },
        "hdbscan_params": {
            "min_cluster_size": hs,
            "min_samples": hm,
        },
        "bt_params": {
            "nr_topics": nr_topics,
            "top_n_words": top_n_words,
        },
        "text_column": selected_text_column,
    }

    run_button = st.sidebar.button("Run Analysis", type="primary")

    # =================================================================
    # 9. Visualization & History Tabs
    # =================================================================
    main_tab, history_tab = st.tabs(["Main Results", "Run History"])

    def load_history():
        path = HISTORY_FILE
        if not os.path.exists(path):
            return []
        try:
            data = json.load(open(path))
        except Exception:
            return []
        for e in data:
            if "outlier_pct" not in e and "outlier_perc" in e:
                e["outlier_pct"] = e.pop("outlier_perc")
        return data

    def save_history(h):
        json.dump(h, open(HISTORY_FILE, "w"), indent=2)

    if "history" not in st.session_state:
        st.session_state.history = load_history()

    if run_button:
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.asarray(embeddings)

        if embeddings.dtype == object or embeddings.ndim != 2:
            try:
                embeddings = np.vstack(embeddings).astype(np.float32)
            except Exception:
                st.error(
                    "Cached embeddings are invalid (object/ragged). Click **Prepare Data** to regenerate."
                )
                st.stop()

        if embeddings.shape[0] != len(docs):
            st.error(
                f"len(docs)={len(docs)} but embeddings.shape[0]={embeddings.shape[0]}.\n"
                "Likely stale cache (e.g., switched sentences↔reports or model). "
                "Use the **Clear cache** button below and regenerate."
            )
            st.stop()

        with st.spinner("Performing topic modeling..."):
            model, reduced, labels, n_topics, outlier_pct = perform_topic_modeling(
                docs, embeddings, get_config_hash(current_config)
            )
        st.session_state.latest_results = (model, reduced, labels)

        entry = {
            "timestamp": str(pd.Timestamp.now()),
            "config": current_config,
            "num_topics": n_topics,
            "outlier_pct": f"{outlier_pct:.2f}%",
            "llm_labels": [
                name
                for name in model.get_topic_info().Name.values
                if ("Unlabelled" not in name and "outlier" not in name)
            ],
        }
        st.session_state.history.insert(0, entry)
        save_history(st.session_state.history)
        st.rerun()

    # --- MAIN TAB ---
    with main_tab:
        if "latest_results" in st.session_state:
            tm, reduced, labs = st.session_state.latest_results

            st.subheader("Experiential Topics Visualisation")
            fig, _ = datamapplot.create_plot(reduced, labs)
            st.pyplot(fig)

            st.subheader("Topic Info")
            st.dataframe(tm.get_topic_info())

            st.subheader("Export results (one row per topic)")

            full_reps = tm.get_topics(full=True)
            llm_reps = full_reps.get("LLM", {})

            llm_names = {}
            for tid, vals in llm_reps.items():
                try:
                    llm_names[tid] = (
                        (vals[0][0] or "").strip().strip('"').strip(".")
                    )
                except Exception:
                    llm_names[tid] = "Unlabelled"

            if not llm_names:
                st.caption("Note: Using default keyword-based topic names.")
                llm_names = (
                    tm.get_topic_info().set_index("Topic")["Name"].to_dict()
                )

            doc_info = tm.get_document_info(docs)[["Document", "Topic"]]

            include_outliers = st.checkbox(
                "Include outlier topic (-1)", value=False
            )
            if not include_outliers:
                doc_info = doc_info[doc_info["Topic"] != -1]

            grouped = (
                doc_info.groupby("Topic")["Document"]
                .apply(list)
                .reset_index(name="texts")
            )
            grouped["topic_name"] = grouped["Topic"].map(llm_names).fillna(
                "Unlabelled"
            )

            export_topics = (
                grouped.rename(columns={"Topic": "topic_id"})[
                    ["topic_id", "topic_name", "texts"]
                ]
                .sort_values("topic_id")
                .reset_index(drop=True)
            )

            SEP = "\n"

            export_csv = export_topics.copy()
            export_csv["texts"] = export_csv["texts"].apply(
                lambda lst: SEP.join(map(str, lst))
            )

            base = os.path.splitext(os.path.basename(CSV_PATH))[0]
            gran = "sentences" if selected_granularity else "reports"
            csv_name = f"topics_{base}_{gran}.csv"
            jsonl_name = f"topics_{base}_{gran}.jsonl"
            csv_path = (EVAL_DIR / csv_name).resolve()
            jsonl_path = (EVAL_DIR / jsonl_name).resolve()

            cL, cC, cR = st.columns(3)

            with cL:
                if st.button("Save CSV to eval/", use_container_width=True):
                    try:
                        export_csv.to_csv(csv_path, index=False)
                        st.success(f"Saved CSV → {csv_path}")
                    except Exception as e:
                        st.error(f"Failed to save CSV: {e}")

            with cC:
                if st.button("Save JSONL to eval/", use_container_width=True):
                    try:
                        with open(jsonl_path, "w", encoding="utf-8") as f:
                            for _, row in export_topics.iterrows():
                                rec = {
                                    "topic_id": int(row["topic_id"]),
                                    "topic_name": row["topic_name"],
                                    "texts": list(map(str, row["texts"])),
                                }
                                f.write(
                                    json.dumps(rec, ensure_ascii=False) + "\n"
                                )
                        st.success(f"Saved JSONL → {jsonl_path}")
                    except Exception as e:
                        st.error(f"Failed to save JSONL: {e}")

            with cR:
                st.download_button(
                    "Download CSV",
                    data=export_csv.to_csv(index=False).encode("utf-8"),
                    file_name=csv_name,
                    mime="text/csv",
                    use_container_width=True,
                )

            st.caption("Preview (one row per topic)")
            st.dataframe(export_csv.head(10))

        else:
            st.info("Click 'Run Analysis' to begin.")

    # --- HISTORY TAB ---
    with history_tab:
        st.subheader("Run History")
        if not st.session_state.history:
            st.info("No runs yet.")
        else:
            for i, entry in enumerate(st.session_state.history):
                with st.expander(f"Run {i+1} — {entry['timestamp']}"):
                    st.write(f"**Topics:** {entry['num_topics']}")
                    st.write(
                        f"**Outliers:** {entry.get('outlier_pct', entry.get('outlier_perc', 'N/A'))}"
                    )
                    st.write("**Topic Labels (default keywords):**")
                    st.write(entry["llm_labels"])
                    with st.expander("Show full configuration"):
                        st.json(entry["config"])
