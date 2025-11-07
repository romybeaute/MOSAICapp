"""
File: app.py
Description: Streamlit app for advanced topic modeling on Innerspeech dataset
             with BERTopic, UMAP, HDBSCAN, LlamaCPP labeling, and CSV upload support.
Last Modified: 06/11/2025
@author: r.beaut
"""

# =====================================================================
# Imports
# =====================================================================

from pathlib import Path
import sys
from llama_cpp import Llama
import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import nltk
import json

# from mosaic.path_utils import CFG, raw_path, proc_path, eval_path, project_root


try:
    from mosaic.path_utils import CFG, raw_path, proc_path, eval_path, project_root  # type: ignore
except Exception:
    # Minimal stand-in so the app works anywhere (Streamlit Cloud, local without MOSAIC, etc.)
    def _env(key: str, default: str) -> Path:
        val = os.getenv(key, default)
        return Path(val).expanduser().resolve()

    # Defaults: app-local data/ eval/ that are safe on Cloud
    _DATA_ROOT = _env("MOSAIC_DATA", str(Path(__file__).parent / "data"))
    _BOX_ROOT  = _env("MOSAIC_BOX",  str(Path(__file__).parent / "data" / "raw"))
    _EVAL_ROOT = _env("MOSAIC_EVAL", str(Path(__file__).parent / "eval"))

    CFG = {
        "data_root": str(_DATA_ROOT),
        "box_root":  str(_BOX_ROOT),
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
from bertopic.representation import LlamaCPP
# from llama_cpp import Llama
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

ACCEPTABLE_TEXT_COLUMNS = [
    "reflection_answer_english",
    "reflection_answer",
    "text",
    "report",
]

def _pick_text_column(df: pd.DataFrame) -> str | None:
    """Return the first matching text column."""
    for col in ACCEPTABLE_TEXT_COLUMNS:
        if col in df.columns:
            return col
    return None


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



# =====================================================================
# 1. Streamlit app setup
# =====================================================================

st.set_page_config(page_title="Advanced Topic Modeling", layout="wide")
st.title("Topic Modelling Dashboard for Phenomenological Reports")

ROOT = project_root()
sys.path.append(str(ROOT / "MULTILINGUAL"))



# =====================================================================
# 2. Dataset paths (using MOSAIC structure)
# =====================================================================

DATASET = "INNERSPEECH"

RAW_DIR  = raw_path(DATASET)
PROC_DIR = proc_path(DATASET, 'preprocessed')
EVAL_DIR = eval_path(DATASET)
CACHE_DIR = PROC_DIR / "cache"

PROC_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "API Translation (Batched)": str(PROC_DIR / "innerspeech_translated_batched_API.csv"),
    "Local Translation (Llama)": str(PROC_DIR / "innerspeech_dataset_translated_llama.csv"),
}

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
    """Loads LlamaCPP quantised model for topic labeling."""
    model_repo  = "NousResearch/Meta-Llama-3-8B-Instruct-GGUF"
    model_file  = "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
    model_path  = hf_hub_download(repo_id=model_repo, filename=model_file)
    return Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=8192,
                 stop=["Q:", "\n"], verbose=False)


@st.cache_data
def load_precomputed_data(docs_file, embeddings_file):
    docs = np.load(docs_file, allow_pickle=True).tolist()
    emb  = np.load(embeddings_file, allow_pickle=True)
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
            st.error(f"Embeddings are invalid (dtype={_embeddings.dtype}, ndim={_embeddings.ndim}). "
                     "Please click **Prepare Data** to regenerate.")
            st.stop()
    _embeddings = np.ascontiguousarray(_embeddings, dtype=np.float32)

    if _embeddings.shape[0] != len(_docs):
        st.error(f"Mismatch between docs and embeddings: len(docs)={len(_docs)} vs "
                 f"embeddings.shape[0]={_embeddings.shape[0]}. "
                 "Delete the cached files for this configuration and regenerate.")
        st.stop()

    config = json.loads(config_hash)

    # Prepare vectorizer parameters
    if "ngram_range" in config["vectorizer_params"]:
        config["vectorizer_params"]["ngram_range"] = tuple(config["vectorizer_params"]["ngram_range"])

    # Load LLM for labeling
    llm = load_llm_model()

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

    umap_model = UMAP(
        random_state=42, metric="cosine",
        **config["umap_params"]
    )
    hdbscan_model = HDBSCAN(
        metric="euclidean", prediction_data=True,
        **config["hdbscan_params"]
    )
    vectorizer_model = CountVectorizer(**config["vectorizer_params"]) if config["use_vectorizer"] else None

    nr_topics_val = None if config["bt_params"]["nr_topics"] == "auto" \
                         else int(config["bt_params"]["nr_topics"])

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=rep_model,
        top_n_words=config["bt_params"]["top_n_words"],
        nr_topics=nr_topics_val,
        verbose=False
    )

    topics, _ = topic_model.fit_transform(_docs, _embeddings)
    info = topic_model.get_topic_info()

    outlier_pct = 0
    if -1 in info.Topic.values:
        outlier_pct = (info.Count[info.Topic == -1].iloc[0] / info.Count.sum()) * 100

    # LLM labels
    raw_labels = [label[0][0] for label in topic_model.get_topics(full=True)["LLM"].values()]
    cleaned_labels = [
        lbl.split(":")[-1].strip().strip('"').strip(".") for lbl in raw_labels
    ]
    final_labels = [lbl if lbl else "Unlabelled" for lbl in cleaned_labels]

    # Map each document to its label
    all_labels = [
        final_labels[topic + topic_model._outliers] if topic != -1 else "Unlabelled"
        for topic in topics
    ]

    reduced = UMAP(
        n_neighbors=15, n_components=2, min_dist=0.0,
        metric="cosine", random_state=42
    ).fit_transform(_embeddings)

    return topic_model, reduced, all_labels, len(info)-1, outlier_pct



# =====================================================================
# 5. CSV → documents → embeddings pipeline
# =====================================================================

def generate_and_save_embeddings(csv_path, docs_file, emb_file,
                                 selected_embedding_model,
                                 split_sentences, device):

    # ---------------------
    # Load & clean CSV
    # ---------------------
    st.info(f"Reading and preparing CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    col = _pick_text_column(df)
    if col is None:
        st.error("CSV must contain one of: " + ", ".join(ACCEPTABLE_TEXT_COLUMNS))
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
            nltk.data.find("tokenizers/punkt")
        except Exception:
            nltk.download("punkt")

        sentences = [s for r in reports for s in nltk.sent_tokenize(r)]
        docs = [s for s in sentences if len(s.split()) > 2]
    else:
        docs = reports

    np.save(docs_file, np.array(docs, dtype=object))
    st.success(f"Prepared {len(docs)} documents")

    # ---------------------
    # Embeddings
    # ---------------------
    st.info(f"Encoding {len(docs)} documents with {selected_embedding_model} on {device}")

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
        convert_to_numpy=True
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    np.save(emb_file, embeddings)

    st.success("Embedding generation complete!")
    st.balloons()
    st.rerun()



# =====================================================================
# 6. Sidebar — dataset, upload, parameters
# =====================================================================

st.sidebar.header("Data Source & Model")

selected_dataset_name = st.sidebar.selectbox("Choose a dataset",
                                             list(DATASETS.keys()))
selected_embedding_model = st.sidebar.selectbox("Choose an embedding model", (
    "intfloat/multilingual-e5-large-instruct",
    "Qwen/Qwen3-Embedding-0.6B",
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-mpnet-base-v2",
))

# --- User CSV upload ---
st.sidebar.subheader("Data Input Method")
source = st.sidebar.radio(
    "Choose data source",
    ["Use preprocessed CSV on server", "Upload my own CSV"],
    index=1
)

uploaded_csv_path = None
if source == "Upload my own CSV":
    up = st.sidebar.file_uploader("Upload a CSV", type=["csv"])
    if up is not None:
        tmp_df = pd.read_csv(up)
        col = _pick_text_column(tmp_df)
        if col is None:
            st.error("CSV must contain a text column such as: " + ", ".join(ACCEPTABLE_TEXT_COLUMNS))
        else:
            if col != "reflection_answer_english":
                tmp_df = tmp_df.rename(columns={col: "reflection_answer_english"})
            uploaded_csv_path = str((PROC_DIR / "uploaded.csv").resolve())
            tmp_df.to_csv(uploaded_csv_path, index=False)
            st.success(f"Uploaded CSV saved to {uploaded_csv_path}")

# Choose final CSV path
if source == "Upload my own CSV" and uploaded_csv_path is not None:
    CSV_PATH = uploaded_csv_path
else:
    CSV_PATH = DATASETS[selected_dataset_name]

# --- Device selection ---
st.sidebar.header("Data Preparation")
selected_device = st.sidebar.radio(
    "Processing device",
    ["GPU (MPS)", "CPU"],
    index=0,
)

selected_granularity = st.sidebar.checkbox("Split reports into sentences", value=True)
granularity_label = "sentences" if selected_granularity else "reports"

# --- Subsample ---
st.sidebar.header("Performance")
subsample_perc = st.sidebar.slider("Subsample (%)", 10, 100, 100, 5)




# =====================================================================
# 7. Precompute filenames and pipeline triggers
# =====================================================================

def get_precomputed_filenames(csv_path, model_name, split_sentences):
    base = os.path.splitext(os.path.basename(csv_path))[0]
    safe_model = re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)
    suf = "sentences" if split_sentences else "reports"
    return (
        str(CACHE_DIR / f"precomputed_{base}_{suf}_docs.npy"),
        str(CACHE_DIR / f"precomputed_{base}_{safe_model}_{suf}_embeddings.npy"),
    )

DOCS_FILE, EMBEDDINGS_FILE = get_precomputed_filenames(
    CSV_PATH, selected_embedding_model, selected_granularity
)

# --- Cache management (after DOCS_FILE / EMBEDDINGS_FILE exist) ---
st.sidebar.markdown("### Cache")
if st.sidebar.button("Clear cached files for this configuration", use_container_width=True):
    try:
        for p in (DOCS_FILE, EMBEDDINGS_FILE):
            if os.path.exists(p):
                os.remove(p)
        # also clear Streamlit caches tied to these functions
        try:
            load_precomputed_data.clear()   # st.cache_data func
        except Exception:
            pass
        try:
            perform_topic_modeling.clear()  # st.cache_data func
        except Exception:
            pass

        st.success("Deleted cached docs/embeddings and cleared caches. Click **Prepare Data** again.")
        st.rerun()
    except Exception as e:
        st.error(f"Failed to delete cache files: {e}")




# =====================================================================
# 8. Prepare Data OR Run Analysis
# =====================================================================

if not os.path.exists(EMBEDDINGS_FILE):
    st.warning(f"No precomputed embeddings found for this configuration ({granularity_label} / {selected_embedding_model}).")
    
    if st.button("Prepare Data for This Configuration"):
        generate_and_save_embeddings(
            CSV_PATH, DOCS_FILE, EMBEDDINGS_FILE,
            selected_embedding_model, selected_granularity, selected_device
        )

else:
    # Load cached data
    docs, embeddings = load_precomputed_data(DOCS_FILE, EMBEDDINGS_FILE)

    # Coerce to 2-D float array even if saved as object
    embeddings = np.asarray(embeddings)
    if embeddings.dtype == object or embeddings.ndim != 2:
        try:
            embeddings = np.vstack(embeddings).astype(np.float32)
        except Exception:
            st.error("Cached embeddings are invalid. Please regenerate them for this configuration.")
            st.stop()

    # Subsample
    if subsample_perc < 100:
        n = int(len(docs) * (subsample_perc / 100))
        idx = np.random.choice(len(docs), size=n, replace=False)
        docs = [docs[i] for i in idx]
        # embeddings = embeddings[idx]
        embeddings = np.asarray(embeddings)
        embeddings = embeddings[idx, :]   # keep it 2-D
        st.warning(f"Running analysis on {subsample_perc}% subsample ({len(docs)} documents)")

    st.metric("Documents to Analyze", len(docs), granularity_label)

    # --- Parameter controls ---
    st.sidebar.header("Model Parameters")

    use_vectorizer = st.sidebar.checkbox("Use CountVectorizer", value=True)

    with st.sidebar.expander("Vectorizer"):
        ng_min = st.slider("Min N-gram", 1, 5, 1)
        ng_max = st.slider("Max N-gram", 1, 5, 2)
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
        nr_topics   = st.text_input("nr_topics", value="auto")
        top_n_words = st.slider("top_n_words", 5, 25, 10)

    # --- Build config ---
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
    }

    # --- Run Button ---
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
        # --- migrate old keys for backward-compat ---
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
                st.error("Cached embeddings are invalid (object/ragged). Click **Prepare Data** to regenerate.")
                st.stop()

        if embeddings.shape[0] != len(docs):
            st.error(f"len(docs)={len(docs)} but embeddings.shape[0]={embeddings.shape[0]}.\n"
                    "Likely stale cache (e.g., switched sentences↔reports or model). "
                    "Use the **Clear cache** button below and regenerate.")
            st.stop()


        with st.spinner("Performing topic modeling..."):
            model, reduced, labels, n_topics, outlier_pct = perform_topic_modeling(
                docs, embeddings, get_config_hash(current_config)
            )
        st.session_state.latest_results = (model, reduced, labels)

        # Save in history
        entry = {
            "timestamp": str(pd.Timestamp.now()),
            "config": current_config,
            "num_topics": n_topics,
            "outlier_pct": f"{outlier_pct:.2f}%",
            "llm_labels": [
                name for name in model.get_topic_info().Name.values
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

            st.subheader("Topic Visualization")
            fig, _ = datamapplot.create_plot(reduced, labs)
            st.pyplot(fig)

            st.subheader("Topic Info")
            st.dataframe(tm.get_topic_info())
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
                    st.write(f"**Outliers:** {entry.get('outlier_pct', entry.get('outlier_perc', 'N/A'))}")
                    st.write("**LLM Labels:**")
                    st.write(entry["llm_labels"])
                    with st.expander("Show full configuration"):
                        st.json(entry["config"])
