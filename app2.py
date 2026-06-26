"""
File: app.py
Description: Streamlit app for advanced topic modeling on Innerspeech dataset
             with BERTopic, UMAP, HDBSCAN. (LLM features disabled for lite deployment)
Last Modified: 21/04/2026
@corresp author: r.beaut@sussex.ac.uk
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

# from huggingface_hub import hf_hub_download, InferenceClient # for the LLM API command
from huggingface_hub import InferenceClient # for the LLM API command

from typing import Any

from io import BytesIO
import zipfile


import hashlib
import uuid
from datetime import datetime
import altair as alt

from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px


#to remove funciton locally defined here, we can use importing from mosaic_core
# from mosaic_core.core_functions import (
#     pick_text_column, list_text_columns, slugify, clean_label,
#     get_config_hash, make_run_id, cleanup_old_cache,
#     load_csv_texts, count_clean_reports, preprocess_texts,
#     run_topic_model, get_topic_labels, get_outlier_stats, get_num_topics,
#     SYSTEM_PROMPT, USER_TEMPLATE, generate_llm_labels,
#     labels_cache_path, load_cached_labels, save_labels_cache,
#     get_hf_status_code,
# )



# =====================================================================
# NLTK setup
# =====================================================================

NLTK_DATA_DIR = "/usr/local/share/nltk_data"
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_DIR)

# try to ensure both punkt_tab (new NLTK) and punkt (old NLTK) are available
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

def _cleanup_old_cache(current_slug: str):
    """Deletes precomputed cache files that do not match the current dataset slug."""
    if not CACHE_DIR.exists():
        return

    removed_count = 0
    cache_files = list(CACHE_DIR.glob("precomputed_*.npy")) + list(CACHE_DIR.glob("precomputed_*_docs.json"))
    for p in cache_files:
        if current_slug not in p.name:
            try:
                p.unlink() 
                removed_count += 1
            except Exception as e:
                print(f"Error deleting {p.name}: {e}")
                
    if removed_count > 0:
        print(f"Auto-cleanup: Removed {removed_count} old cache files.")

# default names we know from MOSAIC but NOT a hard constraint
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
    Return all columns
    makes the selector work with any column name / dtype
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
    **Beauté, R., et al. (2026).**  
    **Mapping of Subjective Accounts into Interpreted Clusters (MOSAIC): Topic Modelling and LLM applied to Stroboscopic Phenomenology**  
    https://doi.org/10.1093/nc/niag008
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

#start add for comparison
RUNS_DIR = EVAL_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

def make_run_id(cfg: dict) -> str:
    cfg_str = json.dumps(cfg, sort_keys=True)
    h = hashlib.md5(cfg_str.encode("utf-8")).hexdigest()[:8]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{h}"

def save_run_snapshot(
    run_id: str,
    tm: BERTopic,
    reduced: np.ndarray,
    labs: list[str],
    dataset_title: str,
    csv_path: str,
    current_config: dict,
) -> dict:
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    info = tm.get_topic_info()
    n_units = int(info["Count"].sum()) if "Count" in info.columns else len(labs)
    outlier_count = 0
    if {"Topic", "Count"}.issubset(info.columns) and (-1 in info["Topic"].values):
        outlier_count = int(info.loc[info["Topic"] == -1, "Count"].iloc[0])

    n_topics = int((info["Topic"] != -1).sum()) if "Topic" in info.columns else None
    outlier_pct = (100.0 * outlier_count / n_units) if n_units else 0.0


    # fig, _ = datamapplot.create_plot(
    #     reduced,
    #     labs,
    #     noise_label="Unlabelled",
    #     noise_color="#CCCCCC",
    #     label_font_size=11,
    #     arrowprops={"arrowstyle": "-", "color": "#333333"},
    # )

    fig, _ = datamapplot.create_plot(
        reduced,
        labs,
        noise_label="Unlabelled",
        noise_color="#CCCCCC",
        figsize=(18, 18),
        dynamic_label_size=True,
        dynamic_label_size_scaling_factor=0.85,
        label_font_size=10,
        label_wrap_width=15,
        label_margin_factor=1.5,
        arrowprops={"arrowstyle": "-", "color": "#333333"}
    )
    fig.suptitle(f"{dataset_title}: MOSAIC Topic Map", fontsize=16, y=0.99)

    plot_png = run_dir / "plot.png"
    fig.savefig(plot_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # save topic info
    topic_info_csv = run_dir / "topic_info.csv"
    info.to_csv(topic_info_csv, index=False)

    # save meta
    meta = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset_title": dataset_title,
        "csv_path": str(csv_path),
        "config": current_config,
        "n_units": int(n_units),
        "n_topics": int(n_topics) if n_topics is not None else None,
        "outlier_count": int(outlier_count),
        "outlier_pct": float(outlier_pct),
        "artifacts": {
            "plot_png": str(plot_png),
            "topic_info_csv": str(topic_info_csv),
        },
    }
    meta_json = run_dir / "meta.json"
    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    meta["artifacts"]["meta_json"] = str(meta_json)

    return meta

#end add for comparison 

with st.sidebar.expander("About the dataset name", expanded=False):
    st.markdown(
        f"""
- The name above is converted to **UPPER CASE** and used as a folder name.
- If the folder doesn’t exist, it will be **created**:
  - Preprocessed CSVs: `{PROC_DIR}`
  - Exports (results): `{EVAL_DIR}`
- If you choose **Use preprocessed CSV on server**, I’ll list CSVs in `{PROC_DIR}`.
- If you **upload** a CSV, it will be saved to `{PROC_DIR}/uploaded.csv`.
- If you **load precomputed embeddings** (`.npy` + `docs.json` from `run_embeddings.py`),
  embedding is skipped and the app runs topic modelling directly on them.
        """.strip()
    )


def _list_server_csvs(proc_dir: Path) -> list[str]:
    # Non-recursive on purpose: per-session uploads live under proc_dir/_uploads/<id>/
    # and must never appear here (otherwise one visitor's data leaks to others).
    return [str(p) for p in sorted(proc_dir.glob("*.csv"))]


DATASETS = None  
HISTORY_FILE = str(PROC_DIR / "run_history.json")

# =====================================================================
# 3. Embedding loaders
# =====================================================================


@st.cache_resource
def load_embedding_model(model_name):
    st.info(f"Loading embedding model '{model_name}'...")
    return SentenceTransformer(model_name)



@st.cache_data
def load_precomputed_data(docs_file, embeddings_file):
    with open(docs_file, "r", encoding="utf-8") as f:
        docs = json.load(f)
    emb = np.load(embeddings_file)
    return docs, emb


# =====================================================================
# 4. LLM loaders
# =====================================================================

# Approximate price for cost estimates in the UI only
# Novita Llama 3 8B is around $0.04 per 1M input tokens
HF_APPROX_PRICE_PER_MTOKENS_USD = 0.04


@st.cache_resource
def get_hf_client(model_id: str):
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            token = st.secrets.get("HF_TOKEN")
        except Exception:
            token = None

    client = InferenceClient(model=model_id, token=token)
    return client, token

def _labels_cache_path(config_hash: str, model_id: str) -> Path:
    safe_model = re.sub(r"[^a-zA-Z0-9_.-]", "_", model_id)
    short_hash = hashlib.md5(config_hash.encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"llm_labels_{safe_model}_{short_hash}.json"

def _hf_status_code(e: Exception) -> int | None:
    """Extract HTTP status code from huggingface_hub error, if present."""
    resp = getattr(e, "response", None)
    return getattr(resp, "status_code", None)



SYSTEM_PROMPT = """You are an expert phenomenologist. Your task is to perform a "phenomenological reduction" on a cluster of subjective reports.

Goal: Identify the invariant structural theme shared by the majority of the reports.

Rules:
1. FOCUS ON STRUCTURE, NOT CONTENT: Do not label the specific object seen (e.g., "Night sky", "Monster"), but the mode of experience (e.g., "Sense of vastness", "Perceiving threatening entities").
2. AVOID OUTLIERS: If a specific detail (like a specific location or object) appears in only 1-2 reports, IGNORE IT.
3. BE CONCISE: Use scientifically precise noun phrases (3-8 words).
4. NO META-COMMENTARY: Output ONLY the label. Do not use quotes or introductory text.
"""



USER_TEMPLATE = """Below are representative sentences from a single cluster of experiences:

{documents}

Keywords: {keywords}

Task: Synthesize the core shared experience into a label.

Critical Constraints:
1. Identify the *common denominator* shared by the majority of these sentences.
2. EXCLUDE details that appear in only one or two sentences (e.g. specific objects like "night sky", "machine", or specific locations), unless they are central to the whole cluster.
3. If the cluster is generic (e.g. just "feeling good"), use a generic label. Do not force a specific image if it isn't there.

Output ONLY the label (3-8 words):"""


def _clean_label(x: str) -> str:
    # handle None
    x = (x or "").strip()
    
    # take first line only (if LLM rambles)
    x = x.splitlines()[0].strip()
    
    # remove surrounding quotes
    x = x.strip(' "\'`')
    
    # remove trailing punctuation (.,;:)
    x = re.sub(r"[.,;:]+$", "", x).strip()
    
    # remove "Experience of..." prefixes
    x = re.sub(
        r"^(Experiential(?:\s+Phenomenon)?|Experience|Experience of|Subjective Experience of|Phenomenon of)\s+",
        "",
        x,
        flags=re.IGNORECASE,
    )

    # final cleanup
    x = x.strip()
    
    # fallback if empty
    return x or "Unlabelled"




def generate_labels_via_chat_completion(
    topic_model: BERTopic,
    docs: list[str],
    config_hash: str,
    model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    max_topics: int = 50,
    max_docs_per_topic: int = 10,
    doc_char_limit: int = 500,
    temperature: float = 0.2, #deterministic, stable outputs.
    force: bool = False,
    system_prompt: str | None = None,
    user_template: str | None = None) -> dict[int, str]:
    """
    Label topics after fitting (fast + stable on Spaces).
    Returns {topic_id: label}.
    """

    # Fall back to the built-in prompts when the UI doesn't override them.
    system_prompt = system_prompt or SYSTEM_PROMPT
    user_template = user_template or USER_TEMPLATE

    st.session_state["hf_last_model_param"] = model_id

    # Fold the prompts into the cache key so editing a prompt regenerates labels.
    _prompt_sig = hashlib.md5((system_prompt + "||" + user_template).encode("utf-8")).hexdigest()[:8]
    cache_path = _labels_cache_path(config_hash + "|" + _prompt_sig, model_id)

    if (not force) and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return {int(k): str(v) for k, v in cached.items()}
        except Exception:
            pass

    client, token = get_hf_client(model_id)
    if not token:
        raise RuntimeError("No HF_TOKEN found in env/secrets.")

    topic_info = topic_model.get_topic_info()
    topic_info = topic_info[topic_info.Topic != -1].head(max_topics)

    labels: dict[int, str] = {}
    prog = st.progress(0)
    total = len(topic_info)

    for i, tid in enumerate(topic_info.Topic.tolist(), start=1):
        words = topic_model.get_topic(tid) or []
        keywords = ", ".join([w for (w, _) in words[:10]])

        try:
            reps = (topic_model.get_representative_docs(tid) or [])[:max_docs_per_topic]
        except Exception:
            reps = []

        # keep prompt small
        reps = [r.replace("\n", " ").strip()[:doc_char_limit] for r in reps if str(r).strip()]
        if reps:
            docs_block = "\n".join([f"- {r}" for r in reps])
        else:
            docs_block = "- (No representative docs available)"

        user_prompt = user_template.format(documents=docs_block, keywords=keywords)
        # Store one example prompt (for UI inspection) – will be overwritten each run
        st.session_state["hf_last_example_prompt"] = user_prompt

        try:
            out = client.chat_completion(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=60, #Upper bound on how many tokens the model is allowed to generate as output for that label
                temperature=temperature,
                stop=["\n"],
            )
            # Store the provider-returned model id (if available)
            provider_model = getattr(out, "model", None)
            if provider_model:
                st.session_state["hf_last_provider_model"] = provider_model
        except Exception as e:
            if _hf_status_code(e) == 402:
                raise RuntimeError(
                    "Hugging Face returned 402 Payment Required for this LLM call. "
                    "You have used up the monthly Inference Provider credits on this "
                    "account. Either upgrade to PRO / enable pay-as-you-go, or skip "
                    "the 'Generate LLM labels (API)' step."
                ) from e

            raise



        usage = getattr(out, "usage", None)
        total_tokens = None

        # `usage` might be a dict (raw JSON) or an object with attributes
        if isinstance(usage, dict):
            total_tokens = usage.get("total_tokens")
        else:
            total_tokens = getattr(usage, "total_tokens", None)

        if total_tokens is not None:
            st.session_state.setdefault("hf_tokens_total", 0)
            st.session_state["hf_tokens_total"] += int(total_tokens)

        raw = out.choices[0].message.content
        labels[int(tid)] = _clean_label(raw)


        prog.progress(int(100 * i / max(total, 1)))

    try:
        cache_path.write_text(json.dumps({str(k): v for k, v in labels.items()}, indent=2), encoding="utf-8")
    except Exception:
        pass

    return labels



# =====================================================================
# 5. Topic modeling function
# =====================================================================


def get_config_hash(cfg):
    return json.dumps(cfg, sort_keys=True)


@st.cache_data
def perform_topic_modeling(_docs, _embeddings, config_hash, dataset_key=""):
    """Fit BERTopic. dataset_key is only for cache invalidation across datasets."""

    # Config-keyed on-disk cache: a different config (incl. the random seed) gets
    # its own files, so changing settings actually recomputes and concurrent users
    # with different data never load each other's model.
    _cfg_key = hashlib.md5((config_hash + "|" + str(dataset_key)).encode("utf-8")).hexdigest()[:10]
    saved_model_path = CACHE_DIR / f"topic_model_{_cfg_key}"
    saved_reduced_path = CACHE_DIR / f"reduced_2d_{_cfg_key}.npy"
    saved_topics_path = CACHE_DIR / f"topics_{_cfg_key}.json"

    # Legacy fixed-name cache shipped by run_pipeline.py (no config key). Only used
    # as a fallback so previously shipped models keep working.
    _legacy_model = CACHE_DIR / "topic_model"
    _legacy_reduced = CACHE_DIR / "reduced_2d.npy"
    _legacy_topics = CACHE_DIR / "topics.json"
    if not (saved_model_path.exists() and saved_reduced_path.exists() and saved_topics_path.exists()):
        if _legacy_model.exists() and _legacy_reduced.exists() and _legacy_topics.exists():
            saved_model_path, saved_reduced_path, saved_topics_path = (
                _legacy_model, _legacy_reduced, _legacy_topics,
            )

    if saved_model_path.exists() and saved_reduced_path.exists() and saved_topics_path.exists():
        topic_model = BERTopic.load(str(saved_model_path))
        reduced = np.load(str(saved_reduced_path))
        with open(saved_topics_path) as f:
            topics = json.load(f)
        info = topic_model.get_topic_info()
        outlier_pct = 0.0
        if -1 in info.Topic.values:
            outlier_pct = (info.Count[info.Topic == -1].iloc[0] / info.Count.sum()) * 100
        name_map = info.set_index("Topic")["Name"].to_dict()
        all_labels = [name_map.get(t, "Unlabelled") for t in topics]
        return topic_model, reduced, all_labels, len(info) - 1, outlier_pct

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

    rep_model = None  # Use BERTopic defaults for representation

    seed = int(config.get("random_seed", 42))

    umap_model = UMAP(random_state=seed, metric="cosine", **config["umap_params"])
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

    # Override BERTopic's default 3 representative documents limit
    doc_df = pd.DataFrame({"Document": _docs, "ID": range(len(_docs)), "Topic": topics})
    repr_docs = topic_model._extract_representative_docs(
        c_tf_idf=topic_model.c_tf_idf_,
        documents=doc_df,
        topics=topic_model.topic_representations_,
        nr_repr_docs=20  # Get up to 20 representative docs per topic (instead of default 3 
    )
    topic_model.representative_docs_ = repr_docs[0]

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
        random_state=seed,
    ).fit_transform(_embeddings)

    # Persist to the config-keyed cache so an identical re-run is instant and the
    # right model is loaded back (never another config's).
    try:
        topic_model.save(str(saved_model_path))
        np.save(str(saved_reduced_path), reduced)
        with open(saved_topics_path, "w") as f:
            json.dump([int(t) for t in topics], f)
    except Exception as e:
        print(f"Could not persist topic-model cache: {e}")

    return topic_model, reduced, all_labels, len(info) - 1, outlier_pct


# =====================================================================
# 5b. Zero-shot helpers
# =====================================================================

class _ZSPassThrough:
    """Dummy UMAP that passes embeddings unchanged (zero-shot doesn't need dim-reduction)."""
    def fit(self, X, y=None): return self
    def transform(self, X): return X

class _ZSDummyClustering:
    """Dummy HDBSCAN that marks every doc as outlier so zero-shot does all the work."""
    def __init__(self): self.labels_ = None
    def fit(self, X, y=None):
        self.labels_ = np.array([-1] * len(X))
        return self


@st.cache_data
def run_zeroshot(_docs, _embeddings, categories, min_similarity, embedding_model_name, dataset_key=""):
    """Run BERTopic zero-shot classification against a fixed list of categories."""
    emb_model = load_embedding_model(embedding_model_name)
    topic_model = BERTopic(
        embedding_model=emb_model,
        umap_model=_ZSPassThrough(),
        hdbscan_model=_ZSDummyClustering(),
        vectorizer_model=CountVectorizer(stop_words="english"),
        zeroshot_topic_list=list(categories),
        zeroshot_min_similarity=min_similarity,
        verbose=False,
    )
    topics, _ = topic_model.fit_transform(list(_docs), np.asarray(_embeddings))
    return topics, topic_model.get_topic_info(), topic_model


# =====================================================================
# 5c. Condition-comparison helpers
# =====================================================================

def _parse_condition_csv(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Accept two CSV formats exported by the main pipeline and return
    {topic_name: [sentence, sentence, ...]}

    Supported formats:
    - "Row per topic"   : columns include `topic_name` and `texts` (pipe-separated)
    - "Long / all-sentences" : columns include `Topic Name` and `Document`
    """
    if "topic_name" in df.columns and "texts" in df.columns:
        out: dict[str, list[str]] = {}
        for _, row in df.iterrows():
            name = str(row["topic_name"]).strip()
            if name in ("Unlabelled", "Outlier", "Too Specific (Idiosyncratic)", ""):
                continue
            texts_raw = row["texts"]
            if not isinstance(texts_raw, str) or not texts_raw.strip():
                continue
            sentences = [s.strip() for s in texts_raw.split(" | ") if s.strip()]
            if sentences:
                out[name] = sentences
        return out
    if "Topic Name" in df.columns and "Document" in df.columns:
        out = {}
        for _, row in df.iterrows():
            name = str(row["Topic Name"]).strip()
            if name in ("Unlabelled", "Outlier", "Too Specific (Idiosyncratic)", ""):
                continue
            sentence = str(row["Document"]).strip()
            if sentence:
                out.setdefault(name, []).append(sentence)
        return out
    return {}


@st.cache_data
def _embed_sentences(sentences: tuple[str, ...], model_name: str) -> np.ndarray:
    """Embed a deduplicated tuple of sentences (cache-friendly)."""
    model = load_embedding_model(model_name)
    return model.encode(list(sentences), show_progress_bar=False, convert_to_numpy=True)


@st.cache_data
def compute_condition_similarity(
    topics_a: dict,
    topics_b: dict,
    model_name: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Given two dicts of {topic_name: [sentences]}, embed all sentences,
    build mean topic vectors, and return (similarity_matrix_df, vec_a, vec_b).
    """
    # Collect all unique sentences across both conditions for a single encode pass
    all_sents_a = [s for sents in topics_a.values() for s in sents]
    all_sents_b = [s for sents in topics_b.values() for s in sents]

    unique_sents = list(dict.fromkeys(all_sents_a + all_sents_b))
    embeddings = np.asarray(_embed_sentences(tuple(unique_sents), model_name), dtype=np.float64)

    # ── Mean-centering (remove the common component) ──────────────────────────
    # Sentence-transformer embeddings are anisotropic: they occupy a narrow cone,
    # so *every* pair of topics scores a high cosine (~0.6–0.99) and even unrelated
    # themes look "strongly correlated". Subtracting the global centroid removes
    # that shared direction and restores a discriminative range (≈ -0.4 … 0.9),
    # where unrelated topics fall near 0 (or negative) and real matches stand out.
    if len(embeddings) > 1:
        embeddings = embeddings - embeddings.mean(axis=0, keepdims=True)

    sent_to_vec = {s: embeddings[i] for i, s in enumerate(unique_sents)}

    def _mean_vec(sents):
        vecs = [sent_to_vec[s] for s in sents if s in sent_to_vec]
        return np.mean(vecs, axis=0) if vecs else None

    topic_names_a = list(topics_a.keys())
    topic_names_b = list(topics_b.keys())

    vecs_a = np.array([v for name in topic_names_a if (v := _mean_vec(topics_a[name])) is not None])
    vecs_b = np.array([v for name in topic_names_b if (v := _mean_vec(topics_b[name])) is not None])

    valid_a = [n for n in topic_names_a if _mean_vec(topics_a[n]) is not None]
    valid_b = [n for n in topic_names_b if _mean_vec(topics_b[n]) is not None]

    sim = cosine_similarity(vecs_a, vecs_b)
    sim_df = pd.DataFrame(sim, index=valid_a, columns=valid_b)
    return sim_df, vecs_a, vecs_b


def generate_and_save_embeddings(
    csv_path,
    docs_file,
    emb_file,
    selected_embedding_model,
    split_sentences,
    device,
    text_col=None,
    min_words: int = 0,
):

    # ---------------------
    # 1. Load & Clean CSV
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

    # identify metadata columns (everything except the text)
    metadata_cols = [c for c in df.columns if c != "reflection_answer_english"]

    # ---------------------
    # 2. Process Text & Metadata
    # ---------------------
    final_docs = []
    removed_texts = []
    final_metadata_rows = []

    granularity_label = "sentences" if split_sentences else "reports"
    total_units_before = 0

    if split_sentences:
        # Loop through every report
        for idx, row in df.iterrows():
            report_text = row["reflection_answer_english"]
            
            # Tokenise
            try:
                sentences = nltk.sent_tokenize(report_text)
            except LookupError:
                nltk.download('punkt')
                sentences = nltk.sent_tokenize(report_text)
            
            total_units_before += len(sentences)
            
            # Check every sentence
            for sent in sentences:
                # Filter by word count
                if min_words > 0 and len(sent.split()) < min_words:
                    removed_texts.append(sent)
                    continue
                
                # Keep valid sentence
                final_docs.append(sent)
                
                # Copy metadata for this sentence
                meta = {c: row[c] for c in metadata_cols}
                meta["_source_row_idx"] = idx  # Useful to trace back to original row
                final_metadata_rows.append(meta)

    else:
        # Report level processing
        total_units_before = len(df)
        
        for idx, row in df.iterrows():
            report_text = row["reflection_answer_english"]
            
            # Filter by word count
            if min_words > 0 and len(report_text.split()) < min_words:
                removed_texts.append(report_text)
                continue
                
            # Keep valid report
            final_docs.append(report_text)
            
            # Copy metadata
            meta = {c: row[c] for c in metadata_cols}
            meta["_source_row_idx"] = idx
            final_metadata_rows.append(meta)

    # ---------------------
    # 3. Update Session State 
    # ---------------------
    total_units_after = len(final_docs)
    removed_count = len(removed_texts)

    st.session_state["last_data_stats"] = {
        "granularity": granularity_label,
        "min_words": int(min_words or 0),
        "total_before": int(total_units_before),
        "total_after": int(total_units_after),
        "removed": int(removed_count),
    }
    st.session_state["last_removed_units"] = removed_texts

    if min_words and min_words > 0:
        st.info(
            f"Preprocessing: started with {total_units_before} {granularity_label}, "
            f"removed {removed_count} shorter than {min_words} words; "
            f"{total_units_after} remaining."
        )
    else:
        st.info(f"Preprocessing: {total_units_after} {granularity_label} prepared.")

    with open(docs_file, "w", encoding="utf-8") as f:
        json.dump(final_docs, f, ensure_ascii=False)
    st.success(f"Prepared {len(final_docs)} documents")

    metadata_file = docs_file.replace("_docs.json", "_metadata.csv")
    pd.DataFrame(final_metadata_rows).to_csv(metadata_file, index=False)

    # ---------------------
    # 5. Generate Embeddings
    # ---------------------
    st.info(
        f"Encoding {len(final_docs)} documents with {selected_embedding_model} on {device}"
    )

    model = load_embedding_model(selected_embedding_model)

    # Device setup
    encode_device = None
    batch_size = 32
    if device == "CPU":
        encode_device = "cpu"
        batch_size = 64
    else:
        import torch
        if torch.cuda.is_available():
            encode_device = "cuda"
        elif torch.backends.mps.is_available():
            encode_device = "mps"
        else:
            encode_device = "cpu"

    embeddings = model.encode(
        final_docs,
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
# 7. Sidebar — dataset, upload, parameters
# =====================================================================

st.sidebar.header("Data Input Method")

# Per-visitor session id. On a shared host (e.g. a Hugging Face Space) every
# visitor runs inside the same process, so without this an uploaded CSV would be
# saved into the shared PROC_DIR and shown in *everyone's* "server CSV" list.
# We use this id to give each session a private upload + cache workspace.
if "_session_id" not in st.session_state:
    st.session_state["_session_id"] = uuid.uuid4().hex[:12]

source = st.sidebar.radio(
    "Choose data source",
    ("Use preprocessed CSV on server", "Upload my own CSV",
     "Load precomputed embeddings (.npy + docs)"),
    index=0,
    key="data_source",
)

uploaded_csv_path = None
CSV_PATH = None  # will be set in the chosen branch
PRECOMPUTED_MODE = False  # set True in the precomputed-embeddings branch

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
elif source == "Upload my own CSV":
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
                up.seek(0)  
                tmp_df = pd.read_csv(up, encoding=encoding)
                success_encoding = encoding
                break  
            except UnicodeDecodeError:
                continue  

        if tmp_df is None:
            st.error("Could not decode file. Please save your CSV as 'CSV UTF-8' in Excel.")
            st.stop()
            
        if tmp_df.empty:
            st.error("Uploaded CSV is empty.")
            st.stop()
            
        print(f"Successfully loaded CSV using {success_encoding} encoding.")


        # Private per-session workspace: the uploaded CSV and ALL derived artifacts
        # (docs, embeddings, metadata, topic model) live here so they are never
        # listed or reused by other visitors sharing this host.
        session_dir = PROC_DIR / "_uploads" / st.session_state["_session_id"]
        session_dir.mkdir(parents=True, exist_ok=True)
        CACHE_DIR = session_dir / "cache"
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        safe_filename = _slugify(os.path.splitext(up.name)[0])
        _cleanup_old_cache(safe_filename)
        uploaded_csv_path = str((session_dir / f"{safe_filename}.csv").resolve())

        tmp_df.to_csv(uploaded_csv_path, index=False)
        st.success("Uploaded CSV saved to a private session workspace (not shared with other users).")
        CSV_PATH = uploaded_csv_path
    else:
        st.info("Upload a CSV to continue.")
        st.stop()

else:  # "Load precomputed embeddings (.npy + docs)"
    st.sidebar.caption(
        "Upload the **`*_embeddings.npy`** and **`*_docs.json`** files produced by "
        "`run_embeddings.py` (e.g. computed on a GPU/HPC with the Qwen 4B model). "
        "Embedding is skipped entirely — the app goes straight to topic modelling. "
        "Make sure the **embedding model** selected below matches the one used to "
        "create the .npy file."
    )
    pre_label = st.sidebar.text_input(
        "Dataset label (used for naming/results)", value="precomputed",
        key="precomp_label",
    )
    up_emb = st.sidebar.file_uploader(
        "Embeddings file (.npy)", type=["npy"], key="precomp_npy"
    )
    up_docs = st.sidebar.file_uploader(
        "Documents file (docs.json — list of strings)", type=["json"], key="precomp_docs"
    )
    if up_emb is None or up_docs is None:
        st.info("Upload **both** the `.npy` embeddings and the `docs.json` to continue.")
        st.stop()

    # Private per-session workspace (same isolation as the upload branch).
    session_dir = PROC_DIR / "_uploads" / st.session_state["_session_id"]
    session_dir.mkdir(parents=True, exist_ok=True)
    CACHE_DIR = session_dir / "cache"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        up_docs.seek(0)
        _docs_list = json.load(up_docs)
    except Exception as e:
        st.error(f"Could not read docs.json: {e}")
        st.stop()
    if not isinstance(_docs_list, list) or not _docs_list:
        st.error("docs.json must be a non-empty JSON list of strings.")
        st.stop()
    _docs_list = [str(d) for d in _docs_list]

    # Synthesize a one-row-per-document CSV so every downstream step (text-column
    # selection, counts, metadata, export) works exactly as for an uploaded CSV.
    safe_filename = _slugify(pre_label or "precomputed")
    CSV_PATH = str((session_dir / f"{safe_filename}.csv").resolve())
    pd.DataFrame({"text": _docs_list}).to_csv(CSV_PATH, index=False)
    st.success(
        f"Loaded {len(_docs_list)} precomputed documents into a private session "
        "workspace (not shared with other users)."
    )
    PRECOMPUTED_MODE = True

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

#preprocessing action: remove sentences with less than N words
min_words = st.sidebar.slider(
    f"Remove {granularity_label} shorter than N words",
    min_value=1,
    max_value=20,
    value=3,  # default = 3 words
    step=1,
    help="Units (sentences or reports) with fewer words than this will be discarded "
         "during preprocessing. After changing, click 'Prepare Data for This Configuration'.",
)

subsample_perc = st.sidebar.slider("Data sampling (%)", 10, 100, 100, 5)

# Random seed for every stochastic step (UMAP for clustering, the 2D projection,
# and subsampling). Defined here so it is in scope for both the subsample below
# and the topic-model config further down.
with st.sidebar.expander("Reproducibility (random seed)"):
    st.caption(
        "UMAP and subsampling are stochastic. A fixed seed makes runs reproducible; "
        "tick 'new seed each run' to explore how stable the topics are."
    )
    randomize_seed = st.checkbox("New random seed each run", value=False)
    if randomize_seed:
        random_seed = int(np.random.randint(0, 1_000_000))
        st.caption(f"This run's seed: `{random_seed}`")
    else:
        random_seed = int(
            st.number_input("Random seed", min_value=0, max_value=1_000_000, value=42, step=1)
        )

st.sidebar.markdown("---")

# ---------------------------------------------------------------------
# Embedding model & device
# ---------------------------------------------------------------------

ALL_EMBEDDING_MODELS = (
    "BAAI/bge-small-en-v1.5",
    "intfloat/multilingual-e5-large-instruct",
    "Qwen/Qwen3-Embedding-0.6B",
    "Qwen/Qwen3-Embedding-4B",
    "sentence-transformers/all-mpnet-base-v2",
)

def _detect_model_from_filename(filename: str) -> str | None:
    """Return the first known model whose sanitized name appears in the filename."""
    for model in ALL_EMBEDDING_MODELS:
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", model)
        if safe in filename:
            return model
    return None

st.sidebar.header("Model Selection")

selected_embedding_model = st.sidebar.selectbox(
    "Choose an embedding model",
    (
        "BAAI/bge-small-en-v1.5",
        "intfloat/multilingual-e5-large-instruct",
        "Qwen/Qwen3-Embedding-0.6B",
        "Qwen/Qwen3-Embedding-4B",
        "sentence-transformers/all-mpnet-base-v2",
    ),
    help="Unsure which model to pick? Check the [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) for performance maximising on Clustering and STS tasks."
)

selected_device = st.sidebar.radio(
    "Processing device",
    ["GPU", "CPU"],
    index=0,
)


@st.cache_data
def _detect_runtime_device() -> str:
    """Return the GPU/accelerator actually available to this process: 'cuda', 'mps' or 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


_runtime_device = _detect_runtime_device()

if _runtime_device == "cuda":
    st.sidebar.success("🟢 GPU (CUDA) detected — it will be used for embedding.")
elif _runtime_device == "mps":
    st.sidebar.success("🟢 Apple GPU (MPS) detected — it will be used for embedding.")
else:
    st.sidebar.warning(
        "🟡 No GPU available — running on **CPU**. "
        "Selecting 'GPU' above has no effect here and falls back to CPU."
    )
    if selected_device == "GPU":
        st.sidebar.caption(
            "On a Hugging Face Space, a GPU only appears here if you upgrade the Space "
            "hardware *and* install a CUDA build of torch (the pinned build is CPU-only)."
        )

# =====================================================================
# 7. Precompute filenames and pipeline triggers
# =====================================================================


def get_precomputed_filenames(csv_path, model_name, split_sentences, text_col,min_words):
    base = os.path.splitext(os.path.basename(csv_path))[0]
    safe_model = re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)
    suf = "sentences" if split_sentences else "reports"

    col_suffix = ""
    if text_col:
        safe_col = re.sub(r"[^a-zA-Z0-9_-]", "_", text_col)
        col_suffix = f"_{safe_col}"

    mw_suffix = f"_minw{int(min_words or 0)}"


    return (
    str(CACHE_DIR / f"precomputed_{base}{col_suffix}_{suf}{mw_suffix}_docs.json"),
    str(CACHE_DIR / f"precomputed_{base}_{safe_model}{col_suffix}_{suf}{mw_suffix}_embeddings.npy"),
    )


DOCS_FILE, EMBEDDINGS_FILE = get_precomputed_filenames(
    CSV_PATH, selected_embedding_model, selected_granularity, selected_text_column, min_words
)

METADATA_FILE = DOCS_FILE.replace("_docs.json", "_metadata.csv")

if PRECOMPUTED_MODE:
    # Persist the uploaded files at the exact cache paths the app expects, so the
    # "embeddings already exist" check below short-circuits straight to topic
    # modelling. Written once (guarded on existence) to avoid re-copying ~100 MB
    # on every Streamlit rerun.
    if not os.path.exists(DOCS_FILE):
        with open(DOCS_FILE, "w", encoding="utf-8") as _f:
            json.dump(_docs_list, _f, ensure_ascii=False)
    if not os.path.exists(EMBEDDINGS_FILE):
        up_emb.seek(0)
        with open(EMBEDDINGS_FILE, "wb") as _f:
            _f.write(up_emb.getbuffer())
    try:
        _emb_rows = np.load(EMBEDDINGS_FILE, mmap_mode="r").shape[0]
    except Exception as e:
        st.error(f"Could not read the uploaded .npy embeddings: {e}")
        st.stop()
    if _emb_rows != len(_docs_list):
        st.error(
            f"Mismatch: the .npy has {_emb_rows} rows but docs.json has "
            f"{len(_docs_list)} entries. Upload matching files (same run)."
        )
        st.stop()

# Cache management
st.sidebar.markdown("### Cache")

with st.sidebar.expander("When do I need to clear the cache?"):
    st.markdown(
        """
**You do NOT need to clear the cache when:**
- Switching to a different dataset — each dataset has its own embedding files, and the topic model cache is automatically invalidated
- Changing any parameter (embedding model, UMAP, HDBSCAN, etc.) — the cache key changes automatically

**You DO need to clear the cache when:**
- Your embeddings are corrupted or were generated from wrong data and you want to regenerate them for the current dataset + settings
- You want to force a fresh topic model run despite using identical settings on the same data

The button below deletes the embedding files for the **currently selected** dataset/settings combination only, then clears all in-memory caches.
        """
    )

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
# 8a. Module-level constants and reusable UI helpers
# =====================================================================

_ZS_DEFAULT_CATEGORIES = """\
Time, Effort and Desire
Peace, Bliss and Silence
Self-Knowledge, Autonomous Cognizance and Insight
Wakeful Presence
Pure Awareness in Dream and Sleep
Luminosity
Thoughts and Feelings
Emptiness and Non-egoic Self-awareness
Sensory Perception in Body and Space
Touching World and Self
Mental Agency
Witness Consciousness"""


def _condition_comparison_ui(embedding_model: str, key_suffix: str = "") -> None:
    """Render the full condition-comparison UI.
    key_suffix keeps widget/session-state keys unique when rendered in multiple places.
    """
    _sim_key = f"cond_sim{key_suffix}"

    cc1, cc2 = st.columns(2)
    with cc1:
        cond_name_a = st.text_input("Condition A name", value="Condition A",
                                    key=f"cond_name_a{key_suffix}")
        cond_file_a = st.file_uploader(
            "Topics summary CSV — Condition A", type=["csv"],
            key=f"cond_file_a{key_suffix}",
            help="The `topics_summary_*.csv` saved from the Main Results tab.",
        )
    with cc2:
        cond_name_b = st.text_input("Condition B name", value="Condition B",
                                    key=f"cond_name_b{key_suffix}")
        cond_file_b = st.file_uploader(
            "Topics summary CSV — Condition B", type=["csv"],
            key=f"cond_file_b{key_suffix}",
            help="The `topics_summary_*.csv` from the pipeline run on Condition B.",
        )

    sim_col, thresh_col = st.columns([1, 1])
    with sim_col:
        cond_threshold = st.slider(
            "Match threshold (centered cosine similarity)",
            min_value=-0.20, max_value=0.95, value=0.50, step=0.01,
            key=f"cond_thresh{key_suffix}",
            help="Topic pairs scoring above this value are considered 'shared'. "
                 "On the mean-centered scale, ≥0.50 is a strong/solid match, "
                 "0.35–0.50 is moderate, and <0.30 is essentially unrelated.",
        )
    with thresh_col:
        st.info(
            f"Embedding model: `{embedding_model}`  \n"
            "Sentences in each topic are embedded, **mean-centered** (the shared "
            "component is removed so unrelated topics no longer look similar), "
            "averaged to one vector per topic, then compared by cosine similarity. "
            "Scores now span roughly −0.4 … 0.9 instead of being squeezed into 0.6–1.0."
        )

    run_cond = st.button(
        "Run Comparison", type="primary",
        key=f"cond_run_btn{key_suffix}",
        disabled=(cond_file_a is None or cond_file_b is None),
    )
    if cond_file_a is None or cond_file_b is None:
        st.info("Upload a CSV for both conditions to enable the comparison.")

    if run_cond and cond_file_a and cond_file_b:
        try:
            df_a = pd.read_csv(cond_file_a)
            df_b = pd.read_csv(cond_file_b)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            return
        topics_a = _parse_condition_csv(df_a)
        topics_b = _parse_condition_csv(df_b)
        if not topics_a:
            st.error("Could not find topics in Condition A CSV. "
                     "Expected `topic_name` + `texts`, or `Topic Name` + `Document`.")
            return
        if not topics_b:
            st.error("Could not find topics in Condition B CSV.")
            return
        total_sents = sum(len(v) for v in topics_a.values()) + sum(len(v) for v in topics_b.values())
        with st.spinner(f"Embedding {total_sents} sentences and computing similarity…"):
            sim_df, _, _ = compute_condition_similarity(topics_a, topics_b, embedding_model)
        st.session_state[_sim_key] = {
            "sim_df": sim_df, "topics_a": topics_a, "topics_b": topics_b,
            "name_a": cond_name_a, "name_b": cond_name_b,
        }

    if _sim_key not in st.session_state:
        return

    cs = st.session_state[_sim_key]
    sim_df: pd.DataFrame = cs["sim_df"]
    topics_a: dict = cs["topics_a"]
    topics_b: dict = cs["topics_b"]
    name_a: str = cs["name_a"]
    name_b: str = cs["name_b"]
    threshold = cond_threshold

    # ── Heatmap ──────────────────────────────────────────────────
    st.subheader("Cosine Similarity Heatmap")

    show_all = st.checkbox(
        "Show ALL topics", value=False, key=f"heat_showall{key_suffix}",
        help="Display the full topic-by-topic matrix. Untick to show only the "
             "largest topics for a more readable grid.",
    )
    if show_all:
        _top_a = list(sim_df.index)
        _top_b = list(sim_df.columns)
    else:
        max_per_cond = st.slider(
            "Max topics per condition to display", 3, 50, 10, 1,
            key=f"heat_topn{key_suffix}",
            help="Shows the N largest topics (by number of sentences) from each "
                 "condition, e.g. 10 gives a 10×10 grid.",
        )
        # Rank topics by size (sentence count) and keep the top-N present in the matrix.
        _top_a = [n for n in sorted(topics_a, key=lambda k: len(topics_a[k]), reverse=True)
                  if n in sim_df.index][:max_per_cond]
        _top_b = [n for n in sorted(topics_b, key=lambda k: len(topics_b[k]), reverse=True)
                  if n in sim_df.columns][:max_per_cond]
    heat_df = sim_df.loc[_top_a, _top_b]
    if len(_top_a) < len(sim_df.index) or len(_top_b) < len(sim_df.columns):
        st.caption(
            f"Showing the {len(_top_a)}×{len(_top_b)} largest topics "
            f"(of {len(sim_df.index)}×{len(sim_df.columns)} total). "
            "Tick 'Show ALL topics' or raise the slider to see more."
        )

    # Centered cosine: use a diverging scale anchored at 0 so unrelated topics
    # (near 0 / negative) read as neutral and real matches stand out in green.
    _vmax = max(0.3, float(np.nanmax(heat_df.values))) if heat_df.size else 1.0
    _vmin = min(-0.3, float(np.nanmin(heat_df.values))) if heat_df.size else -1.0
    _ncols = len(heat_df.columns)
    _nrows = len(heat_df)
    # Annotate values only when the grid is small enough to stay legible.
    _annot = (_nrows * _ncols) <= 400
    fig_heat, ax_heat = plt.subplots(figsize=(max(7, _ncols * 1.4), max(5, _nrows * 1.0)))
    fig_heat.patch.set_facecolor("white")
    sns.heatmap(
        heat_df, annot=_annot, fmt=".2f", cmap="RdYlGn",
        center=0.0, vmin=_vmin, vmax=_vmax, linewidths=0.6, linecolor="#f0f0f0",
        annot_kws={"size": 9, "weight": "bold"},
        cbar_kws={"shrink": 0.75, "pad": 0.02}, ax=ax_heat,
    )
    ax_heat.collections[0].colorbar.set_label("Centered cosine similarity", fontsize=9, labelpad=8)
    ax_heat.collections[0].colorbar.ax.tick_params(labelsize=8)
    ax_heat.set_title(f"Semantic similarity: {name_a}  vs  {name_b}",
                      fontsize=13, pad=14, color="#222222")
    ax_heat.set_xlabel(name_b, fontsize=11, labelpad=10, color="#444444")
    ax_heat.set_ylabel(name_a, fontsize=11, labelpad=10, color="#444444")
    plt.xticks(rotation=35, ha="right", fontsize=8.5)
    plt.yticks(rotation=0, fontsize=8.5)
    plt.tight_layout(pad=1.5)
    st.pyplot(fig_heat)
    buf_heat = BytesIO()
    fig_heat.savefig(buf_heat, format="png", dpi=300, bbox_inches="tight")
    st.download_button("Download heatmap as PNG", data=buf_heat.getvalue(),
                       file_name=f"similarity_heatmap_{name_a}_{name_b}.png", mime="image/png")
    plt.close(fig_heat)

    # ── Greedy matching ───────────────────────────────────────────
    st.subheader(f"Matched topics (threshold ≥ {threshold:.2f})")
    matrix_copy = sim_df.copy()
    shared = []
    unique_a = list(sim_df.index)
    unique_b = list(sim_df.columns)
    while True:
        max_score = float(matrix_copy.max().max())
        if max_score < threshold:
            break
        idx_a, idx_b = matrix_copy.stack().idxmax()
        shared.append((idx_a, idx_b, max_score))
        if idx_a in unique_a: unique_a.remove(idx_a)
        if idx_b in unique_b: unique_b.remove(idx_b)
        matrix_copy.loc[idx_a, :] = -1e9
        matrix_copy.loc[:, idx_b] = -1e9

    if shared:
        shared_df = pd.DataFrame(shared, columns=[name_a, name_b, "cosine_similarity"])
        shared_df = shared_df.sort_values("cosine_similarity", ascending=False).reset_index(drop=True)
        st.success(f"Found **{len(shared)} shared topic pairs** above threshold {threshold:.2f}")
        st.dataframe(shared_df, use_container_width=True)
    else:
        st.warning(f"No topic pairs found above threshold {threshold:.2f}. Try lowering the threshold.")

    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown(f"**Unique to {name_a}** ({len(unique_a)} topics)")
        for t in unique_a: st.markdown(f"- {t}")
    with mc2:
        st.markdown(f"**Unique to {name_b}** ({len(unique_b)} topics)")
        for t in unique_b: st.markdown(f"- {t}")

    # ── Frequency comparison ──────────────────────────────────────
    st.subheader("Frequency comparison")
    counts_a = {name: len(sents) for name, sents in topics_a.items()}
    counts_b = {name: len(sents) for name, sents in topics_b.items()}
    rows_ct = []
    for a_name, b_name, score in sorted(shared, key=lambda x: -x[2]):
        rows_ct.append({"theme": f"{a_name} / {b_name}", name_a: counts_a.get(a_name, 0),
                        name_b: counts_b.get(b_name, 0), "type": "paired", "cosine": round(score, 3)})
    for t in unique_a:
        rows_ct.append({"theme": t, name_a: counts_a.get(t, 0), name_b: 0,
                        "type": f"{name_a}-specific", "cosine": float("nan")})
    for t in unique_b:
        rows_ct.append({"theme": t, name_a: 0, name_b: counts_b.get(t, 0),
                        "type": f"{name_b}-specific", "cosine": float("nan")})
    ct_df = pd.DataFrame(rows_ct)
    ct_df["total"] = ct_df[name_a] + ct_df[name_b]
    ct_df = ct_df.sort_values("total", ascending=False).drop(columns="total").reset_index(drop=True)
    st.dataframe(ct_df, use_container_width=True)

    # Chi-squared test
    counts_for_chi2 = ct_df[[name_a, name_b]].dropna()
    if counts_for_chi2.shape[0] >= 2:
        try:
            chi2_stat, p_val, dof, _ = chi2_contingency(counts_for_chi2)
            chi_col1, chi_col2, chi_col3 = st.columns(3)
            chi_col1.metric("χ² statistic", f"{chi2_stat:.2f}")
            chi_col2.metric("Degrees of freedom", dof)
            chi_col3.metric("p-value", f"{p_val:.4f}")
            if p_val < 0.05:
                st.success(f"p = {p_val:.4f} < 0.05 — statistically significant difference "
                           f"in topic distribution between {name_a} and {name_b}.")
            else:
                st.info(f"p = {p_val:.4f} ≥ 0.05 — no statistically significant difference detected.")
        except ValueError as e:
            st.warning(f"Chi-squared test could not be computed: {e}")

    # Bar chart
    st.subheader("Comparative frequency chart")
    if not ct_df.empty:
        pct_df = ct_df.copy()
        total_a = pct_df[name_a].sum()
        total_b = pct_df[name_b].sum()
        pct_df[f"{name_a} %"] = (pct_df[name_a] / total_a * 100).round(1) if total_a else 0
        pct_df[f"{name_b} %"] = (pct_df[name_b] / total_b * 100).round(1) if total_b else 0
        sorted_themes = pct_df.sort_values(f"{name_a} %", ascending=False)["theme"].tolist()
        formatted_labels = [t.replace(" / ", "\n") for t in sorted_themes]
        plot_melt = pct_df.melt(id_vars="theme",
                                value_vars=[f"{name_a} %", f"{name_b} %"],
                                var_name="Condition", value_name="Percentage")
        plot_melt["Condition"] = plot_melt["Condition"].str.replace(" %", "")
        n_t = len(sorted_themes)
        fig_bar, ax_bar = plt.subplots(figsize=(9, max(4.5, n_t * 0.50 + 1.4)))
        fig_bar.patch.set_facecolor("white")
        ax_bar.set_facecolor("#f8f9fa")
        sns.barplot(data=plot_melt, y="theme", x="Percentage", hue="Condition",
                    palette={name_a: "#5B7BE8", name_b: "#FF8C42"},
                    order=sorted_themes, ax=ax_bar)
        ax_bar.set_yticklabels(formatted_labels, fontsize=8.5, color="#333333")
        ax_bar.set_title(f"Experiential theme frequencies: {name_a}  vs  {name_b}",
                         fontsize=12, pad=14, color="#222222")
        ax_bar.set_xlabel("% of sentences within condition", fontsize=10, color="#555555")
        ax_bar.set_ylabel("")
        ax_bar.tick_params(axis="x", labelsize=8, colors="#777777")
        for spine in ("top", "right"): ax_bar.spines[spine].set_visible(False)
        ax_bar.spines["left"].set_color("#cccccc")
        ax_bar.spines["bottom"].set_color("#cccccc")
        ax_bar.grid(axis="x", color="#e0e0e0", linewidth=0.8, zorder=0)
        ax_bar.grid(axis="y", linestyle="", alpha=0)
        ax_bar.set_axisbelow(True)
        ax_bar.set_xlim(0, plot_melt["Percentage"].max() * 1.18)
        for container in ax_bar.containers:
            lbs = [f"{v:.1f}%" if v > 0 else "" for v in container.datavalues]
            ax_bar.bar_label(container, labels=lbs, label_type="edge", fontsize=7.5,
                             padding=3, color="#444444")
        legend = ax_bar.legend(title="Condition", frameon=True, facecolor="white", edgecolor="#dddddd", fontsize=9)
        legend.get_title().set_fontweight("bold")
        legend.get_title().set_fontsize(9)
        plt.tight_layout(pad=1.5)
        st.pyplot(fig_bar)
        buf_bar = BytesIO()
        fig_bar.savefig(buf_bar, format="png", dpi=200, bbox_inches="tight")
        plt.close(fig_bar)
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button("Download bar chart (PNG)", data=buf_bar.getvalue(),
                               file_name=f"freq_comparison_{name_a}_{name_b}.png",
                               mime="image/png", use_container_width=True)
        with dl2:
            st.download_button("Download contingency table (CSV)",
                               data=ct_df.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"contingency_{name_a}_{name_b}.csv",
                               mime="text/csv", use_container_width=True)


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
            min_words=min_words,
        )

    st.divider()
    st.markdown("**— or —**")

    _showing_tools = st.session_state.get("show_precomputed_tools", False)
    _toggle_label = (
        "Hide analysis tools"
        if _showing_tools
        else "Skip pipeline — I'll upload precomputed files from a previous run"
    )
    if st.button(_toggle_label, key="skip_to_tools_btn"):
        st.session_state["show_precomputed_tools"] = not _showing_tools
        st.rerun()

    if st.session_state.get("show_precomputed_tools"):
        st.info(
            "Upload files you exported from a previous pipeline run. "
            "The embedding model selected in the sidebar will be used for any re-embedding steps."
        )

        sa_zs_tab, sa_cond_tab = st.tabs(["Zero-Shot Classification", "Condition Comparison"])

        with sa_zs_tab:
            st.subheader("Zero-Shot Topic Classification")
            st.caption(
                "Upload the `.npy` embeddings file and `.json` docs file saved by the pipeline "
                "to classify documents into categories without re-running the full pipeline."
            )

            sa_npy = st.file_uploader("Embeddings file (.npy)", type=["npy"], key="sa_zs_npy")
            sa_docs_json = st.file_uploader(
                "Documents file (.json) — list of strings", type=["json"], key="sa_zs_docs"
            )

            # Auto-detect the embedding model from the filename; let the user override.
            _auto_model = _detect_model_from_filename(sa_npy.name if sa_npy else "")
            _default_idx = list(ALL_EMBEDDING_MODELS).index(_auto_model) if _auto_model else 0
            sa_embedding_model = st.selectbox(
                "Embedding model (must match the model used to generate the .npy file)",
                ALL_EMBEDDING_MODELS,
                index=_default_idx,
                key="sa_zs_emb_model",
                help="Auto-detected from the filename when possible. Change if detection is wrong.",
            )
            if _auto_model:
                st.caption(f"Auto-detected from filename: `{_auto_model}`")
            else:
                st.warning("Could not auto-detect the embedding model from the filename. Please select the correct model above.")

            sa_zs_categories_raw = st.text_area(
                "Categories — one per line",
                value=_ZS_DEFAULT_CATEGORIES,
                height=220,
                key="sa_zs_categories",
            )
            sa_zs_min_sim = st.slider(
                "Minimum similarity threshold",
                min_value=0.1, max_value=0.9, value=0.5, step=0.05,
                key="sa_zs_thresh",
                help="Documents with max category similarity below this value are left as 'Unclassified'.",
            )

            sa_run_zs = st.button(
                "Run Zero-Shot Classification",
                type="primary",
                key="sa_zs_run_btn",
                disabled=(sa_npy is None or sa_docs_json is None),
            )
            if sa_npy is None or sa_docs_json is None:
                st.info("Upload both files above to enable classification.")

            if sa_run_zs and sa_npy and sa_docs_json:
                try:
                    sa_embeddings = np.load(sa_npy)
                    sa_docs_list = json.load(sa_docs_json)
                    sa_docs_list = [str(d) if (d is not None and d != "") else "[empty]" for d in sa_docs_list]
                except Exception as e:
                    st.error(f"Could not load files: {e}")
                    st.stop()
                sa_categories = [c.strip() for c in sa_zs_categories_raw.strip().splitlines() if c.strip()]
                if not sa_categories:
                    st.error("Please enter at least one category.")
                    st.stop()
                with st.spinner("Running zero-shot classification…"):
                    sa_zs_topics, sa_zs_topic_info, _ = run_zeroshot(
                        sa_docs_list, sa_embeddings, tuple(sa_categories),
                        sa_zs_min_sim, sa_embedding_model,
                        dataset_key=f"standalone_{len(sa_docs_list)}",
                    )
                st.session_state["sa_zs_results"] = (sa_zs_topics, sa_zs_topic_info, sa_categories, sa_docs_list)
                st.session_state["sa_zs_min_sim"] = sa_zs_min_sim

            if "sa_zs_results" in st.session_state:
                _sa_zs_state = st.session_state["sa_zs_results"]
                if len(_sa_zs_state) == 4:
                    sa_zs_topics, sa_zs_topic_info, sa_categories, sa_docs_list = _sa_zs_state
                else:
                    sa_zs_topics, sa_zs_topic_info, sa_categories = _sa_zs_state
                    sa_docs_list = []
                if not sa_docs_list or len(sa_docs_list) != len(sa_zs_topics):
                    st.warning("Session data mismatch — please re-run the classification.")
                    st.stop()

                sa_total_zs = len(sa_zs_topics)
                sa_classified_zs = sum(1 for t in sa_zs_topics if t != -1)
                sa_unclassified_zs = sa_total_zs - sa_classified_zs

                sa_zm1, sa_zm2, sa_zm3 = st.columns(3)
                sa_zm1.metric("Total documents", f"{sa_total_zs:,}")
                sa_zm2.metric("Classified", f"{sa_classified_zs:,} ({100*sa_classified_zs/sa_total_zs:.1f}%)")
                sa_zm3.metric("Unclassified", f"{sa_unclassified_zs:,} ({100*sa_unclassified_zs/sa_total_zs:.1f}%)")

                sa_zs_name_map = sa_zs_topic_info.set_index("Topic")["Name"].to_dict()
                sa_zs_df = pd.DataFrame({"sentence": sa_docs_list, "topic_id": sa_zs_topics})
                sa_zs_df["category"] = sa_zs_df["topic_id"].map(sa_zs_name_map).fillna("Unclassified")

                sa_zs_plot_df = (
                    sa_zs_topic_info[sa_zs_topic_info["Topic"] != -1]
                    .sort_values("Count", ascending=True)
                    .reset_index(drop=True)
                )

                if not sa_zs_plot_df.empty:
                    st.subheader("Distribution across categories")
                    import matplotlib.cm as _cm
                    import matplotlib.colors as _mcolors
                    _norm = _mcolors.Normalize(vmin=sa_zs_plot_df["Count"].min(), vmax=sa_zs_plot_df["Count"].max())
                    _cmap = _cm.get_cmap("Purples")
                    _bar_colors = [_cmap(0.35 + 0.55 * _norm(v)) for v in sa_zs_plot_df["Count"]]
                    fig_sa_zs, ax_sa_zs = plt.subplots(figsize=(10, max(4, len(sa_zs_plot_df) * 0.62)))
                    fig_sa_zs.patch.set_facecolor("white")
                    ax_sa_zs.set_facecolor("#f8f9fa")
                    bars = ax_sa_zs.barh(sa_zs_plot_df["Name"], sa_zs_plot_df["Count"], color=_bar_colors, edgecolor="white", linewidth=0.8, height=0.65)
                    _max_count = sa_zs_plot_df["Count"].max()
                    for bar in bars:
                        w = bar.get_width()
                        ax_sa_zs.text(w + _max_count * 0.015, bar.get_y() + bar.get_height() / 2, str(int(w)), va="center", ha="left", fontsize=9, color="#333333", fontweight="bold")
                    for spine in ("top", "right"):
                        ax_sa_zs.spines[spine].set_visible(False)
                    ax_sa_zs.set_xlabel("Number of sentences", fontsize=10, color="#555555")
                    ax_sa_zs.invert_yaxis()
                    ax_sa_zs.grid(axis="x", color="#e0e0e0", linewidth=0.8, zorder=0)
                    ax_sa_zs.set_axisbelow(True)
                    ax_sa_zs.set_title(f"Zero-Shot Classification  ·  {sa_classified_zs:,} / {sa_total_zs:,} sentences classified", fontsize=11, color="#333333", pad=12)
                    plt.tight_layout(pad=1.5)
                    st.pyplot(fig_sa_zs)
                    _buf_sa_zs = BytesIO()
                    fig_sa_zs.savefig(_buf_sa_zs, format="png", dpi=200, bbox_inches="tight")
                    st.download_button("Download bar chart (PNG)", data=_buf_sa_zs.getvalue(), file_name="zeroshot_barchart.png", mime="image/png")
                    plt.close(fig_sa_zs)

                st.subheader("Sentences per category")
                for _, row_sa_zs in sa_zs_plot_df.sort_values("Count", ascending=False).iterrows():
                    cat_name = row_sa_zs["Name"]
                    count_sa_zs = row_sa_zs["Count"]
                    with st.expander(f"{cat_name}  ({count_sa_zs} sentences)"):
                        cat_sentences = sa_zs_df[sa_zs_df["category"] == cat_name]["sentence"].reset_index(drop=True)
                        n_show = st.slider("Sentences to show", 5, min(100, len(cat_sentences)), 10, key=f"sa_zs_show_{cat_name}")
                        st.dataframe(cat_sentences.head(n_show).to_frame("sentence"), use_container_width=True)

                with st.expander(f"Unclassified  ({sa_unclassified_zs} sentences)"):
                    unclass = sa_zs_df[sa_zs_df["category"] == "Unclassified"]["sentence"].reset_index(drop=True)
                    st.dataframe(unclass.head(50).to_frame("sentence"), use_container_width=True)

                st.download_button(
                    "Download full classification results (CSV)",
                    data=sa_zs_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="zeroshot_standalone.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

                # ── 2D Scatter plot ───────────────────────────────────────────
                _reduced_path = CACHE_DIR / "reduced_2d.npy"
                _topics_path  = CACHE_DIR / "topics.json"
                _llm_label_files = sorted(CACHE_DIR.glob("llm_labels_*.json"))

                if _reduced_path.exists():
                    _reduced_2d = np.load(_reduced_path)
                    if _reduced_2d.shape[0] == len(sa_zs_topics):
                        st.subheader("2D Map — coloured by zero-shot category")
                        import plotly.graph_objects as go
                        import plotly.express as px
                        _cat_list = [sa_zs_name_map.get(t, "Unclassified") for t in sa_zs_topics]
                        _unique_cats = ["Unclassified"] + [c for c in sorted(set(_cat_list)) if c != "Unclassified"]
                        _palette = px.colors.qualitative.Plotly + px.colors.qualitative.Set2 + px.colors.qualitative.Pastel
                        _scatter_fig = go.Figure()
                        for _i, _cat in enumerate(_unique_cats):
                            _mask = [j for j, c in enumerate(_cat_list) if c == _cat]
                            _color  = "#CCCCCC" if _cat == "Unclassified" else _palette[(_i - 1) % len(_palette)]
                            _opacity = 0.25 if _cat == "Unclassified" else 0.75
                            _size    = 3 if _cat == "Unclassified" else 5
                            _scatter_fig.add_trace(go.Scattergl(
                                x=_reduced_2d[_mask, 0], y=_reduced_2d[_mask, 1],
                                mode="markers", name=_cat,
                                marker=dict(color=_color, size=_size, opacity=_opacity),
                                text=[sa_docs_list[j][:200] for j in _mask],
                                hoverinfo="text+name",
                            ))
                        _scatter_fig.update_layout(
                            height=700, template="simple_white",
                            xaxis=dict(visible=False), yaxis=dict(visible=False),
                            title=dict(text="<b>Documents coloured by zero-shot category</b>", x=0.5, xanchor="center"),
                            legend=dict(title="Category", bgcolor="rgba(255,255,255,0.9)", bordercolor="#dddddd", borderwidth=1, font=dict(size=10)),
                            margin=dict(l=10, r=10, t=60, b=10),
                        )
                        st.plotly_chart(_scatter_fig, use_container_width=True)

                # ── BERTopic cross-reference ──────────────────────────────────
                # Search all dataset caches for a topics.json matching the uploaded doc count
                _n_docs = len(sa_zs_topics)
                _matched_cache = None
                for _candidate in (Path(__file__).parent / "data").glob("*/preprocessed/cache/topics.json"):
                    try:
                        with open(_candidate) as _f:
                            _candidate_topics = json.load(_f)
                        if len(_candidate_topics) == _n_docs:
                            _matched_cache = _candidate.parent
                            break
                    except Exception:
                        continue

                if _matched_cache is None:
                    st.info("BERTopic cross-reference not available — no matching `topics.json` found. Run the full pipeline on this dataset first.")
                else:
                    _topics_path  = _matched_cache / "topics.json"
                    _llm_label_files = sorted(_matched_cache.glob("llm_labels_*.json"))
                    with open(_topics_path) as _f:
                        _bt_topics = json.load(_f)
                    _llm_labels_xref = {}
                    if _llm_label_files:
                        with open(_llm_label_files[-1]) as _f:
                            _llm_labels_xref = {int(k): v for k, v in json.load(_f).items()}
                    else:
                        st.caption("No LLM labels found — showing topic numbers. Run the full pipeline with LLM labelling for better labels.")
                    if len(_bt_topics) == _n_docs:
                        _sa_zs_threshold = st.session_state.get("sa_zs_min_sim", "?")
                        st.subheader("BERTopic × Zero-Shot cross-reference")
                        st.caption(f"Each row shows how a BERTopic topic distributes across your zero-shot categories (rows sum to 100%). Similarity threshold: {_sa_zs_threshold}. '% unclassified' = sentences below threshold not assigned to any category.")
                        _xref_df = pd.DataFrame({
                            "category": [sa_zs_name_map.get(t, "Unclassified") for t in sa_zs_topics],
                            "bt_topic": _bt_topics,
                        })
                        _xref_df["bt_label"] = _xref_df["bt_topic"].map(
                            lambda t: _llm_labels_xref.get(t, f"Topic {t}") if t != -1 else "Outlier"
                        )

                        # Heatmap: % of each BERTopic topic that falls into each zero-shot category
                        # Replace zero-shot outlier topic name with "Unclassified"
                        _zs_outlier_name = sa_zs_name_map.get(-1)
                        if _zs_outlier_name:
                            _xref_df.loc[_xref_df["category"] == _zs_outlier_name, "category"] = "Unclassified"

                        _crosstab_full = pd.crosstab(
                            _xref_df["bt_label"], _xref_df["category"], normalize="index"
                        ) * 100
                        # Keep % unclassified as a separate column, remove the outlier name column
                        _pct_unclassified = _crosstab_full.get("Unclassified", pd.Series(0, index=_crosstab_full.index))
                        _crosstab = _crosstab_full.drop(columns="Unclassified", errors="ignore")
                        _crosstab["% unclassified"] = _pct_unclassified.values

                        import plotly.graph_objects as _go_xref
                        _heat_fig = _go_xref.Figure(data=_go_xref.Heatmap(
                            z=_crosstab.values,
                            x=list(_crosstab.columns),
                            y=list(_crosstab.index),
                            colorscale= "dense",#"bupu",
                            text=[[f"{v:.1f}%" for v in row] for row in _crosstab.values],
                            texttemplate="%{text}",
                            hovertemplate="BERTopic: %{y}<br>Category: %{x}<br>%{text}<extra></extra>",
                            colorbar=dict(title="% of topic"),
                        ))
                        _heat_fig.update_layout(
                            height=max(400, len(_crosstab) * 28),
                            xaxis=dict(title="Zero-shot category", tickangle=-30, side="bottom"),
                            yaxis=dict(title="BERTopic topic", autorange="reversed"),
                            margin=dict(l=10, r=10, t=30, b=120),
                        )
                        st.plotly_chart(_heat_fig, use_container_width=True)
                        try:
                            _heat_img = _heat_fig.to_image(format="png", scale=2)
                            st.download_button("Download heatmap (PNG)", data=_heat_img, file_name="zeroshot_heatmap.png", mime="image/png")
                        except Exception:
                            st.caption("Install `kaleido` to enable heatmap download: `pip install kaleido`")

                        st.caption("Per zero-shot category: top BERTopic topics inside it")
                        for _, _row_xref in sa_zs_plot_df.sort_values("Count", ascending=False).iterrows():
                            _cat_xref = _row_xref["Name"]
                            with st.expander(f"{_cat_xref}"):
                                _top_bt = (
                                    _xref_df[_xref_df["category"] == _cat_xref]
                                    .groupby("bt_label").size()
                                    .sort_values(ascending=False).head(10)
                                    .reset_index()
                                )
                                _top_bt.columns = ["BERTopic Label", "Count"]
                                st.dataframe(_top_bt, use_container_width=True, hide_index=True)

        with sa_cond_tab:
            st.subheader("Condition Comparison — Semantic Similarity")
            with st.expander("How to use", expanded=True):
                st.markdown(
                    f"""
Upload the `topics_summary_*.csv` files saved from previous pipeline runs (one per condition).

**What the comparison does:**
- Sentences in each topic CSV are re-embedded using `{selected_embedding_model}` (set in the sidebar)
- Each topic becomes a mean vector; cosine similarity is computed pairwise across conditions
- Results: heatmap, matched pairs, unique topics, contingency table, chi-squared test, frequency chart
                    """
                )
            _condition_comparison_ui(selected_embedding_model, key_suffix="_sa")

else:
    # Load cached data
    docs, embeddings = load_precomputed_data(DOCS_FILE, EMBEDDINGS_FILE)

    embeddings = np.asarray(embeddings)
    if embeddings.dtype == object or embeddings.ndim != 2:
        try:
            embeddings = np.vstack(embeddings).astype(np.float32)
        except Exception:
            st.error(
                "Cached embeddings are invalid. Please click on \"Clear cached files for this configuration\" in the sidebar and then regenerate them for this configuration."
            )
            st.stop()

    if subsample_perc < 100:
        n = int(len(docs) * (subsample_perc / 100))
        idx = np.random.default_rng(random_seed).choice(len(docs), size=n, replace=False)
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

    c1, c2, c3 = st.columns(3)
    c1.metric("Reports in CSV (cleaned)", n_reports)
    c2.metric(f"Units analysed ({unit})", n_units)


    stats = st.session_state.get("last_data_stats")
    if (
        stats is not None
        and stats.get("granularity") == unit
        and stats.get("min_words", 0) == int(min_words or 0)
    ):
        removed = stats["removed"]
        total_before = stats["total_before"]
        c3.metric("Units removed (< N words)", removed)
        st.caption(
            f"Min-words filter N = {int(min_words or 0)}. "
            f"Started with {total_before} {unit}, kept {stats['total_after']}."
        )
    else:
        c3.metric("Units removed (< N words)", "–")
        st.caption(
            "Change 'Remove units shorter than N words' and click "
            "'Prepare Data for This Configuration' to update removal stats."
        )

    
    with st.expander("Preview preprocessed text (first 10 units)"):
        preview_df = pd.DataFrame({"text": docs[:10]})
        st.dataframe(preview_df)

        removed = st.session_state.get("last_removed_units", [])
        with st.expander(f"Preview removed units ({len(removed)})"):
            if not removed:
                st.caption("No units removed for the current min-words setting.")
            else:
                n_show = st.slider("How many removed units to show", 10, min(500, len(removed)), 50)
                df_removed = pd.DataFrame({
                    "n_words": [len(str(x).split()) for x in removed[:n_show]],
                    "text": removed[:n_show],
                })
                st.dataframe(df_removed, use_container_width=True)
        
                st.download_button(
                    "Download removed units (txt)",
                    data="\n".join(map(str, removed)),
                    file_name="removed_units.txt",
                    mime="text/plain",
                )



    # --- Parameter controls ---
    st.sidebar.header("Model Parameters")

    with st.sidebar.expander("📖 Quick Parameter Guide"):
        st.markdown("""
        **Getting too many small/fragmented topics?**
        - Increase `min_cluster_size` (try 20-30)
        - Increase `n_neighbors` (try 25-30)
        
        **Getting too few broad topics?**
        - Decrease `min_cluster_size` (try 5-8)
        - Decrease `n_neighbors` (try 8-12)
        
        **Too many outliers?**
        - Decrease `min_samples` (try 2-3)
        - Decrease `min_cluster_size`
        
        **Topics not distinct enough?**
        - Decrease `n_neighbors` 
        - Increase `min_samples`
        - Try `min_dist = 0.0`
        
        **General advice:**
        - Start with defaults, run analysis, then adjust
        - Change ONE parameter at a time
        - `n_neighbors` and `min_cluster_size` have the biggest impact
        """)

    use_vectorizer = st.sidebar.checkbox("Use CountVectorizer", value=True,help="Enables bag-of-words representation for topic keyword extraction. Disable if you only want embedding-based clustering without keyword analysis.")

    with st.sidebar.expander("Vectorizer"):
        ng_min = st.slider(
            "Min N-gram", 1, 5, 1,
            help="Minimum word sequence length. 1 = single words ('visual'), 2 = pairs ('visual experience'). Lower values capture basic terms."
        )
        ng_max = st.slider(
            "Max N-gram", 1, 5, 2,
            help="Maximum word sequence length. Higher values capture longer phrases but increase noise. 2-3 is usually optimal."
        )
        min_df = st.slider(
            "Min Doc Freq", 1, 50, 1,
            help="Minimum number of documents a term must appear in. Higher values filter out rare terms, reducing noise but potentially losing unique descriptors."
        )
        

        st.write("Stopwords Configuration")
        use_english_stopwords = st.checkbox("Use standard English stopwords", value=True)
        
        custom_stopwords_input = st.text_area(
            "Custom Stopwords (comma-separated)",
            value="",
            help="Add context-specific words here to force the model to ignore them."
        )


    with st.sidebar.expander("UMAP"):
        st.caption("UMAP reduces high-dimensional embeddings to a lower-dimensional space for clustering.")
        um_n = st.slider(
            "n_neighbors", 2, 50, 15,
            help="How many neighbors to consider for each point. LOW (5-10): DEPENDS ON THE SIZE OF THE DATA, but: preserves fine local structure, more small clusters. HIGH (30-50): captures broader patterns, fewer larger clusters. This is often the most impactful parameter."
        )
        um_c = st.slider(
            "n_components", 2, 20, 5,
            help="Number of dimensions in the reduced space. Higher preserves more information but increases computation. 5-10 is typical for clustering; 2 is used for visualisation."
        )
        um_d = st.slider(
            "min_dist", 0.0, 1.0, 0.0,
            help="Minimum distance between points in reduced space. 0.0: points can clump tightly (better for clustering). Higher values spread points out (better for visualisation but worse for clustering)."
        )
    
    with st.sidebar.expander("HDBSCAN"):
        st.caption("HDBSCAN finds clusters of varying densities and identifies outliers.")
        hs = st.slider(
            "min_cluster_size", 5, 100, 10,
            help="Minimum points required to form a cluster. LOW (5-15): finds more, smaller clusters including niche topics (but can lead to overfitting). HIGH (30-100): only finds major themes, more outliers."
        )
        hm = st.slider(
            "min_samples", 2, 100, 10,
            help="How conservative clustering is. LOW (2-5): more inclusive, fewer outliers, but may merge distinct topics. HIGH (15+): stricter, more outliers, but clusters are more coherent. Should typically be ≤ min_cluster_size."
        )

    with st.sidebar.expander("BERTopic"):
        nr_topics = st.text_input(
            "nr_topics", 
            value="auto",
            help="Target number of topics. 'auto': let the algorithm decide. A number (e.g., '20'): merge similar topics down to this count. Use 'auto' first, then reduce if you have too many topics."
        )
        top_n_words = st.slider(
            "top_n_words", 5, 25, 10,
            help="Number of keywords per topic. More words give richer topic descriptions but may include less relevant terms. 10-15 is usually a good balance for interpretability."
        )

        final_stopwords = []
        if use_english_stopwords:
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
            final_stopwords = list(ENGLISH_STOP_WORDS)
        
        if custom_stopwords_input:
            custom_list = [w.strip().lower() for w in custom_stopwords_input.split(",") if w.strip()]
            final_stopwords.extend(custom_list)
            
        # Pass None if list is empty (default behavior), otherwise pass the list
        vectorizer_stopwords = final_stopwords if final_stopwords else None
        
        current_config = {
            "embedding_model": selected_embedding_model,
            "granularity": granularity_label,
            "min_words": int(min_words or 0),
            "subsample_percent": subsample_perc,
            "use_vectorizer": use_vectorizer,
            "vectorizer_params": {
                "ngram_range": (ng_min, ng_max),
                "min_df": min_df,
                "stop_words": vectorizer_stopwords, 
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
            "random_seed": random_seed,
        }

    run_button = st.sidebar.button("Run Analysis", type="primary")

    # =================================================================
    # 9. Visualisation & History Tabs
    # =================================================================
    main_tab, zeroshot_tab, condition_tab, history_tab, compare_tab = st.tabs(["Main Results", "Zero-Shot Classification", "Condition Comparison", "Run History", "Compare Runs"])

    st.markdown(
        """
        <style>
        /* Highlight the first tab (Main Results) in orange */
        div[data-testid="stTabs"] > div > div > div:first-child button[role="tab"] {
            color: #E8630A;
            font-weight: 700;
            border-bottom: 3px solid #E8630A;
        }
        div[data-testid="stTabs"] > div > div > div:first-child button[role="tab"]:hover {
            color: #E8630A;
            opacity: 0.8;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
                docs, embeddings, get_config_hash(current_config), DOCS_FILE
            )
        st.session_state.latest_results = (model, reduced, labels)
        st.session_state.latest_results_docs_file = DOCS_FILE

        # ==========================================================
        # Load Metadata to calculate Topic Diversity
        # ==========================================================
        if os.path.exists(METADATA_FILE):
            meta_df = pd.read_csv(METADATA_FILE)
            

            if subsample_perc < 100 and 'idx' in locals():
                meta_df = meta_df.iloc[idx].reset_index(drop=True)
            
            # Create a mapping dataframe (whoch topic belongs to which original report)
            # use 'model.topics_' which aligns 1:1 with 'docs' and 'meta_df'
            topic_sources = pd.DataFrame({
                "Topic": model.topics_,
                "Report_ID": meta_df["_source_row_idx"],
                "Sentence": docs
            })

            # Remove outliers (-1) for this specific analysis if desired
            topic_sources = topic_sources[topic_sources["Topic"] != -1]

            # Calculate Aggregated stats per Topic
            diversity_stats = topic_sources.groupby("Topic").agg(
                Total_Sentences=('Sentence', 'count'),
                Unique_Reports=('Report_ID', 'nunique')
            ).reset_index()

            # Calculate a "Repetition Score" 
            # 1.0 = Perfectly diverse (Every sentence comes from a different person)
            # Low = Repetitive (One person said many sentences in this topic)
            diversity_stats["Diversity_Ratio"] = diversity_stats["Unique_Reports"] / diversity_stats["Total_Sentences"]
            
            # Map Topic Names
            if "llm_names" in st.session_state:
                mapping = st.session_state.llm_names
            else:
                mapping = model.get_topic_info().set_index("Topic")["Name"].to_dict()
            
            diversity_stats["Name"] = diversity_stats["Topic"].map(mapping).fillna("Unlabelled")
            
            st.session_state.diversity_stats = diversity_stats


        # Store the exact docs used to fit this model (so export never mismatches)
        st.session_state.latest_docs = docs
        st.session_state.latest_csv_path = CSV_PATH

        run_id = make_run_id(current_config)
        dataset_title = ds_input.strip() or DATASET_DIR
        
        safe_labs = ["Unlabelled" if t == -1 else lab for t, lab in zip(model.topics_, labels)]
        
        meta = save_run_snapshot(
            run_id=run_id,
            tm=model,
            reduced=reduced,
            labs=safe_labs,
            dataset_title=dataset_title,
            csv_path=CSV_PATH,
            current_config=current_config,
        )



        st.session_state.latest_config_hash = get_config_hash(current_config) + "|" + DOCS_FILE
        st.session_state.latest_config = current_config


        entry = {
            "run_id": meta["run_id"],
            "timestamp": meta["timestamp"],
            "config": current_config,
            "num_topics": meta["n_topics"],
            "n_units": meta["n_units"],
            "n_outliers": meta["outlier_count"],
            "outlier_pct": meta["outlier_pct"],  # float
            "artifacts": meta["artifacts"],
            "llm_labels": [
                name
                for name in model.get_topic_info().Name.values
                if ("Unlabelled" not in name and "outlier" not in name)
            ],
        }
        dataset_title = ds_input.strip() or DATASET_DIR
        entry["dataset_title"] = dataset_title
        entry["csv_path"] = CSV_PATH

        st.session_state.history.insert(0, entry)
        save_history(st.session_state.history)
        st.rerun()

    # --- MAIN TAB ---
    with main_tab:
        _results_stale = (
            "latest_results" in st.session_state
            and st.session_state.get("latest_results_docs_file") != DOCS_FILE
        )
        if _results_stale:
            st.warning(
                "The dataset has changed since the last analysis. "
                "Click **Run Analysis** to analyse the current dataset."
            )
        if "latest_results" in st.session_state and not _results_stale:
            tm, reduced, labs = st.session_state.latest_results



            st.subheader("LLM topic labelling (via Hugging Face API)")


            # -------------------------------
            # START Topic modelling stats (pre-LLM)
            # -------------------------------
            info = tm.get_topic_info()
            
            total_units = int(info["Count"].sum()) if "Count" in info.columns else len(getattr(tm, "topics_", []))
            
            # Topics (excluding outliers)
            n_topics = int((info["Topic"] != -1).sum()) if "Topic" in info.columns else len(set(tm.topics_)) - (1 if -1 in tm.topics_ else 0)
            
            # Outliers
            if "Topic" in info.columns and "Count" in info.columns and (-1 in info["Topic"].values):
                outlier_count = int(info.loc[info["Topic"] == -1, "Count"].iloc[0])
            else:
                outlier_count = int(sum(1 for t in getattr(tm, "topics_", []) if t == -1))
            
            outlier_pct = (100.0 * outlier_count / total_units) if total_units else 0.0
            
            st.markdown("#### Topic modelling summary")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Topics found", n_topics)
            c2.metric("Outliers (-1)", outlier_count)
            c3.metric("Outlier rate", f"{outlier_pct:.1f}%")
            c4.metric("Units clustered", total_units)


            with st.expander("Model Quality Metrics (Coherence & Embeddings)"):
                st.caption(
                    "These metrics assess topic quality. **Topic Coherence (C_v)** measures human interpretability "
                    "(how often top words actually appear together in the text), while **Embedding Coherence** "
                    "measures semantic tightness (how close the words are in the vector space)."
                )
                
                if "quality_metrics" not in st.session_state or st.session_state.quality_metrics_hash != get_config_hash(current_config):
                    with st.spinner("Calculating coherence metrics..."):
                        # prepare Data for Gensim (C_v)
                        # IMPORTANT: tokenise the docs the same way BERTopic's CountVectorizer
                        # does (lowercase + the default token pattern: word chars, 2+ length),
                        # otherwise the gensim dictionary keeps original case/punctuation
                        # (e.g. "Death,") and never matches the cleaned topic words
                        # (e.g. "death"), which makes CoherenceModel raise
                        # "unable to interpret topic as either a list of tokens or a list of ids".
                        _token_pattern = re.compile(r"(?u)\b\w\w+\b")
                        tokenized_docs = [_token_pattern.findall(d.lower()) for d in docs]
                        dictionary = Dictionary(tokenized_docs)
                        _vocab = dictionary.token2id

                        # Get top 10 words for every active topic (excluding outliers)
                        unique_topics = [t for t in set(tm.topics_) if t != -1]
                        topics_top_words = []
                        for t in unique_topics:
                            topic_words = tm.get_topic(t)
                            # tm.get_topic() can return False or empty for some topics
                            if topic_words and topic_words is not False:
                                # split n-grams into their component tokens and keep only
                                # tokens that are actually present in the dictionary, so a
                                # single OOV / multi-word phrase can't crash the whole metric
                                words = []
                                for word, _ in topic_words[:10]:
                                    for tok in str(word).lower().split():
                                        if tok in _vocab and tok not in words:
                                            words.append(tok)
                                # C_v needs at least 2 words to compute co-occurrence
                                if len(words) >= 2:
                                    topics_top_words.append(words)

                        # calculate C_v
                        if topics_top_words and len(topics_top_words) > 0:
                            cm = CoherenceModel(
                                topics=topics_top_words,
                                texts=tokenized_docs,
                                dictionary=dictionary,
                                coherence='c_v',
                                processes=1
                            )
                            c_v_score = cm.get_coherence()
                        else:
                            c_v_score = 0.0

                        # calculate Embedding Coherence (Proxy)
                        # average cosine similarity of top 10 words in embedding space
                        emb_coh_score = 0.0

                        active_embedding_model = load_embedding_model(selected_embedding_model)
                        if topics_top_words:
                            total_sim = 0
                            valid_topics = 0
                            for words in topics_top_words:
                                if len(words) < 2: continue
                                
                                word_embs = active_embedding_model.encode(words)
                                
                                sim_matrix = np.inner(word_embs, word_embs)
                                tri_u = sim_matrix[np.triu_indices(len(words), k=1)]
                                
                                if len(tri_u) > 0:
                                    total_sim += np.mean(tri_u)
                                    valid_topics += 1
                            
                            if valid_topics > 0:
                                emb_coh_score = total_sim / valid_topics
                        
                        st.session_state.quality_metrics = (c_v_score, emb_coh_score)
                        st.session_state.quality_metrics_hash = get_config_hash(current_config)
                
                # Retrieve from cache
                c_v, emb_coh = st.session_state.quality_metrics
                
                qc1, qc2 = st.columns(2)
                qc1.metric(
                    "Topic Coherence (C_v)", 
                    f"{c_v:.3f}", 
                    help="Measures how often the top words in a topic appear together in the original text. Good values: 0.5 - 0.7."
                )
                qc2.metric(
                    "Embedding Coherence", 
                    f"{emb_coh:.3f}", 
                    help="Measures how mathematically close the top words are in the vector space. Higher means tighter semantic clusters."
                )
            
            with st.expander("Show topic-size overview"):
                # Show biggest topics first (excluding outliers)
                if {"Topic", "Count", "Name"}.issubset(set(info.columns)):
                    top_sizes = (
                        info[info["Topic"] != -1][["Topic", "Count", "Name"]]
                        .sort_values("Count", ascending=False)
                        .head(15)
                        .reset_index(drop=True)
                    )
                    st.dataframe(top_sizes, use_container_width=True)
                else:
                    st.caption("Topic-size overview unavailable (missing columns in topic info).")



            model_id = st.text_input(
                "HF model id for labelling",
                value="meta-llama/Meta-Llama-3-8B-Instruct",
                help="""
                **Must be a model supported by the Hugging Face Serverless Inference API.**
                
                **Recommended Models:**
                * `meta-llama/Meta-Llama-3-8B-Instruct` (Free tier friendly, fast)
                * `meta-llama/Llama-3.1-8B-Instruct` (Newer, smarter, but may require HF PRO)
                * `mistralai/Mistral-7B-Instruct-v0.3` (Strict instruction following)

                [Check supported models](https://huggingface.co/models?pipeline_tag=text-generation&inference=warm&sort=trending)
                """
            )

            with st.expander("Show LLM configuration and prompts"):
                st.markdown(f"**HF model id (requested):** `{model_id}`")
            
                requested_last = st.session_state.get("hf_last_model_param")
                provider_model = st.session_state.get("hf_last_provider_model")
            
                if requested_last:
                    st.markdown(f"**Last run – requested model id:** `{requested_last}`")
                if provider_model:
                    st.markdown(f"**Last run – provider model (returned):** `{provider_model}`")
                else:
                    st.caption("Run LLM labelling once to see the provider-returned model id.")
            
                st.session_state.setdefault("llm_system_prompt", SYSTEM_PROMPT)
                st.session_state.setdefault("llm_user_template", USER_TEMPLATE)

                st.markdown("**System prompt** (editable):")
                st.text_area("System prompt", key="llm_system_prompt", height=200,
                             label_visibility="collapsed")

                st.markdown(
                    "**User prompt template** (editable — must keep the "
                    "`{documents}` and `{keywords}` placeholders):"
                )
                st.text_area("User prompt template", key="llm_user_template", height=240,
                             label_visibility="collapsed")

                if ("{documents}" not in st.session_state["llm_user_template"]
                        or "{keywords}" not in st.session_state["llm_user_template"]):
                    st.warning(
                        "The user template must contain both `{documents}` and `{keywords}` "
                        "or label generation will fail."
                    )
                if st.button("Reset prompts to default", key="llm_reset_prompts"):
                    st.session_state.pop("llm_system_prompt", None)
                    st.session_state.pop("llm_user_template", None)
                    st.rerun()

                example_prompt = st.session_state.get("hf_last_example_prompt")
                if example_prompt:
                    st.markdown("**Example full prompt for one topic (last run):**")
                    st.code(example_prompt, language="markdown")
                else:
                    st.caption("No example prompt stored yet – run LLM labelling to populate this.")
            
            cA, cB, cC = st.columns([1, 1, 2])

            with cA:
                max_topics = st.slider("Max topics", 5, 120, 40, 5)

            with cB:
                max_docs_per_topic = st.slider(
                    "Docs per topic",
                    min_value=3,
                    max_value=20,
                    value=5,
                    step=1,
                    help="How many representative sentences per topic to show the LLM. Try keeping low value to not spend all tokens",
                    key="llm_docs_per_topic",
                )
                force = st.checkbox(
                    "Force regenerate",
                    value=False,
                    key="llm_force_regenerate",  
                )
            
            
            if cC.button("Generate LLM labels (API)", use_container_width=True):
                try:
                    cfg_hash = st.session_state.get("latest_config_hash", "nohash")
                    llm_names = generate_labels_via_chat_completion(
                        topic_model=tm,
                        docs=docs,
                        config_hash=cfg_hash,
                        model_id=model_id,
                        max_topics=max_topics,
                        max_docs_per_topic=max_docs_per_topic,
                        force=force,
                        system_prompt=st.session_state.get("llm_system_prompt", SYSTEM_PROMPT),
                        user_template=st.session_state.get("llm_user_template", USER_TEMPLATE),
                    )
                    st.session_state.llm_names = llm_names
                    st.success(f"Generated {len(llm_names)} labels.")
                    st.rerun()
                except Exception as e:
                    st.error(f"LLM labelling failed: {e}")


            hf_tokens_total = st.session_state.get("hf_tokens_total", 0)
            if hf_tokens_total:
                approx_cost = hf_tokens_total / 1_000_000 * HF_APPROX_PRICE_PER_MTOKENS_USD
                st.caption(
                    f"Approx. HF LLM usage this session: ~{hf_tokens_total:,} tokens "
                    f"(~${approx_cost:.4f} at "
                    f"${HF_APPROX_PRICE_PER_MTOKENS_USD}/M tokens, "
                    "based on Novita Llama 3 8B pricing). "
                )
                
            
            # Apply labels (LLM overrides keyword names)
            default_map = tm.get_topic_info().set_index("Topic")["Name"].to_dict()
            api_map = st.session_state.get("llm_names", {}) or {}
            llm_names = {**default_map, **api_map}



            labs = []
            for t in tm.topics_:
                if t == -1:
                    labs.append("Unlabelled")
                else:
                    labs.append(llm_names.get(t, "Unlabelled"))



            if outlier_count > 0:
                st.markdown("### Outlier Reduction")
                with st.expander("Assign 'Unlabelled' reports to topics"):
                    st.caption(
                        "**Warning:** Reducing outliers alters the scientific strictness of your model. "
                        "It forces noise points into their nearest semantic topic."
                    )
                    
                    col_red1, col_red2, col_red3 = st.columns([1, 1, 1])
                    
                    with col_red1:
                        red_strategy = st.selectbox(
                            "Strategy", 
                            ["embeddings", "c-tf-idf"],
                            index=0, # Default to embeddings
                            help="Embeddings: Match based on meaning (semantic vectors). c-TF-IDF: Match based on shared keywords."
                        )
                    
                    with col_red2:
                        red_threshold = st.slider(
                            "Similarity Threshold",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.4,
                            step=0.05,
                            help="If an outlier's similarity to the nearest topic is below this number, it stays as an outlier. Higher = Stricter."
                        )

                    with col_red3:
                        st.write("") # Formatting spacer
                        if st.button("Reduce Outliers", use_container_width=True):
                            with st.spinner(f"Reassigning outliers (Strategy: {red_strategy}, Threshold: {red_threshold})..."):
                                try:
                                    new_topics = tm.reduce_outliers(
                                        docs, 
                                        tm.topics_, 
                                        strategy=red_strategy, 
                                        embeddings=embeddings,
                                        threshold=red_threshold
                                    )
                                    
                                    tm.update_topics(docs, topics=new_topics)
                                    

                                    new_info = tm.get_topic_info()
                                    new_name_map = new_info.set_index("Topic")["Name"].to_dict()
                                    
                                    final_labels = []
                                    current_llm_map = st.session_state.get("llm_names", {})
                                    
                                    for t in new_topics:
                                        if t == -1:
                                            final_labels.append("Unlabelled")
                                        else:
                                            lab = current_llm_map.get(t, new_name_map.get(t, f"Topic {t}"))
                                            final_labels.append(lab)

                                    st.session_state.latest_results = (tm, reduced, final_labels)
                                    
                                    st.success(f"Outliers reduced! (Threshold {red_threshold})")
                                    st.rerun()
                                    
                                except Exception as e:
                                    st.error(f"Error reducing outliers: {e}")
            
            
            # VISUALISATION
            # st.subheader("Experiential Topics Visualisation")

            # dataset_title = ds_input.strip() or DATASET_DIR
            # plot_title = f"{dataset_title}: MOSAIC's Experiential Topic Map"
            
            # fig, _ = datamapplot.create_plot(
            #     reduced,
            #     labs,
            #     noise_label="Unlabelled",  
            #     noise_color="#CCCCCC",     
            #     label_font_size=11,        
            #     arrowprops={"arrowstyle": "-", "color": "#333333"} 
            # )
            # fig.suptitle(plot_title, fontsize=16, y=0.99)
            # st.pyplot(fig)
            # VISUALISATION
            st.subheader("Experiential Topics Visualisation")

            with st.expander("Topic map appearance"):
                _appc1, _appc2 = st.columns(2)
                with _appc1:
                    map_label_size = st.slider(
                        "Topic label size", 6, 30, 16, 1,
                        help="Font size of the topic labels on the map.",
                    )
                with _appc2:
                    map_dpi = st.slider(
                        "Image quality (DPI)", 100, 400, 300, 50,
                        help="Higher = sharper image (and a bigger file). 300 is print quality.",
                    )

            dataset_title = ds_input.strip() or DATASET_DIR
            plot_title = f"{dataset_title}: MOSAIC's Experiential Topic Map"

            fig, _ = datamapplot.create_plot(
                reduced,
                labs,
                noise_label="Unlabelled",
                noise_color="#CCCCCC",
                figsize=(18, 18),
                dynamic_label_size=True,
                dynamic_label_size_scaling_factor=0.85,
                label_font_size=map_label_size,
                label_wrap_width=15,
                label_margin_factor=1.5,
                arrowprops={"arrowstyle": "-", "color": "#333333"}
            )
            fig.suptitle(plot_title, fontsize=16, y=0.99)
            # Render a high-DPI PNG and show that (st.pyplot rasterises at a low
            # default DPI, which looks blurry for this dense map).
            _map_buf = BytesIO()
            fig.savefig(_map_buf, format="png", dpi=map_dpi, bbox_inches="tight")
            st.image(_map_buf.getvalue(), use_container_width=True)

            # 2. Interactive Documents & Topics scatter (built directly with plotly)
            _topic_assignments = np.array(tm.topics_)
            _unique_topic_ids = sorted(set(_topic_assignments.tolist()))
            _palette = (
                px.colors.qualitative.Plotly
                + px.colors.qualitative.Set2
                + px.colors.qualitative.Pastel
            )
            _doc_fig = go.Figure()
            for _tid in _unique_topic_ids:
                _mask = np.where(_topic_assignments == _tid)[0]
                _x = reduced[_mask, 0]
                _y = reduced[_mask, 1]
                if _tid == -1:
                    _label = "Unlabelled"
                    _color = "#CCCCCC"
                    _size = 4
                    _opacity = 0.4
                else:
                    _label = llm_names.get(_tid, f"Topic {_tid}")
                    _color = _palette[_tid % len(_palette)]
                    _size = 6
                    _opacity = 0.75
                _hover = [
                    f"<b>{_label}</b><br><br>{docs[i][:300]}{'…' if len(docs[i]) > 300 else ''}"
                    for i in _mask
                ]
                _doc_fig.add_trace(go.Scattergl(
                    x=_x, y=_y,
                    mode="markers",
                    name=_label,
                    marker=dict(color=_color, size=_size, opacity=_opacity),
                    text=_hover,
                    hoverinfo="text",
                ))
            _doc_fig.update_layout(
                title=dict(
                    text="<b>Documents and Topics</b>",
                    x=0.5, xanchor="center",
                    font=dict(size=18, color="#222222"),
                ),
                template="simple_white",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                height=650,
                margin=dict(l=10, r=10, t=60, b=10),
                legend=dict(
                    title="Topics",
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#dddddd",
                    borderwidth=1,
                    font=dict(size=11),
                ),
            )
            st.plotly_chart(_doc_fig, use_container_width=True)

            



            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=map_dpi, bbox_inches="tight")
            png_bytes = buf.getvalue()
            
            base = os.path.splitext(os.path.basename(CSV_PATH))[0]
            gran = "sentences" if selected_granularity else "reports"
            png_name = f"topics_{base}_{gran}_plot.png"
            
            dl_col, save_col = st.columns(2)
            
            with dl_col:
                st.download_button(
                    "Download visualisation as PNG",
                    data=png_bytes,
                    file_name=png_name,
                    mime="image/png",
                    use_container_width=True,
                )
            with save_col:
                if st.button("Save plot to eval/", use_container_width=True):
                    try:
                        plot_path = (EVAL_DIR / png_name).resolve()
                        fig.savefig(plot_path, format="png", dpi=300, bbox_inches="tight")
                        st.success(f"Saved plot → {plot_path}")
                    except Exception as e:
                        st.error(f"Failed to save plot: {e}")





            

            st.subheader("Topic Info")
            st.dataframe(tm.get_topic_info())
            

            st.subheader("Topic Participation Analysis")
            
            if "diversity_stats" in st.session_state:
                diversity_stats = st.session_state.diversity_stats
                
                # Metric Description
                st.caption("""
                **Diversity Ratio:** - **Close to 1.0 (100%):** High consensus. The topic is built from sentences spoken by many different people.
                - **Close to 0.0 (0%):** Low consensus. The topic is mostly one or two people talking a lot (monopolising the topic).
                """)


                st.dataframe(
                    diversity_stats[["Topic", "Name", "Total_Sentences", "Unique_Reports", "Diversity_Ratio"]]
                    .sort_values("Unique_Reports", ascending=False)
                    .style.background_gradient(subset=["Diversity_Ratio"], cmap="RdYlGn"),
                    use_container_width=True
                )

                chart = alt.Chart(diversity_stats).mark_circle(size=100).encode(
                    x=alt.X('Total_Sentences', title='Total Sentences in Topic'),
                    y=alt.Y('Unique_Reports', title='Unique Reports (Participants)'),
                    color=alt.Color('Diversity_Ratio', scale=alt.Scale(scheme='redyellowgreen'), title='Diversity'),
                    tooltip=['Name', 'Total_Sentences', 'Unique_Reports', 'Diversity_Ratio']
                ).properties(
                    title="Are topics driven by group consensus or individual monologues?",
                    height=400
                ).interactive()

                line = alt.Chart(pd.DataFrame({'x': [0, diversity_stats['Total_Sentences'].max()], 'y': [0, diversity_stats['Total_Sentences'].max()]})).mark_line(color='grey', strokeDash=[5,5]).encode(x='x', y='y')

                st.altair_chart(chart + line, use_container_width=True)
                

                st.info("""
                **Topic Distribution: Robustness vs. Idiosyncratic Discovery**
                
                This graph distinguishes between widespread shared experiences (robust structures) and highly detailed personal accounts (idiosyncratic discoveries).

                - **The Diagonal Line (100% Diversity):** *Phenomenological Robustness.* Every sentence comes from a different participant. This indicates a structural invariant shared across the cohort.
                - **The Vertical Drop:** *Idiosyncratic Discovery.* The further a dot drops below the line, the more the topic is defined by a specific individual's detailed account.
                    - **Green (High Diversity):** Represents a shared, inter-subjective pattern.
                    - **Red (Low Diversity):** Represents a deep, specific, or unique individual experience.
                """)

                st.subheader("Topic Filtering")
                min_participants = st.slider(
                    "Hide topics with fewer than N unique reports/participants",
                    min_value=1, 
                    max_value=20, 
                    value=1,
                    help="Topics driven by fewer than this many unique people will be marked as 'Idiosyncratic' and excluded from the main list."
                )

                # Identify "bad" topics using the correctly named variable
                idiosyncratic_topics = diversity_stats[
                    diversity_stats["Unique_Reports"] < min_participants
                ]["Topic"].tolist()

                # Update the labels map for display
                filtered_llm_names = llm_names.copy()
                for t in idiosyncratic_topics:
                    filtered_llm_names[t] = "Too Specific (Idiosyncratic)"

                # Update the visual feedback
                st.write(f"Flagged {len(idiosyncratic_topics)} topics as idiosyncratic.")
            
            else:
                st.caption("Topic participation stats unavailable (Metadata missing or run not finished).")
                filtered_llm_names = llm_names.copy()


            st.subheader("Export results (one row per topic)")

            model_docs = getattr(tm, "docs_", None)
            if model_docs is not None and len(docs) != len(model_docs):
                st.caption("Note: Dataset size changed since model training. Re-run 'Run Analysis' for accurate mapping.")


            docs_for_export = st.session_state.get("latest_docs", getattr(tm, "docs_", docs))
            
            if len(docs_for_export) != len(tm.topics_):
                st.error("Error: Doc/Topic count mismatch. Please re-run **Run Analysis**.")
                st.stop()
            
            doc_info = tm.get_document_info(docs_for_export)[["Document", "Topic"]]


            if os.path.exists(METADATA_FILE):
                try:
                    meta_df = pd.read_csv(METADATA_FILE)
                    if len(meta_df) == len(doc_info):
                        if "_source_row_idx" in meta_df.columns:
                            doc_info["Report_ID"] = meta_df["_source_row_idx"].values
                        if "reflection_answer_english" in meta_df.columns:
                            doc_info["Original_Full_Report"] = meta_df["reflection_answer_english"].values
                except Exception as e:
                    st.warning(f"Could not load metadata IDs: {e}")


            include_outliers = st.checkbox("Include outlier topic (-1)", value=False)
            if not include_outliers:
                doc_info = doc_info[doc_info["Topic"] != -1]


            agg_funcs = {"Document": list}
            if "Report_ID" in doc_info.columns:
                agg_funcs["Report_ID"] = list

            grouped = (
                doc_info.groupby("Topic")
                .agg(agg_funcs)
                .reset_index()
            )
            
            # Renaming for clarity
            grouped = grouped.rename(columns={"Document": "texts"})
            if "Report_ID" in grouped.columns:
                grouped = grouped.rename(columns={"Report_ID": "report_ids"})

            # Map Names
            grouped["topic_name"] = grouped["Topic"].map(filtered_llm_names).fillna("Unlabelled")

            # Reorder columns
            cols = ["Topic", "topic_name", "texts"]
            if "report_ids" in grouped.columns:
                cols.insert(2, "report_ids") # Put IDs right after name
            
            export_topics = grouped[cols].sort_values("Topic").reset_index(drop=True)

            export_csv = export_topics.copy()
            SEP = " | "
            export_csv["texts"] = export_csv["texts"].apply(lambda lst: SEP.join(map(str, lst)))
            if "report_ids" in export_csv.columns:
                export_csv["report_ids"] = export_csv["report_ids"].apply(lambda lst: str(list(lst)))

            base = os.path.splitext(os.path.basename(CSV_PATH))[0]
            gran = "sentences" if selected_granularity else "reports"
            

            st.markdown(
                "**Save this file if you want to compare conditions later** — "
                "the *Condition Comparison* tab requires one of these CSVs per condition."
            )
            csv_name = f"topics_summary_{base}_{gran}.csv"
            st.download_button(
                "⬇ Save Summary CSV for Condition Comparison (Row = Topic)",
                data=export_csv.to_csv(index=False).encode("utf-8-sig"),
                file_name=csv_name,
                mime="text/csv",
                use_container_width=True,
                type="primary",
            )

            # One-click bundle: everything needed to re-run Zero-Shot (.npy + .json)
            # and Condition Comparison (topics summary CSV) later, without re-running.
            st.markdown(
                "**Or grab everything at once** — this `.zip` has the embeddings + docs "
                "(for *Zero-Shot*) and the topics summary (for *Condition Comparison*)."
            )
            _bundle_buf = BytesIO()
            with zipfile.ZipFile(_bundle_buf, "w", zipfile.ZIP_DEFLATED) as _zf:
                _zf.writestr(csv_name, export_csv.to_csv(index=False))
                if os.path.exists(EMBEDDINGS_FILE):
                    _zf.write(EMBEDDINGS_FILE, arcname=os.path.basename(EMBEDDINGS_FILE))
                if os.path.exists(DOCS_FILE):
                    _zf.write(DOCS_FILE, arcname=os.path.basename(DOCS_FILE))
            st.download_button(
                "⬇ Download reusable bundle (.zip): embeddings + docs + topics summary",
                data=_bundle_buf.getvalue(),
                file_name=f"mosaic_bundle_{base}_{gran}.zip",
                mime="application/zip",
                use_container_width=True,
                help="Re-upload the .npy + .json in the Zero-Shot tab, and the topics summary CSV "
                     "in the Condition Comparison tab — no need to re-run the pipeline.",
            )
            st.markdown("---")

            cL, cC, cR = st.columns(3)

            with cL:
                csv_name = f"topics_summary_{base}_{gran}.csv"
                st.download_button(
                    "Save Summary CSV (Row = Topic)",
                    data=export_csv.to_csv(index=False).encode("utf-8-sig"),
                    file_name=csv_name,
                    mime="text/csv",
                    use_container_width=True
                )

            with cC:
                jsonl_name = f"topics_{base}_{gran}.jsonl"
                if st.button("Save JSONL to eval/", use_container_width=True):
                    st.success(f"Saved JSONL")

            with cR:
                long_csv_name = f"all_sentences_{base}_{gran}.csv"
                
                long_df = doc_info.copy()
                long_df["Topic Name"] = long_df["Topic"].map(filtered_llm_names).fillna("Unlabelled")
                
                desired_cols = ["Topic", "Topic Name", "Report_ID", "Document", "Original_Full_Report"]
                final_cols = [c for c in desired_cols if c in long_df.columns]
                long_df = long_df[final_cols]

                st.download_button(
                    "Download All Sentences (Long Format)",
                    data=long_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name=long_csv_name,
                    mime="text/csv",
                    use_container_width=True,
                    help="One row per sentence. Includes Report IDs."
                )

            st.dataframe(export_csv)



            st.subheader("Representative Sentences Export")

            top_docs_list = []
            unique_valid_topics = [t for t in set(tm.topics_) if t != -1]
            SEP = " | "

            for t in unique_valid_topics:
                reps = tm.get_representative_docs(t)
                if reps:
                    topic_label = filtered_llm_names.get(t, f"Topic {t}")
                    top_docs_list.append({
                        "Topic": t,
                        "Topic Name": topic_label,
                        "Sentences": SEP.join(map(str, reps[:10]))
                    })

            if top_docs_list:
                df_top_docs = pd.DataFrame(top_docs_list)
                top_docs_csv_name = f"top_10_representative_sentences_{base}_{gran}.csv"

                st.download_button(
                    label="Download Top 10 Representative Sentences per Topic",
                    data=df_top_docs.to_csv(index=False).encode("utf-8-sig"),
                    file_name=top_docs_csv_name,
                    mime="text/csv",
                    use_container_width=True,
                    help="Extracts up to 10 most mathematically central sentences for each topic."
                )



        else:
            st.info("Click 'Run Analysis' (scroll down left corner - after params selection -) to begin.")

    # --- ZERO-SHOT TAB ---
    with zeroshot_tab:
        st.subheader("Zero-Shot Topic Classification")
        st.caption(
            "Classify your documents into **predefined categories** using semantic similarity. "
            "Uses the same preprocessed docs and embeddings as the main pipeline — no extra embedding step needed."
        )

        zs_categories_raw = st.text_area(
            "Categories — one per line (edit freely)",
            value=_ZS_DEFAULT_CATEGORIES,
            height=260,
            help="These are the predefined topics you want to classify documents into. "
                 "Edit or replace them with categories relevant to your dataset.",
        )

        zs_col1, zs_col2 = st.columns([1, 2])
        with zs_col1:
            zs_min_sim = st.slider(
                "Minimum similarity threshold",
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                step=0.05,
                help=(
                    "Cosine similarity ranges from 0 (completely unrelated) to 1 (identical meaning).\n\n"
                    "Each sentence is compared to every category label. "
                    "If its highest similarity score is below this threshold, it is left as 'Unclassified'.\n\n"
                    "**Typical values:**\n"
                    "- 0.3–0.4 → very lenient, most sentences get assigned (risk of false positives)\n"
                    "- 0.5 → balanced default\n"
                    "- 0.6–0.7 → strict, only high-confidence matches classified\n\n"
                    "If too many sentences are Unclassified, lower the threshold. "
                    "If the categories feel noisy, raise it."
                ),
            )

        with zs_col2:
            st.info(
                f"**{len(docs):,}** documents ready\n\n"
                f"**Embedding model:** `{selected_embedding_model}` (set in the sidebar)\n\n"
                "Zero-shot works by embedding both your documents **and** your category labels "
                "into the same vector space, then assigning each sentence to the nearest category. "
                f"Your documents are already embedded — only the category labels will be re-encoded "
                f"using `{selected_embedding_model}` when you click Run."
            )

        if st.button("Run Zero-Shot Classification", type="primary", key="zs_run_btn"):
            zs_categories = [c.strip() for c in zs_categories_raw.strip().splitlines() if c.strip()]
            if not zs_categories:
                st.error("Please enter at least one category.")
            else:
                with st.spinner(f"Classifying {len(docs):,} documents into {len(zs_categories)} categories…"):
                    try:
                        zs_topics, zs_topic_info, _ = run_zeroshot(
                            docs,
                            embeddings,
                            tuple(zs_categories),
                            zs_min_sim,
                            selected_embedding_model,
                            DOCS_FILE,
                        )
                        st.session_state["zs_results"] = (zs_topics, zs_topic_info, zs_categories)
                        st.session_state["zs_results_docs_file"] = DOCS_FILE
                    except Exception as e:
                        st.error(f"Zero-shot classification failed: {e}")

        if "zs_results" in st.session_state and st.session_state.get("zs_results_docs_file") != DOCS_FILE:
            st.warning("The dataset has changed. Re-run zero-shot classification for the current dataset.")
        elif "zs_results" in st.session_state:
            zs_topics, zs_topic_info, zs_categories = st.session_state["zs_results"]

            # Summary metrics
            total_zs = len(zs_topics)
            classified_zs = sum(1 for t in zs_topics if t != -1)
            unclassified_zs = total_zs - classified_zs

            zm1, zm2, zm3 = st.columns(3)
            zm1.metric("Total documents", f"{total_zs:,}")
            zm2.metric("Classified", f"{classified_zs:,} ({100*classified_zs/total_zs:.1f}%)")
            zm3.metric("Unclassified", f"{unclassified_zs:,} ({100*unclassified_zs/total_zs:.1f}%)")

            # Build per-doc DataFrame
            zs_name_map = zs_topic_info.set_index("Topic")["Name"].to_dict()
            zs_df = pd.DataFrame({"sentence": docs, "topic_id": zs_topics})
            zs_df["category"] = zs_df["topic_id"].map(zs_name_map).fillna("Unclassified")

            # Bar chart (classified topics only, sorted by count)
            zs_plot_df = (
                zs_topic_info[zs_topic_info["Topic"] != -1]
                .sort_values("Count", ascending=True)
                .reset_index(drop=True)
            )

            if not zs_plot_df.empty:
                st.subheader("Distribution across categories")
                import matplotlib.cm as _cm
                import matplotlib.colors as _mcolors

                _norm = _mcolors.Normalize(
                    vmin=zs_plot_df["Count"].min(),
                    vmax=zs_plot_df["Count"].max(),
                )
                _cmap = _cm.get_cmap("Blues")
                _bar_colors = [_cmap(0.35 + 0.55 * _norm(v)) for v in zs_plot_df["Count"]]

                fig_zs, ax_zs = plt.subplots(figsize=(10, max(4, len(zs_plot_df) * 0.62)))
                fig_zs.patch.set_facecolor("white")
                ax_zs.set_facecolor("#f8f9fa")

                bars = ax_zs.barh(
                    zs_plot_df["Name"], zs_plot_df["Count"],
                    color=_bar_colors, edgecolor="white", linewidth=0.8, height=0.65,
                )
                _max_count = zs_plot_df["Count"].max()
                for bar in bars:
                    w = bar.get_width()
                    ax_zs.text(
                        w + _max_count * 0.015,
                        bar.get_y() + bar.get_height() / 2,
                        str(int(w)),
                        va="center", ha="left", fontsize=9,
                        color="#333333", fontweight="bold",
                    )

                for spine in ("top", "right"):
                    ax_zs.spines[spine].set_visible(False)
                ax_zs.spines["left"].set_color("#cccccc")
                ax_zs.spines["bottom"].set_color("#cccccc")
                ax_zs.set_xlabel("Number of sentences", fontsize=10, color="#555555")
                ax_zs.tick_params(axis="y", labelsize=9, colors="#333333")
                ax_zs.tick_params(axis="x", labelsize=8, colors="#777777")
                ax_zs.set_xlim(0, _max_count * 1.15)
                ax_zs.invert_yaxis()
                ax_zs.grid(axis="x", color="#e0e0e0", linewidth=0.8, zorder=0)
                ax_zs.set_axisbelow(True)
                ax_zs.set_title(
                    f"Zero-Shot Classification  ·  {classified_zs:,} / {total_zs:,} sentences classified",
                    fontsize=11, color="#333333", pad=12,
                )
                plt.tight_layout(pad=1.5)
                st.pyplot(fig_zs)

                buf_zs = BytesIO()
                fig_zs.savefig(buf_zs, format="png", dpi=200, bbox_inches="tight")
                st.download_button(
                    "Download chart as PNG",
                    data=buf_zs.getvalue(),
                    file_name=f"zeroshot_chart_{os.path.splitext(os.path.basename(CSV_PATH))[0]}.png",
                    mime="image/png",
                )
                plt.close(fig_zs)

            # Per-category expandable view
            st.subheader("Sentences per category")
            for _, row_zs in zs_plot_df.sort_values("Count", ascending=False).iterrows():
                cat_name = row_zs["Name"]
                count_zs = row_zs["Count"]
                with st.expander(f"{cat_name}  ({count_zs} sentences)"):
                    cat_sentences = zs_df[zs_df["category"] == cat_name]["sentence"].reset_index(drop=True)
                    n_show = st.slider(
                        "Sentences to show", 5, min(100, len(cat_sentences)), 10,
                        key=f"zs_show_{cat_name}"
                    )
                    st.dataframe(
                        cat_sentences.head(n_show).to_frame("sentence"),
                        use_container_width=True,
                    )

            # Unclassified preview
            with st.expander(f"Unclassified  ({unclassified_zs} sentences)"):
                unclass_sentences = zs_df[zs_df["category"] == "Unclassified"]["sentence"].reset_index(drop=True)
                st.dataframe(unclass_sentences.head(50).to_frame("sentence"), use_container_width=True)

            # Download full results
            zs_base = os.path.splitext(os.path.basename(CSV_PATH))[0]
            st.download_button(
                "Download full classification results (CSV)",
                data=zs_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"zeroshot_{zs_base}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # --- CONDITION COMPARISON TAB ---
    with condition_tab:
        st.subheader("Condition Comparison — Semantic Similarity")

        with st.expander("How to use this tab — read first", expanded=True):
            st.markdown(
                f"""
**This tab compares the topics found in two different conditions** by measuring the semantic
similarity between their topic vectors.

**Before you can use this tab, you need to:**

1. **Run the full pipeline on Condition A** (upload its CSV, prepare data, run analysis)
2. In the **Main Results** tab, click the blue **"Save Summary CSV for Condition Comparison"** button — save that file to your computer
3. **Go back to the sidebar**, upload Condition B's CSV, prepare data, and run analysis
4. Again click **"Save Summary CSV for Condition Comparison"** — save that second file
5. Come back here and upload both saved files below

**What the comparison does:**
- All sentences in each topic are embedded using `{selected_embedding_model}` (same model as the sidebar)
- Each topic is represented by the **mean vector** of its sentences
- **Cosine similarity** is computed between every pair of topics across conditions
- A greedy algorithm finds the best-matching pairs above the threshold
- Results include a heatmap, matched/unmatched topic lists, a contingency table, a chi-squared test, and a frequency bar chart
                """
            )

        _condition_comparison_ui(selected_embedding_model)

    # --- HISTORY TAB ---
    with history_tab:
        st.subheader("Run History")
        if not st.session_state.history:
            st.info("No runs yet.")
        else:
            for i, entry in enumerate(st.session_state.history):
                with st.expander(f"Run {i+1} — {entry['timestamp']}"):
                    st.write(f"**Topics:** {entry['num_topics']}")

                    outp = entry.get("outlier_pct", None)
                    if isinstance(outp, (int, float)):
                        st.write(f"**Outliers:** {outp:.2f}%")
                    else:
                        st.write(f"**Outliers:** {outp}")

                    st.write("**Topic Labels (default keywords):**")
                    st.write(entry["llm_labels"])
                    with st.expander("Show full configuration"):
                        st.json(entry["config"])



    with compare_tab:
        st.subheader("Compare runs")
    
        hist = st.session_state.get("history", [])
        if not hist:
            st.info("No runs yet.")
        else:
            # dataset selector
            dataset_options = sorted({e.get("dataset_title", "Unknown") for e in hist})
            chosen_ds = st.selectbox("Dataset", dataset_options)
    
            hist_ds = [e for e in hist if e.get("dataset_title", "Unknown") == chosen_ds]

            
            # Table view
            rows = []
            for e in hist_ds:
                cfg = e.get("config", {}) or {}
                rows.append({
                    "run_id": e.get("run_id", ""),
                    "timestamp": e.get("timestamp", ""),
                    "topics": e.get("num_topics", ""),
                    "outliers_%": e.get("outlier_pct", ""),
                    "min_words": cfg.get("min_words", ""),
                    "granularity": cfg.get("granularity", ""),
                    "embedding": cfg.get("embedding_model", ""),
                    "umap_n": (cfg.get("umap_params") or {}).get("n_neighbors", ""),
                    "umap_dist": (cfg.get("umap_params") or {}).get("min_dist", ""),
                    "hdb_min_cluster": (cfg.get("hdbscan_params") or {}).get("min_cluster_size", ""),
                    "hdb_min_samples": (cfg.get("hdbscan_params") or {}).get("min_samples", ""),
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
    
            # Side-by-side snapshots
            run_ids = [e.get("run_id") for e in hist if e.get("run_id")]
            selected = st.multiselect("Select runs to view plots", run_ids, default=run_ids[:2])
    
            chosen = [e for e in hist if e.get("run_id") in selected]
            if chosen:
                cols = st.columns(min(3, len(chosen)))
                for col, e in zip(cols, chosen[:3]):
                    rid = e.get("run_id", "—")
                    col.markdown(f"**{rid}**")
                    outp = e.get("outlier_pct", 0.0)
                    try:
                        col.caption(f"Topics: {e.get('num_topics','—')} • Outliers: {float(outp):.2f}%")
                    except Exception:
                        col.caption(f"Topics: {e.get('num_topics','—')} • Outliers: {outp}")
    
                    plot_path = (e.get("artifacts") or {}).get("plot_png")
                    if plot_path and os.path.exists(plot_path):
                        col.image(plot_path, use_container_width=True)
                    else:
                        col.caption("No saved plot found.")
    
                for e in chosen[3:]:
                    rid = e.get("run_id", "—")
                    with st.expander(f"{rid} — details"):
                        st.json(e.get("config", {}), expanded=False)
                        plot_path = (e.get("artifacts") or {}).get("plot_png")
                        if plot_path and os.path.exists(plot_path):
                            st.image(plot_path, use_container_width=True)