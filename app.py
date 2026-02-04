"""
File: app.py
Description: Streamlit app for advanced topic modeling on Innerspeech dataset
             with BERTopic, UMAP, HDBSCAN. (LLM features disabled for lite deployment)
Last Modified: 23/12/2025
@author: r.beaut@sussex.ac.uk
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

from io import BytesIO #Download button for the clustering image


import hashlib
from datetime import datetime




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

def _cleanup_old_cache(current_slug: str):
    """Deletes precomputed cache files that do not match the current dataset slug."""
    if not CACHE_DIR.exists():
        return

    removed_count = 0
    cache_files = list(CACHE_DIR.glob("precomputed_*.npy")) + list(CACHE_DIR.glob("precomputed_*_docs.json"))
    for p in cache_files:
        # If the file belongs to a different dataset (doesn't contain the new slug)
        if current_slug not in p.name:
            try:
                p.unlink() # Delete file
                removed_count += 1
            except Exception as e:
                print(f"Error deleting {p.name}: {e}")
                
    if removed_count > 0:
        print(f"Auto-cleanup: Removed {removed_count} old cache files.")

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

    # --- save plot ---
    fig, _ = datamapplot.create_plot(
        reduced,
        labs,
        noise_label="Unlabelled",
        noise_color="#CCCCCC",
        label_font_size=11,
        arrowprops={"arrowstyle": "-", "color": "#333333"},
    )
    fig.suptitle(f"{dataset_title}: MOSAIC Topic Map", fontsize=16, y=0.99)

    plot_png = run_dir / "plot.png"
    fig.savefig(plot_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- save topic info ---
    topic_info_csv = run_dir / "topic_info.csv"
    info.to_csv(topic_info_csv, index=False)

    # --- save meta ---
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
        """.strip()
    )


def _list_server_csvs(proc_dir: Path) -> list[str]:
    return [str(p) for p in sorted(proc_dir.glob("*.csv"))]


DATASETS = None  # keep name for clarity; we’ll fill it when rendering the sidebar
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

# Approximate price for cost estimates in the UI only.
# Novita Llama 3 8B is around $0.04 per 1M input tokens
# and $0.04 per 1M output tokens – adjust if needed.
HF_APPROX_PRICE_PER_MTOKENS_USD = 0.04


#ADDED FOR LLM (START)
@st.cache_resource
def get_hf_client(model_id: str):
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            token = st.secrets.get("HF_TOKEN")
        except Exception:
            token = None

    # Bake the model into the client so you don't pass model= every call
    client = InferenceClient(model=model_id, token=token)
    return client, token

def _labels_cache_path(config_hash: str, model_id: str) -> Path:
    safe_model = re.sub(r"[^a-zA-Z0-9_.-]", "_", model_id)
    return CACHE_DIR / f"llm_labels_{safe_model}_{config_hash}.json"

def _hf_status_code(e: Exception) -> int | None:
    """Extract HTTP status code from a huggingface_hub error, if present."""
    resp = getattr(e, "response", None)
    return getattr(resp, "status_code", None)

# SYSTEM_PROMPT = """You are an expert phenomenologist analysing first-person experiential reports or microphenomenological interviews.

# Your task is to assign a concise label to a cluster of similar reports by identifying the
# shared lived experiential structure or process they describe.

# The label must:
# 1. Describe what changes in experience itself (e.g. boundaries, temporality, embodiment, agency, affect, meaning).
# 2. Capture the underlying experiential process or structural transformation, not surface narrative details.
# 3. Be specific and distinctive, but at the level of experiential structure rather than anecdotal content.
# 4. Use phenomenological language that describes how cognitive, affective, or perceptual processes are lived, rather than analytic or evaluative abstractions.
# 5. Be conceptually focused on a single dominant experiential pattern.
# 6. Be concise and noun-phrase-like.

# Constraints:
# - Output ONLY the label (no explanation).
# - 3–8 words.
# - Avoid surface-specific details unless they reflect a recurring experiential structure.
# - Avoid meta-level analytic terms (e.g. epistemic, estimation, verification, evaluation) unless they directly describe how the process is experienced.
# - Avoid generic wrappers such as "experience of", "state of", or "phenomenon of".
# - No punctuation, no quotes, no extra text.
# - Do not explain your reasoning.
# """


SYSTEM_PROMPT = """You are analysing first-person experiential reports from participants describing their subjective experiences.

Your task is to create a clear, meaningful label for a cluster of similar reports.

The label must:
1. Describe WHAT participants actually experienced in plain, concrete terms
2. Be specific enough that someone could recognize this experience from the label alone
3. Capture the CONTENT of the experience, not just its category
4. Use everyday language that participants themselves might use
5. Be 3-10 words long

BAD examples (too vague/abstract):
- "Temporal consciousness shifts"
- "Embodied affective processing"

GOOD examples (concrete and meaningful):
- "Feeling of merging with surroundings"
- "Losing track of time passing"
- "Vivid childhood memories"
- "Sense of body expanding or shrinking"
- "Feeling deeply calm and peaceful"

Constraints:
- Output ONLY the label
- 3-10 words
- Use language a participant would understand
- No jargon unless participants used it
- No quotes, no punctuation, no explanation
"""

    
# SYSTEM_PROMPT = """You are an expert phenomenologist analysing subjective reflections from specific experiences.
# Your task is to label a cluster of similar experiential reports.

# The title should be:
# 1. HIGHLY SPECIFIC to the experiential characteristic unique to this "phenomenological" cluster
# 2. PHENOMENOLOGICALLY DESCRIPTIVE (focus on *what* was felt/seen).
# 3. DISTINCTIVE enough that it wouldn't apply equally well to other "phenomenological" clusters
# 4. TECHNICALLY PRECISE, using domain-specific terminology where appropriate
# 5. CONCEPTUALLY FOCUSED on the core specificities of this type of experience
# 6. CONCISE and NOUN-PHRASE LIKE (e.g. "body boundary dissolution", not "Experience of body boundary dissolution").


# Constraints:
# - Output ONLY the label (no explanation).
# - 3–7 words.
# - Avoid generic wrappers such as "experience of", "phenomenon of", "state of" unless they are absolutely necessary.
# - No punctuation, no quotes, no extra text.
# - Do not explain your reasoning
# """


# USER_TEMPLATE = """Here is a cluster of participant reports describing a specific phenomenon:

# {documents}

# Top keywords associated with this cluster:
# {keywords}

# Task: Return a single scientifically precise label (3–7 words). Output ONLY the label.
# """

USER_TEMPLATE = """Below are participant reports that were clustered together because they describe similar experiences:

{documents}

Keywords statistically associated with this cluster: {keywords}

Based on WHAT these participants actually describe experiencing, write a single clear label (4-10 words) that captures the shared experience. Focus on the concrete content, not abstract categories.

Output ONLY the label:"""

def _clean_label(x: str) -> str:
    x = (x or "").strip()
    x = x.splitlines()[0].strip()          # first line only
    x = x.strip(' "\'`')
    x = re.sub(r"[.:\-–—]+$", "", x).strip()  # remove trailing punctuation
    # enforce "no punctuation" lightly (optional):
    x = re.sub(r"[^\w\s]", "", x).strip()
    # Optional: de-wrap generic "experience/phenomenon/state" wrappers
    # Leading patterns like "Experiential/Experience of ..."
    x = re.sub(
        r"^(Experiential(?:\s+Phenomenon)?|Experience|Experience of|Subjective Experience of|Phenomenon of)\s+",
        "",
        x,
        flags=re.IGNORECASE,
    )
    # Trailing "experience/phenomenon/state"
    x = re.sub(
        r"\s+(experience|experiences|phenomenon|state|states)$",
        "",
        x,
        flags=re.IGNORECASE,
    )

    x = x.strip()
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
    force: bool = False) -> dict[int, str]:
    """
    Label topics AFTER fitting (fast + stable on Spaces).
    Returns {topic_id: label}.
    """

    # Remember which HF model id we requested on the last run
    st.session_state["hf_last_model_param"] = model_id
    
    cache_path = _labels_cache_path(config_hash, model_id)

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

        user_prompt = USER_TEMPLATE.format(documents=docs_block, keywords=keywords)
        # Store one example prompt (for UI inspection) – will be overwritten each run
        st.session_state["hf_last_example_prompt"] = user_prompt

        # # --- THE KEY PART: chat_completion ---
        # out = client.chat_completion(
        #     model=model_id,
        #     messages=[
        #         {"role": "system", "content": SYSTEM_PROMPT},
        #         {"role": "user", "content": user_prompt},
        #     ],
        #     max_tokens=24,
        #     temperature=temperature,
        #     stop=["\n"],
        # )
        # # ------------------------------------

        # raw = out.choices[0].message.content
        # labels[int(tid)] = _clean_label(raw)


        # --- THE KEY PART: chat_completion ---
        try:
            out = client.chat_completion(
                model=model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=35, #Upper bound on how many tokens the model is allowed to generate as output for that label
                temperature=temperature,
                stop=["\n"],
            )
            # Store the provider-returned model id (if available)
            provider_model = getattr(out, "model", None)
            if provider_model:
                st.session_state["hf_last_provider_model"] = provider_model
        except Exception as e:
            # Nice message for the specific 402 you're seeing
            if _hf_status_code(e) == 402:
                raise RuntimeError(
                    "Hugging Face returned 402 Payment Required for this LLM call. "
                    "You have used up the monthly Inference Provider credits on this "
                    "account. Either upgrade to PRO / enable pay-as-you-go, or skip "
                    "the 'Generate LLM labels (API)' step."
                ) from e
            # Anything else: bubble up the original error
            raise
        # ------------------------------------

        # --- Best-effort local accounting of token usage (this Streamlit session) ---
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
        # ---------------------------------------------------------------------------

        raw = out.choices[0].message.content
        labels[int(tid)] = _clean_label(raw)


        prog.progress(int(100 * i / max(total, 1)))

    try:
        cache_path.write_text(json.dumps({str(k): v for k, v in labels.items()}, indent=2), encoding="utf-8")
    except Exception:
        pass

    return labels
#ADDED FOR LLM (END)



# =====================================================================
# 5. Topic modeling function
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
# 6. CSV → documents → embeddings pipeline
# =====================================================================


# def generate_and_save_embeddings(
#     csv_path,
#     docs_file,
#     emb_file,
#     selected_embedding_model,
#     split_sentences,
#     device,
#     text_col=None,
#     min_words: int = 0, #for removal of sentences with <N words
# ):

#     # ---------------------
#     # Load & clean CSV
#     # ---------------------
#     st.info(f"Reading and preparing CSV: {csv_path}")
#     df = pd.read_csv(csv_path)

#     if text_col is not None and text_col in df.columns:
#         col = text_col
#     else:
#         col = _pick_text_column(df)

#     if col is None:
#         st.error("CSV must contain at least one text column.")
#         return

#     if col != "reflection_answer_english":
#         df = df.rename(columns={col: "reflection_answer_english"})

#     df.dropna(subset=["reflection_answer_english"], inplace=True)
#     df["reflection_answer_english"] = df["reflection_answer_english"].astype(str)
#     df = df[df["reflection_answer_english"].str.strip() != ""]
#     reports = df["reflection_answer_english"].tolist()

#     #change to add data sanity check
#     granularity_label = "sentences" if split_sentences else "reports"
    
#     #change to account for sentence removal when < N words
#     if split_sentences:
#         try:
#             sentences = [s for r in reports for s in nltk.sent_tokenize(r)]
#         except LookupError as e:
#             st.error(f"NLTK tokenizer data not found: {e}")
#             st.stop()

#         total_units_before = len(sentences)

#         if min_words and min_words > 0:
#             docs = [s for s in sentences if len(s.split()) >= min_words]
#             removed_texts = [s for s in sentences if len(s.split()) < min_words]
#         else:
#             docs = sentences
#             removed_texts = []
#     else:
#         total_units_before = len(reports)
#         if min_words and min_words > 0:
#             docs = [r for r in reports if len(str(r).split()) >= min_words]
#             removed_texts = [r for r in reports if len(str(r).split()) < min_words]
#         else:
#             docs = reports
#             removed_texts = []

#     total_units_after = len(docs)
#     removed_units = total_units_before - total_units_after

#     # Store stats for later display in "Dataset summary"
#     st.session_state["last_data_stats"] = {
#         "granularity": granularity_label,
#         "min_words": int(min_words or 0),
#         "total_before": int(total_units_before),
#         "total_after": int(total_units_after),
#         "removed": int(removed_units),
#     }
#     # keep removed texts so the UI can show them
#     st.session_state["last_removed_units"] = removed_texts

#     if min_words and min_words > 0:
        
#         st.info(
#             f"Preprocessing: started with {total_units_before} {granularity_label}, "
#             f"removed {removed_units} shorter than {min_words} words; "
#             f"{total_units_after} remaining."
#         )
#     else:
#         st.info(f"Preprocessing: {total_units_after} {granularity_label} prepared.")

#     np.save(docs_file, np.array(docs, dtype=object))
#     st.success(f"Prepared {len(docs)} documents")

#     # ---------------------
#     # Embeddings
#     # ---------------------
#     st.info(
#         f"Encoding {len(docs)} documents with {selected_embedding_model} on {device}"
#     )

#     model = load_embedding_model(selected_embedding_model)

#     # encode_device = None
#     # batch_size = 32
#     # if device == "CPU":
#     #     encode_device = "cpu"
#     #     batch_size = 64

#     encode_device = None
#     batch_size = 32

#     # If user selected CPU explicitly, skip all checks
#     if device == "CPU":
#         encode_device = "cpu"
#         batch_size = 64
#     else:
#         # User selected GPU. We try CUDA -> MPS -> CPU
#         import torch
#         if torch.cuda.is_available():
#             encode_device = "cuda"
#             st.toast("Using NVIDIA GPU (CUDA)")
#         elif torch.backends.mps.is_available():
#             encode_device = "mps"
#             st.toast("Using Apple GPU (MPS)")
#         else:
#             encode_device = "cpu"
#             st.warning("No GPU found (neither CUDA nor MPS). Falling back to CPU.")

#     embeddings = model.encode(
#         docs,
#         show_progress_bar=True,
#         batch_size=batch_size,
#         device=encode_device,
#         convert_to_numpy=True,
#     )
#     embeddings = np.asarray(embeddings, dtype=np.float32)
#     np.save(emb_file, embeddings)

#     st.success("Embedding generation complete!")
#     st.balloons()
#     st.rerun()



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

    # Standardize the text column name
    if col != "reflection_answer_english":
        df = df.rename(columns={col: "reflection_answer_english"})

    # Clean empty rows
    df.dropna(subset=["reflection_answer_english"], inplace=True)
    df["reflection_answer_english"] = df["reflection_answer_english"].astype(str)
    df = df[df["reflection_answer_english"].str.strip() != ""]

    # Identify metadata columns (everything except the text)
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
            
            # Tokenize
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
    # 3. Update Session State (Crucial for Dashboard UI)
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
        json.dump(docs, f, ensure_ascii=False)
    st.success(f"Prepared {len(docs)} documents")

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

        # # Just save; we’ll choose the text column later
        # uploaded_csv_path = str((PROC_DIR / "uploaded.csv").resolve())
        # tmp_df.to_csv(uploaded_csv_path, index=False)
        # st.success(f"Uploaded CSV saved to {uploaded_csv_path}")
        # CSV_PATH = uploaded_csv_path

        # FIX: Use the original filename to avoid cache collisions
        # We sanitize the name to be safe for file systems
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
    help="Unsure which model to pick? Check the [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) for performance maximising on Clustering and STS tasks."
)

selected_device = st.sidebar.radio(
    "Processing device",
    ["GPU", "CPU"],
    index=0,
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
METADATA_FILE = DOCS_FILE.replace(".npy", "_metadata.csv").replace("_docs.csv", "_metadata.csv")
    
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
            min_words=min_words,
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
                "Cached embeddings are invalid. Please click on \"Clear cached files for this configuration\" in the sidebar and then regenerate them for this configuration."
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
        stopwords = st.select_slider(
            "Stopwords", 
            options=[None, "english"], 
            value=None,
            help="Remove common English words (the, is, at, etc.)"
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
            "min_samples", 2, 100, 5,
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
        
        current_config = {
            "embedding_model": selected_embedding_model,
            "granularity": granularity_label,
            "min_words": int(min_words or 0),
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
    main_tab, history_tab, compare_tab = st.tabs(["Main Results", "Run History", "Compare Runs"])


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
        # Store the exact docs used to fit this model (so export never mismatches)
        st.session_state.latest_docs = docs
        st.session_state.latest_csv_path = CSV_PATH

        # --- AUTO-SAVE RUN SNAPSHOT (plot + stats + topic_info) ---
        run_id = make_run_id(current_config)
        dataset_title = ds_input.strip() or DATASET_DIR
        
        # Make sure outliers show as "Unlabelled" in the saved plot
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


        ### ADD FOR LLM (START)
        st.session_state.latest_config_hash = get_config_hash(current_config)
        st.session_state.latest_config = current_config
        ### ADD FOR LLM (END)

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
        if "latest_results" in st.session_state:
            tm, reduced, labs = st.session_state.latest_results


            ##### ADDED FOR LLM (START)
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

            # -------------------------------
            # END Topic modelling stats (pre-LLM)
            # -------------------------------


            model_id = st.text_input(
                "HF model id for labelling",
                value="meta-llama/Meta-Llama-3-8B-Instruct",
            )
            with st.expander("Show LLM configuration and prompts"):
                # What we *request*
                st.markdown(f"**HF model id (requested):** `{model_id}`")
            
                # What was used on the last run, if available
                requested_last = st.session_state.get("hf_last_model_param")
                provider_model = st.session_state.get("hf_last_provider_model")
            
                if requested_last:
                    st.markdown(f"**Last run – requested model id:** `{requested_last}`")
                if provider_model:
                    st.markdown(f"**Last run – provider model (returned):** `{provider_model}`")
                else:
                    st.caption("Run LLM labelling once to see the provider-returned model id.")
            
                st.markdown("**System prompt:**")
                st.code(SYSTEM_PROMPT, language="markdown")
            
                st.markdown("**User prompt template:**")
                st.code(USER_TEMPLATE, language="markdown")
            
                example_prompt = st.session_state.get("hf_last_example_prompt")
                if example_prompt:
                    st.markdown("**Example full prompt for one topic (last run):**")
                    st.code(example_prompt, language="markdown")
                else:
                    st.caption("No example prompt stored yet – run LLM labelling to populate this.")
            
            cA, cB, cC = st.columns([1, 1, 2])

            with cA:
                max_topics = st.slider("Max topics", 5, 120, 40, 5)
            # max_topics = cA.slider("Max topics", 5, 120, 40, 5)

            with cB:
                max_docs_per_topic = st.slider(
                    "Docs per topic",
                    min_value=2,
                    max_value=40,
                    value=8,
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
                    )
                    st.session_state.llm_names = llm_names
                    st.success(f"Generated {len(llm_names)} labels.")
                    st.rerun()
                except Exception as e:
                    st.error(f"LLM labelling failed: {e}")


            # Approximate HF usage for *this* Streamlit session (local estimate only)
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



            # FIX: Force outliers (Topic -1) to be "Unlabelled" so we can hide them
            labs = []
            for t in tm.topics_:
                if t == -1:
                    labs.append("Unlabelled")
                else:
                    labs.append(llm_names.get(t, "Unlabelled"))
            
            # VISUALISATION
            st.subheader("Experiential Topics Visualisation")

            # Build a nice title from the dataset name
            dataset_title = ds_input.strip() or DATASET_DIR
            plot_title = f"{dataset_title}: MOSAIC's Experiential Topic Map"
            
            # We pass 'noise_label' and 'noise_color' to grey out the outliers
            fig, _ = datamapplot.create_plot(
                reduced,
                labs,
                noise_label="Unlabelled",  # Tells datamapplot: "Do not put a text label on this group"
                noise_color="#CCCCCC",     # Sets the points to a light Grey
                label_font_size=11,        # Optional: Adjust font size
                arrowprops={"arrowstyle": "-", "color": "#333333"} # Optional: darker, simpler arrows
            )
            fig.suptitle(plot_title, fontsize=16, y=0.99)
            st.pyplot(fig)

            # --- Download / save visualisation ---

            # Prepare high-res PNG bytes
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
            png_bytes = buf.getvalue()
            
            # Reuse base / gran for a nice filename later (they’re defined below as well)
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

            st.subheader("Export results (one row per topic)")


            model_docs = getattr(tm, "docs_", None)
            if model_docs is not None and len(docs) != len(model_docs):
                st.caption(
                    "Note: export uses the original documents from the topic-model run. "
                    "The current dataset size is different (e.g. sampling/splitting changed), "
                    "so you may want to re-run topic modelling before exporting."
                )

            # Always export using the same docs the model was trained on
            docs_for_export = st.session_state.get("latest_docs", None)
            
            # Fallback if session_state was cleared
            if docs_for_export is None:
                docs_for_export = getattr(tm, "docs_", None)
            
            # Final fallback (won't usually be needed)
            if docs_for_export is None:
                docs_for_export = docs
            
            # Hard safety check to prevent the ValueError
            if len(docs_for_export) != len(tm.topics_):
                st.error(
                    "Cannot export: the current docs don't match the model (dataset / subsample / filter changed). "
                    "Please re-run **Run Analysis** for the current configuration."
                )
                st.stop()
            
            doc_info = tm.get_document_info(docs_for_export)[["Document", "Topic"]]


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

                # Create a Long Format DataFrame (One row per sentence)
                # This ensures NO text is hidden due to Excel cell limits
                long_format_df = doc_info.copy()
                long_format_df["Topic Name"] = long_format_df["Topic"].map(llm_names).fillna("Unlabelled")
                
                # Reorder columns for clarity
                long_format_df = long_format_df[["Topic", "Topic Name", "Document"]]

                # === NEW: Merge Metadata if available ===
                if os.path.exists(METADATA_FILE):
                    try:
                        meta_df = pd.read_csv(METADATA_FILE)
                        # Only merge if the lengths match exactly (safety check)
                        if len(meta_df) == len(long_format_df):
                            long_format_df = pd.concat([
                                long_format_df.reset_index(drop=True), 
                                meta_df.reset_index(drop=True)
                            ], axis=1)
                    except Exception:
                        pass # If it fails, we just download the data without metadata
                # ========================================
                
                # Define filename
                long_csv_name = f"all_sentences_{base}_{gran}.csv"
                
                st.download_button(
                    "Download All Sentences (Long Format)",
                    data=long_format_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name=long_csv_name,
                    mime="text/csv",
                    use_container_width=True,
                    help="Download a CSV with one row per sentence. Best for checking exactly which sentences belong to which topic."
                )
                
                # st.download_button(
                #     "Download CSV",
                #     data=export_csv.to_csv(index=False).encode("utf-8-sig"),
                #     file_name=csv_name,
                #     mime="text/csv",
                #     use_container_width=True,
                # )

            # st.caption("Preview (one row per topic)")
            st.dataframe(export_csv)

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
                    # st.write(
                    #     f"**Outliers:** {entry.get('outlier_pct', entry.get('outlier_perc', 'N/A'))}"
                    # )
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
