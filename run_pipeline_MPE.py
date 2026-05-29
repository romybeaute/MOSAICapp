"""
Standalone HPC pipeline for MOSAIC, no Streamlit required.

Outputs are saved using the exact same paths and filenames as the Streamlit app,
so can open app.py afterwards and load precomputed results without re-running.

Usage:
    python run_pipeline_MPE.py
    sbatch run_pipeline_MPE.sh
"""

import argparse
import json
import logging
import re
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="Run topic modelling only, skip LLM labelling and plots")
parser.add_argument("--output-dir", type=str, default="outputs_MPE", help="Directory for plots and HTML (default: outputs_MPE)")
parser.add_argument("--llm-model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="HuggingFace model ID for LLM labelling")
args = parser.parse_args()

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for HPC (no display)
import matplotlib.pyplot as plt
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
    handlers=[logging.StreamHandler(), logging.FileHandler("pipeline_MPE.log")],
)
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).parent
DATASET_NAME    = "MPE"
CSV_PATH        = "data/MPE/preprocessed/MPE_dataset_translated_batched.csv"
TEXT_COL        = "phen_report_english"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"
DEVICE          = "cuda"
SPLIT_SENTENCES = True
MIN_WORDS       = 3
HF_TOKEN        = _load_secret("HF_TOKEN")

CONFIG = {
    "umap_params": {
        "n_neighbors": 13,
        "n_components": 10,
        "min_dist": 0.0,
    },
    "hdbscan_params": {
        "min_cluster_size": 11,
        "min_samples": 21,
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
if Path(DOCS_FILE).exists() and Path(EMBEDDINGS_FILE).exists():
    log.info("Step 1 — loading cached docs and embeddings")
    with open(DOCS_FILE, encoding="utf-8") as f:
        docs = json.load(f)
    embeddings = np.load(EMBEDDINGS_FILE)
    log.info(f"Loaded {len(docs)} sentences  →  shape {embeddings.shape}")
else:
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
config_hash = get_config_hash(CONFIG)
_topic_model_path = CACHE_DIR / "topic_model"
_topics_path      = CACHE_DIR / "topics.json"
_reduced_path     = CACHE_DIR / "reduced_2d.npy"

if _topic_model_path.exists() and _topics_path.exists() and _reduced_path.exists():
    log.info("Step 2 — loading cached topic model")
    from bertopic import BERTopic
    topic_model = BERTopic.load(str(_topic_model_path))
    with open(_topics_path) as f:
        topics = json.load(f)
    reduced_2d = np.load(_reduced_path)
else:
    log.info("Step 2 — topic modelling")
    topic_model, reduced_2d, topics = run_topic_model(docs, embeddings, CONFIG)
    topic_model.save(str(_topic_model_path))
    np.save(_reduced_path, reduced_2d)
    with open(_topics_path, "w") as f:
        json.dump(topics, f)

n_topics   = len(set(t for t in topics if t != -1))
n_outliers = sum(1 for t in topics if t == -1)
log.info(f"Topics: {n_topics}  |  Outliers: {n_outliers} ({100*n_outliers/len(topics):.1f}%)")

if args.debug:
    log.info("Debug mode — skipping LLM labelling and plots")
    import sys; sys.exit(0)

# ── Step 3: LLM labelling ─────────────────────────────────────────────────────
log.info("Step 3 — LLM labelling with Qwen3")

LLM_MODEL = args.llm_model

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

# ── Step 4: Save plots ────────────────────────────────────────────────────────
log.info("Step 4 — generating plots")

PLOTS_DIR = PROJECT_ROOT / args.output_dir
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Build per-document label list (LLM label if available, else BERTopic name)
topic_info = topic_model.get_topic_info()
name_map = topic_info.set_index("Topic")["Name"].to_dict()
doc_labels = [
    labels.get(t, name_map.get(t, "Unlabelled")) if t != -1 else "Unlabelled"
    for t in topics
]

# 1. Datamapplot — 2D topic map
try:
    import datamapplot
    fig, _ = datamapplot.create_plot(
        reduced_2d,
        doc_labels,
        noise_label="Unlabelled",
        noise_color="#CCCCCC",
        figsize=(18, 18),
        dynamic_label_size=True,
        dynamic_label_size_scaling_factor=0.85,
        label_font_size=10,
        label_wrap_width=15,
        label_margin_factor=1.5,
        arrowprops={"arrowstyle": "-", "color": "#333333"},
    )
    fig.suptitle(f"{DATASET_NAME}: MOSAIC Topic Map", fontsize=16, y=0.99)
    topic_map_path = PLOTS_DIR / "topic_map.png"
    fig.savefig(topic_map_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Topic map  →  {topic_map_path}")
except Exception as e:
    log.warning(f"datamapplot failed: {e}")

# 2. Bar chart — topic sizes
info_no_outliers = topic_info[topic_info["Topic"] != -1].copy()
info_no_outliers["Label"] = info_no_outliers["Topic"].map(
    lambda t: labels.get(t, name_map.get(t, f"Topic {t}"))
)
info_no_outliers = info_no_outliers.sort_values("Count", ascending=True)

fig, ax = plt.subplots(figsize=(10, max(6, len(info_no_outliers) * 0.35)))
ax.barh(info_no_outliers["Label"], info_no_outliers["Count"], color="#4C72B0")
ax.set_xlabel("Number of documents")
ax.set_title(f"{DATASET_NAME}: Topic sizes ({n_topics} topics, {100*n_outliers/len(topics):.1f}% outliers)")
plt.tight_layout()
bar_path = PLOTS_DIR / "topic_sizes.png"
fig.savefig(bar_path, dpi=200, bbox_inches="tight")
plt.close(fig)
log.info(f"Bar chart  →  {bar_path}")

# 3. Topic info CSV — topics + LLM labels + sizes
info_no_outliers["LLM_Label"] = info_no_outliers["Topic"].map(lambda t: labels.get(t, ""))
info_no_outliers.to_csv(PLOTS_DIR / "topic_info.csv", index=False)
log.info(f"Topic info →  {PLOTS_DIR / 'topic_info.csv'}")

# 4. Interactive HTML — zoomable documents + topics scatter
try:
    import plotly.graph_objects as go
    import plotly.express as px

    topic_assignments = np.array(topics)
    unique_topic_ids = sorted(set(topic_assignments.tolist()))
    palette = px.colors.qualitative.Plotly + px.colors.qualitative.Set2 + px.colors.qualitative.Pastel
    llm_map = {**name_map, **{int(k): v for k, v in labels.items()}}

    doc_fig = go.Figure()
    for tid in unique_topic_ids:
        mask = np.where(topic_assignments == tid)[0]
        x, y = reduced_2d[mask, 0], reduced_2d[mask, 1]
        label = "Unlabelled" if tid == -1 else llm_map.get(tid, f"Topic {tid}")
        color = "#CCCCCC" if tid == -1 else palette[tid % len(palette)]
        size, opacity = (4, 0.4) if tid == -1 else (6, 0.75)
        hover = [f"<b>{label}</b><br><br>{docs[i][:300]}{'…' if len(docs[i]) > 300 else ''}" for i in mask]
        doc_fig.add_trace(go.Scattergl(
            x=x, y=y, mode="markers", name=label,
            marker=dict(color=color, size=size, opacity=opacity),
            text=hover, hoverinfo="text",
        ))

    doc_fig.update_layout(
        title=dict(text=f"<b>{DATASET_NAME}: Documents and Topics</b>", x=0.5, xanchor="center"),
        template="simple_white",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        height=900, margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(title="Topics", bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#dddddd", borderwidth=1, font=dict(size=10)),
    )
    html_path = PLOTS_DIR / "topics_interactive.html"
    doc_fig.write_html(str(html_path))
    log.info(f"Interactive HTML →  {html_path}")
except Exception as e:
    log.warning(f"Interactive HTML failed: {e}")

log.info(f"Done. All outputs saved to {PLOTS_DIR}")
log.info(f"To copy to your Mac: scp -r rb666@artemis:{PLOTS_DIR} ~/Desktop/")
