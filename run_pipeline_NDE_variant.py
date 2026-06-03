"""
NDE pipeline variant — same embeddings as run_pipeline.py, different UMAP/HDBSCAN params.
Topic model and outputs are stored separately so the main NDE cache is never touched.

Usage:
    sbatch run_pipeline_NDE_t1.sh      # trial 1/73  — 78 topics
    sbatch run_pipeline_NDE_t70.sh     # trial 70    — 74 topics
    sbatch run_pipeline_NDE_t87.sh     # trial 87    — 54 topics
"""

import argparse
import json
import logging
import re
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--mcs",        type=int, required=True, help="HDBSCAN min_cluster_size")
parser.add_argument("--ms",         type=int, required=True, help="HDBSCAN min_samples")
parser.add_argument("--nn",         type=int, required=True, help="UMAP n_neighbors")
parser.add_argument("--nc",         type=int, default=10,    help="UMAP n_components (default 10)")
parser.add_argument("--tag",        type=str, required=True, help="Short label for outputs (e.g. t87)")
parser.add_argument("--base-tag",   type=str, default=None,
                    help="Tag of the topic model to load (default: same as --tag). "
                         "Use when running outlier reduction on an existing model, "
                         "e.g. --tag t97_ro50 --base-tag t97")
parser.add_argument("--debug",           action="store_true")
parser.add_argument("--llm-model",       type=str, default="meta-llama/Llama-3.1-8B-Instruct")
parser.add_argument("--nr-repr-docs",    type=int, default=7)
parser.add_argument("--reduce-outliers", action="store_true",
                    help="Reassign outlier sentences to nearest topic using embedding similarity")
parser.add_argument("--outlier-threshold", type=float, default=0.5,
                    help="Min cosine similarity to reassign an outlier (default 0.5)")
args = parser.parse_args()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def _load_secret(key: str) -> str:
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

log_file = f"pipeline_NDE_{args.tag}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
)
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).parent
DATASET_NAME    = "NDE"
CSV_PATH        = "data/NDE/preprocessed/NDE_reports_grouped.csv"
TEXT_COL        = "cleaned_report"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"
DEVICE          = "cuda"
SPLIT_SENTENCES = True
MIN_WORDS       = 3
HF_TOKEN        = _load_secret("HF_TOKEN")

CONFIG = {
    "umap_params": {
        "n_neighbors": args.nn,
        "n_components": args.nc,
        "min_dist": 0.0,
    },
    "hdbscan_params": {
        "min_cluster_size": args.mcs,
        "min_samples":      args.ms,
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

log.info(f"Variant tag: {args.tag}  |  mcs={args.mcs}  ms={args.ms}  nn={args.nn}")

# ── Paths — reuse shared NDE embeddings cache ─────────────────────────────────
CACHE_DIR = get_cache_dir(PROJECT_ROOT, DATASET_NAME)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DOCS_FILE, EMBEDDINGS_FILE = get_precomputed_filenames(
    CACHE_DIR, CSV_PATH, EMBEDDING_MODEL, SPLIT_SENTENCES, TEXT_COL, MIN_WORDS
)

# Variant-specific paths (never overwrites the main NDE topic model)
config_hash = get_config_hash(CONFIG)
_load_tag         = args.base_tag if args.base_tag else args.tag
_topic_model_path = CACHE_DIR / f"topic_model_{_load_tag}"
_topics_path      = CACHE_DIR / f"topics_{_load_tag}.json"
_reduced_path     = CACHE_DIR / f"reduced_2d_{_load_tag}.npy"
# Save outputs under the full tag (separate from the base model)
_topics_save_path = CACHE_DIR / f"topics_{args.tag}.json"
_reduced_save_path = CACHE_DIR / f"reduced_2d_{args.tag}.npy"

# ── Step 1: Load cached embeddings (never re-embeds) ─────────────────────────
if not Path(DOCS_FILE).exists() or not Path(EMBEDDINGS_FILE).exists():
    raise FileNotFoundError(
        "Embeddings not found — run run_pipeline.py (or run_embed.py) first to generate them."
    )

log.info("Step 1 — loading cached docs and embeddings")
with open(DOCS_FILE, encoding="utf-8") as f:
    docs = json.load(f)
embeddings = np.load(EMBEDDINGS_FILE)
log.info(f"Loaded {len(docs)} sentences  →  shape {embeddings.shape}")

# ── Step 2: Topic modelling ───────────────────────────────────────────────────
if _topic_model_path.exists() and Path(_topics_path).exists() and Path(_reduced_path).exists():
    log.info("Step 2 — loading cached topic model for this variant")
    from bertopic import BERTopic
    topic_model = BERTopic.load(str(_topic_model_path))
    with open(_topics_path) as f:
        topics = json.load(f)
    reduced_2d = np.load(_reduced_path)
else:
    log.info("Step 2 — topic modelling")
    topic_model, reduced_2d, topics = run_topic_model(docs, embeddings, CONFIG)
    topic_model.save(str(CACHE_DIR / f"topic_model_{args.tag}"))
    np.save(_reduced_save_path, reduced_2d)
    with open(_topics_save_path, "w") as f:
        json.dump(topics, f)

n_topics   = len(set(t for t in topics if t != -1))
n_outliers = sum(1 for t in topics if t == -1)
log.info(f"Topics: {n_topics}  |  Outliers: {n_outliers} ({100*n_outliers/len(topics):.1f}%)")

# ── Outlier reduction ─────────────────────────────────────────────────────────
if args.reduce_outliers:
    log.info(f"Reducing outliers — strategy=embeddings  threshold={args.outlier_threshold}")
    new_topics = topic_model.reduce_outliers(
        docs, topics,
        strategy="embeddings",
        embeddings=embeddings,
        threshold=args.outlier_threshold,
    )
    topic_model.update_topics(docs, topics=new_topics)
    topics = [int(t) for t in new_topics]  # convert numpy int64 → Python int
    n_outliers_new = sum(1 for t in topics if t == -1)
    log.info(f"After reduction — Outliers: {n_outliers_new} ({100*n_outliers_new/len(topics):.1f}%)  "
             f"(was {n_outliers}, reduced by {n_outliers - n_outliers_new})")
    # Save updated topics under the output tag
    with open(_topics_save_path, "w") as f:
        json.dump(topics, f)

# Re-extract representative docs
import pandas as _pd
_documents_df = _pd.DataFrame({"Document": docs, "Topic": topics})
_repr_docs, _, _, _ = topic_model._extract_representative_docs(
    topic_model.c_tf_idf_,
    _documents_df,
    topic_model.topic_representations_,
    nr_samples=500,
    nr_repr_docs=args.nr_repr_docs,
)
topic_model.representative_docs_ = _repr_docs
log.info(f"Representative docs per topic: {args.nr_repr_docs}")

if args.debug:
    log.info("Debug mode — skipping LLM labelling and plots")
    import sys; sys.exit(0)

# ── Step 3: LLM labelling ─────────────────────────────────────────────────────
log.info("Step 3 — LLM labelling")
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

llm_cache = labels_cache_path(CACHE_DIR, config_hash, LLM_MODEL)
llm_cache.write_text(
    json.dumps({str(k): v for k, v in labels.items()}, indent=2),
    encoding="utf-8",
)
log.info(f"Labelled {len(labels)} topics  →  {llm_cache}")

# ── Step 4: Save plots ────────────────────────────────────────────────────────
log.info("Step 4 — generating plots")

PLOTS_DIR = PROJECT_ROOT / f"outputs_NDE_{args.tag}"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

topic_info = topic_model.get_topic_info()
name_map   = topic_info.set_index("Topic")["Name"].to_dict()
doc_labels = [
    labels.get(t, name_map.get(t, "Unlabelled")) if t != -1 else "Unlabelled"
    for t in topics
]

# 1. Datamapplot
try:
    import datamapplot
    fig, _ = datamapplot.create_plot(
        reduced_2d, doc_labels,
        noise_label="Unlabelled", noise_color="#CCCCCC",
        figsize=(18, 18), dynamic_label_size=True,
        dynamic_label_size_scaling_factor=0.85,
        label_font_size=10, label_wrap_width=15, label_margin_factor=1.5,
        arrowprops={"arrowstyle": "-", "color": "#333333"},
    )
    fig.suptitle(f"NDE [{args.tag}]: MOSAIC Topic Map", fontsize=16, y=0.99)
    fig.savefig(PLOTS_DIR / "topic_map.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"topic_map.png  →  {PLOTS_DIR}")
except Exception as e:
    log.warning(f"datamapplot failed: {e}")

# 2. Bar chart
info_no_outliers = topic_info[topic_info["Topic"] != -1].copy()
info_no_outliers["Label"] = info_no_outliers["Topic"].map(
    lambda t: labels.get(t, name_map.get(t, f"Topic {t}"))
)
info_no_outliers = info_no_outliers.sort_values("Count", ascending=True)
fig, ax = plt.subplots(figsize=(10, max(6, len(info_no_outliers) * 0.35)))
ax.barh(info_no_outliers["Label"], info_no_outliers["Count"], color="#4C72B0")
ax.set_xlabel("Number of documents")
ax.set_title(f"NDE [{args.tag}]: Topic sizes ({n_topics} topics, {100*n_outliers/len(topics):.1f}% outliers)")
plt.tight_layout()
fig.savefig(PLOTS_DIR / "topic_sizes.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# 3. Topic info CSV
info_no_outliers["LLM_Label"] = info_no_outliers["Topic"].map(lambda t: labels.get(t, ""))
cols = ["Topic", "LLM_Label"] + [c for c in info_no_outliers.columns if c not in ("Topic", "LLM_Label")]
info_no_outliers[cols].to_csv(PLOTS_DIR / "topic_info.csv", index=False)

# 3b. Topics summary — one row per topic (for condition comparison in app2.py)
from collections import defaultdict as _dd
import pandas as _pd2
llm_map_full = {**name_map, **{int(k): v for k, v in labels.items()}}
_topic_sents = _dd(list)
for _t, _d in zip(topics, docs):
    if _t != -1:
        _topic_sents[_t].append(_d)
_rows = [{"topic_name": llm_map_full.get(_t, f"Topic {_t}"),
          "texts": " | ".join(_s), "n_sentences": len(_s)}
         for _t, _s in sorted(_topic_sents.items())]
_pd2.DataFrame(_rows).to_csv(PLOTS_DIR / "topics_sentences.csv", index=False)
log.info(f"Topics summary →  {PLOTS_DIR / 'topics_sentences.csv'}")

# 4. Interactive HTML
try:
    import plotly.graph_objects as go
    import plotly.express as px
    topic_assignments = np.array(topics)
    unique_topic_ids  = sorted(set(topic_assignments.tolist()))
    palette = px.colors.qualitative.Plotly + px.colors.qualitative.Set2 + px.colors.qualitative.Pastel
    llm_map = {**name_map, **{int(k): v for k, v in labels.items()}}
    doc_fig = go.Figure()
    for tid in unique_topic_ids:
        mask  = np.where(topic_assignments == tid)[0]
        x, y  = reduced_2d[mask, 0], reduced_2d[mask, 1]
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
        title=dict(text=f"<b>NDE [{args.tag}]: Documents and Topics</b>", x=0.5, xanchor="center"),
        template="simple_white",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        height=900, margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(title="Topics", bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#dddddd", borderwidth=1, font=dict(size=10)),
    )
    doc_fig.write_html(str(PLOTS_DIR / "topics_interactive.html"))
    log.info(f"topics_interactive.html  →  {PLOTS_DIR}")
except Exception as e:
    log.warning(f"Interactive HTML failed: {e}")

log.info(f"Done. Outputs  →  {PLOTS_DIR}")
log.info(f"To copy: scp -r rb666@artemis:{PLOTS_DIR} ~/Desktop/")
