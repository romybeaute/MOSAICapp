"""
run_zeroshot_NDE.py
-------------------
Zero-shot topic classification for the NDE dataset using the 16-item Greyson NDE Scale.

Reuses precomputed Qwen/Qwen3-Embedding-4B embeddings — no re-embedding of documents.
The model is only used to encode the 16 short category strings.

RECOMMENDED WORKFLOW
--------------------
1. Run --diagnose first to see the similarity distribution and choose a threshold:

    sbatch --job-name=NDE_zs_diag \\
           --partition=general --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=00:30:00 \\
           --output=pipeline_NDE_zs_diag_%j.log \\
           --wrap="source .venv/bin/activate && module load CUDA/12.1.1 && PYTHONUNBUFFERED=1 \\
           python run_zeroshot_NDE.py --diagnose"

   Inspect outputs_NDE_zeroshot/diagnose_similarity_dist.png and the logged percentiles.
   Pick the threshold at the elbow / the percentile matching your expected coverage.

2. Then run the full classification with your chosen threshold:

    sbatch --job-name=NDE_zeroshot \\
           --partition=general --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:00:00 \\
           --output=pipeline_NDE_zeroshot_%j.log \\
           --wrap="source .venv/bin/activate && module load CUDA/12.1.1 && PYTHONUNBUFFERED=1 \\
           python run_zeroshot_NDE.py --min-similarity 0.45 --from-variant t97_ro"
"""

import argparse
import json
import logging
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--min-similarity", type=float, default=0.50,
                    help="Cosine similarity threshold; docs below this are 'Unclassified' (default 0.50)")
parser.add_argument("--from-variant", type=str, default=None,
                    help="Tag of a previously run NDE variant (e.g. t97_ro) whose 2D reduction "
                         "is reused for the scatter plot. Falls back to any available reduced_2d_*.npy.")
parser.add_argument("--diagnose", action="store_true",
                    help="Plot the max-similarity distribution across all documents then exit. "
                         "Use this to choose --min-similarity before running the full classification.")
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
    get_cache_dir,
    get_precomputed_filenames,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline_NDE_zeroshot.log"),
    ],
)
log = logging.getLogger(__name__)

# ── NDE dataset config (must match run_pipeline.py / run_pipeline_NDE_variant.py) ──
PROJECT_ROOT    = Path(__file__).parent
DATASET_NAME    = "NDE"
CSV_PATH        = "data/NDE/preprocessed/NDE_reports_grouped.csv"
TEXT_COL        = "cleaned_report"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"
SPLIT_SENTENCES = True
MIN_WORDS       = 3

# ── Greyson NDE Scale — 16 categories ──────────────────────────────────────────
CATEGORIES = [
    "Did time seem to speed up?",
    "Were your thoughts speeded up?",
    "Did scenes from your past come back to you?",
    "Did you suddenly seem to understand everything?",
    "Did you have a feeling of peace or pleasantness?",
    "Did you have a feeling of joy?",
    "Did you feel a sense of harmony or unity with the universe?",
    "Did you see or feel surrounded by a brilliant light?",
    "Were your senses more vivid than usual?",
    "Did you seem to be aware of things going on elsewhere, as if by Extrasensory Perception?",
    "Did scenes from the future come to you?",
    "Did you feel separated from your physical body?",
    "Did you seem to enter some other, unearthly world?",
    "Did you seem to encounter a mystical being or presence?",
    "Did you see deceased spirits or religious figures?",
    "Did you come to a border or point of no return?",
]

log.info(f"Zero-shot  |  categories={len(CATEGORIES)}  min_similarity={args.min_similarity}  diagnose={args.diagnose}")

# ── Paths ───────────────────────────────────────────────────────────────────────
CACHE_DIR = get_cache_dir(PROJECT_ROOT, DATASET_NAME)
DOCS_FILE, EMBEDDINGS_FILE = get_precomputed_filenames(
    CACHE_DIR, CSV_PATH, EMBEDDING_MODEL, SPLIT_SENTENCES, TEXT_COL, MIN_WORDS
)
OUTPUTS_DIR = PROJECT_ROOT / "outputs_NDE_zeroshot"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Step 1: Load precomputed docs and embeddings ────────────────────────────────
if not Path(DOCS_FILE).exists() or not Path(EMBEDDINGS_FILE).exists():
    raise FileNotFoundError(
        f"Embeddings not found at:\n  {EMBEDDINGS_FILE}\n"
        "Run run_pipeline.py (or run_embed.py) first to generate them."
    )

log.info("Step 1 — loading cached docs and embeddings")
with open(DOCS_FILE, encoding="utf-8") as f:
    docs = json.load(f)
embeddings = np.load(EMBEDDINGS_FILE).astype(np.float32)
log.info(f"Loaded {len(docs)} sentences  →  shape {embeddings.shape}")

# ── Step 2: Zero-shot classification ───────────────────────────────────────────
log.info("Step 2 — zero-shot classification")

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic

HF_TOKEN = _load_secret("HF_TOKEN")

log.info(f"Loading embedding model: {EMBEDDING_MODEL}")
emb_model = SentenceTransformer(EMBEDDING_MODEL, token=HF_TOKEN, device="cuda")

# ── Diagnostic mode: show similarity distribution and exit ─────────────────────
if args.diagnose:
    from sklearn.metrics.pairwise import cosine_similarity as _cosine_sim

    log.info("DIAGNOSE MODE — encoding category labels")
    cat_embeddings = emb_model.encode(CATEGORIES, convert_to_numpy=True, show_progress_bar=False)
    cat_embeddings = cat_embeddings.astype(np.float32)

    log.info("Computing cosine similarity for all documents (this may take a moment)")
    # Process in batches to avoid OOM
    BATCH = 4096
    max_sims = []
    argmax_cats = []
    for start in range(0, len(embeddings), BATCH):
        batch = embeddings[start : start + BATCH]
        sim = _cosine_sim(batch, cat_embeddings)          # (batch, 16)
        max_sims.extend(sim.max(axis=1).tolist())
        argmax_cats.extend(sim.argmax(axis=1).tolist())

    max_sims = np.array(max_sims)

    # ── Percentile report ──
    pcts = [10, 25, 50, 75, 90, 95, 99]
    log.info("Max-similarity percentiles across all %d sentences:", len(max_sims))
    for p in pcts:
        log.info("  p%02d : %.4f", p, float(np.percentile(max_sims, p)))

    # ── Per-category stats (ignoring threshold — every doc assigned to nearest) ──
    import pandas as _pd2
    diag_df = _pd2.DataFrame({
        "sentence":    docs,
        "best_cat":    [CATEGORIES[i] for i in argmax_cats],
        "max_sim":     max_sims,
    })
    for thresh in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65):
        classified = (max_sims >= thresh).sum()
        log.info("  thresh=%.2f → classified %d / %d  (%.1f%%)",
                 thresh, classified, len(max_sims), 100 * classified / len(max_sims))

    # ── Histogram ──
    fig_d, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax_hist = axes[0]
    ax_hist.hist(max_sims, bins=80, color="#5B7BE8", edgecolor="white", linewidth=0.4)
    ax_hist.set_xlabel("Max cosine similarity (best matching category)")
    ax_hist.set_ylabel("Number of sentences")
    ax_hist.set_title("Distribution of per-sentence max similarity\n(all docs, no threshold)")
    for thresh_line in (0.40, 0.45, 0.50, 0.55, 0.60):
        ax_hist.axvline(thresh_line, color="red", linewidth=1.0, linestyle="--", alpha=0.6,
                        label=f"{thresh_line}")
    ax_hist.legend(title="Candidate thresholds", fontsize=8)

    # Sorted curve (elbow plot)
    ax_elbow = axes[1]
    sorted_sims = np.sort(max_sims)[::-1]
    ax_elbow.plot(sorted_sims, color="#5B7BE8", linewidth=1.2)
    ax_elbow.set_xlabel("Documents (ranked by similarity)")
    ax_elbow.set_ylabel("Max cosine similarity")
    ax_elbow.set_title("Elbow plot — choose threshold at the 'knee'")
    for thresh_line in (0.40, 0.45, 0.50, 0.55, 0.60):
        ax_elbow.axhline(thresh_line, color="red", linewidth=1.0, linestyle="--", alpha=0.6,
                         label=f"{thresh_line}")
    ax_elbow.legend(title="Candidate thresholds", fontsize=8)

    plt.tight_layout()
    diag_path = OUTPUTS_DIR / "diagnose_similarity_dist.png"
    fig_d.savefig(diag_path, dpi=200, bbox_inches="tight")
    plt.close(fig_d)
    log.info(f"Diagnostic plot  →  {diag_path}")

    # Save per-sentence similarity CSV for manual inspection
    diag_csv = OUTPUTS_DIR / "diagnose_sentences.csv"
    diag_df.to_csv(diag_csv, index=False, encoding="utf-8-sig")
    log.info(f"Diagnostic CSV   →  {diag_csv}")
    log.info("DIAGNOSE MODE done — re-run without --diagnose once you have chosen --min-similarity")
    import sys; sys.exit(0)


class _ZSPassThrough:
    """Dummy UMAP — passes pre-computed embeddings unchanged."""
    def fit(self, X, y=None): return self
    def transform(self, X): return X


class _ZSDummyClustering:
    """Dummy HDBSCAN — marks every doc as outlier so zero-shot handles assignment."""
    def __init__(self): self.labels_ = None
    def fit(self, X, y=None):
        self.labels_ = np.array([-1] * len(X))
        return self


topic_model = BERTopic(
    embedding_model=emb_model,
    umap_model=_ZSPassThrough(),
    hdbscan_model=_ZSDummyClustering(),
    vectorizer_model=CountVectorizer(stop_words="english"),
    zeroshot_topic_list=CATEGORIES,
    zeroshot_min_similarity=args.min_similarity,
    verbose=True,
)

topics, _ = topic_model.fit_transform(docs, embeddings)
topic_info = topic_model.get_topic_info()
name_map = topic_info.set_index("Topic")["Name"].to_dict()
log.info(f"Zero-shot complete")

n_classified   = sum(1 for t in topics if t != -1)
n_unclassified = sum(1 for t in topics if t == -1)
log.info(f"Classified: {n_classified} ({100*n_classified/len(topics):.1f}%)  "
         f"Unclassified: {n_unclassified} ({100*n_unclassified/len(topics):.1f}%)")

# ── Step 3: Save sentence-level CSV ────────────────────────────────────────────
log.info("Step 3 — saving sentence-level results")

import pandas as pd

results_df = pd.DataFrame({
    "sentence": docs,
    "topic_id": topics,
    "category": [name_map.get(t, "Unclassified") for t in topics],
})
results_path = OUTPUTS_DIR / "zeroshot_sentences.csv"
results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
log.info(f"Sentence results  →  {results_path}")

# ── Step 4: Bar chart ───────────────────────────────────────────────────────────
log.info("Step 4 — bar chart")

plot_df = (
    topic_info[topic_info["Topic"] != -1]
    .sort_values("Count", ascending=True)
    .reset_index(drop=True)
)

if not plot_df.empty:
    fig, ax = plt.subplots(figsize=(10, max(5, len(plot_df) * 0.55)))
    ax.barh(plot_df["Name"], plot_df["Count"], color="#5B7BE8")
    ax.set_xlabel("Number of sentences")
    ax.set_title(
        f"NDE Zero-Shot: Greyson Scale categories\n"
        f"({n_classified:,} classified, {n_unclassified:,} unclassified, "
        f"threshold={args.min_similarity})",
        fontsize=11,
    )
    plt.tight_layout()
    bar_path = OUTPUTS_DIR / "zeroshot_barchart.png"
    fig.savefig(bar_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Bar chart  →  {bar_path}")

# ── Step 5: Topic-info CSV ──────────────────────────────────────────────────────
ti_path = OUTPUTS_DIR / "zeroshot_topic_info.csv"
topic_info.to_csv(ti_path, index=False)
log.info(f"Topic info  →  {ti_path}")

# ── Step 6: Interactive scatter (reuse variant's 2D reduction if available) ─────
log.info("Step 6 — interactive scatter plot")

reduced_2d = None

if args.from_variant:
    candidate = CACHE_DIR / f"reduced_2d_{args.from_variant}.npy"
    if candidate.exists():
        reduced_2d = np.load(candidate)
        log.info(f"Loaded 2D reduction from variant '{args.from_variant}'  →  {candidate}")
    else:
        log.warning(f"--from-variant '{args.from_variant}': file not found at {candidate}")

if reduced_2d is None:
    for p in sorted(CACHE_DIR.glob("reduced_2d_*.npy")):
        arr = np.load(p)
        if arr.shape[0] == len(docs):
            reduced_2d = arr
            log.info(f"Auto-found 2D reduction: {p.name}")
            break

if reduced_2d is not None and reduced_2d.shape[0] == len(docs):
    try:
        import plotly.graph_objects as go
        import plotly.express as px

        cat_labels = [name_map.get(t, "Unclassified") for t in topics]
        unique_cats = ["Unclassified"] + [c for c in sorted(set(cat_labels)) if c != "Unclassified"]
        palette = px.colors.qualitative.Plotly + px.colors.qualitative.Set2 + px.colors.qualitative.Pastel

        fig_html = go.Figure()
        for i, cat in enumerate(unique_cats):
            mask = [j for j, c in enumerate(cat_labels) if c == cat]
            color   = "#CCCCCC" if cat == "Unclassified" else palette[(i - 1) % len(palette)]
            opacity = 0.25 if cat == "Unclassified" else 0.75
            size    = 3    if cat == "Unclassified" else 5
            hover = [f"<b>{cat}</b><br><br>{docs[j][:300]}{'…' if len(docs[j]) > 300 else ''}"
                     for j in mask]
            fig_html.add_trace(go.Scattergl(
                x=reduced_2d[mask, 0], y=reduced_2d[mask, 1],
                mode="markers", name=cat,
                marker=dict(color=color, size=size, opacity=opacity),
                text=hover, hoverinfo="text",
            ))

        fig_html.update_layout(
            title=dict(
                text=f"<b>NDE Zero-Shot: Greyson Scale  "
                     f"(threshold={args.min_similarity})</b>",
                x=0.5, xanchor="center",
            ),
            template="simple_white",
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            height=900, margin=dict(l=10, r=10, t=60, b=10),
            legend=dict(
                title="Category", bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#dddddd", borderwidth=1, font=dict(size=10),
            ),
        )
        html_path = OUTPUTS_DIR / "zeroshot_interactive.html"
        fig_html.write_html(str(html_path))
        log.info(f"Interactive HTML  →  {html_path}")
    except Exception as e:
        log.warning(f"Interactive plot failed: {e}")
else:
    log.info("No matching 2D reduction found — skipping scatter plot. "
             "Run a variant first, then pass --from-variant <tag>.")

log.info(f"Done. Outputs  →  {OUTPUTS_DIR}")
log.info(f"To copy: scp -r rb666@artemis:{OUTPUTS_DIR} ~/Desktop/")
