"""
Regenerate MPE plots from an edited topic_info.csv — no GPU, no model needed.

Usage:
    python replot_MPE.py
    python replot_MPE.py --topic-info data/MPE/output/outputs_MPE/topic_info.csv
                         --output-dir data/MPE/output/outputs_MPE

Requires these files copied from Artemis:
    data/MPE/preprocessed/cache/topics.json
    data/MPE/preprocessed/cache/reduced_2d.npy
    data/MPE/preprocessed/cache/*_docs.json
    data/MPE/preprocessed/cache/topic_model/   (only for hierarchy plot)
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--topic-info", default="data/MPE/output/outputs_MPE/topic_info.csv")
parser.add_argument("--output-dir", default="data/MPE/output/outputs_MPE")
parser.add_argument("--cache-dir",  default="data/MPE/preprocessed/cache")
parser.add_argument("--csv",        default="data/MPE/preprocessed/MPE_dataset_translated_batched.csv",
                    help="Original CSV to count number of reports")
args = parser.parse_args()

DATASET_NAME = "MPE"
PLOTS_DIR    = Path(args.output_dir)
CACHE_DIR    = Path(args.cache_dir)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Load edited labels from CSV ───────────────────────────────────────────────
topic_info = pd.read_csv(args.topic_info)
labels = dict(zip(topic_info["Topic"].astype(int), topic_info["LLM_Label"].astype(str)))
name_map = dict(zip(topic_info["Topic"].astype(int), topic_info["Name"].astype(str)))

# ── Load cached data ──────────────────────────────────────────────────────────
docs_files = sorted(CACHE_DIR.glob("*_docs.json"))
if not docs_files:
    raise FileNotFoundError(f"No *_docs.json found in {CACHE_DIR}. Copy it from Artemis first.")

with open(docs_files[0], encoding="utf-8") as f:
    docs = json.load(f)

with open(CACHE_DIR / "topics.json") as f:
    topics = json.load(f)

reduced_2d = np.load(CACHE_DIR / "reduced_2d.npy")

n_topics    = len(set(t for t in topics if t != -1))
n_outliers  = sum(1 for t in topics if t == -1)
n_sentences = len(docs)
n_reports   = len(pd.read_csv(args.csv)) if Path(args.csv).exists() else None
print(f"Topics: {n_topics}  |  Sentences: {n_sentences}  |  Reports: {n_reports}  |  Outliers: {n_outliers} ({100*n_outliers/len(topics):.1f}%)")

# ── Build per-doc label list ──────────────────────────────────────────────────
doc_labels = [
    labels.get(t, name_map.get(t, "Unlabelled")) if t != -1 else "Unlabelled"
    for t in topics
]

# ── 1. Datamapplot ────────────────────────────────────────────────────────────
try:
    import datamapplot
    fig, _ = datamapplot.create_plot(
        reduced_2d,
        doc_labels,
        noise_label="Unlabelled",
        noise_color="#CCCCCC",
        figsize=(18, 18),
        dynamic_label_size=True,
        dynamic_label_size_scaling_factor=1.2,
        label_font_size=16,
        label_wrap_width=12,
        label_margin_factor=1.5,
        arrowprops={"arrowstyle": "-", "color": "#333333"},
    )
    subtitle = f"N = {n_sentences} sentences"
    if n_reports:
        subtitle += f", {n_reports} reports"
    fig.suptitle(f"{DATASET_NAME}: MOSAIC Topic Map ({subtitle})", fontsize=20, y=0.99)
    fig.savefig(PLOTS_DIR / "topic_map.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"topic_map.png  →  {PLOTS_DIR}")
except Exception as e:
    print(f"datamapplot failed: {e}")

# ── 2. Bar chart ──────────────────────────────────────────────────────────────
import matplotlib.cm as cm
import matplotlib.colors as mcolors

info_no_outliers = topic_info[topic_info["Topic"] != -1].copy()
info_no_outliers["Label"] = info_no_outliers["Topic"].map(
    lambda t: labels.get(t, name_map.get(t, f"Topic {t}"))
)
info_no_outliers = info_no_outliers.sort_values("Count", ascending=True)

counts = info_no_outliers["Count"].values
norm   = mcolors.LogNorm(vmin=counts.min(), vmax=counts.max())
cmap   = cm.get_cmap("Purples")
colors = [cmap(0.35 + 0.65 * norm(c)) for c in counts]  # log scale avoids one outlier dominating

fig, ax = plt.subplots(figsize=(10, max(6, len(info_no_outliers) * 0.38)))
bars = ax.barh(info_no_outliers["Label"], counts, color=colors, edgecolor="white", linewidth=0.4)

ax.set_xlabel("Number of sentences", fontsize=11)
ax.set_title(f"{DATASET_NAME}: Topic sizes ({n_topics} topics)", fontsize=13, pad=12)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="y", labelsize=9)

# Colourbar legend
sm = cm.ScalarMappable(cmap=cmap, norm=mcolors.LogNorm(vmin=counts.min(), vmax=counts.max()))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.02)
cbar.set_label("Sentences per topic", fontsize=9)

plt.tight_layout()
fig.savefig(PLOTS_DIR / "topic_sizes.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"topic_sizes.png  →  {PLOTS_DIR}")

# ── 3. Updated topic_info CSV ─────────────────────────────────────────────────
topic_info.to_csv(PLOTS_DIR / "topic_info.csv", index=False)
print(f"topic_info.csv  →  {PLOTS_DIR}")

# ── 4. Interactive HTML ───────────────────────────────────────────────────────
try:
    import plotly.graph_objects as go
    import plotly.express as px

    topic_assignments = np.array(topics)
    unique_topic_ids  = sorted(set(topic_assignments.tolist()))
    palette = px.colors.qualitative.Plotly + px.colors.qualitative.Set2 + px.colors.qualitative.Pastel
    llm_map = {**name_map, **labels}

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
        title=dict(text=f"<b>{DATASET_NAME}: Documents and Topics</b>", x=0.5, xanchor="center"),
        template="simple_white",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        height=900, margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(title="Topics", bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#dddddd", borderwidth=1, font=dict(size=10)),
    )
    doc_fig.write_html(str(PLOTS_DIR / "topics_interactive.html"))
    print(f"topics_interactive.html  →  {PLOTS_DIR}")
except Exception as e:
    print(f"Interactive HTML failed: {e}")

# ── 5. Hierarchy plot — computed directly from keywords, no model needed ─────
try:
    import ast
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    from scipy.cluster.hierarchy import linkage
    import plotly.figure_factory as ff

    info_hier = topic_info[topic_info["Topic"] != -1].copy()
    hier_labels = [labels.get(int(t), f"Topic {t}") for t in info_hier["Topic"]]

    def _parse_repr(r):
        try:
            return " ".join(ast.literal_eval(r))
        except Exception:
            return str(r)

    keyword_docs = [_parse_repr(r) for r in info_hier["Representation"]]

    X = normalize(TfidfVectorizer().fit_transform(keyword_docs).toarray())
    Z = linkage(X, method="ward", metric="euclidean")

    hier_fig = ff.create_dendrogram(
        X,
        labels=hier_labels,
        linkagefun=lambda x: Z,
        orientation="left",
    )
    hier_fig.update_layout(
        title=dict(text=f"<b>{DATASET_NAME}: Topic Hierarchy</b>", x=0.5, xanchor="center"),
        height=max(600, len(hier_labels) * 22),
        margin=dict(l=320, r=50, t=60, b=50),
        template="simple_white",
    )
    hier_fig.write_html(str(PLOTS_DIR / "hierarchy.html"))
    print(f"hierarchy.html  →  {PLOTS_DIR}")
    try:
        hier_fig.write_image(str(PLOTS_DIR / "hierarchy.png"), width=1400,
                             height=max(600, len(hier_labels) * 22), scale=2)
        print(f"hierarchy.png   →  {PLOTS_DIR}")
    except Exception as _e:
        print(f"hierarchy.png skipped (install kaleido: pip install kaleido): {_e}")
except Exception as e:
    print(f"Hierarchy plot failed: {e}")

print(f"\nDone. All outputs updated in {PLOTS_DIR}")
