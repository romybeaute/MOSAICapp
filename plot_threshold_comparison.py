"""
Plot cosine similarity distribution for outlier sentences + threshold comparison table.
Justifies the choice of outlier reduction threshold.

Usage:
    python3 plot_threshold_comparison.py --dataset NDE
    python3 plot_threshold_comparison.py --dataset MPE
    python3 plot_threshold_comparison.py --dataset NDE MPE
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", nargs="+", choices=["NDE", "MPE"], default=["NDE", "MPE"])
parser.add_argument("--output-dir", default="data/figures")
args = parser.parse_args()

Path(args.output_dir).mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.6, 0.7, 0.8]

DATASETS = {
    "NDE": {
        "topics_file":     "data/NDE/preprocessed/cache/topics_t97.json",
        "embeddings_glob": "data/NDE/preprocessed/cache/*Qwen3-Embedding-4B*embeddings.npy",
        "color":           "#4c72b0",
    },
    "MPE": {
        "topics_file":     "data/MPE/preprocessed/cache/topics.json",
        "embeddings_glob": "data/MPE/preprocessed/cache/*Qwen3-Embedding-4B*embeddings.npy",
        "color":           "#dd8452",
    },
}

results = {}

for ds_name in args.dataset:
    cfg  = DATASETS[ds_name]
    embs = sorted(Path(".").glob(cfg["embeddings_glob"]))
    if not embs:
        print(f"Embeddings not found for {ds_name} — skipping. Copy from Artemis first.")
        continue

    print(f"Loading {ds_name}...")
    with open(cfg["topics_file"]) as f:
        topics = json.load(f)
    embeddings = np.load(embs[-1]).astype(np.float32)

    topic_ids   = sorted(set(t for t in topics if t != -1))
    ta          = np.array(topics)
    centroids   = np.vstack([embeddings[ta == tid].mean(axis=0) for tid in topic_ids])
    outlier_idx = np.where(ta == -1)[0]
    n_total     = len(topics)
    n_out_orig  = len(outlier_idx)

    sims      = cosine_similarity(embeddings[outlier_idx], centroids)
    best_sims = sims.max(axis=1)

    thresh_stats = {}
    for t in THRESHOLDS:
        remaining        = int((best_sims < t).sum())
        thresh_stats[t]  = {"remaining": remaining, "pct": 100 * remaining / n_total}

    results[ds_name] = {
        "best_sims":  best_sims,
        "n_total":    n_total,
        "n_out_orig": n_out_orig,
        "pct_orig":   100 * n_out_orig / n_total,
        "thresh_stats": thresh_stats,
        "color":      cfg["color"],
    }
    print(f"  {ds_name}: {n_total} sentences, {n_out_orig} outliers ({100*n_out_orig/n_total:.1f}%)")

if not results:
    raise SystemExit("No data — check embeddings paths.")

# ── Plot 1: similarity distribution ──────────────────────────────────────────
n_ds = len(results)
fig, axes = plt.subplots(1, n_ds, figsize=(6 * n_ds, 5))
if n_ds == 1:
    axes = [axes]

thresh_colors = ["#e07b54", "#4c72b0", "#55a868"]

for ax, (ds_name, r) in zip(axes, results.items()):
    ax.hist(r["best_sims"], bins=60, color=r["color"], alpha=0.75,
            edgecolor="white", linewidth=0.3)
    for t, tc in zip(THRESHOLDS, thresh_colors):
        ax.axvline(t, color=tc, linewidth=2.0, linestyle="--",
                   label=f"τ = {t}  →  {r['thresh_stats'][t]['pct']:.1f}% outliers remain")
    ax.set_xlabel("Max cosine similarity to nearest topic centroid", fontsize=11)
    ax.set_ylabel("Number of outlier sentences", fontsize=11)
    ax.set_title(f"{ds_name}: Outlier sentence similarity distribution\n"
                 f"(original outlier rate: {r['pct_orig']:.1f}%)", fontsize=12)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
p1 = Path(args.output_dir) / "outlier_similarity_distribution.png"
fig.savefig(p1, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"\nDistribution plot  →  {p1}")

# ── Plot 2: comparison table ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.axis("off")

col_labels = ["Dataset", "Original %"] + [f"τ = {t}" for t in THRESHOLDS]
rows = []
for ds_name, r in results.items():
    row = [ds_name, f"{r['pct_orig']:.1f}%"]
    for t in THRESHOLDS:
        row.append(f"{r['thresh_stats'][t]['pct']:.1f}%")
    rows.append(row)

tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1.4, 2.2)

# Highlight selected threshold (τ=0.7, column index 3)
for row_idx in range(len(rows) + 1):
    tbl[row_idx, 3].set_facecolor("#d4e6f7")

# Header row styling
for col_idx in range(len(col_labels)):
    tbl[0, col_idx].set_facecolor("#2c3e50")
    tbl[0, col_idx].set_text_props(color="white", fontweight="bold")

ax.set_title("Outlier reduction threshold comparison (selected: τ = 0.7)",
             fontsize=12, pad=20, fontweight="bold")
plt.tight_layout()
p2 = Path(args.output_dir) / "outlier_threshold_comparison.png"
fig.savefig(p2, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Threshold table    →  {p2}")

# ── Console summary ───────────────────────────────────────────────────────────
print("\n── Summary ──────────────────────────────────────────────────────")
for ds_name, r in results.items():
    print(f"\n{ds_name}  (original: {r['pct_orig']:.1f}% outliers)")
    for t in THRESHOLDS:
        marker = "  ◄ selected" if t == 0.7 else ""
        print(f"  τ = {t}:  {r['thresh_stats'][t]['pct']:.1f}% outliers remain{marker}")
