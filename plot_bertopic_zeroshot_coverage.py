"""
Cross-reference BERTopic topics with Greyson Scale zero-shot coverage.

Shows which BERTopic topics are captured by the Greyson Scale (at a given threshold)
and which represent phenomenological content the Scale does not cover.

Usage:
    python plot_bertopic_zeroshot_coverage.py --threshold 0.60
"""

import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

parser = argparse.ArgumentParser()
parser.add_argument("--threshold", type=float, default=0.55)
parser.add_argument("--diag-csv",    default=str(Path.home() / "Desktop/diagnose_sentences.csv"))
parser.add_argument("--topics-json", default=str(Path.home() / "Desktop/topics_t97_ro.json"))
parser.add_argument("--topic-info",  default=str(Path.home() / "Desktop/topic_info.csv"))
args = parser.parse_args()

# ── Load data ──────────────────────────────────────────────────────────────────
diag    = pd.read_csv(args.diag_csv)
with open(args.topics_json) as f:
    bt_topics = json.load(f)
topic_info = pd.read_csv(args.topic_info)

assert len(diag) == len(bt_topics), (
    f"Mismatch: diagnose_sentences has {len(diag)} rows but topics_t97_ro has {len(bt_topics)} entries. "
    "Make sure both files come from the same NDE run."
)

# ── Build combined dataframe ───────────────────────────────────────────────────
df = diag.copy()
df["bt_topic"] = bt_topics

# BERTopic label map: prefer LLM_Label, fall back to Name when empty
llm_map  = topic_info.set_index("Topic")["LLM_Label"].to_dict() if "LLM_Label" in topic_info.columns else {}
name_map = topic_info.set_index("Topic")["Name"].to_dict()      if "Name"      in topic_info.columns else {}

def _best_label(tid):
    if tid == -1:
        return "Outlier"
    llm = str(llm_map.get(tid, "")).strip()
    if llm:
        return llm
    name = str(name_map.get(tid, "")).strip()
    return name if name else f"Topic {tid}"

label_map = {tid: _best_label(tid) for tid in set(bt_topics)}
df["bt_label"] = df["bt_topic"].map(_best_label)
df["captured"] = df["max_sim"] >= args.threshold

# ── Per-BERTopic-topic coverage stats ─────────────────────────────────────────
rows = []
for tid, grp in df[df["bt_topic"] != -1].groupby("bt_topic"):
    n_total     = len(grp)
    n_captured  = grp["captured"].sum()
    pct         = 100 * n_captured / n_total
    top_cat     = grp[grp["captured"]]["best_cat"].value_counts().idxmax() if n_captured > 0 else "—"
    rows.append({
        "bt_topic":   tid,
        "bt_label":   label_map.get(tid, f"Topic {tid}"),
        "n_sentences": n_total,
        "n_captured": int(n_captured),
        "pct_captured": round(pct, 1),
        "top_greyson_cat": top_cat,
    })

cov = pd.DataFrame(rows).sort_values("pct_captured", ascending=False)

# ── Print summary ──────────────────────────────────────────────────────────────
print(f"\nThreshold: {args.threshold}")
print(f"Total sentences (excl. BERTopic outliers): {len(df[df['bt_topic'] != -1]):,}")
print(f"Captured by Greyson Scale: {df[df['bt_topic'] != -1]['captured'].sum():,} "
      f"({100*df[df['bt_topic'] != -1]['captured'].mean():.1f}%)")
print(f"\nBERTopic topics: {len(cov)}")
print(f"  Fully captured (>50%): {(cov.pct_captured > 50).sum()}")
print(f"  Partially captured (10-50%): {((cov.pct_captured >= 10) & (cov.pct_captured <= 50)).sum()}")
print(f"  Not captured (<10%): {(cov.pct_captured < 10).sum()}")

print(f"\n{'─'*80}")
print("NOT CAPTURED by Greyson Scale (< 10% of sentences above threshold):")
print(f"{'─'*80}")
not_covered = cov[cov.pct_captured < 10].sort_values("n_sentences", ascending=False)
for _, r in not_covered.iterrows():
    print(f"  [{r.pct_captured:4.1f}%]  {r.n_sentences:4d} sentences  →  {r.bt_label}")

print(f"\n{'─'*80}")
print("WELL CAPTURED by Greyson Scale (> 50%):")
print(f"{'─'*80}")
well_covered = cov[cov.pct_captured > 50].sort_values("pct_captured", ascending=False)
for _, r in well_covered.iterrows():
    print(f"  [{r.pct_captured:4.1f}%]  {r.n_sentences:4d} sentences  →  {r.bt_label}  (→ {r.top_greyson_cat})")

# ── Save coverage table ────────────────────────────────────────────────────────
out_csv = Path.home() / "Desktop" / f"bertopic_greyson_coverage_{args.threshold}.csv"
cov.to_csv(out_csv, index=False)
print(f"\nFull coverage table → {out_csv}")

# ── Plot ───────────────────────────────────────────────────────────────────────
cov_plot = cov.sort_values("pct_captured", ascending=True)
n = len(cov_plot)

# Colour by coverage: red = not covered, green = well covered
norm  = mcolors.Normalize(vmin=0, vmax=100)
cmap  = plt.cm.RdYlGn
colors = [cmap(norm(v)) for v in cov_plot["pct_captured"]]

fig, ax = plt.subplots(figsize=(10, max(8, n * 0.28)))
bars = ax.barh(
    cov_plot["bt_label"], cov_plot["pct_captured"],
    color=colors, edgecolor="white", linewidth=0.4, height=0.8,
)
ax.axvline(10, color="#e74c3c", linewidth=1.2, linestyle="--", alpha=0.7, label="10% (not covered)")
ax.axvline(50, color="#27ae60", linewidth=1.2, linestyle="--", alpha=0.7, label="50% (well covered)")
ax.set_xlabel("% of topic sentences captured by Greyson Scale", fontsize=11)
ax.set_title(
    f"BERTopic topics × Greyson Scale coverage  (threshold={args.threshold})\n"
    f"Green = Scale captures this topic well  |  Red = Scale does not cover this topic",
    fontsize=11,
)
ax.tick_params(axis="y", labelsize=7.5)
ax.set_xlim(0, 105)
ax.legend(fontsize=9)

for bar, val in zip(bars, cov_plot["pct_captured"]):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f"{val:.0f}%", va="center", ha="left", fontsize=7, color="#333333")

plt.tight_layout()
out_png = Path.home() / "Desktop" / f"bertopic_greyson_coverage_{args.threshold}.png"
plt.savefig(out_png, dpi=200, bbox_inches="tight")
print(f"Coverage plot → {out_png}")
plt.show()
