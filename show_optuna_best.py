"""
Print the best Optuna trials in a clean table.

Usage:
    python show_optuna_best.py
    python show_optuna_best.py --dataset MPE
    python show_optuna_best.py --top 20 --max-outliers 50
"""

import argparse
import glob
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",      type=str, default="NDE")
parser.add_argument("--top",          type=int, default=10,  help="Number of results to show")
parser.add_argument("--max-outliers", type=float, default=55, help="Max outlier %% to include")
args = parser.parse_args()

pattern = f"outputs/optuna/OPTUNA_*{args.dataset}*results.csv"
files = sorted(glob.glob(pattern))
if not files:
    print(f"No results file found matching: {pattern}")
    raise SystemExit(1)

results_file = files[-1]
print(f"Reading: {results_file}\n")

df = pd.read_csv(results_file)

# Remove duplicate header rows (if multiple runs were appended)
df = df[df["trial"] != "trial"].copy()
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="ignore")

best = (
    df[df["topic_score"] == 1.0]
    .query(f"outlier_pct <= {args.max_outliers}")
    .sort_values("emb_coherence", ascending=False)
    .drop_duplicates(subset=["min_cluster_size", "min_samples", "n_neighbors"])
    .head(args.top)
    [["trial", "emb_coherence", "n_topics", "outlier_pct",
      "min_cluster_size", "min_samples", "n_neighbors"]]
    .reset_index(drop=True)
)

if best.empty:
    print(f"No trials with topic_score=1.0 and outlier_pct ≤ {args.max_outliers}%.")
    print("Try --max-outliers 60")
else:
    best.index += 1
    best.columns = ["trial", "emb_coh", "n_topics", "outliers%", "mcs", "ms", "nn"]
    best["emb_coh"] = best["emb_coh"].round(4)
    best["outliers%"] = best["outliers%"].round(1)
    print(best.to_string())
    print(f"\nBest: trial {int(best.iloc[0]['trial'])} — "
          f"mcs={int(best.iloc[0]['mcs'])}  ms={int(best.iloc[0]['ms'])}  nn={int(best.iloc[0]['nn'])}  "
          f"→ {int(best.iloc[0]['n_topics'])} topics  {best.iloc[0]['outliers%']}% outliers")
