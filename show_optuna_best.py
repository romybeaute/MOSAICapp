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
)
cols = ["trial", "emb_coherence", "n_topics", "outlier_pct",
        "min_cluster_size", "min_samples", "n_neighbors"]
if "n_components" in df.columns:
    cols.append("n_components")
best = (best[cols]
    .reset_index(drop=True)
)

if best.empty:
    print(f"No trials with topic_score=1.0 and outlier_pct ≤ {args.max_outliers}%.")
    print("Try --max-outliers 60")
else:
    best.index += 1
    rename = ["trial", "emb_coh", "n_topics", "outliers%", "mcs", "ms", "nn"]
    if "n_components" in best.columns:
        rename.append("nc")
    best.columns = rename
    best["emb_coh"] = best["emb_coh"].round(4)
    best["outliers%"] = best["outliers%"].round(1)
    print(best.to_string())
    r = best.iloc[0]
    nc = int(r['nc']) if 'nc' in best.columns and str(r['nc']) not in ('', 'nan') else '?'
    nc_str = f"  nc={nc}" if nc != '?' else ""
    print(f"\nBest: trial {int(r['trial'])} — "
          f"mcs={int(r['mcs'])}  ms={int(r['ms'])}  nn={int(r['nn'])}{nc_str}  "
          f"→ {int(r['n_topics'])} topics  {r['outliers%']}% outliers")
    nc_arg = f"--nc {nc} " if nc != '?' else ""
    print(f"\nTo run this config:")
    print(f"  python run_pipeline_NDE_variant.py --mcs {int(r['mcs'])} --ms {int(r['ms'])} "
          f"--nn {int(r['nn'])} {nc_arg}--tag best")
