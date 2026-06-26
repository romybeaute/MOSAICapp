"""
Show examples of sentences reassigned from outlier (-1) to a topic.
Helps manually validate outlier reduction quality.

Usage:
    python3 check_reassignments.py --dataset NDE --threshold 0.7
    python3 check_reassignments.py --dataset MPE --threshold 0.7
"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",   choices=["NDE", "MPE"], default="NDE")
parser.add_argument("--threshold", default="0.7", help="e.g. 0.7 → tag ro70")
parser.add_argument("--n",         type=int, default=5, help="Examples per topic")
parser.add_argument("--seed",      type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)

tag  = f"ro{int(float(args.threshold) * 100)}"
root = Path("data")

if args.dataset == "NDE":
    cache_dir  = root / "NDE/preprocessed/cache"
    orig_file  = cache_dir / "topics_t97.json"
    new_file   = cache_dir / f"topics_t97_{tag}.json"
    docs_file  = sorted(cache_dir.glob("*_docs.json"))[0]
    labels_csv = Path("data/NDE/output/outputs_NDE_t97/topic_info.csv")  # use original edited labels
else:
    cache_dir  = root / "MPE/preprocessed/cache"
    orig_file  = cache_dir / "topics.json"
    new_file   = cache_dir / f"topics_mpe_{tag}.json"   # doesn't exist — MPE saves in-memory only
    docs_file  = sorted(cache_dir.glob("*_docs.json"))[0]
    labels_csv = Path(f"data/MPE/output/outputs_MPE_{tag}/topic_info.csv")

# Load docs
with open(docs_file, encoding="utf-8") as f:
    docs = json.load(f)

# Load original topics
if not orig_file.exists():
    print(f"Original topics file not found: {orig_file}")
    raise SystemExit(1)
with open(orig_file) as f:
    orig_topics = json.load(f)

# For MPE: load from topics_sentences.csv since topics aren't saved separately
if args.dataset == "MPE":
    # Get reassigned sentences from topics_sentences.csv comparison
    # We'll use the output topics_sentences.csv and the original topics
    ts_file = labels_csv.parent / "topics_sentences.csv"
    if not ts_file.exists():
        print(f"Copy outputs from Artemis first: {ts_file}")
        raise SystemExit(1)
    ts_df = pd.read_csv(ts_file)
    label_map = dict(zip(ts_df["topic_name"], ts_df["topic_name"]))

    # Find sentences originally -1 that now have a topic
    print(f"\n{'='*70}")
    print(f"MPE — Outlier reduction threshold={args.threshold}")
    print(f"{'='*70}")
    # We only have sentence-level from topics_sentences.csv
    # Show sample sentences per topic to judge quality
    for _, row in ts_df.iterrows():
        sentences = [s.strip() for s in str(row["texts"]).split(" | ") if s.strip()]
        sample = random.sample(sentences, min(args.n, len(sentences)))
        print(f"\nTopic: {row['topic_name']}  (n={row['n_sentences']})")
        for s in sample:
            print(f"  • {s[:120]}")
    raise SystemExit(0)

# NDE: compare orig vs new topics
if not new_file.exists():
    print(f"Reduced topics file not found: {new_file}")
    print(f"Copy from Artemis: scp rb666@artemis:.../cache/topics_t97_{tag}.json {cache_dir}/")
    raise SystemExit(1)

with open(new_file) as f:
    new_topics = json.load(f)

# Load labels
labels = {}
if labels_csv.exists():
    df = pd.read_csv(labels_csv)
    labels = {int(t): str(v) for t, v in zip(df["Topic"], df["LLM_Label"])
          if str(v) not in ("nan", "", "None")}

# Find reassigned sentences
reassigned = defaultdict(list)
for i, (orig, new) in enumerate(zip(orig_topics, new_topics)):
    if orig == -1 and new != -1:
        reassigned[new].append(docs[i])

n_total = sum(len(v) for v in reassigned.values())
print(f"\n{'='*70}")
print(f"NDE t97 — Outlier reduction threshold={args.threshold}")
print(f"Total reassigned: {n_total} sentences across {len(reassigned)} topics")
print(f"{'='*70}")

for tid in sorted(reassigned.keys()):
    sents = reassigned[tid]
    label = labels.get(tid, f"Topic {tid}")
    sample = random.sample(sents, min(args.n, len(sents)))
    print(f"\nTopic {tid}: {label}  ({len(sents)} reassigned)")
    for s in sample:
        print(f"  • {s[:120]}")
