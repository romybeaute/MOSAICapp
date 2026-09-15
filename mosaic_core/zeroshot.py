"""
Zero-shot classification of documents against user-defined categories.

Documents (sentences or reports) that are already embedded are compared with
embedded category descriptions, and each document is assigned to its closest
category if the match is both strong enough and unambiguous. This is the same
similarity computation BERTopic performs internally with `zeroshot_topic_list`,
exposed here so the threshold can be inspected and tuned.

The work is split in two so that interactive threshold tuning stays cheap:

1. `compute_zeroshot_similarities` encodes the categories once and returns the
   document x category-line cosine matrix (expensive, threshold-independent).
2. `apply_zeroshot_threshold` turns that matrix into assignments (pure numpy).

No Streamlit dependencies.
"""

from __future__ import annotations

import hashlib
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

__all__ = [
    "parse_categories",
    "embeddings_fingerprint",
    "compute_zeroshot_similarities",
    "apply_zeroshot_threshold",
]


def parse_categories(raw: str) -> tuple[list[str], list[str], list[str]]:
    """Parse a block of category definitions into (labels, line_labels, line_texts).

    Each non-empty line is either a bare label ("Anxiety") or
    "Label | description of what belongs to it" — the description is embedded
    together with the label, which anchors the category vector in the same
    register as real sentences instead of an abstract two-word title.
    Several lines may share the same label (e.g. one questionnaire item per
    line); a document's similarity to a label is the max over that label's lines.
    """
    labels: list[str] = []
    line_labels: list[str] = []
    line_texts: list[str] = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if "|" in line:
            label, desc = line.split("|", 1)
            label, desc = label.strip(), desc.strip()
            text = f"{label}: {desc}" if desc else label
        else:
            label, text = line, line
        if not label:
            continue
        if label not in labels:
            labels.append(label)
        line_labels.append(label)
        line_texts.append(text)
    return labels, line_labels, line_texts


def embeddings_fingerprint(embeddings: np.ndarray) -> str:
    """Cheap content hash of document embeddings, for use as a cache key.

    Hashing the full array can be slow for large corpora × high-dim models,
    so hash the shape plus a strided sample of rows — enough to distinguish
    two datasets that happen to have the same number of documents.
    """
    arr = np.ascontiguousarray(embeddings)
    step = max(1, len(arr) // 512)
    h = hashlib.sha1()
    h.update(str(arr.shape).encode())
    h.update(arr[::step].tobytes())
    return h.hexdigest()


def compute_zeroshot_similarities(
    doc_embeddings: np.ndarray,
    line_texts,
    encode: Callable[[list[str]], np.ndarray],
    model_name: str = "the embedding model",
) -> np.ndarray:
    """Cosine similarity between every document and every category line.

    Parameters
    ----------
    doc_embeddings : array of shape (n_docs, dim)
        Precomputed document embeddings.
    line_texts : sequence of str
        Category line texts, as returned by `parse_categories`.
    encode : callable
        Maps a list of strings to an (n, dim) array, e.g.
        ``SentenceTransformer(model_name).encode``. It must be the same model
        that produced `doc_embeddings`.
    model_name : str
        Only used in the error message.

    Returns
    -------
    ndarray of shape (n_docs, n_lines)
    """
    doc_emb = np.asarray(doc_embeddings)

    # Guard: the category lines are encoded with `encode`, while the documents
    # use the (possibly precomputed) `doc_embeddings`. If the dimensions differ,
    # fail here with an actionable message instead of a cryptic matrix-shape error.
    cat_emb = np.asarray(encode(list(line_texts)))
    if cat_emb.ndim != 2 or doc_emb.ndim != 2 or cat_emb.shape[1] != doc_emb.shape[1]:
        raise ValueError(
            f"Embedding-model mismatch: your document embeddings are "
            f"{doc_emb.shape[-1]}-dim, but '{model_name}' produces "
            f"{cat_emb.shape[-1]}-dim vectors. Use the SAME embedding model that "
            "created the document embeddings (e.g. Qwen/Qwen3-Embedding-4B → 2560-dim)."
        )
    return cosine_similarity(doc_emb, cat_emb)


def apply_zeroshot_threshold(line_sims, labels, line_labels, min_similarity, min_margin=0.0):
    """Turn the doc × category-line similarity matrix into assignments.

    A label's similarity is the max over its lines. A document is assigned to
    its best label when best similarity ≥ min_similarity AND the best label
    beats the runner-up by ≥ min_margin (ambiguous docs sit between category
    vectors; the margin filter leaves them Unclassified). Pure numpy — cheap
    enough to re-run live whenever a slider moves.

    Returns (topics, topic_info, per_doc) where `topics` is a list with -1 for
    unclassified, `topic_info` mirrors BERTopic's get_topic_info columns
    (Topic / Name / Count, only categories with Count > 0, plus the -1 row),
    and `per_doc` holds best_category / confidence / margin / runner_up for
    every document regardless of threshold.
    """
    labels = list(labels)
    line_sims = np.asarray(line_sims)
    label_sims = np.column_stack([
        line_sims[:, [i for i, ll in enumerate(line_labels) if ll == lab]].max(axis=1)
        for lab in labels
    ])
    n = label_sims.shape[0]
    order = np.argsort(-label_sims, axis=1)
    best = order[:, 0]
    best_sim = label_sims[np.arange(n), best]
    if len(labels) > 1:
        second = order[:, 1]
        margin = best_sim - label_sims[np.arange(n), second]
        runner_up = [labels[i] for i in second]
    else:
        margin = np.full(n, np.inf)
        runner_up = [""] * n

    assigned = (best_sim >= min_similarity) & (margin >= min_margin)
    topics = np.where(assigned, best, -1)

    rows = []
    counts = np.bincount(topics[assigned], minlength=len(labels)) if assigned.any() else np.zeros(len(labels), int)
    for i, lab in enumerate(labels):
        if counts[i] > 0:
            rows.append({"Topic": i, "Name": lab, "Count": int(counts[i])})
    rows.append({"Topic": -1, "Name": "Unclassified", "Count": int((~assigned).sum())})
    topic_info = pd.DataFrame(rows, columns=["Topic", "Name", "Count"])

    per_doc = pd.DataFrame({
        "best_category": [labels[i] for i in best],
        "confidence": np.round(best_sim, 4),
        "margin": np.round(np.where(np.isfinite(margin), margin, np.nan), 4),
        "runner_up": runner_up,
    })
    return topics.tolist(), topic_info, per_doc
