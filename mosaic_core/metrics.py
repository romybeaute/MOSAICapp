"""
Topic-level quality and robustness metrics.

- `topic_diversity`: how many distinct reports (participants) each topic draws on.
- `embedding_coherence`: C_embed, mean pairwise cosine of a topic's sentences.
- `topic_coherence_cv`: C_v word co-occurrence coherence (requires gensim).

No Streamlit dependencies.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

__all__ = [
    "topic_diversity",
    "embedding_coherence",
    "topic_coherence_cv",
]


def topic_diversity(topics, report_ids, outlier_topic: int = -1) -> pd.DataFrame:
    """Per-topic diversity ratio: distinct source reports / sentences in the topic.

    When reports are split into sentences, a topic can be dominated by a single
    participant who repeats themselves. The diversity ratio separates shared,
    inter-subjective themes from idiosyncratic ones:

    - 1.0: every sentence in the topic comes from a different report.
    - close to 1/n: one report contributed almost all n sentences.

    Parameters
    ----------
    topics : sequence of int
        Topic assignment of each sentence (e.g. ``BERTopic.topics_``).
    report_ids : sequence
        Identifier of the report each sentence was taken from, aligned with `topics`.
    outlier_topic : int
        Topic id to exclude (BERTopic's outlier bin by default).

    Returns
    -------
    DataFrame with columns ``Topic, Total_Sentences, Unique_Reports, Diversity_Ratio``,
    one row per topic, sorted by topic id.
    """
    topics = list(topics)
    report_ids = list(report_ids)
    if len(topics) != len(report_ids):
        raise ValueError(
            f"topics ({len(topics)}) and report_ids ({len(report_ids)}) must have the same length."
        )
    topic_sources = pd.DataFrame({"Topic": topics, "Report_ID": report_ids})
    topic_sources = topic_sources[topic_sources["Topic"] != outlier_topic]

    stats = topic_sources.groupby("Topic").agg(
        Total_Sentences=("Report_ID", "size"),
        Unique_Reports=("Report_ID", "nunique"),
    ).reset_index()
    stats["Diversity_Ratio"] = stats["Unique_Reports"] / stats["Total_Sentences"]
    return stats


def embedding_coherence(embeddings, topics, outlier_topic: int = -1) -> float:
    """Embedding coherence (C_embed), as defined in the MOSAIC paper.

    Each topic's average cosine similarity over all unique pairs of its sentence
    embeddings, then averaged across topics. The outlier topic and topics with
    fewer than two sentences are excluded. Returns 0.0 if no topic qualifies.

    Closed form: for unit vectors, sum_{i<j} cos(e_i,e_j) = (||sum_i u_i||^2 - N) / 2,
    so a topic with thousands of sentences costs O(N*d) instead of an N x N matrix.
    """
    emb = np.asarray(embeddings, dtype=np.float64)
    topic_arr = np.asarray(topics)
    if len(topic_arr) != len(emb):
        raise ValueError(
            f"embeddings ({len(emb)}) and topics ({len(topic_arr)}) must have the same length."
        )
    unit = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
    intra = []
    for t in set(topic_arr.tolist()):
        if t == outlier_topic:
            continue
        u = unit[topic_arr == t]
        n_k = len(u)
        if n_k < 2:
            continue  # a pairwise average needs at least two sentences
        s = u.sum(axis=0)
        intra.append((float(s @ s) - n_k) / (n_k * (n_k - 1)))
    return float(np.mean(intra)) if intra else 0.0


# Same token pattern as BERTopic's default CountVectorizer.
_TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")


def topic_coherence_cv(docs, topic_words, top_n: int = 10) -> float:
    """C_v topic coherence (Röder et al., 2015) computed with gensim.

    Parameters
    ----------
    docs : sequence of str
        The documents the topics were fitted on.
    topic_words : sequence of sequence of str
        Ranked representative words per topic (outliers already excluded), e.g.
        ``[[w for w, _ in topic_model.get_topic(t)] for t in topic_ids]``.
    top_n : int
        Number of top words per topic to use.

    Docs are tokenised the same way BERTopic's CountVectorizer does (lowercase,
    word chars, length ≥ 2); otherwise the dictionary keeps original case and
    punctuation (e.g. "Death,") and never matches the cleaned topic words. N-gram
    topic words are split into tokens and out-of-vocabulary tokens are dropped, so
    a single phrase can't crash the metric. Topics with fewer than 2 usable words
    are skipped. Returns 0.0 if no topic qualifies.
    """
    try:
        from gensim.corpora import Dictionary
        from gensim.models import CoherenceModel
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("topic_coherence_cv requires gensim: pip install gensim") from exc

    tokenized_docs = [_TOKEN_PATTERN.findall(str(d).lower()) for d in docs]
    dictionary = Dictionary(tokenized_docs)
    vocab = dictionary.token2id

    topics_top_words = []
    for words_ranked in topic_words:
        words = []
        for word in list(words_ranked)[:top_n]:
            for tok in str(word).lower().split():
                if tok in vocab and tok not in words:
                    words.append(tok)
        # C_v needs at least 2 words to compute co-occurrence
        if len(words) >= 2:
            topics_top_words.append(words)

    if not topics_top_words:
        return 0.0
    cm = CoherenceModel(
        topics=topics_top_words,
        texts=tokenized_docs,
        dictionary=dictionary,
        coherence="c_v",
        processes=1,
    )
    return float(cm.get_coherence())
