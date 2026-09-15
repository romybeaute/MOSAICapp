"""
mosaic_core: the MOSAIC analysis library, usable without Streamlit.

Modules
-------
core_functions
    Preprocessing, embedding, BERTopic modelling and LLM topic labelling.
zeroshot
    Zero-shot classification of documents against user-defined categories.
comparison
    Centred-cosine comparison of two topic solutions, MAD-calibrated match
    threshold, hub filtering and one-to-one matching.
metrics
    Topic diversity ratio, embedding coherence (C_embed) and C_v coherence.

The lightweight modules (zeroshot, comparison, metrics) are imported here.
`core_functions` pulls in BERTopic and sentence-transformers, so import it
explicitly: ``from mosaic_core import core_functions``.
"""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("MOSAICapp")
except PackageNotFoundError:  # running from a source checkout without install
    __version__ = "unknown"

from mosaic_core.comparison import (
    STRINGENCY_LEVELS,
    calibrate_match_threshold,
    compute_condition_similarity,
    describe_threshold,
    greedy_match,
    pair_zscores,
    parse_condition_csv,
    robust_loc_scale,
    shared_theme_chi2,
)
from mosaic_core.metrics import embedding_coherence, topic_coherence_cv, topic_diversity
from mosaic_core.zeroshot import (
    apply_zeroshot_threshold,
    compute_zeroshot_similarities,
    embeddings_fingerprint,
    parse_categories,
)

__all__ = [
    "__version__",
    # zeroshot
    "parse_categories",
    "embeddings_fingerprint",
    "compute_zeroshot_similarities",
    "apply_zeroshot_threshold",
    # comparison
    "STRINGENCY_LEVELS",
    "parse_condition_csv",
    "compute_condition_similarity",
    "robust_loc_scale",
    "calibrate_match_threshold",
    "describe_threshold",
    "pair_zscores",
    "greedy_match",
    "shared_theme_chi2",
    # metrics
    "topic_diversity",
    "embedding_coherence",
    "topic_coherence_cv",
]
