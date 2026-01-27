"""Pytest fixtures for MOSAIC tests."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_texts():
    """Short phenomenological reports for testing."""
    return [
        "I saw vivid geometric patterns and colors.",
        "There was a feeling of floating outside my body.",
        "Time seemed to slow down completely.",
        "I experienced a deep sense of peace and calm.",
        "The music created visual patterns in my mind.",
    ]


@pytest.fixture
def sample_dataframe(sample_texts):
    """DataFrame with text column and metadata."""
    return pd.DataFrame({
        "id": range(1, len(sample_texts) + 1),
        "text": sample_texts,
        "condition": ["HS", "HS", "DL", "DL", "HS"],
    })


@pytest.fixture
def sample_csv(sample_dataframe):
    """Temporary CSV file with sample data."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        sample_dataframe.to_csv(f, index=False)
        path = f.name
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def sample_embeddings():
    """Random embeddings matching sample_texts length."""
    np.random.seed(42)
    return np.random.randn(5, 384).astype(np.float32)


@pytest.fixture
def larger_corpus():
    """30 documents for topic modeling tests (UMAP needs >15 samples)."""
    base = [
        "I saw a bright light.",
        "The light was blinding and white.",
        "I felt a presence nearby.",
        "The presence was comforting.",
        "Patterns emerged in the visual field.",
        "Geometric patterns were everywhere.",
    ]
    return base * 5


@pytest.fixture
def larger_embeddings(larger_corpus):
    """Embeddings for the larger corpus."""
    np.random.seed(42)
    return np.random.randn(len(larger_corpus), 384).astype(np.float32)


@pytest.fixture
def topic_config():
    """Minimal BERTopic configuration for fast tests."""
    return {
        "umap_params": {"n_neighbors": 5, "n_components": 2, "min_dist": 0.0},
        "hdbscan_params": {"min_cluster_size": 2, "min_samples": 1},
        "bt_params": {"nr_topics": 2, "top_n_words": 3},
        "vectorizer_params": {"stop_words": "english"},
        "use_vectorizer": True,
    }