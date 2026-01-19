"""
Pytest configuration and shared fixtures for MOSAIC-app tests.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame with phenomenological reports."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "text": [
            "I saw vivid geometric patterns and colors.",
            "There was a feeling of floating outside my body.",
            "Time seemed to slow down completely.",
            "I experienced a deep sense of peace and calm.",
            "The music created visual patterns in my mind."
        ],
        "condition": ["HS", "HS", "DL", "DL", "HS"]
    })


@pytest.fixture
def sample_csv_file(sample_dataframe):
    """Create a temporary CSV file with sample data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_dataframe.to_csv(f, index=False)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings (5 documents, 384 dimensions)."""
    np.random.seed(42)
    return np.random.randn(5, 384).astype(np.float32)


@pytest.fixture
def sample_documents():
    """Sample list of documents for testing."""
    return [
        "I saw vivid geometric patterns and colors.",
        "There was a feeling of floating outside my body.",
        "Time seemed to slow down completely.",
        "I experienced a deep sense of peace and calm.",
        "The music created visual patterns in my mind."
    ]


@pytest.fixture
def sample_config():
    """Sample BERTopic configuration dictionary."""
    return {
        "embedding_model": "all-MiniLM-L6-v2",
        "umap_params": {
            "n_neighbors": 15,
            "n_components": 5,
            "min_dist": 0.0,
        },
        "hdbscan_params": {
            "min_cluster_size": 5,
            "min_samples": 3,
        },
        "vectorizer_params": {
            "ngram_range": [1, 2],
            "stop_words": "english",
        },
        "use_vectorizer": True,
        "bt_params": {
            "top_n_words": 10,
            "nr_topics": "auto",
        },
        "granularity": "sentences",
        "min_words": 2,
    }
