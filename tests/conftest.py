"""Pytest fixtures for MOSAIC tests using local dummy dataset."""

import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

@pytest.fixture
def sample_csv():
    """Returns the path to the dummy_dataset.csv file located in the same directory."""
    # Get the directory where this conftest.py file resides
    current_dir = Path(__file__).parent
    file_path = current_dir / "dummy_dataset.csv"
    
    if not file_path.exists():
        pytest.fail(f"Test data file not found at: {file_path}")
        
    return str(file_path)

@pytest.fixture
def sample_dataframe(sample_csv):
    """Loads the CSV into a DataFrame and normalizes column names."""
    df = pd.read_csv(sample_csv)
    
    # Normalize text column name for tests (handle 'report' vs 'text')
    if 'text' not in df.columns:
        if 'report' in df.columns:
            df = df.rename(columns={'report': 'text'})
        else:
            # Fallback: assume first column is text if neither exists
            df = df.rename(columns={df.columns[0]: 'text'})
            
    return df

@pytest.fixture
def sample_texts(sample_dataframe):
    """Returns the list of text reports from the dataframe."""
    return sample_dataframe['text'].tolist()

@pytest.fixture
def sample_embeddings(sample_texts):
    """Generates random embeddings matching the exact length of the CSV data."""
    np.random.seed(42)
    # Generate (n_samples, 384) matrix
    return np.random.randn(len(sample_texts), 384).astype(np.float32)

@pytest.fixture
def larger_corpus(sample_texts):
    """
    Alias for sample_texts. 
    Since the dummy dataset is sufficiently large, we reuse it.
    """
    return sample_texts

@pytest.fixture
def larger_embeddings(sample_embeddings):
    """Alias for sample_embeddings matching the larger corpus."""
    return sample_embeddings

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