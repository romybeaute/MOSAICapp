"""
Integration tests that call real models and APIs.

These are SLOW and should NOT run in CI.
Run manually with: pytest tests/test_integration.py -v

Requires:
- Internet connection
- HF_TOKEN env var (for LLM tests)
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

# Skip entire module if running in CI
pytestmark = pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Integration tests skipped in CI"
)


@pytest.fixture
def integration_csv():
    """CSV with enough data for real embedding."""
    texts = [
        "I saw bright geometric patterns.",
        "Colors were vivid and shifting.",
        "Time felt distorted and slow.",
        "I felt detached from my body.",
        "There was a sense of peace.",
    ] * 6  # 30 docs
    
    df = pd.DataFrame({"text": texts})
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f, index=False)
        path = f.name
    
    yield path
    os.unlink(path)


class TestRealEmbeddings:
    """Tests with actual embedding model."""
    
    def test_compute_embeddings_real(self):
        from mosaic_core.core_functions import compute_embeddings
        
        docs = ["This is a test.", "Another sentence here."]
        embeddings = compute_embeddings(
            docs,
            model_name="all-MiniLM-L6-v2",  # small, fast model
            device="cpu"
        )
        
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] == 384  # MiniLM dimension
        assert embeddings.dtype == np.float32
    
    def test_preprocess_and_embed_real(self, integration_csv):
        from mosaic_core.core_functions import preprocess_and_embed
        
        docs, embeddings = preprocess_and_embed(
            integration_csv,
            model_name="all-MiniLM-L6-v2",
            split_sentences=False,
            min_words=3,
            device="cpu"
        )
        
        assert len(docs) == 30
        assert embeddings.shape == (30, 384)


class TestRealTopicModeling:
    """Full pipeline with real embeddings."""
    
    def test_full_pipeline(self, integration_csv):
        from mosaic_core.core_functions import (
            preprocess_and_embed, run_topic_model,
            get_topic_labels, get_outlier_stats
        )
        
        docs, embeddings = preprocess_and_embed(
            integration_csv,
            model_name="all-MiniLM-L6-v2",
            split_sentences=False,
            device="cpu"
        )
        
        config = {
            "umap_params": {"n_neighbors": 5, "n_components": 2, "min_dist": 0.0},
            "hdbscan_params": {"min_cluster_size": 3, "min_samples": 2},
            "bt_params": {"nr_topics": "auto", "top_n_words": 5},
            "use_vectorizer": True,
        }
        
        model, reduced, topics = run_topic_model(docs, embeddings, config)
        labels = get_topic_labels(model, topics)
        outlier_count, outlier_pct = get_outlier_stats(model)
        
        assert len(topics) == len(docs)
        assert len(labels) == len(docs)
        assert reduced.shape == (len(docs), 2)
        assert 0 <= outlier_pct <= 100


@pytest.mark.skipif(
    not os.environ.get("HF_TOKEN"),
    reason="HF_TOKEN not set"
)
class TestRealLLMLabeling:
    """Tests with actual HuggingFace API."""
    
    def test_generate_labels_real(self, integration_csv):
        from mosaic_core.core_functions import (
            preprocess_and_embed, run_topic_model, generate_llm_labels
        )
        
        docs, embeddings = preprocess_and_embed(
            integration_csv,
            model_name="all-MiniLM-L6-v2",
            split_sentences=False,
            device="cpu"
        )
        
        config = {
            "umap_params": {"n_neighbors": 5, "n_components": 2, "min_dist": 0.0},
            "hdbscan_params": {"min_cluster_size": 3, "min_samples": 2},
            "bt_params": {"nr_topics": 2, "top_n_words": 5},
            "use_vectorizer": True,
        }
        
        model, _, _ = run_topic_model(docs, embeddings, config)
        
        labels = generate_llm_labels(
            model,
            hf_token=os.environ["HF_TOKEN"],
            max_topics=2
        )
        
        assert isinstance(labels, dict)
        assert len(labels) > 0
        assert all(isinstance(v, str) for v in labels.values())