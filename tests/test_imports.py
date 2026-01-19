"""
Tests to verify that all required packages can be imported.
This catches missing dependencies early.
"""

import pytest


class TestImports:
    """Test that all required packages are importable."""

    def test_import_pandas(self):
        """Test pandas import."""
        import pandas
        assert pandas is not None

    def test_import_numpy(self):
        """Test numpy import."""
        import numpy
        assert numpy is not None

    def test_import_streamlit(self):
        """Test streamlit import."""
        import streamlit
        assert streamlit is not None

    def test_import_bertopic(self):
        """Test BERTopic import."""
        import bertopic
        assert bertopic is not None

    def test_import_sentence_transformers(self):
        """Test sentence-transformers import."""
        import sentence_transformers
        assert sentence_transformers is not None

    def test_import_umap(self):
        """Test UMAP import."""
        import umap
        assert umap is not None

    def test_import_hdbscan(self):
        """Test HDBSCAN import."""
        import hdbscan
        assert hdbscan is not None

    def test_import_sklearn(self):
        """Test scikit-learn import."""
        import sklearn
        assert sklearn is not None

    def test_import_nltk(self):
        """Test NLTK import."""
        import nltk
        assert nltk is not None

    def test_import_datamapplot(self):
        """Test datamapplot import."""
        import datamapplot
        assert datamapplot is not None

    def test_import_matplotlib(self):
        """Test matplotlib import."""
        import matplotlib
        assert matplotlib is not None

    def test_import_huggingface_hub(self):
        """Test huggingface_hub import."""
        import huggingface_hub
        assert huggingface_hub is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
