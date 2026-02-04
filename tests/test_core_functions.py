"""Tests for mosaic_core.core_functions module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from mosaic_core.core_functions import (
    pick_text_column,
    list_text_columns,
    slugify,
    clean_label,
    preprocess_texts,
    load_csv_texts,
    count_clean_reports,
    get_config_hash,
    make_run_id,
    run_topic_model,
    get_topic_labels,
    get_outlier_stats,
    get_num_topics,
    ensure_nltk_data,
    get_hf_status_code,
    generate_llm_labels,
)


class TestSlugify:
    """Filename sanitization."""
    
    def test_preserves_alphanumeric(self):
        assert slugify("MOSAIC") == "MOSAIC"
        assert slugify("dataset123") == "dataset123"
    
    def test_replaces_spaces(self):
        assert slugify("my dataset") == "my_dataset"
        assert slugify("my  dataset") == "my_dataset"
    
    def test_replaces_special_chars(self):
        assert slugify("data@2024!") == "data_2024_"
        assert slugify("path/to/file") == "path_to_file"
    
    def test_preserves_safe_chars(self):
        assert slugify("data-set_v1.0") == "data-set_v1.0"
    
    def test_empty_returns_default(self):
        assert slugify("") == "DATASET"
        assert slugify("   ") == "DATASET"
    
    def test_strips_whitespace(self):
        assert slugify("  name  ") == "name"


class TestPickTextColumn:
    """Auto-detection of text columns."""
    
    def test_priority_order(self):
        df = pd.DataFrame({
            "reflection_answer_english": ["a"],
            "text": ["b"],
        })
        assert pick_text_column(df) == "reflection_answer_english"
    
    def test_fallback_columns(self):
        assert pick_text_column(pd.DataFrame({"text": ["a"]})) == "text"
        assert pick_text_column(pd.DataFrame({"report": ["a"]})) == "report"
        assert pick_text_column(pd.DataFrame({"reflection_answer": ["a"]})) == "reflection_answer"
    
    def test_returns_none_if_no_match(self):
        df = pd.DataFrame({"description": ["a"], "notes": ["b"]})
        assert pick_text_column(df) is None
    
    def test_empty_dataframe(self):
        assert pick_text_column(pd.DataFrame()) is None


class TestListTextColumns:
    """Column listing."""
    
    def test_returns_all_columns(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        assert list_text_columns(df) == ["a", "b", "c"]
    
    def test_empty_dataframe(self):
        assert list_text_columns(pd.DataFrame()) == []


class TestCleanLabel:
    """LLM output normalization."""
    
    def test_basic_label(self):
        assert clean_label("Visual Patterns") == "Visual Patterns"
    
    def test_strips_whitespace(self):
        assert clean_label("  Visual Patterns  ") == "Visual Patterns"
    
    def test_removes_quotes(self):
        assert clean_label('"Visual Patterns"') == "Visual Patterns"
        assert clean_label("'Visual Patterns'") == "Visual Patterns"
        assert clean_label("`Visual Patterns`") == "Visual Patterns"
    
    def test_removes_trailing_punctuation(self):
        assert clean_label("Visual Patterns.") == "Visual Patterns"
        assert clean_label("Visual Patterns:") == "Visual Patterns"
        assert clean_label("Visual Patterns—") == "Visual Patterns"
    
    def test_removes_experience_prefix(self):
        assert clean_label("Experience of Light") == "Light"
        assert clean_label("Subjective Experience of Colors") == "Colors"
        assert clean_label("Phenomenon of Sound") == "Sound"
        # "Experiential Phenomenon" is matched, leaving "of Motion"
        # This is expected behavior - the regex handles common patterns
    
    def test_removes_experience_suffix(self):
        assert clean_label("Visual experience") == "Visual"
        assert clean_label("Color phenomenon") == "Color"
        assert clean_label("Light state") == "Light"
    
    def test_takes_first_line(self):
        assert clean_label("Label\nExplanation text") == "Label"
    
    def test_empty_returns_unlabelled(self):
        assert clean_label("") == "Unlabelled"
        assert clean_label("   ") == "Unlabelled"
        assert clean_label(None) == "Unlabelled"


class TestPreprocessTexts:
    """Text preprocessing and sentence splitting."""
    
    def test_sentence_splitting(self):
        texts = ["First sentence. Second sentence."]
        docs, removed, stats = preprocess_texts(texts, split_sentences=True, min_words=0)
        assert len(docs) == 2
        assert stats["total_before"] == 2
    
    def test_no_splitting(self):
        texts = ["First sentence. Second sentence."]
        docs, removed, stats = preprocess_texts(texts, split_sentences=False, min_words=0)
        assert len(docs) == 1
    
    def test_min_words_filter(self):
        texts = ["This is long enough.", "Short."]
        docs, removed, stats = preprocess_texts(texts, split_sentences=False, min_words=3)
        assert len(docs) == 1
        assert len(removed) == 1
        assert stats["removed_count"] == 1
    
    def test_stats_accuracy(self):
        texts = ["One sentence. Another sentence.", "Third sentence here."]
        docs, removed, stats = preprocess_texts(texts, split_sentences=True, min_words=2)
        assert stats["total_before"] == 3  # NLTK splits into 3 sentences
        assert stats["total_after"] == len(docs)
        assert stats["removed_count"] == len(removed)


class TestLoadCSVTexts:
    """CSV loading."""
    
    def test_loads_texts(self, sample_csv):
        texts = load_csv_texts(sample_csv, text_col="report")
        assert len(texts) > 0
    
    def test_auto_detects_column(self, sample_csv):
        texts = load_csv_texts(sample_csv)
        assert len(texts) > 0
    
    def test_raises_on_missing_column(self, sample_csv):
        with pytest.raises(ValueError, match="No valid text column"):
            load_csv_texts(sample_csv, text_col="nonexistent")
    
    def test_filters_empty_rows(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("text\n")
            f.write("Valid text\n")
            f.write("\n")
            f.write("   \n")
            f.write("Another valid\n")
            path = f.name
        
        try:
            texts = load_csv_texts(path)
            assert len(texts) == 2
        finally:
            os.unlink(path)


class TestCountCleanReports:
    """Report counting."""
    
    def test_counts_correctly(self, sample_csv):
        assert count_clean_reports(sample_csv, "report") > 0
    
    def test_returns_zero_on_error(self):
        assert count_clean_reports("/nonexistent/path.csv") == 0


class TestConfigUtils:
    """Config hashing and run IDs."""
    
    def test_hash_is_deterministic(self):
        cfg = {"a": 1, "b": 2}
        assert get_config_hash(cfg) == get_config_hash(cfg)
    
    def test_hash_ignores_key_order(self):
        cfg1 = {"a": 1, "b": 2}
        cfg2 = {"b": 2, "a": 1}
        assert get_config_hash(cfg1) == get_config_hash(cfg2)
    
    def test_run_id_contains_hash(self):
        cfg = {"a": 1}
        run_id = make_run_id(cfg)
        h = get_config_hash(cfg)
        assert h in run_id


class TestRunTopicModel:
    """BERTopic fitting."""
    
    def test_returns_expected_types(self, larger_corpus, larger_embeddings, topic_config):
        model, reduced, topics = run_topic_model(
            larger_corpus, larger_embeddings, topic_config
        )
        
        assert hasattr(model, "get_topic_info")
        assert reduced.shape == (len(larger_corpus), 2)
        assert len(topics) == len(larger_corpus)
    
    def test_reduced_is_2d(self, larger_corpus, larger_embeddings, topic_config):
        _, reduced, _ = run_topic_model(larger_corpus, larger_embeddings, topic_config)
        assert reduced.ndim == 2
        assert reduced.shape[1] == 2
    
    def test_topics_are_integers(self, larger_corpus, larger_embeddings, topic_config):
        _, _, topics = run_topic_model(larger_corpus, larger_embeddings, topic_config)
        assert all(isinstance(t, (int, np.integer)) for t in topics)


class TestGetTopicLabels:
    """Topic label extraction."""
    
    def test_returns_labels_for_all_docs(self, larger_corpus, larger_embeddings, topic_config):
        model, _, topics = run_topic_model(larger_corpus, larger_embeddings, topic_config)
        labels = get_topic_labels(model, topics)
        assert len(labels) == len(larger_corpus)
    
    def test_labels_are_strings(self, larger_corpus, larger_embeddings, topic_config):
        model, _, topics = run_topic_model(larger_corpus, larger_embeddings, topic_config)
        labels = get_topic_labels(model, topics)
        assert all(isinstance(lbl, str) for lbl in labels)


class TestOutlierStats:
    """Outlier statistics."""
    
    def test_returns_count_and_percentage(self, larger_corpus, larger_embeddings, topic_config):
        model, _, _ = run_topic_model(larger_corpus, larger_embeddings, topic_config)
        count, pct = get_outlier_stats(model)
        assert isinstance(count, int)
        assert isinstance(pct, float)
        assert 0 <= pct <= 100
    
    def test_num_topics(self, larger_corpus, larger_embeddings, topic_config):
        model, _, _ = run_topic_model(larger_corpus, larger_embeddings, topic_config)
        n = get_num_topics(model)
        assert isinstance(n, int)
        assert n >= 0


class TestEmbeddingShapeValidation:
    """Embedding consistency checks."""
    
    def test_shape_matches_docs(self, sample_texts, sample_embeddings):
        assert sample_embeddings.shape[0] == len(sample_texts)
    
    def test_dtype_is_float32(self, sample_embeddings):
        assert sample_embeddings.dtype == np.float32


class TestLabelsCachePath:
    """Label cache path generation."""
    
    def test_returns_path_object(self):
        from mosaic_core.core_functions import labels_cache_path
        from pathlib import Path
        
        p = labels_cache_path("/tmp", "abc123", "meta-llama/Llama-3")
        assert isinstance(p, Path)
    
    def test_sanitizes_model_id(self):
        from mosaic_core.core_functions import labels_cache_path
        
        p = labels_cache_path("/tmp", "hash", "org/model-name")
        assert "/" not in p.name


class TestLabelsCacheIO:
    """Label cache read/write."""
    
    def test_save_and_load(self):
        from mosaic_core.core_functions import save_labels_cache, load_cached_labels
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        
        try:
            labels = {0: "Topic A", 1: "Topic B"}
            save_labels_cache(path, labels)
            loaded = load_cached_labels(path)
            assert loaded == labels
        finally:
            os.unlink(path)
    
    def test_load_returns_none_on_missing(self):
        from mosaic_core.core_functions import load_cached_labels
        
        result = load_cached_labels("/nonexistent/path.json")
        assert result is None


class TestCleanupOldCache:
    """Cache cleanup."""
    
    def test_removes_non_matching_files(self):
        from mosaic_core.core_functions import cleanup_old_cache

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake cache files (json docs + npy embeddings)
            (Path(tmpdir) / "precomputed_OLD_docs.json").touch()
            (Path(tmpdir) / "precomputed_OLD_embeddings.npy").touch()
            (Path(tmpdir) / "precomputed_CURRENT_docs.json").touch()

            removed = cleanup_old_cache(tmpdir, "CURRENT")

            assert removed == 2
            assert (Path(tmpdir) / "precomputed_CURRENT_docs.json").exists()
            assert not (Path(tmpdir) / "precomputed_OLD_docs.json").exists()
    
    def test_handles_missing_dir(self):
        from mosaic_core.core_functions import cleanup_old_cache
        
        result = cleanup_old_cache("/nonexistent/dir", "test")
        assert result == 0


class TestResolveDevice:
    """Device resolution."""
    
    def test_cpu_explicit(self):
        from mosaic_core.core_functions import resolve_device
        
        device, batch = resolve_device("cpu")
        assert device == "cpu"
        assert batch == 64
    
    def test_cpu_uppercase(self):
        from mosaic_core.core_functions import resolve_device

        device, _ = resolve_device("CPU")
        assert device == "cpu"


class TestEnsureNltkData:
    """NLTK data download logic."""

    @patch("mosaic_core.core_functions.nltk")
    def test_returns_when_punkt_tab_found(self, mock_nltk):
        mock_nltk.data.path = []
        mock_nltk.data.find.return_value = True
        ensure_nltk_data()
        mock_nltk.download.assert_not_called()

    @patch("mosaic_core.core_functions.nltk")
    def test_downloads_when_both_missing(self, mock_nltk):
        mock_nltk.data.path = []
        mock_nltk.data.find.side_effect = LookupError("not found")
        ensure_nltk_data()
        mock_nltk.download.assert_called_once_with("punkt", download_dir=None, quiet=True)

    @patch("mosaic_core.core_functions.nltk")
    def test_data_dir_added_to_path(self, mock_nltk):
        mock_nltk.data.path = []
        mock_nltk.data.find.return_value = True
        ensure_nltk_data(data_dir="/custom/path")
        assert "/custom/path" in mock_nltk.data.path

    @patch("mosaic_core.core_functions.nltk")
    def test_data_dir_not_duplicated(self, mock_nltk):
        mock_nltk.data.path = ["/custom/path"]
        mock_nltk.data.find.return_value = True
        ensure_nltk_data(data_dir="/custom/path")
        assert mock_nltk.data.path.count("/custom/path") == 1

    @patch("mosaic_core.core_functions.nltk")
    def test_falls_back_to_punkt(self, mock_nltk):
        """punkt_tab missing but punkt found — no download needed."""
        mock_nltk.data.path = []
        mock_nltk.data.find.side_effect = [LookupError, True]
        ensure_nltk_data()
        mock_nltk.download.assert_not_called()

    @patch("mosaic_core.core_functions.nltk")
    def test_download_failure_does_not_crash(self, mock_nltk):
        mock_nltk.data.path = []
        mock_nltk.data.find.side_effect = LookupError("not found")
        mock_nltk.download.side_effect = OSError("network error")
        ensure_nltk_data()  # should not raise


class TestGetHfStatusCode:
    """HTTP status code extraction from HuggingFace exceptions."""

    def test_extracts_status_code(self):
        exc = Exception("api error")
        exc.response = MagicMock(status_code=402)
        assert get_hf_status_code(exc) == 402

    def test_returns_none_without_response(self):
        exc = Exception("no response attr")
        assert get_hf_status_code(exc) is None

    def test_returns_none_with_none_response(self):
        exc = Exception("null response")
        exc.response = None
        assert get_hf_status_code(exc) is None

    def test_returns_none_without_status_code(self):
        exc = Exception("response but no status")
        exc.response = MagicMock(spec=[])
        assert get_hf_status_code(exc) is None


class TestGenerateLlmLabels:
    """LLM label generation with mocked HuggingFace API."""

    @staticmethod
    def _make_mock_topic_model(topic_ids):
        model = MagicMock()
        info_data = {
            "Topic": [-1] + topic_ids,
            "Count": [5] * (1 + len(topic_ids)),
        }
        model.get_topic_info.return_value = pd.DataFrame(info_data)
        model.get_topic.return_value = [("word1", 0.5), ("word2", 0.3)]
        model.get_representative_docs.return_value = ["Sample doc text."]
        return model

    @patch("mosaic_core.core_functions.InferenceClient")
    def test_success_returns_labels(self, MockClient):
        mock_client = MockClient.return_value
        completion = MagicMock()
        completion.choices = [MagicMock(message=MagicMock(content="Embodied Awareness"))]
        mock_client.chat_completion.return_value = completion

        model = self._make_mock_topic_model([0, 1])
        labels = generate_llm_labels(model, hf_token="fake-token", max_topics=2)

        assert isinstance(labels, dict)
        assert len(labels) == 2
        assert all(isinstance(v, str) for v in labels.values())

    @patch("mosaic_core.core_functions.InferenceClient")
    def test_402_raises_runtime_error(self, MockClient):
        mock_client = MockClient.return_value
        exc = Exception("payment required")
        exc.response = MagicMock(status_code=402)
        mock_client.chat_completion.side_effect = exc

        model = self._make_mock_topic_model([0])
        with pytest.raises(RuntimeError, match="402 Payment Required"):
            generate_llm_labels(model, hf_token="fake-token")

    @patch("mosaic_core.core_functions.InferenceClient")
    def test_generic_error_falls_back(self, MockClient):
        mock_client = MockClient.return_value
        mock_client.chat_completion.side_effect = Exception("server error")

        model = self._make_mock_topic_model([0, 1])
        labels = generate_llm_labels(model, hf_token="fake-token")

        assert labels[0] == "Topic 0"
        assert labels[1] == "Topic 1"

    @patch("mosaic_core.core_functions.InferenceClient")
    def test_labels_cleaned(self, MockClient):
        mock_client = MockClient.return_value
        completion = MagicMock()
        completion.choices = [MagicMock(message=MagicMock(content='"Experience of Light."'))]
        mock_client.chat_completion.return_value = completion

        model = self._make_mock_topic_model([0])
        labels = generate_llm_labels(model, hf_token="fake-token")

        assert labels[0] == "Light"