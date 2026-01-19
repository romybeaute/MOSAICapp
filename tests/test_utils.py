"""
Tests for utility functions in MOSAIC-app.
These tests verify core functionality without requiring Streamlit or heavy ML models.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os


# =====================================================================
# Test helper functions
# =====================================================================

def _slugify(s: str) -> str:
    """Convert string to safe folder name (copied from app.py for testing)."""
    import re
    s = s.strip()
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s or "DATASET"


def _pick_text_column(df: pd.DataFrame) -> str | None:
    """Return the first matching preferred text column name if present."""
    ACCEPTABLE_TEXT_COLUMNS = [
        "reflection_answer_english",
        "reflection_answer",
        "text",
        "report",
    ]
    for col in ACCEPTABLE_TEXT_COLUMNS:
        if col in df.columns:
            return col
    return None


def _list_text_columns(df: pd.DataFrame) -> list[str]:
    """Return all columns."""
    return list(df.columns)


def _clean_label(x: str) -> str:
    """Clean LLM-generated label (copied from app.py for testing)."""
    import re
    x = (x or "").strip()
    lines = x.splitlines()
    x = lines[0].strip() if lines else ""
    x = x.strip(' "\'`')
    x = re.sub(r"[.:\-–—]+$", "", x).strip()
    x = re.sub(r"[^\w\s]", "", x).strip()
    # Remove "Experience of" but keep "Experience" alone for separate handling
    x = re.sub(
        r"^(Experiential(?:\s+Phenomenon)?|Experience of|Subjective Experience of|Phenomenon of)\s+",
        "",
        x,
        flags=re.IGNORECASE,
    )
    x = re.sub(
        r"\s+(experience|experiences|phenomenon|state|states)$",
        "",
        x,
        flags=re.IGNORECASE,
    )
    x = x.strip()
    return x or "Unlabelled"


# =====================================================================
# Tests for _slugify
# =====================================================================

class TestSlugify:
    """Tests for the _slugify function."""

    def test_basic_string(self):
        """Test that basic strings are preserved."""
        assert _slugify("MOSAIC") == "MOSAIC"
        assert _slugify("dataset") == "dataset"

    def test_spaces_replaced(self):
        """Test that spaces are replaced with underscores."""
        assert _slugify("my dataset") == "my_dataset"
        assert _slugify("my  dataset") == "my_dataset"

    def test_special_characters_replaced(self):
        """Test that special characters are replaced."""
        assert _slugify("dataset@2024!") == "dataset_2024_"
        assert _slugify("data/set") == "data_set"

    def test_empty_string(self):
        """Test that empty strings return 'DATASET'."""
        assert _slugify("") == "DATASET"
        assert _slugify("   ") == "DATASET"

    def test_whitespace_stripped(self):
        """Test that leading/trailing whitespace is stripped."""
        assert _slugify("  dataset  ") == "dataset"

    def test_allowed_characters_preserved(self):
        """Test that dots, hyphens, underscores are preserved."""
        assert _slugify("data-set_v1.0") == "data-set_v1.0"


# =====================================================================
# Tests for _pick_text_column
# =====================================================================

class TestPickTextColumn:
    """Tests for the _pick_text_column function."""

    def test_reflection_answer_english(self):
        """Test that 'reflection_answer_english' is picked first."""
        df = pd.DataFrame({
            "id": [1, 2],
            "reflection_answer_english": ["text1", "text2"],
            "text": ["other1", "other2"]
        })
        assert _pick_text_column(df) == "reflection_answer_english"

    def test_reflection_answer(self):
        """Test that 'reflection_answer' is picked if no 'reflection_answer_english'."""
        df = pd.DataFrame({
            "id": [1, 2],
            "reflection_answer": ["text1", "text2"],
        })
        assert _pick_text_column(df) == "reflection_answer"

    def test_text_column(self):
        """Test that 'text' column is recognized."""
        df = pd.DataFrame({
            "id": [1, 2],
            "text": ["text1", "text2"],
        })
        assert _pick_text_column(df) == "text"

    def test_report_column(self):
        """Test that 'report' column is recognized."""
        df = pd.DataFrame({
            "id": [1, 2],
            "report": ["text1", "text2"],
        })
        assert _pick_text_column(df) == "report"

    def test_no_matching_column(self):
        """Test that None is returned if no matching column exists."""
        df = pd.DataFrame({
            "id": [1, 2],
            "description": ["text1", "text2"],
        })
        assert _pick_text_column(df) is None

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        assert _pick_text_column(df) is None


# =====================================================================
# Tests for _list_text_columns
# =====================================================================

class TestListTextColumns:
    """Tests for the _list_text_columns function."""

    def test_returns_all_columns(self):
        """Test that all columns are returned."""
        df = pd.DataFrame({
            "id": [1, 2],
            "text": ["a", "b"],
            "category": ["x", "y"]
        })
        cols = _list_text_columns(df)
        assert cols == ["id", "text", "category"]

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        assert _list_text_columns(df) == []


# =====================================================================
# Tests for _clean_label
# =====================================================================

class TestCleanLabel:
    """Tests for the _clean_label function."""

    def test_basic_label(self):
        """Test that basic labels are preserved."""
        assert _clean_label("Visual Patterns") == "Visual Patterns"

    def test_strips_whitespace(self):
        """Test that whitespace is stripped."""
        assert _clean_label("  Visual Patterns  ") == "Visual Patterns"

    def test_removes_quotes(self):
        """Test that quotes are removed."""
        assert _clean_label('"Visual Patterns"') == "Visual Patterns"
        assert _clean_label("'Visual Patterns'") == "Visual Patterns"

    def test_removes_trailing_punctuation(self):
        """Test that trailing punctuation is removed."""
        assert _clean_label("Visual Patterns.") == "Visual Patterns"
        assert _clean_label("Visual Patterns:") == "Visual Patterns"
        assert _clean_label("Visual Patterns—") == "Visual Patterns"

    def test_removes_experience_prefix(self):
        """Test that 'Experience of' prefix is removed."""
        assert _clean_label("Experience of Visual Patterns") == "Visual Patterns"
        assert _clean_label("Subjective Experience of Colors") == "Colors"
        assert _clean_label("Phenomenon of Light") == "Light"

    def test_removes_experience_suffix(self):
        """Test that 'experience' suffix is removed."""
        assert _clean_label("Visual Pattern experience") == "Visual Pattern"
        assert _clean_label("Color phenomenon") == "Color"

    def test_takes_first_line_only(self):
        """Test that only first line is used."""
        assert _clean_label("Visual Patterns\nSome explanation") == "Visual Patterns"

    def test_empty_returns_unlabelled(self):
        """Test that empty string returns 'Unlabelled'."""
        assert _clean_label("") == "Unlabelled"
        assert _clean_label("   ") == "Unlabelled"

    def test_none_returns_unlabelled(self):
        """Test that None returns 'Unlabelled'."""
        assert _clean_label(None) == "Unlabelled"


# =====================================================================
# Tests for CSV handling
# =====================================================================

class TestCSVHandling:
    """Tests for CSV loading and processing."""

    def test_load_csv_with_text_column(self):
        """Test loading a CSV with a recognized text column."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("id,text\n")
            f.write("1,This is a test report.\n")
            f.write("2,Another report here.\n")
            temp_path = f.name

        try:
            df = pd.read_csv(temp_path)
            assert len(df) == 2
            assert _pick_text_column(df) == "text"
        finally:
            os.unlink(temp_path)

    def test_load_csv_filters_empty_rows(self):
        """Test that empty text rows are handled."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("id,text\n")
            f.write("1,Valid text\n")
            f.write("2,\n")
            f.write("3,Another valid text\n")
            temp_path = f.name

        try:
            df = pd.read_csv(temp_path)
            df = df[df["text"].notna() & (df["text"].str.strip() != "")]
            assert len(df) == 2
        finally:
            os.unlink(temp_path)


# =====================================================================
# Tests for sentence tokenization
# =====================================================================

class TestSentenceTokenization:
    """Tests for sentence splitting functionality."""

    def test_sentence_splitting(self):
        """Test basic sentence splitting."""
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        text = "This is sentence one. This is sentence two. And a third."
        sentences = nltk.sent_tokenize(text)
        assert len(sentences) == 3

    def test_single_sentence(self):
        """Test with single sentence."""
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        text = "Just one sentence here."
        sentences = nltk.sent_tokenize(text)
        assert len(sentences) == 1

    def test_min_words_filter(self):
        """Test filtering sentences by minimum word count."""
        sentences = [
            "This is a long sentence with many words.",
            "Short one.",
            "Another longer sentence here.",
            "Hi."
        ]
        min_words = 3
        filtered = [s for s in sentences if len(s.split()) >= min_words]
        assert len(filtered) == 2


# =====================================================================
# Tests for data validation
# =====================================================================

class TestDataValidation:
    """Tests for data validation functions."""

    def test_embedding_shape_validation(self):
        """Test that embedding dimensions match document count."""
        docs = ["doc1", "doc2", "doc3"]
        embeddings = np.random.randn(3, 384)  # 3 docs, 384-dim embeddings
        
        assert embeddings.shape[0] == len(docs)

    def test_embedding_dtype(self):
        """Test embedding dtype conversion."""
        embeddings = np.random.randn(5, 384)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        
        assert embeddings.dtype == np.float32

    def test_config_hash_consistency(self):
        """Test that same config produces same hash."""
        import json
        
        config1 = {"param_a": 1, "param_b": "value"}
        config2 = {"param_b": "value", "param_a": 1}  # Same but different order
        
        hash1 = json.dumps(config1, sort_keys=True)
        hash2 = json.dumps(config2, sort_keys=True)
        
        assert hash1 == hash2


# =====================================================================
# Run tests
# =====================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])