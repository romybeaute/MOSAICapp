"""Tests for mosaic_core.zeroshot."""

import numpy as np
import pytest

from mosaic_core.zeroshot import (
    apply_zeroshot_threshold,
    compute_zeroshot_similarities,
    embeddings_fingerprint,
    parse_categories,
)


class TestParseCategories:
    def test_bare_labels(self):
        labels, line_labels, line_texts = parse_categories("Anxiety\nJoy\n")
        assert labels == ["Anxiety", "Joy"]
        assert line_labels == ["Anxiety", "Joy"]
        assert line_texts == ["Anxiety", "Joy"]

    def test_description_is_embedded_with_label(self):
        _, _, line_texts = parse_categories("Anxiety | feeling afraid")
        assert line_texts == ["Anxiety: feeling afraid"]

    def test_repeated_label_keeps_one_label_many_lines(self):
        labels, line_labels, _ = parse_categories("A | one\nB\nA | two")
        assert labels == ["A", "B"]
        assert line_labels == ["A", "B", "A"]

    def test_skips_blank_lines_and_empty_labels(self):
        labels, line_labels, _ = parse_categories("\n  \n | orphan description\nJoy")
        assert labels == ["Joy"]
        assert line_labels == ["Joy"]

    def test_empty_description_falls_back_to_label(self):
        _, _, line_texts = parse_categories("Joy |   ")
        assert line_texts == ["Joy"]


class TestEmbeddingsFingerprint:
    def test_stable_for_same_array(self):
        e = np.arange(40, dtype=float).reshape(10, 4)
        assert embeddings_fingerprint(e) == embeddings_fingerprint(e.copy())

    def test_differs_for_different_content_same_shape(self):
        e = np.zeros((10, 4))
        f = np.ones((10, 4))
        assert embeddings_fingerprint(e) != embeddings_fingerprint(f)


class TestComputeZeroshotSimilarities:
    def test_cosine_against_encoded_lines(self):
        docs = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
        encode = lambda texts: np.array([[1.0, 0.0], [0.0, 1.0]])
        sims = compute_zeroshot_similarities(docs, ["x", "y"], encode)
        assert sims.shape == (3, 2)
        np.testing.assert_allclose(sims[0], [1.0, 0.0])
        np.testing.assert_allclose(sims[1], [0.0, 1.0])
        np.testing.assert_allclose(sims[2], [np.sqrt(0.5), np.sqrt(0.5)])

    def test_dimension_mismatch_raises(self):
        docs = np.ones((3, 4))
        encode = lambda texts: np.ones((len(texts), 8))
        with pytest.raises(ValueError, match="mismatch"):
            compute_zeroshot_similarities(docs, ["x"], encode, model_name="m")


class TestApplyZeroshotThreshold:
    labels = ["A", "B", "C"]
    line_labels = ["A", "B", "C"]

    def test_assigns_best_label_above_threshold(self):
        sims = np.array([[0.9, 0.1, 0.2],
                         [0.1, 0.8, 0.3]])
        topics, info, per_doc = apply_zeroshot_threshold(sims, self.labels, self.line_labels, 0.5)
        assert topics == [0, 1]
        assert list(per_doc["best_category"]) == ["A", "B"]
        assert list(per_doc["runner_up"]) == ["C", "C"]

    def test_below_threshold_is_unclassified(self):
        sims = np.array([[0.4, 0.1, 0.2]])
        topics, info, per_doc = apply_zeroshot_threshold(sims, self.labels, self.line_labels, 0.5)
        assert topics == [-1]
        # best_category is reported regardless of the threshold
        assert per_doc["best_category"].iloc[0] == "A"
        assert info.set_index("Topic").loc[-1, "Count"] == 1

    def test_margin_leaves_ambiguous_docs_unclassified(self):
        sims = np.array([[0.80, 0.78, 0.1],    # ambiguous: margin 0.02
                         [0.80, 0.50, 0.1]])   # clear: margin 0.30
        topics, _, per_doc = apply_zeroshot_threshold(
            sims, self.labels, self.line_labels, min_similarity=0.5, min_margin=0.1)
        assert topics == [-1, 0]
        np.testing.assert_allclose(per_doc["margin"], [0.02, 0.30])

    def test_label_similarity_is_max_over_its_lines(self):
        labels = ["A", "B"]
        line_labels = ["A", "A", "B"]
        sims = np.array([[0.2, 0.9, 0.5]])
        topics, _, per_doc = apply_zeroshot_threshold(sims, labels, line_labels, 0.5)
        assert topics == [0]
        assert per_doc["confidence"].iloc[0] == pytest.approx(0.9)

    def test_single_label_has_no_margin(self):
        sims = np.array([[0.7], [0.2]])
        topics, _, per_doc = apply_zeroshot_threshold(sims, ["A"], ["A"], 0.5, min_margin=0.3)
        assert topics == [0, -1]
        assert per_doc["margin"].isna().all()
        assert list(per_doc["runner_up"]) == ["", ""]

    def test_topic_info_counts_sum_to_n_docs(self):
        rng = np.random.default_rng(0)
        sims = rng.uniform(0, 1, size=(200, 3))
        topics, info, _ = apply_zeroshot_threshold(sims, self.labels, self.line_labels, 0.6, 0.05)
        assert info["Count"].sum() == 200
        # only categories that received documents are listed, plus the -1 row
        assert (info.loc[info["Topic"] != -1, "Count"] > 0).all()
        assert info["Topic"].iloc[-1] == -1
        for _, row in info.iterrows():
            assert topics.count(row["Topic"]) == row["Count"]
