"""Tests for mosaic_core.metrics (diversity ratio, C_embed, C_v)."""

import numpy as np
import pytest

from mosaic_core.metrics import embedding_coherence, topic_coherence_cv, topic_diversity


class TestTopicDiversity:
    def test_ratio_is_one_when_every_sentence_has_its_own_report(self):
        stats = topic_diversity([0, 0, 0], ["r1", "r2", "r3"])
        row = stats.iloc[0]
        assert row["Total_Sentences"] == 3 and row["Unique_Reports"] == 3
        assert row["Diversity_Ratio"] == 1.0

    def test_single_report_dominating_topic(self):
        stats = topic_diversity([1, 1, 1, 1], ["r1", "r1", "r1", "r1"]).set_index("Topic")
        assert stats.loc[1, "Diversity_Ratio"] == pytest.approx(0.25)

    def test_mixed_topics(self):
        topics = [0, 0, 0, 0, 1, 1, 2]
        reports = ["a", "a", "b", "c", "a", "b", "z"]
        stats = topic_diversity(topics, reports).set_index("Topic")
        assert stats.loc[0, "Diversity_Ratio"] == pytest.approx(3 / 4)
        assert stats.loc[1, "Diversity_Ratio"] == pytest.approx(1.0)
        assert stats.loc[2, "Total_Sentences"] == 1
        assert list(stats.columns) == ["Total_Sentences", "Unique_Reports", "Diversity_Ratio"]

    def test_outliers_are_excluded(self):
        stats = topic_diversity([-1, -1, 0], ["a", "b", "c"])
        assert list(stats["Topic"]) == [0]

    def test_ratio_bounds(self):
        rng = np.random.default_rng(0)
        topics = rng.integers(-1, 10, 1000)
        reports = rng.integers(0, 50, 1000)
        stats = topic_diversity(topics, reports)
        assert (stats["Diversity_Ratio"] > 0).all() and (stats["Diversity_Ratio"] <= 1).all()
        assert stats["Total_Sentences"].sum() == (topics != -1).sum()

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            topic_diversity([0, 1], ["a"])


def _brute_force_cembed(embeddings, topics):
    embeddings = np.asarray(embeddings, dtype=float)
    unit = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    per_topic = []
    for t in sorted(set(topics) - {-1}):
        u = unit[np.asarray(topics) == t]
        if len(u) < 2:
            continue
        s = u @ u.T
        per_topic.append(s[np.triu_indices(len(u), k=1)].mean())
    return float(np.mean(per_topic)) if per_topic else 0.0


class TestEmbeddingCoherence:
    def test_matches_brute_force_pairwise_mean(self):
        rng = np.random.default_rng(1)
        emb = rng.normal(size=(120, 16))
        topics = list(rng.integers(-1, 6, 120))
        assert embedding_coherence(emb, topics) == pytest.approx(_brute_force_cembed(emb, topics))

    def test_identical_sentences_score_one(self):
        emb = np.tile([1.0, 2.0, 3.0], (4, 1))
        assert embedding_coherence(emb, [0, 0, 0, 0]) == pytest.approx(1.0)

    def test_scale_invariant(self):
        rng = np.random.default_rng(2)
        emb = rng.normal(size=(30, 5))
        topics = [0] * 15 + [1] * 15
        assert embedding_coherence(emb * 7.5, topics) == pytest.approx(embedding_coherence(emb, topics))

    def test_singletons_and_outliers_ignored(self):
        emb = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        # topic 0 has two identical vectors (score 1); topic 1 is a singleton; -1 is ignored
        assert embedding_coherence(emb, [0, 0, 1, -1]) == pytest.approx(1.0)

    def test_no_qualifying_topic_returns_zero(self):
        assert embedding_coherence(np.eye(3), [-1, 0, 1]) == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            embedding_coherence(np.eye(3), [0, 0])


class TestTopicCoherenceCv:
    docs = [
        "bright colours and geometric patterns",
        "geometric patterns of bright light",
        "colours shifting into patterns",
        "calm slow breathing and relaxation",
        "deep relaxation and calm breathing",
        "breathing felt calm",
    ] * 5

    def test_returns_score_in_unit_interval(self):
        pytest.importorskip("gensim")
        score = topic_coherence_cv(self.docs, [["colours", "patterns", "geometric"],
                                               ["calm", "breathing", "relaxation"]])
        assert 0.0 <= score <= 1.0

    def test_oov_and_ngrams_do_not_crash(self):
        pytest.importorskip("gensim")
        score = topic_coherence_cv(self.docs, [["bright light", "Patterns", "unseenword"],
                                               ["calm", "breathing"]])
        assert np.isfinite(score)

    def test_no_usable_topic_returns_zero(self):
        pytest.importorskip("gensim")
        assert topic_coherence_cv(self.docs, [["unseenword"], []]) == 0.0
