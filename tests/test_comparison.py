"""Tests for mosaic_core.comparison (centred cosine, MAD threshold, matching)."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from mosaic_core.comparison import (
    MAD_TO_SD,
    calibrate_match_threshold,
    compute_condition_similarity,
    describe_threshold,
    greedy_match,
    pair_zscores,
    parse_condition_csv,
    robust_loc_scale,
    shared_theme_chi2,
)


def _lookup_embedder(vectors: dict):
    """Deterministic stand-in for a sentence-transformer."""
    return lambda sentences: np.array([vectors[s] for s in sentences], dtype=float)


def _sim(values, rows=None, cols=None):
    values = np.asarray(values, dtype=float)
    rows = rows or [f"a{i}" for i in range(values.shape[0])]
    cols = cols or [f"b{j}" for j in range(values.shape[1])]
    return pd.DataFrame(values, index=rows, columns=cols)


class TestParseConditionCsv:
    def test_row_per_topic_format(self):
        df = pd.DataFrame({
            "Topic": [-1, 0, 1, 2],
            "topic_name": ["Noise", "Colours", "Unlabelled", "Calm"],
            "texts": ["x | y", "red things | blue things", "z", "slow breath |  | quiet"],
        })
        out = parse_condition_csv(df)
        # topic -1 and "Unlabelled" are excluded, empty pieces are dropped
        assert out == {"Colours": ["red things", "blue things"],
                       "Calm": ["slow breath", "quiet"]}

    def test_long_format(self):
        df = pd.DataFrame({"Topic Name": ["A", "A", "Outlier", "B"],
                           "Document": ["s1", "s2", "noise", " s3 "]})
        assert parse_condition_csv(df) == {"A": ["s1", "s2"], "B": ["s3"]}

    def test_unknown_format_returns_empty(self):
        assert parse_condition_csv(pd.DataFrame({"foo": [1]})) == {}

    def test_pipe_inside_sentence_is_not_a_separator(self):
        # Only the literal " | " separator splits.
        df = pd.DataFrame({"topic_name": ["A"], "texts": ["either|or | second"]})
        assert parse_condition_csv(df) == {"A": ["either|or", "second"]}


class TestComputeConditionSimilarity:
    def setup_method(self):
        rng = np.random.default_rng(42)
        self.vectors = {f"s{i}": rng.normal(size=12) for i in range(60)}
        self.topics_a = {"A1": ["s0", "s1", "s2"], "A2": ["s3", "s4"], "A3": ["s5", "s6", "s7"]}
        self.topics_b = {"B1": ["s10", "s11"], "B2": ["s12", "s13", "s14"],
                         "B3": ["s15"], "B4": ["s16", "s17"]}

    def test_shapes_and_labels(self):
        sim, raw, va, vb = compute_condition_similarity(
            self.topics_a, self.topics_b, _lookup_embedder(self.vectors))
        assert sim.shape == raw.shape == (3, 4)
        assert list(sim.index) == ["A1", "A2", "A3"]
        assert list(sim.columns) == ["B1", "B2", "B3", "B4"]
        assert va.shape == (3, 12) and vb.shape == (4, 12)
        assert np.all(sim.values <= 1 + 1e-12) and np.all(sim.values >= -1 - 1e-12)

    def test_matches_manual_centred_cosine(self):
        emb = _lookup_embedder(self.vectors)
        sim, _, _, _ = compute_condition_similarity(self.topics_a, self.topics_b, emb)

        def centred(topics):
            all_vecs = np.vstack([self.vectors[s] for t in topics.values() for s in t])
            c = all_vecs.mean(axis=0)
            return {n: np.mean([self.vectors[s] for s in t], axis=0) - c for n, t in topics.items()}

        ca, cb = centred(self.topics_a), centred(self.topics_b)
        a, b = ca["A2"], cb["B4"]
        expected = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
        assert sim.loc["A2", "B4"] == pytest.approx(expected)

    def test_invariant_to_global_shift_of_one_condition(self):
        """Per-condition centring removes any content shift shared by a whole condition."""
        shift = np.full(12, 5.0)
        shifted = dict(self.vectors)
        for t in self.topics_b.values():
            for s in t:
                shifted[s] = self.vectors[s] + shift
        sim1, raw1, _, _ = compute_condition_similarity(
            self.topics_a, self.topics_b, _lookup_embedder(self.vectors))
        sim2, raw2, _, _ = compute_condition_similarity(
            self.topics_a, self.topics_b, _lookup_embedder(shifted))
        np.testing.assert_allclose(sim1.values, sim2.values, atol=1e-10)
        # ...whereas the uncentred score is affected.
        assert not np.allclose(raw1.values, raw2.values)

    def test_independent_of_other_condition_size(self):
        """Adding sentences to B must not change A's topic vectors (no pooled centring)."""
        emb = _lookup_embedder(self.vectors)
        _, _, va1, _ = compute_condition_similarity(self.topics_a, self.topics_b, emb)
        bigger_b = dict(self.topics_b, B5=[f"s{i}" for i in range(20, 60)])
        _, _, va2, _ = compute_condition_similarity(self.topics_a, bigger_b, emb)
        np.testing.assert_allclose(va1, va2)

    def test_duplicate_sentences_are_embedded_once(self):
        calls = []

        def embed(sentences):
            calls.append(list(sentences))
            return np.array([self.vectors[s] for s in sentences])

        compute_condition_similarity({"A": ["s0", "s1"]}, {"B": ["s1", "s2"], "C": ["s0"]}, embed)
        assert len(calls) == 1
        assert sorted(calls[0]) == ["s0", "s1", "s2"]

    def test_empty_input(self):
        sim, raw, _, _ = compute_condition_similarity({}, {}, _lookup_embedder({}))
        assert sim.empty and raw.empty


class TestRobustLocScale:
    def test_median_and_mad(self):
        v = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        med, sd = robust_loc_scale(v)
        assert med == 3.0
        # |v - 3| = [2, 1, 0, 1, 97] -> MAD = 1
        assert sd == pytest.approx(MAD_TO_SD * 1.0)

    def test_matches_normal_sd_on_large_sample(self):
        rng = np.random.default_rng(0)
        med, sd = robust_loc_scale(rng.normal(loc=0.2, scale=0.1, size=200_000))
        assert med == pytest.approx(0.2, abs=2e-3)
        assert sd == pytest.approx(0.1, rel=0.02)

    def test_resistant_to_outliers(self):
        rng = np.random.default_rng(1)
        clean = rng.normal(0, 0.1, size=1000)
        dirty = np.concatenate([clean, np.full(20, 0.9)])
        _, sd_clean = robust_loc_scale(clean)
        _, sd_dirty = robust_loc_scale(dirty)
        assert sd_dirty == pytest.approx(sd_clean, rel=0.1)

    def test_ignores_non_finite(self):
        assert robust_loc_scale([1.0, np.nan, 2.0, np.inf, 3.0]) == robust_loc_scale([1.0, 2.0, 3.0])

    def test_zero_mad_falls_back_to_std(self):
        v = np.array([1.0, 1.0, 1.0, 5.0])
        med, sd = robust_loc_scale(v)
        assert med == 1.0
        assert sd == pytest.approx(np.std(v))

    def test_constant_and_empty_fall_back_to_one(self):
        assert robust_loc_scale(np.full(5, 0.3)) == (0.3, 1.0)
        assert robust_loc_scale(np.array([])) == (0.0, 1.0)


class TestCalibrateMatchThreshold:
    def test_threshold_is_median_plus_k_robust_sd(self):
        rng = np.random.default_rng(3)
        sim = _sim(rng.normal(0, 0.1, size=(20, 25)))
        med, sd = robust_loc_scale(sim.values)
        for k in (3.0, 3.5, 4.0):
            calib = calibrate_match_threshold(sim, k)
            assert calib["threshold"] == pytest.approx(med + k * sd)
            assert calib["median"] == med and calib["robust_sd"] == sd
            assert calib["tail"] == pytest.approx(norm.sf(k))

    def test_diagnostics(self):
        rng = np.random.default_rng(4)
        sim = _sim(rng.normal(0, 0.1, size=(10, 30)))
        calib = calibrate_match_threshold(sim, 3.5)
        assert calib["n_pairs"] == 300
        assert calib["max_possible"] == 10
        expected = 10 * (1 - (1 - norm.sf(3.5)) ** 30)
        assert calib["expected_chance"] == pytest.approx(expected)
        assert calib["background_pct"] == pytest.approx(
            100 * (sim.values < calib["threshold"]).mean())

    def test_higher_k_is_stricter(self):
        rng = np.random.default_rng(5)
        sim = _sim(rng.normal(0, 0.1, size=(15, 15)))
        t = [calibrate_match_threshold(sim, k)["threshold"] for k in (3.0, 3.5, 4.0)]
        assert t[0] < t[1] < t[2]

    def test_genuine_matches_clear_the_threshold(self):
        rng = np.random.default_rng(6)
        values = rng.normal(0, 0.1, size=(12, 12))
        np.fill_diagonal(values[:5, :5], 0.9)  # five true correspondences
        calib = calibrate_match_threshold(_sim(values), 3.5)
        assert np.all(np.diag(values)[:5] > calib["threshold"])
        # the background itself (almost) never does
        off = values[~np.eye(12, dtype=bool)]
        assert (off > calib["threshold"]).mean() < 0.01

    def test_empty_matrix(self):
        calib = calibrate_match_threshold(pd.DataFrame(), 3.5)
        assert calib["n_pairs"] == 0 and calib["expected_chance"] == 0.0

    def test_describe_threshold_inverts_calibration(self):
        rng = np.random.default_rng(7)
        sim = _sim(rng.normal(0, 0.1, size=(8, 9)))
        calib = calibrate_match_threshold(sim, 3.2)
        described = describe_threshold(sim, calib["threshold"])
        assert described["k"] == pytest.approx(3.2)
        assert described["tail"] == pytest.approx(calib["tail"])
        assert described["expected_chance"] == pytest.approx(calib["expected_chance"])


class TestPairZscores:
    def test_hub_topic_is_penalised(self):
        # a0 scores high against every b ("hub"); a1 has one specific match with b1.
        sim = _sim([[0.60, 0.62, 0.61, 0.60, 0.59],
                    [0.00, 0.60, 0.02, -0.01, 0.01],
                    [0.01, -0.02, 0.00, 0.02, -0.01],
                    [-0.01, 0.01, 0.02, 0.00, 0.00],
                    [0.02, 0.00, -0.01, 0.01, 0.02]])
        z = pair_zscores(sim)
        assert z.shape == sim.shape
        # both pairs have a high raw score and stand out in column b1 ...
        assert sim.loc["a0", "b1"] >= sim.loc["a1", "b1"]
        # ... but only the specific pair also stands out in its own row
        assert z.loc["a1", "b1"] > 2.0
        assert z.loc["a0", "b1"] < 2.0
        matches, _, _ = greedy_match(sim, threshold=0.5, z_df=z, min_z=2.0)
        assert [(m["a"], m["b"]) for m in matches] == [("a1", "b1")]

    def test_is_min_of_row_and_column_z(self):
        rng = np.random.default_rng(8)
        sim = _sim(rng.normal(size=(5, 6)))
        z = pair_zscores(sim)
        i, j = 2, 3
        mr, sr = robust_loc_scale(sim.values[i, :])
        mc, sc = robust_loc_scale(sim.values[:, j])
        expected = min((sim.values[i, j] - mr) / sr, (sim.values[i, j] - mc) / sc)
        assert z.iat[i, j] == pytest.approx(expected)

    def test_empty(self):
        assert pair_zscores(pd.DataFrame()).empty


class TestGreedyMatch:
    def test_one_to_one_in_descending_score_order(self):
        sim = _sim([[0.9, 0.8, 0.1],
                    [0.85, 0.2, 0.1],
                    [0.1, 0.1, 0.7]])
        matches, ua, ub = greedy_match(sim, threshold=0.5)
        pairs = [(m["a"], m["b"]) for m in matches]
        # a0-b0 is taken first, so a1 cannot also take b0 even though 0.85 > 0.8
        assert pairs == [("a0", "b0"), ("a2", "b2")]
        assert ua == ["a1"] and ub == ["b1"]
        assert [m["score"] for m in matches] == sorted([m["score"] for m in matches], reverse=True)

    def test_each_topic_used_at_most_once(self):
        rng = np.random.default_rng(9)
        sim = _sim(rng.uniform(0, 1, size=(7, 11)))
        matches, ua, ub = greedy_match(sim, threshold=0.0)
        a_used = [m["a"] for m in matches]
        b_used = [m["b"] for m in matches]
        assert len(set(a_used)) == len(a_used) and len(set(b_used)) == len(b_used)
        assert len(matches) == 7  # min(shape) when everything clears the threshold
        assert set(a_used) | set(ua) == set(sim.index)
        assert set(b_used) | set(ub) == set(sim.columns)

    def test_stops_at_threshold(self):
        sim = _sim([[0.9, 0.1], [0.1, 0.49]])
        matches, ua, ub = greedy_match(sim, threshold=0.5)
        assert [(m["a"], m["b"]) for m in matches] == [("a0", "b0")]
        assert ua == ["a1"] and ub == ["b1"]

    def test_threshold_is_inclusive(self):
        sim = _sim([[0.5]])
        matches, _, _ = greedy_match(sim, threshold=0.5)
        assert len(matches) == 1

    def test_z_rejection_keeps_both_topics_available(self):
        sim = _sim([[0.9, 0.8],
                    [0.7, 0.1]])
        z = _sim([[0.5, 3.0],    # a0-b0 has the top score but is not specific
                  [3.0, 3.0]])
        matches, ua, ub = greedy_match(sim, threshold=0.5, z_df=z, min_z=2.0)
        pairs = {(m["a"], m["b"]) for m in matches}
        # a0 and b0 are both still matched, just not to each other
        assert pairs == {("a0", "b1"), ("a1", "b0")}
        assert ua == [] and ub == []

    def test_reciprocal_best_flag(self):
        sim = _sim([[0.9, 0.3],
                    [0.95, 0.6]])
        matches, _, _ = greedy_match(sim, threshold=0.5)
        by_pair = {(m["a"], m["b"]): m for m in matches}
        # a1's best is b0 and b0's best is a1 -> reciprocal
        assert by_pair[("a1", "b0")]["reciprocal_best"] is True
        # a0's best is b0, but a0-b0 was taken; a0 never matched b1 (0.3 < 0.5)
        assert ("a0", "b1") not in by_pair
        m, _, _ = greedy_match(sim, threshold=0.0)
        flags = {(x["a"], x["b"]): x["reciprocal_best"] for x in m}
        assert flags[("a0", "b1")] is False

    def test_non_finite_scores_are_never_matched(self):
        sim = _sim([[np.nan, 0.6], [0.7, np.inf]])
        matches, _, _ = greedy_match(sim, threshold=0.5)
        assert {(m["a"], m["b"]) for m in matches} == {("a1", "b0"), ("a0", "b1")}

    def test_empty(self):
        assert greedy_match(pd.DataFrame(), 0.5) == ([], [], [])


class TestSharedThemeChi2:
    def test_matches_scipy_and_cramers_v(self):
        table = np.array([[30, 10], [10, 30], [20, 20]])
        res = shared_theme_chi2(table)
        from scipy.stats import chi2_contingency
        chi2, p, dof, _ = chi2_contingency(table)
        assert res["chi2"] == pytest.approx(chi2) and res["p"] == pytest.approx(p)
        assert res["dof"] == dof == 2
        assert res["cramers_v"] == pytest.approx(np.sqrt(chi2 / (120 * 1)))
        assert res["n"] == 120 and res["n_themes"] == 3

    def test_small_expected_fraction(self):
        assert shared_theme_chi2([[30, 10], [10, 30]])["small_expected_frac"] == 0.0
        assert shared_theme_chi2([[3, 1], [1, 3], [50, 50]])["small_expected_frac"] == pytest.approx(4 / 6)

    def test_identical_proportions_give_zero_effect(self):
        res = shared_theme_chi2([[10, 20], [30, 60]])
        assert res["cramers_v"] == pytest.approx(0.0, abs=1e-12)

    def test_drops_empty_rows(self):
        assert shared_theme_chi2([[10, 5], [0, 0], [5, 10]])["n_themes"] == 2

    @pytest.mark.parametrize("table", [[[10, 5]], [[10, 0], [5, 0]], [[1, 2, 3]]])
    def test_untestable_tables_raise(self, table):
        with pytest.raises(ValueError):
            shared_theme_chi2(table)
