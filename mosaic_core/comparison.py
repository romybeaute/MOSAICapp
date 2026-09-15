"""
Comparison of topic solutions obtained separately on two conditions.

Given the topics of condition A and condition B (each a mapping
``{topic_name: [sentence, ...]}``), this module

1. scores every A x B topic pair with a per-condition *centred* cosine
   (`compute_condition_similarity`),
2. derives a match threshold from the comparison's own background distribution
   using a robust median / MAD estimate (`calibrate_match_threshold`),
3. flags "hub" topics that score highly against everything (`pair_zscores`),
4. pairs topics one-to-one (`greedy_match`), and
5. tests whether the shared themes are used in different proportions
   (`shared_theme_chi2`).

No Streamlit dependencies.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, norm
from sklearn.metrics.pairwise import cosine_similarity

__all__ = [
    "MAD_TO_SD",
    "STRINGENCY_LEVELS",
    "EXCLUDED_TOPIC_NAMES",
    "parse_condition_csv",
    "compute_condition_similarity",
    "robust_loc_scale",
    "calibrate_match_threshold",
    "describe_threshold",
    "pair_zscores",
    "greedy_match",
    "shared_theme_chi2",
]

# Topic names that are not themes and must never take part in matching.
EXCLUDED_TOPIC_NAMES = ("Unlabelled", "Outlier", "Too Specific (Idiosyncratic)", "")


# ── Input parsing ────────────────────────────────────────────────────────────

def parse_condition_csv(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Accept two CSV formats exported by the main pipeline and return
    {topic_name: [sentence, sentence, ...]}

    Supported formats:
    - "Row per topic"   : columns include `topic_name` and `texts` (joined with " | ")
    - "Long / all-sentences" : columns include `Topic Name` and `Document`

    Returns an empty dict if neither format is recognised.
    """
    if "topic_name" in df.columns and "texts" in df.columns:
        out: dict[str, list[str]] = {}
        for _, row in df.iterrows():
            # Topic -1 is BERTopic's outlier bin: a grab-bag of unrelated sentences,
            # not a theme. Its mean vector is meaningless, so never match on it.
            if "Topic" in df.columns and pd.to_numeric(row["Topic"], errors="coerce") == -1:
                continue
            name = str(row["topic_name"]).strip()
            if name in EXCLUDED_TOPIC_NAMES:
                continue
            texts_raw = row["texts"]
            if not isinstance(texts_raw, str) or not texts_raw.strip():
                continue
            sentences = [s.strip() for s in texts_raw.split(" | ") if s.strip()]
            if sentences:
                out[name] = sentences
        return out
    if "Topic Name" in df.columns and "Document" in df.columns:
        out = {}
        for _, row in df.iterrows():
            name = str(row["Topic Name"]).strip()
            if name in EXCLUDED_TOPIC_NAMES:
                continue
            sentence = str(row["Document"]).strip()
            if sentence:
                out.setdefault(name, []).append(sentence)
        return out
    return {}


# ── Centred cosine similarity ────────────────────────────────────────────────

def compute_condition_similarity(
    topics_a: dict,
    topics_b: dict,
    embed: Callable[[list[str]], np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Given two dicts of {topic_name: [sentences]}, embed all sentences, build mean
    topic vectors, and return ``(sim_df, raw_df, vecs_a, vecs_b)``:

    - ``sim_df``: centred cosine, topics of A as rows, topics of B as columns.
      All matching decisions use this score.
    - ``raw_df``: uncentred cosine, same shape, for interpretation only.
    - ``vecs_a``, ``vecs_b``: the centred topic vectors.

    `embed` maps a list of unique sentences to an (n, dim) array, e.g.
    ``SentenceTransformer(model_name).encode``.

    Scores are *centered* cosines. Raw sentence-transformer embeddings are
    anisotropic — they occupy a narrow cone, so every pair of topics scores
    ~0.6–0.98 and unrelated themes look "strongly correlated" (measured on real
    data: min raw cosine across a 36x43 topic matrix was 0.55). Centering removes
    that shared component and restores a discriminative range.

    Centering is done **per condition** (each condition's sentences are centered
    on that condition's own centroid) rather than on the pooled centroid, for two
    reasons:

    1. Reproducibility. The pooled centroid is a size-weighted blend of the two
       conditions, so the *same* pair of topics scores differently depending on
       how many sentences the other condition happens to contain.
    2. Interpretability. Each topic vector becomes that topic's deviation from
       its own condition's average, and the size-weighted mean of those deviations
       is exactly zero in both conditions. The background therefore sits *near*
       zero (measured median on real data: -0.01), which is what makes the scale
       discriminative.

    Two things this score is not. It is not a Pearson correlation: correlation is
    the cosine of vectors centred across the dimensions being dotted, whereas the
    centroid subtracted here is a per-dimension mean over sentences. And zero is
    an empirical, not an exact, no-correspondence point — the size-weighted mean
    of the *unnormalised* deviations vanishes, but the mean of their pairwise
    cosines does not. Nothing downstream assumes it does: `calibrate_match_threshold`
    estimates the background median from the matrix rather than fixing it at 0.

    Note also that centring per condition removes the between-condition mean
    difference by construction. That is what makes topics comparable on their
    within-condition profiles, but it means a *global* content shift between the
    two conditions is invisible to this score.
    """
    names_a = list(topics_a.keys())
    names_b = list(topics_b.keys())

    # Single encode pass over the union of both conditions' sentences.
    occ_a = [s for name in names_a for s in topics_a[name]]
    occ_b = [s for name in names_b for s in topics_b[name]]
    unique_sents = list(dict.fromkeys(occ_a + occ_b))
    if not unique_sents:
        return pd.DataFrame(), pd.DataFrame(), np.empty((0, 0)), np.empty((0, 0))

    embeddings = np.asarray(embed(unique_sents), dtype=np.float64)
    pos = {s: i for i, s in enumerate(unique_sents)}

    def _centroid(occurrences):
        """Centroid over sentence *occurrences* (duplicates counted), so that it
        matches the weighting used by the topic means below."""
        idx = [pos[s] for s in occurrences if s in pos]
        return embeddings[idx].mean(axis=0) if idx else np.zeros(embeddings.shape[1])

    centroid_a = _centroid(occ_a)
    centroid_b = _centroid(occ_b)

    def _topic_vecs(topics, names, centroid):
        """Return centered *and* uncentered topic means (same topic order)."""
        vecs, raw, valid = [], [], []
        for name in names:
            idx = [pos[s] for s in topics[name] if s in pos]
            if not idx:
                continue
            mean_vec = embeddings[idx].mean(axis=0)
            raw.append(mean_vec)
            vecs.append(mean_vec - centroid)
            valid.append(name)
        return np.array(vecs), np.array(raw), valid

    vecs_a, raw_a, valid_a = _topic_vecs(topics_a, names_a, centroid_a)
    vecs_b, raw_b, valid_b = _topic_vecs(topics_b, names_b, centroid_b)
    if len(valid_a) == 0 or len(valid_b) == 0:
        return pd.DataFrame(), pd.DataFrame(), vecs_a, vecs_b

    sim_df = pd.DataFrame(cosine_similarity(vecs_a, vecs_b), index=valid_a, columns=valid_b)
    # Uncentered cosine, reported alongside for interpretation only. It answers
    # "how much content do these two topics share in absolute terms", but it is
    # useless as a matching rule: sentence-transformer embeddings are anisotropic,
    # so on real data every pair scores ~0.55-0.98 and the ranking is dominated by
    # how generic a topic is. Matching decisions use the centered score.
    raw_df = pd.DataFrame(cosine_similarity(raw_a, raw_b), index=valid_a, columns=valid_b)
    return sim_df, raw_df, vecs_a, vecs_b


# ── Threshold calibration ────────────────────────────────────────────────────
# A fixed cut-off like "0.50" is not portable: its meaning shifts with the
# embedding model, the corpus and the granularity. Instead, calibrate against the
# comparison's own background.
#
# Rationale for the null: at most min(n_a, n_b) of the n_a x n_b cells can be
# genuine correspondences, so >=95% of the matrix is by construction
# non-corresponding pairs. Robust location/scale (median / MAD) of the whole
# matrix therefore estimate the "no correspondence" distribution almost
# unaffected by the handful of real matches sitting in the tail.
#
# Rejected alternative: a permutation null that reshuffles sentences between
# topics within each condition. Measured on real data it sits at mean 0.00 with
# sd 0.12, so it flags ~15% of all pairs at p<0.05 and admits obvious
# non-matches (e.g. "Sudden perception of deeper reality" vs "Uncontrollable
# laughter", sim 0.27, p=0.012). It tests "more alike than two random sentence
# bags", which is far weaker than "about the same theme" — much too lenient.

# Consistency constant: 1.4826 x MAD estimates the SD of a normal distribution.
MAD_TO_SD = 1.4826

# Stringency presets: k = how many robust SDs above the background median.
# Validated against hand-labelled matches on a 36x43 real comparison, where the
# best achievable F1 sat at a cut-off of ~0.62; k=3.5 reproduces that value
# (0.627) without hard-coding it. That is a single dataset, so treat 3.5 as a
# sensible starting point rather than a validated constant.
STRINGENCY_LEVELS = {
    "Lenient (k = 3.0)": 3.0,
    "Balanced (k = 3.5) — recommended": 3.5,
    "Strict (k = 4.0)": 4.0,
}


def robust_loc_scale(values: np.ndarray) -> tuple[float, float]:
    """Median and MAD-based SD estimate, with fallbacks for degenerate input.

    Non-finite values are ignored. If the MAD is zero (e.g. a 1xN matrix or
    mostly tied values) the plain SD is used instead, and if that is zero too
    the scale falls back to 1.0 so that z-scores stay finite.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    med = float(np.median(v))
    sd = MAD_TO_SD * float(np.median(np.abs(v - med)))
    if not np.isfinite(sd) or sd <= 1e-9:          # e.g. a 1xN matrix
        sd = float(np.std(v))
    if not np.isfinite(sd) or sd <= 1e-9:
        sd = 1.0
    return med, sd


def _expected_chance_matches(tail: float, n_a: int, n_b: int) -> float:
    """Expected chance matches under one-to-one (best-hit) matching: each of the
    n_a topics gets one shot at n_b candidates, so P(its best hit clears the
    bar by chance) = 1 - (1 - tail)^n_b. Capped at the maximum possible."""
    if not (n_a and n_b):
        return 0.0
    return float(min(n_a * (1.0 - (1.0 - tail) ** n_b), float(min(n_a, n_b))))


def calibrate_match_threshold(sim_df: pd.DataFrame, k: float) -> dict:
    """Derive a match threshold from the background distribution of `sim_df`.

    threshold = median(sim_df) + k * 1.4826 * MAD(sim_df)

    Returns the threshold plus the diagnostics needed to judge whether it is too
    lenient or too strict: ``threshold, k, median, robust_sd, tail`` (one-sided
    normal tail probability of k, i.e. the per-pair false-positive rate),
    ``expected_chance`` (expected number of chance matches), ``background_pct``
    (% of pairs below the threshold), ``n_pairs`` and ``max_possible``.
    """
    n_a, n_b = (sim_df.shape if sim_df.size else (0, 0))
    med, sd = robust_loc_scale(sim_df.values if sim_df.size else np.array([]))
    threshold = med + k * sd
    tail = float(norm.sf(k))                       # per-pair false-positive rate
    expected_chance = _expected_chance_matches(tail, n_a, n_b)

    vals = sim_df.values[np.isfinite(sim_df.values)] if sim_df.size else np.array([])
    background_pct = float((vals < threshold).mean() * 100) if vals.size else 0.0
    return {
        "threshold": float(threshold),
        "k": float(k),
        "median": med,
        "robust_sd": sd,
        "tail": tail,
        "expected_chance": float(expected_chance),
        "background_pct": background_pct,
        "n_pairs": int(n_a * n_b),
        "max_possible": int(min(n_a, n_b)) if n_a and n_b else 0,
    }


def describe_threshold(sim_df: pd.DataFrame, threshold: float) -> dict:
    """Express a manually chosen threshold on the calibrated scale.

    Returns the same keys as `calibrate_match_threshold`, with ``k`` being the
    number of robust SDs the given threshold sits above the background median.
    """
    calib = calibrate_match_threshold(sim_df, 3.5)
    calib["k"] = (threshold - calib["median"]) / calib["robust_sd"]
    calib["tail"] = float(norm.sf(calib["k"]))
    n_a, n_b = sim_df.shape
    calib["expected_chance"] = _expected_chance_matches(calib["tail"], n_a, n_b)
    calib["threshold"] = float(threshold)
    calib["background_pct"] = float((sim_df.values < threshold).mean() * 100)
    return calib


def pair_zscores(sim_df: pd.DataFrame) -> pd.DataFrame:
    """Per-pair `z_min`: how far a score stands out within its own row *and* its
    own column, in robust SDs.

    A global threshold cannot catch a "hub" topic — a broad, generic theme that
    scores highly against everything in the other condition. Such a topic is not
    a specific correspondence. Requiring a pair to be an outlier in both its row
    and its column filters those out.
    """
    if sim_df.empty:
        return pd.DataFrame()
    S = sim_df.values
    z_row = np.empty_like(S, dtype=np.float64)
    z_col = np.empty_like(S, dtype=np.float64)
    for i in range(S.shape[0]):
        m, s = robust_loc_scale(S[i, :])
        z_row[i, :] = (S[i, :] - m) / s
    for j in range(S.shape[1]):
        m, s = robust_loc_scale(S[:, j])
        z_col[:, j] = (S[:, j] - m) / s
    return pd.DataFrame(np.minimum(z_row, z_col), index=sim_df.index, columns=sim_df.columns)


# ── Matching ─────────────────────────────────────────────────────────────────

def greedy_match(sim_df: pd.DataFrame, threshold: float,
                 z_df: pd.DataFrame | None = None, min_z: float | None = None):
    """One-to-one greedy matching: repeatedly take the highest remaining score.

    A pair is accepted if its score is ≥ `threshold` and, when `min_z` is given,
    its `z_min` (from `pair_zscores`) is ≥ `min_z`. A pair rejected on `z_min`
    alone is retired, but both of its topics remain available for other pairs.

    Returns (matches, unmatched_a, unmatched_b) where each match is a dict with
    keys ``a, b, score, z_min, reciprocal_best``.
    """
    unmatched_a = list(sim_df.index)
    unmatched_b = list(sim_df.columns)
    matches: list[dict] = []
    if sim_df.empty:
        return matches, unmatched_a, unmatched_b

    S = sim_df.values.astype(np.float64).copy()
    S[~np.isfinite(S)] = -np.inf
    # Reciprocal best hit is a property of the original matrix, so record it first.
    best_for_row = S.argmax(axis=1)
    best_for_col = S.argmax(axis=0)
    rows, cols = list(sim_df.index), list(sim_df.columns)
    work = S.copy()

    n_max = min(S.shape)
    while len(matches) < n_max:
        flat = int(np.argmax(work))
        i, j = divmod(flat, work.shape[1])
        score = float(work[i, j])
        if not np.isfinite(score) or score < threshold:
            break
        z_min = float(z_df.iat[i, j]) if z_df is not None and not z_df.empty else float("nan")
        if min_z is None or not np.isfinite(z_min) or z_min >= min_z:
            matches.append({
                "a": rows[i], "b": cols[j], "score": score, "z_min": z_min,
                "reciprocal_best": bool(best_for_row[i] == j and best_for_col[j] == i),
            })
            unmatched_a.remove(rows[i])
            unmatched_b.remove(cols[j])
            work[i, :] = -np.inf
            work[:, j] = -np.inf
        else:
            # Rejected on z_min only: retire this cell, leave both topics available.
            work[i, j] = -np.inf
    return matches, unmatched_a, unmatched_b


# ── Frequency comparison ─────────────────────────────────────────────────────

def shared_theme_chi2(table) -> dict:
    """χ² test and Cramér's V on a (n_shared_themes x 2) table of sentence counts.

    Only *paired* themes belong in the table. Rows for condition-specific topics
    contain a zero produced by modelling the two conditions separately, not by
    sampling: the topic solution for one condition simply has no such theme.
    Including those rows makes the test reject essentially always.

    Caveat that no restriction can fix: the counts are sentences, and sentences
    are nested within participants/reports, so χ² and p scale with a sample size
    inflated by within-participant repetition. Cramér's V divides n out and is
    the quantity to report; p is descriptive only.

    All-zero rows are dropped. Raises ValueError if fewer than 2 themes remain or
    one condition has no sentences in them.

    Returns a dict with ``chi2, p, dof, cramers_v, n, n_themes``.
    """
    table = np.asarray(table, dtype=float)
    if table.ndim != 2 or table.shape[1] != 2:
        raise ValueError("Expected a table of shape (n_themes, 2).")
    table = table[table.sum(axis=1) > 0]
    if len(table) < 2:
        raise ValueError("Need at least 2 shared themes to test whether their "
                         "proportions differ between conditions.")
    if table.sum(axis=0).min() == 0:
        raise ValueError("One condition has no sentences in the shared themes.")
    chi2_stat, p_val, dof, _ = chi2_contingency(table)
    n_obs = float(table.sum())
    cramers_v = float(np.sqrt(chi2_stat / (n_obs * (min(table.shape) - 1))))
    return {
        "chi2": float(chi2_stat),
        "p": float(p_val),
        "dof": int(dof),
        "cramers_v": cramers_v,
        "n": int(n_obs),
        "n_themes": int(len(table)),
    }
