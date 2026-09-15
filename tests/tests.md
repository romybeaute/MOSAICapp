# MOSAICapp Test Suite

This directory contains the automated tests for the `mosaic_core` library. None of the
tests import Streamlit, and all of them except `test_integration.py` run offline with
small deterministic inputs (fake embeddings instead of real models).

## File Structure

### `conftest.py`
Shared **fixtures**, such as the path to `dummy_dataset.csv`.

### `test_core_functions.py` (fast)
Preprocessing and pipeline helpers: slugify, text-column detection, sentence splitting,
config hashing, label cleaning, LLM label caching, device resolution.

### `test_zeroshot.py` (fast)
Zero-shot classification: category parsing (`Label | description`, repeated labels),
the dimension-mismatch guard, and `apply_zeroshot_threshold` (similarity threshold,
margin rule for ambiguous documents, max-over-lines, single-category case, counts).

### `test_comparison.py` (fast)
Condition comparison:
- **Centred cosine:** agrees with a manual computation, is invariant to a global shift
  of one condition, does not depend on the other condition's size, embeds each unique
  sentence once.
- **MAD threshold:** `robust_loc_scale` (median, 1.4826 x MAD, outlier resistance,
  degenerate fallbacks), `calibrate_match_threshold` (median + k x robust SD,
  diagnostics, true matches clear it while background does not), `describe_threshold`.
- **Hub filtering:** `pair_zscores` penalises topics that score high against everything.
- **Matching:** `greedy_match` is one-to-one, stops at the threshold, keeps both topics
  available after a z-score rejection, flags reciprocal best hits, ignores non-finite scores.
- **Frequency test:** `shared_theme_chi2` agrees with SciPy and computes Cramér's V.
- **CSV parsing:** both export formats and the `" | "` separator.

### `test_metrics.py` (fast)
Diversity ratio (bounds, outlier exclusion), embedding coherence C_embed (agrees with a
brute-force pairwise mean, scale invariance) and C_v coherence (requires gensim).

### `test_no_streamlit.py` (fast)
Checks that `mosaic_core` and its submodules import with Streamlit unavailable.

### `test_integration.py` (slow)
Runs the real ML pipeline: downloads a sentence-transformers model, fits BERTopic on
dummy data and, if `HF_TOKEN` is set, calls the Hugging Face Inference API. Skipped
automatically when `CI=true`.

## How to Run

```bash
pip install -e ".[dev]"

pytest tests/ -v                          # everything
CI=true pytest tests/ -v                  # everything except the slow integration tests
pytest tests/test_comparison.py -v        # one file
```

## Continuous integration

`.github/workflows/tests.yml` runs two jobs on every push and pull request to `main`:

- **app-environment:** the full `requirements.txt` environment on Python 3.11.
- **library-only:** `pip install -e ".[dev]"` without Streamlit, on Python 3.10 and 3.12.
  It fails if Streamlit is installed, to show the library does not depend on it.
