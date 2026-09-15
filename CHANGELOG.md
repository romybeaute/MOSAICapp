# Changelog

Versions follow [semantic versioning](https://semver.org/). Each GitHub release is
archived on Zenodo; the concept DOI
[10.5281/zenodo.18394316](https://doi.org/10.5281/zenodo.18394316) always resolves to
the latest version.

## [2.0.0] — 2026-09-15

### Changed (breaking)
- Single entry point: run `streamlit run app.py`. The former extended interface
  (`app2.py`) is now `app.py`; the former reduced `app.py` has been removed.
- Zero-Shot Classification and Condition Comparison are behind an **Advanced analyses**
  toggle in the sidebar (or `?advanced=1` in the URL).
- `pip install .` installs the analysis library only; the Streamlit interface and its
  plotting libraries are the `[app]` extra.

### Added
- `mosaic_core.zeroshot`: zero-shot classification with similarity and margin thresholds.
- `mosaic_core.comparison`: per-condition centred cosine between topic solutions,
  median/MAD-calibrated match threshold, row/column z-scores against hub topics,
  one-to-one greedy matching, χ² and Cramér's V on shared themes.
- `mosaic_core.metrics`: topic diversity ratio, embedding coherence (C_embed), C_v.
- Tests for all of the above, and a CI job that installs and tests the library without
  Streamlit.
- Questionnaire presets (MPE-92, 11D-ASC) and saved category sets for zero-shot.
- Precomputed-embeddings data source, `run_embeddings.py` for GPU/HPC embedding, and a
  pinned `requirements.lock.txt`.

### Fixed
- Embedding coherence now follows the MOSAIC paper's definition (mean pairwise cosine of a
  topic's sentence embeddings) instead of a top-word proxy.
- Zero-shot hang caused by pickling the embedding model through Streamlit's cache.
- C_v crash from a tokenisation mismatch; matplotlib ≥ 3.9 colormap crash.

## [1.1.0] — 2026-01-27
- JOSS submission version: `mosaic_core` library, test suite, contributing guide, paper.

## [1.0.0] — 2026-01-19
- First archived release. (The git tag is spelled `v.1.0.0`.)

[2.0.0]: https://github.com/romybeaute/MOSAICapp/compare/v1.1.0...v2.0.0
[1.1.0]: https://github.com/romybeaute/MOSAICapp/compare/v.1.0.0...v1.1.0
[1.0.0]: https://github.com/romybeaute/MOSAICapp/releases/tag/v.1.0.0
