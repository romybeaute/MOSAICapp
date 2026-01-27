---
title: MOSAICapp
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
---


# MOSAICapp

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18394317.svg)](https://doi.org/10.5281/zenodo.18394317)

A web application for topic modelling of phenomenological reports using BERTopic and transformer embeddings.

**Web app:** [huggingface.co/spaces/romybeaute/MOSAICapp](https://huggingface.co/spaces/romybeaute/MOSAICapp)

## Statement of Need

Consciousness research increasingly relies on open-ended subjective reports to capture the richness of lived experience. Structured questionnaires like the Altered States of Consciousness scales or the MEQ impose predefined categories that can miss unexpected experiential dimensions.

MOSAICapp provides an alternative: instead of forcing reports into predefined categories, it uses neural topic modelling to discover thematic structure directly from the text. This "wide-angle" approach lets researchers see what participants actually describe before committing to a categorical framework.

The tool is designed for consciousness researchers, phenomenologists, and qualitative researchers working with text data who want computational analysis without writing code.

## Features

- **No-code interface** — upload CSV, configure parameters, download results
- **Sentence-level analysis** — optional segmentation for finer-grained themes
- **Interactive visualisations** — 2D topic maps, hierarchical clustering, topic distributions
- **LLM topic labelling** — automatic generation of interpretable labels (full version)
- **Python API** — `mosaic_core` library for programmatic use and batch processing

## Installation

### Web app (no installation)

Visit [huggingface.co/spaces/romybeaute/MOSAICapp](https://huggingface.co/spaces/romybeaute/MOSAICapp)

### Local installation

```bash
git clone https://github.com/romybeaute/MOSAICapp.git
cd MOSAICapp

# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies and the package
pip install -r requirements.txt
pip install .

# Download NLTK data (required for segmentation)
python -c "import nltk; nltk.download('punkt')"

# Run the app
streamlit run app.py
```

### Library usage

```python
from mosaic_core.core_functions import preprocess_and_embed, run_topic_model

docs, embeddings = preprocess_and_embed("data.csv", text_col="report")

config = {
    "umap_params": {"n_neighbors": 15, "n_components": 5},
    "hdbscan_params": {"min_cluster_size": 10},
    "bt_params": {"nr_topics": "auto"}
}

model, reduced_embeddings, topics = run_topic_model(docs, embeddings, config)
```

## Input format

CSV file with a text column. The app auto-detects columns named `text`, `report`, `reflection_answer`, or `reflection_answer_english`. Any column can also be selected manually.

## How it works

MOSAICapp implements a BERTopic pipeline: texts are embedded using sentence transformers, reduced with UMAP, clustered with HDBSCAN, and labelled using c-TF-IDF (with optional LLM refinement). This approach captures semantic context better than older bag-of-words methods like LDA.

For methodological details, see the [MOSAIC paper](https://arxiv.org/abs/2502.18318).

## Research applications

MOSAICapp has been used to analyse:

- Stroboscopic light experiences from the Dreamachine project
- Descriptions of "pure awareness" from the Minimal Phenomenal Experience study  
- Psychedelic experience reports (DMT, 5-MeO-DMT micro-phenomenological interviews)

## Citation

```bibtex
@article{beaute2025mosaic,
  title={Mapping of Subjective Accounts into Interpreted Clusters (MOSAIC): 
         Topic Modelling and LLM Applied to Stroboscopic Phenomenology},
  author={Beauté, Romy and Schwartzman, David J and Dumas, Guillaume and 
          Crook, Jennifer and Macpherson, Fiona and Barrett, Adam B and Seth, Anil K},
  journal={arXiv preprint arXiv:2502.18318},
  year={2025}
}
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on reporting bugs, suggesting features, and contributing code.

## Tests

**Run everything:**
```bash
pytest tests/ -v
```

**Run only fast tests:**
```bash
pytest tests/test_core_functions.py -v
```

## License

MIT

## Acknowledgements

Built with [BERTopic](https://github.com/MaartenGr/BERTopic) by Maarten Grootendorst. Funded by the Be.AI Leverhulme doctoral scholarship at the Sussex Centre for Consciousness Science.