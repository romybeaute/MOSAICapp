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



---

## 1. Quick Start (No Installation)

The easiest way to use MOSAICapp is via the hosted web interface. No coding or installation is required.

**[Launch MOSAICapp on Hugging Face](https://huggingface.co/spaces/romybeaute/MOSAICapp)**

*Note: The hosted version runs on shared resources. For large datasets or privacy-sensitive data, we recommend the local installation below.*


---

## 2. Local Installation

Run the app on your own machine to use custom GPUs, process sensitive data locally, or modify the code.

### Prerequisites
- Python 3.9+
- Git


### Setup steps

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
```

---

## 3. Configuration & Running


### Run the app
```
streamlit run app.py
```

### LLM Setup (Optional)
To use the Automated Topic Labelling feature (Llama-3), you must provide a Hugging Face Access Token. The app uses this token to access the inference API.

1. Get a Token: Log in to Hugging Face and create a token with "Read" permissions.

2. Configure Local App:

- Create a folder named .streamlit in your root directory.

- Inside it, create a file named secrets.toml.

- Add your token in TOML file:
```
HF_TOKEN = "hf_..."
```

- Note: This file is ignored by Git to protect your credentials.


---

## 4. Running Tests
We include a test suite to verify the installation and core logic. This is useful to check if your environment is set up correctly.

**Run everything:**
```bash
pytest tests/ -v
```

**Run only fast tests:**
```bash
pytest tests/test_core_functions.py -v
```

This will automatically load a dummy dataset included in the repo and verify:

- Data loading (CSV parsing)

- Embedding generation

- Topic modelling pipeline

- Visualisation outputs

---

## 5. Python API (Advanced Usage)
MOSAICapp is also a Python library. You can import `mosaic_core` in your own scripts or Jupyter Notebooks for batch processing or custom analysis pipelines.

### Library usage
```python
from mosaic_core.core_functions import preprocess_and_embed, run_topic_model

# 1. Load and Preprocess
docs, embeddings = preprocess_and_embed("data.csv", text_col="report")

# 2. Configure Parameters
config = {
    "umap_params": {"n_neighbors": 15, "n_components": 5},
    "hdbscan_params": {"min_cluster_size": 10},
    "bt_params": {"nr_topics": "auto"}
}

# 3. Run Model
model, reduced_embeddings, topics = run_topic_model(docs, embeddings, config)
```





## Input format

CSV file with a text column. The app auto-detects columns named `text`, `report`, `reflection_answer`, or `reflection_answer_english`. Any column can also be selected manually.


---


## How it works

MOSAICapp implements a BERTopic pipeline: texts are embedded using sentence transformers, reduced with UMAP, clustered with HDBSCAN, and labelled using c-TF-IDF (with optional LLM refinement). This approach captures semantic context better than older bag-of-words methods like LDA.

For methodological details, see the [MOSAIC paper](https://arxiv.org/abs/2502.18318).



---

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



## License

MIT

## Acknowledgements

Built with [BERTopic](https://github.com/MaartenGr/BERTopic) by Maarten Grootendorst. Funded by the Be.AI Leverhulme doctoral scholarship at the Sussex Centre for Consciousness Science.