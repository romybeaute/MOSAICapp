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

# Quick Start (No Installation)

The easiest way to use MOSAICapp is via the hosted web interface. No coding or installation is required.

**[Launch MOSAICapp on Hugging Face](https://huggingface.co/spaces/romybeaute/MOSAICapp)**

If the space is sleeping, contact me (r.beaut@sussex.ac.uk) to restart it.

*Note: The hosted version runs on shared resources. For large datasets or privacy-sensitive data, we recommend the local installation below.*


---

# Local Installation
## 1. Steps for local installation
Run the app on your own machine to use custom GPUs, process sensitive data locally, or modify the code.

### Prerequisites
- **Python 3.11–3.13** — recommended, and required for the pinned `requirements.lock.txt`
  (it ships NumPy 2.x, which needs Python ≥ 3.11). Python 3.9–3.10 still works with the
  flexible `requirements.txt`.
- Git


### Setup steps

```bash
git clone https://github.com/romybeaute/MOSAICapp.git
cd MOSAICapp

# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies and the package.
# Recommended for tutorials — exact, tested versions (needs Python 3.11–3.13):
pip install -r requirements.lock.txt
# Alternative — latest compatible versions (also works on Python 3.9–3.10):
#   pip install -r requirements.txt
pip install .

# Download NLTK data (required for segmentation)
python -c "import nltk; nltk.download('punkt')"
```

---

## 2. Configuration & Running


### Run the app
```
streamlit run app.py
```
to use the basic version (if only needs topic-modelling)
or use 
```
streamlit run app2.py
```
to use the new, extended version, with zero-shot and comparison between datasets (may be a bit slower)


### Input format

CSV file with a text column. The app auto-detects columns named `text`, `report`, `reflection_answer`, or `reflection_answer_english`. Any column can also be selected manually.


### LLM Topic Labelling — Hugging Face token setup (optional)

The **Automated Topic Labelling** feature turns each topic's keywords into a
readable label using a hosted Llama-3 model via the Hugging Face Inference API.
This is the **only** feature that needs a token — embedding, topic modelling,
zero-shot classification and condition comparison all work **without** one.

If you try to label topics without a token you'll see:

> `LLM labelling failed: No HF_TOKEN found in env/secrets.`

To enable it:

**Step 1 — Create a free Hugging Face token**
1. Sign in (or sign up) at <https://huggingface.co>.
2. Open **Settings → Access Tokens**: <https://huggingface.co/settings/tokens>.
3. Click **Create new token**, give it a name, select the **Read** role, and copy
   the value (it starts with `hf_...`).
   *If you create a fine-grained token instead, also tick
   "Make calls to Inference Providers" so the labelling API works.*

**Step 2 — Give the token to the app** (pick ONE option)

*Option A — secrets file (recommended). From the repo root:*
```bash
mkdir -p .streamlit
printf 'HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"\n' > .streamlit/secrets.toml
```
This creates `.streamlit/secrets.toml` containing a single line:
```toml
HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"
```
The file lives inside your repo folder but is already listed in `.gitignore`,
so it is **never committed**.

*Option B — environment variable (good for a one-off session):*
```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"   # add to ~/.zshrc or ~/.bashrc to persist
streamlit run app2.py
```

**Step 3 — Restart the app** (`streamlit run app2.py`). Topic labelling now works.

> ⚠️ **Never paste your token into code, notebooks, or any committed file.**
> If a token is ever exposed, revoke it on the
> [tokens page](https://huggingface.co/settings/tokens) and create a new one.


---

## 3. Pre-computing embeddings locally (optional)

Embedding is the slow part of the pipeline, and the hosted Space runs on shared CPUs.
If you have a GPU (locally or on an HPC cluster), you can compute the embeddings once
and reuse them — the app will then skip embedding entirely.

Use [`run_embeddings.py`](run_embeddings.py), which does **only** preprocessing +
embedding (no BERTopic fit, no LLM, no `HF_TOKEN` needed).

**Steps:**

1. Open `run_embeddings.py` and edit the config block at the top:
   ```python
   DATASET_NAME    = "MOSAIC"                       # must match the app's "Project/Dataset name"
   CSV_PATH        = "data/my_interviews.csv"       # path to your CSV
   EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"      # must match the model picked in the app
   TEXT_COL        = "sentence"                      # column holding the text (None = auto-detect)
   SPLIT_SENTENCES = True
   MIN_WORDS       = 3
   DEVICE          = "cuda"                          # "cuda" on a GPU, "mps" on a Mac, "cpu" otherwise
   ```

2. Run it:
   ```bash
   python run_embeddings.py
   ```
   This writes two files into `data/<DATASET_NAME>/preprocessed/cache/`:
   `precomputed_..._docs.json` and `precomputed_..._embeddings.npy`.

3. In the app, pick the **same** dataset name, embedding model, and segmentation
   settings. The app finds the cached files and skips embedding.
   (To use them on the hosted Space, upload those two files to the Space's matching
   `cache/` folder via the **Files** tab.)

### On an HPC cluster (SLURM, e.g. Artemis)

A ready-made batch script is provided in [`run_embeddings.sh`](run_embeddings.sh)
(requests 1 GPU). The only extra step is installing the **CUDA** build of PyTorch
(the pinned `requirements.txt` ships the CPU-only build):

```bash
ssh you@cluster
cd MOSAICapp
git pull
module load CUDA/12.1.1
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121 --force-reinstall

# edit run_embeddings.py (CSV_PATH, model, DEVICE="cuda"), then:
sbatch run_embeddings.sh
squeue -u $USER          # monitor; output goes to embed_<jobid>.log
```

---

# Python API (Advanced Usage)
MOSAICapp is also a Python library. You can import `mosaic_core` in your own scripts or Jupyter Notebooks for batch processing or custom analysis pipelines.

## Library usage
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


### Input format

CSV file with a text column. The app auto-detects columns named `text`, `report`, `reflection_answer`, or `reflection_answer_english`. Any column can also be selected manually.


## Running Tests
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
