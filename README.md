---
title: MOSAICapp
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
---

# MOSAIC Topic Dashboard

A Streamlit app for BERTopic-based topic modelling with sentence-transformers embeddings.
**No data bundled** — upload CSV with one text column (any of: `reflection_answer_english`, `reflection_answer`, `text`, `report`).

## Lite Version (Free Hardware)

This Hugging Face Space runs the **`lite` version** of the app.

To make it run on free "CPU basic" hardware, the **LLM-based topic labeling feature has been disabled**. The app will use BERTopic's default keyword-based labels instead.

For the full, original version with LLM features (which requires paid GPU hardware), please see the `main` branch of the [original GitHub repository](https://github.com/romybeaute/MOSAICapp).

## Run Locally (Full Version)

To run the full version on your local machine:

```bash
# Clone the main branch
git clone [https://github.com/romybeaute/MOSAICapp.git](https://github.com/romybeaute/MOSAICapp.git)
cd MOSAICapp

# Install requirements
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"

# Run the app
streamlit run app.py

# ---------------------------------------


### Library Usage (Advanced)
For researchers wishing to run MOSAIC programmatically (e.g., on a computer cluster), 
you can import the core logic directly:

```python
from mosaic_core.analysis import preprocess_and_embed, run_topic_model

# 1. Load and Embed
docs, embeddings = preprocess_and_embed("my_data.csv", text_col="report")

# 2. Configure
config = {
    "umap_params": {"n_neighbors": 15, "n_components": 5},
    "hdbscan_params": {"min_cluster_size": 10},
    "bt_params": {"nr_topics": "auto"}
}

# 3. Run Analysis
model, reduced_data, topics = run_topic_model(docs, embeddings, config)