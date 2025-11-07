---
title: MOSAICapp
colorFrom: indigo
colorTo: blue
sdk: streamlit
app_file: app.py
pinned: false
---


# MOSAIC Topic Dashboard

A Streamlit app for BERTopic-based topic modelling with sentence-transformers embeddings.  
**No data bundled** — upload CSV with one text column (any of: `reflection_answer_english`, `reflection_answer`, `text`, `report`).

## Run locally
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
streamlit run app.py
