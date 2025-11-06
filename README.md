# Innerspeech Topic Dashboard (Upload Your CSV)

A Streamlit app for BERTopic-based topic modelling with sentence-transformers embeddings.  
**No data bundled** — users upload their own CSV with one text column (any of: `reflection_answer_english`, `reflection_answer`, `text`, `report`).

## Run locally
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
streamlit run app.py
