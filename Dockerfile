# ---- Base image ----
# We switch to Python 3.10 because it has the most reliable pre-built wheels
FROM python:3.10-slim

# Workdir inside the container
WORKDIR /app

# ---- System dependencies ----
# Install minimal tools. We clean up the list to keep it small.
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---- Upgrade Pip ----
# Crucial: Upgrading pip helps it distinguish between the 'musl' and 'linux' versions
RUN pip install --upgrade pip

# ---- STEP 1: Install LLM Engine (CPU Version) ----
# We install this specific version for Python 3.10 on Linux CPU.
# This avoids the "musl" error and the build timeout.
RUN pip install llama-cpp-python \
    --no-cache-dir \
    --prefer-binary \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# ---- STEP 2: Install Remaining Packages ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- NLTK data ----
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"

# ---- Copy application files ----
COPY . .

# ---- Command to run the app ----
# TO SWITCH VERSIONS: Change "app.py" to "app_with_LLM.py" in the line below.
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]

# # ---- Base image ----
# FROM python:3.11-slim

# # Workdir inside the container
# WORKDIR /app

# # ---- System dependencies ----
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     curl \
#     git \
#     unzip \
#  && rm -rf /var/lib/apt/lists/*


# # ---- Python deps ----
# COPY requirements.txt ./

# # Torch / sentence-transformers like having the CPU wheel index explicitly
# RUN pip install --no-cache-dir \
#     --extra-index-url https://download.pytorch.org/whl/cpu \
#     -r requirements.txt

# # ---- NLTK data (punkt + stopwords) ----
# RUN mkdir -p /usr/local/share/nltk_data

# # punkt tokenizer
# # RUN curl -L "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip" \
# #       -o /tmp/punkt.zip && \
# #     unzip /tmp/punkt.zip -d /usr/local/share/nltk_data && \
# #     rm /tmp/punkt.zip
# # punkt_tab tokenizer (for NLTK >= 3.9)
# RUN curl -L "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip" \
#       -o /tmp/punkt_tab.zip && \
#     unzip /tmp/punkt_tab.zip -d /usr/local/share/nltk_data && \
#     rm /tmp/punkt_tab.zip


# # stopwords corpus
# RUN curl -L "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/stopwords.zip" \
#       -o /tmp/stopwords.zip && \
#     unzip /tmp/stopwords.zip -d /usr/local/share/nltk_data && \
#     rm /tmp/stopwords.zip

# # ---- Copy app code ----
# # If you only want app.py + data, you can narrow this, but copying all is fine.
# COPY . .

# # ---- Hugging Face port wiring ----
# ENV PORT=7860
# EXPOSE 7860

# # Optional healthcheck; HF will just ignore failures but nice to have
# HEALTHCHECK CMD curl --fail http://localhost:${PORT}/_stcore/health || exit 1

# # ---- Run Streamlit ----
# ENTRYPOINT ["bash", "-c", "streamlit run app_with_LLM.py \
#     --server.port=${PORT} \
#     --server.address=0.0.0.0 \
#     --server.enableCORS=false \
#     --server.enableXsrfProtection=false"]

