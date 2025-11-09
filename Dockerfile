FROM python:3.13.5-slim

WORKDIR /app

# Add 'unzip' to the list of programs to install
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# 1. Copy ONLY the requirements file first
COPY requirements.txt ./

# 2. Run pip install (this layer will now be cached)
# We add your --extra-index-url fix back in, just in case
RUN pip3 install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# 3. Create the standard NLTK data directory
RUN mkdir -p /usr/local/share/nltk_data

# 4. Download and unzip 'punkt'
RUN curl -L "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip" -o /tmp/punkt.zip && \
    unzip /tmp/punkt.zip -d /usr/local/share/nltk_data && \
    rm /tmp/punkt.zip

# 5. Download and unzip 'stopwords'
RUN curl -L "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/stopwords.zip" -o /tmp/stopwords.zip && \
    unzip /tmp/stopwords.zip -d /usr/local/share/nltk_data && \
    rm /tmp/stopwords.zip

# 6. NOW copy the rest of your app
COPY app.py ./

EXPOSE 7860

# HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1


# Run YOUR app.py file, with all flags
# ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableCORS=true", "--server.enableXsrfProtection=false"]

ENTRYPOINT ["bash", "-c", "streamlit run app.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false"]
