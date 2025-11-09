# Use a recent Python image
FROM python:3.11-slim

# System deps (for things like hdbscan, numpy, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# HF will route to port 7860; tell Streamlit to use it
ENV PORT=7860

EXPOSE 7860

CMD ["streamlit", "run", "app.py",
     "--server.port=7860",
     "--server.address=0.0.0.0",
     "--server.enableCORS=false",
     "--server.enableXsrfProtection=false"]
