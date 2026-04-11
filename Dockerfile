# Dockerfile
# Targets Hugging Face Docker Spaces — port 7860 is required by HF.
# Health-check endpoint runs on :8080 (see app.py) for UptimeRobot / HF pings.

FROM python:3.11-slim

WORKDIR /app

# System deps for numpy/torch/hdbscan build wheels + newspaper3k
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev \
    libxslt-dev \
    libffi-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps before copying source (Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data needed by newspaper3k
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Copy source
COPY . .

# Create cache directory (pipeline.py writes here)
RUN mkdir -p cache

# HF Spaces requires port 7860
EXPOSE 7860

CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]
