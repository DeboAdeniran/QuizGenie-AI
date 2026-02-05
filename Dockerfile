# Dockerfile - With rake-nltk installed
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install nltk FIRST
RUN pip install --no-cache-dir nltk==3.8.1

# Install rake-nltk AFTER nltk
RUN pip install --no-cache-dir rake-nltk==1.0.4

# Install the rest
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Download spaCy model
RUN python -c "import spacy; spacy.cli.download('en_core_web_sm')"

COPY . .

EXPOSE 8000
CMD ["python", "ai_server.py"]