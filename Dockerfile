# Dockerfile - Fixed with proper installation order
FROM python:3.11-slim

WORKDIR /app

# 1. Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy requirements first for better caching
COPY requirements.txt .

# 3. Install Python dependencies in two steps
# First install nltk alone
RUN pip install --no-cache-dir nltk==3.8.1

# Then install the rest
RUN pip install --no-cache-dir -r requirements.txt

# 4. Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# 5. Download spaCy model (use smaller model for memory efficiency)
RUN python -c "import spacy; spacy.cli.download('en_core_web_sm')"

# 6. Copy application code
COPY . .

# 7. Expose port
EXPOSE 8000

# 8. Command to run the application
CMD ["python", "ai_server.py"]