# Dockerfile - Optimized for Render deployment
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python packages
# Install core dependencies first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install all packages from requirements.txt (proper order is defined there)
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"

# Download spaCy model using pip (more reliable than spacy download)
RUN pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

# Copy application code
COPY . .

# Create logs directory
RUN mkdir -p logs

# Expose port (Render will set PORT env variable)
EXPOSE 8000

# Use exec form of CMD for proper signal handling
CMD ["python", "-u", "ai_server.py"]