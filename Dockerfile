# Dockerfile - Minimal working version
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages individually in correct order
RUN pip install --no-cache-dir nltk==3.8.1
RUN pip install --no-cache-dir fastapi==0.104.1
RUN pip install --no-cache-dir uvicorn[standard]==0.24.0
RUN pip install --no-cache-dir pydantic==2.5.0
RUN pip install --no-cache-dir python-multipart==0.0.6
RUN pip install --no-cache-dir spacy==3.7.2
RUN pip install --no-cache-dir textstat==0.7.3
RUN pip install --no-cache-dir python-docx==1.1.0
RUN pip install --no-cache-dir PyPDF2==3.0.1
RUN pip install --no-cache-dir torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir numpy==1.24.3
RUN pip install --no-cache-dir scikit-learn==1.3.2
RUN pip install --no-cache-dir loguru==0.7.2
RUN pip install --no-cache-dir aiofiles==23.2.1
RUN pip install --no-cache-dir python-dotenv==1.0.0

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Download spaCy model
RUN python -c "import spacy; spacy.cli.download('en_core_web_sm')"

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["python", "ai_server.py"]