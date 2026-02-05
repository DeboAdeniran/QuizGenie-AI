#!/bin/bash
# start.sh

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Download spaCy model
python -c "import spacy; spacy.cli.download('en_core_web_lg')"

# Start the server
python ai_server.py