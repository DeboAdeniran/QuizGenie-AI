import spacy
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from rake_nltk import Rake
import textstat
from collections import Counter
import re
from typing import List, Dict, Any, Optional
import PyPDF2
import docx
import io
from loguru import logger

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class TextProcessor:
    """
    Handles all text processing, NLP analysis, and content extraction
    """
    
    def __init__(self, spacy_model: str = "en_core_web_lg"):
        """
        Initialize the text processor with spaCy and NLTK
        """
        try:
            self.nlp = spacy.load(spacy_model)
            logger.info(f"Loaded spaCy model: {spacy_model}")
        except OSError:
            logger.warning(f"Model {spacy_model} not found. Downloading...")
            import os
            os.system(f"python -m spacy download {spacy_model}")
            self.nlp = spacy.load(spacy_model)
        
        self.stop_words = set(stopwords.words('english'))
        self.rake = Rake()
        logger.info("TextProcessor initialized successfully")
    
    def is_ready(self) -> bool:
        """Check if the processor is ready"""
        return self.nlp is not None
    
    def extract_text_from_file(self, content: bytes, filename: str) -> str:
        """
        Extract text from various file formats
        """
        file_extension = filename.lower().split('.')[-1]
        
        try:
            if file_extension == 'pdf':
                return self._extract_from_pdf(content)
            elif file_extension in ['doc', 'docx']:
                return self._extract_from_docx(content)
            elif file_extension == 'txt':
                return content.decode('utf-8')
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
        except Exception as e:
            logger.error(f"Error extracting text from {filename}: {str(e)}")
            raise
    
    def _extract_from_pdf(self, content: bytes) -> str:
        """Extract text from PDF"""
        pdf_file = io.BytesIO(content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        return text.strip()
    
    def _extract_from_docx(self, content: bytes) -> str:
        """Extract text from DOCX"""
        doc_file = io.BytesIO(content)
        doc = docx.Document(doc_file)
        
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        
        return text.strip()
    
    def analyze(self,
                text: str,
                analyze_sentiment: bool = True,
                extract_entities: bool = True,
                identify_topics: bool = True) -> Dict[str, Any]:
        """
        Comprehensive text analysis
        """
        # Clean text
        cleaned_text = self._clean_text(text)
        
        # Basic statistics
        word_count = len(word_tokenize(cleaned_text))
        char_count = len(cleaned_text)
        sentences = sent_tokenize(cleaned_text)
        sentence_count = len(sentences)
        
        # Readability analysis
        readability = self._analyze_readability(cleaned_text)
        
        # Process with spaCy
        doc = self.nlp(cleaned_text)
        
        # Extract entities
        entities = []
        if extract_entities:
            entities = self._extract_entities(doc)
        
        # Extract topics
        topics = []
        if identify_topics:
            topics = self._extract_topics(cleaned_text, doc)
        
        # Extract key phrases
        key_phrases = self._extract_key_phrases(cleaned_text)
        
        # Sentiment analysis (basic)
        sentiment = None
        if analyze_sentiment:
            sentiment = self._analyze_sentiment(doc)
        
        return {
            "word_count": word_count,
            "character_count": char_count,
            "sentence_count": sentence_count,
            "readability_score": readability['flesch_reading_ease'],
            "complexity": readability['complexity'],
            "language": doc.lang_,
            "topics": topics,
            "key_phrases": key_phrases,
            "entities": entities,
            "sentiment": sentiment
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\'\"]+', '', text)
        return text.strip()
    
    def _analyze_readability(self, text: str) -> Dict[str, Any]:
        """Analyze text readability"""
        flesch_score = textstat.flesch_reading_ease(text)
        
        # Determine complexity
        if flesch_score >= 70:
            complexity = "easy"
        elif flesch_score >= 50:
            complexity = "medium"
        else:
            complexity = "hard"
        
        return {
            "flesch_reading_ease": round(flesch_score, 2),
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
            "complexity": complexity
        }
    
    def _extract_entities(self, doc: spacy.tokens.Doc) -> List[Dict[str, Any]]:
        """Extract named entities"""
        entities = []
        seen = set()
        
        for ent in doc.ents:
            if ent.text not in seen and len(ent.text) > 2:
                entities.append({
                    "text": ent.text,
                    "type": ent.label_,
                    "confidence": 0.8  # spaCy doesn't provide confidence directly
                })
                seen.add(ent.text)
        
        return entities[:20]  # Limit to top 20
    
    def _extract_topics(self, text: str, doc: spacy.tokens.Doc, max_topics: int = 10) -> List[Dict[str, Any]]:
        """Extract main topics from text"""
        # Use noun chunks and named entities as topic candidates
        topic_candidates = []
        
        # Extract noun chunks
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 4:  # Limit phrase length
                topic_candidates.append(chunk.text.lower())
        
        # Extract named entities as topics
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE']:
                topic_candidates.append(ent.text.lower())
        
        # Count frequencies
        topic_freq = Counter(topic_candidates)
        
        # Get top topics
        topics = []
        for topic, count in topic_freq.most_common(max_topics):
            # Extract keywords from topic
            topic_doc = self.nlp(topic)
            keywords = [token.text for token in topic_doc if not token.is_stop and token.is_alpha]
            
            topics.append({
                "name": topic.title(),
                "confidence": min(count / 10, 1.0),  # Normalize confidence
                "keywords": keywords[:5]
            })
        
        return topics
    
    def _extract_key_phrases(self, text: str, max_phrases: int = 15) -> List[str]:
        """Extract key phrases using RAKE"""
        self.rake.extract_keywords_from_text(text)
        phrases = self.rake.get_ranked_phrases()
        
        # Filter and clean phrases
        filtered_phrases = []
        for phrase in phrases[:max_phrases]:
            if 2 <= len(phrase.split()) <= 5:  # Reasonable phrase length
                filtered_phrases.append(phrase.title())
        
        return filtered_phrases
    
    def _analyze_sentiment(self, doc: spacy.tokens.Doc) -> Dict[str, float]:
        """Basic sentiment analysis"""
        # Simple sentiment based on spaCy
        # For production, use a dedicated sentiment model
        positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'best', 'better'}
        negative_words = {'bad', 'poor', 'terrible', 'worst', 'awful', 'horrible', 'worse'}
        
        pos_count = 0
        neg_count = 0
        total_words = 0
        
        for token in doc:
            if token.is_alpha and not token.is_stop:
                total_words += 1
                if token.text.lower() in positive_words:
                    pos_count += 1
                elif token.text.lower() in negative_words:
                    neg_count += 1
        
        if total_words == 0:
            return {"positive": 0.5, "negative": 0.5, "neutral": 0.0}
        
        pos_score = pos_count / total_words if total_words > 0 else 0
        neg_score = neg_count / total_words if total_words > 0 else 0
        neutral_score = 1 - (pos_score + neg_score)
        
        return {
            "positive": round(pos_score, 3),
            "negative": round(neg_score, 3),
            "neutral": round(neutral_score, 3)
        }
    
    def extract_sentences_by_topic(self, text: str, topic: str, max_sentences: int = 5) -> List[str]:
        """Extract sentences related to a specific topic"""
        doc = self.nlp(text)
        topic_doc = self.nlp(topic.lower())
        
        sentences = []
        for sent in doc.sents:
            # Calculate similarity
            similarity = sent.similarity(topic_doc)
            if similarity > 0.5:  # Threshold for relevance
                sentences.append((sent.text.strip(), similarity))
        
        # Sort by similarity and return top sentences
        sentences.sort(key=lambda x: x[1], reverse=True)
        return [sent[0] for sent in sentences[:max_sentences]]
    
    def split_into_chunks(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Split text into manageable chunks"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks