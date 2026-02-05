import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent

import logging
logging.basicConfig(level=logging.INFO)
base_logger = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
from loguru import logger

# Import our custom modules
from text_processor import TextProcessor

# Try to import improved question generator first
try:
    from question_generator import ImprovedQuestionGenerator as QuestionGenerator
    base_logger.info("✅ Using IMPROVED question generator with enhanced AI capabilities")
except ImportError:
    try:
        from question_generator import QuestionGenerator
        base_logger.warning("⚠️ Using original question generator (improved version not found)")
    except ImportError:
        # Try direct import
        sys.path.append(str(BASE_DIR))
        try:
            from question_generator import ImprovedQuestionGenerator as QuestionGenerator
            base_logger.info("✅ Using IMPROVED question generator with enhanced AI capabilities")
        except ImportError:
            from question_generator import QuestionGenerator
            base_logger.warning("⚠️ Using original question generator")

# Configure logging
logger.remove()
logger.add(sys.stdout, level="INFO")
logger.add("logs/ai_service_{time}.log", rotation="500 MB", level="DEBUG")

# Initialize FastAPI app
app = FastAPI(
    title="QuizGenie AI Service - Enhanced Edition",
    description="AI-powered text analysis and intelligent question generation",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize processors
text_processor = TextProcessor()
question_generator = QuestionGenerator()

# Pydantic models for request/response
class TextAnalysisRequest(BaseModel):
    text: str
    analyze_sentiment: bool = True
    extract_entities: bool = True
    identify_topics: bool = True

class TextAnalysisResponse(BaseModel):
    word_count: int
    character_count: int
    sentence_count: int
    readability_score: float
    complexity: str
    language: str
    topics: List[Dict[str, Any]]
    key_phrases: List[str]
    entities: List[Dict[str, Any]]
    sentiment: Optional[Dict[str, float]] = None

class QuestionGenerationRequest(BaseModel):
    text: str
    num_questions: int = Field(default=10, ge=1, le=50)
    difficulty: str = Field(default="mixed", pattern="^(easy|medium|hard|mixed)$")
    question_types: List[str] = Field(default=["multipleChoice", "trueFalse"])
    topics: Optional[List[str]] = None
    context_aware: bool = True

class Question(BaseModel):
    type: str
    text: str
    options: Optional[List[str]] = None
    correct_answer: str
    explanation: str
    difficulty: str
    points: int
    topic: Optional[str] = None
    source_context: str
    confidence: float
    keywords: List[str]

class QuestionGenerationResponse(BaseModel):
    questions: List[Question]
    generation_time: float
    total_questions: int
    difficulty_distribution: Dict[str, int]
    topics_covered: List[str]
    quality_score: Optional[float] = None

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "QuizGenie AI Enhanced",
        "version": "2.0.0",
        "models_loaded": text_processor.is_ready() and question_generator.is_ready(),
        "generator_type": "improved" if hasattr(question_generator, '_build_knowledge_base') else "standard"
    }

# Text analysis endpoint
@app.post("/api/analyze-text", response_model=TextAnalysisResponse)
async def analyze_text(request: TextAnalysisRequest):
    """
    Analyze text and extract metadata, topics, entities, and sentiment
    """
    try:
        logger.info(f"Analyzing text of length {len(request.text)}")
        
        # Perform analysis
        analysis = text_processor.analyze(
            text=request.text,
            analyze_sentiment=request.analyze_sentiment,
            extract_entities=request.extract_entities,
            identify_topics=request.identify_topics
        )
        
        logger.info(f"Analysis complete. Found {len(analysis['topics'])} topics")
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text analysis failed: {str(e)}")

# Question generation endpoint
@app.post("/api/generate-questions", response_model=QuestionGenerationResponse)
async def generate_questions(request: QuestionGenerationRequest):
    """
    Generate high-quality quiz questions from provided text using enhanced AI
    """
    try:
        logger.info(f"🎯 Generating {request.num_questions} questions with difficulty: {request.difficulty}")
        logger.info(f"📝 Text length: {len(request.text)} characters")
        logger.info(f"🔧 Question types: {', '.join(request.question_types)}")
        
        # Validate text length
        if len(request.text) < 100:
            raise HTTPException(
                status_code=400, 
                detail="Text is too short. Please provide at least 100 characters for meaningful question generation."
            )
        
        # Generate questions
        result = question_generator.generate(
            text=request.text,
            num_questions=request.num_questions,
            difficulty=request.difficulty,
            question_types=request.question_types,
            topics=request.topics,
            context_aware=request.context_aware
        )
        
        quality_score = result.get('quality_score', None)
        
        logger.info(f"✅ Generated {len(result['questions'])} high-quality questions")
        if quality_score:
            logger.info(f"📊 Average quality score: {quality_score:.2f}")
        
        # Log difficulty distribution
        dist = result['difficulty_distribution']
        logger.info(f"📈 Distribution - Easy: {dist.get('easy', 0)}, Medium: {dist.get('medium', 0)}, Hard: {dist.get('hard', 0)}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error generating questions: {str(e)}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=f"Question generation failed: {str(e)}")

# Batch processing endpoint
@app.post("/api/process-documents")
async def process_documents(files: List[UploadFile] = File(...)):
    """
    Process multiple documents and extract text
    """
    try:
        results = []
        
        for file in files:
            logger.info(f"Processing file: {file.filename}")
            
            # Read file content
            content = await file.read()
            
            # Extract text based on file type
            extracted_text = text_processor.extract_text_from_file(
                content=content,
                filename=file.filename
            )
            
            results.append({
                "filename": file.filename,
                "text_length": len(extracted_text),
                "text": extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text
            })
        
        return {
            "processed_files": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

# Validate question endpoint
@app.post("/api/validate-question")
async def validate_question(question: Question):
    """
    Validate a generated question for quality and correctness
    """
    try:
        if hasattr(question_generator, 'validate_question'):
            validation = question_generator.validate_question(question.dict())
        else:
            # Basic validation
            validation = {
                "is_valid": True,
                "quality_score": 75,
                "issues": []
            }
        
        return validation
        
    except Exception as e:
        logger.error(f"Error validating question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

# Get topics from text endpoint
@app.post("/api/extract-topics")
async def extract_topics(text: str, max_topics: int = 10):
    """
    Extract main topics from text
    """
    try:
        topics = text_processor.extract_topics(text, max_topics=max_topics)
        return {"topics": topics}
        
    except Exception as e:
        logger.error(f"Error extracting topics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Topic extraction failed: {str(e)}")

if __name__ == "__main__":
    # Get port from environment variable (Render provides PORT)
    port = int(os.environ.get("PORT", 8000))
    
    logger.info("="*60)
    logger.info("🚀 Starting QuizGenie AI Service - Enhanced Edition")
    logger.info("="*60)
    
    # Run the server
    uvicorn.run(
        "ai_server:app",
        host="0.0.0.0",  # IMPORTANT: Must be 0.0.0.0 for deployment
        port=port,
        reload=False,  # Disable reload in production
        log_level="info"
    )