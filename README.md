# QuizGenie AI - Enhanced Question Generation Service

🎯 **AI-powered quiz question generation with intelligent, conceptual questions**

## 🌟 Features

- **Intelligent Question Generation**: Creates understanding-based questions, not just memorization
- **Multiple Question Types**: Multiple choice, True/False, Short answer
- **Difficulty Levels**: Easy, Medium, Hard, or Mixed
- **Context-Aware**: Extracts key concepts, definitions, processes, and relationships
- **Text Analysis**: Comprehensive NLP analysis with sentiment, entities, and topics
- **Document Processing**: Supports PDF, DOCX, and TXT files
- **RESTful API**: Fast, scalable FastAPI backend
- **Production Ready**: Optimized for deployment on Render

## 🚀 Quick Start

### Local Development

1. **Clone the repository**

```bash
git clone <your-repo-url>
cd quizgenie-ai
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

3. **Download NLTK data**

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

4. **Run the server**

```bash
python ai_server.py
```

5. **Test the API**
   Open http://localhost:8000/docs for interactive API documentation

### Docker Deployment

```bash
docker build -t quizgenie-ai .
docker run -p 8000:8000 quizgenie-ai
```

## 📡 API Endpoints

### Health Check

```bash
GET /health
```

### Generate Questions

```bash
POST /api/generate-questions
Content-Type: application/json

{
  "text": "Your educational content here...",
  "num_questions": 10,
  "difficulty": "mixed",
  "question_types": ["multipleChoice", "trueFalse"],
  "context_aware": true
}
```

### Analyze Text

```bash
POST /api/analyze-text
Content-Type: application/json

{
  "text": "Your text here...",
  "analyze_sentiment": true,
  "extract_entities": true,
  "identify_topics": true
}
```

### Process Documents

```bash
POST /api/process-documents
Content-Type: multipart/form-data

files: [file1.pdf, file2.docx]
```

## 🎓 Question Types Generated

1. **Conceptual Definition Questions**: Tests understanding of key terms
2. **Process/Stage Questions**: Evaluates knowledge of procedures and stages
3. **Role/Responsibility Questions**: Assesses understanding of functions
4. **Historical/Timeline Questions**: Tests temporal knowledge
5. **Relationship Questions**: Evaluates understanding of connections
6. **Comparison Questions**: Tests ability to differentiate concepts
7. **Cause-Effect Questions**: Assesses logical reasoning
8. **Inference Questions**: Tests deeper understanding
9. **Application Questions**: Evaluates practical application
10. **Sequence Questions**: Tests understanding of order

## 🛠️ Tech Stack

- **FastAPI**: Modern, fast web framework
- **spaCy**: Advanced NLP processing
- **Sentence Transformers**: Semantic similarity
- **NLTK**: Natural language toolkit
- **PyTorch**: Machine learning backend
- **Uvicorn**: ASGI server

## 📦 Project Structure

```
quizgenie-ai/
├── ai_server.py              # Main FastAPI application
├── question_generator.py     # Intelligent question generation logic
├── text_processor.py         # NLP and text analysis
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker configuration
├── render.yaml              # Render deployment config
├── DEPLOYMENT_GUIDE.md      # Detailed deployment instructions
└── pre_deploy_check.py      # Pre-deployment validation script
```

## 🔧 Configuration

### Environment Variables

| Variable            | Default | Description              |
| ------------------- | ------- | ------------------------ |
| `PORT`              | 8000    | Server port              |
| `PYTHON_UNBUFFERED` | 1       | Enable real-time logging |

### Model Configuration

The service uses `en_core_web_sm` spaCy model by default for optimal deployment performance. For local development with higher accuracy, you can change to `en_core_web_lg` in the code.

## 🚀 Deployment to Render

See **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** for detailed instructions.

Quick steps:

1. Push code to GitHub
2. Create new Web Service on Render
3. Connect your repository
4. Render auto-deploys using Dockerfile
5. Access your API at `https://your-app.onrender.com`

## 🧪 Testing

Run pre-deployment checks:

```bash
python pre_deploy_check.py
```

Test the API:

```python
import requests

response = requests.post(
    "http://localhost:8000/api/generate-questions",
    json={
        "text": "Climate change is affecting global temperatures...",
        "num_questions": 5,
        "difficulty": "medium"
    }
)
print(response.json())
```

## 📊 Performance

- **Question Generation**: ~2-5 seconds for 10 questions
- **Text Analysis**: ~1-2 seconds for typical documents
- **Memory Usage**: ~400MB RAM (with en_core_web_sm)
- **Concurrent Requests**: Handled by Uvicorn workers

## 🔒 Security

- CORS configured (update for production)
- Input validation with Pydantic models
- Rate limiting recommended for production
- No API key required (add authentication for production)

## 🐛 Troubleshooting

### Common Issues

1. **Module not found errors**
   - Ensure all dependencies in requirements.txt
   - Check installation order (nltk before rake-nltk)

2. **spaCy model not found**
   - Run: `python -m spacy download en_core_web_sm`

3. **Out of memory on deployment**
   - Using en_core_web_sm (13MB) instead of en_core_web_lg (788MB)
   - Upgrade to Render Starter plan if needed

4. **Build timeout**
   - Dockerfile is optimized with caching
   - First build takes 10-15 minutes
   - Subsequent builds are faster

## 📈 Roadmap

- [ ] Add caching for frequently generated questions
- [ ] Support for more document formats
- [ ] Question quality scoring improvements
- [ ] Multi-language support
- [ ] Advanced question types (matching, ordering)
- [ ] Question bank management
- [ ] User authentication and rate limiting

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- NLP powered by [spaCy](https://spacy.io/)
- Sentence embeddings from [Sentence Transformers](https://www.sbert.net/)
- Deployed on [Render](https://render.com/)

## 📞 Support

For issues, questions, or contributions:

- Open an issue on GitHub
- Check DEPLOYMENT_GUIDE.md for deployment help
- Run pre_deploy_check.py before deploying

---

**Made with ❤️ for educators and learners**
