# 📚 Educational Chatbot & Learning Assistant

A comprehensive Streamlit-based educational platform featuring an AI-powered chatbot for PDF-based Q&A with citations and rubric-based answer evaluation. Also includes study planning, essay generation, and summarization tools powered by RAG (Retrieval-Augmented Generation).

---

## 📋 Table of Contents
- [Features](#-features)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Code Documentation](#-code-documentation)
- [Troubleshooting](#-troubleshooting)
- [Deployment](#-deployment)

---

## ✨ Features

### 1. **PDF Q&A Chatbot (RAG-based)**
- Upload PDF documents and ask questions with context-aware answers
- Retrieves relevant passages using FAISS vector similarity search
- Provides page citations with relevance scores
- Caches PDF embeddings for faster subsequent queries
- Semantic search using sentence-transformers

### 2. **Automated Evaluation System**
- Grades student responses against PDF reference material
- Uses rubric-based scoring across 4 dimensions:
  - **Concept Correctness** (40%): Semantic similarity with reference answer
  - **Coverage** (30%): Completeness of key points
  - **Clarity** (20%): Sentence structure and readability
  - **Language** (10%): Grammar, punctuation, capitalization
- Provides detailed feedback with strengths and improvement areas

### 3. **Smart Study Planner**
- Creates personalized study schedules based on exam date
- Topic prioritization by difficulty and importance
- Automatic scheduling of:
  - Daily study sessions with time allocation
  - Periodic revision cycles (every 3 days by default)
  - Mock tests (every 7 days by default)
- Adapts to available daily study hours

### 4. **AI Essay Writer**
- Generates original academic essays on any topic
- Customizable parameters:
  - Word limit (150-2000 words)
  - Tone (Academic, Friendly, Persuasive, Narrative)
  - Optional outline/structure
- Structured output: Introduction → Body sections → Conclusion
- Plagiarism-safe with original content generation

### 5. **Intelligent Summarizer**
- Summarizes text content or entire PDF documents
- Two modes:
  - **Short**: Concise summary under 150 words
  - **Bullets**: 5-8 key bullet points
- Smart text limiting for large documents (first 4000 chars)

---

## 🛠 Technology Stack

### Core Framework
- **Streamlit**: Web UI framework
- **Python 3.10+**: Base language

### LLM Provider
- **Groq**: Cloud-based inference with free tier
  - Default model: `llama-3.1-8b-instant` (8B parameters)
  - Fast inference with low latency
  - API key from [console.groq.com](https://console.groq.com)
  - Other models available: `llama-3.3-70b-versatile`, `mixtral-8x7b-32768`

### RAG & Embeddings
- **sentence-transformers**: Text embeddings
  - Model: `all-MiniLM-L6-v2` (lightweight, fast)
  - Dimensionality: 384
- **FAISS**: Vector database for similarity search
  - IndexFlatIP for cosine similarity
  - CPU-based implementation (faiss-cpu)

### PDF Processing
- **PyPDF2**: Text extraction from PDF documents
- Custom chunking with overlap for context preservation

### Other Libraries
- **python-dotenv**: Environment variable management
- **groq**: Groq API client library

---

## 📁 Project Structure

```
AI-Powered Learning Assistant System/
│
├── app.py                          # Main Streamlit application entry point
├── requirements.txt                # Python dependencies
├── .env                            # Environment configuration (not in git)
├── README.md                       # This file
│
├── config/                         # Configuration modules
│   ├── settings.py                 # LLM, RAG, and app settings
│   └── prompts.py                  # System prompts and templates
│
├── rag/                            # RAG pipeline components
│   ├── pdf_loader.py               # PDF text extraction by page
│   ├── chunker.py                  # Text chunking with overlap
│   ├── embeddings.py               # Sentence-transformer embeddings
│   ├── vector_store.py             # FAISS index operations
│   └── retriever.py                # Retrieval orchestration
│
├── services/                       # Business logic services
│   ├── llm_client.py               # Unified LLM client (Ollama/Groq)
│   ├── pdf_chatbot.py              # PDF Q&A with caching
│   ├── evaluator.py                # Response evaluation engine
│   ├── planner.py                  # Study schedule generation
│   ├── essay_writer.py             # Essay generation
│   └── summarizer.py               # Text/PDF summarization
│
├── utils/                          # Utility functions
│   ├── validators.py               # Input validation
│   └── error_handler.py            # Error handling
│
└── cache/                          # Runtime cache
    └── pdf_indexes/                # Cached PDF embeddings
```

---

## 🚀 Installation

### Prerequisites
- **Python 3.10 or higher**
- **Git** (for cloning)
- **Groq API Key** (free) - Get from [console.groq.com](https://console.groq.com)

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd "AI-Powered Learning Assistant System"
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Get Groq API Key
1. Visit [console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Navigate to API Keys section
4. Create new key and copy it

### Step 5: Configure Environment
Create `.env` file in project root:
```env
GROQ_API_KEY=your_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

### Step 6: Download Embedding Model
The `all-MiniLM-L6-v2` model (~80MB) downloads automatically on first run.

---

## ⚙️ Configuration

### Using `.env` File (Recommended)

Create `.env` in project root:

```env
# Groq API Configuration (Required)
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant

# Optional: Embedding Model Override
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### Alternative: Environment Variables

**Windows PowerShell:**
```powershell
$env:GROQ_API_KEY='your_key_here'
$env:GROQ_MODEL='llama-3.1-8b-instant'
```

**Linux/Mac:**
```bash
export GROQ_API_KEY=your_key_here
export GROQ_MODEL=llama-3.1-8b-instant
```

### Available Groq Models

| Model | Parameters | Context Window | Best For |
|-------|-----------|----------------|----------|
| `llama-3.1-8b-instant` | 8B | 128K | Fast responses, general tasks |
| `llama-3.3-70b-versatile` | 70B | 128K | Complex reasoning, accuracy |
| `mixtral-8x7b-32768` | 47B | 32K | Multi-lingual, coding |
| `gemma2-9b-it` | 9B | 8K | Compact, efficient |

Change model in `.env` by updating `GROQ_MODEL` value.

---

## 🎯 Usage

### Start the Application
```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Run Streamlit app
streamlit run app.py
```

Access at: `http://localhost:8501`

### Feature Guide

#### 📚 PDF Q&A Tab
1. Upload PDF document
2. Type question
3. Click "Get Answer"
4. View answer with citations

#### ✅ Evaluation Tab
1. Upload reference PDF
2. Enter question
3. Paste student answer
4. Click "Evaluate"
5. Review score and feedback

#### 📅 Study Planner Tab
1. Select exam date
2. Set daily study hours
3. Enter topics: `Topic | difficulty(1-5) | priority(1-5)`
4. Click "Create Plan"

#### ✍️ Essay Writer Tab
1. Enter topic
2. Set word limit and tone
3. Add optional outline
4. Click "Generate Essay"

#### 📝 Summarizer Tab
1. Choose mode (Short/Bullets)
2. Select source (Text/PDF)
3. Provide content
4. Click summarize

---

## 📖 Code Documentation

### Key Components

#### `config/settings.py` - Configuration
```python
GROQ_MODEL        # Groq model name (default: llama-3.1-8b-instant)
CHUNK_SIZE        # 800 characters per chunk
CHUNK_OVERLAP     # 150 characters overlap
TOP_K             # 5 similar chunks to retrieve
```

#### `services/llm_client.py` - LLM Interface
```python
class LLMClient:
    def generate(system_prompt, user_prompt, max_tokens) -> str
        # Groq API client with error handling
        # Validates API key on initialization
        # Returns generated text
```

#### `rag/pdf_loader.py` - PDF Processing
```python
def extract_text_by_page(file_bytes) -> List[Dict]
    # Returns: [{"page": 1, "text": "..."}, ...]
```

#### `services/pdf_chatbot.py` - Q&A Service
```python
def answer_question(question, file_bytes, file_name) -> Dict
    # Returns: {"answer": str, "citations": List}
    # Uses @st.cache_resource for caching
```

#### `services/evaluator.py` - Grading
```python
def grade_response(question, student_answer, reference) -> Dict
    # Returns score, details, strengths, improvements
    # Uses semantic similarity + heuristics
```

---

## 🔧 Troubleshooting

### "GROQ_API_KEY not set"
- Add to `.env`: `GROQ_API_KEY=your_key`
- Verify: `echo $env:GROQ_API_KEY` (PowerShell)

### "Ollama request failed"
1. Check Ollama running: `ollama list`
2. Verify at `http://localhost:11434/api/tags`
3. Pull model: `ollama pull llama3`
4. **Or switch to Groq** in `.env`

### "Model decommissioned"
- Update `.env`: `GROQ_MODEL=llama-3.1-8b-instant`

### "No readable pages in PDF"
- PDF is image-based (scanned)
- Use PDF with text layer or pre-process with OCR

### Module not found
```bash
pip install --upgrade -r requirements.txt
```

### Slow first PDF query
- Normal: building embeddings first time
- Subsequent queries use cached index (fast)

---

## 🌐 Deployment

### Streamlit Community Cloud (Free)

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git push -u origin main
   ```

2. **Deploy on [share.streamlit.io](https://share.streamlit.io):**
   - Connect GitHub repo
   - Select `app.py` as main file
   - Add secrets in dashboard:
     ```toml
     GROQ_API_KEY = "your_key_here"
     GROQ_MODEL = "llama-3.1-8b-instant"
     ```
   - Click Deploy

3. **Requirements:**
   - Use `faiss-cpu` in `requirements.txt`
   - Keep dependencies minimal
   - Groq API key required

### Docker Deployment

**Dockerfile:**
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

**Build and run:**
```bash
docker build -t learning-assistant .
docker run -p 8501:8501 \
  -e GROQ_API_KEY=your_key \
  -e GROQ_MODEL=llama-3.1-8b-instant \
  learning-assistant
```

### Environment Variables for Production

```bash
# Required
GROQ_API_KEY=your_production_key

# Optional
GROQ_MODEL=llama-3.3-70b-versatile  # Use larger model
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

---

## 🎓 Educational Use

Designed for:
- Students preparing for exams
- Educators creating materials
- Self-learners organizing study
- Document Q&A with citations

**Ethical Use:** Always verify AI-generated content and cite sources in academic work.

---

## 📄 License

Open-source. Free to use, modify, and distribute.
