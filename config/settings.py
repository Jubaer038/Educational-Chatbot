import os
from dotenv import load_dotenv

load_dotenv()  # Load .env before reading environment variables

# LLM configuration (Groq only)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Embedding configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# RAG parameters
CHUNK_SIZE = 800  # characters
CHUNK_OVERLAP = 150  # characters
TOP_K = 5

# Caching
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache", "pdf_indexes")

# Planner defaults
REVISION_EVERY_N_DAYS = 3
MOCK_TEST_EVERY_N_DAYS = 7
