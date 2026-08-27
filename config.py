"""Configuration settings for Simple RAG Chatbot.

Loads environment variables from .env and provides clean, centralized
defaults for all RAG hyperparameters, file paths, and model settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

# Base Project Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
VECTOR_DB_DIR = BASE_DIR / "vector_db"
LOG_FILE = BASE_DIR / "rag_chatbot.log"

# Ensure essential directories exist
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

# OpenAI API Settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
LLM_MODEL = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini").strip()
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip()

# RAG Pipeline Hyperparameters
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
TOP_K = int(os.getenv("TOP_K", 4))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.25))

# Conversation Settings
MAX_HISTORY = int(os.getenv("MAX_HISTORY", 10))

# Debug and Logging
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").strip().lower() in ("true", "1", "yes")

# Supported Document Extensions
SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".md", ".docx"}
