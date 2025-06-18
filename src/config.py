from pathlib import Path
from dotenv import load_dotenv
import os

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
EMB_DIR  = BASE_DIR / "embeddings"

load_dotenv()

# LLM Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# LLM Provider Selection (Google only)
LLM_PROVIDER = "google"

# Model Configuration (configurable via .env)
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash-exp")

# Google Gemini Configuration  
GOOGLE_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{GOOGLE_MODEL}:generateContent"

# --- Embedding Model Configuration ---
LOCAL_MODEL_DIRECTORY = "E:\\Projects\\CodeFest_Summer_2025\\RAGBOT\\RAGChabot\\embeddings\\single_dense"
EMBEDDING_DIMENSION = 1024  # Confirmed dimension

# --- Remote Embedding API Configuration ---
REMOTE_EMBEDDING_URL = os.getenv("REMOTE_EMBEDDING_URL", "https://api.confusedelectrons.xyz/embed-query-w-sentence-transformers/")
REMOTE_EMBEDDING_API_KEY = os.getenv("REMOTE_EMBEDDING_API_KEY")  # Optional, can be None

# --- Port Configuration ---
BACKEND_PORT = int(os.getenv("BACKEND_PORT", 5000))
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", 3001))
