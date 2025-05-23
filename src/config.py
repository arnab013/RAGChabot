from pathlib import Path
from dotenv import load_dotenv
import os

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
EMB_DIR  = BASE_DIR / "embeddings"

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_ENDPOINT = "https://api.mistral.ai/v1/chat/completions"  # :contentReference[oaicite:0]{index=0}

# Mixtral model names (update if you have access to a newer suffix)
MIXTRAL_MODEL = "open-mixtral-8x22b"

# embedding model
EMB_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
