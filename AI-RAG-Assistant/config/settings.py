import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
VECTOR_DB_DIR = BASE_DIR / "vector_db"
MODELS_DIR = BASE_DIR / "models"

# Embedding
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Local LLM (GGUF)
LLM_MODEL_PATH = os.getenv(
    "LLM_MODEL_PATH",
    str(MODELS_DIR / "llama-3-8b-instruct.Q4_K_M.gguf"),
)
LLM_N_CTX = int(os.getenv("LLM_N_CTX", "4096"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# RAG
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("TOP_K", "4"))

# Optional: OpenAI / other API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
USE_API_LLM = os.getenv("USE_API_LLM", "false").lower() == "true"
