"""
config.py
─────────
Central configuration for the OC-RAG backend.
All environment variables are loaded here and imported by service files.

Create a .env file in the project root with the following:
    GROQ_API_KEY=your_groq_api_key_here
    LLM_MODEL=llama3-70b-8192
    ENVIRONMENT=development

Run the app:
    uvicorn main:app --reload
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# ── LLM ───────────────────────────────────────────────────────────────────────
GROQ_API_KEY : str  = os.getenv("GROQ_API_KEY", "")
LLM_MODEL    : str  = os.getenv("LLM_MODEL", "llama3-70b-8192")

if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY is not set. "
        "Add it to your .env file: GROQ_API_KEY=your_key_here"
    )

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR         = BASE_DIR / "data"
PROGRAMS_DIR     = DATA_DIR / "programs"
STRANDS_DIR      = DATA_DIR / "strands"
CHROMA_DIR       = DATA_DIR / "chroma_db"
DB_PATH          = DATA_DIR / "oc_rag_results.db"
ONTOLOGY_PATH    = BASE_DIR / "ontology" / "lspu_lb_ontology.owl"
THEORY_MAP_PATH  = DATA_DIR / "question_theory_map.json"
QUESTIONS_PATH   = DATA_DIR / "assessment_questions.json"
TEST_PROFILES_PATH = DATA_DIR / "test_profiles.json"

# ── ChromaDB ───────────────────────────────────────────────────────────────────
COLLECTION_NAME : str = os.getenv("COLLECTION_NAME", "lspu_lb_programs")
EMBED_MODEL     : str = "all-MiniLM-L6-v2"
TOP_K_RETRIEVAL : int = 10   # chunks retrieved in Stage 3
TOP_K_RERANKED  : int = 5    # chunks passed to Stage 5 after Stage 4

# ── API ────────────────────────────────────────────────────────────────────────
ENVIRONMENT : str = os.getenv("ENVIRONMENT", "development")
API_TITLE   : str = "OC-RAG College Recommendation API"
API_VERSION : str = "1.0.0"
API_DESC    : str = (
    "Ontology-Contextualized RAG pipeline for LSPU-LB college program recommendation. "
    "Implements 5-stage pipeline: Career Theory Extraction → Ontology Candidate Selection "
    "→ Filtered Retrieval → LLM Reranking → Grounded Generation."
)

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL : str = os.getenv("LOG_LEVEL", "INFO")
