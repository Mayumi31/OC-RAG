"""
main.py
───────
FastAPI entrypoint for the OC-RAG College Recommendation API.

Endpoints:
    GET  /              → API info
    GET  /health        → System health check (ontology, ChromaDB, LLM)
    GET  /questions     → Returns the 8 assessment questions for the frontend
    POST /recommend     → Runs the full OC-RAG pipeline for one student
    GET  /results/{id}  → Retrieves a logged pipeline run by student ID

Run locally:
    uvicorn main:app --reload --port 8000

Environment variables required in .env:
    GROQ_API_KEY=your_key
"""

import json
import logging
import sqlite3
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import config
from models.schemas import (
    RecommendationRequest,
    RecommendationResponse,
    CareerProfile,
    HealthResponse,
    QuestionsResponse,
)
from pipeline.oc_rag_pipeline import run_oc_rag_pipeline

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = getattr(logging, config.LOG_LEVEL, logging.INFO),
    format  = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("oc_rag.main")


# ── Lifespan: warm up heavy resources on startup ───────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Pre-loads the ontology and embedding model on startup
    so the first request is not slow.
    """
    logger.info("OC-RAG API starting up — warming up resources...")
    try:
        from services.stage2_ontology_select import get_ontology_instance
        get_ontology_instance()
        logger.info("Ontology loaded.")
    except Exception as e:
        logger.warning(f"Ontology pre-load failed (will retry on first request): {e}")

    try:
        from services.stage3_retrieval import get_embed_model, get_collection
        get_embed_model()
        get_collection()
        logger.info("Embedding model and ChromaDB collection loaded.")
    except Exception as e:
        logger.warning(f"ChromaDB/embedding pre-load failed (will retry on first request): {e}")

    logger.info("OC-RAG API ready.")
    yield
    logger.info("OC-RAG API shutting down.")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = config.API_TITLE,
    version     = config.API_VERSION,
    description = config.API_DESC,
    lifespan    = lifespan,
)

# CORS — allow React frontend (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
def root():
    return {
        "name"       : config.API_TITLE,
        "version"    : config.API_VERSION,
        "environment": config.ENVIRONMENT,
        "endpoints"  : {
            "health"   : "GET /health",
            "questions": "GET /questions",
            "recommend": "POST /recommend",
            "results"  : "GET /results/{student_id}",
            "docs"     : "GET /docs",
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """
    Checks whether the ontology, ChromaDB collection, and LLM are accessible.
    Use this before running the experiment to confirm all systems are ready.
    """
    ontology_loaded = False
    chromadb_ready  = False

    try:
        from services.stage2_ontology_select import get_ontology_instance
        onto            = get_ontology_instance()
        programs        = list(onto.AcademicProgram.instances())
        ontology_loaded = len(programs) == 9
    except Exception as e:
        logger.warning(f"Health check — ontology error: {e}")

    try:
        from services.stage3_retrieval import get_collection
        col            = get_collection()
        chromadb_ready = col.count() > 0
    except Exception as e:
        logger.warning(f"Health check — ChromaDB error: {e}")

    return HealthResponse(
        status          = "ok" if (ontology_loaded and chromadb_ready) else "degraded",
        ontology_loaded = ontology_loaded,
        chromadb_ready  = chromadb_ready,
        llm_model       = config.LLM_MODEL,
        environment     = config.ENVIRONMENT,
    )


@app.get("/questions", response_model=QuestionsResponse, tags=["Assessment"])
def get_questions():
    """
    Returns the 8 open-ended assessment questions.
    The React frontend calls this to render the questionnaire dynamically.
    """
    try:
        with open(config.QUESTIONS_PATH) as f:
            data = json.load(f)
        return QuestionsResponse(
            total_questions = data["questionnaire_metadata"]["total_questions"],
            questions       = [
                {
                    "id"      : q["id"],
                    "number"  : q["number"],
                    "text"    : q["text"],
                    "probe"   : q.get("follow_up_probe", ""),
                }
                for q in data["questions"]
            ]
        )
    except Exception as e:
        logger.error(f"Failed to load questions: {e}")
        raise HTTPException(status_code=500, detail="Could not load assessment questions.")


@app.post("/recommend", response_model=RecommendationResponse, tags=["Pipeline"])
def recommend(request: RecommendationRequest):
    """
    Runs the full OC-RAG 5-stage pipeline for one student and returns
    a grounded college program recommendation.

    Pipeline stages:
      Stage 1 — Career Theory Signal Extraction (LLM)
      Stage 2 — Ontology-Guided Candidate Selection (owlready2)
      Stage 3 — Ontology-Filtered Document Retrieval (ChromaDB)
      Stage 4 — LLM Reranking (Groq)
      Stage 5 — Grounded Recommendation Generation (Groq)

    All intermediate outputs are logged to SQLite automatically.
    """
    logger.info(
        f"POST /recommend — student_id={request.student_id} "
        f"strand={request.strand} gwa={request.gwa}"
    )

    try:
        result = run_oc_rag_pipeline(
            student_id           = request.student_id,
            strand               = request.strand.value,
            gwa                  = request.gwa,
            answers              = request.answers.model_dump(),
            ground_truth_program = request.ground_truth_program,
            log_to_db            = True,
        )
    except Exception as e:
        logger.error(f"Pipeline error for {request.student_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    # Build response
    s4_output     = result["stage_outputs"]["stage4"]
    s3_output     = result["stage_outputs"]["stage3"]
    s2_output     = result["stage_outputs"]["stage2"]
    career_profile = result["career_profile"]

    return RecommendationResponse(
        student_id          = request.student_id,
        strand              = request.strand.value,
        gwa                 = request.gwa,
        recommendation      = result["recommendation"],
        programs_mentioned  = result["stage_outputs"]["stage5"].get("programs_mentioned", []),
        career_profile      = CareerProfile(**career_profile),
        candidate_programs  = result["candidate_programs"],
        stage3_query_string = s3_output.get("query_string", ""),
        top5_program_codes  = [
            c["metadata"]["program_code"] for c in result["top5_chunks"]
        ],
        rerank_applied      = s4_output.get("rerank_applied", False),
        s2_fallback_level   = s2_output.get("fallback_level", -1),
        total_time_seconds  = result["total_time_seconds"],
        success             = result["success"],
    )


@app.get("/results/{student_id}", tags=["Evaluation"])
def get_results(student_id: str):
    """
    Retrieves the most recent logged pipeline run for a given student ID.
    Useful for reviewing what the pipeline produced for a specific profile.
    """
    try:
        conn   = sqlite3.connect(str(config.DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT * FROM pipeline_runs WHERE student_id = ? ORDER BY id DESC LIMIT 1",
            (student_id,)
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            raise HTTPException(
                status_code=404,
                detail=f"No pipeline run found for student_id '{student_id}'."
            )

        row_dict = dict(row)

        # Parse JSON fields back to lists/dicts for the response
        for field in [
            "s1_self_efficacy_domains", "s1_career_outcomes",
            "s2_candidate_programs", "s3_top_programs",
            "s4_top5_programs", "s4_llm_scores",
            "s5_programs_mentioned", "error_log"
        ]:
            if row_dict.get(field):
                try:
                    row_dict[field] = json.loads(row_dict[field])
                except Exception:
                    pass

        return row_dict

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving results for {student_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
