"""
oc_rag_pipeline.py
──────────────────
OC-RAG Pipeline Orchestrator — coordinates Stage 1 through Stage 5.

Responsibilities:
- Accept a student profile (strand, gwa, answers)
- Pass data between stages in the correct format
- Collect all intermediate outputs for logging
- Log complete pipeline run to SQLite for the experiment runner
- Never silently fail — every stage has fallback handling
- Return the final recommendation + full pipeline log

Pipeline flow:
  Student Answers
      ↓
  Stage 1 → Career Theory Profile (Holland + SCCT)
      ↓
  Stage 2 → Candidate Program Codes (ontology traversal)
      ↓
  Stage 3 → Top 10 Chunks (filtered cosine similarity)
      ↓
  Stage 4 → Top 5 Chunks (LLM reranked)
      ↓
  Stage 5 → Final Recommendation (grounded generation)
      ↓
  SQLite Log
"""

import os
import json
import logging
import sqlite3
import time
from datetime import datetime

from services.stage1_career_theory  import run_stage1
from services.stage2_ontology_select import run_stage2
from services.stage3_retrieval       import run_stage3
from services.stage4_reranking       import run_stage4
from services.stage5_generation      import run_stage5

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
DB_PATH      = os.getenv(
    "DB_PATH",
    os.path.join(os.path.dirname(__file__), "..", "data", "oc_rag_results.db")
)


# ── SQLite Logger ──────────────────────────────────────────────────────────────
def _init_db(db_path: str):
    """Creates the SQLite results table if it does not exist."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            run_timestamp           TEXT,
            condition               TEXT DEFAULT 'OC-RAG',
            student_id              TEXT,
            strand                  TEXT,
            gwa                     REAL,
            ground_truth_program    TEXT,

            -- Stage 1 outputs
            s1_holland_primary      TEXT,
            s1_holland_secondary    TEXT,
            s1_holland_code         TEXT,
            s1_self_efficacy_domains TEXT,
            s1_career_outcomes      TEXT,
            s1_confidence           TEXT,
            s1_notes                TEXT,

            -- Stage 2 outputs
            s2_candidate_programs   TEXT,
            s2_fallback_level       INTEGER,
            s2_fallback_reason      TEXT,

            -- Stage 3 outputs
            s3_query_string         TEXT,
            s3_chunks_returned      INTEGER,
            s3_top_programs         TEXT,

            -- Stage 4 outputs
            s4_top5_programs        TEXT,
            s4_llm_scores           TEXT,
            s4_rerank_applied       INTEGER,

            -- Stage 5 outputs
            s5_recommendation       TEXT,
            s5_programs_mentioned   TEXT,
            s5_generation_success   INTEGER,

            -- Timing
            total_time_seconds      REAL,
            error_log               TEXT
        )
    """)
    conn.commit()
    conn.close()


def _log_to_db(db_path: str, log_data: dict):
    """Writes one pipeline run to SQLite."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        INSERT INTO pipeline_runs (
            run_timestamp, condition, student_id, strand, gwa, ground_truth_program,
            s1_holland_primary, s1_holland_secondary, s1_holland_code,
            s1_self_efficacy_domains, s1_career_outcomes, s1_confidence, s1_notes,
            s2_candidate_programs, s2_fallback_level, s2_fallback_reason,
            s3_query_string, s3_chunks_returned, s3_top_programs,
            s4_top5_programs, s4_llm_scores, s4_rerank_applied,
            s5_recommendation, s5_programs_mentioned, s5_generation_success,
            total_time_seconds, error_log
        ) VALUES (
            :run_timestamp, :condition, :student_id, :strand, :gwa, :ground_truth_program,
            :s1_holland_primary, :s1_holland_secondary, :s1_holland_code,
            :s1_self_efficacy_domains, :s1_career_outcomes, :s1_confidence, :s1_notes,
            :s2_candidate_programs, :s2_fallback_level, :s2_fallback_reason,
            :s3_query_string, :s3_chunks_returned, :s3_top_programs,
            :s4_top5_programs, :s4_llm_scores, :s4_rerank_applied,
            :s5_recommendation, :s5_programs_mentioned, :s5_generation_success,
            :total_time_seconds, :error_log
        )
    """, log_data)
    conn.commit()
    conn.close()


# ── Main Pipeline Function ─────────────────────────────────────────────────────
def run_oc_rag_pipeline(
    student_id           : str,
    strand               : str,
    gwa                  : float,
    answers              : dict,
    ground_truth_program : str = None,
    log_to_db            : bool = True
) -> dict:
    """
    Runs the full OC-RAG pipeline for one student.

    Args:
        student_id           : Unique student identifier (e.g. "STU_001")
        strand               : SHS strand (STEM, ABM, HUMSS, GAS)
        gwa                  : General Weighted Average (float)
        answers              : dict with keys q1..q8
        ground_truth_program : Known correct program code for evaluation (optional)
        log_to_db            : Whether to write results to SQLite

    Returns:
        dict with keys:
            recommendation      : Final recommendation text
            career_profile      : Stage 1 output
            candidate_programs  : Stage 2 output list
            retrieved_chunks    : Stage 3 output list (10 chunks)
            top5_chunks         : Stage 4 output list (5 chunks)
            stage_outputs       : Full intermediate outputs from all stages
            pipeline_log        : Flattened log dict for SQLite
            total_time_seconds  : Float
            success             : bool
    """
    pipeline_start = time.time()
    error_log      = []
    _init_db(DB_PATH)

    logger.info(f"━━━ OC-RAG Pipeline START — {student_id} ({strand}, GWA: {gwa}) ━━━")

    student_info = {"strand": strand, "gwa": gwa, "answers": answers}

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    logger.info("── Stage 1: Career Theory Signal Extraction")
    try:
        t1_start      = time.time()
        career_profile = run_stage1(answers, GROQ_API_KEY)
        t1_time        = round(time.time() - t1_start, 3)
        logger.info(f"   Stage 1 done in {t1_time}s")
    except Exception as e:
        error_log.append(f"Stage 1 error: {e}")
        logger.error(f"Stage 1 fatal error: {e}")
        career_profile = {
            "holland_primary": None, "holland_secondary": None, "holland_code": None,
            "self_efficacy_domains": [], "career_outcome_preferences": [],
            "extraction_confidence": "FALLBACK", "extraction_notes": str(e)
        }

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    logger.info("── Stage 2: Ontology-Guided Candidate Selection")
    try:
        t2_start   = time.time()
        s2_output  = run_stage2(career_profile)
        t2_time    = round(time.time() - t2_start, 3)
        candidate_programs = s2_output["candidate_programs"]
        logger.info(f"   Stage 2 done in {t2_time}s — candidates: {candidate_programs}")
    except Exception as e:
        error_log.append(f"Stage 2 error: {e}")
        logger.error(f"Stage 2 fatal error: {e}")
        s2_output          = {"candidate_programs": [], "fallback_level": 3, "fallback_reason": str(e), "query_summary": {}}
        candidate_programs = []

    # Safety: if Stage 2 returned nothing, use all programs
    if not candidate_programs:
        from services.stage2_ontology_select import get_ontology_instance
        onto               = get_ontology_instance()
        candidate_programs = [p.hasProgramCode[0] for p in onto.AcademicProgram.instances()]
        s2_output["fallback_level"]  = 3
        s2_output["fallback_reason"] = "Stage 2 returned empty list — full program set used."
        logger.warning(f"Stage 2 safety net: using all {len(candidate_programs)} programs.")

    # ── Stage 3 ───────────────────────────────────────────────────────────────
    logger.info("── Stage 3: Ontology-Filtered Document Retrieval")
    try:
        t3_start  = time.time()
        s3_output = run_stage3(career_profile, candidate_programs)
        t3_time   = round(time.time() - t3_start, 3)
        logger.info(f"   Stage 3 done in {t3_time}s — {s3_output['chunks_returned']} chunks retrieved")
    except Exception as e:
        error_log.append(f"Stage 3 error: {e}")
        logger.error(f"Stage 3 fatal error: {e}")
        s3_output = {"query_string": "", "retrieved_chunks": [], "chunks_returned": 0, "candidate_filter": candidate_programs}

    # ── Stage 4 ───────────────────────────────────────────────────────────────
    logger.info("── Stage 4: LLM Reranking")
    retrieved_chunks = s3_output.get("retrieved_chunks", [])
    try:
        t4_start  = time.time()
        s4_output = run_stage4(career_profile, retrieved_chunks, GROQ_API_KEY)
        t4_time   = round(time.time() - t4_start, 3)
        top5_chunks = s4_output["top5_chunks"]
        logger.info(f"   Stage 4 done in {t4_time}s — top 5: {[c['metadata']['program_code'] for c in top5_chunks]}")
    except Exception as e:
        error_log.append(f"Stage 4 error: {e}")
        logger.error(f"Stage 4 fatal error: {e}")
        top5_chunks = retrieved_chunks[:5]
        for i, c in enumerate(top5_chunks, 1):
            c["llm_score"]  = 3
            c["final_rank"] = i
        s4_output = {"top5_chunks": top5_chunks, "scoring_raw": "", "rerank_applied": False, "scores_by_chunk": {}}

    # ── Stage 5 ───────────────────────────────────────────────────────────────
    logger.info("── Stage 5: Grounded Recommendation Generation")
    try:
        t5_start  = time.time()
        s5_output = run_stage5(student_info, top5_chunks, career_profile, GROQ_API_KEY)
        t5_time   = round(time.time() - t5_start, 3)
        logger.info(f"   Stage 5 done in {t5_time}s")
    except Exception as e:
        error_log.append(f"Stage 5 error: {e}")
        logger.error(f"Stage 5 fatal error: {e}")
        s5_output = {
            "recommendation_text": "Recommendation generation failed. Please consult a guidance counselor.",
            "programs_in_context": [], "programs_mentioned": [],
            "grounding_context_used": 0, "prompt_used": "", "generation_success": False
        }

    total_time = round(time.time() - pipeline_start, 3)
    logger.info(f"━━━ OC-RAG Pipeline COMPLETE — {student_id} | Total time: {total_time}s ━━━")

    # ── Build SQLite log dict ──────────────────────────────────────────────────
    pipeline_log = {
        "run_timestamp"          : datetime.utcnow().isoformat(),
        "condition"              : "OC-RAG",
        "student_id"             : student_id,
        "strand"                 : strand,
        "gwa"                    : gwa,
        "ground_truth_program"   : ground_truth_program or "",
        "s1_holland_primary"     : career_profile.get("holland_primary", ""),
        "s1_holland_secondary"   : career_profile.get("holland_secondary", ""),
        "s1_holland_code"        : career_profile.get("holland_code", ""),
        "s1_self_efficacy_domains": json.dumps(career_profile.get("self_efficacy_domains", [])),
        "s1_career_outcomes"     : json.dumps(career_profile.get("career_outcome_preferences", [])),
        "s1_confidence"          : career_profile.get("extraction_confidence", ""),
        "s1_notes"               : career_profile.get("extraction_notes", ""),
        "s2_candidate_programs"  : json.dumps(s2_output.get("candidate_programs", [])),
        "s2_fallback_level"      : s2_output.get("fallback_level", -1),
        "s2_fallback_reason"     : s2_output.get("fallback_reason", ""),
        "s3_query_string"        : s3_output.get("query_string", ""),
        "s3_chunks_returned"     : s3_output.get("chunks_returned", 0),
        "s3_top_programs"        : json.dumps(list(set(
            c["metadata"]["program_code"] for c in s3_output.get("retrieved_chunks", [])
        ))),
        "s4_top5_programs"       : json.dumps([c["metadata"]["program_code"] for c in top5_chunks]),
        "s4_llm_scores"          : json.dumps(s4_output.get("scores_by_chunk", {})),
        "s4_rerank_applied"      : int(s4_output.get("rerank_applied", False)),
        "s5_recommendation"      : s5_output.get("recommendation_text", ""),
        "s5_programs_mentioned"  : json.dumps(s5_output.get("programs_mentioned", [])),
        "s5_generation_success"  : int(s5_output.get("generation_success", False)),
        "total_time_seconds"     : total_time,
        "error_log"              : json.dumps(error_log) if error_log else "",
    }

    if log_to_db:
        _log_to_db(DB_PATH, pipeline_log)
        logger.info(f"Pipeline run logged to SQLite: {DB_PATH}")

    return {
        "recommendation"    : s5_output.get("recommendation_text", ""),
        "career_profile"    : career_profile,
        "candidate_programs": candidate_programs,
        "retrieved_chunks"  : retrieved_chunks,
        "top5_chunks"       : top5_chunks,
        "stage_outputs"     : {
            "stage1": career_profile,
            "stage2": s2_output,
            "stage3": s3_output,
            "stage4": s4_output,
            "stage5": s5_output,
        },
        "pipeline_log"      : pipeline_log,
        "total_time_seconds": total_time,
        "success"           : len(error_log) == 0,
    }
