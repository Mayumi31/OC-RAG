"""
schemas.py
──────────
Pydantic models for all OC-RAG API request and response bodies.
Used by FastAPI for automatic validation, serialization, and OpenAPI docs.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum


# ── Enums ──────────────────────────────────────────────────────────────────────
class StrandEnum(str, Enum):
    STEM  = "STEM"
    ABM   = "ABM"
    HUMSS = "HUMSS"
    GAS   = "GAS"

class HollandTypeEnum(str, Enum):
    Realistic    = "Realistic"
    Investigative = "Investigative"
    Artistic     = "Artistic"
    Social       = "Social"
    Enterprising = "Enterprising"
    Conventional = "Conventional"

class ExtractionConfidenceEnum(str, Enum):
    HIGH     = "HIGH"
    MEDIUM   = "MEDIUM"
    LOW      = "LOW"
    FALLBACK = "FALLBACK"


# ── Request Models ─────────────────────────────────────────────────────────────
class StudentAnswers(BaseModel):
    """The student's 8 open-ended assessment answers."""
    q1: str = Field(..., min_length=10, description="Activities enjoyed most")
    q2: str = Field(..., min_length=10, description="Problem-solving approach")
    q3: str = Field(..., min_length=10, description="Helping or leading experience")
    q4: str = Field(..., min_length=10, description="Academic strengths and weaknesses")
    q5: str = Field(..., min_length=10, description="Ideal career without constraints")
    q6: str = Field(..., min_length=10, description="Preferred type of work")
    q7: str = Field(..., min_length=10, description="Top 3 personal strengths with examples")
    q8: str = Field(..., min_length=10, description="Work and life vision in 10 years")


class RecommendationRequest(BaseModel):
    """Full request body for the /recommend endpoint."""
    student_id           : str          = Field(..., description="Unique student identifier e.g. STU_001")
    strand               : StrandEnum   = Field(..., description="SHS strand: STEM, ABM, HUMSS, or GAS")
    gwa                  : float        = Field(..., ge=60.0, le=100.0, description="General Weighted Average")
    answers              : StudentAnswers
    ground_truth_program : Optional[str] = Field(None, description="Known correct program for evaluation (optional)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "student_id": "STU_001",
                "strand": "HUMSS",
                "gwa": 88.5,
                "answers": {
                    "q1": "I love listening to people and helping them with their problems...",
                    "q2": "I reflect on all sides before responding...",
                    "q3": "I volunteered at the guidance office and listened to students...",
                    "q4": "I am strongest in English, Filipino, and Social Science...",
                    "q5": "I want to be a guidance counselor or psychologist...",
                    "q6": "Option d — helping and caring for people...",
                    "q7": "Empathy, communication, and reflection...",
                    "q8": "I see myself in a school or clinic helping people navigate challenges..."
                },
                "ground_truth_program": "BSPSY"
            }
        }
    }


# ── Intermediate Stage Response Models ────────────────────────────────────────
class CareerProfile(BaseModel):
    """Stage 1 output — Student Career Theory Profile."""
    holland_primary            : Optional[str]       = None
    holland_secondary          : Optional[str]       = None
    holland_code               : Optional[str]       = None
    self_efficacy_domains      : list[str]           = []
    career_outcome_preferences : list[str]           = []
    extraction_confidence      : ExtractionConfidenceEnum = ExtractionConfidenceEnum.FALLBACK
    extraction_notes           : Optional[str]       = None


class OntologyQuerySummary(BaseModel):
    """Stage 2 query metadata."""
    holland_types_queried          : list[str] = []
    holland_individuals_found      : int       = 0
    competency_domains_queried     : list[str] = []
    competency_individuals_found   : int       = 0
    career_outcomes_queried        : list[str] = []
    career_individuals_found       : int       = 0


class CandidateSelectionResult(BaseModel):
    """Stage 2 output — Ontology candidate program selection."""
    candidate_programs : list[str]
    fallback_level     : int   = Field(..., ge=0, le=3)
    fallback_reason    : Optional[str] = None
    query_summary      : OntologyQuerySummary


class ChunkMetadata(BaseModel):
    """Metadata attached to each ChromaDB chunk."""
    program_code      : str
    program_name      : str
    college           : str
    chunk_type        : str
    holland_primary   : str
    holland_secondary : str
    grade_threshold   : float


class RetrievedChunk(BaseModel):
    """A single retrieved chunk from Stage 3 or Stage 4."""
    text              : str
    metadata          : ChunkMetadata
    cosine_similarity : float
    retrieval_rank    : int
    llm_score         : Optional[int]  = None
    final_rank        : Optional[int]  = None


class RetrievalResult(BaseModel):
    """Stage 3 output — filtered document retrieval."""
    query_string      : str
    retrieved_chunks  : list[RetrievedChunk]
    chunks_returned   : int
    candidate_filter  : list[str]


class RerankingResult(BaseModel):
    """Stage 4 output — LLM reranking."""
    top5_chunks      : list[RetrievedChunk]
    rerank_applied   : bool
    scores_by_chunk  : dict


# ── Main Response Models ───────────────────────────────────────────────────────
class RecommendationResponse(BaseModel):
    """
    Full response from the /recommend endpoint.
    Contains the final recommendation plus all intermediate pipeline outputs
    for transparency and evaluation logging.
    """
    student_id              : str
    strand                  : str
    gwa                     : float
    recommendation          : str
    programs_mentioned      : list[str]
    career_profile          : CareerProfile
    candidate_programs      : list[str]
    stage3_query_string     : str
    top5_program_codes      : list[str]
    rerank_applied          : bool
    s2_fallback_level       : int
    total_time_seconds      : float
    success                 : bool

    model_config = {
        "json_schema_extra": {
            "example": {
                "student_id"        : "STU_001",
                "strand"            : "HUMSS",
                "gwa"               : 88.5,
                "recommendation"    : "Based on your responses, we recommend Bachelor of Science in Psychology...",
                "programs_mentioned": ["BSPSY"],
                "career_profile": {
                    "holland_primary"           : "Social",
                    "holland_secondary"         : "Investigative",
                    "holland_code"              : "SI",
                    "self_efficacy_domains"     : ["Counseling", "Communication", "Research"],
                    "career_outcome_preferences": ["Guidance Counseling", "Community Development"],
                    "extraction_confidence"     : "HIGH",
                    "extraction_notes"          : None
                },
                "candidate_programs"    : ["BSPSY", "BSEd", "BSHRM"],
                "stage3_query_string"   : "Social Counseling Communication Research Guidance Counseling",
                "top5_program_codes"    : ["BSPSY", "BSPSY", "BSEd", "BSPSY", "BSEd"],
                "rerank_applied"        : True,
                "s2_fallback_level"     : 0,
                "total_time_seconds"    : 4.32,
                "success"               : True
            }
        }
    }


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""
    status          : str
    ontology_loaded : bool
    chromadb_ready  : bool
    llm_model       : str
    environment     : str


class QuestionsResponse(BaseModel):
    """Response from the /questions endpoint — returns the 8 assessment questions."""
    total_questions : int
    questions       : list[dict]
