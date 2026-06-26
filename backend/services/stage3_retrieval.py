"""
stage3_retrieval.py
───────────────────
Stage 3 of the OC-RAG pipeline — Ontology-Filtered Document Retrieval.

Responsibilities:
- Build the structured query string from the Student Career Theory Profile
  using the fixed template: "[Holland type] [competency domains] [career outcomes]"
- Encode the query string into a 384-dim dense vector using all-MiniLM-L6-v2
- Query ChromaDB with a WHERE clause filtering ONLY to ontology-verified
  candidate program codes from Stage 2
- Return the top 10 chunks with their cosine similarity scores
- Cosine similarity scores are preserved for Stage 4 tie-breaking

Input:  career_profile (Stage 1 output), candidate_programs (Stage 2 output)
Output: list of 10 dicts, each with: text, metadata, cosine_similarity, rank
"""

import os
import logging
from sentence_transformers import SentenceTransformer
import chromadb

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
CHROMA_DIR      = os.getenv(
    "CHROMA_DIR",
    os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "lspu_lb_programs")
EMBED_MODEL     = "all-MiniLM-L6-v2"
TOP_K           = 10   # Number of chunks retrieved for Stage 4

# ── Load embedding model once at module level ──────────────────────────────────
_EMBED_MODEL = None

def get_embed_model() -> SentenceTransformer:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        logger.info(f"Loading embedding model: {EMBED_MODEL}")
        _EMBED_MODEL = SentenceTransformer(EMBED_MODEL)
        logger.info("Embedding model loaded.")
    return _EMBED_MODEL


# ── ChromaDB client ────────────────────────────────────────────────────────────
_CHROMA_COLLECTION = None

def get_collection():
    global _CHROMA_COLLECTION
    if _CHROMA_COLLECTION is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _CHROMA_COLLECTION = client.get_collection(COLLECTION_NAME)
        logger.info(f"ChromaDB collection '{COLLECTION_NAME}' loaded. "
                    f"Total documents: {_CHROMA_COLLECTION.count()}")
    return _CHROMA_COLLECTION


# ── Query String Builder ───────────────────────────────────────────────────────
def _build_query_string(career_profile: dict) -> str:
    """
    Builds the fixed-format query string for Stage 3 embedding.

    Template: "[Holland type] [competency domains] [career outcome preferences]"
    Example:  "Social Counseling Communication Research Guidance Counseling Community Development"

    This fixed concatenated format is used uniformly across all 30 test profiles
    as specified in the OC-RAG methodology.
    """
    parts = []

    # 1. Dominant Holland type
    holland_primary = career_profile.get("holland_primary")
    if holland_primary:
        parts.append(holland_primary)

    # 2. High-confidence competency domains
    domains = career_profile.get("self_efficacy_domains", [])
    if domains:
        parts.extend(domains)

    # 3. Career outcome preferences
    outcomes = career_profile.get("career_outcome_preferences", [])
    if outcomes:
        parts.extend(outcomes)

    query_string = " ".join(parts)
    logger.info(f"Stage 3 query string: '{query_string}'")
    return query_string


# ── WHERE Clause Builder ───────────────────────────────────────────────────────
def _build_where_clause(candidate_programs: list[str]) -> dict:
    """
    Builds the ChromaDB WHERE clause to restrict retrieval
    to only ontology-verified candidate programs from Stage 2.

    Single program: {"program_code": "BSPSY"}
    Multiple programs: {"program_code": {"$in": ["BSPSY", "BSEd"]}}
    """
    if len(candidate_programs) == 1:
        return {"program_code": candidate_programs[0]}
    return {"program_code": {"$in": candidate_programs}}


# ── Main Stage 3 Function ──────────────────────────────────────────────────────
def run_stage3(career_profile: dict, candidate_programs: list[str]) -> dict:
    """
    Runs Stage 3 — Ontology-Filtered Document Retrieval.

    Args:
        career_profile    : Student Career Theory Profile from Stage 1
        candidate_programs: List of program codes from Stage 2

    Returns:
        dict with keys:
            query_string    : The constructed query string
            retrieved_chunks: List of up to 10 dicts, each with:
                              {text, metadata, cosine_similarity, retrieval_rank}
            chunks_returned : int — actual number of chunks returned
            candidate_filter: list of program codes used in WHERE clause
    """
    model      = get_embed_model()
    collection = get_collection()

    # Build query string
    query_string = _build_query_string(career_profile)

    # Fallback: if query string is empty (total Stage 1 failure), use a generic query
    if not query_string.strip():
        query_string = "college program career skills recommendation"
        logger.warning("Stage 3: Empty query string — using generic fallback query.")

    # Encode query
    query_embedding = model.encode([query_string], convert_to_list=True)

    # Build WHERE clause
    where_clause = _build_where_clause(candidate_programs)
    logger.info(f"Stage 3: Querying ChromaDB with WHERE {where_clause}")

    # Query ChromaDB
    # n_results capped at available documents to avoid ChromaDB errors
    available = collection.count()
    n_results  = min(TOP_K, available)

    results = collection.query(
        query_embeddings = query_embedding,
        where            = where_clause,
        n_results        = n_results,
        include          = ["documents", "metadatas", "distances"],
    )

    # Parse results
    # ChromaDB returns cosine distance (0=identical, 2=opposite)
    # Convert to cosine similarity: similarity = 1 - distance
    retrieved_chunks = []
    docs      = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for rank, (doc, meta, dist) in enumerate(zip(docs, metadatas, distances), start=1):
        cosine_similarity = round(1 - dist, 6)
        retrieved_chunks.append({
            "text"            : doc,
            "metadata"        : meta,
            "cosine_similarity": cosine_similarity,
            "retrieval_rank"  : rank,
        })

    logger.info(
        f"Stage 3 complete. Retrieved {len(retrieved_chunks)} chunks from "
        f"{len(set(c['metadata']['program_code'] for c in retrieved_chunks))} programs."
    )

    # Log top 3 for debugging
    for chunk in retrieved_chunks[:3]:
        logger.debug(
            f"  Rank {chunk['retrieval_rank']}: {chunk['metadata']['program_code']} | "
            f"{chunk['metadata']['chunk_type']} | sim={chunk['cosine_similarity']}"
        )

    return {
        "query_string"    : query_string,
        "retrieved_chunks": retrieved_chunks,
        "chunks_returned" : len(retrieved_chunks),
        "candidate_filter": candidate_programs,
    }
