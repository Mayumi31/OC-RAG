"""
ingest_chromadb.py
──────────────────
Chunks all 9 program JSONs into semantically meaningful text segments,
embeds them using all-MiniLM-L6-v2 (384-dim), and stores them in ChromaDB
with metadata that enables Stage 3's WHERE clause filtering by program code.

Run once before starting the pipeline:
    python3 scripts/ingest_chromadb.py

Requirements:
    pip install chromadb sentence-transformers

Expected folder structure (relative to project root):
    data/programs/          ← enriched program JSONs
    data/chroma_db/         ← ChromaDB persistent storage (auto-created)

Each chunk is stored with metadata:
    {
        "program_code": "BSPSY",
        "program_name": "Bachelor of Science in Psychology",
        "college": "...",
        "chunk_type": "overview" | "ideal_student" | "skills" | "careers" |
                      "work_environment" | "strand_alignment" | "holland_profile" |
                      "competency_domains" | "career_outcomes" | "keywords",
        "holland_primary": "Social",
        "holland_secondary": "Investigative",
        "grade_threshold": 85.0
    }

Stage 3 uses the WHERE clause:
    collection.query(
        query_embeddings=[...],
        where={"program_code": {"$in": ["BSPSY", "BSEd"]}},
        n_results=10
    )
"""

import json
import os
import sys

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ── Configuration ──────────────────────────────────────────────────────────────
PROGRAM_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "programs")
CHROMA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
COLLECTION_NAME = "lspu_lb_programs"
EMBED_MODEL     = "all-MiniLM-L6-v2"   # 384-dim, matches Stage 3 design

# ── Chunking Strategy ──────────────────────────────────────────────────────────
# Each program JSON is split into 10 semantically distinct chunk types.
# This gives Stage 3 fine-grained retrieval rather than one large blob per program.
# Each chunk carries the full program metadata so Stage 3 filtering works correctly.

def build_chunks(program: dict) -> list[dict]:
    """
    Converts one program JSON into a list of text chunks with metadata.
    Returns a list of dicts: {"id": str, "text": str, "metadata": dict}
    """
    code     = program["code"]
    name     = program["name"]
    college  = program["college"]
    h_primary   = program.get("holland_primary", "")
    h_secondary = program.get("holland_secondary", "")
    threshold   = float(program.get("grade_threshold", 80))

    # Shared metadata attached to every chunk of this program
    base_meta = {
        "program_code"    : code,
        "program_name"    : name,
        "college"         : college,
        "holland_primary" : h_primary,
        "holland_secondary": h_secondary,
        "grade_threshold" : threshold,
    }

    chunks = []

    # 1. Overview — what the program is about
    chunks.append({
        "id"      : f"{code}_overview",
        "text"    : (
            f"{name} ({code}) — Program Overview. "
            f"{program.get('overview', '')}"
        ),
        "metadata": {**base_meta, "chunk_type": "overview"},
    })

    # 2. Ideal Student — who thrives in this program
    chunks.append({
        "id"      : f"{code}_ideal_student",
        "text"    : (
            f"{name} — Ideal Student Profile. "
            f"{program.get('ideal_student', '')}"
        ),
        "metadata": {**base_meta, "chunk_type": "ideal_student"},
    })

    # 3. Skills — what the program develops
    chunks.append({
        "id"      : f"{code}_skills",
        "text"    : (
            f"{name} — Skills and Competencies Developed. "
            f"Students in this program develop the following skills: "
            f"{program.get('skills', '')}"
        ),
        "metadata": {**base_meta, "chunk_type": "skills"},
    })

    # 4. Careers — job outcomes
    chunks.append({
        "id"      : f"{code}_careers",
        "text"    : (
            f"{name} — Career Pathways and Job Outcomes. "
            f"Graduates of this program pursue careers such as: "
            f"{program.get('careers', '')}"
        ),
        "metadata": {**base_meta, "chunk_type": "careers"},
    })

    # 5. Work Environment — day-to-day professional context
    chunks.append({
        "id"      : f"{code}_work_environment",
        "text"    : (
            f"{name} — Work Environment and Daily Professional Life. "
            f"{program.get('work_environment', '')}"
        ),
        "metadata": {**base_meta, "chunk_type": "work_environment"},
    })

    # 6. Strand Alignment + Grade Threshold — admission context
    chunks.append({
        "id"      : f"{code}_strand_alignment",
        "text"    : (
            f"{name} — Senior High School Strand Alignment and Grade Requirement. "
            f"{program.get('strand_alignment', '')} "
            f"The minimum General Weighted Average (GWA) required for admission "
            f"to {name} is {threshold}."
        ),
        "metadata": {**base_meta, "chunk_type": "strand_alignment"},
    })

    # 7. Holland Profile — personality-environment fit narrative
    rationale = program.get("holland_rationale", "")
    holland_types_str = " and ".join(program.get("holland_types", []))
    chunks.append({
        "id"      : f"{code}_holland_profile",
        "text"    : (
            f"{name} — Personality and Career Type (Holland RIASEC). "
            f"This program aligns with the {holland_types_str} Holland personality types. "
            f"{rationale}"
        ),
        "metadata": {**base_meta, "chunk_type": "holland_profile"},
    })

    # 8. Competency Domains — structured list as natural language
    comp_domains = program.get("competency_domains", [])
    if comp_domains:
        comp_text = ", ".join(comp_domains)
        chunks.append({
            "id"      : f"{code}_competency_domains",
            "text"    : (
                f"{name} — Required Competency Domains. "
                f"Students who excel in this program typically demonstrate strength in: "
                f"{comp_text}. "
                f"These competency areas are central to academic and professional success in {name}."
            ),
            "metadata": {**base_meta, "chunk_type": "competency_domains"},
        })

    # 9. Career Outcome Categories — structured list as natural language
    career_cats = program.get("career_outcome_categories", [])
    if career_cats:
        career_text = ", ".join(career_cats)
        chunks.append({
            "id"      : f"{code}_career_outcomes",
            "text"    : (
                f"{name} — Career Outcome Categories. "
                f"This program prepares graduates for the following career outcome areas: "
                f"{career_text}. "
                f"These represent the primary professional pathways available to {name} graduates."
            ),
            "metadata": {**base_meta, "chunk_type": "career_outcomes"},
        })

    # 10. Keywords — enriched retrieval surface for informal student language
    keywords = program.get("keywords", "")
    if keywords:
        chunks.append({
            "id"      : f"{code}_keywords",
            "text"    : (
                f"{name} — Related Topics and Keywords. "
                f"This program is associated with the following concepts and areas: "
                f"{keywords}"
            ),
            "metadata": {**base_meta, "chunk_type": "keywords"},
        })

    return chunks


# ── Main Ingestion ─────────────────────────────────────────────────────────────
def main():
    # Load programs
    program_files = sorted([
        f for f in os.listdir(PROGRAM_DIR) if f.endswith(".json")
    ])
    if not program_files:
        print(f"ERROR: No JSON files found in {PROGRAM_DIR}")
        sys.exit(1)

    programs = []
    for fname in program_files:
        with open(os.path.join(PROGRAM_DIR, fname)) as f:
            programs.append(json.load(f))
    print(f"Loaded {len(programs)} program JSONs: {[p['code'] for p in programs]}")

    # Build all chunks
    all_chunks = []
    for program in programs:
        chunks = build_chunks(program)
        all_chunks.extend(chunks)
        print(f"  {program['code']:10s} → {len(chunks)} chunks")
    print(f"\nTotal chunks to ingest: {len(all_chunks)}")

    # Load embedding model
    print(f"\nLoading embedding model: {EMBED_MODEL} ...")
    model = SentenceTransformer(EMBED_MODEL)
    print("Model loaded.")

    # Generate embeddings
    texts = [chunk["text"] for chunk in all_chunks]
    print(f"Generating {len(texts)} embeddings ...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_list=True)
    print("Embeddings generated.")

    # Set up ChromaDB
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Drop and recreate collection for clean ingestion
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"Dropped existing collection: {COLLECTION_NAME}")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # cosine similarity as per Stage 3 design
    )
    print(f"Created collection: {COLLECTION_NAME}")

    # Ingest in batches to avoid memory issues
    BATCH_SIZE = 50
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch      = all_chunks[i : i + BATCH_SIZE]
        batch_embs = embeddings[i : i + BATCH_SIZE]

        collection.add(
            ids        = [c["id"] for c in batch],
            embeddings = batch_embs,
            documents  = [c["text"] for c in batch],
            metadatas  = [c["metadata"] for c in batch],
        )
        print(f"  Ingested batch {i // BATCH_SIZE + 1} ({len(batch)} chunks)")

    print(f"\nIngestion complete. Total documents in collection: {collection.count()}")

    # ── Verification: simulate a Stage 3 query ────────────────────────────────
    print("\n── Verification: Stage 3 simulation ─────────────────────────────────")
    test_query  = "Social Counseling Communication Guidance Counseling Community Development"
    test_filter = {"program_code": {"$in": ["BSPSY", "BSEd", "BSHRM"]}}

    query_embedding = model.encode([test_query], convert_to_list=True)

    results = collection.query(
        query_embeddings = query_embedding,
        where            = test_filter,
        n_results        = 5,
        include          = ["documents", "metadatas", "distances"],
    )

    print(f"Query : '{test_query}'")
    print(f"Filter: programs in {list(test_filter['program_code']['$in'])}")
    print(f"Top 5 results:")
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        cosine_sim = 1 - dist   # ChromaDB cosine distance → similarity
        print(f"  [{i+1}] {meta['program_code']:8s} | {meta['chunk_type']:20s} | "
              f"cosine_sim={cosine_sim:.4f}")
        print(f"       {doc[:100]}...")

    print("\nDone. ChromaDB is ready for Stage 3 retrieval.")


if __name__ == "__main__":
    main()
