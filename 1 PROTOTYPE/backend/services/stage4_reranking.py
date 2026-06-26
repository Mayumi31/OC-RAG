"""
stage4_reranking.py
───────────────────
Stage 4 of the OC-RAG pipeline — LLM Reranking.

Responsibilities:
- Take the 10 chunks from Stage 3
- Call Groq API (temperature=0) to score each chunk 1-5 based on
  alignment with the student's Holland type and SCCT signals
- Select the top 5 after sorting by LLM score (descending)
- Use cosine similarity from Stage 3 as a deterministic tiebreaker
  when two chunks have the same LLM score — no additional LLM call needed
- Return the top 5 reranked chunks for Stage 5

This stage resolves the limitation of pure lexical similarity:
a chunk may rank high in Stage 3 due to keyword overlap but not align
well with the student's Holland type or SCCT competency profile.
The LLM reranking corrects for this.

Input:  career_profile (Stage 1), retrieved_chunks (Stage 3)
Output: top 5 reranked chunks with LLM scores and final rank
"""

import json
import logging
from groq import Groq

logger = logging.getLogger(__name__)

LLM_MODEL       = "llama3-70b-8192"
LLM_TEMPERATURE = 0   # Explicitly 0 for deterministic scoring as per methodology


# ── Reranking Prompt Builder ───────────────────────────────────────────────────
def _build_reranking_prompt(career_profile: dict, chunks: list[dict]) -> str:
    """
    Builds the Stage 4 scoring and ranking prompt.
    Instructs the LLM to assign a score 1-5 to each chunk based on
    alignment with the student's Holland type and SCCT signals.
    """
    holland_primary  = career_profile.get("holland_primary", "Not identified")
    holland_secondary = career_profile.get("holland_secondary", "Not identified")
    domains          = career_profile.get("self_efficacy_domains", [])
    outcomes         = career_profile.get("career_outcome_preferences", [])

    # Format chunks for the prompt
    chunks_block = ""
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk["metadata"]
        chunks_block += (
            f"\n[CHUNK {i}]\n"
            f"Program: {meta.get('program_code', 'Unknown')} — {meta.get('program_name', '')}\n"
            f"Chunk type: {meta.get('chunk_type', '')}\n"
            f"Content: {chunk['text'][:600]}\n"
        )

    domains_str  = ", ".join(domains)  if domains  else "Not identified"
    outcomes_str = ", ".join(outcomes) if outcomes else "Not identified"

    prompt = f"""You are a career counselor evaluating how well each program information chunk matches a student's career theory profile.

STUDENT CAREER THEORY PROFILE:
- Primary Holland RIASEC Type: {holland_primary}
- Secondary Holland RIASEC Type: {holland_secondary}
- High-Confidence Competency Domains (Self-Efficacy): {domains_str}
- Career Outcome Preferences: {outcomes_str}

SCORING CRITERIA:
Score each chunk from 1 to 5 based on how well it aligns with the student's Holland type and SCCT signals above:
  5 = Directly and strongly relevant to the student's Holland type AND competency domains AND career outcomes
  4 = Clearly relevant to Holland type AND either competency domains OR career outcomes
  3 = Moderately relevant — matches some signals but not the core profile
  2 = Weakly relevant — surface-level match only, not aligned with the core profile
  1 = Not relevant — does not align with the student's Holland type or SCCT signals

IMPORTANT: Base your scores ONLY on alignment with the student's career theory profile above.
Do NOT favor chunks simply because they contain similar keywords. Focus on conceptual alignment.

CHUNKS TO SCORE:
{chunks_block}

REQUIRED OUTPUT FORMAT:
Respond with ONLY a valid JSON array. No explanation, no preamble, no markdown.
Each object must have exactly: "chunk_number" (int 1-10) and "score" (int 1-5).

[
  {{"chunk_number": 1, "score": 4}},
  {{"chunk_number": 2, "score": 2}},
  ...
]
"""
    return prompt


# ── Score Parser ───────────────────────────────────────────────────────────────
def _parse_scores(raw: str, num_chunks: int) -> dict[int, int]:
    """
    Parses the LLM scoring response into a dict of {chunk_number: score}.
    Falls back to uniform score of 3 if parsing fails.
    """
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines   = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

        scores_list = json.loads(cleaned)
        return {item["chunk_number"]: int(item["score"]) for item in scores_list}

    except Exception as e:
        logger.warning(f"Stage 4: Score parsing failed: {e}. Assigning uniform score of 3.")
        return {i: 3 for i in range(1, num_chunks + 1)}


# ── Main Stage 4 Function ──────────────────────────────────────────────────────
def run_stage4(career_profile: dict, retrieved_chunks: list[dict], groq_api_key: str) -> dict:
    """
    Runs Stage 4 — LLM Reranking.

    Args:
        career_profile  : Student Career Theory Profile from Stage 1
        retrieved_chunks: List of up to 10 chunks from Stage 3
                          (each must have: text, metadata, cosine_similarity, retrieval_rank)
        groq_api_key    : Groq API key string

    Returns:
        dict with keys:
            top5_chunks     : List of 5 dicts, each with:
                              {text, metadata, cosine_similarity, retrieval_rank,
                               llm_score, final_rank}
            scoring_raw     : Raw LLM scoring response (for logging)
            rerank_applied  : bool — True if LLM scoring changed the Stage 3 order
            scores_by_chunk : dict of {chunk_number: score} from LLM
    """
    client = Groq(api_key=groq_api_key)

    # ── Call LLM for scoring ───────────────────────────────────────────────────
    prompt = _build_reranking_prompt(career_profile, retrieved_chunks)

    try:
        response = client.chat.completions.create(
            model       = LLM_MODEL,
            temperature = LLM_TEMPERATURE,
            max_tokens  = 500,
            messages    = [{"role": "user", "content": prompt}]
        )
        raw_scores = response.choices[0].message.content.strip()
        logger.info("Stage 4: LLM scoring response received.")

    except Exception as e:
        logger.warning(f"Stage 4: LLM scoring call failed: {e}. Using cosine similarity order.")
        raw_scores = ""

    # ── Parse scores ───────────────────────────────────────────────────────────
    scores_by_chunk = _parse_scores(raw_scores, len(retrieved_chunks))

    # ── Attach scores to chunks ────────────────────────────────────────────────
    scored_chunks = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        llm_score = scores_by_chunk.get(i, 3)
        scored_chunks.append({
            **chunk,
            "llm_score"   : llm_score,
            "chunk_number": i,
        })

    # ── Sort: primary = LLM score (desc), tiebreaker = cosine similarity (desc)
    # This is deterministic — no additional LLM call needed for ties
    scored_chunks.sort(
        key=lambda c: (c["llm_score"], c["cosine_similarity"]),
        reverse=True
    )

    # ── Select top 5 ──────────────────────────────────────────────────────────
    top5 = scored_chunks[:5]
    for final_rank, chunk in enumerate(top5, start=1):
        chunk["final_rank"] = final_rank

    # ── Check if reranking changed the Stage 3 order ──────────────────────────
    original_order = [c["retrieval_rank"] for c in retrieved_chunks[:5]]
    reranked_order = [c["retrieval_rank"] for c in top5]
    rerank_applied = original_order != reranked_order

    logger.info(
        f"Stage 4 complete. Top 5 programs: "
        f"{[c['metadata']['program_code'] for c in top5]} | "
        f"Reranking changed order: {rerank_applied}"
    )
    logger.info(
        f"Stage 4 scores: "
        + ", ".join(
            f"Chunk {c['chunk_number']}({c['metadata']['program_code']})={c['llm_score']}"
            for c in top5
        )
    )

    return {
        "top5_chunks"    : top5,
        "scoring_raw"    : raw_scores,
        "rerank_applied" : rerank_applied,
        "scores_by_chunk": scores_by_chunk,
    }
