"""
stage5_generation.py
────────────────────
Stage 5 of the OC-RAG pipeline — Grounded Recommendation Generation.

Responsibilities:
- Assemble the final grounded recommendation prompt using:
    (1) Student's SHS strand
    (2) Student's GWA
    (3) Student's original free-text answers (all 8)
    (4) Top 5 reranked chunks from Stage 4
- Call Groq API with strict grounding instructions:
    - Only recommend programs present in the retrieved chunks
    - Only cite grade thresholds present in the chunks
    - Only cite career outcomes present in the chunks
    - Do NOT fabricate any information not in the retrieved context
- Return the final recommendation text + metadata

Three compounding grounding layers active at this stage:
    Stage 2: Ontology pre-filtering (only valid candidate programs reached here)
    Stage 3: Constrained retrieval (only candidate program chunks retrieved)
    Stage 4: Grounded re-ranking (irrelevant chunks deprioritized)
This combination significantly reduces hallucination probability.

Input:  student_info (strand, gwa, answers), top5_chunks (Stage 4)
Output: dict with recommendation text, program recommended, grounding metadata
"""

import json
import logging
from groq import Groq

logger = logging.getLogger(__name__)

LLM_MODEL       = "llama3-70b-8192"
LLM_TEMPERATURE = 0.3   # Slight warmth for natural recommendation language


# ── Recommendation Prompt Builder ─────────────────────────────────────────────
def _build_recommendation_prompt(
    strand        : str,
    gwa           : float,
    student_answers: dict,
    top5_chunks   : list[dict],
    career_profile : dict
) -> str:
    """
    Builds the Stage 5 grounded recommendation prompt.

    The LLM is given:
    - The student's strand, GWA, and original answers
    - The top 5 reranked program chunks as the ONLY allowed context
    - Explicit grounding restrictions to prevent hallucination
    """

    # Format student answers block
    answers_block = "\n".join(
        f"Q{i+1}: {student_answers.get(f'q{i+1}', '[No answer provided]')}"
        for i in range(8)
    )

    # Format context chunks block
    context_block = ""
    for i, chunk in enumerate(top5_chunks, start=1):
        meta = chunk["metadata"]
        context_block += (
            f"\n[CONTEXT {i}] Program: {meta.get('program_code')} — {meta.get('program_name')}\n"
            f"Chunk type: {meta.get('chunk_type')}\n"
            f"Grade threshold: {meta.get('grade_threshold')}\n"
            f"Content:\n{chunk['text']}\n"
            f"{'─' * 60}\n"
        )

    # Extract Holland and SCCT signals for the prompt
    holland     = career_profile.get("holland_primary", "Not identified")
    domains     = ", ".join(career_profile.get("self_efficacy_domains", [])) or "Not identified"
    outcomes    = ", ".join(career_profile.get("career_outcome_preferences", [])) or "Not identified"

    prompt = f"""You are a college program counselor at Laguna State Polytechnic University - Los Baños (LSPU-LB). Your task is to generate a personalized, grounded college program recommendation for a senior high school student.

═══════════════════════════════════════════════════════════
STUDENT INFORMATION
═══════════════════════════════════════════════════════════
Senior High School Strand : {strand}
General Weighted Average  : {gwa}
Holland RIASEC Type       : {holland}
Key Competency Strengths  : {domains}
Career Outcome Preferences: {outcomes}

STUDENT'S ORIGINAL ANSWERS TO THE CAREER ASSESSMENT:
{answers_block}

═══════════════════════════════════════════════════════════
RETRIEVED PROGRAM CONTEXT (your ONLY allowed information source)
═══════════════════════════════════════════════════════════
{context_block}

═══════════════════════════════════════════════════════════
GROUNDING RULES — YOU MUST FOLLOW THESE STRICTLY
═══════════════════════════════════════════════════════════
1. You may ONLY recommend programs that appear in the retrieved context above.
2. You may ONLY cite grade thresholds (GWA requirements) that are explicitly stated in the context above.
3. You may ONLY mention career outcomes and program descriptions that are present in the context above.
4. Do NOT invent, assume, or add any information that is not in the retrieved context.
5. Do NOT recommend programs that are NOT in the retrieved context, even if you know they exist.
6. If the student's GWA is below a program's threshold as stated in the context, you MUST acknowledge this honestly and explain that the student may need to meet the requirement, but you may still recommend the program as a career fit.
7. Base the recommendation on alignment between the student's answers, strand, and the program information in the context.

═══════════════════════════════════════════════════════════
RECOMMENDATION FORMAT
═══════════════════════════════════════════════════════════
Write a clear, warm, and personalized recommendation with the following structure:

1. PRIMARY RECOMMENDATION — State the single most suitable program and why it fits the student based on their answers, strand, and career profile. Cite specific details from both the student's answers and the retrieved context.

2. STRAND AND GRADE ALIGNMENT — Confirm whether the student's strand aligns with the program. State the program's grade threshold as given in the context and compare it to the student's GWA. Be honest if there is a gap.

3. CAREER FIT — Describe the specific career outcomes from the context that match the student's expressed preferences and competency strengths.

4. SECONDARY RECOMMENDATION (optional) — If the context contains a second strong match, briefly recommend it as an alternative.

5. CLOSING — A brief encouraging note to the student.

Write in a warm, professional, and direct tone. Address the student as "you." Keep the total response under 500 words.
"""
    return prompt


# ── Program Code Extractor ─────────────────────────────────────────────────────
def _extract_recommended_programs(top5_chunks: list[dict], recommendation_text: str) -> list[str]:
    """
    Extracts which program codes were actually mentioned in the recommendation.
    Used for logging and evaluation — not for grounding enforcement.
    """
    program_codes = list(set(
        chunk["metadata"]["program_code"]
        for chunk in top5_chunks
    ))
    mentioned = [code for code in program_codes if code in recommendation_text]
    return mentioned


# ── Main Stage 5 Function ──────────────────────────────────────────────────────
def run_stage5(
    student_info  : dict,
    top5_chunks   : list[dict],
    career_profile: dict,
    groq_api_key  : str
) -> dict:
    """
    Runs Stage 5 — Grounded Recommendation Generation.

    Args:
        student_info   : dict with keys: strand (str), gwa (float), answers (dict q1..q8)
        top5_chunks    : Top 5 reranked chunks from Stage 4
        career_profile : Student Career Theory Profile from Stage 1
        groq_api_key   : Groq API key string

    Returns:
        dict with keys:
            recommendation_text     : The full recommendation string
            programs_in_context     : List of program codes in the top 5 chunks
            programs_mentioned      : List of program codes mentioned in recommendation
            grounding_context_used  : Number of chunks used as context
            prompt_used             : The full prompt sent to the LLM (for logging)
            generation_success      : bool
    """
    client = Groq(api_key=groq_api_key)

    strand  = student_info.get("strand", "Unknown")
    gwa     = student_info.get("gwa", 0.0)
    answers = student_info.get("answers", {})

    programs_in_context = list(set(
        chunk["metadata"]["program_code"] for chunk in top5_chunks
    ))

    prompt = _build_recommendation_prompt(
        strand         = strand,
        gwa            = gwa,
        student_answers= answers,
        top5_chunks    = top5_chunks,
        career_profile = career_profile,
    )

    try:
        response = client.chat.completions.create(
            model       = LLM_MODEL,
            temperature = LLM_TEMPERATURE,
            max_tokens  = 1000,
            messages    = [{"role": "user", "content": prompt}]
        )
        recommendation_text = response.choices[0].message.content.strip()
        generation_success  = True
        logger.info("Stage 5: Recommendation generated successfully.")

    except Exception as e:
        logger.error(f"Stage 5: Recommendation generation failed: {e}")
        recommendation_text = (
            "We were unable to generate a recommendation at this time due to a system error. "
            "Please try again or consult a guidance counselor directly."
        )
        generation_success = False

    programs_mentioned = _extract_recommended_programs(top5_chunks, recommendation_text)

    logger.info(
        f"Stage 5 complete. Programs in context: {programs_in_context} | "
        f"Programs mentioned: {programs_mentioned}"
    )

    return {
        "recommendation_text"   : recommendation_text,
        "programs_in_context"   : programs_in_context,
        "programs_mentioned"    : programs_mentioned,
        "grounding_context_used": len(top5_chunks),
        "prompt_used"           : prompt,
        "generation_success"    : generation_success,
    }
