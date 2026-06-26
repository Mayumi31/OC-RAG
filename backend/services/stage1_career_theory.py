"""
stage1_career_theory.py
───────────────────────
Stage 1 of the OC-RAG pipeline — Career Theory Signal Extraction.

Responsibilities:
- Load the question_theory_map.json as explicit grounding context
- Build the extraction prompt with the theory map injected
- Call Groq API (llama3-70b-8192) to extract Holland type + SCCT signals
- Validate the extracted Holland type against the 6 RIASEC types
- Return a structured Student Career Theory Profile dict
- On extraction failure, fall back to SCCT-only signals
- Never silently terminate — always return something usable for Stage 2

Output schema:
{
    "holland_primary": "Social",
    "holland_secondary": "Investigative",
    "holland_code": "SI",
    "self_efficacy_domains": ["Counseling", "Communication", "Research"],
    "career_outcome_preferences": ["Guidance Counseling", "Community Development"],
    "extraction_confidence": "HIGH" | "MEDIUM" | "LOW" | "FALLBACK",
    "extraction_notes": "optional string"
}
"""

import json
import os
import logging
from groq import Groq

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
VALID_HOLLAND_TYPES = {
    "Realistic", "Investigative", "Artistic",
    "Social", "Enterprising", "Conventional"
}

# Paths — override via config.py if needed
THEORY_MAP_PATH = os.getenv(
    "THEORY_MAP_PATH",
    os.path.join(os.path.dirname(__file__), "..", "data", "question_theory_map.json")
)

LLM_MODEL       = os.getenv("LLM_MODEL", "llama3-70b-8192")
LLM_TEMPERATURE = 0.0   # deterministic extraction


# ── Load theory map once at module level ───────────────────────────────────────
def _load_theory_map(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

_THEORY_MAP = None

def get_theory_map() -> dict:
    global _THEORY_MAP
    if _THEORY_MAP is None:
        _THEORY_MAP = _load_theory_map(THEORY_MAP_PATH)
    return _THEORY_MAP


# ── Prompt Builder ─────────────────────────────────────────────────────────────
def _build_extraction_prompt(student_answers: dict, theory_map: dict) -> str:
    """
    Constructs the Stage 1 extraction prompt.
    Injects:
      - RIASEC type definitions with example signals
      - SCCT construct definitions
      - Per-question theory targets and extraction guidance
      - Extraction rules and output schema
    """

    # Serialize key sections of the theory map for injection
    riasec_defs   = theory_map["riasec_definitions"]["types"]
    scct_defs     = theory_map["scct_constructs"]["constructs"]
    q_map         = {q["question_id"]: q for q in theory_map["question_theory_map"]}
    rules         = theory_map["extraction_rules"]
    output_schema = rules["output_json_structure"]["template"]

    # Format RIASEC definitions block
    riasec_block = ""
    for type_name, info in riasec_defs.items():
        riasec_block += (
            f"  {type_name} ({info['code']}): {info['enjoys']}. "
            f"Example signals: {'; '.join(info['example_signals'][:3])}.\n"
        )

    # Format per-question guidance block
    question_guidance_block = ""
    for qid in ["q1","q2","q3","q4","q5","q6","q7","q8"]:
        q      = q_map.get(qid, {})
        answer = student_answers.get(qid, "[No answer provided]")
        question_guidance_block += (
            f"\n[{qid.upper()}] {q.get('question_summary', '')}\n"
            f"Theory target: {q.get('primary_theory_target', '')} | "
            f"RIASEC to watch: {q.get('riasec_types_to_watch', [])} | "
            f"SCCT construct: {q.get('scct_construct', 'None')} | "
            f"Weight: {q.get('weight', '')}\n"
            f"Extraction guidance: {q.get('extraction_guidance', '')}\n"
            f"Student answer: \"{answer}\"\n"
        )

    # Format competency domain map from q4
    comp_domain_map = q_map["q4"].get("competency_domain_map", {})
    comp_map_block  = "\n".join(
        f"  {subj} → {', '.join(domains)}"
        for subj, domains in comp_domain_map.items()
    )

    # Format extraction rules
    holland_rule   = rules["holland_code_assignment"]["rule"]
    efficacy_rule  = rules["self_efficacy_domains"]["rule"]
    outcome_rule   = rules["career_outcome_preferences"]["rule"]
    fallback_rule  = rules["holland_code_assignment"]["fallback"]

    prompt = f"""You are a career theory specialist trained in Holland's RIASEC Typology and Social Cognitive Career Theory (SCCT). Your task is to analyze a student's open-ended questionnaire responses and extract structured career theory signals.

═══════════════════════════════════════════════════════════
THEORETICAL FRAMEWORK
═══════════════════════════════════════════════════════════

HOLLAND RIASEC TYPES — definitions and example signals:
{riasec_block}

SCCT CONSTRUCTS:
  Self-Efficacy: {scct_defs['self_efficacy']['definition']}
  Detection: {scct_defs['self_efficacy']['how_to_detect']}

  Outcome Expectations: {scct_defs['outcome_expectations']['definition']}
  Detection: {scct_defs['outcome_expectations']['how_to_detect']}

═══════════════════════════════════════════════════════════
SUBJECT-TO-COMPETENCY DOMAIN MAP (use for Question 4)
═══════════════════════════════════════════════════════════
{comp_map_block}

═══════════════════════════════════════════════════════════
STUDENT RESPONSES WITH EXTRACTION GUIDANCE PER QUESTION
═══════════════════════════════════════════════════════════
{question_guidance_block}

═══════════════════════════════════════════════════════════
EXTRACTION RULES
═══════════════════════════════════════════════════════════
Holland code assignment: {holland_rule}
Tiebreaker: {fallback_rule}

Self-efficacy domains: {efficacy_rule}
Minimum domains: {rules['self_efficacy_domains']['minimum_domains']} | Maximum: {rules['self_efficacy_domains']['maximum_domains']}
Priority questions for self-efficacy: {rules['self_efficacy_domains']['priority_questions_for_self_efficacy']}

Career outcome preferences: {outcome_rule}
Priority questions for outcomes: {rules['career_outcome_preferences']['priority_questions_for_outcomes']}

═══════════════════════════════════════════════════════════
VALID HOLLAND TYPES (you must use exactly these spellings)
═══════════════════════════════════════════════════════════
Realistic, Investigative, Artistic, Social, Enterprising, Conventional

═══════════════════════════════════════════════════════════
REQUIRED OUTPUT FORMAT
═══════════════════════════════════════════════════════════
Respond with ONLY a valid JSON object. No explanation, no preamble, no markdown.
Use exactly this structure:

{{
  "holland_primary": "<one of the 6 valid types>",
  "holland_secondary": "<one of the 6 valid types or null>",
  "holland_code": "<2-letter code e.g. SI, EC, IR>",
  "self_efficacy_domains": ["<domain 1>", "<domain 2>", ...],
  "career_outcome_preferences": ["<outcome 1>", "<outcome 2>", "<outcome 3>"],
  "extraction_confidence": "<HIGH | MEDIUM | LOW>",
  "extraction_notes": "<brief note on any contradictions or ambiguities, or null>"
}}
"""
    return prompt


# ── Holland Validator ──────────────────────────────────────────────────────────
def _validate_holland(profile: dict) -> tuple[bool, str]:
    """
    Validates the extracted Holland type against the 6 valid RIASEC types.
    Returns (is_valid, reason).
    """
    primary = profile.get("holland_primary", "")
    if not primary:
        return False, "holland_primary is missing"
    if primary not in VALID_HOLLAND_TYPES:
        return False, f"'{primary}' is not a valid RIASEC type"

    secondary = profile.get("holland_secondary")
    if secondary and secondary not in VALID_HOLLAND_TYPES:
        return False, f"holland_secondary '{secondary}' is not a valid RIASEC type"

    if not profile.get("self_efficacy_domains"):
        return False, "self_efficacy_domains is empty"

    if not profile.get("career_outcome_preferences"):
        return False, "career_outcome_preferences is empty"

    return True, "OK"


# ── SCCT-only Fallback ─────────────────────────────────────────────────────────
def _build_fallback_prompt(student_answers: dict) -> str:
    """
    Simplified fallback prompt when Holland extraction fails.
    Extracts only SCCT signals (self-efficacy domains + career outcome preferences).
    Holland fields will be null — Stage 2 will use SCCT-only querying.
    """
    answers_block = "\n".join(
        f"Q{i+1}: {student_answers.get(f'q{i+1}', '[No answer]')}"
        for i in range(8)
    )

    return f"""You are a career counselor. A student has answered 8 career-related questions.
Your task is to extract ONLY their self-efficacy domains and career outcome preferences.
Do NOT attempt to assign a Holland type.

STUDENT ANSWERS:
{answers_block}

Self-efficacy domains: Academic subjects or skill areas where the student expresses confidence or demonstrated strength.
Career outcome preferences: Career paths or job roles the student has expressed interest in.

Respond with ONLY this JSON object. No explanation, no markdown:
{{
  "holland_primary": null,
  "holland_secondary": null,
  "holland_code": null,
  "self_efficacy_domains": ["<domain 1>", "<domain 2>", "<domain 3>"],
  "career_outcome_preferences": ["<outcome 1>", "<outcome 2>", "<outcome 3>"],
  "extraction_confidence": "FALLBACK",
  "extraction_notes": "Holland extraction failed — SCCT-only fallback applied."
}}
"""


# ── LLM Caller ────────────────────────────────────────────────────────────────
def _call_llm(client: Groq, prompt: str, temperature: float = 0.0) -> str:
    """Calls Groq API and returns the raw text response."""
    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=temperature,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


# ── JSON Parser ────────────────────────────────────────────────────────────────
def _parse_json_response(raw: str) -> dict:
    """
    Safely parses LLM JSON response.
    Strips markdown fences if present before parsing.
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines   = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
    return json.loads(cleaned)


# ── Main Stage 1 Function ─────────────────────────────────────────────────────
def run_stage1(student_answers: dict, groq_api_key: str) -> dict:
    """
    Runs Stage 1 — Career Theory Signal Extraction.

    Args:
        student_answers: dict with keys q1..q8, values are student's free-text answers
        groq_api_key: Groq API key string

    Returns:
        Student Career Theory Profile dict with keys:
            holland_primary, holland_secondary, holland_code,
            self_efficacy_domains, career_outcome_preferences,
            extraction_confidence, extraction_notes
    
    Never raises — always returns a usable profile (FALLBACK if all else fails).
    """
    client     = Groq(api_key=groq_api_key)
    theory_map = get_theory_map()

    # ── Attempt 1: Full extraction with theory map ─────────────────────────────
    try:
        logger.info("Stage 1: Running full career theory extraction...")
        prompt   = _build_extraction_prompt(student_answers, theory_map)
        raw      = _call_llm(client, prompt, temperature=LLM_TEMPERATURE)
        profile  = _parse_json_response(raw)
        is_valid, reason = _validate_holland(profile)

        if is_valid:
            logger.info(
                f"Stage 1 complete. Holland: {profile['holland_primary']} | "
                f"Confidence: {profile['extraction_confidence']}"
            )
            return profile
        else:
            logger.warning(f"Stage 1 validation failed: {reason}. Attempting fallback.")

    except Exception as e:
        logger.warning(f"Stage 1 full extraction error: {e}. Attempting fallback.")

    # ── Attempt 2: SCCT-only fallback ─────────────────────────────────────────
    try:
        logger.info("Stage 1: Running SCCT-only fallback extraction...")
        fallback_prompt = _build_fallback_prompt(student_answers)
        raw             = _call_llm(client, fallback_prompt, temperature=LLM_TEMPERATURE)
        profile         = _parse_json_response(raw)
        profile["extraction_confidence"] = "FALLBACK"
        logger.info("Stage 1 fallback complete. Holland will be null — Stage 2 uses SCCT only.")
        return profile

    except Exception as e:
        logger.error(f"Stage 1 fallback also failed: {e}. Returning minimal safe profile.")

    # ── Attempt 3: Hardcoded safe minimum — pipeline must not terminate ────────
    return {
        "holland_primary"          : None,
        "holland_secondary"        : None,
        "holland_code"             : None,
        "self_efficacy_domains"    : [],
        "career_outcome_preferences": [],
        "extraction_confidence"    : "FALLBACK",
        "extraction_notes"         : "All extraction attempts failed. Stage 2 will use full program scan."
    }
