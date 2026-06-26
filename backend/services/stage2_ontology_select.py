"""
stage2_ontology_select.py
─────────────────────────
Stage 2 of the OC-RAG pipeline — Ontology-Guided Candidate Program Selection.

Responsibilities:
- Load lspu_lb_ontology.owl using owlready2
- Query programs using AND/OR logic:
    Holland match AND (competency match OR career outcome match)
- Apply 3-level cascading fallback:
    Level 1: Holland AND (competency OR career) — primary query
    Level 2: Holland only — if Level 1 returns nothing
    Level 3: All programs on competency/career — if Level 2 returns nothing
- Return list of candidate program codes for Stage 3 WHERE clause filtering
- Never returns an empty list — pipeline must always continue

Input:  Student Career Theory Profile from Stage 1
Output: List of candidate program code strings e.g. ["BSPSY", "BSEd", "BSHRM"]
"""

import os
import logging
from owlready2 import get_ontology, World

logger = logging.getLogger(__name__)

# ── Path ───────────────────────────────────────────────────────────────────────
ONTOLOGY_PATH = os.getenv(
    "ONTOLOGY_PATH",
    os.path.join(os.path.dirname(__file__), "..", "ontology", "lspu_lb_ontology.owl")
)

# ── Load ontology once at module level ─────────────────────────────────────────
_ONTO = None

def get_ontology_instance():
    global _ONTO
    if _ONTO is None:
        logger.info(f"Loading ontology from: {ONTOLOGY_PATH}")
        _ONTO = get_ontology(f"file://{ONTOLOGY_PATH}").load()
        logger.info("Ontology loaded successfully.")
    return _ONTO


# ── Individual Resolvers ───────────────────────────────────────────────────────
def _resolve_holland_individuals(onto, holland_types: list) -> set:
    """Resolves Holland type strings to ontology individuals."""
    targets = set()
    for ht in [h for h in holland_types if h]:
        ind = onto.search_one(label=ht)
        if ind:
            targets.add(ind)
        else:
            logger.warning(f"Stage 2: Holland type '{ht}' not found in ontology.")
    return targets


def _resolve_competency_individuals(onto, domains: list) -> set:
    """Resolves competency domain strings to ontology individuals."""
    targets = set()
    for domain in domains:
        ind = onto.search_one(label=domain)
        if ind:
            targets.add(ind)
        else:
            logger.debug(f"Stage 2: Competency domain '{domain}' not found in ontology — skipping.")
    return targets


def _resolve_career_individuals(onto, outcomes: list) -> set:
    """Resolves career outcome strings to ontology individuals."""
    targets = set()
    for outcome in outcomes:
        ind = onto.search_one(label=outcome)
        if ind:
            targets.add(ind)
        else:
            logger.debug(f"Stage 2: Career outcome '{outcome}' not found in ontology — skipping.")
    return targets


# ── Query Logic ────────────────────────────────────────────────────────────────
def _query_programs(onto, holland_targets: set, comp_targets: set, career_targets: set) -> list[str]:
    """
    Primary AND/OR query:
    Program qualifies if: Holland match AND (competency match OR career outcome match)
    """
    candidates = []
    for program in onto.AcademicProgram.instances():
        p_holland    = set(program.alignsWithHollandType)
        p_competency = set(program.requiresCompetency)
        p_career     = set(program.leadsToCareer)

        holland_match    = bool(p_holland & holland_targets)
        competency_match = bool(p_competency & comp_targets)
        career_match     = bool(p_career & career_targets)

        if holland_match and (competency_match or career_match):
            candidates.append(program.hasProgramCode[0])

    return candidates


def _query_holland_only(onto, holland_targets: set) -> list[str]:
    """
    Fallback Level 1: Holland type match only.
    Used when primary AND/OR query returns no candidates.
    """
    return [
        p.hasProgramCode[0]
        for p in onto.AcademicProgram.instances()
        if set(p.alignsWithHollandType) & holland_targets
    ]


def _query_all_on_scct(onto, comp_targets: set, career_targets: set) -> list[str]:
    """
    Fallback Level 2: All programs queried on competency and/or career outcome.
    Used when Holland type is null or produces no results.
    """
    return [
        p.hasProgramCode[0]
        for p in onto.AcademicProgram.instances()
        if (set(p.requiresCompetency) & comp_targets)
        or (set(p.leadsToCareer) & career_targets)
    ]


def _query_all_programs(onto) -> list[str]:
    """
    Final safety net: return all programs.
    Used only when all other queries return nothing.
    Ensures the pipeline never terminates silently.
    """
    return [p.hasProgramCode[0] for p in onto.AcademicProgram.instances()]


# ── Main Stage 2 Function ──────────────────────────────────────────────────────
def run_stage2(career_profile: dict) -> dict:
    """
    Runs Stage 2 — Ontology-Guided Candidate Program Selection.

    Args:
        career_profile: Student Career Theory Profile from Stage 1

    Returns:
        dict with keys:
            candidate_programs: list of program code strings
            fallback_level: 0 (primary), 1 (holland only), 2 (scct only), 3 (all programs)
            fallback_reason: string explaining why fallback was triggered, or None
            query_summary: dict summarizing what was queried
    """
    onto = get_ontology_instance()

    # Extract signals from Stage 1 profile
    holland_primary   = career_profile.get("holland_primary")
    holland_secondary = career_profile.get("holland_secondary")
    efficacy_domains  = career_profile.get("self_efficacy_domains", [])
    career_outcomes   = career_profile.get("career_outcome_preferences", [])
    confidence        = career_profile.get("extraction_confidence", "FALLBACK")

    # Resolve ontology individuals
    holland_types    = [h for h in [holland_primary, holland_secondary] if h]
    holland_targets  = _resolve_holland_individuals(onto, holland_types)
    comp_targets     = _resolve_competency_individuals(onto, efficacy_domains)
    career_targets   = _resolve_career_individuals(onto, career_outcomes)

    query_summary = {
        "holland_types_queried"   : holland_types,
        "holland_individuals_found": len(holland_targets),
        "competency_domains_queried": efficacy_domains,
        "competency_individuals_found": len(comp_targets),
        "career_outcomes_queried" : career_outcomes,
        "career_individuals_found": len(career_targets),
    }

    # ── Level 0: Primary AND/OR query ─────────────────────────────────────────
    if holland_targets and (comp_targets or career_targets):
        candidates = _query_programs(onto, holland_targets, comp_targets, career_targets)
        if candidates:
            logger.info(f"Stage 2 [Level 0 — AND/OR]: {len(candidates)} candidates: {candidates}")
            return {
                "candidate_programs": candidates,
                "fallback_level"    : 0,
                "fallback_reason"   : None,
                "query_summary"     : query_summary,
            }
        else:
            logger.warning("Stage 2 [Level 0]: No candidates from AND/OR query. Trying Level 1.")

    # ── Level 1: Holland-only fallback ────────────────────────────────────────
    if holland_targets:
        candidates = _query_holland_only(onto, holland_targets)
        if candidates:
            logger.info(f"Stage 2 [Level 1 — Holland only]: {len(candidates)} candidates: {candidates}")
            return {
                "candidate_programs": candidates,
                "fallback_level"    : 1,
                "fallback_reason"   : "AND/OR query returned no results — broadened to Holland type only.",
                "query_summary"     : query_summary,
            }
        else:
            logger.warning("Stage 2 [Level 1]: Holland-only query returned nothing. Trying Level 2.")

    # ── Level 2: SCCT-only fallback ───────────────────────────────────────────
    if comp_targets or career_targets:
        candidates = _query_all_on_scct(onto, comp_targets, career_targets)
        if candidates:
            logger.info(f"Stage 2 [Level 2 — SCCT only]: {len(candidates)} candidates: {candidates}")
            return {
                "candidate_programs": candidates,
                "fallback_level"    : 2,
                "fallback_reason"   : "Holland type unavailable or unresolved — SCCT-only query across all programs.",
                "query_summary"     : query_summary,
            }
        else:
            logger.warning("Stage 2 [Level 2]: SCCT-only query returned nothing. Falling back to all programs.")

    # ── Level 3: All programs safety net ──────────────────────────────────────
    candidates = _query_all_programs(onto)
    logger.warning(f"Stage 2 [Level 3 — All programs]: Returning all {len(candidates)} programs.")
    return {
        "candidate_programs": candidates,
        "fallback_level"    : 3,
        "fallback_reason"   : "All targeted queries returned no results — returning full program set.",
        "query_summary"     : query_summary,
    }
