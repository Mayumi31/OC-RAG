"""
build_ontology.py
Generates lspu_lb_ontology.owl from the 9 enriched program JSONs.
Encodes:
  - Holland RIASEC types (6 individuals)
  - Academic Programs (9 individuals)
  - Competency domains (individuals)
  - Career outcome categories (individuals)
  - Object properties: alignsWithHollandType, requiresCompetency, leadsToCareer
  - Data properties: hasGradeThreshold, hasStrandAlignment, hasProgramCode
Run:
  python3 build_ontology.py
Output:
  lspu_lb_ontology.owl
"""

import json
import os
from owlready2 import (
    get_ontology, Thing, ObjectProperty, DataProperty,
    AllDisjoint, types
)

# ── paths ──────────────────────────────────────────────────────────────────────
PROGRAM_DIR = "/home/claude/programs"
OUTPUT_PATH = "/home/claude/lspu_lb_ontology.owl"
BASE_IRI    = "http://lspu-lb.edu.ph/ocrag/ontology#"

# ── load all program JSONs ─────────────────────────────────────────────────────
programs = []
for fname in sorted(os.listdir(PROGRAM_DIR)):
    if fname.endswith(".json"):
        with open(os.path.join(PROGRAM_DIR, fname)) as f:
            programs.append(json.load(f))

print(f"Loaded {len(programs)} program JSONs: {[p['code'] for p in programs]}")

# ── create ontology ────────────────────────────────────────────────────────────
onto = get_ontology(BASE_IRI)

with onto:

    # ── Classes ────────────────────────────────────────────────────────────────
    class HollandType(Thing):
        """One of the six Holland RIASEC personality-environment types."""

    class AcademicProgram(Thing):
        """An academic degree program offered at LSPU-LB."""

    class CompetencyDomain(Thing):
        """A competency area that a program requires or develops."""

    class CareerOutcome(Thing):
        """A career pathway or outcome that a program leads to."""

    # Make the four main classes disjoint
    AllDisjoint([HollandType, AcademicProgram, CompetencyDomain, CareerOutcome])

    # ── Object Properties ──────────────────────────────────────────────────────
    class alignsWithHollandType(ObjectProperty):
        """Links an AcademicProgram to its compatible Holland RIASEC type(s)."""
        domain  = [AcademicProgram]
        range   = [HollandType]

    class requiresCompetency(ObjectProperty):
        """Links an AcademicProgram to the competency domains it requires."""
        domain  = [AcademicProgram]
        range   = [CompetencyDomain]

    class leadsToCareer(ObjectProperty):
        """Links an AcademicProgram to its career outcome categories."""
        domain  = [AcademicProgram]
        range   = [CareerOutcome]

    # ── Data Properties ────────────────────────────────────────────────────────
    class hasProgramCode(DataProperty):
        """Short code identifier for the program, e.g. BSPSY."""
        domain  = [AcademicProgram]
        range   = [str]

    class hasGradeThreshold(DataProperty):
        """Minimum GWA required for admission to this program."""
        domain  = [AcademicProgram]
        range   = [float]

    class hasStrandAlignment(DataProperty):
        """Recommended SHS strand(s) for this program."""
        domain  = [AcademicProgram]
        range   = [str]

    class hasCollegeName(DataProperty):
        """Full name of the college offering this program."""
        domain  = [AcademicProgram]
        range   = [str]

    # ── Holland RIASEC type individuals ───────────────────────────────────────
    RIASEC_TYPES = [
        "Realistic",
        "Investigative",
        "Artistic",
        "Social",
        "Enterprising",
        "Conventional"
    ]

    holland_individuals = {}
    for ht in RIASEC_TYPES:
        ind = HollandType(ht.replace(" ", "_"))
        ind.label = [ht]
        holland_individuals[ht] = ind
        print(f"  Created HollandType: {ht}")

    # Make all 6 Holland types disjoint from each other
    AllDisjoint(list(holland_individuals.values()))

    # ── Competency domain individuals ──────────────────────────────────────────
    # Collect all unique competency domains across all programs
    all_competencies = set()
    for p in programs:
        for c in p.get("competency_domains", []):
            all_competencies.add(c)

    competency_individuals = {}
    for comp in sorted(all_competencies):
        safe_name = comp.replace(" ", "_").replace("/", "_").replace("-", "_")
        ind = CompetencyDomain(safe_name)
        ind.label = [comp]
        competency_individuals[comp] = ind
    print(f"  Created {len(competency_individuals)} CompetencyDomain individuals")

    # ── Career outcome individuals ─────────────────────────────────────────────
    all_outcomes = set()
    for p in programs:
        for o in p.get("career_outcome_categories", []):
            all_outcomes.add(o)

    career_individuals = {}
    for outcome in sorted(all_outcomes):
        safe_name = outcome.replace(" ", "_").replace("/", "_").replace("-", "_")
        ind = CareerOutcome(safe_name)
        ind.label = [outcome]
        career_individuals[outcome] = ind
    print(f"  Created {len(career_individuals)} CareerOutcome individuals")

    # ── Academic program individuals + property assertions ─────────────────────
    for p in programs:
        code = p["code"]
        safe_code = code.replace(" ", "_")

        prog = AcademicProgram(safe_code)
        prog.label          = [p["name"]]
        prog.hasProgramCode = [code]
        prog.hasGradeThreshold  = [float(p.get("grade_threshold", 80))]
        prog.hasStrandAlignment = [p.get("strand_alignment", "")]
        prog.hasCollegeName     = [p.get("college", "")]

        # alignsWithHollandType — primary AND secondary
        for ht in p.get("holland_types", []):
            if ht in holland_individuals:
                prog.alignsWithHollandType.append(holland_individuals[ht])

        # requiresCompetency
        for comp in p.get("competency_domains", []):
            if comp in competency_individuals:
                prog.requiresCompetency.append(competency_individuals[comp])

        # leadsToCareer
        for outcome in p.get("career_outcome_categories", []):
            if outcome in career_individuals:
                prog.leadsToCareer.append(career_individuals[outcome])

        print(f"  Created AcademicProgram: {code} "
              f"| Holland: {p.get('holland_types')} "
              f"| Competencies: {len(p.get('competency_domains', []))} "
              f"| Outcomes: {len(p.get('career_outcome_categories', []))}")

# ── save ───────────────────────────────────────────────────────────────────────
onto.save(file=OUTPUT_PATH, format="rdfxml")
print(f"\nOntology saved to: {OUTPUT_PATH}")

# ── quick verification ─────────────────────────────────────────────────────────
print("\n── Verification ──────────────────────────────────────────────────────")
verify_onto = get_ontology(f"file://{OUTPUT_PATH}").load()

programs_found   = list(verify_onto.AcademicProgram.instances())
holland_found    = list(verify_onto.HollandType.instances())
competency_found = list(verify_onto.CompetencyDomain.instances())
career_found     = list(verify_onto.CareerOutcome.instances())

print(f"AcademicProgram instances : {len(programs_found)}")
print(f"HollandType instances     : {len(holland_found)}")
print(f"CompetencyDomain instances: {len(competency_found)}")
print(f"CareerOutcome instances   : {len(career_found)}")

print("\n── Sample traversal: Social Holland type → programs ──────────────────")
social = verify_onto.search_one(iri=f"*Social")
if social:
    matched = [
        p for p in verify_onto.AcademicProgram.instances()
        if social in p.alignsWithHollandType
    ]
    print(f"Programs aligning with Social: {[p.hasProgramCode[0] for p in matched]}")

print("\n── Sample traversal: Counseling competency → programs ────────────────")
counseling = verify_onto.search_one(label="Counseling")
if counseling:
    matched = [
        p for p in verify_onto.AcademicProgram.instances()
        if counseling in p.requiresCompetency
    ]
    print(f"Programs requiring Counseling: {[p.hasProgramCode[0] for p in matched]}")

print("\n── Grade thresholds ──────────────────────────────────────────────────")
for p in sorted(programs_found, key=lambda x: x.hasProgramCode[0]):
    print(f"  {p.hasProgramCode[0]:10s}  GWA threshold: {p.hasGradeThreshold[0]}")

print("\nDone.")
