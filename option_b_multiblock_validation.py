#!/usr/bin/env python3
"""
Option B Execution: Multi-block DCE validation with 42 unique tasks.

Tasks:
1. Audit all code/manuscript dependencies on one-block assumption
2. Prepare 42 unique task multinomial prompts (respondent-facing)
3. Dry-run mapping to held-out respondents
4. Draft data provider provenance query

No API calls until design provenance verified.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

WORKSPACE = Path(__file__).resolve().parent
DCE_FILE = WORKSPACE / "analysis_output" / "dce_encoded.csv"

def load_dce_data(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def normalize_profile(wait, eff, se, cash, origin):
    """Normalize to canonical form."""
    if eff > 1:
        eff = eff / 100.0
    return (float(wait), float(eff), float(se), float(cash), int(origin))

def respondent_facing_label(attr: str, value: Any) -> str:
    """
    Convert analysis-coded value to respondent-facing label.
    Based on Table S3 coding dictionary.
    """
    if attr == "wait":
        if value == 0:
            return "walk-in / 0 months"
        elif value == 1:
            return "1 month"
        elif value == 3:
            return "3 months"
        elif value == 6:
            return "6 months"
        else:
            return f"{value} months"
    
    elif attr == "eff":
        # Table S3: 0.50 → 50%
        eff_val = value * 100 if value <= 1 else value
        return f"{int(eff_val)}%"
    
    elif attr == "se":
        # Table S3: level 1 → 10%, level 2 → 20%, level 3 → 30%
        se_map = {0: "0%", 1: "10%", 2: "20%", 3: "30%"}
        return se_map.get(int(value), f"{int(value)}%")
    
    elif attr == "cash":
        return f"{int(value)} RMB"
    
    elif attr == "origin":
        # Table S3: 0 = Domestic, 1 = Imported
        return "Domestic" if int(value) == 0 else "Imported"
    
    return str(value)

def audit_one_block_dependencies():
    """
    1. Audit all quantities and code paths that depend on one-block assumption.
    """
    print("=" * 80)
    print("1. AUDIT: ONE-BLOCK ASSUMPTION DEPENDENCIES")
    print("=" * 80)
    print()
    
    dependencies = {
        "Manuscript Claims (Supplement)": [
            "Line 118: 'All 1,027 respondents completed the same six choice tasks'",
            "Line 118: '1 unique design block shared by all 1027 respondents'",
            "Line 118: '12 administered non-opt-out profiles'",
            "Line 118: '12/216 = 5.6% coverage'",
        ],
        "Held-out Validation (78 tasks)": [
            "Table S15: '78 tasks under 11-of-12-cell canonical match'",
            "Table S15: 3-alt full subset is EMPTY (canonical match)",
            "Table S15: Only 2-alt subset valid (78 tasks, 6.3% coverage)",
            "Root cause: wait ∈ {0,6}, eff ∈ {0.5,0.7}, se ∈ {1,2,3} match only 78/1236 tasks",
        ],
        "Analysis Code": [
            "heldout_dce_validation.py: Assumes 6 same tasks for all respondents",
            "table1_qwen72b_static.py: Uses 64-state grid lookup (cash=0, origin=0)",
            "run_qwen72b_rule_prompted.py: Grid-based validation only",
        ],
        "Current Validation Stats": [
            "Held-out respondents: 205",
            "Held-out tasks: 205 × 6 = 1,230 (theoretical)",
            "Actually matched: 78 (6.3%) under 11-of-12-cell rule",
            "Unmatched: 1,152 tasks (cash/origin mismatch)",
        ]
    }
    
    for category, items in dependencies.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  • {item}")
    
    print()
    print("IMPACT OF SEVEN-BLOCK DISCOVERY:")
    print("-" * 80)
    print("  • 'Same six choice tasks' = FALSE")
    print("  • '1 unique design block' = FALSE (actually 7 blocks)")
    print("  • '12 administered profiles' = MISLEADING (actually 84 across all blocks)")
    print("  • '5.6% coverage' = ONLY applies if restricted to single block")
    print()
    print("  → All manuscript claims need correction")
    print("  → Held-out validation strategy needs revision")
    print("  → Code paths assume wrong design structure")
    print()
    
    return dependencies

def build_task_prompts(rows: list[dict]) -> dict:
    """
    2. Build 42 unique full-task multinomial prompts with respondent-facing labels.
    """
    print("=" * 80)
    print("2. BUILD 42 UNIQUE TASK PROMPTS (Respondent-Facing)")
    print("=" * 80)
    print()
    
    # Build unique tasks
    tasks = defaultdict(lambda: {"respondents": [], "count": 0, "block": None})
    
    # Group by respondent
    resp_data = defaultdict(lambda: {"tasks": {}})
    
    for row in rows:
        resp_id = str(row.get("RespondentID", "")).strip()
        task_id = str(row.get("Choiceset", "")).strip()
        alt = row.get("Alt", "").strip().upper()
        
        # Skip wait=2 anomaly
        try:
            wait = float(row.get("WaitTime", 0))
        except:
            continue
        if wait == 2:
            continue
        
        if alt not in ["A", "B", "C"]:
            continue
        
        # Parse profile
        try:
            eff = float(row.get("VaccineEfficacy", 0))
            se = float(row.get("SideEffects", 0))
            cash = float(row.get("CashIncentives", 0))
            origin = int(float(row.get("VaccineOrigin", 0)))
        except:
            continue
        
        prof = normalize_profile(wait, eff, se, cash, origin)
        
        if resp_id not in resp_data:
            resp_data[resp_id] = {"tasks": {}}
        
        if task_id not in resp_data[resp_id]["tasks"]:
            resp_data[resp_id]["tasks"][task_id] = {}
        
        resp_data[resp_id]["tasks"][task_id][alt] = prof
    
    # Build unique task signatures
    task_sigs = {}
    
    for resp_id, data in resp_data.items():
        for task_id, alts in data["tasks"].items():
            if "A" in alts and "B" in alts and "C" in alts:
                a_prof = alts["A"]
                b_prof = alts["B"]
                
                # Task signature: (A, B, OPT_OUT)
                task_sig = (a_prof, b_prof, "OPT_OUT")
                
                if task_sig not in task_sigs:
                    task_sigs[task_sig] = {
                        "task_id": len(task_sigs) + 1,
                        "respondents": [],
                        "count": 0
                    }
                
                task_sigs[task_sig]["respondents"].append(resp_id)
                task_sigs[task_sig]["count"] += 1
    
    print(f"Total unique task signatures: {len(task_sigs)}")
    print()
    
    # Generate respondent-facing prompts
    prompts = []
    
    for i, (task_sig, data) in enumerate(sorted(task_sigs.items(), key=lambda x: -x[1]["count"]), 1):
        a_prof, b_prof, _ = task_sig
        
        # Build respondent-facing prompt
        prompt = {
            "task_number": i,
            "frequency": data["count"],
            "num_respondents": len(set(data["respondents"])),
            "alternative_a": {
                "waiting_time": respondent_facing_label("wait", a_prof[0]),
                "vaccine_effectiveness": respondent_facing_label("eff", a_prof[1]),
                "risk_of_side_effects": respondent_facing_label("se", a_prof[2]),
                "cash_incentive": respondent_facing_label("cash", a_prof[3]),
                "vaccine_origin": respondent_facing_label("origin", a_prof[4]),
            },
            "alternative_b": {
                "waiting_time": respondent_facing_label("wait", b_prof[0]),
                "vaccine_effectiveness": respondent_facing_label("eff", b_prof[1]),
                "risk_of_side_effects": respondent_facing_label("se", b_prof[2]),
                "cash_incentive": respondent_facing_label("cash", b_prof[3]),
                "vaccine_origin": respondent_facing_label("origin", b_prof[4]),
            },
            "alternative_c": "Do not receive the vaccine",
            "coded_a": {"wait": a_prof[0], "eff": a_prof[1], "se": a_prof[2], "cash": a_prof[3], "origin": a_prof[4]},
            "coded_b": {"wait": b_prof[0], "eff": b_prof[1], "se": b_prof[2], "cash": b_prof[3], "origin": b_prof[4]},
        }
        
        prompts.append(prompt)
    
    # Display sample
    print("Top 5 task prompts (respondent-facing):")
    print("-" * 80)
    
    for prompt in prompts[:5]:
        print(f"\nTask {prompt['task_number']} ({prompt['num_respondents']} respondents):")
        print(f"Alternative A:")
        for k, v in prompt["alternative_a"].items():
            print(f"  - {k.replace('_', ' ').title()}: {v}")
        print(f"Alternative B:")
        for k, v in prompt["alternative_b"].items():
            print(f"  - {k.replace('_', ' ').title()}: {v}")
        print(f"Alternative C: {prompt['alternative_c']}")
    
    print()
    
    return {
        "num_unique_tasks": len(prompts),
        "prompts": prompts,
        "total_task_instances": sum(p["frequency"] for p in prompts)
    }

def dry_run_heldout_mapping(rows: list[dict], task_data: dict) -> dict:
    """
    3. Dry-run mapping of 42 unique tasks to held-out respondents.
    """
    print("=" * 80)
    print("3. DRY-RUN: 42 TASKS → HELD-OUT RESPONDENT MAPPING")
    print("=" * 80)
    print()
    
    # Apply 80/20 split (seed=2026)
    import random
    random.seed(2026)
    
    # Get all respondent IDs
    all_resps = set()
    for row in rows:
        resp_id = str(row.get("RespondentID", "")).strip()
        if resp_id:
            try:
                wait = float(row.get("WaitTime", 0))
                if wait != 2:  # Exclude anomaly
                    all_resps.add(resp_id)
            except:
                pass
    
    all_resps = sorted(list(all_resps))
    n_total = len(all_resps)
    
    # 80/20 split
    n_test = int(n_total * 0.2)
    test_resps = set(random.sample(all_resps, n_test))
    train_resps = set(all_resps) - test_resps
    
    print(f"Total respondents: {n_total}")
    print(f"Training (80%): {len(train_resps)}")
    print(f"Held-out (20%): {len(test_resps)}")
    print()
    
    # Map tasks to held-out respondents
    heldout_tasks = {}
    
    for row in rows:
        resp_id = str(row.get("RespondentID", "")).strip()
        if resp_id not in test_resps:
            continue
        
        task_id = str(row.get("Choiceset", "")).strip()
        alt = row.get("Alt", "").strip().upper()
        
        try:
            wait = float(row.get("WaitTime", 0))
            if wait == 2:
                continue
            eff = float(row.get("VaccineEfficacy", 0))
            se = float(row.get("SideEffects", 0))
            cash = float(row.get("CashIncentives", 0))
            origin = int(float(row.get("VaccineOrigin", 0)))
            prof = normalize_profile(wait, eff, se, cash, origin)
        except:
            continue
        
        key = (resp_id, task_id)
        if key not in heldout_tasks:
            heldout_tasks[key] = {"resp": resp_id, "task": task_id, "profiles": {}}
        
        if alt in ["A", "B", "C"]:
            heldout_tasks[key]["profiles"][alt] = prof
    
    # Count valid held-out tasks
    valid_tasks = [t for t in heldout_tasks.values() if len(t.get("profiles", {})) == 3]
    
    print(f"Held-out respondent-tasks: {len(heldout_tasks)}")
    print(f"Valid complete tasks (A+B+C): {len(valid_tasks)}")
    print()
    
    # Count unique task signatures in held-out
    heldout_sigs = set()
    for t in valid_tasks:
        a = t["profiles"].get("A")
        b = t["profiles"].get("B")
        if a and b:
            sig = (a, b, "OPT_OUT")
            heldout_sigs.add(sig)
    
    print(f"Unique task signatures in held-out: {len(heldout_sigs)}")
    print(f"Expected LLM API calls (42 tasks × R=10): {42 * 10} = 420 calls")
    print()
    
    # Coverage comparison
    print("COVERAGE COMPARISON:")
    print("-" * 80)
    print("  Old approach (11-of-12-cell match):")
    print("    • Tasks evaluated: 78 / 1,230 = 6.3%")
    print("    • Coverage limited by cash/origin mismatch")
    print()
    print("  New approach (42 unique full-task):")
    print("    • Tasks evaluated: ~1,230 / 1,230 = 100%")
    print("    • All held-out tasks have LLM probabilities")
    print("    • No cash/origin mismatch (full 5 attributes)")
    print()
    
    return {
        "n_train": len(train_resps),
        "n_test": len(test_resps),
        "valid_heldout_tasks": len(valid_tasks),
        "unique_heldout_sigs": len(heldout_sigs),
        "api_calls_estimate": 42 * 10
    }

def draft_provenance_query():
    """
    4. Draft provenance query for data provider.
    """
    print("=" * 80)
    print("4. DATA PROVIDER PROVENANCE QUERY (DRAFT)")
    print("=" * 80)
    print()
    
    query = """
Subject: DCE Design Provenance Query - Wuhan Vaccination Study

Dear [Data Provider],

We are conducting a validation analysis of the Wuhan vaccination-timing DCE 
(N=1,027 respondents) and have discovered a discrepancy between the manuscript 
description and the reconstructed data structure.

MANUSCRIPT CLAIM:
- All 1,027 respondents completed the same 6 choice tasks
- 1 unique design block shared by all respondents
- 12 administered non-opt-out profiles per respondent

DATA RECONSTRUCTION FINDS:
- 7 distinct design blocks (not 1)
- 42 unique task signatures (not 6)
- 84 unique non-opt-out profiles (not 12)
- Respondents distributed across blocks: 250, 241, 222, 95, 83, 80, 55

QUESTIONS:

1. DESIGN INTENT
   Was the seven-block structure intentional? 
   Were different respondent groups administered different choice tasks?

2. ORIGINAL DESIGN MATRIX
   Can you provide:
   - The intended experimental design (blocks × tasks × profiles)
   - Original questionnaire versions if multiple were used
   - Any documentation of block assignment criteria

3. QUESTIONNAIRE WORDING
   Can you provide the exact wording used for:
   - Attribute labels (e.g., "Waiting time" vs "Wait time")
   - Side-effect descriptions
   - Opt-out alternative wording
   - Choice task introduction text
   - Response format instructions

4. ANALYTIC RECOMMENDATION
   Given this structure, should validation:
   - Use all 7 blocks (n=1,027, 42 unique tasks, 84 profiles)?
   - Restrict to largest block (Block 1, n=250, 6 tasks, ~12 profiles)?
   - Treat as separate strata for block-level analysis?

The current analysis has identified this as a potential design-provenance 
issue affecting validation strategy and manuscript accuracy. We want to ensure 
our reporting correctly reflects the administered experimental design.

Thank you for your assistance.

Best regards,
[Name]
"""
    
    print(query)
    print()
    
    return query

def generate_execution_summary(audit: dict, tasks: dict, heldout: dict, query: str):
    """
    Generate final execution summary.
    """
    print("=" * 80)
    print("OPTION B EXECUTION SUMMARY")
    print("=" * 80)
    print()
    
    print("PROVISIONAL ANALYTIC TRUTH (Option B):")
    print("-" * 80)
    print(f"  • Design blocks: 7 (not 1)")
    print(f"  • Unique choice-set signatures: {tasks['num_unique_tasks']} (not 6)")
    print(f"  • Unique non-opt-out profiles: 84 (not 12)")
    print(f"  • Total respondents: 1,027 (across all blocks)")
    print()
    
    print("HELD-OUT VALIDATION (NEW):")
    print("-" * 80)
    print(f"  • Training respondents: {heldout['n_train']}")
    print(f"  • Held-out respondents: {heldout['n_test']}")
    print(f"  • Valid complete tasks: {heldout['valid_heldout_tasks']}")
    print(f"  • Unique task signatures in held-out: {heldout['unique_heldout_sigs']}")
    print(f"  • Estimated API calls: {heldout['api_calls_estimate']} (42 tasks × 10 reps)")
    print()
    
    print("NEXT STEPS (BLOCKED):")
    print("-" * 80)
    print("  ☐ Send provenance query to data provider")
    print("  ☐ Await confirmation of seven-block structure")
    print("  ☐ Obtain original questionnaire wording")
    print("  ☐ Update manuscript design description (after confirmation)")
    print("  ☐ Finalize task-level multinomial prompts")
    print("  ☐ Verify mapping before Qwen API calls")
    print()
    
    print("MANUSCRIPT CHANGES REQUIRED:")
    print("-" * 80)
    print("  Supplement Line 118: 'same six choice tasks' → 'seven blocks'")
    print("  Supplement Line 118: '1 unique design block' → '7 design blocks'")
    print("  Supplement Line 118: '12 administered profiles' → '84 profiles across blocks'")
    print("  Table S15: Update 78-task validation to reflect 42-task full coverage")
    print()
    
    print("NO API CALLS MADE")
    print("=" * 80)

def main():
    print("=" * 80)
    print("OPTION B: MULTI-BLOCK DCE VALIDATION")
    print("Seven blocks, 42 tasks, 84 profiles")
    print("=" * 80)
    print()
    
    # Load data
    rows = load_dce_data(DCE_FILE)
    rows_filtered = [r for r in rows if float(r.get("WaitTime", 0)) != 2]
    
    print(f"Data loaded: {len(rows)} rows, {len(rows_filtered)} after wait≠2 filter")
    print()
    
    # Execute all tasks
    audit = audit_one_block_dependencies()
    tasks = build_task_prompts(rows_filtered)
    heldout = dry_run_heldout_mapping(rows_filtered, tasks)
    query = draft_provenance_query()
    
    # Generate summary
    generate_execution_summary(audit, tasks, heldout, query)

if __name__ == "__main__":
    main()
