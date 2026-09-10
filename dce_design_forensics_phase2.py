#!/usr/bin/env python3
"""
DCE Design Forensics - Phase 2: Deep reconstruction with respondent-facing values.

This script:
1. Reconstructs the TRUE DCE design from data (distinguishing 12 vs 84 profiles)
2. Uses respondent-facing values (not analysis codes) for prompts
3. Identifies why initial audit found 84 profiles vs manuscript claim of 12

Key question: Is this a data problem or extraction bug?
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

WORKSPACE = Path(__file__).resolve().parent
DCE_FILE = WORKSPACE / "analysis_output" / "dce_encoded.csv"

# ============================================================================
# RESPONDENT-FACING VALUE MAPPING (from Table S3)
# ============================================================================

def wait_display(wait_coded: float) -> str:
    """Wait time: coded {0,1,3,6} → display {walk-in/0, 1, 3, 6 months}"""
    mapping = {0.0: "walk-in / 0 months", 1.0: "1 month", 3.0: "3 months", 6.0: "6 months"}
    return mapping.get(wait_coded, f"{wait_coded} months")

def efficacy_display(eff_coded: float) -> str:
    """Efficacy: coded {0.5, 0.7, 0.95} → display {50%, 70%, 95%}"""
    mapping = {0.5: "50%", 0.7: "70%", 0.95: "95%", 0.0: "0%"}
    return mapping.get(eff_coded, f"{eff_coded*100:.0f}%")

def side_effects_display(se_coded: float) -> str:
    """Side effects: coded {1,2,3} → display {10%, 20%, 30%}"""
    # Coded level 1 = 10%, 2 = 20%, 3 = 30%
    mapping = {0.0: "0%", 1.0: "10%", 2.0: "20%", 3.0: "30%"}
    return mapping.get(se_coded, f"{se_coded*10:.0f}%")

def origin_display(origin_coded: int) -> str:
    """Origin: coded {0,1} → display {Domestic, Imported}"""
    mapping = {0: "Domestic", 1: "Imported"}
    return mapping.get(origin_coded, "Unknown")

def cash_display(cash_coded: float) -> str:
    """Cash: display as RMB"""
    return f"{int(cash_coded)} RMB"

# ============================================================================
# DATA LOADING
# ============================================================================

def load_dce_data(path: Path) -> list[dict]:
    """Load raw DCE data."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def parse_row(row: dict) -> dict:
    """Parse DCE row with ALL attributes."""
    result = {}
    
    # IDs
    result["respondent_id"] = str(row.get("RespondentID", "")).strip()
    result["task_id"] = str(row.get("Choiceset", "")).strip()
    result["alt"] = row.get("Alt", "").strip().upper()
    result["chid"] = str(row.get("chid", "")).strip()
    
    # Choice outcome
    try:
        result["choice"] = int(float(row.get("Choice", 0)))
    except:
        result["choice"] = 0
    
    # Five attributes (raw coded values)
    result["wait"] = float(row.get("WaitTime", 0))
    result["eff"] = float(row.get("VaccineEfficacy", 0))
    result["se"] = float(row.get("SideEffects", 0))
    result["cash"] = float(row.get("CashIncentives", 0))
    result["origin"] = int(float(row.get("VaccineOrigin", 0)))
    
    # Respondent-facing display values
    result["wait_display"] = wait_display(result["wait"])
    result["eff_display"] = efficacy_display(result["eff"])
    result["se_display"] = side_effects_display(result["se"])
    result["cash_display"] = cash_display(result["cash"])
    result["origin_display"] = origin_display(result["origin"])
    
    return result

def remove_wait_anomaly(rows: list[dict]) -> list[dict]:
    """Remove wait=2 anomalous rows."""
    filtered = []
    for row in rows:
        parsed = parse_row(row)
        if parsed["wait"] != 2.0:
            filtered.append(row)
    return filtered

# ============================================================================
# DESIGN RECONSTRUCTION
# ============================================================================

def analyze_design_structure(rows: list[dict]) -> dict:
    """
    Deep analysis of DCE design structure.
    Key question: Is it 12 profiles / 1 block or 84 profiles / multiple blocks?
    """
    print("=" * 80)
    print("DESIGN STRUCTURE RECONSTRUCTION")
    print("=" * 80)
    print()
    
    # Group by respondent
    respondents = defaultdict(lambda: {"tasks": defaultdict(lambda: {"A": None, "B": None, "C": None})})
    
    for row in rows:
        parsed = parse_row(row)
        resp_id = parsed["respondent_id"]
        task_id = parsed["task_id"]
        alt = parsed["alt"]
        
        if alt in ["A", "B", "C"]:
            respondents[resp_id]["tasks"][task_id][alt] = parsed
    
    print(f"Total respondents: {len(respondents)}")
    
    # Check task completeness per respondent
    complete_respondents = 0
    incomplete_respondents = 0
    task_counts = []
    
    for resp_id, data in respondents.items():
        tasks = data["tasks"]
        complete_tasks = 0
        for task_id, alts in tasks.items():
            if alts["A"] and alts["B"] and alts["C"]:
                complete_tasks += 1
        task_counts.append(len(tasks))
        if complete_tasks == 6:
            complete_respondents += 1
        else:
            incomplete_respondents += 1
    
    print(f"Respondents with 6 complete tasks: {complete_respondents}")
    print(f"Respondents with incomplete tasks: {incomplete_respondents}")
    print(f"Task count distribution: {Counter(task_counts)}")
    print()
    
    # Reconstruct blocks
    blocks = defaultdict(list)
    
    for resp_id, data in respondents.items():
        tasks = data["tasks"]
        
        # Build block signature: sorted tuple of task signatures
        task_sigs = []
        for task_id in sorted(tasks.keys(), key=lambda x: int(x) if x.isdigit() else 999):
            alts = tasks[task_id]
            if alts["A"] and alts["B"]:
                # Task signature: (A_profile, B_profile)
                a_sig = (alts["A"]["wait"], alts["A"]["eff"], alts["A"]["se"], 
                        alts["A"]["cash"], alts["A"]["origin"])
                b_sig = (alts["B"]["wait"], alts["B"]["eff"], alts["B"]["se"],
                        alts["B"]["cash"], alts["B"]["origin"])
                task_sig = (a_sig, b_sig)
                task_sigs.append(task_sig)
        
        block_sig = tuple(task_sigs)
        blocks[block_sig].append(resp_id)
    
    print(f"Unique design blocks: {len(blocks)}")
    print()
    
    # Analyze blocks
    for i, (block_sig, resp_ids) in enumerate(sorted(blocks.items(), key=lambda x: -len(x[1])), 1):
        print(f"Block {i}: {len(resp_ids)} respondents, {len(block_sig)} tasks")
        if len(block_sig) == 6:
            print("  Tasks:")
            for j, (a_sig, b_sig) in enumerate(block_sig, 1):
                print(f"    Task {j}: A(wait={a_sig[0]:.0f}) vs B(wait={b_sig[0]:.0f})")
        print()
    
    return {
        "total_respondents": len(respondents),
        "complete_respondents": complete_respondents,
        "unique_blocks": len(blocks),
        "blocks": blocks
    }

def analyze_unique_profiles(rows: list[dict]) -> dict:
    """
    Analyze unique profiles with respondent-facing values.
    """
    print("=" * 80)
    print("UNIQUE PROFILE ANALYSIS (Respondent-Facing Values)")
    print("=" * 80)
    print()
    
    profiles = defaultdict(list)
    
    for row in rows:
        parsed = parse_row(row)
        if parsed["alt"] == "C":  # Skip opt-out
            continue
        
        # Profile signature with coded values
        coded_sig = (parsed["wait"], parsed["eff"], parsed["se"], 
                    parsed["cash"], parsed["origin"])
        
        # Profile signature with display values
        display_sig = (parsed["wait_display"], parsed["eff_display"], 
                      parsed["se_display"], parsed["cash_display"], 
                      parsed["origin_display"])
        
        profiles[coded_sig].append({
            "parsed": parsed,
            "display": display_sig
        })
    
    print(f"Total unique non-opt-out profiles: {len(profiles)}")
    print()
    
    # Check if all profiles appear in all respondents
    all_respondents = set()
    for rows_list in profiles.values():
        for item in rows_list:
            all_respondents.add(item["parsed"]["respondent_id"])
    
    print(f"Total unique respondents: {len(all_respondents)}")
    
    # Find universal profiles (appear in all respondents)
    universal_profiles = []
    for sig, rows_list in profiles.items():
        resp_ids = set(item["parsed"]["respondent_id"] for item in rows_list)
        if len(resp_ids) == len(all_respondents):
            universal_profiles.append(sig)
    
    print(f"Profiles appearing in ALL respondents: {len(universal_profiles)}")
    print()
    
    # Show all unique profiles with display values
    print("All unique profiles (coded → display):")
    print("-" * 80)
    
    for i, (coded_sig, rows_list) in enumerate(sorted(profiles.items()), 1):
        display_sig = rows_list[0]["display"]
        resp_count = len(set(item["parsed"]["respondent_id"] for item in rows_list))
        row_count = len(rows_list)
        
        print(f"\nProfile {i}:")
        print(f"  Coded:   wait={coded_sig[0]:.0f}, eff={coded_sig[1]:.2f}, se={coded_sig[2]:.1f}, "
              f"cash={coded_sig[3]:.0f}, origin={coded_sig[4]}")
        print(f"  Display: wait={display_sig[0]}, eff={display_sig[1]}, se={display_sig[2]}, "
              f"cash={display_sig[3]}, origin={display_sig[4]}")
        print(f"  Frequency: {row_count} rows from {resp_count} respondents")
    
    print()
    
    return {
        "unique_profiles": len(profiles),
        "universal_profiles": len(universal_profiles),
        "profiles": profiles
    }

def generate_respondent_facing_prompts(rows: list[dict]) -> None:
    """
    Generate example prompts with respondent-facing values.
    """
    print("=" * 80)
    print("RESPONDENT-FACING PROMPT EXAMPLES")
    print("=" * 80)
    print()
    
    # Find a complete choice task
    respondents = defaultdict(lambda: {"tasks": defaultdict(lambda: {"A": None, "B": None, "C": None})})
    
    for row in rows:
        parsed = parse_row(row)
        resp_id = parsed["respondent_id"]
        task_id = parsed["task_id"]
        alt = parsed["alt"]
        
        if alt in ["A", "B", "C"]:
            respondents[resp_id]["tasks"][task_id][alt] = parsed
    
    # Find first complete task
    for resp_id, data in respondents.items():
        for task_id, alts in data["tasks"].items():
            if alts["A"] and alts["B"] and alts["C"]:
                a = alts["A"]
                b = alts["B"]
                
                print("=" * 80)
                print("EXAMPLE: Task-level multinomial prompt (respondent-facing)")
                print("=" * 80)
                print()
                print("You are evaluating vaccination choices.")
                print()
                print("Choice task:")
                print()
                print("Alternative A:")
                print(f"- Waiting time: {a['wait_display']}")
                print(f"- Vaccine effectiveness: {a['eff_display']}")
                print(f"- Risk of side effects: {a['se_display']}")
                print(f"- Cash incentive: {a['cash_display']}")
                print(f"- Vaccine origin: {a['origin_display']}")
                print()
                print("Alternative B:")
                print(f"- Waiting time: {b['wait_display']}")
                print(f"- Vaccine effectiveness: {b['eff_display']}")
                print(f"- Risk of side effects: {b['se_display']}")
                print(f"- Cash incentive: {b['cash_display']}")
                print(f"- Vaccine origin: {b['origin_display']}")
                print()
                print("Alternative C:")
                print("- Do not receive the vaccine / Opt out")
                print()
                print("Which would you choose?")
                print()
                print("Respond exactly in this format:")
                print("Choice: A or B or C")
                print("P(A): probability between 0 and 100")
                print("P(B): probability between 0 and 100")
                print("P(C): probability between 0 and 100")
                print("Reasoning: one short sentence")
                print()
                print("Note: P(A) + P(B) + P(C) should equal 100.")
                print()
                
                print("=" * 80)
                print("EXAMPLE: Profile-level binary prompt (respondent-facing)")
                print("=" * 80)
                print()
                print("You are evaluating a vaccination decision.")
                print()
                print("Vaccine profile:")
                print(f"- Waiting time: {a['wait_display']}")
                print(f"- Vaccine effectiveness: {a['eff_display']}")
                print(f"- Risk of side effects: {a['se_display']}")
                print(f"- Cash incentive: {a['cash_display']}")
                print(f"- Vaccine origin: {a['origin_display']}")
                print()
                print("Would you get vaccinated now under this profile (vs. opting out)?")
                print()
                print("Respond exactly in this format:")
                print("Decision: Yes or No")
                print("Probability: a number between 0 and 100")
                print("Reasoning: one short sentence")
                print()
                return

def main():
    print("=" * 80)
    print("DCE DESIGN FORENSICS - PHASE 2")
    print("Deep reconstruction with respondent-facing values")
    print("=" * 80)
    print()
    
    # Load data
    raw_rows = load_dce_data(DCE_FILE)
    print(f"Total rows: {len(raw_rows)}")
    
    rows = remove_wait_anomaly(raw_rows)
    print(f"After removing wait=2: {len(rows)}")
    print()
    
    # Run analyses
    design = analyze_design_structure(rows)
    profiles = analyze_unique_profiles(rows)
    generate_respondent_facing_prompts(rows)
    
    # Summary
    print("=" * 80)
    print("SUMMARY: MANUSCRIPT CLAIMS vs DATA RECONSTRUCTION")
    print("=" * 80)
    print()
    
    print("Manuscript claims:")
    print("  - Unique blocks: 1")
    print("  - Unique tasks: 6 (same for all respondents)")
    print("  - Unique non-opt-out profiles: 12")
    print()
    
    print("Data reconstruction finds:")
    print(f"  - Unique blocks: {design['unique_blocks']}")
    print(f"  - Complete respondents: {design['complete_respondents']}")
    print(f"  - Unique profiles (coded): {profiles['unique_profiles']}")
    print(f"  - Universal profiles (in all respondents): {profiles['universal_profiles']}")
    print()
    
    # Diagnosis
    print("=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    print()
    
    if design['unique_blocks'] == 1 and profiles['universal_profiles'] == 12:
        print("✓ Data CONSISTENT with manuscript claims")
        print("  → The 84-profile finding was an extraction bug")
        print("  → True design: 12 profiles, 1 block, 6 tasks")
    else:
        print("✗ Data INCONSISTENT with manuscript claims")
        print(f"  → Found {design['unique_blocks']} blocks, not 1")
        print(f"  → Found {profiles['universal_profiles']} universal profiles, not 12")
        print(f"  → Found {profiles['unique_profiles']} total unique profiles")
        print()
        print("Possible explanations:")
        print("  1. Data includes multiple experimental waves/blocks")
        print("  2. Manuscript description is incorrect")
        print("  3. Data preprocessing changed the design")
    
    print()
    print("=" * 80)
    print("FORENSICS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
