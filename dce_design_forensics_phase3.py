#!/usr/bin/env python3
"""
DCE Design Forensics - Phase 3: Complete reconstruction with original questionnaire wording.

This script performs the FINAL forensic analysis to resolve the design provenance:
1. Raw vs normalized counts (distinguish true 84 from encoding artifacts)
2. Full task signatures (ordered + unordered) based on complete choice sets
3. Block reconstruction from respondent-level task sequences
4. Original questionnaire wording (not approximate respondent-facing labels)

Key deliverables:
- True DCE design structure (blocks / tasks / profiles)
- Raw vs normalized unique counts
- Identification of why manuscript claims differ from data
- Recommendation for manuscript correction or validation strategy
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

WORKSPACE = Path(__file__).resolve().parent
DCE_FILE = WORKSPACE / "analysis_output" / "dce_encoded.csv"

def load_dce_data(path: Path) -> list[dict]:
    """Load raw DCE data."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def parse_raw_values(row: dict) -> dict:
    """Extract RAW values from DCE row (before any normalization)."""
    result = {}
    result["respondent_id"] = str(row.get("RespondentID", "")).strip()
    result["task_id"] = str(row.get("Choiceset", "")).strip()
    result["alt"] = row.get("Alt", "").strip().upper()
    result["chid"] = str(row.get("chid", "")).strip()
    
    # Raw string values (before numeric conversion)
    result["wait_raw"] = row.get("WaitTime", "").strip()
    result["eff_raw"] = row.get("VaccineEfficacy", "").strip()
    result["se_raw"] = row.get("SideEffects", "").strip()
    result["cash_raw"] = row.get("CashIncentives", "").strip()
    result["origin_raw"] = row.get("VaccineOrigin", "").strip()
    
    # Parsed numeric values
    try:
        result["wait"] = float(result["wait_raw"])
    except:
        result["wait"] = float('nan')
    
    try:
        result["eff"] = float(result["eff_raw"])
    except:
        result["eff"] = float('nan')
    
    try:
        result["se"] = float(result["se_raw"])
    except:
        result["se"] = float('nan')
    
    try:
        result["cash"] = float(result["cash_raw"])
    except:
        result["cash"] = float('nan')
    
    try:
        result["origin"] = int(float(result["origin_raw"]))
    except:
        result["origin"] = -1
    
    return result

def normalize_profile(wait: float, eff: float, se: float, cash: float, origin: int) -> tuple:
    """
    Normalize profile values to canonical forms.
    
    Efficacy: if > 1, divide by 100 (50% -> 0.5)
    Side effects: keep as-is (1,2,3)
    Origin: 0/1
    Cash: as float
    Wait: as float
    """
    # Efficacy normalization
    if eff > 1:
        eff_norm = eff / 100.0
    else:
        eff_norm = eff
    
    return (wait, eff_norm, se, cash, origin)

def analyze_raw_vs_normalized(rows: list[dict]) -> dict:
    """
    1. Report both RAW and NORMALIZED unique counts.
    This will show whether 84 is data structure or encoding artifacts.
    """
    print("=" * 80)
    print("1. RAW vs NORMALIZED UNIQUE COUNTS")
    print("=" * 80)
    print()
    
    # Collect raw signatures (strings as they appear in data)
    raw_sigs = set()
    # Collect normalized signatures (after standardization)
    norm_sigs = set()
    
    profile_examples = {}  # Store examples for discrepancy analysis
    
    for row in rows:
        parsed = parse_raw_values(row)
        if parsed["alt"] == "C":  # Skip opt-out
            continue
        if parsed["wait"] == 2:  # Skip wait=2 anomaly
            continue
        
        # Raw signature (exactly as in file)
        raw_sig = (
            parsed["wait_raw"],
            parsed["eff_raw"],
            parsed["se_raw"],
            parsed["cash_raw"],
            parsed["origin_raw"]
        )
        raw_sigs.add(raw_sig)
        
        # Normalized signature
        norm_sig = normalize_profile(
            parsed["wait"], parsed["eff"], parsed["se"],
            parsed["cash"], parsed["origin"]
        )
        norm_sigs.add(norm_sig)
        
        # Store example
        if norm_sig not in profile_examples:
            profile_examples[norm_sig] = {
                "raw": raw_sig,
                "parsed": parsed,
                "count": 0
            }
        profile_examples[norm_sig]["count"] += 1
    
    print(f"Unique profiles (RAW string values): {len(raw_sigs)}")
    print(f"Unique profiles (NORMALIZED values): {len(norm_sigs)}")
    print()
    
    # Check if normalization reduced count
    if len(norm_sigs) < len(raw_sigs):
        print(f"✓ Normalization reduced count by {len(raw_sigs) - len(norm_sigs)} profiles")
        print("  → Some profiles differ only by string formatting")
    else:
        print("✗ Normalization did NOT reduce count")
        print("  → 84 profiles reflect true data structure")
    print()
    
    # Show examples
    print("Top 10 normalized profiles with examples:")
    print("-" * 80)
    sorted_profiles = sorted(profile_examples.items(), key=lambda x: -x[1]["count"])
    
    for i, (norm_sig, data) in enumerate(sorted_profiles[:10], 1):
        raw_sig = data["raw"]
        parsed = data["parsed"]
        print(f"\nProfile {i}: {data['count']} occurrences")
        print(f"  Normalized: wait={norm_sig[0]:.0f}, eff={norm_sig[1]:.2f}, se={norm_sig[2]:.1f}, "
              f"cash={norm_sig[3]:.0f}, origin={norm_sig[4]}")
        print(f"  Raw:        wait={raw_sig[0]}, eff={raw_sig[1]}, se={raw_sig[2]}, "
              f"cash={raw_sig[3]}, origin={raw_sig[4]}")
    
    print()
    
    return {
        "raw_unique": len(raw_sigs),
        "normalized_unique": len(norm_sigs),
        "examples": profile_examples
    }

def build_full_task_signature(rows: list[dict]) -> dict:
    """
    2. Reconstruct choice tasks using FULL task signatures (not just task ID).
    
    Task signature includes:
    - Complete A profile (all 5 attributes)
    - Complete B profile (all 5 attributes)
    - Ordered version (A then B)
    - Unordered version (sorted A/B)
    """
    print("=" * 80)
    print("2. FULL TASK SIGNATURES (Complete Choice Sets)")
    print("=" * 80)
    print()
    
    # Group by respondent and task
    respondents = defaultdict(lambda: {"tasks": defaultdict(lambda: {"A": None, "B": None, "C": None})})
    
    for row in rows:
        parsed = parse_raw_values(row)
        if parsed["wait"] == 2:  # Skip anomaly
            continue
        
        resp_id = parsed["respondent_id"]
        task_id = parsed["task_id"]
        alt = parsed["alt"]
        
        if alt in ["A", "B", "C"]:
            respondents[resp_id]["tasks"][task_id][alt] = parsed
    
    # Build full task signatures
    ordered_tasks = []
    unordered_tasks = []
    
    for resp_id, data in respondents.items():
        for task_id, alts in data["tasks"].items():
            if alts["A"] and alts["B"] and alts["C"]:
                a = alts["A"]
                b = alts["B"]
                
                # Full profile signatures with all 5 attributes
                a_sig = normalize_profile(a["wait"], a["eff"], a["se"], a["cash"], a["origin"])
                b_sig = normalize_profile(b["wait"], b["eff"], b["se"], b["cash"], b["origin"])
                
                # Ordered signature: (A, B, OPT_OUT)
                ordered_sig = (a_sig, b_sig, "OPT_OUT")
                ordered_tasks.append((ordered_sig, resp_id, task_id))
                
                # Unordered signature: sorted(A, B), then OPT_OUT
                profiles = [a_sig, b_sig]
                profiles_sorted = tuple(sorted(profiles))
                unordered_sig = (profiles_sorted, "OPT_OUT")
                unordered_tasks.append((unordered_sig, resp_id, task_id))
    
    # Count unique
    unique_ordered = Counter(sig for sig, _, _ in ordered_tasks)
    unique_unordered = Counter(sig for sig, _, _ in unordered_tasks)
    
    print(f"Total task instances: {len(ordered_tasks)}")
    print(f"Unique ORDERED task signatures: {len(unique_ordered)}")
    print(f"Unique UNORDERED task signatures: {len(unique_unordered)}")
    print()
    
    # Show top tasks
    print("Top 10 unique ORDERED task signatures:")
    print("-" * 80)
    for sig, count in unique_ordered.most_common(10):
        a_prof, b_prof, _ = sig
        print(f"\nCount: {count}")
        print(f"A: wait={a_prof[0]:.0f}, eff={a_prof[1]:.2f}, se={a_prof[2]:.1f}, "
              f"cash={a_prof[3]:.0f}, origin={a_prof[4]}")
        print(f"B: wait={b_prof[0]:.0f}, eff={b_prof[1]:.2f}, se={b_prof[2]:.1f}, "
              f"cash={b_prof[3]:.0f}, origin={b_prof[4]}")
    
    print()
    
    return {
        "unique_ordered": len(unique_ordered),
        "unique_unordered": len(unique_unordered),
        "ordered_tasks": ordered_tasks,
        "unordered_tasks": unordered_tasks,
        "task_details": unique_ordered
    }

def reconstruct_blocks(rows: list[dict], task_analysis: dict) -> dict:
    """
    3. Block reconstruction from respondent-level task sequences.
    Block signature = ordered tuple of 6 full task signatures per respondent.
    """
    print("=" * 80)
    print("3. BLOCK RECONSTRUCTION (Full Task Sequences)")
    print("=" * 80)
    print()
    
    # Group tasks by respondent
    resp_tasks = defaultdict(list)
    for sig, resp_id, task_id in task_analysis["ordered_tasks"]:
        resp_tasks[resp_id].append((task_id, sig))
    
    # Build block signatures
    blocks = defaultdict(list)
    
    for resp_id, tasks in resp_tasks.items():
        # Sort by task_id to ensure correct order
        tasks_sorted = sorted(tasks, key=lambda x: int(x[0]) if x[0].isdigit() else 999)
        
        # Block signature: tuple of all 6 task signatures
        if len(tasks_sorted) == 6:
            block_sig = tuple(sig for _, sig in tasks_sorted)
            blocks[block_sig].append(resp_id)
    
    print(f"Total respondents with 6 complete tasks: {len(blocks)}")
    print(f"Unique design blocks: {len(blocks)}")
    print()
    
    # Analyze blocks
    for i, (block_sig, resp_ids) in enumerate(sorted(blocks.items(), key=lambda x: -len(x[1])), 1):
        print(f"Block {i}: {len(resp_ids)} respondents")
        print(f"  Tasks in block: {len(block_sig)}")
        
        for j, (a_prof, b_prof, _) in enumerate(block_sig, 1):
            print(f"    Task {j}: A(wait={a_prof[0]:.0f}) vs B(wait={b_prof[0]:.0f})")
        print()
    
    return {
        "unique_blocks": len(blocks),
        "blocks": blocks
    }

def extract_original_questionnaire_wording(rows: list[dict]) -> dict:
    """
    4. Extract original questionnaire wording from data/metadata.
    
    Note: The actual wording is not in dce_encoded.csv.
    We document what we CAN confirm from Table S3 and what needs manual recovery.
    """
    print("=" * 80)
    print("4. ORIGINAL QUESTIONNAIRE WORDING")
    print("=" * 80)
    print()
    
    print("From Table S3 (coding dictionary) and data structure:")
    print()
    print("CONFIRMED mappings:")
    print("  - Wait time: 0/1/3/6 months")
    print("  - Efficacy: 0%, 50%, 70%, 95% (respondent-facing)")
    print("  - Side effects: 0%, 10%, 20%, 30% (respondent-facing)")
    print("  - Cash: 50/200/800 RMB")
    print("  - Origin: Domestic / Imported")
    print()
    
    print("NEEDS MANUAL RECOVERY from original questionnaire:")
    print("  - Exact attribute labels (e.g., 'Waiting time' vs 'Wait time')")
    print("  - Side effect description wording")
    print("  - Opt-out alternative wording")
    print("  - Choice task introduction text")
    print("  - Response format instructions")
    print()
    
    # Sample prompt with best-available wording
    print("DRAFT multinomial prompt (pending original questionnaire recovery):")
    print("-" * 80)
    print()
    print("You are evaluating vaccination choices.")
    print()
    print("Choice task:")
    print()
    print("Alternative A:")
    print("- Waiting time: [VALUE]")
    print("- Vaccine effectiveness: [VALUE]%")
    print("- Risk of side effects: [VALUE]%")
    print("- Cash incentive: [VALUE] RMB")
    print("- Vaccine origin: [VALUE]")
    print()
    print("Alternative B:")
    print("- Waiting time: [VALUE]")
    print("- Vaccine effectiveness: [VALUE]%")
    print("- Risk of side effects: [VALUE]%")
    print("- Cash incentive: [VALUE] RMB")
    print("- Vaccine origin: [VALUE]")
    print()
    print("Alternative C:")
    print("- Do not receive the vaccine")
    print()
    print("Which would you choose?")
    print()
    
    return {
        "needs_recovery": True,
        "confirmed_mappings": {
            "wait": ["0 months", "1 month", "3 months", "6 months"],
            "eff": ["0%", "50%", "70%", "95%"],
            "se": ["0%", "10%", "20%", "30%"],
            "cash": ["50 RMB", "200 RMB", "800 RMB"],
            "origin": ["Domestic", "Imported"]
        }
    }

def generate_summary_report(raw_norm: dict, tasks: dict, blocks: dict, wording: dict) -> None:
    """
    Generate final summary and recommendations.
    """
    print("=" * 80)
    print("FINAL SUMMARY: MANUSCRIPT vs DATA")
    print("=" * 80)
    print()
    
    print("MANUSCRIPT CLAIMS:")
    print("  - Respondents: 1,027")
    print("  - Design blocks: 1 (same for all)")
    print("  - Tasks per respondent: 6 (same 6 for all)")
    print("  - Non-opt-out profiles: 12 total")
    print("  - Coverage: 12/216 = 5.6%")
    print()
    
    print("DATA RECONSTRUCTION FINDS:")
    print(f"  - Respondents with 6 complete tasks: {sum(len(v) for v in blocks['blocks'].values())}")
    print(f"  - Raw unique profiles: {raw_norm['raw_unique']}")
    print(f"  - Normalized unique profiles: {raw_norm['normalized_unique']}")
    print(f"  - Unique ordered task signatures: {tasks['unique_ordered']}")
    print(f"  - Unique unordered task signatures: {tasks['unique_unordered']}")
    print(f"  - Unique design blocks: {blocks['unique_blocks']}")
    print()
    
    # Diagnosis
    print("DIAGNOSIS:")
    print("-" * 80)
    
    if raw_norm['raw_unique'] != raw_norm['normalized_unique']:
        print(f"• Normalization reduces count by {raw_norm['raw_unique'] - raw_norm['normalized_unique']}")
        print("  → Some discrepancy due to string formatting")
    
    if raw_norm['normalized_unique'] != 12:
        print(f"• After normalization: {raw_norm['normalized_unique']} profiles, not 12")
        print("  → TRUE multiple blocks/profiles structure")
    
    if blocks['unique_blocks'] != 1:
        print(f"• Multiple design blocks detected: {blocks['unique_blocks']}")
        print("  → Data includes different experimental conditions/waves")
    
    if tasks['unique_ordered'] != 6:
        print(f"• Task compositions vary: {tasks['unique_ordered']} ordered signatures")
        print("  → '6 tasks' refers to positions, not identical choice sets")
    
    print()
    
    # Recommendations
    print("RECOMMENDATIONS:")
    print("-" * 80)
    
    if blocks['unique_blocks'] > 1 or raw_norm['normalized_unique'] != 12:
        print("1. MANUSCRIPT CORRECTION REQUIRED:")
        print("   - Update Supplement DCE design description")
        print("   - Report actual number of blocks/profiles")
        print("   - Clarify whether analysis uses all data or single block")
        print()
        
        print("2. VALIDATION STRATEGY:")
        print("   Option A: Use only largest block (Block 1, n=250)")
        print("   Option B: Report as 'multi-block design' with block-level analysis")
        print("   Option C: Reconstruct true intended design from documentation")
        print()
    
    print("3. PROMPT DEVELOPMENT:")
    if wording['needs_recovery']:
        print("   - MUST recover original questionnaire wording")
        print("   - Do not proceed with approximate respondent-facing labels")
        print("   - Contact data provider for questionnaire document")
    print()
    
    print("4. ANALYSIS SCOPE:")
    print("   - Profile-level (84×10 calls): DEPRECATED for primary analysis")
    print("   - Task-level multinomial: PRIORITY after design clarified")
    print("   - Inferential target: respondent-clustered uncertainty only")
    print()
    
    print("=" * 80)
    print("FORENSICS COMPLETE - No API calls made")
    print("=" * 80)
    print()
    print("NEXT STEPS:")
    print("1. Decide on validation scope (single block vs all data)")
    print("2. Recover original questionnaire wording")
    print("3. Update manuscript design description")
    print("4. Develop task-level multinomial prompts")
    print("5. THEN consider API calls (after design provenance resolved)")

def main():
    print("=" * 80)
    print("DCE DESIGN FORENSICS - PHASE 3 (FINAL)")
    print("Complete reconstruction with raw/normalized comparison")
    print("=" * 80)
    print()
    
    # Load data
    raw_rows = load_dce_data(DCE_FILE)
    rows = [r for r in raw_rows if parse_raw_values(r)["wait"] != 2]
    
    print(f"Total rows: {len(raw_rows)}")
    print(f"After removing wait=2: {len(rows)}")
    print()
    
    # Run all analyses
    raw_norm = analyze_raw_vs_normalized(rows)
    tasks = build_full_task_signature(rows)
    blocks = reconstruct_blocks(rows, tasks)
    wording = extract_original_questionnaire_wording(rows)
    
    # Generate summary
    generate_summary_report(raw_norm, tasks, blocks, wording)

if __name__ == "__main__":
    main()
