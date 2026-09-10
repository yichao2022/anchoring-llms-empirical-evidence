#!/usr/bin/env python3
"""
DCE Design Forensics - Reconstruct true experimental design from analytic data.

This script performs deep forensics on the DCE data to resolve the discrepancy:
- Manuscript claims: 1 unique block, 6 tasks, 12 non-opt-out profiles
- Initial audit found: 84 unique profiles

Goal: Identify exactly why the counts differ.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

WORKSPACE = Path(__file__).resolve().parent
DCE_FILE = WORKSPACE / "analysis_output" / "dce_encoded.csv"

def load_dce_data(path: Path) -> list[dict]:
    """Load raw DCE data."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def parse_row(row: dict) -> dict:
    """Parse a DCE row with attribute normalization."""
    result = {}
    
    # Respondent ID
    result["respondent_id"] = row.get("RespondentID", row.get("respondent_id", "")).strip()
    
    # Task ID
    result["task_id"] = str(row.get("Choiceset", row.get("choiceset", ""))).strip()
    
    # Alternative label (A/B/C)
    result["alt"] = row.get("Alt", row.get("alt", "")).strip().upper()
    
    # Choice outcome
    try:
        result["choice"] = int(float(row.get("Choice", row.get("choice", 0))))
    except:
        result["choice"] = 0
    
    # Five administered attributes with normalization
    
    # Wait time
    try:
        result["wait"] = float(row.get("WaitTime", row.get("wait", 0)))
    except:
        result["wait"] = float('nan')
    
    # Efficacy - check for percentage vs proportion
    eff_raw = row.get("VaccineEfficacy", row.get("eff", row.get("Vaccine Efficacy", "0")))
    try:
        eff_val = float(eff_raw)
        # Normalize: if > 1, assume percentage (50) and divide by 100; else proportion (0.5)
        if eff_val > 1:
            result["eff"] = eff_val / 100.0
        else:
            result["eff"] = eff_val
    except:
        result["eff"] = float('nan')
    
    # Side effects - check for scale
    se_raw = row.get("SideEffects", row.get("se", row.get("Side Effects", "0")))
    try:
        se_val = float(se_raw)
        # Could be 1/2/3 or 10/20/30
        result["se"] = se_val
    except:
        result["se"] = float('nan')
    
    # Cash incentive
    try:
        result["cash"] = float(row.get("CashIncentives", row.get("cash", row.get("Cash Incentives", 0))))
    except:
        result["cash"] = float('nan')
    
    # Origin - could be 0/1 or strings
    origin_raw = row.get("VaccineOrigin", row.get("origin", row.get("Vaccine Origin", "0")))
    try:
        origin_val = float(origin_raw)
        result["origin"] = int(origin_val)
    except:
        # String value - normalize
        origin_str = str(origin_raw).strip().lower()
        if origin_str in ['0', 'reference', 'ref', 'domestic']:
            result["origin"] = 0
        elif origin_str in ['1', 'non-reference', 'nonref', 'imported', 'foreign']:
            result["origin"] = 1
        else:
            result["origin"] = -1  # Unknown
    
    return result

def remove_wait_anomaly(rows: list[dict]) -> list[dict]:
    """Remove wait=2 anomalous rows."""
    filtered = []
    removed = 0
    for row in rows:
        parsed = parse_row(row)
        if parsed["wait"] == 2:
            removed += 1
            continue
        filtered.append(row)
    return filtered, removed

def analyze_profile_normalization(rows: list[dict]) -> dict:
    """
    Analyze attribute normalization issues.
    Check for different encodings of the same semantic value.
    """
    print("=" * 80)
    print("5. ATTRIBUTE NORMALIZATION ANALYSIS")
    print("=" * 80)
    print()
    
    raw_values = {
        "eff": [],
        "se": [],
        "origin": [],
    }
    
    for row in rows:
        parsed = parse_row(row)
        if parsed["alt"] != "C":  # Non-opt-out
            raw_values["eff"].append((parsed["eff"], row.get("VaccineEfficacy", row.get("eff", ""))))
            raw_values["se"].append((parsed["se"], row.get("SideEffects", row.get("se", ""))))
            raw_values["origin"].append((parsed["origin"], row.get("VaccineOrigin", row.get("origin", ""))))
    
    # Efficacy analysis
    print("Efficacy encoding check:")
    eff_unique = set(v[0] for v in raw_values["eff"])
    print(f"  Unique normalized values: {sorted(eff_unique)}")
    eff_raw_examples = {}
    for norm, raw in raw_values["eff"]:
        if norm not in eff_raw_examples:
            eff_raw_examples[norm] = set()
        eff_raw_examples[norm].add(str(raw))
    for norm in sorted(eff_raw_examples.keys()):
        print(f"    {norm}: raw examples = {eff_raw_examples[norm]}")
    print()
    
    # Side effects analysis
    print("Side effects encoding check:")
    se_unique = set(v[0] for v in raw_values["se"])
    print(f"  Unique values: {sorted(se_unique)}")
    se_raw_examples = {}
    for norm, raw in raw_values["se"]:
        if norm not in se_raw_examples:
            se_raw_examples[norm] = set()
        se_raw_examples[norm].add(str(raw))
    for se in sorted(se_raw_examples.keys()):
        print(f"    {se}: raw examples = {se_raw_examples[se]}")
    print()
    
    # Origin analysis
    print("Origin encoding check:")
    origin_unique = set(v[0] for v in raw_values["origin"])
    print(f"  Unique values: {sorted(origin_unique)}")
    origin_raw_examples = {}
    for norm, raw in raw_values["origin"]:
        if norm not in origin_raw_examples:
            origin_raw_examples[norm] = set()
        origin_raw_examples[norm].add(str(raw))
    for orig in sorted(origin_raw_examples.keys()):
        print(f"    {orig}: raw examples = {origin_raw_examples[orig]}")
    print()
    
    return raw_values

def analyze_unique_profiles(rows: list[dict]) -> dict:
    """
    1. Define profile signature using ONLY the five administered DCE attributes.
    Report: total unique non-opt-out profile signatures, frequency, respondents per profile.
    """
    print("=" * 80)
    print("1. UNIQUE PROFILE SIGNATURES (5 attributes)")
    print("=" * 80)
    print()
    
    # Profile signature: (wait, eff, se, cash, origin)
    profile_rows = defaultdict(list)
    
    for row in rows:
        parsed = parse_row(row)
        if parsed["alt"] == "C":  # Skip opt-out
            continue
        
        sig = (
            parsed["wait"],
            parsed["eff"],
            parsed["se"],
            parsed["cash"],
            parsed["origin"]
        )
        profile_rows[sig].append(parsed)
    
    print(f"Total unique non-opt-out profile signatures: {len(profile_rows)}")
    print()
    
    # Sort by frequency
    sorted_profiles = sorted(profile_rows.items(), key=lambda x: len(x[1]), reverse=True)
    
    print("Top 20 most frequent profiles:")
    print("-" * 80)
    print(f"{'Rank':<6} {'Wait':<6} {'Eff':<6} {'SE':<6} {'Cash':<8} {'Origin':<8} {'Count':<8} {'Unique Resp':<12}")
    print("-" * 80)
    
    for i, (sig, rows_list) in enumerate(sorted_profiles[:20], 1):
        wait, eff, se, cash, origin = sig
        respondents = set(r["respondent_id"] for r in rows_list)
        print(f"{i:<6} {wait:<6.1f} {eff:<6.2f} {se:<6.1f} {cash:<8.0f} {origin:<8} {len(rows_list):<8} {len(respondents):<12}")
    
    print()
    
    # Check if any profile appears in all respondents (should be 12 if design is balanced)
    all_respondents = set(parse_row(r)["respondent_id"] for r in rows if parse_row(r)["respondent_id"])
    print(f"Total unique respondents: {len(all_respondents)}")
    
    universal_profiles = []
    for sig, rows_list in sorted_profiles:
        respondents = set(r["respondent_id"] for r in rows_list)
        if len(respondents) == len(all_respondents):
            universal_profiles.append(sig)
    
    print(f"Profiles appearing in ALL respondents: {len(universal_profiles)}")
    print()
    
    return {
        "unique_profiles": len(profile_rows),
        "profile_data": profile_rows,
        "universal_profiles": universal_profiles
    }

def analyze_choice_tasks(rows: list[dict]) -> dict:
    """
    2. Reconstruct each respondent's six actual choice sets.
    Define task_sig = (canonicalized_A_profile, canonicalized_B_profile, "OPT_OUT")
    Report: unique ordered task signatures, unique unordered A/B signatures, frequencies.
    """
    print("=" * 80)
    print("2. CHOICE TASK SIGNATURES")
    print("=" * 80)
    print()
    
    # Group by respondent and task
    respondent_tasks = defaultdict(lambda: defaultdict(dict))
    
    for row in rows:
        parsed = parse_row(row)
        resp_id = parsed["respondent_id"]
        task_id = parsed["task_id"]
        alt = parsed["alt"]
        
        if alt in ["A", "B", "C"]:
            respondent_tasks[resp_id][task_id][alt] = parsed
    
    # Build task signatures
    ordered_tasks = []
    unordered_tasks = []
    
    for resp_id, tasks in respondent_tasks.items():
        for task_id, alts in tasks.items():
            if "A" in alts and "B" in alts and "C" in alts:
                a = alts["A"]
                b = alts["B"]
                
                # Ordered signature (A then B)
                ordered_sig = (
                    (a["wait"], a["eff"], a["se"], a["cash"], a["origin"]),
                    (b["wait"], b["eff"], b["se"], b["cash"], b["origin"]),
                    "OPT_OUT"
                )
                ordered_tasks.append((ordered_sig, resp_id, task_id))
                
                # Unordered signature (A and B sorted)
                profiles = [
                    (a["wait"], a["eff"], a["se"], a["cash"], a["origin"]),
                    (b["wait"], b["eff"], b["se"], b["cash"], b["origin"])
                ]
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
    
    # Show top ordered tasks
    print("Top 10 ordered task signatures (A profile, B profile, OPT_OUT):")
    print("-" * 80)
    for sig, count in unique_ordered.most_common(10):
        a_prof, b_prof, _ = sig
        print(f"Count={count:5d}")
        print(f"  A: wait={a_prof[0]:.1f}, eff={a_prof[1]:.2f}, se={a_prof[2]:.1f}, cash={a_prof[3]:.0f}, origin={a_prof[4]}")
        print(f"  B: wait={b_prof[0]:.1f}, eff={b_prof[1]:.2f}, se={b_prof[2]:.1f}, cash={b_prof[3]:.0f}, origin={b_prof[4]}")
        print()
    
    return {
        "ordered_tasks": ordered_tasks,
        "unordered_tasks": unordered_tasks,
        "unique_ordered": len(unique_ordered),
        "unique_unordered": len(unique_unordered),
        "unique_ordered_details": unique_ordered,
        "unique_unordered_details": unique_unordered
    }

def analyze_design_blocks(rows: list[dict], task_analysis: dict) -> dict:
    """
    3. Reconstruct respondent-level design blocks.
    Define block_sig = ordered tuple of all six task signatures.
    Report: unique design blocks, respondents per block.
    """
    print("=" * 80)
    print("3. DESIGN BLOCKS")
    print("=" * 80)
    print()
    
    # Group tasks by respondent
    respondent_tasks = defaultdict(list)
    for sig, resp_id, task_id in task_analysis["ordered_tasks"]:
        respondent_tasks[resp_id].append((task_id, sig))
    
    # Build block signatures (sorted by task_id to ensure order)
    blocks = defaultdict(list)
    
    for resp_id, tasks in respondent_tasks.items():
        # Sort by task_id to ensure consistent ordering
        tasks_sorted = sorted(tasks, key=lambda x: x[0])
        block_sig = tuple(sig for _, sig in tasks_sorted)
        blocks[block_sig].append(resp_id)
    
    print(f"Total respondents: {len(respondent_tasks)}")
    print(f"Unique design blocks: {len(blocks)}")
    print()
    
    # Show block sizes
    print("Block size distribution:")
    block_sizes = Counter(len(resps) for resps in blocks.values())
    for size, count in sorted(block_sizes.items()):
        print(f"  {count} blocks with {size} respondents")
    print()
    
    # Check if single universal block
    if len(blocks) == 1:
        print("✓ All respondents share the same design block")
    else:
        print("✗ Multiple design blocks detected!")
        print("\nBlock details:")
        for i, (block_sig, respondents) in enumerate(blocks.items(), 1):
            print(f"\nBlock {i}: {len(respondents)} respondents")
            print(f"  Tasks in block: {len(block_sig)}")
            for j, task_sig in enumerate(block_sig[:3], 1):  # Show first 3 tasks
                a_prof, b_prof, _ = task_sig
                print(f"    Task {j}: A(wait={a_prof[0]:.0f}) vs B(wait={b_prof[0]:.0f})")
            if len(block_sig) > 3:
                print(f"    ... and {len(block_sig)-3} more tasks")
    
    return {
        "unique_blocks": len(blocks),
        "blocks": blocks,
        "is_single_block": len(blocks) == 1
    }

def analyze_by_task_id(rows: list[dict]) -> dict:
    """
    4. Cross-tab by nominal task ID.
    For each task ID 1-6 report: distinct A profiles, distinct B profiles, distinct A/B signatures.
    """
    print("=" * 80)
    print("4. CROSS-TAB BY NOMINAL TASK ID")
    print("=" * 80)
    print()
    
    # Group by task_id
    task_profiles = defaultdict(lambda: {"A": [], "B": []})
    
    for row in rows:
        parsed = parse_row(row)
        if parsed["alt"] in ["A", "B"]:
            task_id = parsed["task_id"]
            alt = parsed["alt"]
            prof = (parsed["wait"], parsed["eff"], parsed["se"], parsed["cash"], parsed["origin"])
            task_profiles[task_id][alt].append(prof)
    
    print(f"{'Task ID':<10} {'Distinct A':<12} {'Distinct B':<12} {'A/B Pairs':<12} {'Notes'}")
    print("-" * 80)
    
    for task_id in sorted(task_profiles.keys(), key=lambda x: int(x) if x.isdigit() else 999):
        a_profiles = set(task_profiles[task_id]["A"])
        b_profiles = set(task_profiles[task_id]["B"])
        
        # Count unique A/B pairs
        pairs = set()
        for a in task_profiles[task_id]["A"]:
            for b in task_profiles[task_id]["B"]:
                pairs.add((a, b))
        
        notes = ""
        if len(a_profiles) > 1 or len(b_profiles) > 1:
            notes = "MULTIPLE VARIANTS"
        
        print(f"{task_id:<10} {len(a_profiles):<12} {len(b_profiles):<12} {len(pairs):<12} {notes}")
    
    print()
    
    # Check if task IDs are consistent across respondents
    all_task_ids = set()
    for row in rows:
        parsed = parse_row(row)
        all_task_ids.add(parsed["task_id"])
    
    print(f"Task IDs found in data: {sorted(all_task_ids)}")
    print()
    
    return task_profiles

def main():
    print("=" * 80)
    print("DCE DESIGN FORENSICS REPORT")
    print("=" * 80)
    print()
    print(f"Data file: {DCE_FILE}")
    print()
    
    # Load data
    raw_rows = load_dce_data(DCE_FILE)
    print(f"Total rows in DCE file: {len(raw_rows)}")
    
    # Remove wait=2 anomaly
    rows, removed = remove_wait_anomaly(raw_rows)
    print(f"Rows after removing wait=2 anomaly: {len(rows)} (removed {removed})")
    print()
    
    # Run all analyses
    normalization = analyze_profile_normalization(rows)
    profile_analysis = analyze_unique_profiles(rows)
    task_analysis = analyze_choice_tasks(rows)
    block_analysis = analyze_design_blocks(rows, task_analysis)
    task_id_analysis = analyze_by_task_id(rows)
    
    # Summary
    print("=" * 80)
    print("SUMMARY: MANUSCRIPT CLAIMS vs DATA RECONSTRUCTION")
    print("=" * 80)
    print()
    
    print("Manuscript claims:")
    print("  - Unique design blocks: 1")
    print("  - Unique task compositions: 6")
    print("  - Unique non-opt-out profiles: 12")
    print()
    
    print("Analytic-data reconstruction finds:")
    print(f"  - Unique design blocks: {block_analysis['unique_blocks']}")
    print(f"  - Unique ordered task compositions: {task_analysis['unique_ordered']}")
    print(f"  - Unique unordered task compositions: {task_analysis['unique_unordered']}")
    print(f"  - Unique normalized non-opt-out profiles: {profile_analysis['unique_profiles']}")
    print(f"  - Profiles appearing in ALL respondents: {len(profile_analysis['universal_profiles'])}")
    print()
    
    # Identify discrepancy
    print("=" * 80)
    print("DISCREPANCY ANALYSIS")
    print("=" * 80)
    print()
    
    if profile_analysis['unique_profiles'] != 12:
        print(f"⚠️  Profile count mismatch: expected 12, found {profile_analysis['unique_profiles']}")
        print()
        print("Possible explanations:")
        print("  1. Multiple experimental blocks (different respondents saw different profiles)")
        print("  2. Attribute encoding differences (e.g., 50 vs 0.50 for efficacy)")
        print("  3. Side effects scale differences (1/2/3 vs 10/20/30)")
        print("  4. Origin encoding (0/1 vs strings)")
        print("  5. Data includes multiple waves/batches")
        print()
    
    if not block_analysis['is_single_block']:
        print(f"⚠️  Multiple design blocks detected: {block_analysis['unique_blocks']} blocks")
        print("This contradicts the manuscript claim of '1 unique design block'")
        print()
    
    # Save detailed results
    output_dir = WORKSPACE / "forensics_output"
    output_dir.mkdir(exist_ok=True)
    
    # Save profile frequencies
    with open(output_dir / "profile_frequencies.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["wait", "eff", "se", "cash", "origin", "row_count", "respondent_count"])
        for sig, rows_list in profile_analysis["profile_data"].items():
            respondents = set(r["respondent_id"] for r in rows_list)
            writer.writerow([*sig, len(rows_list), len(respondents)])
    
    # Save task signatures
    with open(output_dir / "task_signatures.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_sig_hash", "A_wait", "A_eff", "A_se", "A_cash", "A_origin",
                        "B_wait", "B_eff", "B_se", "B_cash", "B_origin", "frequency"])
        for sig, count in task_analysis["unique_ordered_details"].most_common():
            a_prof, b_prof, _ = sig
            # Create hash for signature
            sig_hash = hash(sig) % 100000
            writer.writerow([sig_hash, *a_prof, *b_prof, count])
    
    print(f"Detailed forensics output saved to: {output_dir}/")
    print("  - profile_frequencies.csv")
    print("  - task_signatures.csv")
    print()
    
    print("=" * 80)
    print("FORENSICS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
