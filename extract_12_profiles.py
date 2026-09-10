#!/usr/bin/env python3
"""
Extract 12 unique non-opt-out profiles from original design file.
Prepare 120 Qwen calls (12 profiles × 10 reps).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent
DESIGN_FILE = Path("/Users/cary/Documents/Discrete choice experiment task (1) 3_副本3/CLD_Vaccination_DCE_Design_NonEfficient.csv")
OUTPUT_CSV = WORKSPACE / "dce_12_profiles_clean.csv"

def load_design_file(path: Path) -> list[dict]:
    """Load original DCE design file."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)

def extract_profiles(rows: list[dict]) -> dict:
    """Extract 12 unique non-opt-out profiles from design file."""
    profiles = {}
    
    for row in rows:
        option = row.get("Option", "").strip()
        if option not in ["A", "B"]:
            continue
        
        # Parse attributes from design file
        wait = row.get("AWE", "").strip()
        origin = row.get("VaccineOrigin", "").strip()
        efficacy = row.get("Efficacy", "").strip()
        side_effects = row.get("SideEffects", "").strip()
        cash = row.get("CashIncentive", "").strip()
        
        # Skip incomplete
        if not all([wait, origin, efficacy, side_effects, cash]):
            continue
        
        # Create profile signature
        prof_sig = (wait, origin, efficacy, side_effects, cash)
        
        if prof_sig not in profiles:
            profiles[prof_sig] = {
                "profile_id": len(profiles) + 1,
                "wait": wait,
                "wait_label": row.get("AWE_Label", ""),
                "origin": origin,
                "origin_label": row.get("VaccineOrigin_Label", ""),
                "efficacy": efficacy,
                "efficacy_label": row.get("Efficacy_Label", ""),
                "side_effects": side_effects,
                "side_effects_label": row.get("SideEffects_Label", ""),
                "cash": cash,
                "cash_label": row.get("CashIncentive_Label", ""),
                "occurrences": []
            }
        
        profiles[prof_sig]["occurrences"].append({
            "task": row.get("Task", ""),
            "option": option
        })
    
    return profiles

def build_prompt(profile: dict) -> str:
    """Build Qwen prompt with exact questionnaire wording."""
    return f"""You are evaluating vaccination choices for COVID-19 vaccines.

Please consider the following vaccine profile and indicate whether you would accept it:

Vaccine Profile:
- {profile['wait_label']}
- {profile['origin_label']}
- {profile['efficacy_label']}
- {profile['side_effects_label']}
- {profile['cash_label']}

Question: Would you choose to receive this vaccine? 
Please respond with ONLY "Yes" or "No".

Additionally, what is the probability (between 0 and 1) that you would accept this vaccine?"""

def main():
    print("=" * 80)
    print("EXTRACTING 12 UNIQUE PROFILES FROM DESIGN FILE")
    print("=" * 80)
    print()
    
    # Load design file
    rows = load_design_file(DESIGN_FILE)
    print(f"Loaded {len(rows)} rows from design file")
    print()
    
    # Extract profiles
    profiles = extract_profiles(rows)
    print(f"Extracted {len(profiles)} unique non-opt-out profiles")
    print()
    
    # Build output records
    records = []
    for prof_sig, data in sorted(profiles.items(), key=lambda x: x[1]['profile_id']):
        record = {
            "profile_id": data['profile_id'],
            "wait": data['wait'],
            "wait_label": data['wait_label'],
            "origin": data['origin'],
            "origin_label": data['origin_label'],
            "efficacy": data['efficacy'],
            "efficacy_label": data['efficacy_label'],
            "side_effects": data['side_effects'],
            "side_effects_label": data['side_effects_label'],
            "cash": data['cash'],
            "cash_label": data['cash_label'],
            "n_occurrences_in_design": len(data['occurrences']),
            "prompt_text": build_prompt(data),
            "expected_repeats": 10,
            "planned_api_calls": 10,
            "status": "ready_for_qwen"
        }
        records.append(record)
        
        print(f"Profile {data['profile_id']}:")
        print(f"  Wait: {data['wait_label']}")
        print(f"  Origin: {data['origin_label']}")
        print(f"  Efficacy: {data['efficacy_label']}")
        print(f"  Side Effects: {data['side_effects_label']}")
        print(f"  Cash: {data['cash_label']}")
        print(f"  Occurs in {len(data['occurrences'])} task-options")
        print()
    
    # Export to CSV
    fieldnames = [
        "profile_id", "wait", "wait_label", "origin", "origin_label",
        "efficacy", "efficacy_label", "side_effects", "side_effects_label",
        "cash", "cash_label", "n_occurrences_in_design", "prompt_text",
        "expected_repeats", "planned_api_calls", "status"
    ]
    
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total unique profiles: {len(records)}")
    print(f"Total planned API calls: {len(records) * 10}")
    print(f"Output: {OUTPUT_CSV}")
    print()
    print("Next: Execute 120 Qwen calls (12 × 10)")
    print("Then: Map to held-out 205 respondents × 6 tasks × 2 alternatives = 2,460 rows")

if __name__ == "__main__":
    main()
