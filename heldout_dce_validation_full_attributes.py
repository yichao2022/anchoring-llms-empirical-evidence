#!/usr/bin/env python3
"""
Held-out DCE validation with full five attributes (NOT 64-state grid lookup).

This script performs two validation approaches:
A. Profile-level binary validation: 12 unique non-opt-out profiles
B. Task-level multinomial validation: 6 unique choice tasks

Key differences from legacy heldout_dce_validation.py:
- Uses full 5 DCE attributes (wait, eff, se, cash, origin) for LLM queries
- Does NOT lookup from cash=0/origin=0 64-state grid
- Generates fresh LLM predictions for real administered DCE profiles
- Task-level validation uses proper multinomial scoring (P_A + P_B + P_optout = 1)

API call estimate:
- Profile-level: 12 profiles × R=10 repeats = 120 calls
- Task-level: 6 tasks × R=10 repeats = 60 calls
Total: ~180 calls (not thousands)

Requirements:
- DASHSCOPE_API_KEY or local vLLM endpoint
- Original 64-state experiment used temperature=0.0, max_tokens=512
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expit

# Configuration
SEED = 2026
TRAIN_FRAC = 0.80
LAMBDA = 0.25  # Fixed for primary analysis
REPEATS = 10

WORKSPACE = Path(__file__).resolve().parent
DCE_FILE = WORKSPACE / "analysis_output" / "dce_encoded.csv"

# Output files
OUTPUT_PROFILES_RAW = WORKSPACE / "llm_raw_outputs_qwen72b_heldout_real_profiles.csv"
OUTPUT_PROFILES_PARSED = WORKSPACE / "llm_parsed_outputs_qwen72b_heldout_real_profiles.csv"
OUTPUT_TASKS_RAW = WORKSPACE / "llm_raw_outputs_qwen72b_heldout_tasks.csv"
OUTPUT_TASKS_PARSED = WORKSPACE / "llm_parsed_outputs_qwen72b_heldout_tasks.csv"

# ============================================================================
# PROMPT TEMPLATES (matched to original 64-state experiment)
# ============================================================================

SYSTEM_PROMPT = """You are evaluating a vaccination decision based on the provided information."""

USER_PROMPT_TEMPLATE_BINARY = """You are evaluating a vaccination decision.

Vaccine profile:
- Waiting time: {wait} months
- Vaccine effectiveness: {eff}
- Side effect risk: {se}
- Cash incentive: {cash} RMB
- Vaccine origin: {origin}

Would you get vaccinated now under this profile (vs. opting out)?

Respond exactly in this format:
Decision: Yes or No
Probability: a number between 0 and 100
Reasoning: one short sentence"""

USER_PROMPT_TEMPLATE_MULTINOMIAL = """You are evaluating vaccination choices.

Choice task:
Alternative A:
- Waiting time: {wait_a} months
- Vaccine effectiveness: {eff_a}
- Side effect risk: {se_a}
- Cash incentive: {cash_a} RMB
- Vaccine origin: {origin_a}

Alternative B:
- Waiting time: {wait_b} months
- Vaccine effectiveness: {eff_b}
- Side effect risk: {se_b}
- Cash incentive: {cash_b} RMB
- Vaccine origin: {origin_b}

Alternative C (Opt-out): No vaccination

Which would you choose?

Respond exactly in this format:
Choice: A or B or C
P(A): probability between 0 and 100
P(B): probability between 0 and 100
P(C): probability between 0 and 100
Reasoning: one short sentence

Note: P(A) + P(B) + P(C) should equal 100."""

# ============================================================================
# DATA LOADING AND DEDUPLICATION
# ============================================================================

def load_dce_data(path: Path) -> list[dict]:
    """Load DCE encoded data with all attributes."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def parse_dce_row(row: dict) -> dict:
    """Extract relevant fields from DCE row."""
    # Map column names
    result = {
        "respondent_id": row.get("RespondentID", row.get("respondent_id", "")),
        "choice": int(float(row.get("Choice", row.get("choice", 0)))),
        "wait": float(row.get("WaitTime", row.get("wait", 0))),
        "eff": float(row.get("VaccineEfficacy", row.get("eff", row.get("Vaccine Efficacy", 0)))),
        "se": float(row.get("SideEffects", row.get("se", row.get("Side Effects", 0)))),
        "cash": float(row.get("CashIncentives", row.get("cash", row.get("Cash Incentives", 0)))),
        "origin": int(float(row.get("VaccineOrigin", row.get("origin", row.get("Vaccine Origin", 0))))),
        "alt": row.get("Alt", row.get("alt", "")).strip(),
        "choiceset": row.get("Choiceset", row.get("choiceset", "")),
    }
    return result

def deduplicate_profiles(rows: list[dict]) -> list[dict]:
    """
    Extract unique non-opt-out profiles (cash != 0).
    Returns list of unique profiles with full 5 attributes.
    """
    seen = {}
    for row in rows:
        parsed = parse_dce_row(row)
        # Skip opt-out (alt == "C" or cash == 0 and origin == 0 and eff == 0)
        if parsed["alt"] == "C":
            continue
        # Create key with all 5 attributes
        key = (
            parsed["wait"],
            parsed["eff"],
            parsed["se"],
            parsed["cash"],
            parsed["origin"]
        )
        if key not in seen:
            seen[key] = parsed
    
    unique_profiles = list(seen.values())
    return unique_profiles

def deduplicate_choice_tasks(rows: list[dict]) -> list[dict]:
    """
    Extract unique choice tasks (A/B/C triplets).
    Each task is identified by respondent_id + choiceset.
    Since all respondents see same 6 tasks, deduplicate by choiceset only.
    """
    tasks = defaultdict(lambda: {"A": None, "B": None, "C": None})
    
    for row in rows:
        parsed = parse_dce_row(row)
        task_key = parsed["choiceset"]  # All respondents share same 6 tasks
        alt = parsed["alt"]
        
        if alt in ["A", "B", "C"]:
            tasks[task_key][alt] = parsed
    
    # Convert to list of complete tasks
    complete_tasks = []
    for task_id, alts in tasks.items():
        if alts["A"] is not None and alts["B"] is not None and alts["C"] is not None:
            complete_tasks.append({
                "task_id": task_id,
                "A": alts["A"],
                "B": alts["B"],
                "C": alts["C"],
            })
    
    return complete_tasks

# ============================================================================
# PROMPT GENERATION
# ============================================================================

def generate_binary_prompt(profile: dict) -> str:
    """Generate binary acceptance prompt for a single profile."""
    origin_str = "reference" if profile["origin"] == 0 else "non-reference"
    return USER_PROMPT_TEMPLATE_BINARY.format(
        wait=profile["wait"],
        eff=profile["eff"],
        se=profile["se"],
        cash=int(profile["cash"]),
        origin=origin_str
    )

def generate_multinomial_prompt(task: dict) -> str:
    """Generate multinomial choice prompt for a choice task."""
    origin_a = "reference" if task["A"]["origin"] == 0 else "non-reference"
    origin_b = "reference" if task["B"]["origin"] == 0 else "non-reference"
    
    return USER_PROMPT_TEMPLATE_MULTINOMIAL.format(
        wait_a=task["A"]["wait"],
        eff_a=task["A"]["eff"],
        se_a=task["A"]["se"],
        cash_a=int(task["A"]["cash"]),
        origin_a=origin_a,
        wait_b=task["B"]["wait"],
        eff_b=task["B"]["eff"],
        se_b=task["B"]["se"],
        cash_b=int(task["B"]["cash"]),
        origin_b=origin_b,
    )

# ============================================================================
# RESPONDENT SPLIT (80/20)
# ============================================================================

def split_respondents(rows: list[dict]) -> tuple[set[str], set[str]]:
    """Split respondents 80/20 for train/test."""
    rids = sorted({parse_dce_row(row)["respondent_id"] for row in rows})
    rng = random.Random(SEED)
    shuffled = rids[:]
    rng.shuffle(shuffled)
    n_train = int(round(TRAIN_FRAC * len(shuffled)))
    n_train = max(1, min(n_train, len(shuffled) - 1))
    train_ids = set(shuffled[:n_train])
    test_ids = set(shuffled[n_train:])
    return train_ids, test_ids

def remove_wait_anomaly(rows: list[dict]) -> list[dict]:
    """Remove wait=2 anomaly before split."""
    filtered = []
    for row in rows:
        parsed = parse_dce_row(row)
        if parsed["wait"] != 2:
            filtered.append(row)
    return filtered

# ============================================================================
# AUDIT / DRY-RUN
# ============================================================================

def audit_validation_setup():
    """
    Print comprehensive audit of validation setup without calling API.
    """
    print("=" * 80)
    print("HELD-OUT DCE VALIDATION - FULL ATTRIBUTES AUDIT")
    print("=" * 80)
    print()
    
    # Load data
    print("1. DATA LOADING")
    print("-" * 80)
    if not DCE_FILE.exists():
        print(f"❌ DCE file not found: {DCE_FILE}")
        sys.exit(1)
    
    raw_rows = load_dce_data(DCE_FILE)
    print(f"   Total rows in DCE: {len(raw_rows)}")
    
    # Remove wait=2 anomaly
    rows = remove_wait_anomaly(raw_rows)
    print(f"   After removing wait=2 anomaly: {len(rows)} rows")
    
    # Split respondents
    train_ids, test_ids = split_respondents(rows)
    print(f"   Train respondents: {len(train_ids)} ({100*TRAIN_FRAC:.0f}%)")
    print(f"   Test respondents:  {len(test_ids)} ({100*(1-TRAIN_FRAC):.0f}%)")
    print(f"   Seed: {SEED}")
    print()
    
    # Deduplicate unique profiles (non-opt-out)
    print("2. UNIQUE PROFILE EXTRACTION")
    print("-" * 80)
    unique_profiles = deduplicate_profiles(rows)
    print(f"   Unique non-opt-out profiles: {len(unique_profiles)}")
    
    # Show attribute distributions
    cash_values = sorted(set(p["cash"] for p in unique_profiles))
    origin_values = sorted(set(p["origin"] for p in unique_profiles))
    print(f"   Cash incentive values: {cash_values}")
    print(f"   Origin values: {origin_values}")
    print()
    
    # Show all unique profiles
    print("   Unique profile combinations (wait, eff, se, cash, origin):")
    for i, p in enumerate(unique_profiles, 1):
        print(f"     {i:2d}. wait={p['wait']:4.1f}, eff={p['eff']:.2f}, se={p['se']:4.1f}, "
              f"cash={p['cash']:5.0f}, origin={p['origin']}")
    print()
    
    # Deduplicate choice tasks
    print("3. UNIQUE CHOICE TASK EXTRACTION")
    print("-" * 80)
    unique_tasks = deduplicate_choice_tasks(rows)
    print(f"   Unique choice tasks: {len(unique_tasks)}")
    print()
    
    for i, task in enumerate(unique_tasks, 1):
        print(f"   Task {task['task_id']}:")
        print(f"     A: wait={task['A']['wait']}, eff={task['A']['eff']:.2f}, "
              f"se={task['A']['se']}, cash={task['A']['cash']:.0f}, origin={task['A']['origin']}")
        print(f"     B: wait={task['B']['wait']}, eff={task['B']['eff']:.2f}, "
              f"se={task['B']['se']}, cash={task['B']['cash']:.0f}, origin={task['B']['origin']}")
        print(f"     C: Opt-out")
        print()
    
    # API call estimates
    print("4. API CALL ESTIMATES")
    print("-" * 80)
    profile_calls = len(unique_profiles) * REPEATS
    task_calls = len(unique_tasks) * REPEATS
    total_calls = profile_calls + task_calls
    print(f"   Profile-level binary validation:")
    print(f"     {len(unique_profiles)} profiles × {REPEATS} repeats = {profile_calls} calls")
    print()
    print(f"   Task-level multinomial validation:")
    print(f"     {len(unique_tasks)} tasks × {REPEATS} repeats = {task_calls} calls")
    print()
    print(f"   Total estimated API calls: {total_calls}")
    print()
    
    # Prompt templates
    print("5. PROMPT TEMPLATES")
    print("-" * 80)
    print()
    print("   SYSTEM PROMPT:")
    print(f"   {SYSTEM_PROMPT!r}")
    print()
    print("   USER PROMPT TEMPLATE (BINARY):")
    print(generate_binary_prompt(unique_profiles[0]))
    print()
    if unique_tasks:
        print("   USER PROMPT TEMPLATE (MULTINOMIAL):")
        print(generate_multinomial_prompt(unique_tasks[0]))
    print()
    
    # Decoding parameters
    print("6. DECODING PARAMETERS")
    print("-" * 80)
    print("   temperature: 0.0")
    print("   max_tokens: 512 (local: 128)")
    print("   stop sequences: [\\nsystem, system You, ...] (local only)")
    print()
    
    # Output schema
    print("7. OUTPUT SCHEMA")
    print("-" * 80)
    print("   Profile-level (BINARY):")
    print("     - profile_id: unique identifier for profile")
    print("     - repeat: repetition number (1-10)")
    print("     - wait, eff, se, cash, origin: full attributes")
    print("     - decision: Yes/No")
    print("     - probability_0_100: 0-100 scale")
    print("     - probability_0_1: 0-1 scale")
    print("     - reasoning: extracted rationale")
    print()
    print("   Task-level (MULTINOMIAL):")
    print("     - task_id: task identifier")
    print("     - repeat: repetition number (1-10)")
    print("     - choice: A/B/C")
    print("     - prob_A, prob_B, prob_C: probabilities (sum to 100)")
    print("     - reasoning: extracted rationale")
    print()
    
    # Mapping key
    print("8. MAPPING KEY")
    print("-" * 80)
    print("   Profile-level mapping:")
    print("     (wait, eff, se, cash, origin) → LLM probability")
    print()
    print("   Task-level mapping:")
    print("     (task_id) → (P_A, P_B, P_C)")
    print()
    
    # Validation approach summary
    print("9. VALIDATION APPROACH SUMMARY")
    print("-" * 80)
    print("   Primary: Task-level multinomial log loss")
    print("     - 6 unique tasks × 205 held-out respondents = 1,230 observations")
    print("     - Proper scoring rule for 3-alternative DCE")
    print("     - EFR: P~_i = λ P^LLM_i + (1-λ) P^DCE_i")
    print("     - All 5 attributes fully represented")
    print()
    print("   Secondary: Profile-level binary log loss")
    print("     - 12 unique profiles × matched respondents")
    print("     - Sensitivity analysis only")
    print()
    
    # Lambda
    print("10. ANALYSIS PARAMETERS")
    print("-" * 80)
    print(f"    λ (lambda) for primary analysis: {LAMBDA} (fixed)")
    print(f"    λ sensitivity: [0.0, 0.25, 0.5, 0.75, 1.0] (after primary)")
    print(f"    Random seed: {SEED}")
    print()
    
    print("=" * 80)
    print("AUDIT COMPLETE - No API calls made")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Review audit output above")
    print("  2. Confirm all unique profiles/tasks are correct")
    print("  3. Run with --mode dry-run to generate prompts without API calls")
    print("  4. Run with --mode api to make actual API calls")
    print()
    
    return {
        "unique_profiles": unique_profiles,
        "unique_tasks": unique_tasks,
        "train_ids": train_ids,
        "test_ids": test_ids,
        "total_api_calls": total_calls
    }

def main():
    parser = argparse.ArgumentParser(
        description="Held-out DCE validation with full 5 attributes"
    )
    parser.add_argument(
        "--mode",
        choices=["audit", "dry-run", "api"],
        default="audit",
        help="audit: print setup summary; dry-run: generate prompts; api: call API"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE,
        help="Output directory for generated files"
    )
    args = parser.parse_args()
    
    if args.mode == "audit":
        audit_validation_setup()
    elif args.mode == "dry-run":
        print("Dry-run mode: generating prompts without API calls...")
        # TODO: implement dry-run
        print("Not yet implemented")
    elif args.mode == "api":
        print("API mode: this will make actual API calls")
        print("Not yet implemented - please confirm audit first")
        sys.exit(1)

if __name__ == "__main__":
    main()
