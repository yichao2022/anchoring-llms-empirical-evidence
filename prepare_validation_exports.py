#!/usr/bin/env python3
"""
Export 42 DCE tasks to CSV for manual audit.
Prepare multinomial prompt pipeline (API execution DISABLED).
"""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

WORKSPACE = Path(__file__).resolve().parent
DCE_FILE = WORKSPACE / "analysis_output" / "dce_encoded.csv"
OUTPUT_CSV = WORKSPACE / "dce_tasks_42_export.csv"
OUTPUT_JSON = WORKSPACE / "dce_tasks_42_export.json"

def load_dce_data(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def normalize_profile(wait, eff, se, cash, origin):
    if eff > 1:
        eff = eff / 100.0
    return (float(wait), float(eff), float(se), float(cash), int(origin))

def respondent_facing_label(attr: str, value: Any) -> str:
    """Convert analysis-coded value to respondent-facing label (Table S3)."""
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
        eff_val = value * 100 if value <= 1 else value
        return f"{int(eff_val)}%"
    elif attr == "se":
        se_map = {0: "0%", 1: "10%", 2: "20%", 3: "30%"}
        return se_map.get(int(value), f"{int(value)}%")
    elif attr == "cash":
        return f"{int(value)} RMB"
    elif attr == "origin":
        return "Domestic" if int(value) == 0 else "Imported"
    return str(value)

def build_multinomial_prompt(a_prof: tuple, b_prof: tuple) -> str:
    """
    Build multinomial prompt (structure ready, wording provisional).
    
    Note: Final wording depends on original questionnaire recovery.
    Current labels from Table S3 coding dictionary.
    """
    # Provisional prompt structure
    prompt = f"""You are evaluating vaccination choices for COVID-19 vaccines.

Please consider the following three alternatives and indicate which one you would choose:

Alternative A:
- Waiting time: {respondent_facing_label('wait', a_prof[0])}
- Vaccine effectiveness: {respondent_facing_label('eff', a_prof[1])}
- Risk of side effects: {respondent_facing_label('se', a_prof[2])}
- Cash incentive: {respondent_facing_label('cash', a_prof[3])}
- Vaccine origin: {respondent_facing_label('origin', a_prof[4])}

Alternative B:
- Waiting time: {respondent_facing_label('wait', b_prof[0])}
- Vaccine effectiveness: {respondent_facing_label('eff', b_prof[1])}
- Risk of side effects: {respondent_facing_label('se', b_prof[2])}
- Cash incentive: {respondent_facing_label('cash', b_prof[3])}
- Vaccine origin: {respondent_facing_label('origin', b_prof[4])}

Alternative C:
- Do not receive the vaccine

Question: Which alternative would you choose? Please respond with ONLY the letter (A, B, or C).

Additionally, if you were forced to assign probabilities to each alternative (summing to 1), what would they be?
Please provide P(A), P(B), P(C) as three numbers between 0 and 1 that sum to 1."""
    
    return prompt

def export_tasks(rows: list[dict]) -> dict:
    """
    Export 42 tasks to CSV with full metadata.
    """
    print("=" * 80)
    print("EXPORTING 42 DCE TASKS TO CSV")
    print("=" * 80)
    print()
    
    # Build blocks and tasks
    resp_data = defaultdict(lambda: {"tasks": {}, "block": None})
    
    for row in rows:
        resp_id = str(row.get("RespondentID", "")).strip()
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
        
        if alt not in ["A", "B", "C"]:
            continue
        
        if task_id not in resp_data[resp_id]["tasks"]:
            resp_data[resp_id]["tasks"][task_id] = {}
        resp_data[resp_id]["tasks"][task_id][alt] = prof
    
    # Build blocks
    block_sigs = defaultdict(list)
    
    for resp_id, data in resp_data.items():
        tasks = data["tasks"]
        if len(tasks) != 6:
            continue
        
        # Build block signature
        task_sigs = []
        for task_id in sorted(tasks.keys(), key=lambda x: int(x) if x.isdigit() else 999):
            alts = tasks[task_id]
            if "A" in alts and "B" in alts:
                sig = (alts["A"], alts["B"], "OPT_OUT")
                task_sigs.append(sig)
        
        if len(task_sigs) == 6:
            block_sig = tuple(task_sigs)
            block_sigs[block_sig].append(resp_id)
    
    # Assign block IDs
    blocks = {}
    for i, (block_sig, resp_ids) in enumerate(sorted(block_sigs.items(), key=lambda x: -len(x[1])), 1):
        blocks[block_sig] = {
            "block_id": i,
            "respondents": resp_ids,
            "n_respondents": len(resp_ids)
        }
    
    print(f"Identified {len(blocks)} blocks")
    for block_sig, block_data in blocks.items():
        print(f"  Block {block_data['block_id']}: {block_data['n_respondents']} respondents")
    print()
    
    # Build task-level data
    all_tasks = []
    task_id_global = 0
    
    for block_sig, block_data in blocks.items():
        block_id = block_data["block_id"]
        
        for task_idx, task_sig in enumerate(block_sig, 1):
            a_prof, b_prof, _ = task_sig
            task_id_global += 1
            
            # Count held-out respondents for this task
            random.seed(2026)
            all_resps = list(resp_data.keys())
            n_test = int(len(all_resps) * 0.2)
            test_resps = set(random.sample(all_resps, n_test))
            
            heldout_count = 0
            for resp_id in block_data["respondents"]:
                if resp_id in test_resps:
                    heldout_count += 1
            
            task_record = {
                "block_id": block_id,
                "task_id_within_block": task_idx,
                "task_global_id": task_id_global,
                "task_signature": str(task_sig),
                "A_wait": a_prof[0],
                "A_eff_coded": a_prof[1],
                "A_eff_display": respondent_facing_label("eff", a_prof[1]),
                "A_se_coded": a_prof[2],
                "A_se_display": respondent_facing_label("se", a_prof[2]),
                "A_cash": a_prof[3],
                "A_origin_coded": a_prof[4],
                "A_origin_display": respondent_facing_label("origin", a_prof[4]),
                "B_wait": b_prof[0],
                "B_eff_coded": b_prof[1],
                "B_eff_display": respondent_facing_label("eff", b_prof[1]),
                "B_se_coded": b_prof[2],
                "B_se_display": respondent_facing_label("se", b_prof[2]),
                "B_cash": b_prof[3],
                "B_origin_coded": b_prof[4],
                "B_origin_display": respondent_facing_label("origin", b_prof[4]),
                "C_label": "Do not receive the vaccine",
                "prompt_text": build_multinomial_prompt(a_prof, b_prof),
                "n_respondents_total": block_data["n_respondents"],
                "n_heldout_respondents_receiving_task": heldout_count,
                "expected_repeats": 10,
                "planned_api_calls": 10,  # 1 task × 10 repeats
                "provenance_status": "pending_provider_confirmation",
                "wording_status": "provisional_from_Table_S3",
            }
            
            all_tasks.append(task_record)
    
    # Export to CSV
    fieldnames = [
        "block_id", "task_id_within_block", "task_global_id", "task_signature",
        "A_wait", "A_eff_coded", "A_eff_display", "A_se_coded", "A_se_display",
        "A_cash", "A_origin_coded", "A_origin_display",
        "B_wait", "B_eff_coded", "B_eff_display", "B_se_coded", "B_se_display",
        "B_cash", "B_origin_coded", "B_origin_display",
        "C_label", "prompt_text",
        "n_respondents_total", "n_heldout_respondents_receiving_task",
        "expected_repeats", "planned_api_calls",
        "provenance_status", "wording_status"
    ]
    
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_tasks)
    
    print(f"✓ Exported {len(all_tasks)} tasks to {OUTPUT_CSV}")
    print()
    
    # Export to JSON (machine-readable companion)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "n_blocks": len(blocks),
                "n_tasks": len(all_tasks),
                "n_respondents_total": 1027,
                "heldout_split": "80/20 (seed=2026)",
                "provenance_status": "pending_provider_confirmation",
                "wording_status": "provisional_from_Table_S3",
                "api_execution": "DISABLED"
            },
            "tasks": all_tasks
        }, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Exported JSON companion to {OUTPUT_JSON}")
    print()
    
    # Display sample
    print("Sample CSV records (first 3):")
    print("-" * 80)
    for task in all_tasks[:3]:
        print(f"\nTask {task['task_global_id']} (Block {task['block_id']}, Task {task['task_id_within_block']}):")
        print(f"  A: {task['A_wait']}mo, {task['A_eff_display']}, {task['A_se_display']}, "
              f"{task['A_cash']} RMB, {task['A_origin_display']}")
        print(f"  B: {task['B_wait']}mo, {task['B_eff_display']}, {task['B_se_display']}, "
              f"{task['B_cash']} RMB, {task['B_origin_display']}")
        print(f"  Respondents: {task['n_respondents_total']} total, "
              f"{task['n_heldout_respondents_receiving_task']} held-out")
        print(f"  Status: {task['provenance_status']} / {task['wording_status']}")
    
    print()
    
    # Summary statistics
    total_api_calls = sum(t["planned_api_calls"] for t in all_tasks)
    total_heldout_obs = sum(t["n_heldout_respondents_receiving_task"] * 6 for t in all_tasks)
    
    print("SUMMARY STATISTICS:")
    print("-" * 80)
    print(f"  Total blocks: {len(blocks)}")
    print(f"  Total unique tasks: {len(all_tasks)}")
    print(f"  Total planned API calls: {total_api_calls}")
    print(f"  Total held-out task observations: ~{total_heldout_obs // len(all_tasks) * len(all_tasks)}")
    print()
    
    return {
        "n_blocks": len(blocks),
        "n_tasks": len(all_tasks),
        "total_api_calls": total_api_calls,
        "csv_path": str(OUTPUT_CSV),
        "json_path": str(OUTPUT_JSON)
    }

def prepare_prompt_pipeline():
    """
    Prepare multinomial prompt pipeline structure.
    API execution DISABLED until provenance confirmed.
    """
    print("=" * 80)
    print("MULTINOMIAL PROMPT PIPELINE (API EXECUTION DISABLED)")
    print("=" * 80)
    print()
    
    pipeline = {
        "status": "READY_BUT_FROZEN",
        "execution_blocked": True,
        "block_reason": "pending_provider_confirmation_and_original_wordings",
        "components": {
            "input": {
                "source": "dce_tasks_42_export.csv",
                "n_tasks": 42,
                "repetitions_per_task": 10,
                "total_api_calls": 420
            },
            "prompt_template": {
                "structure": "multinomial_choice_task",
                "alternatives": ["A", "B", "C"],
                "output_format": "probability_simplex",
                "constraints": ["P(A) + P(B) + P(C) = 1", "each P in [0,1]"]
            },
            "model_config": {
                "model": "Qwen2.5-72B-Instruct",
                "temperature": 0,
                "max_tokens": 500,
                "expected_output": "P(A), P(B), P(C) as three numbers"
            },
            "parser": {
                "target": "probability_triple",
                "validation": "simplex_sum_to_one",
                "fallback": "choice_letter_to_one_hot"
            },
            "output": {
                "file": "llm_parsed_outputs_qwen72b_heldout_tasks.csv",
                "columns": ["task_id", "repetition", "P_A", "P_B", "P_C", "raw_response"]
            }
        },
        "unblocking_conditions": [
            "1. Data provider confirms seven-block structure",
            "2. Original questionnaire wording obtained",
            "3. Prompt text updated with verified wording",
            "4. Dry-run validation passed",
            "5. User explicit approval to execute"
        ]
    }
    
    print(json.dumps(pipeline, indent=2))
    print()
    
    print("⚠️  PIPELINE FROZEN: API execution disabled")
    print("   Unblock when: provenance confirmed + wording verified")
    print()
    
    return pipeline

def draft_final_query():
    """
    Final data provider provenance query for approval.
    """
    query = """Subject: DCE Design Provenance Confirmation - Wuhan Vaccination Study (N=1,027)

Dear Colleague,

We are conducting validation analyses of the Wuhan vaccination-timing discrete choice experiment and have reconstructed the following structure from the analytic data (N=18,485 alt-rows after excluding one wait=2 anomaly):

RECONSTRUCTED DESIGN STRUCTURE:
- 7 distinct respondent groups (design blocks)
- Block sizes: 250, 241, 222, 95, 83, 80, 55 respondents
- 42 unique choice-set compositions (A/B/Opt-out task signatures)
- 84 unique non-opt-out profiles (across all blocks)
- Each block contains 6 tasks; tasks vary across blocks

This reconstruction suggests the administered design was a multi-block experiment rather than a single uniform block administered to all 1,027 respondents.

REQUEST FOR CONFIRMATION:

1. DESIGN INTENT
   Was the seven-block structure the intended administered design?
   If yes: What was the block assignment criteria (randomization, stratification, wave/timing)?
   If no: Is the seven-block pattern a data artifact or reconstruction error?

2. ORIGINAL DESIGN DOCUMENTATION
   Please provide if available:
   - Original experimental design matrix (blocks × tasks × profiles)
   - Questionnaire versions used (if multiple versions exist)
   - Block randomization protocol or assignment log
   - Any design documentation from the original study

3. QUESTIONNAIRE WORDING (CRITICAL)
   The current analysis uses Table S3 coding dictionary for respondent-facing labels. Please confirm or provide the exact wording used for:
   - Attribute labels (e.g., "Waiting time" vs "Wait time", "Vaccine effectiveness" vs "Efficacy")
   - Side-effect descriptions (e.g., "Risk of side effects: 10%" vs "Side effect burden: Level 1")
   - Opt-out alternative (e.g., "Do not receive the vaccine" vs "Opt out")
   - Choice task introduction text
   - Response format instructions
   
   Accurate wording is essential for LLM validation prompts to match the human respondent experience.

4. ANALYTIC RECOMMENDATION
   Based on the reconstructed structure, we are proceeding with:
   - Full-sample analysis using all 7 blocks (n=1,027)
   - Task-level multinomial validation on 42 unique choice sets
   - Complete 5-attribute profiles (no cash/origin mismatch)
   
   Please confirm this aligns with the study design or advise if a restricted analysis (e.g., largest block only) is preferred.

MANUSCRIPT IMPLICATIONS:

If the seven-block structure is confirmed, the following manuscript claims will be updated:
- "1 unique design block shared by all 1027 respondents" → "7 design blocks"
- "same six choice tasks" → "42 unique choice-set compositions across blocks"
- "12 administered non-opt-out profiles" → "84 profiles across full design"
- "12/216 = 5.6% coverage" → recalculated full-design coverage
- Held-out validation: 78-task (6.3%) → 1,229-task (100%) coverage

TIMELINE:
We have prepared the validation pipeline but are holding API calls pending your confirmation. Please respond at your earliest convenience so we can proceed with accurate design documentation.

Thank you for your assistance with this design provenance verification.

Best regards,
Yichao Jin
Ph.D. Candidate, UT Dallas
[Email: Yichao.Jin@UTDallas.edu]
"""
    
    return query

def main():
    print("=" * 80)
    print("PREPARING DCE VALIDATION EXPORTS")
    print("Seven blocks, 42 tasks, API execution DISABLED")
    print("=" * 80)
    print()
    
    # Load data
    rows = load_dce_data(DCE_FILE)
    rows_filtered = [r for r in rows if float(r.get("WaitTime", 0)) != 2]
    
    print(f"Data: {len(rows)} rows, {len(rows_filtered)} after wait≠2 filter")
    print()
    
    # Export tasks
    export_result = export_tasks(rows_filtered)
    
    # Prepare pipeline
    pipeline = prepare_prompt_pipeline()
    
    # Draft query
    query = draft_final_query()
    
    # Final summary
    print("=" * 80)
    print("DELIVERABLES READY")
    print("=" * 80)
    print()
    print(f"1. CSV Export: {export_result['csv_path']}")
    print(f"   - {export_result['n_tasks']} tasks with full metadata")
    print(f"   - {export_result['n_blocks']} blocks identified")
    print(f"   - All fields including provenance/wording status")
    print()
    print(f"2. JSON Companion: {export_result['json_path']}")
    print(f"   - Machine-readable format")
    print(f"   - Pipeline metadata")
    print()
    print("3. Prompt Pipeline:")
    print(f"   - Status: FROZEN (API execution DISABLED)")
    print(f"   - Planned calls: {export_result['total_api_calls']}")
    print(f"   - Unblock: provenance + wording confirmed")
    print()
    print("4. Data Provider Query:")
    print("   - Ready for approval (see below)")
    print()
    print("=" * 80)
    print("DATA PROVIDER QUERY (FOR APPROVAL)")
    print("=" * 80)
    print()
    print(query)

if __name__ == "__main__":
    main()
