#!/usr/bin/env python3
"""
Extract 6 unique DCE tasks for multinomial validation.
Each task includes Alternative A, Alternative B, and Opt-out (C).
"""
import csv
from collections import defaultdict
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent
DCE_FILE = WORKSPACE / "analysis_output" / "dce_encoded.csv"

def load_dce_data(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def parse_row(row):
    return {
        "respondent_id": str(row.get("RespondentID", "")).strip(),
        "task_id": str(row.get("Choiceset", "")).strip(),
        "alt": row.get("Alt", "").strip().upper(),
        "wait": float(row.get("WaitTime", 0)),
        "eff": float(row.get("VaccineEfficacy", 0)),
        "se": float(row.get("SideEffects", 0)),
        "cash": float(row.get("CashIncentives", 0)),
        "origin": int(float(row.get("VaccineOrigin", 0))),
        "choice": int(float(row.get("Choice", 0))),
    }

def origin_label(origin_code):
    return "Domestic" if origin_code == 0 else "Imported"

def se_label(se_code):
    """Map side-effect code to percentage."""
    mapping = {1: 0.1, 2: 1, 3: 10}
    return mapping.get(se_code, se_code)

def wait_label(wait):
    """Format waiting time."""
    if wait == 0:
        return "walk-in (no wait)"
    else:
        return f"{int(wait)} months"

def build_task_signature(tasks_data):
    """Build unique task signature from A, B profiles."""
    a = tasks_data.get("A")
    b = tasks_data.get("B")
    if not a or not b:
        return None
    
    # Ordered signature (A then B)
    sig = (
        (a["wait"], a["eff"], a["se"], a["cash"], a["origin"]),
        (b["wait"], b["eff"], b["se"], b["cash"], b["origin"])
    )
    return sig

def format_profile_for_prompt(profile):
    """Format profile with respondent-facing labels."""
    lines = []
    lines.append(f"- Waiting time: {wait_label(profile['wait'])}")
    lines.append(f"- Vaccine effectiveness: {int(profile['eff']*100)}%")
    lines.append(f"- Risk of side effects: {se_label(profile['se'])}%")
    lines.append(f"- Cash incentive: {int(profile['cash'])} RMB")
    lines.append(f"- Vaccine origin: {origin_label(profile['origin'])}")
    return "\n".join(lines)

def create_multinomial_prompt(task_data):
    """Create respondent-facing multinomial prompt."""
    a = task_data.get("A")
    b = task_data.get("B")
    
    if not a or not b:
        return None
    
    prompt = f"""You are evaluating COVID-19 vaccination choices.

Please consider the following three alternatives and indicate which you would choose:

Alternative A:
{format_profile_for_prompt(a)}

Alternative B:
{format_profile_for_prompt(b)}

Alternative C:
- Do not receive the vaccine (opt out)

Question: Which alternative would you choose? Please respond with ONLY "A", "B", or "C".

Additionally, please provide the probability (between 0 and 1) that you would choose each alternative. The probabilities should sum to 1.

Format your response as:
Choice: [A/B/C]
P(A): [probability]
P(B): [probability]
P(C): [probability]"""

    return prompt

def main():
    print("Loading DCE data...")
    rows = load_dce_data(DCE_FILE)
    
    # Group by respondent and task
    resp_tasks = defaultdict(lambda: defaultdict(lambda: {"A": None, "B": None, "C": None}))
    
    for row in rows:
        parsed = parse_row(row)
        if parsed["wait"] == 2:  # Skip anomaly
            continue
        if parsed["alt"] in ["A", "B", "C"]:
            resp_tasks[parsed["respondent_id"]][parsed["task_id"]][parsed["alt"]] = parsed
    
    print(f"Loaded {len(resp_tasks)} respondents")
    
    # Collect unique task signatures (from first respondent with complete data)
    unique_tasks = {}
    
    for resp_id, tasks in resp_tasks.items():
        for task_id, alts in tasks.items():
            if alts["A"] and alts["B"] and alts["C"]:
                sig = build_task_signature(alts)
                if sig and sig not in unique_tasks:
                    unique_tasks[sig] = {
                        "task_id": task_id,
                        "A": alts["A"],
                        "B": alts["B"],
                        "C": alts["C"],
                        "first_respondent": resp_id
                    }
    
    print(f"\nFound {len(unique_tasks)} unique task signatures")
    
    # Check if we have exactly 6
    if len(unique_tasks) != 6:
        print(f"WARNING: Expected 6 tasks, found {len(unique_tasks)}")
    
    # Generate prompts
    print("\n" + "="*80)
    print("MULTINOMIAL TASK PROMPTS")
    print("="*80)
    
    tasks_list = []
    for i, (sig, data) in enumerate(sorted(unique_tasks.items(), key=lambda x: x[1]['task_id']), 1):
        prompt = create_multinomial_prompt(data)
        tasks_list.append({
            "task_num": i,
            "task_id": data["task_id"],
            "prompt": prompt,
            "A": sig[0],
            "B": sig[1]
        })
        
        print(f"\n--- Task {i} (Task ID: {data['task_id']}) ---")
        print(prompt)
        print("-"*80)
    
    # Save to file
    output_file = WORKSPACE / "multinomial_tasks_6.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task_num", "task_id", "prompt", "A_wait", "A_eff", "A_se", "A_cash", "A_origin",
                        "B_wait", "B_eff", "B_se", "B_cash", "B_origin"])
        for task in tasks_list:
            writer.writerow([
                task["task_num"], task["task_id"], task["prompt"],
                *task["A"], *task["B"]
            ])
    
    print(f"\nSaved {len(tasks_list)} tasks to: {output_file}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Unique tasks found: {len(unique_tasks)}")
    print(f"Expected: 6")
    print(f"Status: {'✓ MATCH' if len(unique_tasks) == 6 else '✗ MISMATCH'}")
    print(f"\nNext step: Run 6 tasks × 10 repetitions = 60 API calls")

if __name__ == "__main__":
    main()
