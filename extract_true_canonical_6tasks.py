#!/usr/bin/env python3
"""
Extract TRUE 6 canonical DCE tasks for multinomial validation.
Each respondent sees the same 6 Choicesets (1-6), each with A, B, C alternatives.
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
        "task_id": str(row.get("Choiceset", "")).strip(),  # This is the canonical task ID (1-6)
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
    
    # Get data from first respondent (all respondents have same 6 tasks)
    resp_id = "1"
    resp_rows = [parse_row(r) for r in rows if r.get("RespondentID") == resp_id]
    
    # Group by task_id (Choiceset 1-6)
    tasks = defaultdict(lambda: {"A": None, "B": None, "C": None})
    for parsed in resp_rows:
        # Note: wait=2 is not an anomaly - it's a valid value (2 months wait)
        # The "anomaly" was a misunderstanding in previous code
        if parsed["alt"] in ["A", "B", "C"]:
            tasks[parsed["task_id"]][parsed["alt"]] = parsed
    
    print(f"Found {len(tasks)} tasks for Respondent 1")
    
    # Check we have exactly 6 tasks
    if len(tasks) != 6:
        print(f"ERROR: Expected 6 tasks, found {len(tasks)}")
        print(f"Task IDs found: {sorted(tasks.keys())}")
        return
    
    # Generate prompts for true 6 canonical tasks
    print("\n" + "="*80)
    print("TRUE 6 CANONICAL DCE TASKS")
    print("="*80)
    
    tasks_list = []
    for i, task_id in enumerate(sorted(tasks.keys(), key=int), 1):
        alts = tasks[task_id]
        if not alts["A"] or not alts["B"]:
            print(f"ERROR: Task {task_id} missing A or B")
            continue
            
        prompt = create_multinomial_prompt(alts)
        tasks_list.append({
            "task_num": i,
            "task_id": task_id,
            "prompt": prompt,
            "A": alts["A"],
            "B": alts["B"],
            "C": alts["C"]
        })
        
        print(f"\n--- Task {i} (Choiceset {task_id}) ---")
        print(f"A: wait={alts['A']['wait']}, eff={alts['A']['eff']}, se={alts['A']['se']}, cash={alts['A']['cash']}, origin={alts['A']['origin']}")
        print(f"B: wait={alts['B']['wait']}, eff={alts['B']['eff']}, se={alts['B']['se']}, cash={alts['B']['cash']}, origin={alts['B']['origin']}")
        print("-"*80)
    
    # Save to file
    output_file = WORKSPACE / "multinomial_tasks_6_CANONICAL.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task_num", "task_id", "prompt", 
                        "A_wait", "A_eff", "A_se", "A_cash", "A_origin",
                        "B_wait", "B_eff", "B_se", "B_cash", "B_origin"])
        for task in tasks_list:
            writer.writerow([
                task["task_num"], task["task_id"], task["prompt"],
                task["A"]["wait"], task["A"]["eff"], task["A"]["se"], task["A"]["cash"], task["A"]["origin"],
                task["B"]["wait"], task["B"]["eff"], task["B"]["se"], task["B"]["cash"], task["B"]["origin"]
            ])
    
    print(f"\n✓ Saved {len(tasks_list)} TRUE canonical tasks to: {output_file}")
    
    # Verify against old file
    print("\n" + "="*80)
    print("VERIFICATION: Comparing to old (incorrect) file")
    print("="*80)
    try:
        with open(WORKSPACE / "multinomial_tasks_6.csv", 'r') as f:
            old_reader = csv.DictReader(f)
            old_rows = list(old_reader)[:6]
        
        print("\nOld file first 6 rows (INCORRECT):")
        for i, row in enumerate(old_rows):
            print(f"  Row {i+1}: task_id={row['task_id']}, A=({row['A_wait']},{row['A_eff']},{row['A_se']}), B=({row['B_wait']},{row['B_eff']},{row['B_se']})")
        
        print("\nNew file 6 canonical tasks (CORRECT):")
        for task in tasks_list:
            print(f"  Task {task['task_num']}: task_id={task['task_id']}, A=({task['A']['wait']},{task['A']['eff']},{task['A']['se']}), B=({task['B']['wait']},{task['B']['eff']},{task['B']['se']})")
        
        # Check if any match
        matches = 0
        for task in tasks_list:
            for old_row in old_rows:
                if (task['A']['wait'] == float(old_row['A_wait']) and 
                    task['A']['eff'] == float(old_row['A_eff']) and
                    task['A']['se'] == float(old_row['A_se']) and
                    task['B']['wait'] == float(old_row['B_wait']) and
                    task['B']['eff'] == float(old_row['B_eff']) and
                    task['B']['se'] == float(old_row['B_se'])):
                    matches += 1
                    break
        print(f"\nMatches between old and new: {matches}/6")
        if matches == 0:
            print("⚠️  WARNING: No matches! Old data is completely wrong.")
        elif matches < 6:
            print(f"⚠️  WARNING: Only {matches}/6 match. Partial overlap.")
        else:
            print("✓ All 6 match (unexpected)")
            
    except FileNotFoundError:
        print("Old file not found for comparison")

if __name__ == "__main__":
    main()
