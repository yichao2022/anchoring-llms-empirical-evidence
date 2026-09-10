#!/usr/bin/env python3
"""
Execute 60 Qwen2.5-72B calls for 6 canonical DCE tasks (multinomial validation).
6 tasks × 10 reps = 60 calls
Uses respondent-facing values (70% not 0.7, Domestic/Imported not reference/non-reference)
"""
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from datetime import datetime

MODEL = "qwen2.5:72b"
OUTPUT_RAW = Path("llm_raw_outputs_multinomial_6tasks_canonical.csv")
OUTPUT_PARSED = Path("llm_parsed_outputs_multinomial_6tasks_canonical.csv")

# Canonical 6 tasks with respondent-facing values
CANONICAL_6_TASKS = [
    {
        "task_num": 1,
        "A": {"wait": 2, "eff": 50, "se": 20, "cash": 800, "origin": "Domestic"},
        "B": {"wait": 6, "eff": 70, "se": 30, "cash": 50, "origin": "Imported"},
    },
    {
        "task_num": 2,
        "A": {"wait": 3, "eff": 70, "se": 10, "cash": 200, "origin": "Domestic"},
        "B": {"wait": 0, "eff": 50, "se": 20, "cash": 50, "origin": "Domestic"},
    },
    {
        "task_num": 3,
        "A": {"wait": 3, "eff": 70, "se": 30, "cash": 200, "origin": "Imported"},
        "B": {"wait": 1, "eff": 95, "se": 10, "cash": 800, "origin": "Imported"},
    },
    {
        "task_num": 4,
        "A": {"wait": 0, "eff": 95, "se": 30, "cash": 800, "origin": "Imported"},
        "B": {"wait": 3, "eff": 70, "se": 30, "cash": 200, "origin": "Domestic"},
    },
    {
        "task_num": 5,
        "A": {"wait": 1, "eff": 50, "se": 10, "cash": 50, "origin": "Domestic"},
        "B": {"wait": 6, "eff": 95, "se": 10, "cash": 200, "origin": "Domestic"},
    },
    {
        "task_num": 6,
        "A": {"wait": 6, "eff": 95, "se": 20, "cash": 800, "origin": "Imported"},
        "B": {"wait": 1, "eff": 50, "se": 20, "cash": 50, "origin": "Imported"},
    },
]

MULTINOMIAL_PROMPT_TEMPLATE = """You are evaluating vaccination choices. Consider these three alternatives:

Alternative A:
- Waiting time: {A_wait} months
- Vaccine effectiveness: {A_eff}%
- Risk of side effects: {A_se}%
- Cash incentive: {A_cash} RMB
- Vaccine origin: {A_origin}

Alternative B:
- Waiting time: {B_wait} months
- Vaccine effectiveness: {B_eff}%
- Risk of side effects: {B_se}%
- Cash incentive: {B_cash} RMB
- Vaccine origin: {B_origin}

Alternative C:
- Do not receive the vaccine / Opt out

Based on the information above, estimate the probability that a person described by these attributes would choose each alternative.

Respond exactly in this format:
P(A): [probability between 0 and 1]
P(B): [probability between 0 and 1]
P(C): [probability between 0 and 1]

Ensure P(A) + P(B) + P(C) = 1."""


def build_prompt(task: dict) -> str:
    """Build respondent-facing multinomial prompt."""
    return MULTINOMIAL_PROMPT_TEMPLATE.format(
        A_wait=task["A"]["wait"],
        A_eff=task["A"]["eff"],
        A_se=task["A"]["se"],
        A_cash=task["A"]["cash"],
        A_origin=task["A"]["origin"],
        B_wait=task["B"]["wait"],
        B_eff=task["B"]["eff"],
        B_se=task["B"]["se"],
        B_cash=task["B"]["cash"],
        B_origin=task["B"]["origin"],
    )


def call_ollama(prompt: str) -> tuple[str, bool]:
    """Call Ollama and return (response_text, success)."""
    messages = [{"role": "user", "content": prompt}]
    try:
        result = subprocess.run(
            ["ollama", "run", MODEL, json.dumps(messages)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            return result.stdout.strip(), True
        else:
            return f"Error: {result.stderr}", False
    except Exception as e:
        return f"Exception: {e}", False


def parse_multinomial_response(text: str) -> dict | None:
    """Parse P(A), P(B), P(C) from response."""
    import re
    p_a_match = re.search(r'P\(A\):\s*(0\.\d+|1\.0|1|0)', text, re.IGNORECASE)
    p_b_match = re.search(r'P\(B\):\s*(0\.\d+|1\.0|1|0)', text, re.IGNORECASE)
    p_c_match = re.search(r'P\(C\):\s*(0\.\d+|1\.0|1|0)', text, re.IGNORECASE)
    
    if not (p_a_match and p_b_match and p_c_match):
        return None
    
    p_a = float(p_a_match.group(1))
    p_b = float(p_b_match.group(1))
    p_c = float(p_c_match.group(1))
    
    # Normalize to sum to 1
    total = p_a + p_b + p_c
    if abs(total - 1.0) > 0.01:
        p_a /= total
        p_b /= total
        p_c /= total
    
    return {"P_A": p_a, "P_B": p_b, "P_C": p_c}


def main():
    # Check for resume
    existing = set()
    if OUTPUT_RAW.exists():
        with open(OUTPUT_RAW) as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing.add((int(row["task_num"]), int(row["repetition"])))
        print(f"Resuming from {len(existing)} existing calls")

    # Initialize CSV
    if not OUTPUT_RAW.exists():
        with open(OUTPUT_RAW, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task_num", "repetition", "raw_response", "status", "timestamp"])
    if not OUTPUT_PARSED.exists():
        with open(OUTPUT_PARSED, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task_num", "repetition", "P_A", "P_B", "P_C", "choice"])

    total_calls = len(CANONICAL_6_TASKS) * 10
    completed = 0

    print(f"Executing {total_calls} multinomial calls (6 tasks × 10 reps)")
    print(f"Model: {MODEL}")
    print("=" * 72)

    for task in CANONICAL_6_TASKS:
        task_num = task["task_num"]
        print(f"\nTask {task_num}/6: A({task['A']['wait']}mo,{task['A']['eff']}%) vs B({task['B']['wait']}mo,{task['B']['eff']}%)")
        
        prompt = build_prompt(task)
        
        for rep in range(1, 11):
            if (task_num, rep) in existing:
                print(f"  Rep {rep}/10: ✓ (cached)")
                completed += 1
                continue
            
            response, success = call_ollama(prompt)
            timestamp = datetime.now().isoformat()
            
            # Save raw
            with open(OUTPUT_RAW, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([task_num, rep, response, "success" if success else "error", timestamp])
            
            # Parse and save
            if success:
                parsed = parse_multinomial_response(response)
                if parsed:
                    with open(OUTPUT_PARSED, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([task_num, rep, parsed["P_A"], parsed["P_B"], parsed["P_C"], ""])
                    print(f"  Rep {rep}/10: ✓ ({completed+1}/{total_calls})")
                else:
                    print(f"  Rep {rep}/10: ✗ parse error ({completed+1}/{total_calls})")
            else:
                print(f"  Rep {rep}/10: ✗ ({completed+1}/{total_calls})")
            
            completed += 1

    print("\n" + "=" * 72)
    print("COMPLETE")
    print(f"Total calls: {total_calls}")
    print(f"Output: {OUTPUT_RAW}")
    print(f"Parsed: {OUTPUT_PARSED}")


if __name__ == "__main__":
    main()
