#!/usr/bin/env python3
"""
Execute 420 Qwen 7B calls for 42 DCE tasks (multinomial validation) using Ollama.
42 tasks × 10 reps = 420 calls
"""
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any

WORKSPACE = Path(__file__).resolve().parent
INPUT_CSV = WORKSPACE / "multinomial_tasks_6.csv"
OUTPUT_RAW = WORKSPACE / "llm_raw_outputs_multinomial_42tasks.csv"
OUTPUT_PARSED = WORKSPACE / "llm_parsed_outputs_multinomial_42tasks.csv"
MODEL = "qwen2.5:7b"

def load_tasks(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def call_ollama(prompt: str, task_num: int, rep: int) -> dict:
    """Call Ollama."""
    try:
        # Format messages for chat
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        # Call ollama (no temperature control in CLI)
        result = subprocess.run(
            ["ollama", "run", MODEL, json.dumps(messages)],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            return {
                "task_num": task_num,
                "repetition": rep,
                "raw_response": result.stdout.strip(),
                "status": "success"
            }
        else:
            return {
                "task_num": task_num,
                "repetition": rep,
                "raw_response": f"Error: {result.stderr}",
                "status": "error"
            }
    except subprocess.TimeoutExpired:
        return {
            "task_num": task_num,
            "repetition": rep,
            "raw_response": "Timeout",
            "status": "timeout"
        }
    except Exception as e:
        return {
            "task_num": task_num,
            "repetition": rep,
            "raw_response": str(e),
            "status": "error"
        }

def parse_multinomial(response: str) -> dict:
    """Extract probabilities from multinomial response."""
    import re
    
    # Look for P(A), P(B), P(C) patterns
    pa = re.search(r'P\(A\):\s*(\d+\.?\d*)', response, re.IGNORECASE)
    pb = re.search(r'P\(B\):\s*(\d+\.?\d*)', response, re.IGNORECASE)
    pc = re.search(r'P\(C\):\s*(\d+\.?\d*)', response, re.IGNORECASE)
    
    # Look for Choice: A/B/C
    choice_match = re.search(r'Choice:\s*([ABC])', response, re.IGNORECASE)
    
    return {
        "P_A": float(pa.group(1)) if pa else None,
        "P_B": float(pb.group(1)) if pb else None,
        "P_C": float(pc.group(1)) if pc else None,
        "choice": choice_match.group(1).upper() if choice_match else None,
        "raw_truncated": response[:200]
    }

def main():
    print("=" * 80)
    print("EXECUTING 420 OLLAMA CALLS (42 multinomial tasks × 10 reps)")
    print(f"Model: {MODEL}")
    print("=" * 80)
    print()
    
    # Load tasks
    tasks = load_tasks(INPUT_CSV)
    print(f"Loaded {len(tasks)} tasks")
    
    # Resume capability
    raw_results = []
    parsed_results = []
    start_task = 1
    start_rep = 1
    
    if OUTPUT_RAW.exists():
        with open(OUTPUT_RAW, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            raw_results = list(reader)
            if raw_results:
                last = raw_results[-1]
                start_task = int(last["task_num"])
                start_rep = int(last["repetition"]) + 1
                if start_rep > 10:
                    start_task += 1
                    start_rep = 1
                print(f"Resuming from Task {start_task}, Rep {start_rep}")
    
    print()
    
    total = len(tasks) * 10
    completed = len(raw_results)
    
    for task in tasks:
        task_num = int(task["task_num"])
        
        if task_num < start_task:
            continue
        
        prompt = task["prompt"]
        print(f"Task {task_num}/42: wait={task['A_wait']}m vs {task['B_wait']}m")
        
        for rep in range(1, 11):
            if task_num == start_task and rep < start_rep:
                continue
            
            result = call_ollama(prompt, task_num, rep)
            raw_results.append(result)
            
            # Parse
            parsed = parse_multinomial(result["raw_response"])
            parsed_results.append({
                "task_num": task_num,
                "repetition": rep,
                **parsed
            })
            
            completed += 1
            status_char = "✓" if result["status"] == "success" else "✗"
            print(f"  Rep {rep}/10: {status_char} ({completed}/{total})")
            
            # Save incremental
            with open(OUTPUT_RAW, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["task_num", "repetition", "raw_response", "status"])
                writer.writeheader()
                writer.writerows(raw_results)
            
            with open(OUTPUT_PARSED, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["task_num", "repetition", "P_A", "P_B", "P_C", "choice", "raw_truncated"])
                writer.writeheader()
                writer.writerows(parsed_results)
        
        print()
    
    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"Total calls: {completed}")
    print(f"Success: {sum(1 for r in raw_results if r['status'] == 'success')}")
    print(f"Output: {OUTPUT_RAW}")

if __name__ == "__main__":
    main()
