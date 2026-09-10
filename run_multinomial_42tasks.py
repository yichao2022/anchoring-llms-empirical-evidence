#!/usr/bin/env python3
"""
Execute 420 Qwen calls for 42 DCE tasks (multinomial validation).
42 tasks × 10 reps = 420 calls
"""
from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent
INPUT_CSV = WORKSPACE / "multinomial_tasks_6.csv"
OUTPUT_RAW = WORKSPACE / "llm_raw_outputs_multinomial_42tasks.csv"
OUTPUT_PARSED = WORKSPACE / "llm_parsed_outputs_multinomial_42tasks.csv"

API_BASE = "http://127.0.0.1:8000/v1"
MODEL_ID = "/Volumes/ORICO/mlx-models/Qwen2-72B-4bit"

def load_tasks(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def call_qwen(prompt: str, task_num: int, rep: int) -> dict:
    """Call local vLLM endpoint."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Respond to the user's question directly."},
        {"role": "user", "content": prompt}
    ]
    
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": 0.7,
        "top_p": 1.0,
        "max_tokens": 500
    }
    
    req = urllib.request.Request(
        f"{API_BASE}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return {
                "task_num": task_num,
                "repetition": rep,
                "raw_response": result["choices"][0]["message"]["content"],
                "finish_reason": result["choices"][0].get("finish_reason", ""),
                "prompt_tokens": result["usage"].get("prompt_tokens", 0),
                "completion_tokens": result["usage"].get("completion_tokens", 0),
                "status": "success"
            }
    except Exception as e:
        return {
            "task_num": task_num,
            "repetition": rep,
            "raw_response": str(e),
            "finish_reason": "error",
            "prompt_tokens": 0,
            "completion_tokens": 0,
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
    print("EXECUTING 420 QWEN CALLS (42 multinomial tasks × 10 reps)")
    print(f"Endpoint: {API_BASE}")
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
    completed = 0
    
    for task in tasks:
        task_num = int(task["task_num"])
        
        if task_num < start_task:
            completed += 10
            continue
        
        prompt = task["prompt"]
        print(f"Task {task_num}/42: {task['A_wait']}m vs {task['B_wait']}m wait")
        
        for rep in range(1, 11):
            if task_num == start_task and rep < start_rep:
                continue
            
            result = call_qwen(prompt, task_num, rep)
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
            print(f"  Rep {rep}/10: {status_char}")
            
            # Save incremental
            with open(OUTPUT_RAW, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["task_num", "repetition", "raw_response", "finish_reason", "prompt_tokens", "completion_tokens", "status"])
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

if __name__ == "__main__":
    main()
