#!/usr/bin/env python3
"""
Execute 60 Qwen calls for 6 canonical DCE tasks (multinomial).
10 repetitions per task for stability.
"""
from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path
from datetime import datetime
import os

WORKSPACE = Path("/tmp/behavioral-digital-twins")
INPUT_CSV = WORKSPACE / "multinomial_tasks_6.csv"
OUTPUT_RAW = WORKSPACE / "llm_raw_outputs_multinomial_6tasks_canonical_NEW.csv"
OUTPUT_PARSED = WORKSPACE / "llm_parsed_outputs_multinomial_6tasks_canonical_NEW.csv"
PROGRESS_LOG = WORKSPACE / "qwen_progress_6tasks.log"

# API configuration - using local Ollama
API_KEY = "ollama"  # Ollama doesn't need auth
API_BASE = "http://127.0.0.1:11434/v1"
MODEL_ID = "qwen2.5:72b"

def log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(PROGRESS_LOG, "a") as f:
        f.write(line + "\n")

def parse_probabilities(text: str) -> dict:
    """Parse P(A), P(B), P(C) from LLM response."""
    import re
    text = text.strip()
    
    # Try to find probabilities
    p_a_match = re.search(r'P\(A\)\s*[:=]\s*(0?\.\d+)', text, re.IGNORECASE)
    p_b_match = re.search(r'P\(B\)\s*[:=]\s*(0?\.\d+)', text, re.IGNORECASE)
    p_c_match = re.search(r'P\(C\)\s*[:=]\s*(0?\.\d+)', text, re.IGNORECASE)
    
    # Also try without parentheses
    if not p_a_match:
        p_a_match = re.search(r'P_A\s*[:=]\s*(0?\.\d+)', text, re.IGNORECASE)
    if not p_b_match:
        p_b_match = re.search(r'P_B\s*[:=]\s*(0?\.\d+)', text, re.IGNORECASE)
    if not p_c_match:
        p_c_match = re.search(r'P_C\s*[:=]\s*(0?\.\d+)', text, re.IGNORECASE)
    
    probs = {}
    if p_a_match:
        probs['P_A'] = float(p_a_match.group(1))
    if p_b_match:
        probs['P_B'] = float(p_b_match.group(1))
    if p_c_match:
        probs['P_C'] = float(p_c_match.group(1))
    
    # If we have less than 3 probabilities, try to extract any numbers
    if len(probs) < 3:
        numbers = re.findall(r'\b(0?\.\d+)\b', text)
        if len(numbers) >= 3:
            if 'P_A' not in probs:
                probs['P_A'] = float(numbers[0])
            if 'P_B' not in probs:
                probs['P_B'] = float(numbers[1])
            if 'P_C' not in probs:
                probs['P_C'] = float(numbers[2])
    
    return probs

def call_qwen(prompt: str, task_num: int, rep: int) -> dict:
    """Call Qwen API for a single task."""
    messages = [
        {"role": "system", "content": "You are a respondent completing a choice task. Provide your choice and the probability you would choose each alternative."},
        {"role": "user", "content": prompt}
    ]
    
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    req = urllib.request.Request(
        f"{API_BASE}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        },
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            return {
                "task_num": task_num,
                "rep": rep,
                "raw_response": content,
                "success": True,
                "error": None
            }
    except Exception as e:
        return {
            "task_num": task_num,
            "rep": rep,
            "raw_response": "",
            "success": False,
            "error": str(e)
        }

def main():
    log("=" * 60)
    log("Starting 6 Canonical Tasks LLM Validation")
    log(f"Model: {MODEL_ID}")
    log(f"Tasks: 6, Reps per task: 10, Total calls: 60")
    log("=" * 60)
    
    # Load tasks
    tasks = []
    with open(INPUT_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tasks.append(row)
    
    log(f"Loaded {len(tasks)} tasks")
    
    # Run API calls
    results = []
    total = len(tasks) * 10
    completed = 0
    
    for task in tasks:
        task_num = int(task['task_num'])
        prompt = task['prompt']
        
        for rep in range(1, 11):  # 10 repetitions
            result = call_qwen(prompt, task_num, rep)
            results.append(result)
            completed += 1
            
            status = "✓" if result["success"] else "✗"
            log(f"[{completed}/{total}] Task {task_num} Rep {rep}: {status}")
            
            if result["success"]:
                probs = parse_probabilities(result["raw_response"])
                log(f"    P(A)={probs.get('P_A', 'N/A')}, P(B)={probs.get('P_B', 'N/A')}, P(C)={probs.get('P_C', 'N/A')}")
            else:
                log(f"    Error: {result['error']}")
    
    # Save raw outputs
    with open(OUTPUT_RAW, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['task_num', 'rep', 'raw_response', 'success', 'error'])
        for r in results:
            writer.writerow([r['task_num'], r['rep'], r['raw_response'], r['success'], r['error']])
    
    log(f"\nSaved raw outputs to: {OUTPUT_RAW}")
    
    # Parse and save probabilities
    with open(OUTPUT_PARSED, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['task_num', 'rep', 'P_A', 'P_B', 'P_C', 'raw_response'])
        for r in results:
            if r['success']:
                probs = parse_probabilities(r['raw_response'])
                writer.writerow([
                    r['task_num'], 
                    r['rep'], 
                    probs.get('P_A', ''),
                    probs.get('P_B', ''),
                    probs.get('P_C', ''),
                    r['raw_response'][:200]  # Truncated for readability
                ])
    
    log(f"Saved parsed outputs to: {OUTPUT_PARSED}")
    
    # Summary
    success_count = sum(1 for r in results if r['success'])
    log(f"\n{'='*60}")
    log(f"SUMMARY: {success_count}/{total} successful calls")
    log(f"{'='*60}")

if __name__ == "__main__":
    main()
