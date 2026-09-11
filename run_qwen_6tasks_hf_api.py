#!/usr/bin/env python3
"""
Execute 60 Qwen calls for 6 canonical DCE tasks (multinomial) via Hugging Face Inference API.
10 repetitions per task for stability.
Matches Table S1: Qwen2.5-72B-Instruct via HF Inference API, 2026-05-19 settings.
"""
from __future__ import annotations

import csv
import json
import os
import urllib.request
from pathlib import Path
from datetime import datetime

WORKSPACE = Path("/tmp/behavioral-digital-twins")
INPUT_CSV = WORKSPACE / "multinomial_tasks_6.csv"
OUTPUT_RAW = WORKSPACE / "llm_raw_outputs_multinomial_6tasks_hf_api.csv"
OUTPUT_PARSED = WORKSPACE / "llm_parsed_outputs_multinomial_6tasks_hf_api.csv"
PROGRESS_LOG = WORKSPACE / "qwen_progress_6tasks_hf_api.log"

# HF Inference API configuration
API_KEY = os.environ.get("HF_API_TOKEN", "")
API_BASE = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-72B-Instruct"
MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"

# Table S1 parameters
TEMPERATURE = 0.7
TOP_P = 1.0
MAX_TOKENS = 512
REPEATS = 10

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
    """Call Qwen via HF Inference API for a single task."""
    messages = [
        {"role": "system", "content": "You are a respondent completing a choice task. Provide your choice and the probability you would choose each alternative."},
        {"role": "user", "content": prompt}
    ]
    
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS
    }
    
    req = urllib.request.Request(
        API_BASE,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        },
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Parse probabilities
            probs = parse_probabilities(content)
            
            return {
                "task_num": task_num,
                "repetition": rep,
                "raw_response": content,
                "P_A": probs.get("P_A", ""),
                "P_B": probs.get("P_B", ""),
                "P_C": probs.get("P_C", ""),
                "finish_reason": result.get("choices", [{}])[0].get("finish_reason", ""),
                "prompt_tokens": result.get("usage", {}).get("prompt_tokens", 0),
                "completion_tokens": result.get("usage", {}).get("completion_tokens", 0),
            }
    except Exception as e:
        log(f"Error calling API for task {task_num}, rep {rep}: {e}")
        return {
            "task_num": task_num,
            "repetition": rep,
            "raw_response": str(e),
            "P_A": "",
            "P_B": "",
            "P_C": "",
            "finish_reason": "error",
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

def main():
    # Clear progress log
    if PROGRESS_LOG.exists():
        PROGRESS_LOG.unlink()
    
    # Load tasks
    tasks = []
    with open(INPUT_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            tasks.append(row)
    
    log(f"Loaded {len(tasks)} tasks from {INPUT_CSV}")
    
    # Initialize output files
    with open(OUTPUT_RAW, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_num", "repetition", "raw_response", "finish_reason", "prompt_tokens", "completion_tokens"])
    
    with open(OUTPUT_PARSED, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_num", "repetition", "P_A", "P_B", "P_C"])
    
    # Execute calls
    total = len(tasks) * REPEATS
    completed = 0
    
    for task in tasks:
        task_num = int(task["task_num"])
        prompt = task["prompt"]
        
        for rep in range(1, REPEATS + 1):
            result = call_qwen(prompt, task_num, rep)
            
            # Write raw output
            with open(OUTPUT_RAW, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    result["task_num"],
                    result["repetition"],
                    result["raw_response"],
                    result["finish_reason"],
                    result["prompt_tokens"],
                    result["completion_tokens"]
                ])
            
            # Write parsed output
            with open(OUTPUT_PARSED, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    result["task_num"],
                    result["repetition"],
                    result["P_A"],
                    result["P_B"],
                    result["P_C"]
                ])
            
            completed += 1
            log(f"Completed {completed}/{total}: Task {task_num}, Rep {rep}")
    
    log(f"All {total} calls completed")
    log(f"Raw outputs: {OUTPUT_RAW}")
    log(f"Parsed outputs: {OUTPUT_PARSED}")

if __name__ == "__main__":
    if not API_KEY:
        print("Error: HF_API_TOKEN environment variable not set")
        exit(1)
    main()
