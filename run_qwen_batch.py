#!/usr/bin/env python3
"""
Execute 120 Qwen calls for 12 DCE profiles.
Batch mode with progress tracking.
"""
from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path
from datetime import datetime

WORKSPACE = Path("/tmp/behavioral-digital-twins")
INPUT_CSV = WORKSPACE / "dce_12_profiles_clean.csv"
OUTPUT_RAW = WORKSPACE / "llm_raw_outputs_12profiles.csv"
OUTPUT_PARSED = WORKSPACE / "llm_parsed_outputs_12profiles.csv"
PROGRESS_LOG = WORKSPACE / "qwen_progress.log"

API_BASE = "http://127.0.0.1:8000/v1"
MODEL_ID = "/Volumes/ORICO/mlx-models/Qwen2-72B-4bit"

def log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(PROGRESS_LOG, "a") as f:
        f.write(line + "\n")

def call_qwen(prompt: str, profile_id: int, rep: int) -> dict:
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
                "profile_id": profile_id,
                "repetition": rep,
                "raw_response": result["choices"][0]["message"]["content"],
                "finish_reason": result["choices"][0].get("finish_reason", ""),
                "prompt_tokens": result["usage"].get("prompt_tokens", 0),
                "completion_tokens": result["usage"].get("completion_tokens", 0),
                "status": "success"
            }
    except Exception as e:
        return {
            "profile_id": profile_id,
            "repetition": rep,
            "raw_response": str(e),
            "finish_reason": "error",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "status": "error"
        }

def main():
    log("=" * 80)
    log("STARTING: 120 Qwen calls (12 profiles × 10 reps)")
    log(f"Endpoint: {API_BASE}")
    log("=" * 80)
    
    # Load profiles
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        profiles = list(csv.DictReader(f))
    
    log(f"Loaded {len(profiles)} profiles")
    
    raw_results = []
    total = len(profiles) * 10
    completed = 0
    
    for profile in profiles:
        profile_id = int(profile["profile_id"])
        prompt = profile["prompt_text"]
        
        log(f"Profile {profile_id}: Starting 10 reps...")
        
        for rep in range(1, 11):
            result = call_qwen(prompt, profile_id, rep)
            raw_results.append(result)
            completed += 1
            
            if result["status"] == "success":
                log(f"  Rep {rep}/10: OK ({result['completion_tokens']} tokens)")
            else:
                log(f"  Rep {rep}/10: ERROR - {result['raw_response'][:50]}")
        
        log(f"Profile {profile_id}: Complete ({completed}/{total} total)")
    
    # Save
    with open(OUTPUT_RAW, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["profile_id", "repetition", "raw_response", "finish_reason", "prompt_tokens", "completion_tokens", "status"])
        writer.writeheader()
        writer.writerows(raw_results)
    
    successes = sum(1 for r in raw_results if r['status'] == 'success')
    log("=" * 80)
    log(f"COMPLETE: {completed} calls, {successes} success, {completed-successes} errors")
    log(f"Output: {OUTPUT_RAW}")
    log("=" * 80)

if __name__ == "__main__":
    main()
