#!/usr/bin/env python3
"""
Execute 120 Qwen calls for 12 DCE profiles.
Local vLLM endpoint: http://127.0.0.1:8000/v1
"""
from __future__ import annotations

import csv
import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

WORKSPACE = Path(__file__).resolve().parent
INPUT_CSV = WORKSPACE / "dce_12_profiles_clean.csv"
OUTPUT_RAW = WORKSPACE / "llm_raw_outputs_12profiles.csv"
OUTPUT_PARSED = WORKSPACE / "llm_parsed_outputs_12profiles.csv"

API_BASE = "http://127.0.0.1:8000/v1"
MODEL_ID = "/Volumes/ORICO/mlx-models/Qwen2-72B-4bit"

def load_profiles(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def call_qwen(prompt: str, profile_id: int, rep: int) -> dict:
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

def parse_probability(response: str) -> dict:
    """Extract probability from response."""
    import re
    
    # Look for patterns like "0.7" or "70%" or "Yes" / "No"
    lines = response.strip().split("\n")
    
    # Try to find a number between 0 and 1
    prob = None
    for line in lines:
        # Look for decimal probability
        match = re.search(r'(\d+\.?\d*)', line)
        if match:
            val = float(match.group(1))
            if 0 <= val <= 1:
                prob = val
                break
            elif 0 <= val <= 100:
                prob = val / 100
                break
    
    # If no number found, check for Yes/No
    if prob is None:
        if "yes" in response.lower():
            prob = 1.0
        elif "no" in response.lower():
            prob = 0.0
    
    return {
        "parsed_probability": prob if prob is not None else "",
        "parsed_yes_no": "Yes" if "yes" in response.lower() else ("No" if "no" in response.lower() else "")
    }

def main():
    print("=" * 80)
    print("EXECUTING 120 QWEN CALLS (12 profiles × 10 reps)")
    print(f"Endpoint: {API_BASE}")
    print("=" * 80)
    print()
    
    # Load profiles
    profiles = load_profiles(INPUT_CSV)
    print(f"Loaded {len(profiles)} profiles")
    
    # Check for existing output and resume
    start_profile = 1
    start_rep = 1
    raw_results = []
    parsed_results = []
    
    if OUTPUT_RAW.exists():
        with open(OUTPUT_RAW, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            raw_results = list(reader)
            if raw_results:
                last = raw_results[-1]
                start_profile = int(last["profile_id"])
                start_rep = int(last["repetition"]) + 1
                if start_rep > 10:
                    start_profile += 1
                    start_rep = 1
                print(f"Resuming from Profile {start_profile}, Rep {start_rep}")
                # Reload parsed results too
                if OUTPUT_PARSED.exists():
                    with open(OUTPUT_PARSED, newline="", encoding="utf-8") as pf:
                        parsed_results = list(csv.DictReader(pf))
    
    print()
    
    total = len(profiles) * 10
    completed = 0
    
    for profile in profiles:
        profile_id = int(profile["profile_id"])
        
        # Skip completed profiles
        if profile_id < start_profile:
            completed += 10
            continue
        
        prompt = profile["prompt_text"]
        
        print(f"Profile {profile_id}: {profile['efficacy_label']}, {profile['side_effects_label']}, {profile['cash_label']}")
        
        for rep in range(1, 11):
            # Skip completed reps for current profile
            if profile_id == start_profile and rep < start_rep:
                continue
            
            result = call_qwen(prompt, profile_id, rep)
            raw_results.append(result)
            
            # Parse
            parsed = parse_probability(result["raw_response"])
            parsed_results.append({
                "profile_id": profile_id,
                "repetition": rep,
                **parsed,
                "raw_response": result["raw_response"][:500]  # Truncate for CSV
            })
            
            completed += 1
            if result["status"] == "success":
                print(f"  Rep {rep}/10: OK (tokens: {result['prompt_tokens']}+{result['completion_tokens']})")
            else:
                print(f"  Rep {rep}/10: ERROR - {result['raw_response'][:100]}")
            
            # Save incremental progress
            with open(OUTPUT_RAW, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["profile_id", "repetition", "raw_response", "finish_reason", "prompt_tokens", "completion_tokens", "status"])
                writer.writeheader()
                writer.writerows(raw_results)
            
            with open(OUTPUT_PARSED, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["profile_id", "repetition", "parsed_probability", "parsed_yes_no", "raw_response"])
                writer.writeheader()
                writer.writerows(parsed_results)
        
        print()
    
    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"Total calls: {completed}")
    print(f"Success: {sum(1 for r in raw_results if r['status'] == 'success')}")
    print(f"Errors: {sum(1 for r in raw_results if r['status'] == 'error')}")
    print(f"Raw outputs: {OUTPUT_RAW}")
    print(f"Parsed outputs: {OUTPUT_PARSED}")

if __name__ == "__main__":
    main()
