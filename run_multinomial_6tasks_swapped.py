#!/usr/bin/env python3
"""
A/B Position Bias Sensitivity Test: Swapped Alternatives
6 tasks × swapped A/B × 10 reps = 60 calls
After running, probabilities need to be mapped back to original labels for comparison.
"""
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from datetime import datetime
import re

MODEL = "qwen2.5:72b"
OUTPUT_RAW = Path("llm_raw_outputs_multinomial_6tasks_swapped.csv")
OUTPUT_PARSED = Path("llm_parsed_outputs_multinomial_6tasks_swapped.csv")

# Canonical 6 tasks - with attributes swapped
# Task 1: original A=(2mo,50%,1%,800,Dom), B=(6mo,70%,10%,50,Imp)
# Swapped: A=(6mo,70%,10%,50,Imp), B=(2mo,50%,1%,800,Dom)
SWAPPED_6_TASKS = [
    {
        "task_num": 1,
        # These are what the LLM sees as "Alternative A" (originally B)
        "display_A": {"wait": 6, "eff": 70, "se": 10, "cash": 50, "origin": "Imported"},
        # These are what the LLM sees as "Alternative B" (originally A)
        "display_B": {"wait": 2, "eff": 50, "se": 1, "cash": 800, "origin": "Domestic"},
        # Original labels for post-processing mapping
        "original_A_was": "display_B",  # display_B was originally A
        "original_B_was": "display_A",  # display_A was originally B
    },
    {
        "task_num": 2,
        "display_A": {"wait": 0, "eff": 50, "se": 1, "cash": 50, "origin": "Domestic"},
        "display_B": {"wait": 3, "eff": 70, "se": 0.1, "cash": 200, "origin": "Domestic"},
        "original_A_was": "display_B",
        "original_B_was": "display_A",
    },
    {
        "task_num": 3,
        "display_A": {"wait": 1, "eff": 95, "se": 0.1, "cash": 800, "origin": "Imported"},
        "display_B": {"wait": 3, "eff": 70, "se": 10, "cash": 200, "origin": "Imported"},
        "original_A_was": "display_B",
        "original_B_was": "display_A",
    },
    {
        "task_num": 4,
        "display_A": {"wait": 3, "eff": 70, "se": 10, "cash": 200, "origin": "Domestic"},
        "display_B": {"wait": 0, "eff": 95, "se": 10, "cash": 800, "origin": "Imported"},
        "original_A_was": "display_B",
        "original_B_was": "display_A",
    },
    {
        "task_num": 5,
        "display_A": {"wait": 6, "eff": 95, "se": 0.1, "cash": 200, "origin": "Domestic"},
        "display_B": {"wait": 1, "eff": 50, "se": 0.1, "cash": 50, "origin": "Domestic"},
        "original_A_was": "display_B",
        "original_B_was": "display_A",
    },
    {
        "task_num": 6,
        "display_A": {"wait": 1, "eff": 50, "se": 1, "cash": 50, "origin": "Imported"},
        "display_B": {"wait": 6, "eff": 95, "se": 1, "cash": 800, "origin": "Imported"},
        "original_A_was": "display_B",
        "original_B_was": "display_A",
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
    """Build respondent-facing multinomial prompt with swapped A/B."""
    return MULTINOMIAL_PROMPT_TEMPLATE.format(
        A_wait=task["display_A"]["wait"],
        A_eff=task["display_A"]["eff"],
        A_se=task["display_A"]["se"],
        A_cash=task["display_A"]["cash"],
        A_origin=task["display_A"]["origin"],
        B_wait=task["display_B"]["wait"],
        B_eff=task["display_B"]["eff"],
        B_se=task["display_B"]["se"],
        B_cash=task["display_B"]["cash"],
        B_origin=task["display_B"]["origin"],
    )


def call_ollama(prompt: str) -> tuple[str, bool]:
    """Call Ollama and return (response_text, success)."""
    messages = [{"role": "user", "content": prompt}]
    
    payload = {
        "model": MODEL,
        "messages": messages,
        "options": {
            "temperature": 0.7,
            "num_predict": 500,
        },
        "stream": False,
    }
    
    try:
        result = subprocess.run(
            ["curl", "-s", "-X", "POST", 
             "http://localhost:11434/api/chat",
             "-H", "Content-Type: application/json",
             "-d", json.dumps(payload)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        
        if result.returncode != 0:
            return f"CURL_ERROR: {result.stderr}", False
        
        response = json.loads(result.stdout)
        content = response.get("message", {}).get("content", "")
        return content, True
        
    except subprocess.TimeoutExpired:
        return "TIMEOUT", False
    except Exception as e:
        return f"ERROR: {e}", False


def parse_response(response: str) -> dict:
    """Parse multinomial probabilities from LLM response."""
    result = {"P_A": None, "P_B": None, "P_C": None, "parse_status": "FAIL"}
    
    # Extract P(A), P(B), P(C)
    pa_match = re.search(r'P\(A\)\s*[:=]\s*([0-9.]+)', response, re.IGNORECASE)
    pb_match = re.search(r'P\(B\)\s*[:=]\s*([0-9.]+)', response, re.IGNORECASE)
    pc_match = re.search(r'P\(C\)\s*[:=]\s*([0-9.]+)', response, re.IGNORECASE)
    
    if pa_match and pb_match and pc_match:
        try:
            p_a = float(pa_match.group(1))
            p_b = float(pb_match.group(1))
            p_c = float(pc_match.group(1))
            
            # Normalize if needed (proportional, not softmax)
            total = p_a + p_b + p_c
            if total > 0 and abs(total - 1.0) > 0.01:
                p_a, p_b, p_c = p_a/total, p_b/total, p_c/total
            
            result = {
                "P_A": p_a,
                "P_B": p_b,
                "P_C": p_c,
                "parse_status": "SUCCESS"
            }
        except ValueError:
            pass
    
    return result


def map_swapped_to_original(parsed: dict) -> dict:
    """
    Map probabilities from swapped labels back to original labels.
    LLM outputs P_swapped(A) and P_swapped(B), but these correspond to original B and A.
    """
    # In swapped version:
    # - LLM's "A" is actually original "B"
    # - LLM's "B" is actually original "A"
    # - "C" stays the same
    
    p_swapped_a = parsed.get("P_A")
    p_swapped_b = parsed.get("P_B")
    p_c = parsed.get("P_C")
    
    if p_swapped_a is None or p_swapped_b is None or p_c is None:
        return parsed
    
    return {
        "P_A": p_swapped_b,  # LLM's B is original A
        "P_B": p_swapped_a,  # LLM's A is original B
        "P_C": p_c,          # C stays the same
        "parse_status": parsed.get("parse_status", "UNKNOWN"),
        "note": "Swapped A/B: mapped back to original labels"
    }


def main():
    """Run swapped A/B position bias test."""
    print("=" * 60)
    print("A/B Position Bias Sensitivity Test (Swapped Alternatives)")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Tasks: 6 canonical DCE tasks")
    print(f"Repetitions: 10 per task")
    print(f"Total calls: 60")
    print(f"Output: {OUTPUT_RAW}, {OUTPUT_PARSED}")
    print("=" * 60)
    print()
    
    # Initialize output files
    with open(OUTPUT_RAW, "w", newline="", encoding="utf-8") as f_raw:
        raw_writer = csv.writer(f_raw)
        raw_writer.writerow(["task_num", "rep", "model", "timestamp", "raw_response", "success"])
    
    with open(OUTPUT_PARSED, "w", newline="", encoding="utf-8") as f_parsed:
        parsed_writer = csv.writer(f_parsed)
        parsed_writer.writerow(["task_num", "rep", "P_A_original", "P_B_original", "P_C", 
                               "P_A_swapped_raw", "P_B_swapped_raw", "parse_status"])
    
    # Run experiment
    total_calls = 0
    successful_calls = 0
    
    for task in SWAPPED_6_TASKS:
        task_num = task["task_num"]
        print(f"\n--- Task {task_num} (SWAPPED A/B) ---")
        
        for rep in range(1, 11):  # 10 repetitions
            total_calls += 1
            timestamp = datetime.now().isoformat()
            
            # Build prompt with swapped A/B
            prompt = build_prompt(task)
            
            # Call Ollama
            print(f"  Task {task_num}, Rep {rep}/10 ... ", end="", flush=True)
            response, success = call_ollama(prompt)
            
            # Save raw output
            with open(OUTPUT_RAW, "a", newline="", encoding="utf-8") as f_raw:
                raw_writer = csv.writer(f_raw)
                raw_writer.writerow([
                    task_num,
                    rep,
                    MODEL,
                    timestamp,
                    response,
                    "SUCCESS" if success else "FAIL"
                ])
            
            if success:
                # Parse probabilities
                parsed = parse_response(response)
                
                # Map back to original labels
                mapped = map_swapped_to_original(parsed)
                
                with open(OUTPUT_PARSED, "a", newline="", encoding="utf-8") as f_parsed:
                    parsed_writer = csv.writer(f_parsed)
                    parsed_writer.writerow([
                        task_num,
                        rep,
                        mapped.get("P_A"),
                        mapped.get("P_B"),
                        mapped.get("P_C"),
                        parsed.get("P_A"),  # raw swapped values
                        parsed.get("P_B"),
                        mapped.get("parse_status", "FAIL")
                    ])
                
                successful_calls += 1
                print(f"OK (P_A_orig={mapped.get('P_A', 'N/A'):.3f}, P_B_orig={mapped.get('P_B', 'N/A'):.3f})")
            else:
                print(f"FAIL: {response[:50]}")
    
    print("\n" + "=" * 60)
    print(f"Complete! Success: {successful_calls}/{total_calls}")
    print(f"Raw outputs: {OUTPUT_RAW}")
    print(f"Parsed outputs (mapped to original labels): {OUTPUT_PARSED}")
    print("=" * 60)


if __name__ == "__main__":
    main()
