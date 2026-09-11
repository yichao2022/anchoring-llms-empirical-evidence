#!/usr/bin/env python3
"""
A/B Position Bias Sensitivity Test: Swapped Alternatives
6 tasks × swapped A/B × 10 reps = 60 calls
Uses JSON Lines format to avoid CSV escaping issues.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from datetime import datetime
import re

MODEL = "qwen2.5:72b"
OUTPUT_JSONL = Path("llm_outputs_multinomial_6tasks_swapped.jsonl")

# Canonical 6 tasks - with attributes swapped
SWAPPED_6_TASKS = [
    {
        "task_num": 1,
        "display_A": {"wait": 6, "eff": 70, "se": 10, "cash": 50, "origin": "Imported"},
        "display_B": {"wait": 2, "eff": 50, "se": 1, "cash": 800, "origin": "Domestic"},
    },
    {
        "task_num": 2,
        "display_A": {"wait": 0, "eff": 50, "se": 1, "cash": 50, "origin": "Domestic"},
        "display_B": {"wait": 3, "eff": 70, "se": 0.1, "cash": 200, "origin": "Domestic"},
    },
    {
        "task_num": 3,
        "display_A": {"wait": 1, "eff": 95, "se": 0.1, "cash": 800, "origin": "Imported"},
        "display_B": {"wait": 3, "eff": 70, "se": 10, "cash": 200, "origin": "Imported"},
    },
    {
        "task_num": 4,
        "display_A": {"wait": 3, "eff": 70, "se": 10, "cash": 200, "origin": "Domestic"},
        "display_B": {"wait": 0, "eff": 95, "se": 10, "cash": 800, "origin": "Imported"},
    },
    {
        "task_num": 5,
        "display_A": {"wait": 6, "eff": 95, "se": 0.1, "cash": 200, "origin": "Domestic"},
        "display_B": {"wait": 1, "eff": 50, "se": 0.1, "cash": 50, "origin": "Domestic"},
    },
    {
        "task_num": 6,
        "display_A": {"wait": 1, "eff": 50, "se": 1, "cash": 50, "origin": "Imported"},
        "display_B": {"wait": 6, "eff": 95, "se": 1, "cash": 800, "origin": "Imported"},
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
    messages = [{"role": "user", "content": prompt}]
    payload = {
        "model": MODEL,
        "messages": messages,
        "options": {"temperature": 0.7, "num_predict": 500},
        "stream": False,
    }
    
    try:
        result = subprocess.run(
            ["curl", "-s", "-X", "POST", 
             "http://localhost:11434/api/chat",
             "-H", "Content-Type: application/json",
             "-d", json.dumps(payload)],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            return f"CURL_ERROR: {result.stderr}", False
        response = json.loads(result.stdout)
        content = response.get("message", {}).get("content", "")
        return content, True
    except Exception as e:
        return f"ERROR: {e}", False


def parse_response(response: str) -> dict:
    result = {"P_A": None, "P_B": None, "P_C": None, "parse_status": "FAIL"}
    
    pa_match = re.search(r'P\(A\)\s*[:=]\s*([0-9.]+)', response, re.IGNORECASE)
    pb_match = re.search(r'P\(B\)\s*[:=]\s*([0-9.]+)', response, re.IGNORECASE)
    pc_match = re.search(r'P\(C\)\s*[:=]\s*([0-9.]+)', response, re.IGNORECASE)
    
    if pa_match and pb_match and pc_match:
        try:
            p_a = float(pa_match.group(1))
            p_b = float(pb_match.group(1))
            p_c = float(pc_match.group(1))
            
            total = p_a + p_b + p_c
            if total > 0 and abs(total - 1.0) > 0.01:
                p_a, p_b, p_c = p_a/total, p_b/total, p_c/total
            
            result = {"P_A": p_a, "P_B": p_b, "P_C": p_c, "parse_status": "SUCCESS"}
        except ValueError:
            pass
    
    return result


def map_swapped_to_original(parsed: dict) -> dict:
    """Map swapped labels back to original: LLM's A is original B, LLM's B is original A."""
    p_swapped_a = parsed.get("P_A")
    p_swapped_b = parsed.get("P_B")
    p_c = parsed.get("P_C")
    
    if p_swapped_a is None or p_swapped_b is None or p_c is None:
        return parsed
    
    return {
        "P_A_original": p_swapped_b,  # LLM's B is original A
        "P_B_original": p_swapped_a,  # LLM's A is original B
        "P_C": p_c,
        "parse_status": parsed.get("parse_status", "UNKNOWN")
    }


def main():
    print("=" * 60)
    print("A/B Position Bias Sensitivity Test (Swapped Alternatives)")
    print("=" * 60)
    
    # Clear output file
    OUTPUT_JSONL.write_text("")
    
    total_calls = 0
    successful_calls = 0
    
    for task in SWAPPED_6_TASKS:
        task_num = task["task_num"]
        print(f"\n--- Task {task_num} (SWAPPED A/B) ---")
        
        for rep in range(1, 11):
            total_calls += 1
            timestamp = datetime.now().isoformat()
            prompt = build_prompt(task)
            
            print(f"  Rep {rep}/10 ... ", end="", flush=True)
            response, success = call_ollama(prompt)
            
            parsed = parse_response(response) if success else {"P_A": None, "P_B": None, "P_C": None, "parse_status": "FAIL"}
            mapped = map_swapped_to_original(parsed) if success else {}
            
            record = {
                "task_num": task_num,
                "rep": rep,
                "model": MODEL,
                "timestamp": timestamp,
                "prompt": prompt,
                "raw_response": response,
                "success": success,
                "P_A_swapped": parsed.get("P_A"),
                "P_B_swapped": parsed.get("P_B"),
                "P_C": parsed.get("P_C"),
                "P_A_original": mapped.get("P_A_original"),
                "P_B_original": mapped.get("P_B_original"),
                "parse_status": parsed.get("parse_status", "FAIL")
            }
            
            with open(OUTPUT_JSONL, "a") as f:
                f.write(json.dumps(record) + "\n")
            
            if success and parsed.get("parse_status") == "SUCCESS":
                successful_calls += 1
                print(f"OK (orig A={mapped['P_A_original']:.3f}, B={mapped['P_B_original']:.3f})")
            else:
                print(f"FAIL")
    
    print(f"\n{'='*60}")
    print(f"Complete! {successful_calls}/{total_calls} successful")
    print(f"Output: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
