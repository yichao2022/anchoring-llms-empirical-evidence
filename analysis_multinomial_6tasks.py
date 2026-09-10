#!/usr/bin/env python3
"""
Compare Pure-DCE / Raw LLM / EFR on canonical 6 tasks multinomial validation.
"""
import csv
import json
import numpy as np
import pandas as pd
from collections import defaultdict

# Load LLM results
with open('llm_parsed_outputs_multinomial_6tasks_canonical.csv') as f:
    reader = csv.DictReader(f)
    llm_rows = list(reader)

# Aggregate LLM probs by task
llm_task_probs = defaultdict(lambda: {'P_A': [], 'P_B': [], 'P_C': []})
for row in llm_rows:
    t = int(row['task_num'])
    llm_task_probs[t]['P_A'].append(float(row['P_A']))
    llm_task_probs[t]['P_B'].append(float(row['P_B']))
    llm_task_probs[t]['P_C'].append(float(row['P_C']))

# Average across reps
llm_probs = {}
for t in range(1, 7):
    if t in llm_task_probs:
        llm_probs[t] = {
            'P_A': np.mean(llm_task_probs[t]['P_A']),
            'P_B': np.mean(llm_task_probs[t]['P_B']),
            'P_C': np.mean(llm_task_probs[t]['P_C']),
        }

# Save aggregated LLM probs
with open('llm_aggregated_6tasks.json', 'w') as f:
    json.dump({str(k): v for k, v in llm_probs.items()}, f, indent=2)

print("LLM Aggregated Probabilities:")
print("=" * 60)
for t in sorted(llm_probs.keys()):
    p = llm_probs[t]
    print(f"Task {t}: P_A={p['P_A']:.4f}, P_B={p['P_B']:.4f}, P_C={p['P_C']:.4f}")

print("\n✓ Analysis complete")
print("  - LLM probs aggregated: llm_aggregated_6tasks.json")
print("\nNext: Need Pure-DCE model predictions for same 6 tasks")
print("      Then compute multinomial log loss and Brier scores")
