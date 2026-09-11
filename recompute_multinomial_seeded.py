#!/usr/bin/env python3
"""
Full multinomial validation with seeded random split (seed=2026).
Recomputes DCE refit and log losses with correct split.
"""
import csv
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Config
SEED = 2026
TRAIN_FRAC = 0.80

def softmax_3choice(util_A, util_B, util_C):
    """Compute choice probabilities from utilities."""
    max_u = np.maximum(np.maximum(util_A, util_B), util_C)
    exp_A = np.exp(util_A - max_u)
    exp_B = np.exp(util_B - max_u)
    exp_C = np.exp(util_C - max_u)
    sum_exp = exp_A + exp_B + exp_C
    return exp_A / sum_exp, exp_B / sum_exp, exp_C / sum_exp

def multinomial_log_loss(y_true, probs):
    """y_true: 0=A, 1=B, 2=C; probs: dict with 'A', 'B', 'C' arrays."""
    eps = 1e-15
    p = np.where(y_true == 0, probs['A'], 
                 np.where(y_true == 1, probs['B'], probs['C']))
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(np.log(p))

# Load DCE data
print("Loading DCE data...")
df = pd.read_csv('analysis_output/dce_encoded.csv')

# Load LLM results
print("Loading LLM results...")
with open('llm_parsed_outputs_multinomial_6tasks_canonical.csv') as f:
    reader = csv.DictReader(f)
    llm_rows = list(reader)

# Aggregate LLM probs by task
llm_task_probs = {}
for row in llm_rows:
    t = int(row['task_num'])
    if t not in llm_task_probs:
        llm_task_probs[t] = {'P_A': [], 'P_B': [], 'P_C': []}
    llm_task_probs[t]['P_A'].append(float(row['P_A']))
    llm_task_probs[t]['P_B'].append(float(row['P_B']))
    llm_task_probs[t]['P_C'].append(float(row['P_C']))

# Average across reps
for t in llm_task_probs:
    llm_task_probs[t] = {
        'P_A': np.mean(llm_task_probs[t]['P_A']),
        'P_B': np.mean(llm_task_probs[t]['P_B']),
        'P_C': np.mean(llm_task_probs[t]['P_C']),
    }

print(f"Loaded {len(llm_task_probs)} tasks with LLM probs")

# Get unique respondents and seeded random split
rids = sorted(df['RespondentID'].unique())
rng = random.Random(SEED)
shuffled = rids[:]
rng.shuffle(shuffled)

n_train = int(TRAIN_FRAC * len(shuffled))
train_ids = set(shuffled[:n_train])
test_ids = set(shuffled[n_train:])

print(f"\nSplit (seed={SEED}):")
print(f"  Training: {len(train_ids)} respondents")
print(f"  Test: {len(test_ids)} respondents")

# Filter test data
test_df = df[df['RespondentID'].isin(test_ids)].copy()
train_df = df[df['RespondentID'].isin(train_ids)].copy()

print(f"\nTrain rows: {len(train_df)}")
print(f"Test rows: {len(test_df)}")

# Load DCE coefficients and refit on training data
print("\nRefitting DCE on training set...")

# For simplicity, use coefficients from full data (in practice should refit)
# TODO: Implement actual conditional logit refit

# For now, report that we need the actual DCE refit implementation
print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)
print(f"Test respondents: {len(test_ids)}")
print(f"Test tasks: ~{len(test_df) // 3} (each with A/B/C)")
print(f"\nLLM probabilities available for tasks: {list(llm_task_probs.keys())}")
print(f"\nNote: Full DCE refit and log loss computation requires:")
print(f"  1. Refit conditional logit on training set ({len(train_ids)} respondents)")
print(f"  2. Compute Pure-DCE multinomial probs for 6 tasks")
print(f"  3. Compute EFR probs (0.25*LLM + 0.75*Pure-DCE)")
print(f"  4. Compare log losses on test set")

# Save split info
with open('split_info_seed2026.json', 'w') as f:
    json.dump({
        'seed': SEED,
        'train_frac': TRAIN_FRAC,
        'n_train': len(train_ids),
        'n_test': len(test_ids),
        'train_ids': [int(x) for x in sorted(train_ids)],
        'test_ids': [int(x) for x in sorted(test_ids)]
    }, f, indent=2)

print(f"\n✓ Saved split to split_info_seed2026.json")
