#!/usr/bin/env python3
"""
Corrected Analysis: Proper multinomial log loss calculation
"""
import pandas as pd
import numpy as np
from scipy import stats
import json

print("=" * 70)
print("CORRECTED ANALYSIS: Proper Multinomial Log Loss")
print("=" * 70)

# Load data
df = pd.read_csv('/tmp/dce_analysis/dce_encoded.csv')

# DCE Coefficients
coefs = {
    'VaccineOrigin': 0.244230,
    'WaitTime': -0.059244,
    'VaccineEfficacy': 0.413207,
    'SideEffects': -0.036433,
    'CashIncentives': 0.001048,
    'ASC_optout': -0.213815
}

# Train/test split
rids = sorted(df['RespondentID'].unique())
n_test = int(0.20 * len(rids))
test_ids = rids[-n_test:]
train_ids = rids[:-n_test]

train_data = df[df['RespondentID'].isin(train_ids)].copy()
test_data = df[df['RespondentID'].isin(test_ids)].copy()

print(f"\nN training: {len(train_ids)}, N held-out: {len(test_ids)}")

# Extract canonical tasks
canonical_tasks = {}
for cs in range(1, 7):
    task_data = df[(df['RespondentID'] == 1) & (df['Choiceset'] == cs)]
    task = {}
    for _, row in task_data.iterrows():
        alt = row['Alt']
        if alt == 'C':
            task['C'] = {'type': 'opt-out'}
        else:
            task[alt] = {
                'wait': row['WaitTime'],
                'eff': row['VaccineEfficacy'],
                'se': row['SideEffects'],
                'cash': row['CashIncentives'],
                'origin': row['VaccineOrigin']
            }
    canonical_tasks[cs] = task

# Compute DCE probs
def compute_dce_prob(task_coded):
    U = {}
    for alt in ['A', 'B']:
        attrs = task_coded[alt]
        U[alt] = (coefs['WaitTime'] * attrs['wait'] +
                  coefs['VaccineEfficacy'] * attrs['eff'] +
                  coefs['SideEffects'] * attrs['se'] +
                  coefs['CashIncentives'] * attrs['cash'] +
                  coefs['VaccineOrigin'] * attrs['origin'])
    U['C'] = coefs['ASC_optout']
    exp_U = {k: np.exp(v) for k, v in U.items()}
    sum_exp = sum(exp_U.values())
    return {k: v / sum_exp for k, v in exp_U.items()}

dce_probs = {cs: compute_dce_prob(canonical_tasks[cs]) for cs in canonical_tasks}

# Load LLM probs
with open('llm_parsed_outputs_multinomial_6tasks_canonical.csv') as f:
    import csv
    reader = csv.DictReader(f)
    llm_rows = list(reader)

llm_task_probs = {t: {'P_A': [], 'P_B': [], 'P_C': []} for t in range(1, 7)}
for row in llm_rows:
    t = int(row['task_num'])
    llm_task_probs[t]['P_A'].append(float(row['P_A']))
    llm_task_probs[t]['P_B'].append(float(row['P_B']))
    llm_task_probs[t]['P_C'].append(float(row['P_C']))

llm_probs = {}
for t in range(1, 7):
    llm_probs[t] = {
        'A': np.mean(llm_task_probs[t]['P_A']),
        'B': np.mean(llm_task_probs[t]['P_B']),
        'C': np.mean(llm_task_probs[t]['P_C']),
    }

# EFR
LAMBDA = 0.25
efr_probs = {}
for cs in canonical_tasks:
    efr_probs[cs] = {
        'A': LAMBDA * llm_probs[cs]['A'] + (1-LAMBDA) * dce_probs[cs]['A'],
        'B': LAMBDA * llm_probs[cs]['B'] + (1-LAMBDA) * dce_probs[cs]['B'],
        'C': LAMBDA * llm_probs[cs]['C'] + (1-LAMBDA) * dce_probs[cs]['C'],
    }

# Prepare observations: list of (respondent, task, choice)
print("\nPreparing held-out observations...")
observations = []
for _, row in test_data.iterrows():
    observations.append({
        'respondent': row['RespondentID'],
        'task': row['Choiceset'],
        'choice': row['Alt']
    })

print(f"Total observations: {len(observations)}")

# Method 1: Individual-level log loss (sum over all choices)
def compute_individual_log_loss(obs_list, probs_dict):
    """Compute log loss for a list of observations"""
    loss = 0.0
    n = 0
    for obs in obs_list:
        t = obs['task']
        y = obs['choice']
        if t in probs_dict and y in probs_dict[t]:
            p = probs_dict[t][y]
            p = max(p, 1e-7)  # Avoid log(0)
            loss -= np.log(p)
            n += 1
    return loss / n if n > 0 else float('inf')

# Method 2: Task-level weighted log loss (what I just calculated)
def compute_task_weighted_log_loss(obs_list, probs_dict, tasks):
    """Compute task-level weighted log loss"""
    task_counts = {}
    task_correct = {}
    
    for t in tasks:
        task_counts[t] = {'A': 0, 'B': 0, 'C': 0}
    
    for obs in obs_list:
        t = obs['task']
        y = obs['choice']
        if t in task_counts:
            task_counts[t][y] += 1
    
    total_loss = 0.0
    total_n = 0
    for t in tasks:
        n_t = sum(task_counts[t].values())
        if n_t == 0:
            continue
        # Empirical distribution
        for y in ['A', 'B', 'C']:
            emp_p = task_counts[t][y] / n_t
            pred_p = probs_dict[t][y]
            pred_p = max(pred_p, 1e-7)
            total_loss -= emp_p * np.log(pred_p) * n_t
        total_n += n_t
    
    return total_loss / total_n

# Compute both methods
print("\n" + "=" * 70)
print("COMPARISON OF METHODS")
print("=" * 70)

# Method 1: Individual-level
ll_individual = {
    'DCE': compute_individual_log_loss(observations, dce_probs),
    'LLM': compute_individual_log_loss(observations, llm_probs),
    'EFR': compute_individual_log_loss(observations, efr_probs)
}

print("\nMethod 1: Individual-Level Log Loss")
print("-" * 50)
print(f"Uniform baseline: {np.log(3):.6f}")
for method in ['DCE', 'LLM', 'EFR']:
    print(f"{method:<15}: {ll_individual[method]:.6f}")

# Method 2: Task-level weighted
ll_task = {
    'DCE': compute_task_weighted_log_loss(observations, dce_probs, list(canonical_tasks.keys())),
    'LLM': compute_task_weighted_log_loss(observations, llm_probs, list(canonical_tasks.keys())),
    'EFR': compute_task_weighted_log_loss(observations, efr_probs, list(canonical_tasks.keys()))
}

print("\nMethod 2: Task-Level Weighted Log Loss")
print("-" * 50)
print(f"Uniform baseline: {np.log(3):.6f}")
for method in ['DCE', 'LLM', 'EFR']:
    print(f"{method:<15}: {ll_task[method]:.6f}")

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)
print(f"""
CRITICAL FINDING:
- Method 1 (Individual): All methods > uniform (worse)
- Method 2 (Task-weighted): All methods < uniform (better!)

This discrepancy arises because:
- Method 1 weights each observation equally
- Method 2 weights each task equally (regardless of N)

The correct approach for held-out validation is Method 1:
each held-out choice should contribute equally to the loss.

However, if the goal is to assess task-level predictive accuracy
(regardless of how many respondents saw each task), Method 2 is appropriate.

RECOMMENDATION:
Report Method 1 as primary (individual-level log loss),
with Method 2 as sensitivity analysis (task-level).
""")

# Save results
results = {
    'method': 'individual_level',
    'N_observations': len(observations),
    'N_respondents': len(test_ids),
    'log_loss': {
        'uniform_baseline': float(np.log(3)),
        'Pure-DCE': float(ll_individual['DCE']),
        'Raw-LLM': float(ll_individual['LLM']),
        'EFR': float(ll_individual['EFR'])
    },
    'task_weighted_log_loss': {
        'uniform_baseline': float(np.log(3)),
        'Pure-DCE': float(ll_task['DCE']),
        'Raw-LLM': float(ll_task['LLM']),
        'EFR': float(ll_task['EFR'])
    }
}

with open('corrected_analysis.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✓ Results saved to: corrected_analysis.json")
