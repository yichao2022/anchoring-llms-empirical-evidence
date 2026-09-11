#!/usr/bin/env python3
"""
Analyze 6 canonical tasks multinomial validation with NEW correct data.
Compare Raw LLM, EFR, and Pure-DCE predictions against held-out human choices.
"""
import csv
import json
import numpy as np
from pathlib import Path
import sys

WORKSPACE = Path("/tmp/behavioral-digital-twins")

# Load LLM predictions
llm_data = []
with open(WORKSPACE / "llm_parsed_outputs_multinomial_6tasks_canonical_NEW.csv", 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        llm_data.append({
            'task_num': int(row['task_num']),
            'rep': int(row['rep']),
            'P_A': float(row['P_A']) if row['P_A'] else 0,
            'P_B': float(row['P_B']) if row['P_B'] else 0,
            'P_C': float(row['P_C']) if row['P_C'] else 0,
        })

print(f"Loaded {len(llm_data)} LLM predictions")

# Aggregate by task (average over 10 reps)
llm_by_task = {}
for row in llm_data:
    task_num = row['task_num']
    if task_num not in llm_by_task:
        llm_by_task[task_num] = {'P_A': [], 'P_B': [], 'P_C': []}
    llm_by_task[task_num]['P_A'].append(row['P_A'])
    llm_by_task[task_num]['P_B'].append(row['P_B'])
    llm_by_task[task_num]['P_C'].append(row['P_C'])

# Average probabilities
for task_num in llm_by_task:
    llm_by_task[task_num]['P_A_mean'] = np.mean(llm_by_task[task_num]['P_A'])
    llm_by_task[task_num]['P_B_mean'] = np.mean(llm_by_task[task_num]['P_B'])
    llm_by_task[task_num]['P_C_mean'] = np.mean(llm_by_task[task_num]['P_C'])

print(f"\nLLM predictions by task:")
for task_num in sorted(llm_by_task.keys()):
    t = llm_by_task[task_num]
    print(f"  Task {task_num}: P(A)={t['P_A_mean']:.3f}, P(B)={t['P_B_mean']:.3f}, P(C)={t['P_C_mean']:.3f}")

# Load DCE data for held-out validation
print("\n" + "="*60)
print("Loading DCE held-out data...")
print("="*60)

# Load the 6 canonical tasks
tasks_6 = []
with open(WORKSPACE / "multinomial_tasks_6.csv", 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        tasks_6.append({
            'task_num': int(row['task_num']),
            'task_id': row['task_id'],
            'A': {'wait': float(row['A_wait']), 'eff': float(row['A_eff']), 
                  'se': float(row['A_se']), 'cash': float(row['A_cash']), 'origin': int(row['A_origin'])},
            'B': {'wait': float(row['B_wait']), 'eff': float(row['B_eff']),
                  'se': float(row['B_se']), 'cash': float(row['B_cash']), 'origin': int(row['B_origin'])},
        })

print(f"Loaded {len(tasks_6)} canonical tasks")

# Load DCE responses
dce_responses = []
with open(WORKSPACE / "analysis_output" / "dce_encoded.csv", 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        dce_responses.append(row)

print(f"Loaded {len(dce_responses)} DCE response rows")

# Map task_id to task_num
task_id_to_num = {t['task_id']: t['task_num'] for t in tasks_6}

# Get held-out split (last 205 respondents)
respondent_ids = sorted(set(int(r['RespondentID']) for r in dce_responses if r['RespondentID']))
held_out_ids = set(respondent_ids[-205:])  # Last 205
print(f"\nHeld-out respondents: {len(held_out_ids)} (IDs {min(held_out_ids)}-{max(held_out_ids)})")

# Aggregate human choices by task
human_choices = {task_num: {'A': 0, 'B': 0, 'C': 0, 'total': 0} 
                 for task_num in range(1, 7)}

for row in dce_responses:
    resp_id = int(row.get('RespondentID', 0))
    if resp_id not in held_out_ids:
        continue
    
    task_id = row.get('Choiceset', '')
    if task_id not in task_id_to_num:
        continue
    
    task_num = task_id_to_num[task_id]
    alt = row.get('Alt', '').strip().upper()
    choice = int(float(row.get('Choice', 0)))
    
    if alt in ['A', 'B', 'C']:
        human_choices[task_num]['total'] += 1
        if choice == 1:
            human_choices[task_num][alt] += 1

print("\nHuman choices by task (held-out):")
for task_num in range(1, 7):
    h = human_choices[task_num]
    if h['total'] > 0:
        p_a = h['A'] / h['total']
        p_b = h['B'] / h['total']
        p_c = h['C'] / h['total']
        print(f"  Task {task_num}: N={h['total']}, P(A)={p_a:.3f}, P(B)={p_b:.3f}, P(C)={p_c:.3f}")
    else:
        print(f"  Task {task_num}: N=0")

# Calculate multinomial log loss
print("\n" + "="*60)
print("CALCULATING MULTINOMIAL LOG LOSS")
print("="*60)

def multinomial_log_loss(probs, choices):
    """Calculate multinomial log loss."""
    losses = []
    for task_num in range(1, 7):
        if task_num not in probs or choices[task_num]['total'] == 0:
            continue
        
        p_a = probs[task_num].get('P_A_mean', probs[task_num].get('P_A', 0.333))
        p_b = probs[task_num].get('P_B_mean', probs[task_num].get('P_B', 0.333))
        p_c = probs[task_num].get('P_C_mean', probs[task_num].get('P_C', 0.333))
        
        # Normalize to sum to 1
        total = p_a + p_b + p_c
        if total > 0:
            p_a, p_b, p_c = p_a/total, p_b/total, p_c/total
        
        h = choices[task_num]
        n = h['total']
        
        # Multinomial log loss: -[n_A*log(p_A) + n_B*log(p_B) + n_C*log(p_C)]
        loss = 0
        if h['A'] > 0 and p_a > 0:
            loss -= h['A'] * np.log(p_a)
        if h['B'] > 0 and p_b > 0:
            loss -= h['B'] * np.log(p_b)
        if h['C'] > 0 and p_c > 0:
            loss -= h['C'] * np.log(p_c)
        
        losses.append(loss)
    
    return np.mean(losses) if losses else 0

# Raw LLM log loss
raw_llm_loss = multinomial_log_loss(llm_by_task, human_choices)
print(f"\nRaw LLM multinomial log loss: {raw_llm_loss:.4f}")

# Uniform baseline (1/3 each)
uniform_probs = {task_num: {'P_A': 1/3, 'P_B': 1/3, 'P_C': 1/3} for task_num in range(1, 7)}
uniform_loss = multinomial_log_loss(uniform_probs, human_choices)
print(f"Uniform (1/3 each) log loss: {uniform_loss:.4f}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Total held-out observations: {sum(human_choices[t]['total'] for t in range(1,7))}")
print(f"Expected: 205 respondents × 6 tasks = 1,230")
print(f"\nMultinomial Log Loss:")
print(f"  Uniform (1/3): {uniform_loss:.4f}")
print(f"  Raw LLM:       {raw_llm_loss:.4f}")

# Save results
results = {
    'uniform_log_loss': uniform_loss,
    'raw_llm_log_loss': raw_llm_loss,
    'held_out_n': sum(human_choices[t]['total'] for t in range(1,7)),
    'held_out_respondents': len(held_out_ids),
    'llm_by_task': {str(k): {kk: float(vv) if isinstance(vv, (int, float, np.number)) else vv 
                              for kk, vv in v.items()} 
                    for k, v in llm_by_task.items()},
    'human_choices': human_choices
}

with open(WORKSPACE / "analysis_6tasks_canonical_NEW.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSaved analysis to: analysis_6tasks_canonical_NEW.json")
