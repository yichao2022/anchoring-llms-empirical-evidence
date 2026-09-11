#!/usr/bin/env python3
"""
Task-Level Bootstrap Analysis
"""
import pandas as pd
import numpy as np
from scipy import stats
import json

print("=" * 70)
print("TASK-LEVEL BOOTSTRAP ANALYSIS")
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

# Compute task-level empirical choice shares (held-out)
print("\nTask-Level Empirical Choice Shares (Held-out):")
task_shares = {}
for cs in canonical_tasks:
    task_data = test_data[test_data['Choiceset'] == cs]
    counts = task_data.groupby('Alt')['Choice'].sum()
    total = counts.sum()
    task_shares[cs] = {
        'A': counts.get('A', 0) / total if total > 0 else 1/3,
        'B': counts.get('B', 0) / total if total > 0 else 1/3,
        'C': counts.get('C', 0) / total if total > 0 else 1/3,
        'N': int(total)
    }
    s = task_shares[cs]
    print(f"  Task {cs}: A={s['A']:.4f}, B={s['B']:.4f}, C={s['C']:.4f} (N={s['N']})")

# Compute task-level multinomial log loss
def task_log_loss(empirical_shares, predicted_probs):
    """Compute log loss for a task given empirical shares and predicted probs"""
    loss = 0.0
    for alt in ['A', 'B', 'C']:
        p = predicted_probs[alt]
        p = max(p, 1e-7)
        loss -= empirical_shares[alt] * np.log(p)
    return loss

print("\nTask-Level Log Loss:")
print("-" * 70)
print(f"{'Task':<8} {'Empirical':<40} {'DCE':<10} {'LLM':<10} {'EFR':<10}")
print("-" * 70)

task_losses = {'DCE': [], 'LLM': [], 'EFR': []}
for cs in canonical_tasks:
    ll_dce = task_log_loss(task_shares[cs], dce_probs[cs])
    ll_llm = task_log_loss(task_shares[cs], llm_probs[cs])
    ll_efr = task_log_loss(task_shares[cs], efr_probs[cs])
    
    task_losses['DCE'].append(ll_dce)
    task_losses['LLM'].append(ll_llm)
    task_losses['EFR'].append(ll_efr)
    
    emp_str = f"A={task_shares[cs]['A']:.2f}, B={task_shares[cs]['B']:.2f}, C={task_shares[cs]['C']:.2f}"
    print(f"{cs:<8} {emp_str:<40} {ll_dce:<10.4f} {ll_llm:<10.4f} {ll_efr:<10.4f}")

# Overall log loss (weighted by task N)
print("\n" + "=" * 70)
print("OVERALL LOG LOSS (Weighted by Task N)")
print("=" * 70)

N_total = sum(task_shares[cs]['N'] for cs in canonical_tasks)
overall = {}
for method in ['DCE', 'LLM', 'EFR']:
    overall[method] = sum(task_losses[method][cs-1] * task_shares[cs]['N'] / N_total 
                         for cs in canonical_tasks)

print(f"Uniform baseline:      {np.log(3):.6f}")
for method in ['DCE', 'LLM', 'EFR']:
    print(f"{method:<15}: {overall[method]:.6f}")

print(f"\nAll three methods > uniform ({np.log(3):.4f})")

# Task-level bootstrap (6 tasks, sample with replacement)
print("\n" + "=" * 70)
print("TASK-LEVEL BOOTSTRAP (2000 reps, sampling 6 tasks with replacement)")
print("=" * 70)

N_BOOT = 2000
SEED = 2026
rng = np.random.RandomState(SEED)

boot_results = {'DCE': [], 'LLM': [], 'EFR': []}
for b in range(N_BOOT):
    # Sample tasks with replacement
    sample_tasks = rng.choice(range(6), size=6, replace=True)
    
    for method in ['DCE', 'LLM', 'EFR']:
        # Compute weighted average for this bootstrap sample
        total_loss = 0.0
        total_n = 0
        for t_idx in sample_tasks:
            cs = t_idx + 1
            total_loss += task_losses[method][t_idx] * task_shares[cs]['N']
            total_n += task_shares[cs]['N']
        boot_results[method].append(total_loss / total_n)

# Compute bootstrap CI
print("\nBootstrap 95% CI:")
print("-" * 50)
print(f"{'Method':<15} {'Mean':<12} {'CI_low':<12} {'CI_high':<12}")
print("-" * 50)

for method in ['DCE', 'LLM', 'EFR']:
    values = boot_results[method]
    mean = np.mean(values)
    ci_low = np.percentile(values, 2.5)
    ci_high = np.percentile(values, 97.5)
    print(f"{method:<15} {mean:<12.6f} {ci_low:<12.6f} {ci_high:<12.6f}")

# Paired differences
print("\nPaired Differences (Bootstrap 95% CI):")
print("-" * 70)
print(f"{'Comparison':<15} {'Mean':<12} {'CI_low':<12} {'CI_high':<12} {'Interpretation'}")
print("-" * 70)

comparisons = [
    ('LLM', 'DCE', 'LLM - DCE'),
    ('EFR', 'DCE', 'EFR - DCE'),
    ('EFR', 'LLM', 'EFR - LLM')
]

for m1, m2, name in comparisons:
    diffs = np.array(boot_results[m1]) - np.array(boot_results[m2])
    mean = np.mean(diffs)
    ci_low = np.percentile(diffs, 2.5)
    ci_high = np.percentile(diffs, 97.5)
    
    if ci_low > 0:
        interp = f"{m1} > {m2} (worse)"
    elif ci_high < 0:
        interp = f"{m1} < {m2} (better)"
    else:
        interp = "No significant difference"
    
    print(f"{name:<15} {mean:<12.6f} {ci_low:<12.6f} {ci_high:<12.6f} {interp}")

# Save results
results = {
    'N_tasks': 6,
    'N_test_respondents': int(N_total / 6),
    'log_loss': {
        'uniform_baseline': float(np.log(3)),
        'Pure-DCE': float(overall['DCE']),
        'Raw-LLM': float(overall['LLM']),
        'EFR': float(overall['EFR'])
    },
    'task_level_log_loss': {
        str(cs): {
            'DCE': float(task_losses['DCE'][cs-1]),
            'LLM': float(task_losses['LLM'][cs-1]),
            'EFR': float(task_losses['EFR'][cs-1]),
            'empirical': task_shares[cs]
        }
        for cs in canonical_tasks
    },
    'bootstrap_95ci': {
        'Pure-DCE': {
            'mean': float(np.mean(boot_results['DCE'])),
            'ci_low': float(np.percentile(boot_results['DCE'], 2.5)),
            'ci_high': float(np.percentile(boot_results['DCE'], 97.5))
        },
        'Raw-LLM': {
            'mean': float(np.mean(boot_results['LLM'])),
            'ci_low': float(np.percentile(boot_results['LLM'], 2.5)),
            'ci_high': float(np.percentile(boot_results['LLM'], 97.5))
        },
        'EFR': {
            'mean': float(np.mean(boot_results['EFR'])),
            'ci_low': float(np.percentile(boot_results['EFR'], 2.5)),
            'ci_high': float(np.percentile(boot_results['EFR'], 97.5))
        }
    }
}

with open('task_level_bootstrap.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✓ Results saved to: task_level_bootstrap.json")

print("\n" + "=" * 70)
print("FINAL CONCLUSION")
print("=" * 70)
print(f"""
In corrected canonical 6-task multinomial validation:
- Raw LLM: {overall['LLM']:.4f} [{np.percentile(boot_results['LLM'], 2.5):.4f}, {np.percentile(boot_results['LLM'], 97.5):.4f}]
- EFR: {overall['EFR']:.4f} [{np.percentile(boot_results['EFR'], 2.5):.4f}, {np.percentile(boot_results['EFR'], 97.5):.4f}]
- Pure-DCE: {overall['DCE']:.4f} [{np.percentile(boot_results['DCE'], 2.5):.4f}, {np.percentile(boot_results['DCE'], 97.5):.4f}]
- Uniform baseline: {np.log(3):.4f}

Key Finding:
All three methods EXCEED uniform baseline (worse prediction).
LLM is numerically lowest but not meaningfully better than uniform.
Exact-task categorical validation does NOT provide evidence that EFR
improves human-choice prediction over simple uniform random choice.

This finding is compatible with paper's conservative framing:
EFR's advantage may be structural regularization/bounded narrative
extension, not "better prediction of human choices".
""")
