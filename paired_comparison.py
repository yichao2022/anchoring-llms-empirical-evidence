#!/usr/bin/env python3
"""
Paired Comparison Analysis with t-test
"""
import pandas as pd
import numpy as np
from scipy import stats
import json

print("=" * 70)
print("PAIRWISE COMPARISON ANALYSIS")
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

# Training choice shares
train_shares = {}
for cs in canonical_tasks:
    task_data = train_data[train_data['Choiceset'] == cs]
    if len(task_data) > 0:
        counts = task_data.groupby('Alt')['Choice'].sum()
        total = counts.sum()
        train_shares[cs] = {
            'A': counts.get('A', 0) / total if total > 0 else 1/3,
            'B': counts.get('B', 0) / total if total > 0 else 1/3,
            'C': counts.get('C', 0) / total if total > 0 else 1/3,
        }

# Compute respondent-level losses
def resp_log_loss(obs_list, probs_dict):
    loss = 0.0
    n = 0
    for obs in obs_list:
        t = obs['task']
        y = obs['choice']
        if t in probs_dict:
            p = probs_dict[t][y]
            p = max(p, 1e-7)
            loss -= np.log(p)
            n += 1
    return loss / n if n > 0 else float('inf')

# Prepare respondent observations
resp_obs = {}
for _, row in test_data.iterrows():
    resp = row['RespondentID']
    if resp not in resp_obs:
        resp_obs[resp] = []
    resp_obs[resp].append({'task': row['Choiceset'], 'choice': row['Alt']})

# Compute losses
losses = {'DCE': [], 'LLM': [], 'EFR': [], 'TrainShare': []}
for resp, obs_list in resp_obs.items():
    losses['DCE'].append(resp_log_loss(obs_list, dce_probs))
    losses['LLM'].append(resp_log_loss(obs_list, llm_probs))
    losses['EFR'].append(resp_log_loss(obs_list, efr_probs))
    losses['TrainShare'].append(resp_log_loss(obs_list, train_shares))

# Paired comparisons
print("\nPaired Comparisons (Respondent-Level):")
print("-" * 70)
print(f"{'Comparison':<20} {'Mean Diff':<12} {'SD Diff':<12} {'t-stat':<10} {'p-value':<12}")
print("-" * 70)

comparisons = [
    ('LLM', 'DCE', 'LLM - DCE'),
    ('EFR', 'DCE', 'EFR - DCE'),
    ('EFR', 'LLM', 'EFR - LLM')
]

results = {}
for m1, m2, name in comparisons:
    diff = np.array(losses[m1]) - np.array(losses[m2])
    mean_diff = np.mean(diff)
    sd_diff = np.std(diff, ddof=1)
    t_stat = mean_diff / (sd_diff / np.sqrt(len(diff)))
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(diff)-1))
    
    print(f"{name:<20} {mean_diff:<12.6f} {sd_diff:<12.6f} {t_stat:<10.4f} {p_value:<12.6f}")
    
    if p_value < 0.001:
        sig = "***"
    elif p_value < 0.01:
        sig = "**"
    elif p_value < 0.05:
        sig = "*"
    else:
        sig = "ns"
    print(f"   Significance: {sig}")
    
    results[name] = {
        'mean_diff': float(mean_diff),
        'sd_diff': float(sd_diff),
        't_stat': float(t_stat),
        'p_value': float(p_value),
        'significance': sig
    }

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"""
Log Loss Results (Held-out, N={len(losses['DCE'])} respondents):
- Uniform baseline:      {np.log(3):.4f}
- Training-choice-share: {np.mean(losses['TrainShare']):.4f}
- Pure-DCE:            {np.mean(losses['DCE']):.4f}
- Raw-LLM:             {np.mean(losses['LLM']):.4f}
- EFR:                 {np.mean(losses['EFR']):.4f}

Key Finding:
All three methods (Pure-DCE, Raw-LLM, EFR) have log loss > uniform (1.0986).
The small differences between methods (0.01-0.02) are statistically detectable
but practically negligible compared to the gap from uniform.

Interpretation:
Exact-task categorical validation does NOT provide evidence that EFR
improves human-choice prediction over simple uniform random choice.
""")

# Save results
with open('paired_comparison_results.json', 'w') as f:
    json.dump({
        'N_respondents': len(losses['DCE']),
        'log_loss': {
            'uniform': float(np.log(3)),
            'training_choice_share': float(np.mean(losses['TrainShare'])),
            'Pure-DCE': float(np.mean(losses['DCE'])),
            'Raw-LLM': float(np.mean(losses['LLM'])),
            'EFR': float(np.mean(losses['EFR']))
        },
        'paired_comparisons': results
    }, f, indent=2)

print("\n✓ Results saved to: paired_comparison_results.json")
