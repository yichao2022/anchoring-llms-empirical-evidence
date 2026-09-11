#!/usr/bin/env python3
"""
Fixed Bootstrap Analysis: Proper respondent-cluster paired bootstrap
"""
import pandas as pd
import numpy as np
import json
from collections import defaultdict

print("=" * 70)
print("FIXED BOOTSTRAP ANALYSIS")
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

llm_task_probs = defaultdict(lambda: {'P_A': [], 'P_B': [], 'P_C': []})
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

# Prepare respondent-level observations
print("\nPreparing respondent-level data...")

# Group test data by respondent
resp_obs = {}
for _, row in test_data.iterrows():
    resp = row['RespondentID']
    if resp not in resp_obs:
        resp_obs[resp] = []
    resp_obs[resp].append({
        'task': row['Choiceset'],
        'choice': row['Alt']
    })

print(f"   {len(resp_obs)} held-out respondents")

# Compute log loss for a single respondent
def resp_log_loss(obs_list, probs_dict):
    """obs_list: list of {'task': t, 'choice': y}"""
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

# Compute respondent-level losses
print("\nComputing respondent-level losses...")
resp_losses = {'DCE': {}, 'LLM': {}, 'EFR': {}, 'TrainShare': {}}

for resp, obs_list in resp_obs.items():
    resp_losses['DCE'][resp] = resp_log_loss(obs_list, dce_probs)
    resp_losses['LLM'][resp] = resp_log_loss(obs_list, llm_probs)
    resp_losses['EFR'][resp] = resp_log_loss(obs_list, efr_probs)
    resp_losses['TrainShare'][resp] = resp_log_loss(obs_list, train_shares)

# Average losses
print("\nAverage Log Loss (Held-out):")
print(f"   Uniform baseline:      {np.log(3):.6f}")
for method in ['TrainShare', 'LLM', 'EFR', 'DCE']:
    avg = np.mean(list(resp_losses[method].values()))
    print(f"   {method:<15}: {avg:.6f}")

# Respondent-cluster bootstrap
print("\nRunning respondent-cluster bootstrap (2000 reps)...")
N_BOOT = 2000
SEED = 2026
rng = np.random.RandomState(SEED)

resp_list = list(resp_obs.keys())
n_resps = len(resp_list)

boot_diffs = {'LLM-DCE': [], 'EFR-DCE': [], 'EFR-LLM': []}

for b in range(N_BOOT):
    # Sample respondents with replacement
    sample_resps = rng.choice(resp_list, size=n_resps, replace=True)
    
    # Compute average losses for this sample
    losses = {'DCE': [], 'LLM': [], 'EFR': []}
    for resp in sample_resps:
        losses['DCE'].append(resp_losses['DCE'][resp])
        losses['LLM'].append(resp_losses['LLM'][resp])
        losses['EFR'].append(resp_losses['EFR'][resp])
    
    avg_dce = np.mean(losses['DCE'])
    avg_llm = np.mean(losses['LLM'])
    avg_efr = np.mean(losses['EFR'])
    
    boot_diffs['LLM-DCE'].append(avg_llm - avg_dce)
    boot_diffs['EFR-DCE'].append(avg_efr - avg_dce)
    boot_diffs['EFR-LLM'].append(avg_efr - avg_llm)

print("\nPaired Differences (95% CI):")
print("-" * 70)
print(f"{'Difference':<15} {'Mean':<12} {'CI_low':<12} {'CI_high':<12} {'Sig?'}")
print("-" * 70)

for diff_name in ['LLM-DCE', 'EFR-DCE', 'EFR-LLM']:
    values = boot_diffs[diff_name]
    mean = np.mean(values)
    ci_low = np.percentile(values, 2.5)
    ci_high = np.percentile(values, 97.5)
    
    # Significance check
    if ci_low > 0:
        sig = f"{diff_name.split('-')[0]} > {diff_name.split('-')[1]}"
    elif ci_high < 0:
        sig = f"{diff_name.split('-')[0]} < {diff_name.split('-')[1]}"
    else:
        sig = "No sig diff"
    
    print(f"{diff_name:<15} {mean:<12.6f} {ci_low:<12.6f} {ci_high:<12.6f} {sig}")

# Save results
results = {
    'N_train_respondents': int(len(train_ids)),
    'N_test_respondents': int(len(test_ids)),
    'log_loss': {
        'uniform_baseline': float(np.log(3)),
        'training_choice_share': float(np.mean(list(resp_losses['TrainShare'].values()))),
        'Pure-DCE': float(np.mean(list(resp_losses['DCE'].values()))),
        'Raw-LLM': float(np.mean(list(resp_losses['LLM'].values()))),
        'EFR': float(np.mean(list(resp_losses['EFR'].values())))
    },
    'paired_bootstrap_95ci': {
        'LLM-DCE': {
            'mean': float(np.mean(boot_diffs['LLM-DCE'])),
            'ci_low': float(np.percentile(boot_diffs['LLM-DCE'], 2.5)),
            'ci_high': float(np.percentile(boot_diffs['LLM-DCE'], 97.5))
        },
        'EFR-DCE': {
            'mean': float(np.mean(boot_diffs['EFR-DCE'])),
            'ci_low': float(np.percentile(boot_diffs['EFR-DCE'], 2.5)),
            'ci_high': float(np.percentile(boot_diffs['EFR-DCE'], 97.5))
        },
        'EFR-LLM': {
            'mean': float(np.mean(boot_diffs['EFR-LLM'])),
            'ci_low': float(np.percentile(boot_diffs['EFR-LLM'], 2.5)),
            'ci_high': float(np.percentile(boot_diffs['EFR-LLM'], 97.5))
        }
    }
}

with open('fixed_bootstrap_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✓ Results saved to: fixed_bootstrap_results.json")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
In corrected canonical 6-task multinomial validation:
- Raw LLM: {results['log_loss']['Raw-LLM']:.4f}
- EFR: {results['log_loss']['EFR']:.4f}
- Pure-DCE: {results['log_loss']['Pure-DCE']:.4f}
- Training-choice-share: {results['log_loss']['training_choice_share']:.4f}
- Uniform baseline: {results['log_loss']['uniform_baseline']:.4f}

All three methods EXCEED uniform baseline (worse prediction).

Paired bootstrap shows whether differences are significant.
""")
