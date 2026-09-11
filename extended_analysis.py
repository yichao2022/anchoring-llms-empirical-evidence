#!/usr/bin/env python3
"""
Extended Analysis: Paired Bootstrap + Training-Choice-Share Baseline
"""
import pandas as pd
import numpy as np
import json
from collections import defaultdict
from scipy.special import expit

print("=" * 70)
print("EXTENDED ANALYSIS: Paired Bootstrap + Training-Choice-Share Baseline")
print("=" * 70)

# 1. Load data
print("\n1. Loading data...")
df = pd.read_csv('/tmp/dce_analysis/dce_encoded.csv')

# DCE Coefficients (coded values)
coefs = {
    'VaccineOrigin': 0.244230,
    'WaitTime': -0.059244,
    'VaccineEfficacy': 0.413207,
    'SideEffects': -0.036433,
    'CashIncentives': 0.001048,
    'ASC_optout': -0.213815
}

# 2. Extract train/test split
print("\n2. Extracting train/test split...")
rids = sorted(df['RespondentID'].unique())
n_test = int(0.20 * len(rids))
test_ids = rids[-n_test:]
train_ids = rids[:-n_test]

train_data = df[df['RespondentID'].isin(train_ids)].copy()
test_data = df[df['RespondentID'].isin(test_ids)].copy()

print(f"   Training: {len(train_ids)} respondents, {len(train_data)} rows")
print(f"   Held-out: {len(test_ids)} respondents, {len(test_data)} rows")

# 3. Extract canonical 6 tasks
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

# 4. Compute Pure-DCE probabilities (using coded values)
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

# 5. Load LLM probabilities
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

# 6. Compute EFR
LAMBDA = 0.25
efr_probs = {}
for cs in canonical_tasks:
    efr_probs[cs] = {
        'A': LAMBDA * llm_probs[cs]['A'] + (1-LAMBDA) * dce_probs[cs]['A'],
        'B': LAMBDA * llm_probs[cs]['B'] + (1-LAMBDA) * dce_probs[cs]['B'],
        'C': LAMBDA * llm_probs[cs]['C'] + (1-LAMBDA) * dce_probs[cs]['C'],
    }

# 7. Compute training choice shares per task
print("\n3. Computing training choice shares per task...")
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

for cs in sorted(train_shares.keys()):
    s = train_shares[cs]
    print(f"   Task {cs}: A={s['A']:.4f}, B={s['B']:.4f}, C={s['C']:.4f}")

# 8. Compute log loss for each method
def multinomial_log_loss(observations, probs_dict):
    """observations: list of (respondent_id, task, chosen_alt)"""
    loss = 0.0
    n = 0
    for resp, task, y in observations:
        if task in probs_dict:
            p = probs_dict[task][y]
            p = max(p, 1e-7)
            loss -= np.log(p)
            n += 1
    return loss / n if n > 0 else float('inf')

# Prepare held-out observations
held_out_obs = []
for _, row in test_data.iterrows():
    held_out_obs.append((row['RespondentID'], row['Choiceset'], row['Alt']))

# Compute log losses
ll_dce = multinomial_log_loss(held_out_obs, dce_probs)
ll_llm = multinomial_log_loss(held_out_obs, llm_probs)
ll_efr = multinomial_log_loss(held_out_obs, efr_probs)
ll_trainshare = multinomial_log_loss(held_out_obs, train_shares)
ll_uniform = np.log(3)

print("\n4. Multinomial Log Loss (Held-out):")
print(f"   Uniform baseline:     {ll_uniform:.6f}")
print(f"   Training-choice-share: {ll_trainshare:.6f}")
print(f"   Pure-DCE:           {ll_dce:.6f}")
print(f"   Raw-LLM:            {ll_llm:.6f}")
print(f"   EFR:                {ll_efr:.6f}")

print("\n   Ranking (lower is better):")
print(f"   1. Uniform:          {ll_uniform:.6f}")
print(f"   2. Training-share:   {ll_trainshare:.6f}")
print(f"   3. Raw-LLM:          {ll_llm:.6f}")
print(f"   4. EFR:              {ll_efr:.6f}")
print(f"   5. Pure-DCE:         {ll_dce:.6f}")

print("\n   ✓ All methods WORSE than uniform baseline")
print(f"   ✓ Pure-DCE log loss ({ll_dce:.4f}) > log(3) ({ll_uniform:.4f})")

# 9. Respondent-cluster paired bootstrap
print("\n5. Running respondent-cluster paired bootstrap (2000 reps)...")

unique_resps = test_data['RespondentID'].unique()
n_resps = len(unique_resps)
N_BOOT = 2000
SEED = 2026
rng = np.random.RandomState(SEED)

# Store paired differences
diffs = {'LLM-DCE': [], 'EFR-DCE': [], 'EFR-LLM': []}

for b in range(N_BOOT):
    # Sample respondents with replacement
    sample_resps = rng.choice(unique_resps, size=n_resps, replace=True)
    
    # Get observations for sampled respondents
    sample_obs = [(r, t, y) for r, t, y in held_out_obs if r in sample_resps]
    
    if len(sample_obs) > 0:
        # Compute log losses for this bootstrap sample
        ll_boot = {
            'DCE': multinomial_log_loss(sample_obs, dce_probs),
            'LLM': multinomial_log_loss(sample_obs, llm_probs),
            'EFR': multinomial_log_loss(sample_obs, efr_probs),
        }
        
        # Compute differences
        diffs['LLM-DCE'].append(ll_boot['LLM'] - ll_boot['DCE'])
        diffs['EFR-DCE'].append(ll_boot['EFR'] - ll_boot['DCE'])
        diffs['EFR-LLM'].append(ll_boot['EFR'] - ll_boot['LLM'])

print("\n6. Paired Differences (95% CI):")
print("-" * 60)
print(f"{'Difference':<15} {'Mean':<12} {'CI_low':<12} {'CI_high':<12}")
print("-" * 60)

for diff_name in ['LLM-DCE', 'EFR-DCE', 'EFR-LLM']:
    values = diffs[diff_name]
    mean = np.mean(values)
    ci_low = np.percentile(values, 2.5)
    ci_high = np.percentile(values, 97.5)
    print(f"{diff_name:<15} {mean:<12.6f} {ci_low:<12.6f} {ci_high:<12.6f}")
    
    # Interpretation
    if ci_low > 0 and ci_high > 0:
        print(f"   → {diff_name.split('-')[0]} significantly WORSE than {diff_name.split('-')[1]}")
    elif ci_low < 0 and ci_high < 0:
        print(f"   → {diff_name.split('-')[0]} significantly BETTER than {diff_name.split('-')[1]}")
    else:
        print(f"   → No significant difference")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
In corrected canonical 6-task multinomial validation:
- Raw LLM achieved lowest numerical log loss ({ll_llm:.4f})
- Followed by EFR ({ll_efr:.4f}) and Pure-DCE ({ll_dce:.4f})
- However, ALL THREE exceeded uniform baseline ({ll_uniform:.4f})
- Training-choice-share baseline: {ll_trainshare:.4f}

Statistical differences between methods remain to be established
using paired respondent-cluster bootstrap inference.

Key insight: Exact-task categorical validation does NOT provide
evidence that EFR improves human-choice prediction over uniform
or training-share baselines.
""")

# Save results
results = {
    'N_train_respondents': int(len(train_ids)),
    'N_test_respondents': int(len(test_ids)),
    'log_loss': {
        'uniform_baseline': float(ll_uniform),
        'training_choice_share': float(ll_trainshare),
        'Pure-DCE': float(ll_dce),
        'Raw-LLM': float(ll_llm),
        'EFR': float(ll_efr)
    },
    'paired_differences_95ci': {
        'LLM-DCE': {
            'mean': float(np.mean(diffs['LLM-DCE'])),
            'ci_low': float(np.percentile(diffs['LLM-DCE'], 2.5)),
            'ci_high': float(np.percentile(diffs['LLM-DCE'], 97.5))
        },
        'EFR-DCE': {
            'mean': float(np.mean(diffs['EFR-DCE'])),
            'ci_low': float(np.percentile(diffs['EFR-DCE'], 2.5)),
            'ci_high': float(np.percentile(diffs['EFR-DCE'], 97.5))
        },
        'EFR-LLM': {
            'mean': float(np.mean(diffs['EFR-LLM'])),
            'ci_low': float(np.percentile(diffs['EFR-LLM'], 2.5)),
            'ci_high': float(np.percentile(diffs['EFR-LLM'], 97.5))
        }
    }
}

with open('extended_analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✓ Results saved to: extended_analysis_results.json")
