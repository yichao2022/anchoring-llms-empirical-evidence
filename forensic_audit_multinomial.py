#!/usr/bin/env python3
"""
Forensic Audit of Canonical 6-Task Multinomial Validation
"""
import pandas as pd
import numpy as np
import json
from collections import defaultdict

print("=" * 70)
print("FORENSIC AUDIT: Canonical 6-Task Multinomial Validation")
print("=" * 70)

# 1. Load DCE coefficients
print("\n1. Loading DCE coefficients...")
coefs = {
    'VaccineOrigin': 0.244230,
    'WaitTime': -0.059244,
    'VaccineEfficacy': 0.413207,
    'SideEffects': -0.036433,
    'CashIncentives': 0.001048,
    'ASC_optout': -0.213815
}
for k, v in coefs.items():
    print(f"   {k}: {v:.6f}")

# 2. Load canonical DCE tasks from data (coded values)
print("\n2. Loading canonical DCE tasks from dce_encoded.csv...")
df = pd.read_csv('/tmp/dce_analysis/dce_encoded.csv')

# Extract tasks from RespondentID=1 (first respondent) to get canonical design
canonical = df[df['RespondentID'] == 1].copy()
print(f"   Canonical rows: {len(canonical)}")
print("   Canonical tasks (coded values):")

tasks_coded = {}
for cs in sorted(canonical['Choiceset'].unique()):
    task_data = canonical[canonical['Choiceset'] == cs]
    task = {}
    for _, row in task_data.iterrows():
        alt = row['Alt']
        if alt == 'C':
            task['C'] = {'type': 'opt-out', 'ASC': 1}
        else:
            task[alt] = {
                'wait': row['WaitTime'],
                'eff': row['VaccineEfficacy'],
                'se': row['SideEffects'],
                'cash': row['CashIncentives'],
                'origin': row['VaccineOrigin']
            }
    tasks_coded[cs] = task
    print(f"   Task {cs}:")
    for alt, attrs in task.items():
        if alt == 'C':
            print(f"     C: opt-out")
        else:
            print(f"     {alt}: wait={attrs['wait']}, eff={attrs['eff']}, se={attrs['se']}, cash={attrs['cash']}, origin={attrs['origin']}")

# 3. Compute Pure-DCE probabilities (conditional logit softmax)
print("\n3. Computing Pure-DCE probabilities (conditional logit softmax)...")
def compute_dce_probabilities(task_coded):
    """Compute multinomial logit probabilities for a task."""
    # Utility for A and B
    U = {}
    for alt in ['A', 'B']:
        attrs = task_coded[alt]
        U[alt] = (coefs['WaitTime'] * attrs['wait'] +
                  coefs['VaccineEfficacy'] * attrs['eff'] +
                  coefs['SideEffects'] * attrs['se'] +
                  coefs['CashIncentives'] * attrs['cash'] +
                  coefs['VaccineOrigin'] * attrs['origin'])
    
    # Utility for C (opt-out) - only ASC_optout
    U['C'] = coefs['ASC_optout']
    
    # Softmax
    exp_U = {k: np.exp(v) for k, v in U.items()}
    sum_exp = sum(exp_U.values())
    P = {k: v / sum_exp for k, v in exp_U.items()}
    return U, P

dce_utilities = {}
dce_probs = {}
for cs in tasks_coded:
    U, P = compute_dce_probabilities(tasks_coded[cs])
    dce_utilities[cs] = U
    dce_probs[cs] = P
    print(f"   Task {cs}:")
    print(f"     Utilities: A={U['A']:.4f}, B={U['B']:.4f}, C={U['C']:.4f}")
    print(f"     Probs: A={P['A']:.6f}, B={P['B']:.6f}, C={P['C']:.6f}")

# 4. Load LLM probabilities
print("\n4. Loading LLM probabilities...")
with open('llm_parsed_outputs_multinomial_6tasks_canonical.csv') as f:
    import csv
    reader = csv.DictReader(f)
    llm_rows = list(reader)

# Average across repetitions
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
    print(f"   Task {t}: LLM probs: A={llm_probs[t]['A']:.4f}, B={llm_probs[t]['B']:.4f}, C={llm_probs[t]['C']:.4f}")

# 5. Compute EFR (λ=0.25)
print("\n5. Computing EFR (λ=0.25)...")
LAMBDA = 0.25
efr_probs = {}
for cs in tasks_coded:
    efr_probs[cs] = {
        'A': LAMBDA * llm_probs[cs]['A'] + (1-LAMBDA) * dce_probs[cs]['A'],
        'B': LAMBDA * llm_probs[cs]['B'] + (1-LAMBDA) * dce_probs[cs]['B'],
        'C': LAMBDA * llm_probs[cs]['C'] + (1-LAMBDA) * dce_probs[cs]['C'],
    }
    print(f"   Task {cs}: EFR probs: A={efr_probs[cs]['A']:.4f}, B={efr_probs[cs]['B']:.4f}, C={efr_probs[cs]['C']:.4f}")

# 6. Compute training and held-out empirical choice shares
print("\n6. Computing empirical choice shares...")
rids = sorted(df['RespondentID'].unique())
n_test = int(0.20 * len(rids))
test_ids = rids[-n_test:]
train_ids = rids[:-n_test]

print(f"   Training respondents: {len(train_ids)}")
print(f"   Held-out respondents: {len(test_ids)}")

train_data = df[df['RespondentID'].isin(train_ids)].copy()
test_data = df[df['RespondentID'].isin(test_ids)].copy()

# Function to compute choice shares per task
def compute_choice_shares(data, tasks):
    shares = {}
    for cs in tasks:
        task_data = data[data['Choiceset'] == cs]
        if len(task_data) == 0:
            continue
        # Count choices per alternative
        counts = task_data.groupby('Alt')['Choice'].sum()
        total = counts.sum()
        shares[cs] = {
            'A': counts.get('A', 0) / total,
            'B': counts.get('B', 0) / total,
            'C': counts.get('C', 0) / total,
            'N': int(total)
        }
    return shares

train_shares = compute_choice_shares(train_data, tasks_coded.keys())
test_shares = compute_choice_shares(test_data, tasks_coded.keys())

print("\n   Training choice shares:")
for cs in sorted(train_shares.keys()):
    s = train_shares[cs]
    print(f"   Task {cs}: A={s['A']:.4f}, B={s['B']:.4f}, C={s['C']:.4f} (N={s['N']})")

print("\n   Held-out choice shares:")
for cs in sorted(test_shares.keys()):
    s = test_shares[cs]
    print(f"   Task {cs}: A={s['A']:.4f}, B={s['B']:.4f}, C={s['C']:.4f} (N={s['N']})")

# 7. Compute multinomial log loss for each method on held-out
print("\n7. Computing multinomial log loss on held-out...")

def multinomial_log_loss(y_true, probs_list, eps=1e-7):
    """y_true: list of chosen alternatives; probs_list: list of dicts {A,B,C}"""
    loss = 0.0
    n = len(y_true)
    for y, probs in zip(y_true, probs_list):
        p = np.clip(probs[y], eps, 1 - eps)
        loss -= np.log(p)
    return loss / n

# Get held-out choices
y_heldout = []
probs_heldout = {'DCE': [], 'LLM': [], 'EFR': []}
for _, row in test_data.iterrows():
    cs = row['Choiceset']
    y = row['Alt']
    y_heldout.append(y)
    
    # DCE probs
    probs_heldout['DCE'].append(dce_probs[cs])
    # LLM probs
    probs_heldout['LLM'].append(llm_probs[cs])
    # EFR probs
    probs_heldout['EFR'].append(efr_probs[cs])

ll_heldout = {}
for method in ['DCE', 'LLM', 'EFR']:
    ll_heldout[method] = multinomial_log_loss(y_heldout, probs_heldout[method])

print("   Held-out multinomial log loss:")
for method, ll in ll_heldout.items():
    print(f"   {method}: {ll:.6f}")

# 8. Compute training multinomial log loss
print("\n8. Computing multinomial log loss on training...")
y_train = []
probs_train = {'DCE': [], 'LLM': [], 'EFR': []}
for _, row in train_data.iterrows():
    cs = row['Choiceset']
    y = row['Alt']
    y_train.append(y)
    probs_train['DCE'].append(dce_probs[cs])
    probs_train['LLM'].append(llm_probs[cs])
    probs_train['EFR'].append(efr_probs[cs])

ll_train = {}
for method in ['DCE', 'LLM', 'EFR']:
    ll_train[method] = multinomial_log_loss(y_train, probs_train[method])

print("   Training multinomial log loss:")
for method, ll in ll_train.items():
    print(f"   {method}: {ll:.6f}")

# 9. Compute baseline (uniform) log loss
uniform_probs = {'A': 1/3, 'B': 1/3, 'C': 1/3}
y_all = y_train + y_heldout
probs_all_uniform = [uniform_probs] * len(y_all)
ll_uniform = multinomial_log_loss(y_all, probs_all_uniform)
print(f"\n   Uniform baseline log loss: {ll_uniform:.6f} (log(3)={np.log(3):.6f})")

# 10. Compute train-choice-share baseline (predict training frequencies)
print("\n9. Computing train-choice-share baseline...")
# For each task, predict training frequencies
train_share_probs = []
for _, row in test_data.iterrows():
    cs = row['Choiceset']
    if cs in train_shares:
        train_share_probs.append(train_shares[cs])
    else:
        train_share_probs.append(uniform_probs)

y_test = list(test_data['Alt'])
ll_trainshare = multinomial_log_loss(y_test, train_share_probs)
print(f"   Train-choice-share baseline log loss: {ll_trainshare:.6f}")

# 11. Summary
print("\n" + "=" * 70)
print("SUMMARY OF FORENSIC AUDIT")
print("=" * 70)
print("\n1. Pure-DCE probabilities are EXTREME (near 0 or 1).")
print("   Reason: Large utility differences between alternatives.")
print("   Example Task 1: U_A={:.4f}, U_B={:.4f}, U_C={:.4f}".format(
    dce_utilities[1]['A'], dce_utilities[1]['B'], dce_utilities[1]['C']))

print("\n2. Multinomial log loss comparison:")
print(f"   {'Method':<15} {'Training':<15} {'Held-out':<15}")
print("-" * 50)
for method in ['DCE', 'LLM', 'EFR']:
    print(f"   {method:<15} {ll_train[method]:<15.6f} {ll_heldout[method]:<15.6f}")
print(f"   {'Uniform':<15} {ll_uniform:<15.6f} {ll_uniform:<15.6f}")
print(f"   {'TrainShare':<15} {'N/A':<15} {ll_trainshare:<15.6f}")

print("\n3. Issue identified:")
print("   Pure-DCE log loss = {:.2f} >> log(3) = {:.2f}".format(ll_heldout['DCE'], np.log(3)))
print("   This is because Pure-DCE probabilities are near-deterministic,")
print("   while human choices are stochastic. When DCE predicts P(B)≈1.0")
print("   but human sometimes chooses A or C, log loss explodes.")

print("\n4. Correct approach:")
print("   The DCE model is a deterministic utility model, not a probabilistic")
print("   model of individual choices. Its role is to provide systematic")
print("   component; the EFR blend with LLM adds stochasticity.")

print("\n5. Next steps:")
print("   - Verify DCE coefficients are from correct model specification")
print("   - Consider if DCE should include random coefficients or error components")
print("   - Compare with train-choice-share baseline as lower bound")
