#!/usr/bin/env python3
"""
Final Analysis: Multinomial Log Loss & Brier Score with Bootstrap CI
"""
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.special import expit

# 1. Load data
print("="*70)
print("FINAL ANALYSIS: Multinomial Log Loss & Brier Score")
print("="*70)

print("\n1. Loading data...")
with open('three_method_comparison.json') as f:
    three_method = json.load(f)

df = pd.read_csv('/tmp/dce_analysis/dce_encoded.csv')
print(f"   Total DCE rows: {len(df)}")

# 2. Extract held-out respondents (20% test set)
print("\n2. Extracting held-out respondents...")
rids = sorted(df['RespondentID'].unique())
n_test = int(0.20 * len(rids))
test_ids = rids[-n_test:]
heldout = df[df['RespondentID'].isin(test_ids)].copy()
print(f"   Held-out: {len(test_ids)} respondents, {len(heldout)} rows")

# 3. Prepare multinomial evaluation data
print("\n3. Preparing multinomial evaluation data...")
human_choices = []
for (resp, cs), group in heldout.groupby(['RespondentID', 'Choiceset']):
    chosen = group[group['Choice'] == 1]['Alt'].values
    if len(chosen) > 0:
        alt_map = {'A': 0, 'B': 1, 'C': 2}
        human_choices.append({
            'respondent_id': resp,
            'choiceset': cs,
            'chosen_alt': alt_map.get(chosen[0], -1)
        })

human_df = pd.DataFrame(human_choices)
print(f"   Human choices: {len(human_df)}")
print(f"   A={sum(human_df['chosen_alt']==0)}, B={sum(human_df['chosen_alt']==1)}, C={sum(human_df['chosen_alt']==2)}")

# 4. Map to task probabilities
print("\n4. Mapping to task probabilities...")
probs_list = []
for _, row in human_df.iterrows():
    cs = row['choiceset']
    y = row['chosen_alt']
    if str(cs) in three_method:
        tm = three_method[str(cs)]
        probs_list.append({
            'respondent_id': row['respondent_id'],
            'choiceset': cs,
            'y_true': y,
            **{f'P_{alt}_{method}': tm[method][alt] 
               for method in ['DCE', 'LLM', 'EFR'] 
               for alt in ['A', 'B', 'C']}
        })

probs_df = pd.DataFrame(probs_list)
print(f"   Mapped: {len(probs_df)} observations")

# 5. Compute metrics
print("\n5. Computing metrics...")

def multinomial_log_loss(y_true, probs):
    eps = 1e-7
    probs = np.clip(probs, eps, 1-eps)
    probs = probs / probs.sum(axis=1, keepdims=True)
    n = len(y_true)
    loss = 0.0
    for i in range(n):
        loss -= np.log(probs[i, y_true[i]])
    return loss / n

def multiclass_brier(y_true, probs):
    n = len(y_true)
    brier = 0.0
    for i in range(n):
        y_onehot = np.zeros(3)
        y_onehot[y_true[i]] = 1
        brier += np.sum((probs[i] - y_onehot) ** 2)
    return brier / n

y_true = probs_df['y_true'].values
probs_dce = probs_df[['P_A_DCE', 'P_B_DCE', 'P_C_DCE']].values
probs_llm = probs_df[['P_A_LLM', 'P_B_LLM', 'P_C_LLM']].values
probs_efr = probs_df[['P_A_EFR', 'P_B_EFR', 'P_C_EFR']].values

ll_dce = multinomial_log_loss(y_true, probs_dce)
ll_llm = multinomial_log_loss(y_true, probs_llm)
ll_efr = multinomial_log_loss(y_true, probs_efr)

br_dce = multiclass_brier(y_true, probs_dce)
br_llm = multiclass_brier(y_true, probs_llm)
br_efr = multiclass_brier(y_true, probs_efr)

print(f"\n   Log Loss - DCE: {ll_dce:.6f}, LLM: {ll_llm:.6f}, EFR: {ll_efr:.6f}")
print(f"   Brier    - DCE: {br_dce:.6f}, LLM: {br_llm:.6f}, EFR: {br_efr:.6f}")

# 6. Respondent-cluster bootstrap (2000 reps)
print("\n6. Running respondent-cluster bootstrap (2000 reps)...")

N_BOOT = 2000
SEED = 2026
rng = np.random.RandomState(SEED)

unique_resps = probs_df['respondent_id'].unique()
n_resps = len(unique_resps)

ll_boot = {'DCE': [], 'LLM': [], 'EFR': []}
br_boot = {'DCE': [], 'LLM': [], 'EFR': []}

for b in range(N_BOOT):
    # Sample respondents with replacement
    sample_resps = rng.choice(unique_resps, size=n_resps, replace=True)
    
    # Get all rows for sampled respondents
    mask = probs_df['respondent_id'].isin(sample_resps)
    y_boot = probs_df.loc[mask, 'y_true'].values
    
    if len(y_boot) > 0:
        for method, probs in [('DCE', probs_dce[mask]), ('LLM', probs_llm[mask]), ('EFR', probs_efr[mask])]:
            ll_boot[method].append(multinomial_log_loss(y_boot, probs))
            br_boot[method].append(multiclass_brier(y_boot, probs))

# Compute CIs
def compute_ci(values, ci=95):
    alpha = (100 - ci) / 2
    return {
        'mean': np.mean(values),
        'ci_low': np.percentile(values, alpha),
        'ci_high': np.percentile(values, 100 - alpha)
    }

results = {
    'N_observations': int(len(probs_df)),
    'N_respondents': int(probs_df['respondent_id'].nunique()),
    'multinomial_log_loss': {
        method: {
            'point': float({'DCE': ll_dce, 'LLM': ll_llm, 'EFR': ll_efr}[method]),
            **{k: float(v) for k, v in compute_ci(ll_boot[method]).items()}
        }
        for method in ['DCE', 'LLM', 'EFR']
    },
    'multiclass_brier': {
        method: {
            'point': float({'DCE': br_dce, 'LLM': br_llm, 'EFR': br_efr}[method]),
            **{k: float(v) for k, v in compute_ci(br_boot[method]).items()}
        }
        for method in ['DCE', 'LLM', 'EFR']
    }
}

# 7. Print final results
print("\n" + "="*70)
print("FINAL RESULTS (with 95% CI)")
print("="*70)

print("\nMultinomial Log Loss:")
for method in ['DCE', 'LLM', 'EFR']:
    r = results['multinomial_log_loss'][method]
    print(f"  {method:<10}: {r['point']:.6f} [{r['ci_low']:.6f}, {r['ci_high']:.6f}]")

print("\nMulticlass Brier:")
for method in ['DCE', 'LLM', 'EFR']:
    r = results['multiclass_brier'][method]
    print(f"  {method:<10}: {r['point']:.6f} [{r['ci_low']:.6f}, {r['ci_high']:.6f}]")

# 8. Save
with open('final_metrics_bootstrap.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✓ Results saved to: final_metrics_bootstrap.json")
