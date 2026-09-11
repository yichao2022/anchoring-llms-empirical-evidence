#!/usr/bin/env python3
"""
Two null distributions for Spearman ρ on the 64-state grid:
1. Uniform-noise null: replace LLM with U(0,1)
2. Permutation null: shuffle actual LLM probs across states
"""
import json, csv, numpy as np
from scipy.stats import spearmanr
from collections import defaultdict

LAMBDA = 0.25
N_NULL = 10000

# ── P_emp from 6-param conditional logit ──
coefs = {
    'WaitTime': -0.059244, 'VaccineEfficacy': 0.413207,
    'SideEffects': -0.036433, 'CashIncentives': 0.001048,
    'VaccineOrigin': 0.244230, 'ASC_optout': -0.213815,
}
wait_levels = [0, 2, 4, 6]
eff_levels = [0.3, 0.5, 0.7, 0.9]
se_levels = [0, 1, 2, 3]

P_emp_64 = []
grid_key = []  # (wait, eff, se) tuples in order
for w in wait_levels:
    for e in eff_levels:
        for s in se_levels:
            U_A = coefs['WaitTime']*w + coefs['VaccineEfficacy']*e + coefs['SideEffects']*s
            p_emp = np.exp(U_A) / (np.exp(U_A) + np.exp(coefs['ASC_optout']))
            P_emp_64.append(p_emp)
            grid_key.append((w, e, s))

P_emp_64 = np.array(P_emp_64)
grid_key_to_idx = {k: i for i, k in enumerate(grid_key)}

# ── Load LLM probs from parsed CSV ──
def load_llm_parsed(filepath):
    state_probs = defaultdict(list)
    with open(filepath) as f:
        for row in csv.DictReader(f):
            w, e, s = float(row['wait']), float(row['eff']), float(row['se'])
            key = (w, e, s)
            p = float(row['probability_0_1'])
            state_probs[key].append(p)
    # Average per state, map to 64-state order
    llm = np.zeros(64)
    for key, idx in grid_key_to_idx.items():
        if key in state_probs:
            llm[idx] = np.mean(state_probs[key])
    return llm

# ── Compute nulls for each model ──
models = {
    'Qwen2.5-72B': 'llm_parsed_outputs_qwen72b_unconstrained.csv',
    'DeepSeek V4': 'llm_parsed_outputs_deepseek_unconstrained.csv',
    'MiroThinker': 'llm_parsed_outputs_mirothinker_unconstrained.csv',
}

np.random.seed(2026)
all_results = {}

for name, fpath in models.items():
    llm = load_llm_parsed(fpath)
    p_efr = LAMBDA * llm + (1 - LAMBDA) * P_emp_64
    rho_obs = spearmanr(p_efr, P_emp_64).statistic

    # Uniform null
    uni_rho = []
    for _ in range(N_NULL):
        U = np.random.uniform(0, 1, 64)
        rho_n = spearmanr(LAMBDA*U + (1-LAMBDA)*P_emp_64, P_emp_64).statistic
        uni_rho.append(rho_n)
    uni_rho = np.array(uni_rho)

    # Permutation null
    perm_rho = []
    for _ in range(N_NULL):
        perm = np.random.permutation(llm)
        rho_n = spearmanr(LAMBDA*perm + (1-LAMBDA)*P_emp_64, P_emp_64).statistic
        perm_rho.append(rho_n)
    perm_rho = np.array(perm_rho)

    all_results[name] = {
        'observed_rho': float(rho_obs),
        'uniform': {
            'mean': float(uni_rho.mean()), 'sd': float(uni_rho.std()),
            'ci95': [float(np.percentile(uni_rho, 2.5)), float(np.percentile(uni_rho, 97.5))],
            'p': float((uni_rho >= rho_obs).mean()),
        },
        'permutation': {
            'mean': float(perm_rho.mean()), 'sd': float(perm_rho.std()),
            'ci95': [float(np.percentile(perm_rho, 2.5)), float(np.percentile(perm_rho, 97.5))],
            'p': float((perm_rho >= rho_obs).mean()),
        },
    }

    print(f"\n{'='*55}")
    print(f"{name}")
    print(f"  Observed EFR ρ = {rho_obs:.4f}")
    print(f"  Uniform null:   mean={uni_rho.mean():.4f}, 95%CI=[{np.percentile(uni_rho,2.5):.4f}, {np.percentile(uni_rho,97.5):.4f}], p={all_results[name]['uniform']['p']:.4f}")
    print(f"  Permutation:    mean={perm_rho.mean():.4f}, 95%CI=[{np.percentile(perm_rho,2.5):.4f}, {np.percentile(perm_rho,97.5):.4f}], p={all_results[name]['permutation']['p']:.4f}")

with open('results/null_comparison_3models.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved results/null_comparison_3models.json")
