#!/usr/bin/env python3
"""
Compare Pure-DCE / Raw LLM / EFR on canonical 6 tasks multinomial validation.
Computes multinomial log loss and multiclass Brier score with bootstrap CI.
"""
from __future__ import annotations

import csv
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Constants
LAMBDA = 0.25  # EFR mixing parameter
N_BOOTSTRAP = 2000
SEED = 2026


def load_llm_results(path: Path) -> dict:
    """Load LLM multinomial results and aggregate by task."""
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Aggregate by task
    task_probs = defaultdict(lambda: {'P_A': [], 'P_B': [], 'P_C': []})
    for row in rows:
        task = int(row['task_num'])
        task_probs[task]['P_A'].append(float(row['P_A']))
        task_probs[task]['P_B'].append(float(row['P_B']))
        task_probs[task]['P_C'].append(float(row['P_C']))
    
    # Average across repetitions
    llm_probs = {}
    for task, probs in task_probs.items():
        llm_probs[task] = {
            'P_A': np.mean(probs['P_A']),
            'P_B': np.mean(probs['P_B']),
            'P_C': np.mean(probs['P_C']),
        }
    
    return llm_probs


def load_held_out_data(path: Path) -> pd.DataFrame:
    """Load held-out DCE data."""
    return pd.read_csv(path)


def multinomial_log_loss(y_true: np.ndarray, probs: np.ndarray, eps: float = 1e-7) -> float:
    """
    y_true: array of shape (n,) with values 0, 1, 2 (for A, B, C)
    probs: array of shape (n, 3) with P(A), P(B), P(C)
    """
    probs = np.clip(probs, eps, 1 - eps)
    probs = probs / probs.sum(axis=1, keepdims=True)  # normalize
    n = len(y_true)
    loss = 0.0
    for i in range(n):
        loss -= np.log(probs[i, y_true[i]])
    return loss / n


def multiclass_brier(y_true: np.ndarray, probs: np.ndarray) -> float:
    """
    y_true: array of shape (n,) with values 0, 1, 2
    probs: array of shape (n, 3)
    """
    n = len(y_true)
    brier = 0.0
    for i in range(n):
        # One-hot encode true label
        y_onehot = np.zeros(3)
        y_onehot[y_true[i]] = 1
        brier += np.sum((probs[i] - y_onehot) ** 2)
    return brier / n


def respondent_cluster_bootstrap(y_true: np.ndarray, probs: dict, 
                                 respondent_ids: np.ndarray,
                                 n_bootstrap: int = 2000,
                                 seed: int = 2026) -> dict:
    """
    Bootstrap at respondent cluster level.
    Returns CI for log_loss and brier for each method.
    """
    rng = np.random.RandomState(seed)
    unique_respondents = np.unique(respondent_ids)
    n_respondents = len(unique_respondents)
    
    results = {'Pure-DCE': {'log_loss': [], 'brier': []},
                 'Raw-LLM': {'log_loss': [], 'brier': []},
                 'EFR': {'log_loss': [], 'brier': []}}
    
    for b in range(n_bootstrap):
        # Sample respondents with replacement
        sampled_respondents = rng.choice(unique_respondents, size=n_respondents, replace=True)
        
        # Get all rows for sampled respondents
        mask = np.isin(respondent_ids, sampled_respondents)
        y_boot = y_true[mask]
        
        for method in ['Pure-DCE', 'Raw-LLM', 'EFR']:
            probs_boot = probs[method][mask]
            
            if len(y_boot) > 0:
                ll = multinomial_log_loss(y_boot, probs_boot)
                br = multiclass_brier(y_boot, probs_boot)
                results[method]['log_loss'].append(ll)
                results[method]['brier'].append(br)
    
    # Compute CIs
    ci = {}
    for method in results:
        ci[method] = {
            'log_loss': {
                'mean': np.mean(results[method]['log_loss']),
                'ci_low': np.percentile(results[method]['log_loss'], 2.5),
                'ci_high': np.percentile(results[method]['log_loss'], 97.5),
            },
            'brier': {
                'mean': np.mean(results[method]['brier']),
                'ci_low': np.percentile(results[method]['brier'], 2.5),
                'ci_high': np.percentile(results[method]['brier'], 97.5),
            }
        }
    
    return ci


def main():
    # File paths
    llm_path = Path("llm_parsed_outputs_multinomial_6tasks_canonical.csv")
    heldout_path = Path("dce_encoded.csv")  # Need to filter for held-out
    
    if not llm_path.exists():
        print(f"LLM results not found: {llm_path}")
        print("Run run_multinomial_6tasks_canonical.py first")
        return
    
    # Load data
    print("Loading LLM results...")
    llm_probs = load_llm_results(llm_path)
    print(f"Loaded {len(llm_probs)} tasks")
    
    print("\nLoading held-out DCE data...")
    df = load_held_out_data(heldout_path)
    
    # Filter for held-out respondents (20% test set)
    # Use same split as in heldout_dce_validation.py
    rids = sorted(df['RespondentID'].unique())
    n_test = int(0.20 * len(rids))
    test_ids = rids[-n_test:]  # Last 20%
    heldout = df[df['RespondentID'].isin(test_ids)].copy()
    
    print(f"Held-out respondents: {len(test_ids)}")
    print(f"Held-out rows: {len(heldout)}")
    
    # Prepare data for multinomial evaluation
    # Need to map each task to P_A, P_B, P_C
    # And get actual choices (y)
    
    # For each task, get:
    # - LLM: P_A, P_B, P_C from llm_probs
    # - Pure-DCE: need to compute from fitted model
    # - EFR: 0.25 * LLM + 0.75 * Pure-DCE
    
    print("\nComputing metrics...")
    # TODO: Implement full analysis once LLM results are ready
    
    print("\n" + "=" * 72)
    print("Analysis ready - waiting for LLM results")
    print("=" * 72)


if __name__ == "__main__":
    main()
