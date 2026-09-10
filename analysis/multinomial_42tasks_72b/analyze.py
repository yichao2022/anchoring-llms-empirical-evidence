"""Analyze multinomial validation results"""
import csv
import json
import numpy as np
from pathlib import Path

def analyze():
    with open('llm_parsed_outputs_multinomial_42tasks.csv') as f:
        rows = list(csv.DictReader(f))
    
    # Aggregate by task
    tasks = {}
    for r in rows:
        t = int(r['task_num'])
        if t not in tasks:
            tasks[t] = {'P_A': [], 'P_B': [], 'P_C': []}
        tasks[t]['P_A'].append(float(r['P_A']))
        tasks[t]['P_B'].append(float(r['P_B']))
        tasks[t]['P_C'].append(float(r['P_C']))
    
    # Calculate statistics
    print("Task | A (mean±sd) | B (mean±sd) | C (mean±sd) | N")
    print("-" * 70)
    for t in sorted(tasks.keys()):
        d = tasks[t]
        n = len(d['P_A'])
        print(f"{t:2d}   | {np.mean(d['P_A']):.3f}±{np.std(d['P_A']):.3f} | "
              f"{np.mean(d['P_B']):.3f}±{np.std(d['P_B']):.3f} | "
              f"{np.mean(d['P_C']):.3f}±{np.std(d['P_C']):.3f} | {n}")
    
    # Save aggregated
    with open('aggregated_by_task.json', 'w') as f:
        result = {}
        for t, d in tasks.items():
            result[t] = {
                'P_A_mean': np.mean(d['P_A']),
                'P_A_std': np.std(d['P_A']),
                'P_B_mean': np.mean(d['P_B']),
                'P_B_std': np.std(d['P_B']),
                'P_C_mean': np.mean(d['P_C']),
                'P_C_std': np.std(d['P_C']),
                'N': len(d['P_A'])
            }
        json.dump(result, f, indent=2)

if __name__ == '__main__':
    analyze()
