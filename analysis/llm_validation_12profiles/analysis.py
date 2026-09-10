#!/usr/bin/env python3
"""
Deeper statistical analysis of LLM validation results.
"""
import csv
import statistics
from collections import defaultdict

def load_data():
    with open('/tmp/behavioral-digital-twins/llm_parsed_outputs_12profiles.csv') as f:
        return list(csv.DictReader(f))

def generate_statistics_table(rows):
    """Generate a markdown table for the paper."""
    profiles = defaultdict(list)
    for r in rows:
        profiles[r['profile_id']].append(float(r['parsed_probability']))
    
    print("\n" + "=" * 80)
    print("MARKDOWN TABLE FOR MANUSCRIPT")
    print("=" * 80)
    
    print("\n| Profile | Mean | SD | Min | Max | 95% CI | Accept % |")
    print("|---------|------|-----|-----|-----|--------|----------|")
    
    for pid in sorted(profiles.keys(), key=int):
        probs = profiles[pid]
        mean = statistics.mean(probs)
        stdev = statistics.stdev(probs) if len(probs) > 1 else 0
        mn = min(probs)
        mx = max(probs)
        ci_lower = mean - 1.96 * stdev / (len(probs) ** 0.5) if stdev > 0 else mean
        ci_upper = mean + 1.96 * stdev / (len(probs) ** 0.5) if stdev > 0 else mean
        yes_pct = sum(1 for p in probs if p > 0.5) / len(probs) * 100
        
        print(f"| {pid:>7} | {mean:.3f} | {stdev:.3f} | {mn:.3f} | {mx:.3f} | [{ci_lower:.3f}, {ci_upper:.3f}] | {yes_pct:.1f}% |")
    
    all_probs = [float(r['parsed_probability']) for r in rows]
    overall_mean = statistics.mean(all_probs)
    overall_std = statistics.stdev(all_probs)
    overall_ci = 1.96 * overall_std / (len(all_probs) ** 0.5)
    yes_pct = sum(1 for p in all_probs if p > 0.5) / len(all_probs) * 100
    
    print(f"\n| **Overall** | **{overall_mean:.3f}** | **{overall_std:.3f}** | **{min(all_probs):.3f}** | **{max(all_probs):.3f}** | **[{overall_mean-overall_ci:.3f}, {overall_mean+overall_ci:.3f}]** | **{yes_pct:.1f}%** |")

if __name__ == "__main__":
    rows = load_data()
    generate_statistics_table(rows)
