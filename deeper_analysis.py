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

def analyze_by_profile(rows):
    """Analyze probability distribution by profile."""
    profiles = defaultdict(lambda: {'probs': [], 'yes_no': []})
    
    for r in rows:
        pid = r['profile_id']
        prob = float(r['parsed_probability']) if r['parsed_probability'] != 'N/A' else None
        profiles[pid]['probs'].append(prob)
        profiles[pid]['yes_no'].append(r['parsed_yes_no'])
    
    print("=" * 80)
    print("DETAILED PROFILE ANALYSIS")
    print("=" * 80)
    
    for pid in sorted(profiles.keys(), key=int):
        data = profiles[pid]
        probs = [p for p in data['probs'] if p is not None]
        
        if probs:
            mean = statistics.mean(probs)
            stdev = statistics.stdev(probs) if len(probs) > 1 else 0
            median = statistics.median(probs)
            min_val = min(probs)
            max_val = max(probs)
            yes_count = sum(1 for y in data['yes_no'] if y == 'Yes')
            yes_pct = yes_count / len(data['yes_no']) * 100
            
            print(f"\nProfile {pid}:")
            print(f"  Mean:   {mean:.4f}")
            print(f"  Median: {median:.4f}")
            print(f"  StdDev: {stdev:.4f}")
            print(f"  Range:  [{min_val:.4f}, {max_val:.4f}]")
            print(f"  Yes%:   {yes_pct:.1f}%")
            print(f"  CV:     {(stdev/mean*100):.1f}%" if mean > 0 else "  CV:     N/A")

def analyze_variance_by_feature(rows):
    """Analyze variance by efficacy, side effects, cash."""
    print("\n" + "=" * 80)
    print("VARIANCE ANALYSIS BY FEATURE")
    print("=" * 80)
    
    # Group by efficacy level
    efficacy_groups = defaultdict(list)
    side_effect_groups = defaultdict(list)
    cash_groups = defaultdict(list)
    
    for r in rows:
        prob = float(r['parsed_probability'])
        # Parse the profile from raw response context
        efficacy_groups['all'].append(prob)
    
    print("\nOverall Variance:")
    print(f"  CV (Coefficient of Variation): {(statistics.stdev([float(r['parsed_probability']) for r in rows])/statistics.mean([float(r['parsed_probability']) for r in rows]))*100:.1f}%")
    
    print(f"\nInterpretation:")
    print(f"  - CV < 10%: Low variance (consistent responses)")
    print(f"  - CV 10-20%: Moderate variance")
    print(f"  - CV > 20%: High variance (inconsistent responses)")

def check_convergence(rows):
    """Check if 10 reps per profile converged."""
    print("\n" + "=" * 80)
    print("CONVERGENCE CHECK (10 reps per profile)")
    print("=" * 80)
    
    profiles = defaultdict(list)
    for r in rows:
        profiles[r['profile_id']].append(float(r['parsed_probability']))
    
    for pid in sorted(profiles.keys(), key=int):
        probs = profiles[pid]
        first_5_mean = statistics.mean(probs[:5])
        last_5_mean = statistics.mean(probs[5:])
        diff = abs(first_5_mean - last_5_mean)
        
        converged = "✓" if diff < 0.1 else "✗"
        print(f"  Profile {pid}: First 5={first_5_mean:.4f}, Last 5={last_5_mean:.4f}, Diff={diff:.4f} {converged}")

def generate_statistics_table(rows):
    """Generate a markdown table for the paper."""
    profiles = defaultdict(lambda: {'probs': []})
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
        stdev = statistics.stdev(probs)
        mn = min(probs)
        mx = max(probs)
        ci_lower = mean - 1.96 * stdev / (len(probs) ** 0.5)
        ci_upper = mean + 1.96 * stdev / (len(probs) ** 0.5)
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
    analyze_by_profile(rows)
    analyze_variance_by_feature(rows)
    check_convergence(rows)
    generate_statistics_table(rows)
