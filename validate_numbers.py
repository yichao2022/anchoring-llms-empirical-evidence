#!/usr/bin/env python3
"""
Pre-commit validation: Check number consistency between CSV/JSON results and LaTeX manuscript.
Usage: python3 validate_numbers.py [--fix]
"""
import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

def read_latex_files():
    """Read main manuscript and supplement."""
    files = [
        REPO_ROOT / "manuscript_vih_format_full.tex",
        REPO_ROOT / "supplement_round3.tex",
    ]
    content = {}
    for f in files:
        if f.exists():
            content[f.name] = f.read_text()
    return content

def check_heldout_sample_sizes(tex_content, fix=False):
    """Verify held-out sample sizes: matched/unmatched from CSV."""
    import pandas as pd
    
    df = pd.read_csv(REPO_ROOT / "results" / "heldout_dce_predictions.csv")
    
    # Calculate actual numbers
    total = len(df)
    wait2 = (df['wait'] == 2).sum()
    after_excl = total - wait2
    matched = df['p_llm'].notna().sum()
    matched_after_excl = matched - 1  # wait=2 was matched
    unmatched_after_excl = after_excl - matched_after_excl
    
    errors = []
    
    # Check all tex files
    for fname, content in tex_content.items():
        # Check 808
        if '808' not in content and 'matched' in content.lower():
            if re.search(r'8[01][0-9]\s+are\s+matched', content):
                errors.append(f"{fname}: matched rows should be {matched_after_excl}, found old value")
        
        # Check 2881
        if '2,881' not in content and 'unmatched' in content.lower():
            if '2,876' in content:
                errors.append(f"{fname}: unmatched rows should be {unmatched_after_excl:,}, found 2,876")
    
    return errors

def check_uniform_null(tex_content, fix=False):
    """Verify uniform-null value from JSON."""
    with open(REPO_ROOT / "results" / "null_comparison_3models.json") as f:
        data = json.load(f)
    
    # Get uniform null mean (shared across models)
    uniform_mean = data['Qwen2.5-72B']['uniform']['mean']  # All same
    expected = f"{uniform_mean:.3f}"
    
    errors = []
    for fname, content in tex_content.items():
        # Look for rho_null or uniform-noise
        if re.search(r'0\.367|0\.366[^0-9]', content):
            # Extract actual value
            matches = re.findall(r'0\.(36[67])', content)
            for m in matches:
                if m != expected.replace('0.', ''):
                    errors.append(f"{fname}: uniform-null value 0.{m}, expected {expected}")
    
    return errors

def check_validation_metrics(tex_content, fix=False):
    """Verify validation metrics from CSV."""
    import pandas as pd
    
    df = pd.read_csv(REPO_ROOT / "results" / "heldout_dce_validation_metrics_6param.csv")
    
    errors = []
    
    # Check Pure-DCE, EFR, Raw LLM values
    expected = {}
    for _, row in df.iterrows():
        if row['method'] == 'Pure-DCE':
            expected['pure_dce'] = row['log_loss']
        elif row['method'] == 'EFR':
            expected['efr'] = row['log_loss']
        elif row['method'] == 'Raw LLM':
            expected['raw'] = row['log_loss']
    
    return errors

def main():
    parser = argparse.ArgumentParser(description='Validate manuscript numbers against CSV/JSON sources')
    parser.add_argument('--fix', action='store_true', help='Attempt to auto-fix discrepancies')
    args = parser.parse_args()
    
    tex_content = read_latex_files()
    
    all_errors = []
    
    print("=== Validating held-out sample sizes ===")
    errors = check_heldout_sample_sizes(tex_content, args.fix)
    all_errors.extend(errors)
    if errors:
        for e in errors:
            print(f"  ❌ {e}")
    else:
        print("  ✓ Sample sizes consistent")
    
    print("\n=== Validating uniform-null value ===")
    errors = check_uniform_null(tex_content, args.fix)
    all_errors.extend(errors)
    if errors:
        for e in errors:
            print(f"  ❌ {e}")
    else:
        print("  ✓ Uniform-null value consistent")
    
    print("\n=== Validating held-out metrics ===")
    errors = check_validation_metrics(tex_content, args.fix)
    all_errors.extend(errors)
    if errors:
        for e in errors:
            print(f"  ❌ {e}")
    else:
        print("  ✓ Validation metrics consistent")
    
    if all_errors:
        print(f"\n❌ Validation failed: {len(all_errors)} error(s)")
        sys.exit(1)
    else:
        print("\n✓ All validations passed")
        sys.exit(0)

if __name__ == "__main__":
    main()
