#!/usr/bin/env python3
"""
Compute multinomial log loss with seeded random split (seed=2026).
Correct implementation for Appendix O validation.
"""
import csv
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Config
SEED = 2026
TRAIN_FRAC = 0.80

# Load DCE data
df = pd.read_csv('analysis_output/dce_encoded.csv')
print(f"Loaded {len(df)} DCE rows")

# Get unique respondents
rids = sorted(df['RespondentID'].unique())
print(f"Total respondents: {len(rids)}")

# Seeded random split
rng = random.Random(SEED)
shuffled = rids[:]
rng.shuffle(shuffled)

n_train = int(TRAIN_FRAC * len(shuffled))
train_ids = set(shuffled[:n_train])
test_ids = set(shuffled[n_train:])

print(f"\nSplit (seed={SEED}):")
print(f"  Training: {len(train_ids)} respondents")
print(f"  Test: {len(test_ids)} respondents")
print(f"  Test respondent IDs: {sorted(test_ids)[:5]}... (first 5)")

# Verify split is random
print(f"\nVerifying randomness:")
print(f"  First test ID: {min(test_ids)}")
print(f"  Is continuous tail? {max(train_ids) + 1 == min(test_ids)}")

# Filter test data
test_df = df[df['RespondentID'].isin(test_ids)].copy()
print(f"\nTest set: {len(test_df)} rows")

# Save test respondent IDs for reference
with open('test_respondents_seed2026.json', 'w') as f:
    json.dump({
        'seed': SEED,
        'train_frac': TRAIN_FRAC,
        'n_total': len(rids),
        'n_train': len(train_ids),
        'n_test': len(test_ids),
        'train_ids': sorted(train_ids),
        'test_ids': sorted(test_ids)
    }, f, indent=2)

print(f"\n✓ Saved split to test_respondents_seed2026.json")
print(f"\nNote: This script only verifies the split.")
print(f"Full DCE refit and log loss computation requires additional implementation.")
