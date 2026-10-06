#!/usr/bin/env python3
"""
Audit held-out validation numbers with anomaly exclusion.
Prints the exact pipeline:
raw = 3690 → anomaly excluded = 3689 → matched = ? → unmatched = ?
"""

import pandas as pd
import random
import numpy as np
from pathlib import Path

SEED = 2026
TRAIN_FRAC = 0.80

# Load DCE data
df = pd.read_csv('analysis_output/dce_encoded.csv')
print(f'1. RAW total rows: {len(df)}')

# Get unique respondents
rids = sorted(df['RespondentID'].unique())
rng = random.Random(SEED)
shuffled = rids[:]
rng.shuffle(shuffled)
n_train = int(round(TRAIN_FRAC * len(shuffled)))
train_ids = set(shuffled[:n_train])
test_ids = set(shuffled[n_train:])

print(f'2. Train respondents: {len(train_ids)}')
print(f'3. Test respondents: {len(test_ids)} (should include Respondent 1)')

# Split data
df_train = df[df['RespondentID'].isin(train_ids)]
df_test = df[df['RespondentID'].isin(test_ids)]

print(f'4. Test rows (raw, pre-exclusion): {len(df_test)} = {len(test_ids)} respondents × 6 tasks × 3 alternatives')
assert len(df_test) == len(test_ids) * 6 * 3, "Test rows should be 205 × 6 × 3 = 3,690"

# Check wait=2
test_wait2 = df_test[df_test['WaitTime'] == 2]
print(f'5. Test rows with wait=2 (anomaly): {len(test_wait2)}')
if len(test_wait2) > 0:
    print(f'   RespondentID: {test_wait2["RespondentID"].iloc[0]}')

# Exclude wait=2
df_test_clean = df_test[df_test['WaitTime'] != 2].copy()
print(f'6. Test rows after wait=2 exclusion: {len(df_test_clean)}')
assert len(df_test_clean) == len(df_test) - 1, "Should have exactly 1 less row"

# Calculate matched/unmatched
# This is a placeholder - actual matching logic depends on LLM grid availability
# For now, we assert the arithmetic
print(f'\n=== ASSERTION CHECK ===')
print(f'Test rows after exclusion: {len(df_test_clean)}')
print(f'Expected: matched + unmatched = {len(df_test_clean)}')

# The actual matched/unmatched numbers come from heldout_dce_validation.py output
# But those were calculated on 3,690 rows (including wait=2)
# We need to recalculate on 3,689 rows

print(f'\n=== CURRENT SITUATION ===')
print(f'From heldout_dce_validation.py: matched=809, unmatched=2,881')
print(f'Sum: 809 + 2,881 = {809 + 2,881}')
print(f'But after exclusion should be: {len(df_test_clean)}')
print(f'Discrepancy: {809 + 2881 - len(df_test_clean)}')

print(f'\n=== CORRECT ARITHMETIC ===')
print(f'If matched = 809 and total = 3,689, then unmatched = 3,689 - 809 = {3689 - 809}')
print(f'If unmatched = 2,881 and total = 3,689, then matched = 3,689 - 2,881 = {3689 - 2881}')

print(f'\n=== CONCLUSION ===')
print(f'One of these must change:')
print(f'  Option A: matched = 808, unmatched = 2,881 (sum = 3,689)')
print(f'  Option B: matched = 809, unmatched = 2,880 (sum = 3,689)')
print(f'  Option C: use 3,690 total (keep wait=2, no exclusion)')

# Verify Respondent 1 is in test set
resp1_in_test = 1 in test_ids
print(f'\nRespondent 1 in test set: {resp1_in_test}')
print(f'Respondent 1 has {len(df_test[df_test["RespondentID"]==1])} rows in test set (including wait=2)')
