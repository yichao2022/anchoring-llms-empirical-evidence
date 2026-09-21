#!/usr/bin/env python3
"""Regenerate Table S7 matched-row counts from canonical held-out IDs + canonical six tasks.

Single source of truth:
  - canonical_split_seed2026.json  -> canonical 205 held-out respondent IDs
  - dce_tasks_42_export.csv        -> canonical six tasks (task_global_id 1..6) attribute levels
  - analysis_output/dce_encoded.csv-> respondent-level choice rows

Matching rule (as documented in Supplement Table S7 note):
  matched = held-out rows whose (wait, eff, se) triple is in the literal overlap
  between the intended DCE design and the LLM 64-state grid:
    DCE design levels : wait {0,1,3,6}, eff {0,0.5,0.7,0.95}, se {0,1,2,3}
    LLM grid levels   : wait {0,2,4,6}, eff {0.3,0.5,0.7,0.9}, se {0,1,2,3}
    overlap           : wait {0,6}, eff {0.5,0.7}, se {1,2,3}  (2x2x3 = 12 cells)
  cash and origin are held at grid reference values and do NOT participate.
  opt-out rows (eff=0 or se=0) are excluded.

Prints respondent x matched-row counts and the exact N that must appear
consistently in main text, supplement contents, and Table S7.
"""
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent

# 1. canonical held-out IDs
split = json.load(open(REPO / "canonical_split_seed2026.json"))
test_ids = set(split["test_ids"])
print(f"canonical split: n_test = {split['n_test']}, test_ids = {len(test_ids)}")

# 2. canonical six tasks = task_global_id 1..6 (the LLM-administered tasks)
tasks = pd.read_csv(REPO / "dce_tasks_42_export.csv")
six = tasks[tasks["task_global_id"].isin(range(1, 7))].copy()
print(f"canonical six tasks: {len(six)} (task_global_id 1..6)")

# 3. respondent-level rows
dce = pd.read_csv(REPO / "analysis_output" / "dce_encoded.csv")
dce = dce[dce["RespondentID"].isin(test_ids)].copy()
print(f"held-out rows (205 respondents x 6 tasks x 3 alts): {len(dce)}")
print(f"  distinct held-out respondents in data: {dce['RespondentID'].nunique()}")

# 4. matching rule
DESIGN = dict(wait={0, 1, 3, 6}, eff={0, 0.5, 0.7, 0.95}, se={0, 1, 2, 3})
GRID = dict(wait={0, 2, 4, 6}, eff={0.3, 0.5, 0.7, 0.9}, se={0, 1, 2, 3})
overlap_wait = DESIGN["wait"] & GRID["wait"]
overlap_eff = DESIGN["eff"] & GRID["eff"]
overlap_se = DESIGN["se"] & GRID["se"]
print(f"overlap: wait {sorted(overlap_wait)} x eff {sorted(overlap_eff)} x se {sorted(overlap_se)} "
      f"= {len(overlap_wait)*len(overlap_eff)*len(overlap_se)} candidate cells")

# non-opt-out = all three focal attributes positive (cash/origin excluded per note)
dce["nonopt"] = (dce["VaccineEfficacy"] > 0) & (dce["SideEffects"] > 0)

# matched: non-opt-out AND (wait,eff,se) in overlap
matched = dce[
    dce["nonopt"]
    & dce["WaitTime"].isin(overlap_wait)
    & dce["VaccineEfficacy"].isin(overlap_eff)
    & dce["SideEffects"].isin(overlap_se)
].copy()
print(f"\n=== MATCHED ROWS ===")
print(f"alt-level rows: {len(matched)}")
print(f"distinct respondents contributing: {matched['RespondentID'].nunique()}")
print(f"distinct (wait,eff,se) cells populated: {matched[['WaitTime','VaccineEfficacy','SideEffects']].drop_duplicates().shape[0]} / 12")
print(f"distinct choicesets (respondent x task): {matched.groupby(['RespondentID','Choiceset']).ngroups}")
print(f"  alt rows per respondent-task:")
print(matched.groupby(['RespondentID', 'Choiceset']).size().value_counts().sort_index().to_string())
print(f"  tasks with exactly 2 matched alts: {(matched.groupby(['RespondentID','Choiceset']).size() == 2).sum()}")
print(f"  tasks with exactly 3 matched alts: {(matched.groupby(['RespondentID','Choiceset']).size() == 3).sum()}")

# respondents x matched-row count distribution
rc = matched.groupby("RespondentID").size()
print(f"\nrespondents x matched-row counts: min={rc.min()}, max={rc.max()}, mean={rc.mean():.2f}")
print(rc.value_counts().sort_index().to_string())

# 5. sanity: full held-out pool
full = dce[dce["nonopt"]]
print(f"\nheld-out non-opt-out alt-rows: {len(full)}")
print(f"  expected: 205 x 6 x 2 (A+B non-opt-out) = 2460 (opt-out C excluded)")
# wait=2 anomaly check
w2 = dce[(dce["WaitTime"] == 2)]
print(f"  wait=2 rows in held-out: {len(w2)}")

matched.to_csv(REPO / "outputs" / "table_s7_matched_canonical.csv", index=False)
print(f"\nwrote outputs/table_s7_matched_canonical.csv")
