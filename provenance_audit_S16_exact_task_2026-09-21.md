# Table S16 Exact-Task Validation — Provenance Audit

**Date**: 2026-09-21
**Auditor**: Hermes (per user request)
**Repo**: `~/bdt_repo` (origin: yichao2022/behavioral-digital-twins, HEAD c81de6a)
**Scope**: Verify that the primary exact-task multinomial validation (Table S16,
main text L342: Pure-DCE 0.911 / EFR 0.923 / Raw LLM 1.020 / uniform 1.099,
N=205 respondents, 1,230 choices) is built on the true DCE design.

---

## Executive summary

**Verdict: FAIL.** The Table S16 exact-task validation is built on a false
design assumption ("all 1,027 respondents completed the same six choice tasks").
The actual DCE uses a **7-block design** (42 distinct tasks; each respondent sees
one block of 6). Three concrete defects, each independently invalidating the
reported numbers:

1. **Held-out split uses "last 205 by RespondentID", not the canonical
   seed=2026 random split** → wrong held-out respondents.
2. **`Choiceset 1-6` is treated as globally identical tasks**, but the same
   Choiceset index means *different* (wait, eff, se) profiles across blocks →
   heterogeneous tasks are pooled into one "task" for multinomial log loss.
3. **The LLM multinomial prompts were built from Respondent 1 only**
   (block 1's tasks) → LLM probabilities cover one block, not all seven.

This is not a wording problem; the reported log losses are not defined on
well-posed "exact tasks".

---

## Evidence

### 1. The design is 7 blocks × 6 tasks, not 6 global tasks

`dce_tasks_42_export.csv` (in repo at c81de6a):

| block_id | tasks | n_respondents_total | n_heldout_receiving_task |
|---|---|---|---|
| 1 | 6 | 250 | 57 |
| 2 | 6 | 241 | 46 |
| 3 | 6 | 222 | 38 |
| 4 | 6 | 95 | 21 |
| 5 | 6 | 83 | 14 |
| 6 | 6 | 80 | 18 |
| 7 | 6 | 55 | 10 |
| **Σ** | **42** | **1,026** | **204** |

All 205 held-out respondents map to exactly one block
(57/46/38/21/14/18/10 = 204; the 205th is the wait=2 anomaly respondent
whose row is excluded).

Per-block non-opt-out profile sets are **disjoint across blocks** (verified by
exhaustive comparison of the 8 distinct (wait,eff,se)-triple patterns among
held-out respondents against the 7 block profile sets; the 8th pattern is the
anomaly respondent).

### 2. Same Choiceset index ≠ same task content

In `analysis_output/dce_encoded.csv`, held-out respondents:
- Every respondent has Choicesets {1,2,3,4,5,6} (18 alt-rows each).
- But Choiceset=1 alone carries **13 distinct (wait, eff, se) cells**
  (e.g. (0,0.5,2)×103, (0,0.7,1)×38, (1,0.5,2)×59, (6,0.95,2)×21, …).
  The task content behind "Choiceset 1" differs by block.

⇒ Aggregating all respondents' "Choiceset=1" choices as one multinomial task
pools different alternatives across respondents.

### 3. Held-out split in the S16 script is wrong

`analysis_6tasks_canonical_NEW.py` (c81de6a), lines 83–86:

```python
# Get held-out split (last 205 respondents)
respondent_ids = sorted(set(int(r['RespondentID']) for r in dce_responses if r['RespondentID']))
held_out_ids = set(respondent_ids[-205:])  # Last 205
```

This selects the **205 largest RespondentIDs**, not the canonical
seed=2026 random split (`canonical_split_seed2026.json`: 822 train / 205 test
via `random.Random(2026).shuffle`). The two sets differ (verified: the
canonical test set contains e.g. Respondent 1; "last 205 by ID" does not).

### 4. LLM multinomial probabilities come from one respondent / one block

`extract_true_canonical_6tasks.py` (c81de6a):

```python
# Get data from first respondent (all respondents have same 6 tasks)
resp_id = "1"
resp_rows = [parse_row(r) for r in rows if r.get("RespondentID") == resp_id]
```

The 6 multinomial prompts (and hence `llm_parsed_outputs_multinomial_6tasks_canonical_NEW.csv`
P_A/P_B/P_C) are derived from **Respondent 1's** tasks, i.e. block 1.
Respondents in blocks 2–7 answered *different* tasks; their choices are scored
against block-1 LLM probabilities.

### 5. Manuscript text claims the false assumption

- main text L118: "All 1,027 respondents completed the **same six choice tasks**"
- supplement L374: "All 1,027 respondents completed the same six choice tasks"
- supplement L1055: same claim
- supplement Table S3/S16 notes repeat the same six-task claim

These statements are contradicted by `dce_tasks_42_export.csv`.

---

## Downstream impact

| Artifact | Claim | Status |
|---|---|---|
| Table S16 exact-task log losses | 0.911 / 0.923 / 1.020 / 1.099 | Invalid (pooled heterogeneous tasks; wrong split; wrong LLM probs) |
| Main L342 caption | same numbers | Invalid |
| Figure 1 (exact-task panel) | derived from same pipeline | Invalid |
| Section 5.1 / Abstract numbers | 1,230 choices | Invalid (205×6 pooled across blocks) |
| Table S7 matched subset N=813 | block-weighted, arithmetic OK | Valid as *alternative-level* matching, but caption/description must reflect block design |
| 3,690 / 3,689 / 2,876 pool counts | correct | Valid |

## Recommended fix (structural)

1. **Admit the block design.** Every respondent completes 6 tasks *within their
   assigned block*; tasks differ across blocks.
2. **Rebuild exact-task validation per block:**
   - Load canonical split from `canonical_split_seed2026.json` (not "last 205").
   - For each of the 7 blocks, extract that block's 6 administered tasks
     (from `dce_tasks_42_export.csv`, columns A_*/B_*).
   - Run LLM multinomial prompts per block (42 prompts, not 6) — requires
     re-running the LLM (Qwen2.5-72B, 10 reps) on the 42 block tasks.
   - Score held-out respondents against their **own block's** LLM probabilities,
     grouped by (block, task), then pool log loss over (respondent, task).
   - Primary validation then has 42 distinct task-level cells; sample size is
     still 205 respondents × 6 tasks = 1,230 respondent-tasks, but the
     denominator is per-block.
3. **Alternative (cheaper, weaker):** If the LLM cannot be re-run, restrict the
   exact-task validation to block 1 only (57 held-out respondents, 342
   respondent-tasks) and clearly label it a single-block validation. The main
   text "1,230 choices" claim must be dropped.
4. **Fix the manuscript design description** (L118/L374/L1055) to state the
   7-block design and 42 unique tasks; the "5.6% coverage" statement must be
   recomputed (currently 12/216 applies only within a block).

## Reproducibility

All numbers above were recomputed from repo files at c81de6a:
- `dce_tasks_42_export.csv` — block/task/respondent assignment
- `analysis_output/dce_encoded.csv` — respondent choices
- `canonical_split_seed2026.json` — canonical split
- `analysis_6tasks_canonical_NEW.py` — S16 pipeline (shown to be defective)
- `extract_true_canonical_6tasks.py` — prompt builder (shown to be defective)
- `regen_table_s7_counts.py` — S7 matched counts (correct arithmetic)

Audit queries (Python, pandas) are in the conversation log; can be re-run
on request.
