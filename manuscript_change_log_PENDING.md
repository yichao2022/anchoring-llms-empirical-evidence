# DCE Design Provenance - Manuscript Change Log
# Pending Provider Confirmation
# Generated: 2026-09-09
# Status: DO NOT APPLY until provenance confirmed

## Current Manuscript Claims (TO BE REVISED)

### Supplement Line 118 (approximate):
"All N = 1,027 respondents completed the same six choice tasks with two non-opt-out alternatives per task... The 12 administered non-opt-out profiles per respondent (1 unique design block shared by all 1027 respondents) cover 12/216 = 5.6% of the non-opt-out candidate space."

### Table S15:
- 78-task validation (6.3% coverage under 11-of-12-cell match)
- "Only 2-alternative subset valid (A vs B, opt-out excluded)"
- "Three-alternative full subset is EMPTY under canonical match"

## Reconstructed Design Facts (PROVISIONAL)

Based on data forensics (pending provider confirmation):
- Design blocks: 7 (not 1)
- Choice tasks per block: 6
- Unique choice-set compositions: 42 (not 6)
- Unique non-opt-out profiles: 84 (not 12)
- Block respondent counts: 250, 241, 222, 95, 83, 80, 55
- Held-out task coverage: 1,229/1,229 (reconstructed mapping, prior to LLM execution)

## Proposed Manuscript Changes (PENDING)

### 1. Design Description (Supplement)
**Current:** "1 unique design block shared by all 1027 respondents"
**Proposed:** "Respondents were assigned to one of seven DCE blocks, each comprising six choice tasks (n = 250, 241, 222, 95, 83, 80, 55 per block)"

**Current:** "same six choice tasks"
**Proposed:** "42 unique choice-set compositions across blocks"

**Current:** "12 administered non-opt-out profiles per respondent"
**Proposed:** "84 unique non-opt-out profiles across the full design (12 per block)"

**Current:** "12/216 = 5.6% coverage"
**Proposed:** "84/216 = 38.9% full-design coverage (12 per block × 7 blocks)"

### 2. Held-out Validation (Table S15)
**Current:** "78-task validation (6.3% coverage)"
**Proposed:** "Full-task multinomial validation: 1,229 held-out task observations with complete five-attribute coverage"

**Current:** "3-alt full subset is EMPTY"
**Proposed:** DELETE (new validation uses full three-alternative tasks)

**Current:** "Only 2-alt subset valid"
**Proposed:** DELETE (replaced by full multinomial)

### 3. Anomaly Documentation
**Current:** "one anomalous wait=2 alt-row... excluded as data anomaly"
**Proposed:** CONFIRMED (pending provider) or REVISED if provider clarifies intent

## Blocked Until

- [ ] Provider confirms seven-block design
- [ ] Provider provides original questionnaire wording
- [ ] Provider clarifies wait=2 anomaly
- [ ] CSV wording_status updated from "provisional_from_Table_S3" to "confirmed_from_original_questionnaire"
- [ ] API pipeline unfrozen
- [ ] 420 LLM calls executed
- [ ] Results parsed and validated

## Internal Working Notes (Not for Manuscript)

- 1,229/1,229 coverage = "reconstructed full-task mapping achieves complete held-out task coverage prior to LLM execution"
- Do NOT write "100% held-out validation coverage" until after API execution
- Lambda remains fixed at 0.25 (no re-optimization during validation)
- Profile-level binary analysis: DEPRECATED (secondary only)
- Task-level multinomial: PRIMARY

## Application Checklist

When applying changes:
1. Update Supplement design description
2. Update Table S15 (replace 78-task with full multinomial)
3. Update Figure 1 caption if needed
4. Update Methods section if needed
5. Verify all cross-references
6. Recompile PDFs
7. Push to GitHub with descriptive commit message

## Files Modified

- supplement_round3.tex (design description, Table S15)
- main_round3_blinded.tex (if design claims appear)
- manuscript_round3_blinded.pdf
- supplement_round3.pdf

## Verification Command

grep -n "one.*block\|same.*six\|12.*profile\|5.6%\|78.*task" supplement_round3.tex
