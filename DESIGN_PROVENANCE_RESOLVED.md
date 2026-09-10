# DCE Design Provenance - RESOLVED
# Source: CLD_Vaccination_DCE_Design_NonEfficient.csv (Original Design File)
# Date: 2026-09-09

## CRITICAL FINDING: Design File Shows Only ONE Block

The original design file contains **only 6 tasks** (not 42), confirming:

### Original Design Structure:
```
Task 1: A(wait=3, eff=70%, se=0.1%, cash=800, origin=Imported) vs 
        B(wait=3, eff=70%, se=10%, cash=800, origin=Imported)
Task 2: A(wait=3, eff=70%, se=1%, cash=800, origin=Domestic) vs 
        B(wait=1, eff=70%, se=0.1%, cash=50, origin=Imported)
Task 3: A(wait=3, eff=50%, se=10%, cash=800, origin=Domestic) vs 
        B(wait=1, eff=50%, se=1%, cash=800, origin=Domestic)
Task 4: A(wait=1, eff=50%, se=10%, cash=50, origin=Domestic) vs 
        B(wait=1, eff=95%, se=1%, cash=200, origin=Imported)
Task 5: A(wait=3, eff=95%, se=1%, cash=50, origin=Domestic) vs 
        B(wait=0, eff=95%, se=10%, cash=800, origin=Imported)
Task 6: A(wait=0, eff=95%, se=10%, cash=800, origin=Domestic) vs 
        B(wait=0, eff=70%, se=0.1%, cash=800, origin=Domestic)
```

### Key Discrepancies Found:

| Attribute | Design File | Current Analysis | Issue |
|-----------|-------------|------------------|-------|
| **Side Effects** | 0.1%, 1%, 10% | 10%, 20%, 30% | **MAJOR** |
| **Wait Time** | 0, 1, 3 months | 0, 1, 3, 6 months | Missing 6mo in design? |
| **Efficacy** | 50%, 70%, 95% | 50%, 70%, 95% | ✓ Match |
| **Cash** | 50, 200, 800 RMB | 50, 200, 800 RMB | ✓ Match |
| **Origin** | Domestic, Imported | Domestic, Imported | ✓ Match |

### Respondent-Facing Wording (from Design File):

| Attribute | Design File Wording | Current Guess |
|-----------|---------------------|---------------|
| Wait | "Average Waiting Time: X months" / "Walk-in appointment" | "Waiting time: X months" |
| Origin | "Origin of Vaccine: Domestic/Imported" | "Vaccine origin: Domestic/Imported" |
| Efficacy | "Vaccine Efficacy: X %" | "Vaccine effectiveness: X%" |
| Side Effects | "Side Effects: X %" | "Risk of side effects: X%" |
| Cash | "Cash Incentive: X RMB" | "Cash incentive: X RMB" |
| Opt-out | "None of these" | "Do not receive the vaccine" |

## ROOT CAUSE ANALYSIS

### Why 7 Blocks in Data?

The design file shows only ONE block with 6 tasks. The seven-block pattern in the analytic data suggests:

1. **Versioning**: Multiple questionnaire versions were fielded
2. **Randomization**: Task order was randomized across respondents
3. **Data Processing**: Some preprocessing created artificial blocks

### Side-Effect Coding Discrepancy

**Design file**: Side effects coded as **0.1%, 1%, 10%** (probabilities)
**Current analysis**: Side effects coded as **10%, 20%, 30%** (levels 1,2,3)

This is a **critical discrepancy**. The design file shows actual percentages, not levels.

## CORRECTED UNDERSTANDING

### True Design:
- **1 design block** (as manuscript claims)
- **6 tasks** (as manuscript claims)
- **Side effects**: 0.1%, 1%, 10% (not 10%, 20%, 30%)
- **12 unique non-opt-out profiles** (need to verify)

### Data Reconstruction Error:
- The 7-block pattern is likely from **task order randomization**
- Respondents received the same 6 tasks in different orders
- Our reconstruction incorrectly identified different orderings as different blocks

## IMMEDIATE ACTIONS REQUIRED

1. **Re-analyze data**: Group by task content, not by respondent task sequence
2. **Correct side-effect mapping**: 0.1%→Level 1, 1%→Level 2, 10%→Level 3
3. **Verify 12 profiles**: Reconstruct from design file
4. **Update CSV**: Use design file wording, not Table S3
5. **Re-validate**: 6 tasks, not 42

## QUESTIONS FOR OWNER

1. Was task order randomized across respondents?
2. Why do side effects show as 10%/20%/30% in data but 0.1%/1%/10% in design?
3. Is there a data processing script that transformed the raw responses?
4. Should we use the design file as the canonical reference?
