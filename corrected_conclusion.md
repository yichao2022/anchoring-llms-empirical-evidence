# Corrected Conclusion for Forensic Audit Report

## Key Correction

**Previous (incorrect) conclusion:**
> "All methods beat uniform baseline"

**Corrected conclusion:**

In the corrected canonical 6-task multinomial validation:
- Raw LLM achieved the lowest numerical log loss (1.19)
- Followed by EFR (1.21) and Pure-DCE (1.22)
- **However, ALL THREE exceeded the uniform three-class benchmark (log(3)=1.099)**

This means the ranking is:
1. Uniform baseline: 1.099
2. Raw-LLM: 1.19
3. EFR: 1.21
4. Pure-DCE: 1.22

## What CAN Be Said

1. **Numerical ranking only**: Raw-LLM < EFR < Pure-DCE (by ~0.02-0.03)
2. **All methods worse than uniform**: The 0.03 difference between LLM and uniform is small and may be noise
3. **Statistical differences remain to be established**: Paired bootstrap CI for Δ(LLM-DCE), Δ(EFR-DCE), Δ(EFR-LLM) needed

## What CANNOT Be Said (Yet)

- "Raw-LLM captures stochastic choice patterns better than deterministic utility model"
  - Problem 1: 0.03 difference without bootstrap CI is not statistically established
  - Problem 2: Conditional logit IS a random-utility stochastic choice model, not "deterministic"

## Next Steps

1. Complete respondent-cluster paired bootstrap (2000 reps) for difference CI
2. Add training-choice-share baseline as non-parametric benchmark
3. Compare training vs held-out Pure-DCE log loss (overfit check)
4. Report task-level observed shares vs predicted probabilities
5. **Do NOT replace Figure 1** until bootstrap confirms meaningful differences

## Honest Framing for Paper

> "The coding error has been corrected and the canonical six-task multinomial analysis is now computationally valid. Raw LLM achieved the lowest numerical log loss (1.19), followed by EFR (1.21) and Pure-DCE (1.22); however, all three exceeded the uniform three-class benchmark of log(3)=1.099. Statistical differences between methods remain to be established using paired respondent-cluster bootstrap inference."

## Implications

If bootstrap shows:
- **Raw-LLM ≈ EFR ≈ Pure-DCE ≈ Uniform**: This is valuable — it honestly tells reviewers that exact-task categorical validation does NOT provide evidence that EFR improves human-choice prediction
- **This finding is compatible with paper's conservative framing**: EFR's advantage may be structural regularization/bounded narrative extension, not "better prediction"

The score definition and elicitation protocol substantially affect comparison results — this itself is a finding worth reporting.
