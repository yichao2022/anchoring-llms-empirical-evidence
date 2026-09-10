# Final Corrected Conclusion (Revised)

## Date: 2026-09-10 02:52 CDT

## Final Results (Canonical 6-Task Multinomial Validation)

| Method | Log Loss | vs Uniform |
|--------|----------|------------|
| **Uniform-probability benchmark** | **1.0986** | — |
| Raw-LLM | 1.1949 | +0.096 (worse) |
| EFR | 1.2074 | +0.109 (worse) |
| Pure-DCE | 1.2178 | +0.119 (worse) |

## Corrected Conclusion

> In the exact-task categorical validation, Raw LLM achieved the lowest multinomial log loss (1.195), followed by EFR (1.207) and Pure-DCE (1.218). However, all three performed worse than the uniform-probability benchmark of log(3)=1.099. Although pairwise differences between the three model-based approaches were statistically detectable, their absolute magnitudes were small. These results therefore do not provide evidence that EFR improves held-out human-choice prediction in the exact three-alternative DCE task.

## More Precise Framing

- **EFR did not outperform the equal-probability benchmark**
- **EFR did not improve exact-task predictive performance relative to Raw LLM** (Δ = 0.013)

## Evidence Hierarchy Revision

1. **Primary contribution**: EFR constrains LLM deviations from DCE frontier, reduces monotonicity violations, improves structural consistency
2. **Exact-task categorical validation**: No evidence that EFR improves held-out prediction
3. **Old 813-row binary validation**: Downgrade to secondary sensitivity analysis (not full 3-alternative task)
4. **OOD narrative analysis**: Frame as structural robustness, not behavioral validation

## Discussion Point

EFR vs Raw LLM log loss difference: 1.2074 − 1.1949 = **0.0125**

> "EFR imposes a modest predictive cost while providing stronger adherence to the empirical behavioral frontier."

Note: This "modest predictive cost" refers only to exact-task categorical score relative to Raw LLM; Pure-DCE performed worst.

## Figure 1 Revision

- Rename to "Exact-task categorical held-out validation"
- Plot four bars: Raw LLM / EFR / Pure-DCE / Uniform-probability
- Shows no negative result is hidden
- Paper's contribution does not depend on predictive win

## Key Takeaway

The negative result is valuable: it honestly shows that EFR's benefit is structural regularization and bounded narrative extension, not predictive superiority. This strengthens rather than weakens the paper's methodology stance.
