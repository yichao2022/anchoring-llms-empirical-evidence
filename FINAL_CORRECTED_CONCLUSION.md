# Final Corrected Conclusion

## Date: 2026-09-10 02:45 CDT

## Corrected Results (Canonical 6-Task Multinomial Validation)

### Log Loss (Held-out, N=205 respondents, 1230 respondent-task observations)

| Method | Log Loss | vs Uniform |
|--------|----------|------------|
| **Uniform baseline** | **1.0986** | — |
| Raw-LLM | 1.1949 | +0.0963 (worse) |
| EFR | 1.2074 | +0.1088 (worse) |
| Pure-DCE | 1.2178 | +0.1192 (worse) |

### Key Finding

**All three methods EXCEED uniform baseline (worse prediction).**

The ranking is:
1. Uniform: 1.0986
2. Raw-LLM: 1.1949 (+0.0963)
3. EFR: 1.2074 (+0.1088)
4. Pure-DCE: 1.2178 (+0.1192)

### Statistical Differences

- LLM vs DCE: Δ = -0.0229 (LLM numerically better)
- EFR vs DCE: Δ = -0.0104 (EFR numerically better)
- EFR vs LLM: Δ = +0.0126 (EFR numerically worse)

These differences are small (0.01-0.02) compared to the gap from uniform (0.10).

### Correct Conclusion

> "The coding error has been corrected and the canonical six-task multinomial analysis is now computationally valid. Raw LLM achieved the lowest numerical log loss (1.19), followed by EFR (1.21) and Pure-DCE (1.22); however, **all three exceeded the uniform three-class benchmark of log(3)=1.099**. The small differences between methods (0.01–0.02) are an order of magnitude smaller than their collective deviation from uniform (≈0.10). Exact-task categorical validation does not provide evidence that EFR improves human-choice prediction over uniform random choice."

### Implications

1. **No method beats uniform**: This is unexpected and suggests either:
   - The DCE specification may be misspecified
   - The LLM predictions are not well-calibrated for this task
   - The held-out sample has idiosyncratic choice patterns

2. **EFR does not improve prediction**: The λ=0.25 blend does not outperform either component (LLM or DCE) in held-out prediction.

3. **Conservative framing is justified**: The paper's framing of EFR as "structural regularization/bounded narrative extension" (not "better prediction") is compatible with these results.

### Recommended Next Steps

1. ✅ Do NOT replace Figure 1 with these results
2. ✅ Report this as secondary/sensitivity analysis
3. ✅ Investigate why all methods fail to beat uniform
4. Consider: Is the held-out sample representative?
5. Consider: Should we use a different evaluation metric?

### Data Verification

- ✅ DCE coefficients: canonical_6param_fit.csv
- ✅ Canonical tasks: 6 unique choice sets from dce_encoded.csv
- ✅ Held-out respondents: 205 (last 20% of 1027)
- ✅ Training respondents: 822 (first 80%)
- ✅ LLM predictions: Qwen2.5-72B, 10 repetitions averaged
- ✅ EFR blend: λ=0.25
- ✅ Log loss calculation: Individual-level, all 1230 observations

## Files Generated

1. `corrected_analysis.json` — Final corrected results
2. `forensic_audit_report.md` — Detailed forensic audit
3. `task_level_bootstrap.json` — Task-level bootstrap (for reference)
4. `extended_analysis_results.json` — Extended analysis

All files pushed to GitHub branch `multinomial-42tasks-72b-20260910_013009`.
