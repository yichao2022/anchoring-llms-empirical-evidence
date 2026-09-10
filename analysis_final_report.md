# Multinomial Validation: 6 Canonical Tasks (Qwen2.5-72B)

## Summary
- **Model**: Qwen2.5-72B-Instruct (Ollama)
- **Tasks**: 6 canonical DCE tasks
- **Calls**: 60 (6 tasks × 10 reps)
- **Success**: 60/60 (100%)

## Task-Level Probabilities

| Task | Alt | Pure-DCE | Raw-LLM | EFR (λ=0.25) |
|------|-----|----------|---------|--------------|
| 1 | A | 0.0008 | 0.4100 | 0.1031 |
| 1 | B | 0.9992 | 0.3400 | 0.8344 |
| 1 | C | 0.0000 | 0.2500 | 0.0625 |
| 2 | A | 0.9998 | 0.4100 | 0.8524 |
| 2 | B | 0.0002 | 0.3300 | 0.0826 |
| 2 | C | 0.0000 | 0.2600 | 0.0650 |
| 3 | A | 0.0000 | 0.2150 | 0.0538 |
| 3 | B | 1.0000 | 0.6750 | 0.9187 |
| 3 | C | 0.0000 | 0.1100 | 0.0275 |
| 4 | A | 1.0000 | 0.5450 | 0.8862 |
| 4 | B | 0.0000 | 0.3000 | 0.0750 |
| 4 | C | 0.0000 | 0.1550 | 0.0388 |
| 5 | A | 0.0000 | 0.3100 | 0.0775 |
| 5 | B | 1.0000 | 0.5100 | 0.8775 |
| 5 | C | 0.0000 | 0.1800 | 0.0450 |
| 6 | A | 1.0000 | 0.4400 | 0.8600 |
| 6 | B | 0.0000 | 0.3500 | 0.0875 |
| 6 | C | 0.0000 | 0.2100 | 0.0525 |

## Key Observations

1. **Pure-DCE**: Extreme polarization (nearly deterministic)
2. **Raw-LLM**: More dispersed, considers opt-out (P_C > 0)
3. **EFR**: Balanced between DCE and LLM

## Next Steps

Need held-out human choices to compute:
- Multinomial log loss
- Multiclass Brier score
- Bootstrap CIs (2000 reps)
