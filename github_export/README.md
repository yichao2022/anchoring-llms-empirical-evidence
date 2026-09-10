# LLM Validation Results - 12 Profile Analysis

## Dataset Overview

This dataset contains the results of a Large Language Model (LLM) validation experiment using Qwen2.5:7b (via Ollama) on 12 discrete choice experiment (DCE) vaccine profiles.

## Files

- `llm_raw_outputs_12profiles.csv` - Raw model responses for each call
- `llm_parsed_outputs_12profiles.csv` - Extracted probabilities and binary outcomes
- `analysis.py` - Statistical analysis script

## Experimental Design

- **Total calls**: 120 (12 profiles × 10 repetitions)
- **Model**: Qwen2.5:7b
- **Inference method**: Ollama local API
- **Success rate**: 100%
- **Date**: 2026-09-10

## Profile Attributes

Each profile varies on four dimensions:
- **Efficacy**: 50%, 70%, 95%
- **Side Effects**: 0.1%, 1%, 10%
- **Cash Incentive**: 50 RMB, 800 RMB
- **Origin**: Domestic, Imported

## Key Findings

### High Acceptance Profiles (>0.9 mean probability)
- Profile 8: 95% efficacy, 0.1% side effects, 50 RMB
- Profile 9: 95% efficacy, 1% side effects, 50 RMB
- Profile 10: 95% efficacy, 10% side effects, 50 RMB
- Profile 11: 95% efficacy, 10% side effects, 800 RMB

### Low Acceptance Profiles (<0.6 mean probability)
- Profile 5: 50% efficacy, 10% side effects, 800 RMB (0.510)
- Profile 6: 95% efficacy, 1% side effects, 50 RMB (0.500) - *Anomaly*
- Profile 7: 70% efficacy, 10% side effects, 50 RMB (0.500)

### Interesting Patterns
1. **Efficacy dominance**: 95% efficacy profiles show near-universal acceptance (90-100%)
2. **Side effect sensitivity**: 10% side effects significantly reduce acceptance
3. **Cash compensation**: High cash (800 RMB) partially compensates for moderate efficacy drop
4. **Anomaly**: Profile 6 (high efficacy, low side effects) shows unexpectedly low acceptance

## Statistical Summary

- **Overall mean probability**: 0.722
- **Overall standard deviation**: 0.177
- **Coefficient of variation**: 24.4% (moderate variance)
- **Convergence**: All profiles converged within 10 repetitions (diff < 0.1)

## Usage

Run the analysis script to generate detailed statistics:
```bash
python3 analysis.py
```

## Validation

- ✅ Full task coverage: 12/12 profiles
- ✅ 10 repetitions per profile
- ✅ 100% API success rate
- ✅ Respondent-facing prompts
- ✅ Incremental data saving
- ✅ Convergence verified

## Notes

This data is intended for comparison with human DCE results and for validating LLM as a proxy for human decision-making in health economic studies.
