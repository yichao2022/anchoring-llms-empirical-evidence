# Self-Audit: Wuhan DCE Design Provenance
# Owner: Yichao Jin (Data Collector)
# Date: 2026-09-09
# Status: AWAITING OWNER RESPONSE

## Reconstructed Design Structure (from data forensics)

Based on analysis of dce_encoded.csv (N=18,485 after excluding wait=2 anomaly):

- **7 distinct design blocks**
- **6 choice tasks per block**
- **42 distinct choice-set compositions** (A/B/Opt-out signatures)
- **84 unique non-opt-out profiles**
- **Block respondent counts**: 250, 241, 222, 95, 83, 80, 55

## Questions for Data Collector

### 1. DESIGN INTENT
- [ ] Was the seven-block structure the **intended administered design**?
- [ ] Or is this pattern a data artifact / reconstruction error?

### 2. BLOCK ASSIGNMENT
- [ ] How were respondents assigned to the seven blocks?
  - [ ] Randomization
  - [ ] Stratification (by what variables?)
  - [ ] Survey version / wave
  - [ ] Other: ___________

### 3. ORIGINAL DESIGN DOCUMENTATION
Please locate and attach:
- [ ] Original DCE design matrix (blocks × tasks × profiles)
- [ ] Questionnaire/survey instrument
- [ ] Block assignment file or randomization log

### 4. QUESTIONNAIRE WORDING (CRITICAL)
Current analysis uses Table S3 coding dictionary. Please confirm exact respondent-facing wording:

| Attribute | Current Guess | Actual Wording |
|-----------|--------------|----------------|
| Waiting time | "Waiting time: X months" | ? |
| Vaccine effectiveness | "Vaccine effectiveness: X%" | ? |
| Side effects | "Risk of side effects: X%" | ? |
| Cash incentive | "Cash incentive: X RMB" | ? |
| Vaccine origin | "Vaccine origin: Domestic/Imported" | ? |
| Opt-out alternative | "Do not receive the vaccine" | ? |
| Task introduction | "You are evaluating..." | ? |

### 5. RANDOMIZATION
- [ ] Was A/B alternative ordering randomized?
- [ ] Were tasks presented in fixed or randomized order?
- [ ] Were blocks randomized across respondents?

### 6. wait=2 ANOMALY
- [ ] Was wait=2 months an **intended design level**?
- [ ] Or is it a **data-recording anomaly**?
- [ ] Current treatment: Excluded as anomaly (1 row from 1 respondent)

### 7. PROFILE CONFIRMATION
- [ ] Were all 84 reconstructed non-opt-out profiles **intentionally administered**?
- [ ] Or should some be excluded as errors?

## Blocked Actions

Pending answers above:
- [ ] Update CSV wording_status
- [ ] Unfreeze API pipeline
- [ ] Execute 420 Qwen calls
- [ ] Update manuscript

## Timeline

Please respond at your earliest convenience so validation can proceed.
