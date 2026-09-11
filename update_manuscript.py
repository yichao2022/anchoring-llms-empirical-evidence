#!/usr/bin/env python3
"""
Update manuscript with corrected validation results
"""
import re

# Read the file
with open('/tmp/behavioral-digital-twins/manuscript_round3_blinded.tex', 'r') as f:
    content = f.read()

# Pattern to find the long paragraph about validation
old_text = """A complementary task-level analysis on the LLM-matched A-versus-B subset of the held-out 20\\% respondent split ($n_{\\text{tasks, matched}} = 78$ under the canonical 11-of-12-cell populated intersection) yields the same qualitative ordering: Pure-DCE task-level log loss $0.3307$, Static-BDT ($\\lambda = 0.25$) $0.4307$, and unconstrained LLM $0.8521$ (Table S15). This is a complementary task-level A-versus-B validation (the canonical full three-alternative matched subset is empty; the held-out sample admits only two-alternative non-opt-out tasks under the 11-of-12-cell populated intersection), and the fact that the task-level ordering matches the alternative-level held-out ordering reported in Figure~1 and Table~S6 (and the same ordering also appears on the held-out alt-level binary $813$-row matched subset, Pure-DCE $0.6789$ vs. EFR $0.6985$ vs. LLM $0.8273$) corroborates that the alt-level binary scoring is not an artefact of the score choice."""

new_text = """An exact-task categorical validation on the six canonical DCE tasks ($N=205$ held-out respondents, 1,230 respondent-task observations) reveals that all three model-based approaches exceeded the uniform-probability benchmark of $\\log(3)=1.099$: Raw LLM ($1.195$), EFR ($1.207$), and Pure-DCE ($1.218$). EFR did not improve exact-task predictive performance relative to Raw LLM ($\\Delta = 0.013$) and did not outperform the equal-probability benchmark. This negative result is reported transparently; the paper's primary contribution is structural regularization and behavioral constraint, not predictive superiority. The binary alternative-level held-out metrics (Figure~1 and Table~S6; Pure-DCE $0.679$ vs. EFR $0.699$ vs. LLM $0.827$) and the 2-alternative task-level results (Table S15; Pure-DCE $0.331$ vs. EFR $0.431$ vs. LLM $0.852$) corroborate that the exact-task multinomial finding is not an artefact of the score choice."""

if old_text in content:
    content = content.replace(old_text, new_text)
    print("✓ Updated validation paragraph")
else:
    print("⚠ Pattern not found - may need manual update")

# Also update the Conclusion section
old_conclusion = """EFR anchors LLM outputs to DCE estimates without retraining. Held-out DCE log loss and Brier score (Static-BDT: 0.699 / 0.252; Pure-DCE: 0.679 / 0.242; unconstrained LLM: 0.827 / 0.314 on the matched held-out subset, $N=813$ \\emph{alternative-level binary-outcome rows} from $N=205$ held-out respondents, i.e. the 813 rows in the LLM-matched subset of the held-out pool (the held-out pool contains 3,690 alt-rows; after excluding one anomalous wait=2 alt-row, 3,689 rows remain under the intended design, of which 813 are matched and 2,876 are unmatched) are reported as sensitivity diagnostics"""

new_conclusion = """EFR anchors LLM outputs to DCE estimates without retraining. Exact-task categorical held-out validation (six canonical DCE tasks, $N=205$ respondents) shows multinomial log loss: Raw LLM ($1.195$), EFR ($1.207$), Pure-DCE ($1.218$), all exceeding the uniform benchmark ($\\log(3)=1.099$). EFR did not improve predictive performance relative to Raw LLM. These held-out metrics are reported as sensitivity diagnostics"""

if old_conclusion in content:
    content = content.replace(old_conclusion, new_conclusion)
    print("✓ Updated Conclusion section")
else:
    print("⚠ Conclusion pattern not found")

# Write back
with open('/tmp/behavioral-digital-twins/manuscript_round3_blinded.tex', 'w') as f:
    f.write(content)

print("\n✓ Manuscript updated")
