#!/usr/bin/env python3
"""
Generate Figure 1: Held-out predictive performance with uniform benchmark
"""
import matplotlib.pyplot as plt
import numpy as np

# Data
methods = ['Pure-DCE', 'EFR', 'Raw LLM', 'Uniform']
log_loss = [1.2178, 1.2074, 1.1949, 1.0986]
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

fig, ax = plt.subplots(figsize=(8, 6))

x = np.arange(len(methods))
bars = ax.bar(x, log_loss, color=colors, edgecolor='black', linewidth=1.2)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, log_loss)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add horizontal line for uniform
ax.axhline(y=1.0986, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Uniform benchmark')

ax.set_ylabel('Multinomial Log Loss', fontsize=12)
ax.set_title('Exact-Task Categorical Held-Out Validation\n(N=205 respondents, 6 canonical tasks)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=11)
ax.set_ylim(0, 1.4)

# Add annotation
ax.text(0.5, 0.95, 'All methods > uniform benchmark', 
        transform=ax.transAxes, fontsize=10, style='italic',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('/tmp/behavioral-digital-twins/figures/figure_1_heldout_predictive.png', dpi=300, bbox_inches='tight')
plt.savefig('/tmp/behavioral-digital-twins/figures/figure_1_heldout_predictive.pdf', bbox_inches='tight')
print("✓ Figure 1 saved: figure_1_heldout_predictive.png and .pdf")
