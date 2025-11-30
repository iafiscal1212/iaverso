#!/usr/bin/env python3
"""
Figure 2: Stability Under Perturbation
Conceptual comparison: System with Veto vs System without Veto
NO real data - pure illustrative curves
"""

import matplotlib.pyplot as plt
import numpy as np

# Set up figure
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Generate conceptual curves (NOT real data)
np.random.seed(42)
x = np.linspace(0, 10, 100)

# System WITH veto - maintains stability
# Starts stable, perturbations cause temporary dips, recovers
y_with_veto = np.ones_like(x) * 0.85
# Add perturbation responses that recover
perturbations = [2, 4, 6, 8]
for p in perturbations:
    dip = 0.15 * np.exp(-2 * (x - p)**2)
    recovery = 0.12 * np.exp(-0.5 * (x - p - 0.5)**2) * (x > p)
    y_with_veto = y_with_veto - dip + recovery
# Add slight noise for realism
y_with_veto += np.random.normal(0, 0.02, len(x))
y_with_veto = np.clip(y_with_veto, 0, 1)

# System WITHOUT veto - progressive degradation
y_without_veto = np.ones_like(x) * 0.85
# Each perturbation causes permanent damage
cumulative_damage = 0
for i, p in enumerate(perturbations):
    damage = 0.18 * (1 + i * 0.3)  # Increasing damage
    cumulative_damage += damage
    mask = x >= p
    y_without_veto[mask] = y_without_veto[mask] - damage * np.exp(-0.3 * (x[mask] - p))
# Add noise and trend toward collapse
y_without_veto += np.random.normal(0, 0.02, len(x))
y_without_veto = y_without_veto - 0.02 * x  # Drift downward
y_without_veto = np.clip(y_without_veto, 0, 1)

# Plotting
ax.plot(x, y_with_veto, color='#27AE60', linewidth=3, label='With Structural Veto',
        zorder=5)
ax.plot(x, y_without_veto, color='#7F8C8D', linewidth=3, label='Without Veto',
        linestyle='--', alpha=0.8, zorder=4)

# Mark perturbation events
for i, p in enumerate(perturbations):
    ax.axvline(x=p, color='#E74C3C', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.scatter([p], [0.95], marker='v', color='#E74C3C', s=100, zorder=10)
    if i == 0:
        ax.text(p, 0.98, 'Perturbation', fontsize=9, ha='center',
                color='#E74C3C', style='italic')

# Collapse threshold
ax.axhline(y=0.3, color='#E74C3C', linestyle='--', alpha=0.4, linewidth=2)
ax.text(9.5, 0.32, 'Collapse\nThreshold', fontsize=9, ha='center',
        color='#E74C3C', alpha=0.7)

# Stability zone
ax.fill_between(x, 0.7, 1.0, alpha=0.1, color='#27AE60')
ax.text(0.5, 0.92, 'Stability Zone', fontsize=9, color='#27AE60', alpha=0.8)

# Annotations
ax.annotate('Recovery', xy=(4.8, 0.78), xytext=(5.5, 0.65),
            fontsize=10, color='#27AE60',
            arrowprops=dict(arrowstyle='->', color='#27AE60', lw=1.5))

ax.annotate('Cumulative\nDamage', xy=(7, 0.35), xytext=(8, 0.5),
            fontsize=10, color='#7F8C8D',
            arrowprops=dict(arrowstyle='->', color='#7F8C8D', lw=1.5))

# Labels and formatting
ax.set_xlabel('Perturbation Intensity (conceptual)', fontsize=12)
ax.set_ylabel('Structural Coherence (abstract index)', fontsize=12)
ax.set_title('Stability Under Perturbation: With vs Without Structural Veto',
             fontsize=14, fontweight='bold', pad=15)

ax.set_xlim(0, 10)
ax.set_ylim(0, 1.05)
ax.set_xticks([])  # No numeric ticks - conceptual only
ax.set_yticks([0, 0.5, 1.0])
ax.set_yticklabels(['Low', 'Medium', 'High'])

ax.legend(loc='lower left', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3)

# Footer note
ax.text(5, -0.08, 'Conceptual illustration - not empirical data',
        fontsize=9, ha='center', color='gray', style='italic',
        transform=ax.transAxes)

plt.tight_layout()
plt.savefig('/root/NEO_EVA/paper/figures/fig2_stability_curves.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/root/NEO_EVA/paper/figures/fig2_stability_curves.pdf',
            bbox_inches='tight', facecolor='white')
print("Figure 2 saved: Stability Curves")
