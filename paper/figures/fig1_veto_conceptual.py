#!/usr/bin/env python3
"""
Figure 1: Conceptual Diagram of Structural Veto
Pure conceptual illustration - no implementation details
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set up figure with clean academic style
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')

# Color palette - academic/professional
colors = {
    'perturbation': '#E74C3C',    # Red
    'coherence': '#3498DB',        # Blue
    'resistance': '#27AE60',       # Green
    'transition': '#9B59B6',       # Purple
    'background': '#F8F9FA',       # Light grey
    'arrow': '#2C3E50',            # Dark blue-grey
    'text': '#1A1A2E'              # Near black
}

# Title
ax.text(5, 7.5, 'Structural Veto: Conceptual Flow',
        fontsize=18, fontweight='bold', ha='center', va='center',
        color=colors['text'])

# Central boxes
box_style = dict(boxstyle="round,pad=0.3", facecolor='white',
                 edgecolor=colors['arrow'], linewidth=2)

# Box 1: Internal Perturbation (top left)
box1 = FancyBboxPatch((0.8, 4.5), 2.2, 1.2,
                       boxstyle="round,pad=0.1",
                       facecolor=colors['perturbation'],
                       edgecolor='white', linewidth=2, alpha=0.8)
ax.add_patch(box1)
ax.text(1.9, 5.1, 'Internal\nPerturbation', fontsize=11, fontweight='bold',
        ha='center', va='center', color='white')

# Box 2: Accumulated Coherence (top right)
box2 = FancyBboxPatch((4, 5.5), 2.2, 1.2,
                       boxstyle="round,pad=0.1",
                       facecolor=colors['coherence'],
                       edgecolor='white', linewidth=2, alpha=0.8)
ax.add_patch(box2)
ax.text(5.1, 6.1, 'Accumulated\nCoherence', fontsize=11, fontweight='bold',
        ha='center', va='center', color='white')

# Box 3: Endogenous Resistance (center)
box3 = FancyBboxPatch((4, 3), 2.2, 1.5,
                       boxstyle="round,pad=0.1",
                       facecolor=colors['resistance'],
                       edgecolor='white', linewidth=3, alpha=0.9)
ax.add_patch(box3)
ax.text(5.1, 3.75, 'Endogenous\nResistance', fontsize=12, fontweight='bold',
        ha='center', va='center', color='white')
ax.text(5.1, 3.2, '(VETO)', fontsize=10, fontweight='bold',
        ha='center', va='center', color='white', style='italic')

# Box 4: Modulated Transition (bottom)
box4 = FancyBboxPatch((4, 0.8), 2.2, 1.2,
                       boxstyle="round,pad=0.1",
                       facecolor=colors['transition'],
                       edgecolor='white', linewidth=2, alpha=0.8)
ax.add_patch(box4)
ax.text(5.1, 1.4, 'Modulated\nTransition', fontsize=11, fontweight='bold',
        ha='center', va='center', color='white')

# Box 5: Structural Preservation (right side)
box5 = FancyBboxPatch((7.2, 3), 2.2, 1.5,
                       boxstyle="round,pad=0.1",
                       facecolor=colors['coherence'],
                       edgecolor='white', linewidth=2, alpha=0.7)
ax.add_patch(box5)
ax.text(8.3, 3.75, 'Structural\nPreservation', fontsize=11, fontweight='bold',
        ha='center', va='center', color='white')

# Arrows with annotations
arrow_style = dict(arrowstyle='->', color=colors['arrow'],
                   connectionstyle='arc3,rad=0.1', lw=2)

# Arrow 1: Perturbation -> Resistance
ax.annotate('', xy=(4, 4), xytext=(3, 5),
            arrowprops=dict(arrowstyle='->', color=colors['perturbation'],
                           lw=2.5, connectionstyle='arc3,rad=-0.2'))
ax.text(3.2, 4.7, 'triggers', fontsize=9, color=colors['text'],
        style='italic', rotation=-30)

# Arrow 2: Coherence -> Resistance
ax.annotate('', xy=(5.1, 4.5), xytext=(5.1, 5.5),
            arrowprops=dict(arrowstyle='->', color=colors['coherence'],
                           lw=2.5))
ax.text(5.3, 5, 'informs', fontsize=9, color=colors['text'], style='italic')

# Arrow 3: Resistance -> Transition
ax.annotate('', xy=(5.1, 2), xytext=(5.1, 3),
            arrowprops=dict(arrowstyle='->', color=colors['resistance'],
                           lw=2.5))
ax.text(5.3, 2.5, 'modulates', fontsize=9, color=colors['text'], style='italic')

# Arrow 4: Resistance -> Preservation
ax.annotate('', xy=(7.2, 3.75), xytext=(6.2, 3.75),
            arrowprops=dict(arrowstyle='->', color=colors['resistance'],
                           lw=2.5))
ax.text(6.5, 4.1, 'maintains', fontsize=9, color=colors['text'], style='italic')

# Arrow 5: Feedback loop (Preservation -> Coherence)
ax.annotate('', xy=(6.2, 6.1), xytext=(8.3, 4.5),
            arrowprops=dict(arrowstyle='->', color=colors['coherence'],
                           lw=2, connectionstyle='arc3,rad=0.4',
                           linestyle='dashed', alpha=0.7))
ax.text(7.8, 5.5, 'reinforces', fontsize=9, color=colors['text'],
        style='italic', rotation=40)

# Key principles (bottom annotation)
principles = [
    "• Coherence is a dynamic resource",
    "• Not all transitions are structurally viable",
    "• Resistance emerges endogenously"
]

for i, p in enumerate(principles):
    ax.text(1, 2.2 - i*0.5, p, fontsize=10, color=colors['text'],
            fontweight='normal', style='italic')

# Footer
ax.text(5, 0.2, 'No goals • No rewards • No semantics • Pure structural dynamics',
        fontsize=9, ha='center', va='center', color=colors['arrow'],
        style='italic', alpha=0.8)

plt.tight_layout()
plt.savefig('/root/NEO_EVA/paper/figures/fig1_structural_veto_concept.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/root/NEO_EVA/paper/figures/fig1_structural_veto_concept.pdf',
            bbox_inches='tight', facecolor='white')
print("Figure 1 saved: Structural Veto Conceptual Diagram")
