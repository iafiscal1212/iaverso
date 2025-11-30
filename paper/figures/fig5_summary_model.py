#!/usr/bin/env python3
"""
Figure 5: Summary Conceptual Model of Structural Veto Theory
Integrated illustration showing all key components
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Ellipse, Wedge
import numpy as np

# Set up figure
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Color palette
colors = {
    'coherence': '#3498DB',
    'resistance': '#27AE60',
    'irreversibility': '#9B59B6',
    'perturbation': '#E74C3C',
    'agency': '#F39C12',
    'veto': '#1ABC9C',
    'background': '#F8F9FA',
    'text': '#2C3E50',
    'light': '#ECF0F1'
}

# Title
ax.text(7, 9.5, 'STRUCTURAL VETO: Integrated Conceptual Model',
        fontsize=18, fontweight='bold', ha='center', color=colors['text'])

# === CENTRAL HEXAGON: Core Theory ===
# Create hexagonal arrangement of core concepts
center_x, center_y = 7, 5.5
radius = 2.5

concepts = [
    ('COHERENCE', colors['coherence'], 'Dynamic\nResource'),
    ('RESISTANCE', colors['resistance'], 'Endogenous\nResponse'),
    ('IRREVERSIBILITY', colors['irreversibility'], 'Temporal\nAsymmetry'),
    ('FORBIDDEN\nZONES', colors['perturbation'], 'Structural\nBoundaries'),
    ('MINIMAL\nAGENCY', colors['agency'], 'Emergent\nProperty'),
    ('VETO', colors['veto'], 'Core\nMechanism')
]

angles = np.linspace(np.pi/2, np.pi/2 + 2*np.pi, len(concepts), endpoint=False)

for i, (name, color, desc) in enumerate(concepts):
    x = center_x + radius * np.cos(angles[i])
    y = center_y + radius * np.sin(angles[i])

    # Draw concept circle
    circle = Circle((x, y), 0.8, facecolor=color, edgecolor='white',
                    linewidth=3, alpha=0.85, zorder=5)
    ax.add_patch(circle)

    # Add text
    ax.text(x, y + 0.1, name, fontsize=9, fontweight='bold',
            ha='center', va='center', color='white', zorder=6)
    ax.text(x, y - 0.25, desc, fontsize=7, ha='center', va='center',
            color='white', style='italic', zorder=6)

    # Connect to center
    ax.plot([center_x, x], [center_y, y], color=color, linewidth=2,
            alpha=0.4, zorder=1)

# Central hub
central = Circle((center_x, center_y), 0.6, facecolor=colors['text'],
                  edgecolor='white', linewidth=3, zorder=10)
ax.add_patch(central)
ax.text(center_x, center_y, 'FORM', fontsize=11, fontweight='bold',
        ha='center', va='center', color='white', zorder=11)

# === LEFT SIDE: Input/Trigger ===
# Perturbation input
ax.annotate('', xy=(4, 5.5), xytext=(1.5, 5.5),
            arrowprops=dict(arrowstyle='->', color=colors['perturbation'],
                           lw=3, mutation_scale=20))
ax.text(1.5, 6.2, 'INTERNAL\nPERTURBATION', fontsize=10, fontweight='bold',
        ha='center', color=colors['perturbation'])
ax.text(1.5, 4.8, '(any state\ntransition)', fontsize=8, ha='center',
        color=colors['text'], style='italic')

# === RIGHT SIDE: Output/Result ===
# Modulated transition output
ax.annotate('', xy=(12.5, 5.5), xytext=(10, 5.5),
            arrowprops=dict(arrowstyle='->', color=colors['resistance'],
                           lw=3, mutation_scale=20))
ax.text(12.5, 6.2, 'MODULATED\nTRANSITION', fontsize=10, fontweight='bold',
        ha='center', color=colors['resistance'])
ax.text(12.5, 4.8, '(coherence\npreserved)', fontsize=8, ha='center',
        color=colors['text'], style='italic')

# === BOTTOM: Key Principles ===
principles_y = 1.5
principles = [
    ("No goals", "Structure, not objectives"),
    ("No rewards", "Geometry, not optimization"),
    ("No semantics", "Form, not meaning"),
    ("No supervision", "Endogenous, not imposed")
]

for i, (title, desc) in enumerate(principles):
    x = 2 + i * 3.3
    # Box
    box = FancyBboxPatch((x - 1.2, principles_y - 0.5), 2.4, 1.2,
                          boxstyle="round,pad=0.05",
                          facecolor=colors['light'],
                          edgecolor=colors['text'],
                          linewidth=1.5, alpha=0.9)
    ax.add_patch(box)
    ax.text(x, principles_y + 0.2, title, fontsize=10, fontweight='bold',
            ha='center', va='center', color=colors['text'])
    ax.text(x, principles_y - 0.15, desc, fontsize=8, ha='center',
            va='center', color=colors['text'], style='italic')

ax.text(7, 0.6, 'FOUNDATIONAL PRINCIPLES', fontsize=11, fontweight='bold',
        ha='center', color=colors['text'])

# === TOP: Theoretical Position ===
theories_y = 8.3
theories = [
    ("IIT", "measures\nintegration"),
    ("FEP", "minimizes\nsurprise"),
    ("VETO", "preserves\nform")
]

ax.text(7, 8.8, 'Theoretical Landscape', fontsize=11, fontweight='bold',
        ha='center', color=colors['text'])

for i, (name, desc) in enumerate(theories):
    x = 4 + i * 3
    color = colors['veto'] if name == 'VETO' else colors['light']
    edge = colors['veto'] if name == 'VETO' else colors['text']
    weight = 'bold' if name == 'VETO' else 'normal'

    box = FancyBboxPatch((x - 0.9, theories_y - 0.4), 1.8, 0.9,
                          boxstyle="round,pad=0.05",
                          facecolor=color,
                          edgecolor=edge,
                          linewidth=2, alpha=0.9)
    ax.add_patch(box)
    text_color = 'white' if name == 'VETO' else colors['text']
    ax.text(x, theories_y + 0.15, name, fontsize=10, fontweight='bold',
            ha='center', va='center', color=text_color)
    ax.text(x, theories_y - 0.15, desc, fontsize=7, ha='center',
            va='center', color=text_color)

# Connecting lines between theories
ax.plot([4.9, 6.1], [theories_y, theories_y], color=colors['text'],
        linewidth=1, linestyle=':', alpha=0.5)
ax.plot([7.9, 9.1], [theories_y, theories_y], color=colors['text'],
        linewidth=1, linestyle=':', alpha=0.5)

# === AGI Implication Arrow ===
ax.annotate('', xy=(13.5, 3), xytext=(10.5, 4.5),
            arrowprops=dict(arrowstyle='->', color=colors['agency'],
                           lw=2.5, connectionstyle='arc3,rad=-0.2'))
ax.text(12.8, 3.5, 'Path to\nMinimal AGI', fontsize=9, fontweight='bold',
        ha='center', color=colors['agency'], rotation=-30)

# Footer
ax.text(7, 0.1, 'Structural Veto Theory — Carmen Esteban, 2025',
        fontsize=9, ha='center', color='gray', style='italic')

plt.tight_layout()
plt.savefig('/root/NEO_EVA/paper/figures/fig5_summary_model.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/root/NEO_EVA/paper/figures/fig5_summary_model.pdf',
            bbox_inches='tight', facecolor='white')
print("Figure 5 saved: Summary Conceptual Model")
