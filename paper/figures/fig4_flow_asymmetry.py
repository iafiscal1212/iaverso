#!/usr/bin/env python3
"""
Figure 4: Conceptual Flow Asymmetry
Illustrating irreversibility and directional dynamics
NO numerical data - pure conceptual illustration
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.colors import LinearSegmentedColormap

# Set up figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Color palette
colors = {
    'forward': '#27AE60',
    'backward': '#E74C3C',
    'neutral': '#3498DB',
    'arrow': '#2C3E50',
    'light': '#ECF0F1'
}

# === LEFT PANEL: Flow Direction Diagram ===
ax1 = axes[0]
ax1.set_xlim(-3, 3)
ax1.set_ylim(-3, 3)
ax1.set_aspect('equal')
ax1.axis('off')
ax1.set_title('Directional Flow (Forward ≠ Backward)', fontsize=13, fontweight='bold', pad=10)

# Create flow field visualization
np.random.seed(42)
n_arrows = 12
theta = np.linspace(0, 2*np.pi, n_arrows, endpoint=False)
r = 1.8

# Forward flow - dominant direction (clockwise bias)
for i, t in enumerate(theta):
    x = r * np.cos(t)
    y = r * np.sin(t)
    # Asymmetric flow - stronger in one direction
    dx = -0.4 * np.sin(t) + 0.15 * np.cos(t)
    dy = 0.4 * np.cos(t) + 0.15 * np.sin(t)

    # Arrow thickness based on flow strength
    width = 0.08 + 0.04 * (1 + np.sin(t))
    ax1.annotate('', xy=(x + dx, y + dy), xytext=(x, y),
                arrowprops=dict(arrowstyle='->', color=colors['forward'],
                               lw=2 + np.sin(t), mutation_scale=15))

# Add reverse arrows (weaker)
for i in range(4):
    t = theta[i * 3]
    x = r * np.cos(t) * 0.6
    y = r * np.sin(t) * 0.6
    dx = 0.2 * np.sin(t)
    dy = -0.2 * np.cos(t)
    ax1.annotate('', xy=(x + dx, y + dy), xytext=(x, y),
                arrowprops=dict(arrowstyle='->', color=colors['backward'],
                               lw=1.5, alpha=0.5, linestyle='--'))

# Central vortex
circle = Circle((0, 0), 0.4, facecolor=colors['light'], edgecolor=colors['arrow'],
                linewidth=2, zorder=10)
ax1.add_patch(circle)
ax1.text(0, 0, 'Core', fontsize=10, ha='center', va='center', fontweight='bold')

# Legend elements
ax1.plot([], [], color=colors['forward'], linewidth=3, label='Dominant flow')
ax1.plot([], [], color=colors['backward'], linewidth=2, linestyle='--',
         alpha=0.5, label='Suppressed reverse')
ax1.legend(loc='lower left', fontsize=10)

# Annotations
ax1.text(0, 2.6, 'Irreversibility Signature:', fontsize=11, ha='center',
         color=colors['arrow'], fontweight='bold')
ax1.text(0, 2.2, 'Forward transitions ≫ Backward transitions',
         fontsize=10, ha='center', color=colors['arrow'], style='italic')

# === RIGHT PANEL: Asymmetry Visualization ===
ax2 = axes[1]

# Create abstract asymmetry bars
categories = ['Forward\nTransitions', 'Backward\nTransitions', 'Net\nAsymmetry']
# Conceptual values (not real data)
forward_val = 0.75
backward_val = 0.25
asymmetry_val = 0.50

bar_colors = [colors['forward'], colors['backward'], colors['neutral']]
bar_values = [forward_val, backward_val, asymmetry_val]

bars = ax2.bar(categories, bar_values, color=bar_colors, edgecolor='white',
               linewidth=2, alpha=0.8)

# Add "expected" reference line
ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, alpha=0.6,
            label='Symmetric expectation')

# Annotations on bars
ax2.text(0, forward_val + 0.03, 'High', ha='center', fontsize=11, fontweight='bold',
         color=colors['forward'])
ax2.text(1, backward_val + 0.03, 'Low', ha='center', fontsize=11, fontweight='bold',
         color=colors['backward'])
ax2.text(2, asymmetry_val + 0.03, 'Strong', ha='center', fontsize=11, fontweight='bold',
         color=colors['neutral'])

# Formatting
ax2.set_ylabel('Relative Frequency (abstract)', fontsize=11)
ax2.set_title('Flow Asymmetry: Markovian Null Rejected', fontsize=13,
              fontweight='bold', pad=10)
ax2.set_ylim(0, 1)
ax2.set_yticks([0, 0.5, 1])
ax2.set_yticklabels(['None', 'Symmetric', 'Full'])
ax2.legend(loc='upper right', fontsize=10)

# Add explanatory text
ax2.text(1, -0.18, 'The system exhibits statistically significant\nirreversibility, '
         'incompatible with Markovian null models.',
         ha='center', fontsize=10, color=colors['arrow'], style='italic',
         transform=ax2.transData)

# Main title
fig.suptitle('Figure 4: Flow Asymmetry and Irreversibility',
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('/root/NEO_EVA/paper/figures/fig4_flow_asymmetry.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/root/NEO_EVA/paper/figures/fig4_flow_asymmetry.pdf',
            bbox_inches='tight', facecolor='white')
print("Figure 4 saved: Flow Asymmetry")
