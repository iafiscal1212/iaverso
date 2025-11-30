#!/usr/bin/env python3
"""
Figure 3: Conceptual Manifold with Forbidden Regions
Stylized artistic representation - NO real topology
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Circle, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap

# Set up figure
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Create artistic background gradient
np.random.seed(123)
x = np.linspace(-5, 5, 200)
y = np.linspace(-4, 4, 160)
X, Y = np.meshgrid(x, y)

# Create abstract "potential landscape"
Z = np.sin(X/2) * np.cos(Y/2) + 0.5 * np.exp(-((X-1)**2 + (Y-1)**2)/4)
Z += 0.3 * np.exp(-((X+2)**2 + (Y-1.5)**2)/3)
Z -= 0.8 * np.exp(-((X+1)**2 + (Y+1.5)**2)/2)  # Forbidden region
Z -= 0.6 * np.exp(-((X-2.5)**2 + (Y+0.5)**2)/1.5)  # Another forbidden region

# Custom colormap: blue (stable) -> white -> red (unstable)
colors_custom = ['#1E3A5F', '#2E5A8F', '#4A90B8', '#87CEEB', '#FFFFFF',
                 '#FFB6C1', '#FF6B6B', '#CC4444', '#8B0000']
cmap = LinearSegmentedColormap.from_list('stability', colors_custom)

# Plot the landscape
im = ax.contourf(X, Y, Z, levels=30, cmap=cmap, alpha=0.7)

# Add contour lines
ax.contour(X, Y, Z, levels=15, colors='white', alpha=0.3, linewidths=0.5)

# Mark "forbidden regions" with hatching
forbidden1 = Ellipse((-1, -1.5), 2.5, 2, angle=15,
                      facecolor='none', edgecolor='#8B0000',
                      linewidth=3, linestyle='--', alpha=0.9)
ax.add_patch(forbidden1)
ax.text(-1, -1.5, 'Forbidden\nRegion', fontsize=10, ha='center', va='center',
        color='#8B0000', fontweight='bold', style='italic')

forbidden2 = Ellipse((2.5, 0.5), 2, 1.5, angle=-20,
                      facecolor='none', edgecolor='#8B0000',
                      linewidth=3, linestyle='--', alpha=0.9)
ax.add_patch(forbidden2)
ax.text(2.5, 0.5, 'High\nDisruption', fontsize=10, ha='center', va='center',
        color='#8B0000', fontweight='bold', style='italic')

# Mark "stability basin"
stability = Ellipse((0, 1.5), 3, 2, angle=0,
                     facecolor='none', edgecolor='#1E3A5F',
                     linewidth=3, alpha=0.9)
ax.add_patch(stability)
ax.text(0, 2.5, 'Stability Basin', fontsize=11, ha='center',
        color='#1E3A5F', fontweight='bold')

# Add conceptual trajectory
t = np.linspace(0, 2*np.pi, 100)
traj_x = 0.5 + 1.5 * np.cos(t) * (1 + 0.3 * np.sin(3*t))
traj_y = 1.2 + 1.2 * np.sin(t) * (1 + 0.2 * np.cos(2*t))
ax.plot(traj_x, traj_y, color='#27AE60', linewidth=2.5, alpha=0.8,
        label='Coherent Trajectory')

# Add arrows showing veto deflection
ax.annotate('', xy=(-0.5, -0.2), xytext=(0.2, 0.5),
            arrowprops=dict(arrowstyle='->', color='#E74C3C',
                           lw=2, connectionstyle='arc3,rad=0.3'))
ax.text(-0.3, 0.3, 'Deflected by\nVeto', fontsize=9, color='#E74C3C',
        style='italic', ha='center')

# Add "attractor" point
ax.scatter([0], [1.5], s=200, color='#1E3A5F', marker='*', zorder=10,
           edgecolor='white', linewidth=1.5)
ax.text(0.3, 1.7, 'Attractor', fontsize=10, color='#1E3A5F', fontweight='bold')

# Labels
ax.set_xlabel('Abstract Dimension 1', fontsize=11, style='italic')
ax.set_ylabel('Abstract Dimension 2', fontsize=11, style='italic')
ax.set_title('Conceptual State Space: Stability and Forbidden Regions',
             fontsize=14, fontweight='bold', pad=15)

# Remove numeric ticks
ax.set_xticks([])
ax.set_yticks([])

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='#27AE60', linewidth=2.5, label='Coherent trajectory'),
    Line2D([0], [0], marker='*', color='#1E3A5F', markersize=15,
           linestyle='None', label='Stable attractor'),
    Ellipse((0,0), 0.3, 0.2, facecolor='none', edgecolor='#8B0000',
            linewidth=2, linestyle='--', label='Forbidden region'),
    Ellipse((0,0), 0.3, 0.2, facecolor='none', edgecolor='#1E3A5F',
            linewidth=2, label='Stability basin')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
          framealpha=0.9)

# Color bar
cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
cbar.set_ticks([])
cbar.ax.set_ylabel('Structural Coherence\n← Low    High →',
                   fontsize=10, rotation=270, labelpad=20)

# Footer
ax.text(0.5, -0.05, 'Artistic representation - not actual manifold topology',
        fontsize=9, ha='center', color='gray', style='italic',
        transform=ax.transAxes)

ax.set_xlim(-5, 5)
ax.set_ylim(-4, 4)

plt.tight_layout()
plt.savefig('/root/NEO_EVA/paper/figures/fig3_conceptual_manifold.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/root/NEO_EVA/paper/figures/fig3_conceptual_manifold.pdf',
            bbox_inches='tight', facecolor='white')
print("Figure 3 saved: Conceptual Manifold")
