import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Define colors
colors = plt.cm.Blues(np.linspace(0, 1, 7))

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Set limits and remove ticks
ax.set_xlim(0, 8)
ax.set_ylim(0, 2)
ax.set_xticks([])
ax.set_yticks([])

# Draw rectangles
previous_layers = [0.1, 0.3, 0.05, 0.25]
upcoming_layers = [0.05, 0.1, 0.15]

for i, value in enumerate(previous_layers):
    rect = patches.Rectangle((i, 1), 0.8, 1, linewidth=1, edgecolor='none', facecolor=colors[i])
    ax.add_patch(rect)
    ax.text(i + 0.4, 1.5, f'{value}', ha='center', va='center', color='black', fontsize=12)

# Current Layer
rect = patches.Rectangle((4, 1), 0.8, 1, linewidth=1, edgecolor='none', facecolor=colors[4])
ax.add_patch(rect)
ax.text(4.4, 1.5, 'Current Layer', ha='center', va='center', color='black', fontsize=12)
ax.annotate('', xy=(4.4, 1.05), xytext=(4.4, 0.95), arrowprops=dict(arrowstyle='->', color='black'))

for i, value in enumerate(upcoming_layers):
    rect = patches.Rectangle((5 + i, 1), 0.8, 1, linewidth=1, edgecolor='none', facecolor=colors[5 + i])
    ax.add_patch(rect)
    ax.text(5.4 + i, 1.5, f'{value}', ha='center', va='center', color='black', fontsize=12)

# Labels for groups
ax.text(2, 0.5, 'Previous Layers', ha='center', va='center', color='black', fontsize=14)
ax.text(6, 0.5, 'Upcoming Layers', ha='center', va='center', color='black', fontsize=14)

# Gradient Legend
gradient = np.linspace(0, 1, 256).reshape(1, 256)
ax.imshow(gradient, aspect='auto', cmap='Blues', extent=[0, 8, 0, 0.2])
ax.text(0, 0.3, 'Shallow', ha='left', va='center', color='black', fontsize=14)
ax.text(8, 0.3, 'Deep', ha='right', va='center', color='black', fontsize=14)

plt.box(False)
plt.show()
