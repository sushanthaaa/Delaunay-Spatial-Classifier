#!/usr/bin/env python3
"""
Generate 2D Buckets Dynamic Insertion Demonstration

This script creates a figure demonstrating how 2D Buckets classify
dynamic insertion queries in O(1) time, showing both:
1. Single-class bucket (top-left, purely green region)
2. Multi-class bucket (Bucket [0,6], split by green/blue boundary)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D
from scipy.spatial import Delaunay, Voronoi

# Use the same style as publication figures
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.dpi'] = 150

# Color scheme matching publication figures
CLASS_COLORS = {
    0: '#2E8B57',  # Green (Class 0)
    1: '#4A90D9',  # Blue (Class 1)
    2: '#FFD700'   # Yellow (Class 2)
}
CLASS_NAMES = {0: 'Green', 1: 'Blue', 2: 'Yellow'}

def load_wine_data():
    """Load wine training data."""
    train_path = 'data/train/wine_train.csv'
    data = np.loadtxt(train_path, delimiter=',')
    X = data[:, :2]
    y = data[:, 2].astype(int)
    return X, y

def classify_point(X, y, point):
    """Classify a point using nearest vertex."""
    distances = np.sqrt(np.sum((X - point)**2, axis=1))
    nearest_idx = np.argmin(distances)
    return y[nearest_idx]

def generate_dynamic_insertion_figure(output_path='figures_publication/wine/6_dynamic_insertion_demo.png'):
    """Generate dynamic insertion demonstration figure."""
    
    # Load data
    X, y = load_wine_data()
    n_points = len(X)
    
    # Compute SRR grid
    k = int(np.ceil(np.sqrt(n_points)))
    min_x, max_x = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    min_y, max_y = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    step_x = (max_x - min_x) / k
    step_y = (max_y - min_y) / k
    
    print(f"SRR Grid: {k}×{k}, step=({step_x:.3f}, {step_y:.3f})")
    
    # Create figure with 2x2 layout
    fig = plt.figure(figsize=(16, 14))
    
    # ============================================
    # Panel (a): Full view with two insertion points marked
    # ============================================
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title('(a) Overview: Two Dynamic Insertion Queries', fontweight='bold', fontsize=13)
    
    # Draw SRR grid
    for i in range(k + 1):
        ax1.axvline(min_x + i * step_x, color='gray', alpha=0.3, linewidth=0.5)
        ax1.axhline(min_y + i * step_y, color='gray', alpha=0.3, linewidth=0.5)
    
    # Plot training points
    for cls in [0, 1, 2]:
        mask = y == cls
        if mask.sum() > 0:
            marker = ['o', 's', '^'][cls]
            ax1.scatter(X[mask, 0], X[mask, 1], c=CLASS_COLORS[cls], 
                       marker=marker, s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Compute Voronoi for decision boundaries
    try:
        vor = Voronoi(X)
        # Draw decision boundaries where Voronoi edges separate different classes
        for ridge_idx, (p1, p2) in enumerate(vor.ridge_vertices):
            if -1 in [p1, p2]:
                continue
            point_idxs = vor.ridge_points[ridge_idx]
            if y[point_idxs[0]] != y[point_idxs[1]]:
                v1, v2 = vor.vertices[p1], vor.vertices[p2]
                ax1.plot([v1[0], v2[0]], [v1[1], v2[1]], 'k-', linewidth=2.5, alpha=0.8)
    except:
        pass
    
    # Define query points
    # Query 1: Top-left in pure GREEN region (single-class bucket)
    query1 = (-3.5, 2.0)  # Clearly in green region
    query1_bucket = (10, 1)  # Row 10, Col 1 (top-left area)
    
    # Query 2: At the GREEN-BLUE boundary (multi-class bucket)
    # Based on data analysis, green-blue boundary is around x=1.2-1.4, y=0.2-0.5
    # Green is in upper/right side (positive y), Blue is in lower side
    query2_green = (1.5, 0.5)   # In green side of the boundary
    query2_blue = (1.3, -0.5)   # In blue side of the boundary (lower y)
    
    # Mark query points with large markers
    ax1.scatter(*query1, c='red', s=300, marker='*', edgecolors='black', linewidth=2, zorder=10)
    ax1.annotate('Query A', query1, xytext=(query1[0]-0.7, query1[1]+0.4), fontsize=11, 
                fontweight='bold', color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    ax1.scatter(*query2_green, c='red', s=300, marker='*', edgecolors='black', linewidth=2, zorder=10)
    ax1.scatter(*query2_blue, c='red', s=300, marker='*', edgecolors='black', linewidth=2, zorder=10)
    ax1.annotate('Query B1', query2_green, xytext=(query2_green[0]-1.0, query2_green[1]-0.5), fontsize=11,
                fontweight='bold', color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax1.annotate('Query B2', query2_blue, xytext=(query2_blue[0]+0.3, query2_blue[1]+0.6), fontsize=11,
                fontweight='bold', color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    # Highlight the two buckets of interest
    # Bucket for Query A (single-class)
    bucket_a_col = int((query1[0] - min_x) / step_x)
    bucket_a_row = int((query1[1] - min_y) / step_y)
    bucket_a_x = min_x + bucket_a_col * step_x
    bucket_a_y = min_y + bucket_a_row * step_y
    rect_a = Rectangle((bucket_a_x, bucket_a_y), step_x, step_y, 
                       fill=False, edgecolor='red', linewidth=3, linestyle='--')
    ax1.add_patch(rect_a)
    ax1.text(bucket_a_x + step_x/2, bucket_a_y + step_y + 0.1, f'Bucket A\n[{bucket_a_row},{bucket_a_col}]',
            ha='center', fontsize=9, color='red', fontweight='bold')
    
    # Bucket for Query B (multi-class)
    bucket_b_col = int((query2_green[0] - min_x) / step_x)
    bucket_b_row = int((query2_green[1] - min_y) / step_y)
    bucket_b_x = min_x + bucket_b_col * step_x
    bucket_b_y = min_y + bucket_b_row * step_y
    rect_b = Rectangle((bucket_b_x, bucket_b_y), step_x, step_y,
                       fill=False, edgecolor='purple', linewidth=3, linestyle='--')
    ax1.add_patch(rect_b)
    ax1.text(bucket_b_x + step_x/2, bucket_b_y - 0.25, f'Bucket B\n[{bucket_b_row},{bucket_b_col}]',
            ha='center', fontsize=9, color='purple', fontweight='bold')
    
    ax1.set_xlim(min_x, max_x)
    ax1.set_ylim(min_y, max_y)
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    
    # ============================================
    # Panel (b): Single-Class Bucket Detail (Query A)
    # ============================================
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title('(b) Single-Class Bucket: Query A → Green', fontweight='bold', fontsize=13)
    
    # Zoom into bucket A area
    margin = step_x * 1.5
    zoom_min_x = bucket_a_x - margin
    zoom_max_x = bucket_a_x + step_x + margin
    zoom_min_y = bucket_a_y - margin
    zoom_max_y = bucket_a_y + step_y + margin
    
    # Color the bucket
    rect_fill = Rectangle((bucket_a_x, bucket_a_y), step_x, step_y,
                          facecolor=CLASS_COLORS[0], alpha=0.4, edgecolor='red', linewidth=3)
    ax2.add_patch(rect_fill)
    
    # Draw adjacent grid cells
    for r in range(-1, 3):
        for c in range(-1, 3):
            bx = bucket_a_x + c * step_x
            by = bucket_a_y + r * step_y
            if bx >= min_x and by >= min_y and bx < max_x and by < max_y:
                rect = Rectangle((bx, by), step_x, step_y,
                                fill=False, edgecolor='gray', linewidth=0.5)
                ax2.add_patch(rect)
    
    # Plot nearby training points
    mask = ((X[:, 0] >= zoom_min_x) & (X[:, 0] <= zoom_max_x) &
            (X[:, 1] >= zoom_min_y) & (X[:, 1] <= zoom_max_y))
    for cls in [0, 1, 2]:
        cls_mask = mask & (y == cls)
        if cls_mask.sum() > 0:
            marker = ['o', 's', '^'][cls]
            ax2.scatter(X[cls_mask, 0], X[cls_mask, 1], c=CLASS_COLORS[cls],
                       marker=marker, s=100, edgecolors='black', linewidth=1, zorder=5)
    
    # Mark query point
    ax2.scatter(*query1, c='red', s=400, marker='*', edgecolors='black', linewidth=2, zorder=10)
    
    # Add classification result
    pred_class = classify_point(X, y, np.array(query1))
    ax2.text(query1[0], query1[1] - 0.2, f'→ Class {pred_class} ({CLASS_NAMES[pred_class]})',
            ha='center', fontsize=12, fontweight='bold', color=CLASS_COLORS[pred_class])
    
    # Algorithm box
    algo_text = """O(1) Classification:
1. Bucket lookup → O(1)
2. num_classes = 1
3. Return dominant_class
   = Green (0)

Total: O(1)"""
    ax2.text(0.02, 0.98, algo_text, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='green', linewidth=2))
    
    ax2.set_xlim(zoom_min_x, zoom_max_x)
    ax2.set_ylim(zoom_min_y, zoom_max_y)
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    
    # ============================================
    # Panel (c): Multi-Class Bucket Detail (Query B - Green side)
    # ============================================
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title('(c) Multi-Class Bucket: Query B1 → Green', fontweight='bold', fontsize=13)
    
    # Zoom into bucket B area
    zoom_min_x = bucket_b_x - margin
    zoom_max_x = bucket_b_x + step_x + margin
    zoom_min_y = bucket_b_y - margin
    zoom_max_y = bucket_b_y + step_y + margin
    
    # Create fine grid for class coloring
    fine_n = 30
    fine_x = np.linspace(bucket_b_x, bucket_b_x + step_x, fine_n)
    fine_y = np.linspace(bucket_b_y, bucket_b_y + step_y, fine_n)
    
    for i in range(fine_n - 1):
        for j in range(fine_n - 1):
            cx = (fine_x[i] + fine_x[i+1]) / 2
            cy = (fine_y[j] + fine_y[j+1]) / 2
            cls = classify_point(X, y, np.array([cx, cy]))
            rect = Rectangle((fine_x[i], fine_y[j]),
                            fine_x[i+1] - fine_x[i],
                            fine_y[j+1] - fine_y[j],
                            facecolor=CLASS_COLORS[cls], edgecolor='none', alpha=0.5)
            ax3.add_patch(rect)
    
    # Draw bucket boundary
    rect = Rectangle((bucket_b_x, bucket_b_y), step_x, step_y,
                     fill=False, edgecolor='purple', linewidth=3)
    ax3.add_patch(rect)
    
    # Draw decision boundary through bucket
    try:
        for ridge_idx, (p1, p2) in enumerate(vor.ridge_vertices):
            if -1 in [p1, p2]:
                continue
            point_idxs = vor.ridge_points[ridge_idx]
            if y[point_idxs[0]] != y[point_idxs[1]]:
                v1, v2 = vor.vertices[p1], vor.vertices[p2]
                if (min(v1[0], v2[0]) <= bucket_b_x + step_x and 
                    max(v1[0], v2[0]) >= bucket_b_x and
                    min(v1[1], v2[1]) <= bucket_b_y + step_y and
                    max(v1[1], v2[1]) >= bucket_b_y):
                    ax3.plot([v1[0], v2[0]], [v1[1], v2[1]], 'k-', linewidth=3, alpha=0.9)
    except:
        pass
    
    # Plot nearby training points
    mask = ((X[:, 0] >= zoom_min_x) & (X[:, 0] <= zoom_max_x) &
            (X[:, 1] >= zoom_min_y) & (X[:, 1] <= zoom_max_y))
    for cls in [0, 1, 2]:
        cls_mask = mask & (y == cls)
        if cls_mask.sum() > 0:
            marker = ['o', 's', '^'][cls]
            ax3.scatter(X[cls_mask, 0], X[cls_mask, 1], c=CLASS_COLORS[cls],
                       marker=marker, s=100, edgecolors='black', linewidth=1, zorder=5)
    
    # Mark query point B1 (green side)
    ax3.scatter(*query2_green, c='red', s=400, marker='*', edgecolors='black', linewidth=2, zorder=10)
    pred_class = classify_point(X, y, np.array(query2_green))
    ax3.annotate(f'B1 → {CLASS_NAMES[pred_class]}', query2_green, 
                xytext=(query2_green[0] - 0.4, query2_green[1] + 0.3),
                fontsize=12, fontweight='bold', color=CLASS_COLORS[pred_class],
                arrowprops=dict(arrowstyle='->', color=CLASS_COLORS[pred_class], lw=2))
    
    # Algorithm box
    algo_text = """O(1) Classification:
1. Bucket lookup → O(1)
2. num_classes = 2
3. Check linked list:
   → Green region polygon
   → Point-in-polygon: YES
4. Return Green (0)

Total: O(1)"""
    ax3.text(0.02, 0.98, algo_text, transform=ax3.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='green', linewidth=2))
    
    ax3.set_xlim(zoom_min_x, zoom_max_x)
    ax3.set_ylim(zoom_min_y, zoom_max_y)
    ax3.set_xlabel('Feature 1')
    ax3.set_ylabel('Feature 2')
    
    # ============================================
    # Panel (d): Multi-Class Bucket Detail (Query B - Blue side)
    # ============================================
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title('(d) Multi-Class Bucket: Query B2 → Blue', fontweight='bold', fontsize=13)
    
    # Same zoom as panel c
    for i in range(fine_n - 1):
        for j in range(fine_n - 1):
            cx = (fine_x[i] + fine_x[i+1]) / 2
            cy = (fine_y[j] + fine_y[j+1]) / 2
            cls = classify_point(X, y, np.array([cx, cy]))
            rect = Rectangle((fine_x[i], fine_y[j]),
                            fine_x[i+1] - fine_x[i],
                            fine_y[j+1] - fine_y[j],
                            facecolor=CLASS_COLORS[cls], edgecolor='none', alpha=0.5)
            ax4.add_patch(rect)
    
    # Draw bucket boundary
    rect = Rectangle((bucket_b_x, bucket_b_y), step_x, step_y,
                     fill=False, edgecolor='purple', linewidth=3)
    ax4.add_patch(rect)
    
    # Draw decision boundary
    try:
        for ridge_idx, (p1, p2) in enumerate(vor.ridge_vertices):
            if -1 in [p1, p2]:
                continue
            point_idxs = vor.ridge_points[ridge_idx]
            if y[point_idxs[0]] != y[point_idxs[1]]:
                v1, v2 = vor.vertices[p1], vor.vertices[p2]
                if (min(v1[0], v2[0]) <= bucket_b_x + step_x and 
                    max(v1[0], v2[0]) >= bucket_b_x and
                    min(v1[1], v2[1]) <= bucket_b_y + step_y and
                    max(v1[1], v2[1]) >= bucket_b_y):
                    ax4.plot([v1[0], v2[0]], [v1[1], v2[1]], 'k-', linewidth=3, alpha=0.9)
    except:
        pass
    
    # Plot nearby training points
    for cls in [0, 1, 2]:
        cls_mask = mask & (y == cls)
        if cls_mask.sum() > 0:
            marker = ['o', 's', '^'][cls]
            ax4.scatter(X[cls_mask, 0], X[cls_mask, 1], c=CLASS_COLORS[cls],
                       marker=marker, s=100, edgecolors='black', linewidth=1, zorder=5)
    
    # Mark query point B2 (blue side)
    ax4.scatter(*query2_blue, c='red', s=400, marker='*', edgecolors='black', linewidth=2, zorder=10)
    pred_class = classify_point(X, y, np.array(query2_blue))
    ax4.annotate(f'B2 → {CLASS_NAMES[pred_class]}', query2_blue,
                xytext=(query2_blue[0] + 0.3, query2_blue[1] - 0.3),
                fontsize=12, fontweight='bold', color=CLASS_COLORS[pred_class],
                arrowprops=dict(arrowstyle='->', color=CLASS_COLORS[pred_class], lw=2))
    
    # Algorithm box
    algo_text = """O(1) Classification:
1. Bucket lookup → O(1)
2. num_classes = 2
3. Check linked list:
   → Green region: NO
   → Blue region polygon
   → Point-in-polygon: YES
4. Return Blue (1)

Total: O(1)"""
    ax4.text(0.02, 0.98, algo_text, transform=ax4.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='blue', linewidth=2))
    
    ax4.set_xlim(zoom_min_x, zoom_max_x)
    ax4.set_ylim(zoom_min_y, zoom_max_y)
    ax4.set_xlabel('Feature 1')
    ax4.set_ylabel('Feature 2')
    
    # Add legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CLASS_COLORS[0], 
               markersize=12, label='Class 0 (Green)', markeredgecolor='black'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=CLASS_COLORS[1],
               markersize=12, label='Class 1 (Blue)', markeredgecolor='black'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=CLASS_COLORS[2],
               markersize=12, label='Class 2 (Yellow)', markeredgecolor='black'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
               markersize=15, label='Query Point', markeredgecolor='black'),
        Line2D([0], [0], color='black', linewidth=3, label='Decision Boundary'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=5, 
              bbox_to_anchor=(0.5, 0.02), fontsize=11)
    
    # Save
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()
    
    return output_path

if __name__ == '__main__':
    output = generate_dynamic_insertion_figure()
    print(f"Dynamic insertion demo generated: {output}")
