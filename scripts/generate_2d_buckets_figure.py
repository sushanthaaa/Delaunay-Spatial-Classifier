#!/usr/bin/env python3
"""
Generate 2D Buckets Visualization Figure

This script generates a figure demonstrating the 2D Buckets data structure
for O(1) dynamic classification with decision boundary polygons.
"""

import sys
import os

# Add scripts directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
from scipy.spatial import Delaunay, Voronoi
from scipy.spatial import ConvexHull

# Use the same style as publication figures
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.dpi'] = 150

# Color scheme matching publication figures
CLASS_COLORS = {
    0: '#2E8B57',  # Green
    1: '#4A90D9',  # Blue  
    2: '#FFD700'   # Yellow/Gold
}

def load_wine_data():
    """Load wine training data."""
    train_path = 'data/train/wine_train.csv'
    data = np.loadtxt(train_path, delimiter=',')
    X = data[:, :2]
    y = data[:, 2].astype(int)
    return X, y

def compute_voronoi_class_regions(X, y):
    """Compute Voronoi diagram and assign classes."""
    vor = Voronoi(X)
    return vor

def classify_point(X, y, point):
    """Classify a point using nearest vertex."""
    distances = np.sqrt(np.sum((X - point)**2, axis=1))
    nearest_idx = np.argmin(distances)
    return y[nearest_idx]

def generate_2d_buckets_figure(output_path='figures_publication/wine/5_2d_buckets.png'):
    """Generate comprehensive 2D Buckets visualization."""
    
    # Load data
    X, y = load_wine_data()
    n_points = len(X)
    
    # Compute SRR grid size
    k = int(np.ceil(np.sqrt(n_points)))
    print(f"SRR Grid Size: {k}x{k} = {k*k} buckets")
    
    # Compute bounding box
    min_x, max_x = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    min_y, max_y = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    step_x = (max_x - min_x) / k
    step_y = (max_y - min_y) / k
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # ============================================
    # Panel (a): Training Data with SRR Grid
    # ============================================
    ax = axes[0, 0]
    ax.set_title('(a) Training Data + SRR Grid', fontweight='bold')
    
    # Plot grid lines
    for i in range(k + 1):
        ax.axvline(min_x + i * step_x, color='gray', alpha=0.3, linewidth=0.5)
        ax.axhline(min_y + i * step_y, color='gray', alpha=0.3, linewidth=0.5)
    
    # Plot points by class
    for cls in [0, 1, 2]:
        mask = y == cls
        if mask.sum() > 0:
            marker = ['o', 's', '^'][cls]
            ax.scatter(X[mask, 0], X[mask, 1], c=CLASS_COLORS[cls], 
                      marker=marker, s=60, alpha=0.8, edgecolors='black', linewidth=0.5,
                      label=f'Class {cls}')
    
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.legend(loc='upper right')
    ax.set_xlabel('Feature 1 (Alcohol)')
    ax.set_ylabel('Feature 2 (Malic Acid)')
    ax.text(0.02, 0.98, f'{k}×{k} Grid', transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ============================================
    # Panel (b): Bucket Classification
    # ============================================
    ax = axes[0, 1]
    ax.set_title('(b) 2D Buckets: Class Assignments', fontweight='bold')
    
    single_class_count = 0
    multi_class_count = 0
    
    # For each bucket, determine dominant class
    bucket_colors = []
    for r in range(k):
        for c in range(k):
            bucket_min_x = min_x + c * step_x
            bucket_max_x = bucket_min_x + step_x
            bucket_min_y = min_y + r * step_y
            bucket_max_y = bucket_min_y + step_y
            
            # Sample 5 points
            sample_points = [
                (bucket_min_x + step_x/2, bucket_min_y + step_y/2),  # center
                (bucket_min_x + step_x*0.1, bucket_min_y + step_y*0.1),
                (bucket_max_x - step_x*0.1, bucket_min_y + step_y*0.1),
                (bucket_min_x + step_x*0.1, bucket_max_y - step_y*0.1),
                (bucket_max_x - step_x*0.1, bucket_max_y - step_y*0.1),
            ]
            
            classes = [classify_point(X, y, np.array(pt)) for pt in sample_points]
            unique_classes = set(classes)
            
            if len(unique_classes) == 1:
                single_class_count += 1
                alpha = 0.5
            else:
                multi_class_count += 1
                alpha = 0.3
            
            # Use dominant class color
            from collections import Counter
            dominant_class = Counter(classes).most_common(1)[0][0]
            
            rect = Rectangle((bucket_min_x, bucket_min_y), step_x, step_y,
                            facecolor=CLASS_COLORS[dominant_class], 
                            edgecolor='black', alpha=alpha, linewidth=0.5)
            ax.add_patch(rect)
            
            # Mark multi-class buckets with striped pattern
            if len(unique_classes) > 1:
                ax.plot([bucket_min_x, bucket_max_x], [bucket_min_y, bucket_max_y],
                       'k-', alpha=0.3, linewidth=0.5)
    
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.text(0.02, 0.98, f'Single-class: {single_class_count}\nMulti-class: {multi_class_count}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ============================================
    # Panel (c): Zoom into Multi-class Bucket
    # ============================================
    ax = axes[1, 0]
    ax.set_title('(c) Zoom: Multi-class Bucket Detail', fontweight='bold')
    
    # Find a multi-class bucket to zoom into
    zoom_bucket = None
    for r in range(k):
        for c in range(k):
            bucket_min_x = min_x + c * step_x
            bucket_max_x = bucket_min_x + step_x
            bucket_min_y = min_y + r * step_y
            bucket_max_y = bucket_min_y + step_y
            
            sample_points = [
                (bucket_min_x + step_x/2, bucket_min_y + step_y/2),
                (bucket_min_x + step_x*0.1, bucket_min_y + step_y*0.1),
                (bucket_max_x - step_x*0.1, bucket_min_y + step_y*0.1),
                (bucket_min_x + step_x*0.1, bucket_max_y - step_y*0.1),
                (bucket_max_x - step_x*0.1, bucket_max_y - step_y*0.1),
            ]
            classes = [classify_point(X, y, np.array(pt)) for pt in sample_points]
            
            if len(set(classes)) >= 2:
                zoom_bucket = (r, c, bucket_min_x, bucket_max_x, bucket_min_y, bucket_max_y)
                break
        if zoom_bucket:
            break
    
    if zoom_bucket:
        r, c, bmin_x, bmax_x, bmin_y, bmax_y = zoom_bucket
        
        # Create fine grid within bucket
        fine_n = 20
        fine_x = np.linspace(bmin_x, bmax_x, fine_n)
        fine_y = np.linspace(bmin_y, bmax_y, fine_n)
        
        for i in range(fine_n - 1):
            for j in range(fine_n - 1):
                cx = (fine_x[i] + fine_x[i+1]) / 2
                cy = (fine_y[j] + fine_y[j+1]) / 2
                cls = classify_point(X, y, np.array([cx, cy]))
                
                rect = Rectangle((fine_x[i], fine_y[j]), 
                                fine_x[i+1] - fine_x[i],
                                fine_y[j+1] - fine_y[j],
                                facecolor=CLASS_COLORS[cls], 
                                edgecolor='none', alpha=0.6)
                ax.add_patch(rect)
        
        # Draw bucket boundary
        ax.plot([bmin_x, bmax_x, bmax_x, bmin_x, bmin_x],
               [bmin_y, bmin_y, bmax_y, bmax_y, bmin_y],
               'k-', linewidth=2)
        
        # Plot nearby training points
        margin = step_x * 0.5
        mask = ((X[:, 0] >= bmin_x - margin) & (X[:, 0] <= bmax_x + margin) &
                (X[:, 1] >= bmin_y - margin) & (X[:, 1] <= bmax_y + margin))
        
        for cls in [0, 1, 2]:
            cls_mask = mask & (y == cls)
            if cls_mask.sum() > 0:
                marker = ['o', 's', '^'][cls]
                ax.scatter(X[cls_mask, 0], X[cls_mask, 1], c=CLASS_COLORS[cls],
                          marker=marker, s=100, edgecolors='black', linewidth=1.5, zorder=10)
        
        ax.set_xlim(bmin_x - margin*0.5, bmax_x + margin*0.5)
        ax.set_ylim(bmin_y - margin*0.5, bmax_y + margin*0.5)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.text(0.02, 0.98, f'Bucket [{r},{c}]\nLinked List:\n→ Class regions', 
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ============================================
    # Panel (d): Classification Flowchart
    # ============================================
    ax = axes[1, 1]
    ax.set_title('(d) O(1) Dynamic Classification Flow', fontweight='bold')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Draw flowchart boxes
    box_props = dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', linewidth=2)
    arrow_props = dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=2)
    
    # Step 1: Query Point
    ax.annotate('Query Point\n(x, y)', xy=(2, 9), fontsize=11, ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='lightblue', edgecolor='black'))
    
    # Step 2: Bucket Lookup
    ax.annotate('', xy=(2, 7.5), xytext=(2, 8.5), arrowprops=arrow_props)
    ax.annotate('O(1) Bucket Lookup\ncol = (x - min_x) / step_x\nrow = (y - min_y) / step_y', 
               xy=(2, 7), fontsize=10, ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black'))
    
    # Step 3: Decision Diamond
    ax.annotate('', xy=(2, 5), xytext=(2, 6), arrowprops=arrow_props)
    ax.annotate('Single\nClass?', xy=(2, 4.5), fontsize=11, ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='black', pad=0.5))
    
    # Branch: Yes (O(1))
    ax.annotate('', xy=(0.5, 3), xytext=(1.3, 4), arrowprops=arrow_props)
    ax.text(0.3, 3.5, 'Yes', fontsize=9)
    ax.annotate('Return\ndominant_class\nO(1)', xy=(0.5, 2), fontsize=10, ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='green', linewidth=2))
    
    # Branch: No (check regions)
    ax.annotate('', xy=(3.5, 3), xytext=(2.7, 4), arrowprops=arrow_props)
    ax.text(3.3, 3.5, 'No', fontsize=9)
    ax.annotate('Point-in-Polygon\nfor each region\nO(1)', xy=(3.8, 2), fontsize=10, ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange', linewidth=2))
    
    # Final output
    ax.annotate('', xy=(2, 0.8), xytext=(0.5, 1.5), arrowprops=arrow_props)
    ax.annotate('', xy=(2, 0.8), xytext=(3.8, 1.5), arrowprops=arrow_props)
    ax.annotate('Predicted Class', xy=(2, 0.3), fontsize=12, ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='lightblue', edgecolor='blue', linewidth=2),
               fontweight='bold')
    
    # Legend/stats on right side
    stats_text = f"""
    Wine Dataset Stats:
    ─────────────────
    Training points: {n_points}
    Grid size: {k}×{k} = {k*k}
    
    Bucket Types:
    ─────────────────
    Single-class: {single_class_count}
    Multi-class: {multi_class_count}
    
    Complexity:
    ─────────────────
    Bucket lookup: O(1)
    Single-class: O(1)
    Multi-class: O(1)*
    
    * O(k) where k is const
    """
    ax.text(6.5, 5, stats_text, fontsize=9, va='center', ha='left',
           fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='gray'))
    
    # Save figure
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved figure to: {output_path}")
    plt.close()
    
    return output_path

if __name__ == '__main__':
    output = generate_2d_buckets_figure()
    print(f"2D Buckets visualization generated: {output}")
