#!/usr/bin/env python3
"""
Generate simple 2D Buckets demo figure - single image showing dynamic queries
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from scipy.spatial import Voronoi
import os

# Color scheme
CLASS_COLORS = {
    0: '#2E8B57',  # Green
    1: '#4A90D9',  # Blue  
    2: '#FFD700'   # Yellow
}

def load_wine_data():
    data = np.loadtxt('data/train/wine_train.csv', delimiter=',')
    return data[:, :2], data[:, 2].astype(int)

def classify_point(X, y, point):
    distances = np.sqrt(np.sum((X - point)**2, axis=1))
    return y[np.argmin(distances)]

def generate_simple_demo():
    X, y = load_wine_data()
    n = len(X)
    k = int(np.ceil(np.sqrt(n)))  # 12x12 grid
    
    # Bounding box
    min_x, max_x = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    min_y, max_y = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    step_x = (max_x - min_x) / k
    step_y = (max_y - min_y) / k
    
    print(f"Grid: {k}x{k}, bounds: x=[{min_x:.2f},{max_x:.2f}], y=[{min_y:.2f},{max_y:.2f}]")
    print(f"Step: ({step_x:.3f}, {step_y:.3f})")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title('(d) Decision Boundaries + SRR Grid (11×11) with Dynamic Queries', fontweight='bold', fontsize=13)
    
    # Draw SRR grid
    for i in range(k + 1):
        ax.axvline(min_x + i * step_x, color='gray', alpha=0.3, linewidth=0.5, linestyle='--')
        ax.axhline(min_y + i * step_y, color='gray', alpha=0.3, linewidth=0.5, linestyle='--')
    
    # Plot Delaunay edges (simplified - just draw triangulation edges)
    from scipy.spatial import Delaunay
    tri = Delaunay(X)
    for simplex in tri.simplices:
        for i in range(3):
            p1, p2 = X[simplex[i]], X[simplex[(i+1)%3]]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=0.5, alpha=0.6)
    
    # Plot training points
    for cls in [0, 1, 2]:
        mask = y == cls
        marker = ['o', 's', '^'][cls]
        ax.scatter(X[mask, 0], X[mask, 1], c=CLASS_COLORS[cls],
                  marker=marker, s=50, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    # Draw decision boundaries using Voronoi
    try:
        vor = Voronoi(X)
        for ridge_idx, (p1, p2) in enumerate(vor.ridge_vertices):
            if -1 in [p1, p2]:
                continue
            point_idxs = vor.ridge_points[ridge_idx]
            if y[point_idxs[0]] != y[point_idxs[1]]:
                v1, v2 = vor.vertices[p1], vor.vertices[p2]
                ax.plot([v1[0], v2[0]], [v1[1], v2[1]], 'k-', linewidth=2.5, alpha=0.9)
    except:
        pass
    
    # ====== BUCKET [11,0] - Top-left corner, outside convex hull ======
    bucket_a_row, bucket_a_col = 11, 0
    bucket_a_x = min_x + bucket_a_col * step_x
    bucket_a_y = min_y + bucket_a_row * step_y
    
    # Highlight single-class bucket
    rect_a = Rectangle((bucket_a_x, bucket_a_y), step_x, step_y,
                        fill=False, edgecolor='red', linewidth=3, linestyle='-')
    ax.add_patch(rect_a)
    ax.text(bucket_a_x + step_x/2, bucket_a_y + step_y + 0.1, f'Bucket[{bucket_a_row},{bucket_a_col}]', 
            ha='center', fontsize=9, color='red', fontweight='bold')
    
    # Query point in single-class bucket
    qa = (bucket_a_x + step_x * 0.5, bucket_a_y + step_y * 0.5)
    qa_class = classify_point(X, y, np.array(qa))
    ax.scatter(*qa, c=CLASS_COLORS[qa_class], s=500, marker='*', 
              edgecolors='black', linewidth=2, zorder=10)
    ax.annotate(f'→ Class {qa_class}', qa, xytext=(qa[0]+0.5, qa[1]+0.1),
               fontsize=11, fontweight='bold', color=CLASS_COLORS[qa_class])
    
    # ====== BUCKET [6,5] - Multi-class (Green-Blue boundary) ======
    bucket_b_row, bucket_b_col = 6, 5
    bucket_b_x = min_x + bucket_b_col * step_x
    bucket_b_y = min_y + bucket_b_row * step_y
    
    # Highlight multi-class bucket
    rect_b = Rectangle((bucket_b_x, bucket_b_y), step_x, step_y,
                        fill=False, edgecolor='purple', linewidth=3, linestyle='-')
    ax.add_patch(rect_b)
    ax.text(bucket_b_x + step_x/2, bucket_b_y - 0.15, f'Bucket[{bucket_b_row},{bucket_b_col}]', 
            ha='center', fontsize=9, color='purple', fontweight='bold')
    
    # Two query points in multi-class bucket
    # Point more towards green region (upper-right of bucket)
    qb1 = (bucket_b_x + step_x * 0.75, bucket_b_y + step_y * 0.75)
    qb1_class = classify_point(X, y, np.array(qb1))
    ax.scatter(*qb1, c=CLASS_COLORS[qb1_class], s=500, marker='*',
              edgecolors='black', linewidth=2, zorder=10)
    ax.annotate(f'→ Class {qb1_class}', qb1, xytext=(qb1[0]+0.4, qb1[1]+0.1),
               fontsize=11, fontweight='bold', color=CLASS_COLORS[qb1_class])
    
    # Point more towards blue region (lower-left of bucket)
    qb2 = (bucket_b_x + step_x * 0.25, bucket_b_y + step_y * 0.25)
    qb2_class = classify_point(X, y, np.array(qb2))
    ax.scatter(*qb2, c=CLASS_COLORS[qb2_class], s=500, marker='*',
              edgecolors='black', linewidth=2, zorder=10)
    ax.annotate(f'→ Class {qb2_class}', qb2, xytext=(qb2[0]-0.9, qb2[1]-0.1),
               fontsize=11, fontweight='bold', color=CLASS_COLORS[qb2_class])
    
    # Print bucket info
    print(f"\nBucket[{bucket_a_row},{bucket_a_col}] (Single-class)")
    print(f"  Query: ({qa[0]:.2f}, {qa[1]:.2f}) → Class {qa_class}")
    print(f"\nBucket[{bucket_b_row},{bucket_b_col}] (Multi-class: Green-Blue)")
    print(f"  Query 1: ({qb1[0]:.2f}, {qb1[1]:.2f}) → Class {qb1_class}")
    print(f"  Query 2: ({qb2[0]:.2f}, {qb2[1]:.2f}) → Class {qb2_class}")
    
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    
    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CLASS_COLORS[0], markersize=10, label='Class 0', markeredgecolor='black'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=CLASS_COLORS[1], markersize=10, label='Class 1', markeredgecolor='black'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=CLASS_COLORS[2], markersize=10, label='Class 2', markeredgecolor='black'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', markersize=15, label='Dynamic Query', markeredgecolor='black'),
        Line2D([0], [0], color='black', linewidth=2.5, label='Decision Boundary'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # Save
    output = 'figures_publication/wine/4_decision_boundaries_with_queries.png'
    os.makedirs(os.path.dirname(output), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output}")
    plt.close()
    return output

if __name__ == '__main__':
    generate_simple_demo()
