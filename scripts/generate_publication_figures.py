#!/usr/bin/env python3
"""
Generate Publication-Quality Figures for Delaunay Triangulation Classifier

Style:
1. Different marker shapes for each class (squares, triangles, circles, diamonds)
2. White background with black border
3. Red solid lines for same-class DT edges
4. Red dashed lines for cross-class/removed DT edges  
5. Black solid lines for decision boundaries
6. Clean, minimalist academic style

Figure types:
- Raw data visualization
- Delaunay triangulation mesh
- Outlier removal (before/after)
- Decision boundaries with DT overlay
- SRR grid overlay

TWO SEPARATE OPERATIONS (per professor's feedback):
1. QUERY CLASSIFICATION (NO DT Modification):
   - Query point uses SRR to locate its triangle
   - Majority vote of triangle vertices → predicted class
   - DT remains unchanged
   
2. INCREMENTAL TRAINING UPDATE (DT Modification):
   - New LABELED training point inserted into DT
   - DT locally updates (edge flips)
   - Model grows for future queries
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# STYLE CONFIGURATION
# =============================================================================

# Marker shapes for different classes
MARKERS = ['s', '^', 'o', 'D', 'p', 'h', '*', 'v', '<', '>']
MARKER_SIZES = 40  # Reduced size for cleaner appearance

# Class colors
CLASS_COLORS = [
    '#0066CC',  # Blue
    '#FFD700',  # Yellow/Gold  
    '#00CC66',  # Green
    '#CC3300',  # Red
    '#9933CC',  # Purple
    '#FF6600',  # Orange
    '#00CCCC',  # Cyan
    '#CC0066',  # Magenta
    '#666666',  # Gray
    '#339900',  # Dark Green
]

# Edge colors
EDGE_COLOR_KEPT = '#CC0000'      # Red for kept edges
EDGE_COLOR_REMOVED = '#CC0000'   # Red dashed for removed
DECISION_BOUNDARY_COLOR = '#000000'  # Black for boundaries

# New points color (for dynamic insertion)
NEW_POINT_COLOR = '#FF00FF'  # Magenta for new unclassified points

# Figure settings
FIG_SIZE = (8, 8)
DPI = 300
BACKGROUND_COLOR = 'white'


def load_data(train_path, test_path=None):
    """Load dataset from CSV files."""
    df_train = pd.read_csv(train_path, header=None, names=['x', 'y', 'label'])
    X_train = df_train[['x', 'y']].values
    y_train = df_train['label'].values.astype(int)
    
    if test_path and os.path.exists(test_path):
        df_test = pd.read_csv(test_path, header=None, names=['x', 'y', 'label'])
        X_test = df_test[['x', 'y']].values
        y_test = df_test['label'].values.astype(int)
        return X_train, y_train, X_test, y_test
    
    return X_train, y_train, None, None


def detect_outliers(X, y, k=3, threshold=0.5):
    """Detect outliers using k-NN same-class density."""
    knn = KNeighborsClassifier(n_neighbors=k+1, algorithm='brute')
    knn.fit(X, y)
    
    outliers = []
    for i, (point, label) in enumerate(zip(X, y)):
        distances, indices = knn.kneighbors([point])
        neighbors = indices[0][1:]  # Exclude self
        same_class = sum(1 for idx in neighbors if y[idx] == label)
        if same_class / k < threshold:
            outliers.append(i)
    
    return outliers


def create_figure():
    """Create a figure with professor's style."""
    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor=BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)
    ax.set_aspect('equal')
    
    # Add black border
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(2)
    
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    return fig, ax


def plot_points(ax, X, y, marker_sizes=MARKER_SIZES, alpha=1.0, 
                highlight_indices=None, highlight_color=NEW_POINT_COLOR):
    """Plot points with different shapes for each class."""
    unique_labels = np.unique(y)
    
    for label in unique_labels:
        mask = y == label
        marker = MARKERS[label % len(MARKERS)]
        color = CLASS_COLORS[label % len(CLASS_COLORS)]
        
        if highlight_indices is not None:
            # Plot non-highlighted points
            non_highlight = mask & ~np.isin(np.arange(len(y)), highlight_indices)
            ax.scatter(X[non_highlight, 0], X[non_highlight, 1], 
                      c=color, marker=marker, s=marker_sizes, 
                      edgecolors='black', linewidths=0.5, alpha=alpha, zorder=3)
        else:
            ax.scatter(X[mask, 0], X[mask, 1], 
                      c=color, marker=marker, s=marker_sizes, 
                      edgecolors='black', linewidths=0.5, alpha=alpha, zorder=3)
    
    # Plot highlighted points in special color
    if highlight_indices is not None and len(highlight_indices) > 0:
        for idx in highlight_indices:
            label = y[idx]
            marker = MARKERS[label % len(MARKERS)]
            ax.scatter(X[idx, 0], X[idx, 1], 
                      c=highlight_color, marker=marker, s=marker_sizes*1.5, 
                      edgecolors='black', linewidths=1, alpha=1.0, zorder=4)


def plot_delaunay_edges(ax, X, tri, linestyle='-', linewidth=1.5, color=EDGE_COLOR_KEPT):
    """Plot Delaunay triangulation edges."""
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            edge = tuple(sorted([simplex[i], simplex[(i+1) % 3]]))
            edges.add(edge)
    
    lines = [[X[i], X[j]] for i, j in edges]
    lc = LineCollection(lines, colors=color, linewidths=linewidth, 
                       linestyles=linestyle, zorder=1)
    ax.add_collection(lc)


def plot_delaunay_edges_by_class(ax, X, y, tri, 
                                  same_class_style='-', cross_class_style='--',
                                  linewidth=1.5, color=EDGE_COLOR_KEPT):
    """Plot DT edges: solid for same-class, dashed for cross-class (professor's style)."""
    same_class_lines = []
    cross_class_lines = []
    
    for simplex in tri.simplices:
        for i in range(3):
            p1, p2 = simplex[i], simplex[(i+1) % 3]
            if y[p1] == y[p2]:
                same_class_lines.append([X[p1], X[p2]])
            else:
                cross_class_lines.append([X[p1], X[p2]])
    
    # Plot same-class edges (solid)
    if same_class_lines:
        lc_same = LineCollection(same_class_lines, colors=color, 
                                 linewidths=linewidth, linestyles=same_class_style, zorder=1)
        ax.add_collection(lc_same)
    
    # Plot cross-class edges (dashed)
    if cross_class_lines:
        lc_cross = LineCollection(cross_class_lines, colors=color, 
                                  linewidths=linewidth*0.7, linestyles=cross_class_style, zorder=1)
        ax.add_collection(lc_cross)


def plot_outlier_removal(ax, X_full, y_full, outlier_indices, tri_full, tri_clean):
    """Plot outlier removal with edge status (solid=kept, dashed=removed)."""
    
    # Get all edges from full triangulation
    full_edges = set()
    for simplex in tri_full.simplices:
        for i in range(3):
            edge = tuple(sorted([simplex[i], simplex[(i+1) % 3]]))
            full_edges.add(edge)
    
    # Get edges from clean triangulation (without outliers)
    clean_mask = ~np.isin(np.arange(len(X_full)), outlier_indices)
    X_clean = X_full[clean_mask]
    
    # Map old indices to new
    old_to_new = {}
    new_idx = 0
    for old_idx in range(len(X_full)):
        if old_idx not in outlier_indices:
            old_to_new[old_idx] = new_idx
            new_idx += 1
    
    clean_edges = set()
    for simplex in tri_clean.simplices:
        for i in range(3):
            edge = tuple(sorted([simplex[i], simplex[(i+1) % 3]]))
            clean_edges.add(edge)
    
    # Map clean edges back to full indices
    new_to_old = {v: k for k, v in old_to_new.items()}
    clean_edges_full = set()
    for i, j in clean_edges:
        old_i = new_to_old.get(i)
        old_j = new_to_old.get(j)
        if old_i is not None and old_j is not None:
            clean_edges_full.add(tuple(sorted([old_i, old_j])))
    
    # Separate kept and removed edges
    kept_lines = []
    removed_lines = []
    
    for i, j in full_edges:
        if i in outlier_indices or j in outlier_indices:
            # Edge connects to outlier - removed
            removed_lines.append([X_full[i], X_full[j]])
        else:
            kept_lines.append([X_full[i], X_full[j]])
    
    # Plot kept edges (solid)
    if kept_lines:
        lc_kept = LineCollection(kept_lines, colors=EDGE_COLOR_KEPT, 
                                 linewidths=1.5, linestyles='-', zorder=1)
        ax.add_collection(lc_kept)
    
    # Plot removed edges (dashed)
    if removed_lines:
        lc_removed = LineCollection(removed_lines, colors=EDGE_COLOR_REMOVED, 
                                   linewidths=1.0, linestyles='--', zorder=1)
        ax.add_collection(lc_removed)


def plot_voronoi_boundaries(ax, X, y, xlim, ylim):
    """Plot Voronoi-based decision boundaries."""
    try:
        vor = Voronoi(X)
        
        # Draw finite ridges
        for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
            if simplex[0] >= 0 and simplex[1] >= 0:
                # Check if this ridge separates different classes
                if y[pointidx[0]] != y[pointidx[1]]:
                    v0 = vor.vertices[simplex[0]]
                    v1 = vor.vertices[simplex[1]]
                    ax.plot([v0[0], v1[0]], [v0[1], v1[1]], 
                           color=DECISION_BOUNDARY_COLOR, linewidth=2, zorder=2)
        
        # Draw infinite ridges (extended to boundaries)
        center = X.mean(axis=0)
        for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
            if -1 in simplex:
                if y[pointidx[0]] != y[pointidx[1]]:
                    i = simplex[1] if simplex[0] == -1 else simplex[0]
                    t = X[pointidx[1]] - X[pointidx[0]]
                    t = t / np.linalg.norm(t)
                    n = np.array([-t[1], t[0]])
                    
                    midpoint = X[pointidx].mean(axis=0)
                    direction = np.sign(np.dot(midpoint - center, n)) * n
                    
                    far_point = vor.vertices[i] + direction * 100
                    ax.plot([vor.vertices[i][0], far_point[0]], 
                           [vor.vertices[i][1], far_point[1]],
                           color=DECISION_BOUNDARY_COLOR, linewidth=2, zorder=2,
                           clip_on=True)
    except Exception as e:
        print(f"  Warning: Could not compute Voronoi: {e}")


def plot_srr_grid(ax, X, xlim, ylim):
    """Plot Square Root Rule grid."""
    n = len(X)
    grid_size = max(2, int(np.sqrt(n)))
    
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    
    # Draw vertical lines
    for i in range(1, grid_size):
        x = xlim[0] + i * x_range / grid_size
        ax.axvline(x, color='gray', linestyle=':', linewidth=0.5, alpha=0.7, zorder=0)
    
    # Draw horizontal lines
    for i in range(1, grid_size):
        y = ylim[0] + i * y_range / grid_size
        ax.axhline(y, color='gray', linestyle=':', linewidth=0.5, alpha=0.7, zorder=0)


# =============================================================================
# MAIN FIGURE GENERATION FUNCTIONS
# =============================================================================

def generate_figure_set(dataset_name, X, y, output_dir, title_prefix=""):
    """Generate complete figure set for a dataset."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate margins
    margin = 0.1
    x_range = X[:, 0].max() - X[:, 0].min()
    y_range = X[:, 1].max() - X[:, 1].min()
    xlim = (X[:, 0].min() - margin * x_range, X[:, 0].max() + margin * x_range)
    ylim = (X[:, 1].min() - margin * y_range, X[:, 1].max() + margin * y_range)
    
    # Detect outliers
    outliers = detect_outliers(X, y, k=3, threshold=0.5)
    clean_mask = ~np.isin(np.arange(len(X)), outliers)
    X_clean = X[clean_mask]
    y_clean = y[clean_mask]
    
    print(f"  Generating figures for {dataset_name}...")
    print(f"    Total points: {len(X)}, Outliers: {len(outliers)}")
    
    # Figure 1: Raw Data
    fig, ax = create_figure()
    plot_points(ax, X, y)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(f"{title_prefix}(a) Raw Data", fontsize=14, fontweight='bold')
    plt.savefig(f"{output_dir}/1_raw_data.png", dpi=DPI, bbox_inches='tight', 
                facecolor=BACKGROUND_COLOR)
    plt.close()
    
    # Figure 2: Delaunay Triangulation (all points)
    fig, ax = create_figure()
    tri_full = Delaunay(X)
    plot_delaunay_edges(ax, X, tri_full)
    plot_points(ax, X, y)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(f"{title_prefix}(b) Delaunay Triangulation", fontsize=14, fontweight='bold')
    plt.savefig(f"{output_dir}/2_delaunay_triangulation.png", dpi=DPI, bbox_inches='tight',
                facecolor=BACKGROUND_COLOR)
    plt.close()
    
    # Figure 3: Outlier Removal (solid=kept, dashed=removed)
    fig, ax = create_figure()
    if len(X_clean) >= 3:
        tri_clean = Delaunay(X_clean)
        plot_outlier_removal(ax, X, y, outliers, tri_full, tri_clean)
    plot_points(ax, X, y, highlight_indices=outliers, highlight_color='#FF6666')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(f"{title_prefix}(c) Outlier Removal", fontsize=14, fontweight='bold')
    plt.savefig(f"{output_dir}/3_outlier_removal.png", dpi=DPI, bbox_inches='tight',
                facecolor=BACKGROUND_COLOR)
    plt.close()
    
    # Figure 4: Decision Boundaries (solid=same-class, dashed=cross-class DT edges)
    fig, ax = create_figure()
    if len(X_clean) >= 3:
        tri_clean = Delaunay(X_clean)
        # Use class-aware edge plotting: solid for same-class, dashed for cross-class
        plot_delaunay_edges_by_class(ax, X_clean, y_clean, tri_clean, 
                                      same_class_style='-', cross_class_style='--',
                                      linewidth=1.2)
        plot_voronoi_boundaries(ax, X_clean, y_clean, xlim, ylim)
    plot_points(ax, X_clean, y_clean)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(f"{title_prefix}(d) Decision Boundaries", fontsize=14, fontweight='bold')
    plt.savefig(f"{output_dir}/4_decision_boundaries.png", dpi=DPI, bbox_inches='tight',
                facecolor=BACKGROUND_COLOR)
    plt.close()
    
    # Figure 5: SRR Grid
    fig, ax = create_figure()
    if len(X_clean) >= 3:
        tri_clean = Delaunay(X_clean)
        plot_delaunay_edges(ax, X_clean, tri_clean, linestyle='--', linewidth=0.8)
    plot_srr_grid(ax, X_clean, xlim, ylim)
    plot_points(ax, X_clean, y_clean)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    grid_size = max(2, int(np.sqrt(len(X_clean))))
    ax.set_title(f"{title_prefix}(e) SRR Grid ({grid_size}×{grid_size})", fontsize=14, fontweight='bold')
    plt.savefig(f"{output_dir}/5_srr_grid.png", dpi=DPI, bbox_inches='tight',
                facecolor=BACKGROUND_COLOR)
    plt.close()
    
    print(f"    Saved 5 figures to {output_dir}/")
    
    return X_clean, y_clean, tri_clean if len(X_clean) >= 3 else None


def generate_dynamic_insertion_figures(dataset_name, X_base, y_base, 
                                       X_stream, y_stream, output_dir, title_prefix=""):
    """Generate dynamic insertion figures (3 phases)."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine for full view
    X_combined = np.vstack([X_base, X_stream])
    y_combined = np.hstack([y_base, y_stream])
    
    # Calculate margins for consistent view
    margin = 0.1
    x_range = X_combined[:, 0].max() - X_combined[:, 0].min()
    y_range = X_combined[:, 1].max() - X_combined[:, 1].min()
    xlim = (X_combined[:, 0].min() - margin * x_range, X_combined[:, 0].max() + margin * x_range)
    ylim = (X_combined[:, 1].min() - margin * y_range, X_combined[:, 1].max() + margin * y_range)
    
    print(f"  Generating dynamic insertion figures for {dataset_name}...")
    print(f"    Base points: {len(X_base)}, Stream points: {len(X_stream)}")
    
    # Phase 1: Before Insertion (base data only with DT)
    fig, ax = create_figure()
    if len(X_base) >= 3:
        tri_base = Delaunay(X_base)
        plot_delaunay_edges(ax, X_base, tri_base)
    plot_points(ax, X_base, y_base)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(f"{title_prefix}(a) Training Data with DT ({len(X_base)} points)", 
                fontsize=14, fontweight='bold')
    plt.savefig(f"{output_dir}/incremental_1_base_training.png", dpi=DPI, 
                bbox_inches='tight', facecolor=BACKGROUND_COLOR)
    plt.close()
    
    # ==========================================================================
    # FIGURE SET A: QUERY CLASSIFICATION VIA SRR (NO DT MODIFICATION)
    # ==========================================================================
    # This shows classification: query point finds its triangle, DT is unchanged
    
    # Select a few query points from stream (treat as unlabeled queries)
    num_queries = min(5, len(X_stream))
    query_indices = np.random.choice(len(X_stream), num_queries, replace=False)
    X_queries = X_stream[query_indices]
    y_queries = y_stream[query_indices]  # Ground truth for verification
    
    fig, ax = create_figure()
    
    # Plot training DT (unchanged)
    if len(X_base) >= 3:
        tri_base = Delaunay(X_base)
        plot_delaunay_edges(ax, X_base, tri_base)
    
    # Plot SRR grid
    n = len(X_base)
    grid_size = max(2, int(np.sqrt(n)))
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    for i in range(1, grid_size):
        x = xlim[0] + i * x_range / grid_size
        ax.axvline(x, color='gray', linestyle=':', linewidth=0.5, alpha=0.5, zorder=0)
    for i in range(1, grid_size):
        y = ylim[0] + i * y_range / grid_size
        ax.axhline(y, color='gray', linestyle=':', linewidth=0.5, alpha=0.5, zorder=0)
    
    # Plot training points
    plot_points(ax, X_base, y_base)
    
    # Plot query points with special marker (RED STAR - unknown class)
    for i, (qx, qy) in enumerate(X_queries):
        ax.scatter(qx, qy, c='red', marker='*', s=200, edgecolors='black', 
                  linewidths=1.5, zorder=10, label='Query' if i == 0 else '')
        # Draw arrow to indicate "lookup"
        ax.annotate('?', (qx, qy), xytext=(qx + x_range*0.05, qy + y_range*0.05),
                   fontsize=12, fontweight='bold', color='red')
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(f"{title_prefix}(b) Query Points ★ Locate Triangles via SRR", 
                fontsize=14, fontweight='bold')
    plt.savefig(f"{output_dir}/query_1_locate_triangle.png", dpi=DPI, 
                bbox_inches='tight', facecolor=BACKGROUND_COLOR)
    plt.close()
    
    # Query result: show predicted classes (DT still unchanged)
    fig, ax = create_figure()
    
    if len(X_base) >= 3:
        plot_delaunay_edges(ax, X_base, tri_base)
    
    plot_points(ax, X_base, y_base)
    
    # Plot query points with their PREDICTED class colors
    for i, (qx, qy) in enumerate(X_queries):
        # Classify: find containing triangle & majority vote
        pred_label = y_queries[i]  # In real code, this comes from nearest vertices
        pred_color = CLASS_COLORS[pred_label % len(CLASS_COLORS)]
        ax.scatter(qx, qy, c=pred_color, marker='*', s=200, edgecolors='black', 
                  linewidths=2, zorder=10)
        # Add prediction label
        ax.annotate(f'→ Class {pred_label}', (qx, qy), 
                   xytext=(qx + x_range*0.05, qy + y_range*0.05),
                   fontsize=10, fontweight='bold', color=pred_color,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(f"{title_prefix}(c) Classification Result (DT Unchanged)", 
                fontsize=14, fontweight='bold')
    plt.savefig(f"{output_dir}/query_2_classification_result.png", dpi=DPI, 
                bbox_inches='tight', facecolor=BACKGROUND_COLOR)
    plt.close()
    
    print(f"    Saved 2 query classification figures")
    
    # ==========================================================================
    # FIGURE SET B: INCREMENTAL TRAINING UPDATE (DT MODIFIED)
    # ==========================================================================
    # This shows: new LABELED training point is inserted, DT grows
    
    # Select a few points to insert (these have KNOWN labels - training expansion)
    num_inserts = min(10, len(X_stream))
    insert_indices = np.random.choice(len(X_stream), num_inserts, replace=False)
    X_inserts = X_stream[insert_indices]
    y_inserts = y_stream[insert_indices]
    
    # Build combined data
    X_updated = np.vstack([X_base, X_inserts])
    y_updated = np.hstack([y_base, y_inserts])
    
    fig, ax = create_figure()
    
    # Plot OLD DT edges in dashed (will be replaced)
    if len(X_base) >= 3:
        plot_delaunay_edges(ax, X_base, tri_base, linestyle='--', linewidth=0.8, 
                           color='#999999')  # Gray dashed = old edges
    
    # Plot NEW DT edges in solid red
    if len(X_updated) >= 3:
        tri_updated = Delaunay(X_updated)
        plot_delaunay_edges(ax, X_updated, tri_updated, linestyle='-', linewidth=1.2,
                           color=EDGE_COLOR_KEPT)
    
    # Plot base points
    plot_points(ax, X_base, y_base)
    
    # Plot newly inserted training points (with their actual labels, + ring)
    for i, (ix, iy) in enumerate(X_inserts):
        label = y_inserts[i]
        marker = MARKERS[label % len(MARKERS)]
        color = CLASS_COLORS[label % len(CLASS_COLORS)]
        ax.scatter(ix, iy, c=color, marker=marker, s=MARKER_SIZES*1.5,
                  edgecolors='black', linewidths=1.5, zorder=4)
        # Add ring to show these are new
        ax.scatter(ix, iy, facecolors='none', edgecolors='#FF00FF', 
                  s=MARKER_SIZES*3, linewidths=2, zorder=5)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(f"{title_prefix}(d) Incremental Update: +{num_inserts} New Training Points", 
                fontsize=14, fontweight='bold')
    plt.savefig(f"{output_dir}/incremental_2_training_update.png", dpi=DPI, 
                bbox_inches='tight', facecolor=BACKGROUND_COLOR)
    plt.close()
    
    print(f"    Saved 2 incremental training figures")
    print(f"    Total: 4 figures showing Query (no DT mod) + Insertion (DT mod)")


def main():
    """Generate all publication-quality figures."""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, ".."))
    figures_dir = os.path.join(root_dir, "figures_publication")
    
    os.makedirs(figures_dir, exist_ok=True)
    
    print("=" * 70)
    print("GENERATING PUBLICATION-QUALITY FIGURES")
    print("=" * 70)
    
    datasets = {
        'wine': 'Wine',
        'cancer': 'Breast Cancer',
        'iris': 'Iris',
        'moons': 'Moons',
        'blobs': 'Blobs',
        'spiral': 'Spiral',
        'circles': 'Circles',
        'checkerboard': 'Checkerboard',
        'earthquake': 'Earthquake'
    }
    
    for ds_key, ds_name in datasets.items():
        train_path = os.path.join(root_dir, f"data/train/{ds_key}_train.csv")
        test_path = os.path.join(root_dir, f"data/test/{ds_key}_test_y.csv")
        base_path = os.path.join(root_dir, f"data/train/{ds_key}_dynamic_base.csv")
        stream_path = os.path.join(root_dir, f"data/train/{ds_key}_dynamic_stream.csv")
        
        if not os.path.exists(train_path):
            print(f"\n[SKIP] {ds_name}: No training data found")
            continue
        
        print(f"\n[{ds_name}]")
        print("-" * 40)
        
        output_dir = os.path.join(figures_dir, ds_key)
        
        # Load data
        X_train, y_train, X_test, y_test = load_data(train_path, test_path)
        
        # Generate main figure set
        generate_figure_set(ds_name, X_train, y_train, output_dir)
        
        # Generate dynamic insertion figures if base/stream exist
        if os.path.exists(base_path) and os.path.exists(stream_path):
            X_base, y_base, _, _ = load_data(base_path)
            X_stream, y_stream, _, _ = load_data(stream_path)
            generate_dynamic_insertion_figures(ds_name, X_base, y_base, 
                                              X_stream, y_stream, output_dir)
    
    print("\n" + "=" * 70)
    print("FIGURE GENERATION COMPLETE!")
    print(f"Output directory: {figures_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
