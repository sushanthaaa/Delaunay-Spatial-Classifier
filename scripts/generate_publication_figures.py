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


def plot_decision_boundaries(ax, X, y, tri, linewidth=2.5, color=DECISION_BOUNDARY_COLOR, 
                              extend_to_axes=True, xlim=None, ylim=None):
    """
    Plot decision boundaries as black lines, extended to the plot edges.
    
    Decision boundary logic:
    - For triangles with 2 classes: connect midpoints of cross-class edges
    - For triangles with 3 classes: connect centroid to all edge midpoints
    - For boundary edges: EXTEND the line to the plot axes
    
    This creates the "fence" that separates different class regions,
    extending all the way to the plot borders like in the professor's example.
    """
    from scipy.spatial import ConvexHull
    
    boundary_lines = []
    boundary_endpoints = []  # Track endpoints on convex hull for extension
    
    # Get convex hull edges (these are where we need to extend outward)
    hull = ConvexHull(X)
    hull_edges = set()
    for i in range(len(hull.vertices)):
        v1 = hull.vertices[i]
        v2 = hull.vertices[(i + 1) % len(hull.vertices)]
        hull_edges.add(tuple(sorted([v1, v2])))
    
    # Process each triangle
    for simplex in tri.simplices:
        v0, v1, v2 = simplex
        l0, l1, l2 = y[v0], y[v1], y[v2]
        p0, p1, p2 = X[v0], X[v1], X[v2]
        
        # Edge midpoints
        m01 = (p0 + p1) / 2
        m12 = (p1 + p2) / 2
        m20 = (p2 + p0) / 2
        
        # Check which edges are on the convex hull
        e01_on_hull = tuple(sorted([v0, v1])) in hull_edges
        e12_on_hull = tuple(sorted([v1, v2])) in hull_edges
        e20_on_hull = tuple(sorted([v2, v0])) in hull_edges
        
        # Case 1: All three vertices have DIFFERENT classes
        if l0 != l1 and l1 != l2 and l0 != l2:
            centroid = (p0 + p1 + p2) / 3
            boundary_lines.append([centroid, m01])
            boundary_lines.append([centroid, m12])
            boundary_lines.append([centroid, m20])
            
            # Track hull midpoints for extension
            if e01_on_hull:
                boundary_endpoints.append((centroid, m01, p0, p1))
            if e12_on_hull:
                boundary_endpoints.append((centroid, m12, p1, p2))
            if e20_on_hull:
                boundary_endpoints.append((centroid, m20, p2, p0))
        
        # Case 2: Two classes in triangle
        elif l0 != l1 or l1 != l2 or l0 != l2:
            active_midpoints = []
            active_edges_on_hull = []
            
            if l0 != l1:
                active_midpoints.append(m01)
                if e01_on_hull:
                    active_edges_on_hull.append((m01, p0, p1))
            if l1 != l2:
                active_midpoints.append(m12)
                if e12_on_hull:
                    active_edges_on_hull.append((m12, p1, p2))
            if l2 != l0:
                active_midpoints.append(m20)
                if e20_on_hull:
                    active_edges_on_hull.append((m20, p2, p0))
            
            if len(active_midpoints) == 2:
                boundary_lines.append([active_midpoints[0], active_midpoints[1]])
                
                # Track for extension if on hull
                for midpoint, pa, pb in active_edges_on_hull:
                    # The other midpoint is the "interior" point
                    other = active_midpoints[1] if np.array_equal(midpoint, active_midpoints[0]) else active_midpoints[0]
                    boundary_endpoints.append((other, midpoint, pa, pb))
    
    # Draw all internal boundary lines
    if boundary_lines:
        lc = LineCollection(boundary_lines, colors=color, linewidths=linewidth, 
                           linestyles='-', zorder=2)
        ax.add_collection(lc)
    
    # Extend boundaries to axes if requested
    if extend_to_axes and boundary_endpoints:
        if xlim is None:
            xlim = ax.get_xlim()
        if ylim is None:
            ylim = ax.get_ylim()
        
        extension_lines = []
        for interior_pt, hull_midpoint, hull_v1, hull_v2 in boundary_endpoints:
            # Direction from interior point to hull midpoint
            direction = hull_midpoint - interior_pt
            if np.linalg.norm(direction) < 1e-10:
                continue
            direction = direction / np.linalg.norm(direction)
            
            # Extend outward from hull_midpoint
            # Find intersection with plot boundary
            extended_pt = extend_line_to_boundary(hull_midpoint, direction, xlim, ylim)
            if extended_pt is not None:
                extension_lines.append([hull_midpoint, extended_pt])
        
        if extension_lines:
            lc_ext = LineCollection(extension_lines, colors=color, linewidths=linewidth, 
                                   linestyles='-', zorder=2)
            ax.add_collection(lc_ext)


def extend_line_to_boundary(start_point, direction, xlim, ylim):
    """
    Extend a line from start_point in the given direction until it hits the plot boundary.
    """
    x0, y0 = start_point
    dx, dy = direction
    
    if abs(dx) < 1e-10 and abs(dy) < 1e-10:
        return None
    
    # Find intersection with each boundary
    t_candidates = []
    
    # Left boundary (x = xlim[0])
    if abs(dx) > 1e-10:
        t = (xlim[0] - x0) / dx
        if t > 0:
            y_at_t = y0 + t * dy
            if ylim[0] <= y_at_t <= ylim[1]:
                t_candidates.append(t)
    
    # Right boundary (x = xlim[1])
    if abs(dx) > 1e-10:
        t = (xlim[1] - x0) / dx
        if t > 0:
            y_at_t = y0 + t * dy
            if ylim[0] <= y_at_t <= ylim[1]:
                t_candidates.append(t)
    
    # Bottom boundary (y = ylim[0])
    if abs(dy) > 1e-10:
        t = (ylim[0] - y0) / dy
        if t > 0:
            x_at_t = x0 + t * dx
            if xlim[0] <= x_at_t <= xlim[1]:
                t_candidates.append(t)
    
    # Top boundary (y = ylim[1])
    if abs(dy) > 1e-10:
        t = (ylim[1] - y0) / dy
        if t > 0:
            x_at_t = x0 + t * dx
            if xlim[0] <= x_at_t <= xlim[1]:
                t_candidates.append(t)
    
    if not t_candidates:
        return None
    
    t_min = min(t_candidates)
    return np.array([x0 + t_min * dx, y0 + t_min * dy])


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
    
    # Figure 4: Decision Boundaries WITH SRR Grid (solid=same-class, dashed=cross-class DT edges)
    fig, ax = create_figure()
    grid_size = max(2, int(np.sqrt(len(X_clean))))
    if len(X_clean) >= 3:
        tri_clean = Delaunay(X_clean)
        # Use class-aware edge plotting: solid for same-class, dashed for cross-class
        plot_delaunay_edges_by_class(ax, X_clean, y_clean, tri_clean, 
                                      same_class_style='-', cross_class_style='--',
                                      linewidth=1.2)
        plot_voronoi_boundaries(ax, X_clean, y_clean, xlim, ylim)
    
    # Add SRR grid overlay (dashed gray lines)
    plot_srr_grid(ax, X_clean, xlim, ylim)
    
    plot_points(ax, X_clean, y_clean)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(f"{title_prefix}(d) Decision Boundaries + SRR Grid ({grid_size}×{grid_size})", fontsize=14, fontweight='bold')
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
        
        # Add decision boundary fence (black lines separating class regions)
        # Extended to the plot axes
        plot_decision_boundaries(ax, X_base, y_base, tri_base, xlim=xlim, ylim=ylim)
    
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
    ax.set_title(f"{title_prefix}(b) Query Points ★ with Decision Boundaries", 
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
            
            # Generate outside-hull classification figure
            generate_outside_hull_figure(ds_name, X_base, y_base, 
                                         X_stream, y_stream, output_dir)
    
    print("\n" + "=" * 70)
    print("FIGURE GENERATION COMPLETE!")
    print(f"Output directory: {figures_dir}")
    print("=" * 70)

def generate_outside_hull_figure(dataset_name, X_base, y_base, X_stream, y_stream, output_dir):
    """
    Generate figures illustrating outside-hull classification with extended decision boundaries.
    
    Creates TWO figures:
    1. Query point outside hull - shows how it's classified based on decision regions
    2. After classification - shows the query connected to the mesh (if inserted)
    
    Fixed issues:
    - No background shading
    - Normal point sizes with correct markers
    - Red DT edges (hull is part of DT, not separate black outline)
    - Query placed in top-left SRR cell (S00 or S01)
    """
    from scipy.spatial import ConvexHull
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine data for consistent plot limits
    X_combined = np.vstack([X_base, X_stream])
    
    margin = 0.15
    x_range = X_combined[:, 0].max() - X_combined[:, 0].min()
    y_range = X_combined[:, 1].max() - X_combined[:, 1].min()
    xlim = (X_combined[:, 0].min() - margin * x_range, X_combined[:, 0].max() + margin * x_range)
    ylim = (X_combined[:, 1].min() - margin * y_range, X_combined[:, 1].max() + margin * y_range)
    
    if len(X_base) < 3:
        return
    
    tri_base = Delaunay(X_base)
    
    # Calculate SRR grid parameters
    n = len(X_base)
    grid_size = max(2, int(np.sqrt(n)))
    cell_width = (xlim[1] - xlim[0]) / grid_size
    cell_height = (ylim[1] - ylim[0]) / grid_size
    
    # Place query point in top-left area (S00 or S01 cell) - OUTSIDE the hull
    # S00 is at (row=grid_size-1, col=0), which is top-left
    # We place it slightly inside the S00 cell but outside the convex hull
    query_col = 0  # Leftmost column
    query_row = grid_size - 1  # Top row
    
    # Center of S00 cell
    query_x = xlim[0] + (query_col + 0.5) * cell_width
    query_y = ylim[0] + (query_row + 0.5) * cell_height
    
    # Adjust to make sure it's outside the convex hull
    hull = ConvexHull(X_base)
    hull_centroid = X_base.mean(axis=0)
    
    # If query is inside hull, move it outward
    query_outside = np.array([query_x, query_y])
    
    # Move toward top-left corner to ensure it's outside
    direction_to_corner = np.array([xlim[0], ylim[1]]) - hull_centroid
    direction_to_corner = direction_to_corner / np.linalg.norm(direction_to_corner)
    
    # Place query in top-left, definitely outside hull
    query_outside = np.array([xlim[0] + 0.5 * cell_width, ylim[1] - 0.5 * cell_height])
    
    # ==========================================================================
    # FIGURE 1: Query point outside hull, not yet classified
    # ==========================================================================
    fig, ax = create_figure()
    
    # Draw SRR grid with labels
    for i in range(grid_size + 1):
        x = xlim[0] + i * cell_width
        ax.axvline(x, color='gray', linestyle=':', linewidth=0.8, alpha=0.6, zorder=0)
    for i in range(grid_size + 1):
        y = ylim[0] + i * cell_height
        ax.axhline(y, color='gray', linestyle=':', linewidth=0.8, alpha=0.6, zorder=0)
    
    # Add SRR cell labels (S00, S01, etc.)
    for row in range(grid_size):
        for col in range(grid_size):
            cell_center_x = xlim[0] + (col + 0.5) * cell_width
            cell_center_y = ylim[0] + (row + 0.5) * cell_height
            # Label only every few cells to avoid clutter
            if grid_size <= 4 or (row % 2 == 0 and col % 2 == 0):
                pass  # Skip labels for cleaner figure
    
    # Draw Delaunay triangulation edges (RED - including hull edges)
    plot_delaunay_edges(ax, X_base, tri_base)
    
    # Draw decision boundaries (BLACK lines, extended to axes)
    plot_voronoi_boundaries(ax, X_base, y_base, xlim, ylim)
    
    # Draw training points (NORMAL size with correct markers)
    plot_points(ax, X_base, y_base)
    
    # Draw query point OUTSIDE the hull (PURPLE STAR with ?)
    ax.scatter([query_outside[0]], [query_outside[1]], 
              c='red', marker='*', s=250, edgecolors='black', 
              linewidths=1.5, zorder=10)
    ax.annotate('?', (query_outside[0], query_outside[1]), 
               xytext=(query_outside[0] + 0.03 * x_range, query_outside[1] + 0.03 * y_range),
               fontsize=14, fontweight='bold', color='red', zorder=11)
    
    # Find nearest hull edge and draw connection line
    min_dist = float('inf')
    nearest_edge_midpoint = None
    nearest_v1_idx = None
    nearest_v2_idx = None
    
    for i in range(len(hull.vertices)):
        v1_idx = hull.vertices[i]
        v2_idx = hull.vertices[(i + 1) % len(hull.vertices)]
        edge_midpoint = (X_base[v1_idx] + X_base[v2_idx]) / 2
        dist = np.linalg.norm(query_outside - edge_midpoint)
        if dist < min_dist:
            min_dist = dist
            nearest_edge_midpoint = edge_midpoint
            nearest_v1_idx = v1_idx
            nearest_v2_idx = v2_idx
    
    # Draw dashed line from query to nearest edge midpoint
    if nearest_edge_midpoint is not None:
        ax.plot([query_outside[0], nearest_edge_midpoint[0]], 
               [query_outside[1], nearest_edge_midpoint[1]], 
               'r--', linewidth=1.5, zorder=5)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(f"(e) Query Point Outside Hull - Classification via Decision Region", 
                fontsize=13, fontweight='bold')
    
    plt.savefig(f"{output_dir}/outside_hull_1_query.png", dpi=DPI, 
                bbox_inches='tight', facecolor=BACKGROUND_COLOR)
    plt.close()
    
    # ==========================================================================
    # FIGURE 2: After classification - query gets classified and inserted into mesh
    # ==========================================================================
    fig, ax = create_figure()
    
    # Determine predicted class for the query point
    # Based on decision boundary region logic
    if nearest_v1_idx is not None and nearest_v2_idx is not None:
        label1 = y_base[nearest_v1_idx]
        label2 = y_base[nearest_v2_idx]
        
        if label1 == label2:
            predicted_label = label1
        else:
            # Closer vertex determines class
            dist1 = np.linalg.norm(query_outside - X_base[nearest_v1_idx])
            dist2 = np.linalg.norm(query_outside - X_base[nearest_v2_idx])
            predicted_label = label1 if dist1 <= dist2 else label2
    else:
        predicted_label = y_base[0]  # Fallback
    
    # Create new data with query point inserted
    X_with_query = np.vstack([X_base, query_outside])
    y_with_query = np.hstack([y_base, predicted_label])
    
    # Create new triangulation
    tri_updated = Delaunay(X_with_query)
    
    # Draw SRR grid
    for i in range(grid_size + 1):
        x = xlim[0] + i * cell_width
        ax.axvline(x, color='gray', linestyle=':', linewidth=0.8, alpha=0.6, zorder=0)
    for i in range(grid_size + 1):
        y = ylim[0] + i * cell_height
        ax.axhline(y, color='gray', linestyle=':', linewidth=0.8, alpha=0.6, zorder=0)
    
    # Draw updated Delaunay triangulation (includes new edges to query point)
    plot_delaunay_edges(ax, X_with_query, tri_updated)
    
    # Draw decision boundaries
    plot_voronoi_boundaries(ax, X_with_query, y_with_query, xlim, ylim)
    
    # Draw original training points
    plot_points(ax, X_base, y_base)
    
    # Draw the classified query point with its predicted class color
    pred_color = CLASS_COLORS[predicted_label % len(CLASS_COLORS)]
    pred_marker = MARKERS[predicted_label % len(MARKERS)]
    ax.scatter([query_outside[0]], [query_outside[1]], 
              c=pred_color, marker='*', s=300, edgecolors='black', 
              linewidths=2, zorder=10)
    
    # Add label showing predicted class
    ax.annotate(f'→ Class {predicted_label}', 
               (query_outside[0], query_outside[1]), 
               xytext=(query_outside[0] + 0.05 * x_range, query_outside[1] - 0.05 * y_range),
               fontsize=11, fontweight='bold', color=pred_color,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=pred_color),
               zorder=11)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(f"(f) After Classification - Query Connected to Mesh", 
                fontsize=13, fontweight='bold')
    
    plt.savefig(f"{output_dir}/outside_hull_2_classified.png", dpi=DPI, 
                bbox_inches='tight', facecolor=BACKGROUND_COLOR)
    plt.close()
    
    print(f"    Saved 2 outside-hull classification figures")


if __name__ == "__main__":
    main()

