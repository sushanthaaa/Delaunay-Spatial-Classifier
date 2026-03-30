#!/usr/bin/env python3
"""
Unified Publication Figure Generator for Delaunay Triangulation Classifier

Generates ALL publication figures from a single script, matching the visual
style of the V19 paper:
  - Red solid lines: same-class DT edges
  - Red dashed lines: cross-class DT edges
  - Black solid lines: decision boundaries (extended to plot borders)
  - Distinct marker shapes per class: squares, triangles, circles, diamonds
  - White background, black border frame, no axis ticks
  - Gray dotted grid for SRR/2D Buckets overlay

Figure categories:
  A. Per-dataset pipeline figures (1–7 per dataset)
  B. Summary comparison charts (accuracy, speedup, dynamic, scalability)

Usage:
  python scripts/generate_publication_figures.py                        # All
  python scripts/generate_publication_figures.py --datasets moons,wine  # Specific
  python scripts/generate_publication_figures.py --summary-only         # Charts only
"""

import argparse
import math
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from scipy.spatial import Delaunay, Voronoi, ConvexHull
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings('ignore')

# =============================================================================
# STYLE CONFIGURATION — matches V19 paper exactly
# =============================================================================

# Marker shapes per class (up to 10 classes)
MARKERS = ['s', '^', 'o', 'D', 'p', 'h', '*', 'v', '<', '>']
MARKER_SIZE = 45

# Class colors (colorblind-accessible, matches paper)
CLASS_COLORS = [
    '#0066CC',  # Blue (class 0)
    '#FFD700',  # Yellow/Gold (class 1)
    '#00CC66',  # Green (class 2)
    '#CC3300',  # Red (class 3)
    '#9933CC',  # Purple (class 4)
    '#FF6600',  # Orange (class 5)
    '#00CCCC',  # Cyan (class 6)
    '#CC0066',  # Magenta (class 7)
    '#666666',  # Gray (class 8)
    '#339900',  # Dark Green (class 9)
]

# Edge colors
EDGE_SAME_CLASS = '#CC0000'       # Red solid for same-class DT edges
EDGE_CROSS_CLASS = '#CC0000'      # Red dashed for cross-class DT edges
DECISION_BOUNDARY = '#000000'     # Black for decision boundaries
GRID_COLOR = '#888888'            # Gray for SRR grid lines

# Figure settings
FIG_SIZE = (7, 7)
DPI = 300
BG_COLOR = 'white'

ALL_DATASETS = [
    'moons', 'circles', 'spiral', 'gaussian_quantiles', 'cassini',
    'checkerboard', 'blobs', 'earthquake',
    'wine', 'cancer', 'bloodmnist'
]

DATASET_NAMES = {
    'moons': 'Moons', 'circles': 'Circles', 'spiral': 'Spiral',
    'gaussian_quantiles': 'Gaussian Quantiles', 'cassini': 'Cassini',
    'checkerboard': 'Checkerboard', 'blobs': 'Blobs',
    'earthquake': 'Earthquake', 'wine': 'Wine',
    'cancer': 'Breast Cancer', 'bloodmnist': 'BloodMNIST'
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_csv(path):
    """Load x,y,label CSV (no header). Returns X, y."""
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path, header=None, names=['x', 'y', 'label'])
    return df[['x', 'y']].values, df['label'].values.astype(int)


def compute_bounds(X, margin_frac=0.08):
    """Compute plot bounds with relative margin."""
    rx = X[:, 0].max() - X[:, 0].min()
    ry = X[:, 1].max() - X[:, 1].min()
    mx = max(rx * margin_frac, 1e-3)
    my = max(ry * margin_frac, 1e-3)
    return (X[:, 0].min() - mx, X[:, 0].max() + mx,
            X[:, 1].min() - my, X[:, 1].max() + my)


# =============================================================================
# OUTLIER DETECTION (mirrors C++ Phase 1 logic)
# =============================================================================

def detect_outliers(X, y, k=3, multiplier=3.0):
    """Detect outliers using DT-based same-class connectivity.

    Mirrors the C++ implementation:
    1. Build temporary DT
    2. Compute median edge length
    3. Threshold = median * multiplier
    4. Build same-class adjacency graph (edges < threshold)
    5. DFS for connected components
    6. Remove components with < k members
    """
    if len(X) < 4:
        return np.zeros(len(X), dtype=bool)

    tri = Delaunay(X)

    # Compute all edge lengths
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            e = tuple(sorted([simplex[i], simplex[(i + 1) % 3]]))
            edges.add(e)

    edge_lengths = []
    for i, j in edges:
        d = np.sqrt(np.sum((X[i] - X[j]) ** 2))
        edge_lengths.append((i, j, d))

    if not edge_lengths:
        return np.zeros(len(X), dtype=bool)

    lengths_only = sorted([e[2] for e in edge_lengths])
    median_len = lengths_only[len(lengths_only) // 2]
    threshold = median_len * multiplier

    # Build same-class adjacency
    n = len(X)
    adj = [[] for _ in range(n)]
    for i, j, d in edge_lengths:
        if y[i] == y[j] and d < threshold:
            adj[i].append(j)
            adj[j].append(i)

    # DFS for connected components
    visited = [False] * n
    outlier_mask = np.zeros(n, dtype=bool)

    for start in range(n):
        if visited[start]:
            continue
        component = []
        stack = [start]
        visited[start] = True
        while stack:
            curr = stack.pop()
            component.append(curr)
            for nb in adj[curr]:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        if len(component) < k:
            for idx in component:
                outlier_mask[idx] = True

    return outlier_mask


# =============================================================================
# DRAWING PRIMITIVES — Paper Style
# =============================================================================

def create_figure(figsize=FIG_SIZE):
    """Create figure with paper style: white bg, black border, no ticks."""
    fig, ax = plt.subplots(figsize=figsize, facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_aspect('equal', adjustable='datalim')
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(2)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax


def draw_points(ax, X, y, size=MARKER_SIZE, alpha=1.0, zorder=5):
    """Draw points with distinct markers per class (paper style)."""
    for label in np.unique(y):
        mask = y == label
        color = CLASS_COLORS[label % len(CLASS_COLORS)]
        marker = MARKERS[label % len(MARKERS)]
        ax.scatter(X[mask, 0], X[mask, 1], c=color, marker=marker,
                   s=size, edgecolors='black', linewidths=0.5,
                   alpha=alpha, zorder=zorder)


def draw_dt_edges_by_class(ax, X, y, tri, lw_same=1.5, lw_cross=1.0):
    """Draw DT edges: solid red = same-class, dashed red = cross-class."""
    same_lines, cross_lines = [], []
    seen = set()
    for simplex in tri.simplices:
        for i in range(3):
            a, b = simplex[i], simplex[(i + 1) % 3]
            e = (min(a, b), max(a, b))
            if e in seen:
                continue
            seen.add(e)
            if y[a] == y[b]:
                same_lines.append([X[a], X[b]])
            else:
                cross_lines.append([X[a], X[b]])

    if same_lines:
        ax.add_collection(LineCollection(
            same_lines, colors=EDGE_SAME_CLASS,
            linewidths=lw_same, linestyles='-', zorder=1))
    if cross_lines:
        ax.add_collection(LineCollection(
            cross_lines, colors=EDGE_CROSS_CLASS,
            linewidths=lw_cross, linestyles='--', zorder=1))


def draw_dt_edges_uniform(ax, X, tri, color=EDGE_SAME_CLASS, lw=1.2,
                          ls='-'):
    """Draw all DT edges in uniform style."""
    lines = []
    seen = set()
    for simplex in tri.simplices:
        for i in range(3):
            a, b = simplex[i], simplex[(i + 1) % 3]
            e = (min(a, b), max(a, b))
            if e in seen:
                continue
            seen.add(e)
            lines.append([X[a], X[b]])
    if lines:
        ax.add_collection(LineCollection(
            lines, colors=color, linewidths=lw, linestyles=ls, zorder=1))


def draw_decision_boundaries(ax, X, y, xlim, ylim, lw=2.5):
    """Draw Voronoi-based decision boundaries as black lines, extended to axes."""
    try:
        vor = Voronoi(X)
    except Exception:
        return

    center = X.mean(axis=0)

    # Finite ridges
    for pidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        if y[pidx[0]] == y[pidx[1]]:
            continue
        if simplex[0] >= 0 and simplex[1] >= 0:
            v0 = vor.vertices[simplex[0]]
            v1 = vor.vertices[simplex[1]]
            ax.plot([v0[0], v1[0]], [v0[1], v1[1]],
                    color=DECISION_BOUNDARY, linewidth=lw, zorder=2)

    # Infinite ridges — extend to plot boundary
    for pidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        if y[pidx[0]] == y[pidx[1]]:
            continue
        if -1 not in simplex:
            continue

        finite_idx = simplex[1] if simplex[0] == -1 else simplex[0]
        t = X[pidx[1]] - X[pidx[0]]
        t = t / np.linalg.norm(t)
        n = np.array([-t[1], t[0]])

        midpoint = X[pidx].mean(axis=0)
        direction = np.sign(np.dot(midpoint - center, n)) * n

        far = vor.vertices[finite_idx] + direction * 100
        ax.plot([vor.vertices[finite_idx][0], far[0]],
                [vor.vertices[finite_idx][1], far[1]],
                color=DECISION_BOUNDARY, linewidth=lw, zorder=2,
                clip_on=True)


def draw_srr_grid(ax, X, xlim, ylim):
    """Draw SRR grid (ceil(sqrt(n)) x ceil(sqrt(n))) as gray dotted lines."""
    n = len(X)
    k = max(2, int(math.ceil(math.sqrt(n))))
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    for i in range(1, k):
        ax.axvline(xlim[0] + i * x_range / k, color=GRID_COLOR,
                   linestyle=':', linewidth=0.6, alpha=0.6, zorder=0)
        ax.axhline(ylim[0] + i * y_range / k, color=GRID_COLOR,
                   linestyle=':', linewidth=0.6, alpha=0.6, zorder=0)
    return k


def make_legend(ax, n_classes):
    """Add class legend with matching markers."""
    handles = []
    for i in range(n_classes):
        handles.append(Line2D(
            [0], [0], marker=MARKERS[i % len(MARKERS)], color='w',
            markerfacecolor=CLASS_COLORS[i % len(CLASS_COLORS)],
            markeredgecolor='black', markersize=8,
            label=f'Class {i}'))
    ax.legend(handles=handles, loc='best', fontsize=8, framealpha=0.9)


def save_fig(fig, path):
    """Save and close figure."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=DPI, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close(fig)
    print(f"    Saved: {os.path.basename(path)}")


# =============================================================================
# PER-DATASET PIPELINE FIGURES
# =============================================================================

def fig_1_raw_data(X, y, name, out_dir):
    """Figure 1: Raw data points with class markers."""
    fig, ax = create_figure()
    draw_points(ax, X, y)
    xlim = compute_bounds(X)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(xlim[2], xlim[3])
    n_cls = len(np.unique(y))
    make_legend(ax, n_cls)
    ax.set_title(f'{name} — (a) Raw Data (n={len(X)}, {n_cls} classes)',
                 fontsize=12, fontweight='bold')
    save_fig(fig, f'{out_dir}/1_raw_data.png')


def fig_2_delaunay_triangulation(X, y, name, out_dir):
    """Figure 2: DT mesh with same-class (solid) and cross-class (dashed) edges."""
    fig, ax = create_figure()
    tri = Delaunay(X)
    draw_dt_edges_by_class(ax, X, y, tri)
    draw_points(ax, X, y)
    xlim = compute_bounds(X)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(xlim[2], xlim[3])
    ax.set_title(f'{name} — (b) Delaunay Triangulation '
                 f'({len(tri.simplices)} triangles)',
                 fontsize=12, fontweight='bold')
    save_fig(fig, f'{out_dir}/2_delaunay_triangulation.png')


def fig_3_outlier_removal(X, y, name, out_dir):
    """Figure 3: Outlier removal — kept edges solid, removed edges dashed,
    outlier points highlighted."""
    outlier_mask = detect_outliers(X, y, k=3)
    n_outliers = outlier_mask.sum()

    X_clean = X[~outlier_mask]
    y_clean = y[~outlier_mask]

    fig, ax = create_figure()

    # Draw full DT with class-aware edges
    if len(X) >= 3:
        tri_full = Delaunay(X)
        # Edges touching outliers → thin dashed gray
        # Edges between clean points → solid red / dashed red by class
        seen = set()
        kept_same, kept_cross, removed = [], [], []
        for simplex in tri_full.simplices:
            for i in range(3):
                a, b = simplex[i], simplex[(i + 1) % 3]
                e = (min(a, b), max(a, b))
                if e in seen:
                    continue
                seen.add(e)
                if outlier_mask[a] or outlier_mask[b]:
                    removed.append([X[a], X[b]])
                elif y[a] == y[b]:
                    kept_same.append([X[a], X[b]])
                else:
                    kept_cross.append([X[a], X[b]])

        if kept_same:
            ax.add_collection(LineCollection(
                kept_same, colors=EDGE_SAME_CLASS,
                linewidths=1.5, linestyles='-', zorder=1))
        if kept_cross:
            ax.add_collection(LineCollection(
                kept_cross, colors=EDGE_CROSS_CLASS,
                linewidths=1.0, linestyles='--', zorder=1))
        if removed:
            ax.add_collection(LineCollection(
                removed, colors='#CC0000',
                linewidths=0.6, linestyles='--', alpha=0.4, zorder=1))

    # Draw clean points normally
    draw_points(ax, X[~outlier_mask], y[~outlier_mask])

    # Highlight outliers with red X
    if n_outliers > 0:
        ax.scatter(X[outlier_mask, 0], X[outlier_mask, 1],
                   c='red', marker='x', s=80, linewidths=2,
                   zorder=10, label=f'Outliers ({n_outliers})')
        ax.legend(loc='best', fontsize=8, framealpha=0.9)

    xlim = compute_bounds(X)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(xlim[2], xlim[3])
    ax.set_title(f'{name} — (c) Outlier Removal '
                 f'({len(X)}→{len(X_clean)} points, {n_outliers} removed)',
                 fontsize=11, fontweight='bold')
    save_fig(fig, f'{out_dir}/3_outlier_removal.png')
    return X_clean, y_clean


def fig_4_decision_boundaries(X, y, name, out_dir):
    """Figure 4: DT mesh + decision boundaries (black) + SRR grid."""
    if len(X) < 3:
        return

    fig, ax = create_figure()
    tri = Delaunay(X)
    xlim = compute_bounds(X)

    draw_dt_edges_by_class(ax, X, y, tri, lw_same=1.2, lw_cross=0.8)
    draw_decision_boundaries(ax, X, y, (xlim[0], xlim[1]), (xlim[2], xlim[3]))
    k = draw_srr_grid(ax, X, (xlim[0], xlim[1]), (xlim[2], xlim[3]))
    draw_points(ax, X, y)

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(xlim[2], xlim[3])
    ax.set_title(f'{name} — (d) Decision Boundaries + '
                 f'SRR Grid ({k}x{k})',
                 fontsize=11, fontweight='bold')
    save_fig(fig, f'{out_dir}/4_decision_boundaries.png')


def fig_5_srr_grid(X, y, name, out_dir):
    """Figure 5: SRR grid overlay on DT (dashed edges)."""
    if len(X) < 3:
        return

    fig, ax = create_figure()
    tri = Delaunay(X)
    xlim = compute_bounds(X)

    draw_dt_edges_uniform(ax, X, tri, color='#CC0000', lw=0.8, ls='--')
    k = draw_srr_grid(ax, X, (xlim[0], xlim[1]), (xlim[2], xlim[3]))
    draw_points(ax, X, y)

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(xlim[2], xlim[3])
    n_cls = len(np.unique(y))
    make_legend(ax, n_cls)
    ax.set_title(f'{name} — (e) SRR Grid ({k}x{k} = {k*k} buckets)',
                 fontsize=12, fontweight='bold')
    save_fig(fig, f'{out_dir}/5_srr_grid.png')


def fig_6_dynamic_update(X_base, y_base, X_stream, y_stream, name, out_dir):
    """Figure 6: Dynamic insertion — before/after with DT update."""
    if len(X_base) < 3:
        return

    n_insert = min(5, len(X_stream))
    X_new = X_stream[:n_insert]
    y_new = y_stream[:n_insert]

    X_combined = np.vstack([X_base, X_new])
    y_combined = np.hstack([y_base, y_new])
    xlim = compute_bounds(X_combined)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG_COLOR)

    # Panel (a): Before insertion
    ax = axes[0]
    ax.set_facecolor(BG_COLOR)
    ax.set_aspect('equal', adjustable='datalim')
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(2)
    ax.set_xticks([])
    ax.set_yticks([])

    tri_before = Delaunay(X_base)
    draw_dt_edges_by_class(ax, X_base, y_base, tri_before)
    draw_points(ax, X_base, y_base)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(xlim[2], xlim[3])
    ax.set_title(f'(a) Before: {len(X_base)} points',
                 fontsize=12, fontweight='bold')

    # Panel (b): After insertion
    ax = axes[1]
    ax.set_facecolor(BG_COLOR)
    ax.set_aspect('equal', adjustable='datalim')
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(2)
    ax.set_xticks([])
    ax.set_yticks([])

    tri_after = Delaunay(X_combined)
    draw_dt_edges_by_class(ax, X_combined, y_combined, tri_after)
    draw_points(ax, X_base, y_base)

    # Highlight new points with magenta ring
    for i in range(n_insert):
        label = y_new[i]
        color = CLASS_COLORS[label % len(CLASS_COLORS)]
        marker = MARKERS[label % len(MARKERS)]
        ax.scatter(X_new[i, 0], X_new[i, 1], c=color, marker=marker,
                   s=MARKER_SIZE * 1.5, edgecolors='black', linewidths=1.5,
                   zorder=6)
        ax.scatter(X_new[i, 0], X_new[i, 1], facecolors='none',
                   edgecolors='#FF00FF', s=MARKER_SIZE * 3,
                   linewidths=2, zorder=7)

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(xlim[2], xlim[3])
    ax.set_title(f'(b) After: +{n_insert} points inserted (O(1) each)',
                 fontsize=12, fontweight='bold')

    fig.suptitle(f'{name} — Dynamic Update', fontsize=14, fontweight='bold',
                 y=1.02)
    plt.tight_layout()
    save_fig(fig, f'{out_dir}/6_dynamic_update.png')


def fig_7_query_classification(X, y, X_stream, y_stream, name, out_dir):
    """Figure 7: Query classification via 2D Buckets — DT unchanged."""
    if len(X) < 3:
        return

    fig, ax = create_figure()
    tri = Delaunay(X)
    xlim = compute_bounds(X)

    draw_dt_edges_by_class(ax, X, y, tri, lw_same=1.0, lw_cross=0.7)
    draw_decision_boundaries(ax, X, y, (xlim[0], xlim[1]), (xlim[2], xlim[3]),
                             lw=2.0)
    draw_srr_grid(ax, X, (xlim[0], xlim[1]), (xlim[2], xlim[3]))
    draw_points(ax, X, y)

    # Query points (red stars with predicted class annotation)
    n_queries = min(3, len(X_stream))
    for i in range(n_queries):
        qx, qy = X_stream[i]
        # Classify by nearest vertex
        dists = np.sum((X - np.array([qx, qy])) ** 2, axis=1)
        pred = y[np.argmin(dists)]
        pred_color = CLASS_COLORS[pred % len(CLASS_COLORS)]

        ax.scatter(qx, qy, c='red', marker='*', s=200, edgecolors='black',
                   linewidths=1.5, zorder=10)
        rx = (xlim[1] - xlim[0])
        ax.annotate(f'Class {pred}', (qx, qy),
                    xytext=(qx + 0.05 * rx, qy + 0.05 * rx),
                    fontsize=9, fontweight='bold', color=pred_color,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    zorder=11)

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(xlim[2], xlim[3])
    ax.set_title(f'{name} — Query Classification (DT unchanged)',
                 fontsize=11, fontweight='bold')
    save_fig(fig, f'{out_dir}/7_query_classification.png')


# =============================================================================
# SUMMARY COMPARISON CHARTS
# =============================================================================

def read_benchmark_csv(csv_path):
    """Read cpp_benchmark CSV robustly, handling commas in method names.

    The CSV has columns: method,accuracy,avg_inference_us,train_time_ms,speedup_vs_knn
    Method names like 'LibSVM C++ (RBF, adaptive)' contain embedded commas,
    so we parse the last 4 fields as numeric and join the rest as method.
    """
    rows = []
    with open(csv_path) as f:
        header = f.readline().strip().split(',')
        for line in f:
            parts = line.strip().split(',')
            # Last 4 fields are always numeric
            numeric = parts[-4:]
            method = ','.join(parts[:-4])
            rows.append({
                'method': method,
                'accuracy': float(numeric[0]),
                'avg_inference_us': float(numeric[1]),
                'train_time_ms': float(numeric[2]),
                'speedup_vs_knn': float(numeric[3]),
            })
    return pd.DataFrame(rows)


def chart_accuracy_comparison(root_dir, figures_dir):
    """Bar chart comparing accuracy across all datasets and algorithms."""
    results = []
    for ds in ALL_DATASETS:
        csv_path = f'{root_dir}/results/cpp_benchmark_{ds}.csv'
        if os.path.exists(csv_path):
            df = read_benchmark_csv(csv_path)
            for _, row in df.iterrows():
                results.append({
                    'dataset': ds,
                    'method': row['method'].replace('**', '').strip(),
                    'accuracy': row['accuracy'] * 100
                })

    if not results:
        print("  No benchmark CSVs found. Run benchmarks first.")
        return

    df = pd.DataFrame(results)
    methods = df['method'].unique()
    datasets = df['dataset'].unique()

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(datasets))
    width = 0.8 / len(methods)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, method in enumerate(methods):
        vals = []
        for ds in datasets:
            sub = df[(df['dataset'] == ds) & (df['method'] == method)]
            vals.append(sub['accuracy'].values[0] if len(sub) > 0 else 0)
        label = method.split('(')[0].strip() if '(' in method else method
        ax.bar(x + i * width, vals, width, label=label,
               color=colors[i % len(colors)], edgecolor='black', linewidth=0.3)

    ax.set_xlabel('Dataset', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title('Classification Accuracy Comparison', fontsize=13,
                 fontweight='bold')
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels([DATASET_NAMES.get(d, d) for d in datasets],
                       rotation=45, ha='right', fontsize=9)
    ax.legend(loc='lower right', fontsize=8)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_fig(fig, f'{figures_dir}/summary_accuracy.png')


def chart_speedup_comparison(root_dir, figures_dir):
    """Bar chart showing speedup vs KNN for each dataset."""
    datasets_found = []
    speedups = []

    for ds in ALL_DATASETS:
        csv_path = f'{root_dir}/results/cpp_benchmark_{ds}.csv'
        if not os.path.exists(csv_path):
            continue
        df = read_benchmark_csv(csv_path)
        row = df[df['method'].str.contains('Delaunay', case=False, na=False)]
        if not row.empty and 'speedup_vs_knn' in df.columns:
            speedups.append(row['speedup_vs_knn'].values[0])
            datasets_found.append(DATASET_NAMES.get(ds, ds))

    if not datasets_found:
        print("  No speedup data found.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#2ca02c' if s >= 1000 else '#ff7f0e' for s in speedups]
    bars = ax.bar(datasets_found, speedups, color=colors,
                  edgecolor='black', linewidth=0.5)

    for bar, s in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{s:.0f}x', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Dataset', fontsize=11)
    ax.set_ylabel('Speedup vs KNN', fontsize=11)
    ax.set_title('Inference Speedup: Delaunay vs FLANN KNN',
                 fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.tight_layout()
    save_fig(fig, f'{figures_dir}/summary_speedup.png')


def chart_dynamic_comparison(root_dir, figures_dir):
    """Grouped bar chart: dynamic insert/move/delete vs DT rebuild."""
    datasets_found = []
    dt_rebuilds = []
    our_inserts = []
    our_moves = []
    our_deletes = []

    for ds in ALL_DATASETS:
        csv_path = f'{root_dir}/results/cpp_benchmark_{ds}.csv'
        if not os.path.exists(csv_path):
            continue

        # Dynamic results are printed in stdout, not CSV, so read from
        # ablation_dynamic_ CSVs instead
        dyn_csv = f'{root_dir}/results/ablation_dynamic_{ds}.csv'
        if os.path.exists(dyn_csv):
            df = pd.read_csv(dyn_csv)
            if len(df) > 0:
                row = df.iloc[0]
                datasets_found.append(DATASET_NAMES.get(ds, ds))
                our_inserts.append(row['avg_insert_ns'] / 1e6)
                our_moves.append(row['avg_move_ns'] / 1e6)
                our_deletes.append(row['avg_delete_ns'] / 1e6)

    if not datasets_found:
        print("  No dynamic ablation data found.")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(datasets_found))
    width = 0.25

    ax.bar(x - width, our_inserts, width, label='Insert', color='#2ca02c',
           edgecolor='black', linewidth=0.3)
    ax.bar(x, our_moves, width, label='Move', color='#ff7f0e',
           edgecolor='black', linewidth=0.3)
    ax.bar(x + width, our_deletes, width, label='Delete', color='#1f77b4',
           edgecolor='black', linewidth=0.3)

    ax.set_xlabel('Dataset', fontsize=11)
    ax.set_ylabel('Time (ms)', fontsize=11)
    ax.set_title('Dynamic Update Performance (Insert / Move / Delete)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets_found, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_fig(fig, f'{figures_dir}/summary_dynamic.png')


def chart_scalability(root_dir, figures_dir):
    """Two-panel plot: O(n log n) training + O(1) inference."""
    train_csv = f'{root_dir}/results/scalability_train.csv'
    infer_csv = f'{root_dir}/results/scalability_inference.csv'

    if not os.path.exists(train_csv) or not os.path.exists(infer_csv):
        print("  No scalability CSVs found. Run scalability_test.py first.")
        return

    df_t = pd.read_csv(train_csv)
    df_i = pd.read_csv(infer_csv)

    # Determine column names (handle both old and new formats)
    n_col = 'n'
    t_col = 'time_s' if 'time_s' in df_t.columns else 'train_time_ms'
    i_col = 'time_us' if 'time_us' in df_i.columns else 'inference_us'

    n_vals = df_t[n_col].values
    t_vals = df_t[t_col].values
    i_vals = df_i[i_col].values

    # Convert ms to s if needed
    if t_col == 'train_time_ms':
        t_vals = t_vals / 1000.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Training
    ax = axes[0]
    ax.loglog(n_vals, t_vals, 'bo-', linewidth=2, markersize=7,
              label='Measured')
    c = t_vals[0] / (n_vals[0] * np.log2(n_vals[0]))
    ref = c * n_vals * np.log2(n_vals.astype(float))
    ax.loglog(n_vals, ref, 'r--', linewidth=1.5, alpha=0.6,
              label='O(n log n) reference')
    ax.set_xlabel('Training points (n)', fontsize=11)
    ax.set_ylabel('Training time (s)', fontsize=11)
    ax.set_title('Training Complexity', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Inference
    ax = axes[1]
    ax.semilogx(n_vals, i_vals, 'gs-', linewidth=2, markersize=7,
                label='Measured')
    mean_i = np.mean(i_vals)
    ax.axhline(y=mean_i, color='r', linestyle='--', linewidth=1.5, alpha=0.6,
               label=f'O(1) mean = {mean_i:.3f} us')
    ax.set_xlabel('Training points (n)', fontsize=11)
    ax.set_ylabel('Inference time (us/point)', fontsize=11)
    ax.set_title('Inference Complexity: O(1) via 2D Buckets',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(i_vals) * 2 if len(i_vals) > 0 else 1)

    plt.tight_layout()
    save_fig(fig, f'{figures_dir}/summary_scalability.png')


def chart_ablation_accuracy(root_dir, figures_dir):
    """Grouped bar chart showing ablation accuracy impact."""
    summary_csv = f'{root_dir}/results/ablation_summary.csv'
    if not os.path.exists(summary_csv):
        print("  No ablation_summary.csv found. Run ablation_study.py first.")
        return

    df = pd.read_csv(summary_csv)

    # Filter to main ablation variants
    variants = ['Full Pipeline', 'Without SRR Grid',
                'Without Outlier Removal', 'Nearest Vertex Only']
    variant_labels = ['Full\nPipeline', 'No 2D\nBuckets',
                      'No Outlier\nRemoval', 'Nearest\nVertex']

    datasets = df['dataset'].unique()
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(datasets))
    width = 0.8 / len(variants)
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']

    for i, (var, label) in enumerate(zip(variants, variant_labels)):
        vals = []
        for ds in datasets:
            sub = df[(df['dataset'] == ds) &
                     (df['variant'].str.contains(var.split()[0], na=False))]
            if len(sub) > 0:
                vals.append(sub['accuracy'].values[0] * 100)
            else:
                vals.append(0)
        ax.bar(x + i * width, vals, width, label=label,
               color=colors[i], edgecolor='black', linewidth=0.3)

    ax.set_xlabel('Dataset', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title('Ablation Study: Component Contribution',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([DATASET_NAMES.get(d, d) for d in datasets],
                       rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_fig(fig, f'{figures_dir}/summary_ablation.png')


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def generate_dataset_figures(ds_key, root_dir, figures_dir):
    """Generate all pipeline figures for a single dataset."""
    name = DATASET_NAMES.get(ds_key, ds_key)
    out_dir = f'{figures_dir}/{ds_key}'
    os.makedirs(out_dir, exist_ok=True)

    train_path = f'{root_dir}/data/train/{ds_key}_train.csv'
    test_path = f'{root_dir}/data/test/{ds_key}_test_y.csv'
    base_path = f'{root_dir}/data/train/{ds_key}_dynamic_base.csv'
    stream_path = f'{root_dir}/data/train/{ds_key}_dynamic_stream.csv'

    X_train, y_train = load_csv(train_path)
    if X_train is None:
        print(f"  [SKIP] {name}: training data not found")
        return

    print(f"\n  [{name}] ({len(X_train)} points, "
          f"{len(np.unique(y_train))} classes)")

    # Fig 1: Raw data
    fig_1_raw_data(X_train, y_train, name, out_dir)

    # Fig 2: Delaunay triangulation
    fig_2_delaunay_triangulation(X_train, y_train, name, out_dir)

    # Fig 3: Outlier removal (returns clean data)
    X_clean, y_clean = fig_3_outlier_removal(
        X_train, y_train, name, out_dir)

    # Fig 4: Decision boundaries + SRR grid (on clean data)
    fig_4_decision_boundaries(X_clean, y_clean, name, out_dir)

    # Fig 5: SRR grid overlay
    fig_5_srr_grid(X_clean, y_clean, name, out_dir)

    # Fig 6 & 7: Dynamic update and query classification
    X_base, y_base = load_csv(base_path)
    X_stream, y_stream = load_csv(stream_path)

    if X_base is not None and X_stream is not None:
        fig_6_dynamic_update(X_base, y_base, X_stream, y_stream,
                             name, out_dir)
        fig_7_query_classification(X_clean, y_clean, X_stream, y_stream,
                                   name, out_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication figures for "
                    "Delaunay Triangulation Classifier")
    parser.add_argument('--datasets', default='all',
                        help='Comma-separated dataset names or "all"')
    parser.add_argument('--summary-only', action='store_true',
                        help='Generate only summary charts (no per-dataset)')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, ".."))
    figures_dir = f'{root_dir}/figures'
    os.makedirs(figures_dir, exist_ok=True)

    if args.datasets == 'all':
        datasets = ALL_DATASETS
    else:
        datasets = [d.strip() for d in args.datasets.split(',')]

    np.random.seed(42)

    print("=" * 70)
    print("PUBLICATION FIGURE GENERATOR")
    print(f"Output: {figures_dir}")
    print("=" * 70)

    # Per-dataset pipeline figures
    if not args.summary_only:
        for ds in datasets:
            generate_dataset_figures(ds, root_dir, figures_dir)

    # Summary comparison charts
    print("\n  [SUMMARY CHARTS]")
    chart_accuracy_comparison(root_dir, figures_dir)
    chart_speedup_comparison(root_dir, figures_dir)
    chart_dynamic_comparison(root_dir, figures_dir)
    chart_scalability(root_dir, figures_dir)
    chart_ablation_accuracy(root_dir, figures_dir)

    print("\n" + "=" * 70)
    print("FIGURE GENERATION COMPLETE")
    print(f"Output: {figures_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()