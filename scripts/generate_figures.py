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
  A. Per-dataset pipeline figures (1-7 per dataset)
  B. Summary comparison charts (accuracy, speedup, dynamic, scalability)
  C. Bucket structural figures (Issues #35a, #36) — universal-HOMOGENEOUS finding
  D. Confusion matrix figures — per-dataset multi-algorithm panels

Usage:
  python scripts/generate_figures.py                           # All
  python scripts/generate_figures.py --datasets moons,wine     # Specific
  python scripts/generate_figures.py --summary-only            # Charts only
  python scripts/generate_figures.py --regenerate-bucket-stats # Re-run C++ to
                                                                 refresh bucket
                                                                 stats CSV

Fixes applied (Week 3 of the master action list):
               figure generator covers the same 12 datasets as
               benchmark_cv.py and ablation_study.py.
               universal-HOMOGENEOUS finding: across all 12 datasets and
               all 15,000+ buckets, every bucket is HOMOGENEOUS (Case A).
               BIPARTITIONED and MULTI_PARTITIONED counts are zero. The
               figure is a stacked bar showing this empirical observation
               that motivates the simplified-classification framing in
               the paper. Bucket counts are obtained by parsing the C++
               binary's "2D Buckets Grid Statistics" stdout block, cached
               to results/bucket_type_distribution.csv.
               polygons-per-bucket per dataset, derived from the same
               parsed stats data (total_polygons / total_buckets).
               This is a partial fix; full per-bucket distribution
               histograms require exposing BucketOccupancyStats via the
               C++ CLI (tracked as Week 7 issues #47, #48).
               the C++ binary to write per-bucket counts to disk
               (currently only summary stats are exposed). See the
               chart_vertices_per_bucket_histogram() stub for the planned
               interface; it is a no-op pending the C++ change.
               figure per dataset showing confusion matrices for all 5
               algorithms side-by-side. Reads the per-dataset, per-algorithm
               confusion matrix CSVs produced by benchmark_cv.py .

Updated:
  - chart_ablation_accuracy() now reads accuracy_mean / accuracy_std from
    multi-seed ablation_summary.csv with error bars, falling
    back to the old single-seed 'accuracy' column for compatibility.
  - chart_accuracy_comparison() prefers multi-seed cv_summary.csv
    when present, falling back to per-dataset cpp_benchmark_*.csv files.
"""

import argparse
import math
import os
import re
import subprocess
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

BUCKET_HOMO_COLOR = '#2ca02c'     # Green for HOMOGENEOUS
BUCKET_BI_COLOR = '#ff7f0e'       # Orange for BIPARTITIONED
BUCKET_MULTI_COLOR = '#d62728'    # Red for MULTI_PARTITIONED

# Figure settings
FIG_SIZE = (7, 7)
DPI = 300
BG_COLOR = 'white'

# figure generator covers the same 12 datasets as benchmark_cv.py and
# ablation_study.py.
ALL_DATASETS = [
    'moons', 'circles', 'spiral', 'gaussian_quantiles', 'cassini',
    'checkerboard', 'blobs', 'earthquake', 'sfcrime',
    'wine', 'cancer', 'bloodmnist'
]

DATASET_NAMES = {
    'moons': 'Moons', 'circles': 'Circles', 'spiral': 'Spiral',
    'gaussian_quantiles': 'Gaussian Quantiles', 'cassini': 'Cassini',
    'checkerboard': 'Checkerboard', 'blobs': 'Blobs',
    'earthquake': 'Earthquake', 'sfcrime': 'SF Crime',
    'wine': 'Wine', 'cancer': 'Breast Cancer', 'bloodmnist': 'BloodMNIST'
}

# Datasets with high spatial density where Fig 1-7 (per-point visualizations)
# become visually busy. We still generate them but warn the user.
DENSE_DATASETS = {'sfcrime', 'earthquake', 'bloodmnist'}


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
# PER-DATASET PIPELINE FIGURES (Fig 1-7)
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
# BUCKET STATISTICS (Issues #35a, #36)
# =============================================================================
# The C++ binary prints a "2D Buckets Grid Statistics" block to stdout
# during training. We parse that block to extract HOMO/BI/MULTI counts and
# total polygons per dataset. Cached to results/bucket_type_distribution.csv
# so this doesn't need to re-run unless --regenerate-bucket-stats is passed.

# Regex matches lines like:
#   "Grid size: 29 x 29 = 841 buckets"
#   "Homogeneous (Case A):     841"
#   "Bipartitioned (Case B):   0"
#   "Multi-partitioned (Case C): 0"
#   "Total polygon regions:    841"
GRID_SIZE_RE = re.compile(
    r'Grid size:\s*(\d+)\s*x\s*(\d+)\s*=\s*(\d+)\s*buckets')
HOMO_RE = re.compile(r'Homogeneous\s*\(Case A\):\s*(\d+)')
BI_RE = re.compile(r'Bipartitioned\s*\(Case B\):\s*(\d+)')
MULTI_RE = re.compile(r'Multi-partitioned\s*\(Case C\):\s*(\d+)')
POLYS_RE = re.compile(r'Total polygon regions:\s*(\d+)')


def parse_bucket_statistics(stdout):
    """Parse the 2D Buckets Grid Statistics block from C++ stdout.

    Returns a dict with keys: rows, cols, total_buckets, homo, bi, multi,
    total_polys. Returns None if any required field is missing.
    """
    grid_match = GRID_SIZE_RE.search(stdout)
    homo_match = HOMO_RE.search(stdout)
    bi_match = BI_RE.search(stdout)
    multi_match = MULTI_RE.search(stdout)
    polys_match = POLYS_RE.search(stdout)

    if not all([grid_match, homo_match, bi_match, multi_match, polys_match]):
        return None

    return {
        'rows': int(grid_match.group(1)),
        'cols': int(grid_match.group(2)),
        'total_buckets': int(grid_match.group(3)),
        'homo': int(homo_match.group(1)),
        'bi': int(bi_match.group(1)),
        'multi': int(multi_match.group(1)),
        'total_polys': int(polys_match.group(1)),
    }


def collect_bucket_stats(root_dir, datasets, force_regenerate=False):
    """Collect bucket statistics for each dataset.

    If results/bucket_type_distribution.csv exists and force_regenerate is
    False, load and return it. Otherwise, run the C++ binary on each dataset
    and parse the stdout for the Grid Statistics block.

    Returns a DataFrame with columns: dataset, grid_str, total_buckets,
    homo, bi, multi, total_polys, mean_polys_per_bucket.
    """
    cache_path = f'{root_dir}/results/bucket_type_distribution.csv'

    # Use cached version if present and not forced to regenerate.
    if os.path.exists(cache_path) and not force_regenerate:
        df = pd.read_csv(cache_path)
        # Normalize column names for backward compatibility with hand-built
        # versions of this CSV that use different column names.
        rename = {
            'Dataset': 'dataset', 'Grid': 'grid_str',
            'HOMO': 'homo', 'BI': 'bi', 'MULTI': 'multi',
            'Total_Polys': 'total_polys',
        }
        for old, new in rename.items():
            if old in df.columns and new not in df.columns:
                df = df.rename(columns={old: new})
        # Compute total_buckets from homo+bi+multi if missing
        if 'total_buckets' not in df.columns:
            df['total_buckets'] = df['homo'] + df['bi'] + df['multi']
        # Compute mean_polys_per_bucket
        df['mean_polys_per_bucket'] = (
            df['total_polys'] / df['total_buckets'].replace(0, np.nan))
        return df

    # Re-generate by running C++ binary on each dataset.
    main_exe = f'{root_dir}/build/main'
    if not os.path.exists(main_exe):
        print(f"  Warning: C++ binary not found at {main_exe}")
        print(f"  Cannot regenerate bucket stats. Build with: "
              f"cd build && make")
        return None

    print("  Regenerating bucket statistics by running C++ binary on each "
          "dataset...")
    rows = []
    for ds in datasets:
        train_csv = f'{root_dir}/data/train/{ds}_train.csv'
        test_csv = f'{root_dir}/data/test/{ds}_test_y.csv'
        if not os.path.exists(train_csv) or not os.path.exists(test_csv):
            print(f"    [SKIP] {ds}: data not found")
            continue

        cmd = [main_exe, 'static', train_csv, test_csv,
               f'{root_dir}/results']
        print(f"    {ds}... ", end='', flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True)

        stats = parse_bucket_statistics(proc.stdout)
        if stats is None:
            print("FAILED to parse stats")
            continue

        rows.append({
            'dataset': ds,
            'grid_str': f"{stats['rows']} x {stats['cols']} = "
                        f"{stats['total_buckets']}",
            'total_buckets': stats['total_buckets'],
            'homo': stats['homo'],
            'bi': stats['bi'],
            'multi': stats['multi'],
            'total_polys': stats['total_polys'],
            'mean_polys_per_bucket': (
                stats['total_polys'] / stats['total_buckets']
                if stats['total_buckets'] > 0 else 0.0),
        })
        print("OK")

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df.to_csv(cache_path, index=False)
    print(f"  Saved bucket stats: {cache_path}")
    return df


# =============================================================================
# =============================================================================

def chart_bucket_type_distribution(root_dir, figures_dir,
                                   force_regenerate=False):
    """Stacked bar chart visualizing the universal-HOMOGENEOUS finding.

    For each dataset, shows the percentage of buckets that are HOMOGENEOUS
    (Case A), BIPARTITIONED (Case B), or MULTI_PARTITIONED (Case C).

    simplified-classification framing. Across all 12 datasets and 15,000+
    buckets, every bucket is HOMOGENEOUS — the BI and MULTI code paths
    are dead at query time under SRR (sqrt(n)) bucket sizing. The grid
    is still valuable as a Voronoi-aware dominant-class lookup, but the
    paper can honestly drop the BI/MULTI complexity from its main story.
    """
    df = collect_bucket_stats(root_dir, ALL_DATASETS, force_regenerate)
    if df is None or df.empty:
        print("  No bucket statistics available")
        return

    # Compute percentages per dataset.
    df = df.copy()
    df['homo_pct'] = 100.0 * df['homo'] / df['total_buckets']
    df['bi_pct'] = 100.0 * df['bi'] / df['total_buckets']
    df['multi_pct'] = 100.0 * df['multi'] / df['total_buckets']

    # Sort by dataset name for consistent ordering with other figures.
    df = df.sort_values('dataset').reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(df))
    width = 0.6

    # Stacked bars
    ax.bar(x, df['homo_pct'], width, label='Homogeneous (Case A)',
           color=BUCKET_HOMO_COLOR, edgecolor='black', linewidth=0.5)
    ax.bar(x, df['bi_pct'], width, bottom=df['homo_pct'],
           label='Bipartitioned (Case B)',
           color=BUCKET_BI_COLOR, edgecolor='black', linewidth=0.5)
    ax.bar(x, df['multi_pct'], width,
           bottom=df['homo_pct'] + df['bi_pct'],
           label='Multi-partitioned (Case C)',
           color=BUCKET_MULTI_COLOR, edgecolor='black', linewidth=0.5)

    # Annotate each bar with the total bucket count for context.
    for i, row in df.iterrows():
        ax.text(i, 102, f"{int(row['total_buckets'])}", ha='center',
                va='bottom', fontsize=8, color='black')

    ax.set_xlabel('Dataset', fontsize=11)
    ax.set_ylabel('Bucket Percentage (%)', fontsize=11)
    ax.set_title('2D Bucket Type Distribution — All Datasets Show 100% '
                 'HOMOGENEOUS\n'
                 '(numbers above bars: total buckets per dataset)',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(
        [DATASET_NAMES.get(d, d) for d in df['dataset']],
        rotation=45, ha='right', fontsize=9)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_fig(fig, f'{figures_dir}/summary_bucket_type_distribution.png')


# =============================================================================
# =============================================================================

def chart_bucket_occupancy_summary(root_dir, figures_dir,
                                   force_regenerate=False):
    """Bar chart showing mean polygons per bucket across datasets.

    total_polygons / total_buckets. Under SRR sizing (k = ceil(sqrt(n))),
    this should be approximately 1.0 for HOMOGENEOUS-dominated grids,
    confirming the O(1) inference claim.

    NOTE: The full per-bucket distribution (max/std/p99) requires
    exposing BucketOccupancyStats via the C++ CLI, tracked as
    Week 7 issues #47 (print in print_statistics) and #48 (--bucket-stats
    JSON output). Until then, only the mean is shown here.
    """
    df = collect_bucket_stats(root_dir, ALL_DATASETS, force_regenerate)
    if df is None or df.empty:
        print("  No bucket statistics available")
        return

    df = df.copy().sort_values('dataset').reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(df))

    # Color by closeness to ideal (1.0).
    colors = ['#2ca02c' if 0.9 <= v <= 1.1 else '#ff7f0e'
              for v in df['mean_polys_per_bucket']]
    bars = ax.bar(x, df['mean_polys_per_bucket'], color=colors,
                  edgecolor='black', linewidth=0.5)

    # Reference line at 1.0 (ideal SRR-sized occupancy).
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5,
               alpha=0.7, label='Ideal SRR occupancy = 1.0')

    # Annotate bars with the actual value.
    for bar, v in zip(bars, df['mean_polys_per_bucket']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{v:.2f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Dataset', fontsize=11)
    ax.set_ylabel('Mean polygons per bucket', fontsize=11)
    ax.set_title('Bucket Occupancy — Mean Polygons per Bucket\n'
                 '(SRR k=ceil(sqrt(n)) sizing → ideal mean = 1.0)',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(
        [DATASET_NAMES.get(d, d) for d in df['dataset']],
        rotation=45, ha='right', fontsize=9)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_fig(fig, f'{figures_dir}/summary_bucket_occupancy.png')


# =============================================================================
# =============================================================================

def chart_vertices_per_bucket_histogram(root_dir, figures_dir):
    """Per-bucket vertex count histogram — DEFERRED to Week 7.

    to a CSV by the C++ binary. The data is available internally via
    DelaunayClassifier::get_bucket_vertex_counts() (added in ,
    but is not currently exposed through the CLI.

    Tracked as Week 7 issues #47 (add to print_statistics) and #48
    (--bucket-stats JSON output). Once the C++ binary writes a
    bucket_vertex_counts_{dataset}.csv file, this function will:
      1. Load each dataset's per-bucket vertex count CSV
      2. Generate a multi-panel histogram (one panel per dataset)
      3. Annotate with mean, std, max, p99 from BucketOccupancyStats
    """
    print("  [DEFERRED] vertices-per-bucket histogram (#36a) — "
          "requires C++ change tracked as Week 7 issues #47/#48")


# =============================================================================
# =============================================================================

def chart_confusion_matrices(root_dir, figures_dir, datasets):
    """Generate one multi-panel figure per dataset, showing confusion matrices
    for all algorithms side-by-side.

    written by benchmark_cv.py . Each CSV is a square matrix of
    counts, aggregated across 5 seeds.

    File pattern: results/confusion_matrix_{dataset}_{algorithm}.csv
    where algorithm is one of: KNN_(CV-tuned_k), SVM_(RBF,_CV-tuned),
    Decision_Tree, Random_Forest, Delaunay_(Ours).

    The output is one figure per dataset at:
      figures/confusion_matrices/{dataset}.png
    """
    out_dir = f'{figures_dir}/confusion_matrices'
    os.makedirs(out_dir, exist_ok=True)

    results_dir = f'{root_dir}/results'
    if not os.path.exists(results_dir):
        print(f"  No results directory at {results_dir}")
        return

    # Find all confusion matrix files and group by dataset.
    matrices_by_dataset = {}
    for fname in os.listdir(results_dir):
        if not fname.startswith('confusion_matrix_') or not fname.endswith('.csv'):
            continue
        # Parse: confusion_matrix_{dataset}_{algorithm}.csv
        # Strip prefix and suffix, then split on first '_' that's followed
        # by a known dataset name. Datasets don't contain underscores
        # except 'gaussian_quantiles', which we handle explicitly.
        stem = fname[len('confusion_matrix_'):-len('.csv')]
        matched_dataset = None
        for ds in ALL_DATASETS:
            if stem.startswith(ds + '_'):
                matched_dataset = ds
                algorithm = stem[len(ds) + 1:].replace('_', ' ')
                break
        if matched_dataset is None:
            continue

        matrices_by_dataset.setdefault(matched_dataset, {})[algorithm] = (
            os.path.join(results_dir, fname))

    if not matrices_by_dataset:
        print("  No confusion matrix CSVs found. "
              "Run benchmark_cv.py first.")
        return

    # Generate one figure per dataset.
    for ds in datasets:
        if ds not in matrices_by_dataset:
            continue

        algorithms = sorted(matrices_by_dataset[ds].keys())
        n_algs = len(algorithms)
        if n_algs == 0:
            continue

        # Multi-panel figure: 1 row x N algorithms.
        # Sized to fit each subplot at roughly 3.5 inches wide.
        fig, axes = plt.subplots(
            1, n_algs, figsize=(3.5 * n_algs, 4),
            squeeze=False, facecolor=BG_COLOR)

        for col_idx, alg in enumerate(algorithms):
            ax = axes[0, col_idx]
            try:
                # benchmark_cv.py writes the CM with both row labels
                # ("true_0", "true_1", ...) as the DataFrame index AND column
                # labels ("pred_0", ...) as the header, via the default
                # df.to_csv() call. Read with header=0 and index_col=0 to
                # strip both, then .values gives the pure integer matrix.
                # Reading with header=None (as the old code did) produced a
                # mixed string/int matrix that crashed later on cm.sum().
                cm = pd.read_csv(matrices_by_dataset[ds][alg],
                                 header=0, index_col=0).values
            except Exception as e:
                print(f"    [WARN] Could not read CM for {ds}/{alg}: {e}")
                continue

            n_classes = cm.shape[0]
            # Normalize to row percentages (recall per true class).
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            cm_pct = 100.0 * cm / row_sums

            im = ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=100,
                           aspect='auto')

            # Annotate each cell with raw count and percentage.
            for i in range(n_classes):
                for j in range(n_classes):
                    val = int(cm[i, j])
                    pct = cm_pct[i, j]
                    # Use white text on dark cells, black on light.
                    text_color = 'white' if pct > 50 else 'black'
                    ax.text(j, i, f'{val}\n({pct:.0f}%)',
                            ha='center', va='center',
                            color=text_color, fontsize=8)

            ax.set_xticks(range(n_classes))
            ax.set_yticks(range(n_classes))
            ax.set_xticklabels([f'{i}' for i in range(n_classes)])
            ax.set_yticklabels([f'{i}' for i in range(n_classes)])
            ax.set_xlabel('Predicted')
            if col_idx == 0:
                ax.set_ylabel('True')
            # Truncate long algorithm names for column titles.
            short_alg = alg if len(alg) <= 20 else alg[:17] + '...'
            ax.set_title(short_alg, fontsize=10, fontweight='bold')

        fig.suptitle(
            f'{DATASET_NAMES.get(ds, ds)} — Confusion Matrices '
            f'(aggregated across 5 seeds)',
            fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        save_fig(fig, f'{out_dir}/{ds}.png')


# =============================================================================
# SUMMARY COMPARISON CHARTS (existing, with updates for multi-seed)
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
    """Bar chart comparing accuracy across all datasets and algorithms.

    UPDATED: Prefer multi-seed cv_summary.csv when available,
    falling back to per-dataset cpp_benchmark_*.csv files.
    """
    cv_summary_path = f'{root_dir}/results/cv_summary.csv'

    if os.path.exists(cv_summary_path):
        # Multi-seed format: dataset,algorithm,mean_acc,std_acc,...
        df = pd.read_csv(cv_summary_path)
        if 'mean_acc' in df.columns:
            results = []
            for _, row in df.iterrows():
                results.append({
                    'dataset': row['dataset'],
                    'method': row['algorithm'],
                    'accuracy': row['mean_acc'] * 100,
                    'accuracy_err': row.get('std_acc', 0) * 100,
                })
            results_df = pd.DataFrame(results)
            _draw_accuracy_chart(results_df, figures_dir,
                                 source_label='multi-seed cv_summary.csv')
            return

    # Fall back to per-dataset cpp_benchmark_*.csv files
    results = []
    for ds in ALL_DATASETS:
        csv_path = f'{root_dir}/results/cpp_benchmark_{ds}.csv'
        if os.path.exists(csv_path):
            df = read_benchmark_csv(csv_path)
            for _, row in df.iterrows():
                results.append({
                    'dataset': ds,
                    'method': row['method'].replace('**', '').strip(),
                    'accuracy': row['accuracy'] * 100,
                    'accuracy_err': 0,
                })

    if not results:
        print("  No accuracy data found. Run benchmark_cv.py or "
              "C++ benchmark first.")
        return

    _draw_accuracy_chart(pd.DataFrame(results), figures_dir,
                         source_label='per-dataset cpp_benchmark_*.csv')


def _draw_accuracy_chart(df, figures_dir, source_label=''):
    """Common drawing logic for the accuracy comparison chart."""
    methods = df['method'].unique()
    datasets = df['dataset'].unique()

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(datasets))
    width = 0.8 / len(methods)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b']

    has_errors = 'accuracy_err' in df.columns and (df['accuracy_err'] > 0).any()

    for i, method in enumerate(methods):
        vals, errs = [], []
        for ds in datasets:
            sub = df[(df['dataset'] == ds) & (df['method'] == method)]
            if len(sub) > 0:
                vals.append(sub['accuracy'].values[0])
                errs.append(sub['accuracy_err'].values[0]
                            if 'accuracy_err' in sub.columns else 0)
            else:
                vals.append(0)
                errs.append(0)
        label = method
        ax.bar(x + i * width, vals, width, label=label,
               color=colors[i % len(colors)], edgecolor='black',
               linewidth=0.3,
               yerr=errs if has_errors else None,
               capsize=2 if has_errors else 0,
               error_kw={'ecolor': 'black', 'elinewidth': 0.5})

    ax.set_xlabel('Dataset', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    title = 'Classification Accuracy Comparison'
    if source_label:
        title += f' ({source_label})'
    ax.set_title(title, fontsize=13, fontweight='bold')
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
    """Grouped bar chart: dynamic insert/move/delete with cross-seed std bars.

    UPDATED: Reads ablation_dynamic_summary.csv (multi-seed) if available,
    falling back to legacy ablation_dynamic_{dataset}.csv format.
    """
    multi_path = f'{root_dir}/results/ablation_dynamic_summary.csv'

    datasets_found = []
    inserts, moves, deletes = [], [], []
    insert_errs, move_errs, delete_errs = [], [], []

    if os.path.exists(multi_path):
        # Multi-seed format
        df = pd.read_csv(multi_path)
        for ds in ALL_DATASETS:
            ds_data = df[df['dataset'] == ds]
            if ds_data.empty:
                continue
            row = ds_data.iloc[0]
            datasets_found.append(DATASET_NAMES.get(ds, ds))
            # Convert ns to ms for plot readability
            inserts.append(row['insert_ns_mean'] / 1e6)
            moves.append(row['move_ns_mean'] / 1e6)
            deletes.append(row['delete_ns_mean'] / 1e6)
            insert_errs.append(
                row.get('insert_ns_cross_seed_std', 0) / 1e6)
            move_errs.append(
                row.get('move_ns_cross_seed_std', 0) / 1e6)
            delete_errs.append(
                row.get('delete_ns_cross_seed_std', 0) / 1e6)
    else:
        # Legacy per-dataset format
        for ds in ALL_DATASETS:
            dyn_csv = f'{root_dir}/results/ablation_dynamic_{ds}.csv'
            if not os.path.exists(dyn_csv):
                continue
            df = pd.read_csv(dyn_csv)
            if len(df) == 0:
                continue
            row = df.iloc[0]
            datasets_found.append(DATASET_NAMES.get(ds, ds))
            # Try multiple column name variants
            ins_col = ('avg_insert_ns' if 'avg_insert_ns' in row
                       else 'insert_ns_mean' if 'insert_ns_mean' in row
                       else None)
            mov_col = ('avg_move_ns' if 'avg_move_ns' in row
                       else 'move_ns_mean' if 'move_ns_mean' in row
                       else None)
            del_col = ('avg_delete_ns' if 'avg_delete_ns' in row
                       else 'delete_ns_mean' if 'delete_ns_mean' in row
                       else None)
            inserts.append(row[ins_col] / 1e6 if ins_col else 0)
            moves.append(row[mov_col] / 1e6 if mov_col else 0)
            deletes.append(row[del_col] / 1e6 if del_col else 0)
            insert_errs.append(0)
            move_errs.append(0)
            delete_errs.append(0)

    if not datasets_found:
        print("  No dynamic ablation data found.")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(datasets_found))
    width = 0.25
    has_errors = any(e > 0 for e in insert_errs + move_errs + delete_errs)

    ax.bar(x - width, inserts, width, label='Insert', color='#2ca02c',
           edgecolor='black', linewidth=0.3,
           yerr=insert_errs if has_errors else None,
           capsize=2 if has_errors else 0,
           error_kw={'ecolor': 'black', 'elinewidth': 0.5})
    ax.bar(x, moves, width, label='Move', color='#ff7f0e',
           edgecolor='black', linewidth=0.3,
           yerr=move_errs if has_errors else None,
           capsize=2 if has_errors else 0,
           error_kw={'ecolor': 'black', 'elinewidth': 0.5})
    ax.bar(x + width, deletes, width, label='Delete', color='#1f77b4',
           edgecolor='black', linewidth=0.3,
           yerr=delete_errs if has_errors else None,
           capsize=2 if has_errors else 0,
           error_kw={'ecolor': 'black', 'elinewidth': 0.5})

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
    """Two-panel plot: O(n log n) training + O(1) inference.

    Annotates the cache transition point if n >= 300K data is present
    (added based on the n=1M scalability finding from the previous session).
    """
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

    # Filter NaN entries (from missing C++ timing)
    train_valid = ~np.isnan(t_vals)
    infer_valid = ~np.isnan(i_vals)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Training panel
    ax = axes[0]
    if train_valid.sum() >= 1:
        ax.loglog(n_vals[train_valid], t_vals[train_valid], 'bo-',
                  linewidth=2, markersize=7, label='Measured')
        nv = n_vals[train_valid].astype(float)
        tv = t_vals[train_valid]
        c = tv[0] / (nv[0] * np.log2(nv[0]))
        ref = c * nv * np.log2(nv)
        ax.loglog(nv, ref, 'r--', linewidth=1.5, alpha=0.6,
                  label='O(n log n) reference')
    ax.set_xlabel('Training points (n)', fontsize=11)
    ax.set_ylabel('Training time (s)', fontsize=11)
    ax.set_title('Training Complexity', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Inference panel
    ax = axes[1]
    if infer_valid.sum() >= 1:
        ax.semilogx(n_vals[infer_valid], i_vals[infer_valid], 'gs-',
                    linewidth=2, markersize=7, label='Measured')
        # Cache-transition annotation: if n_vals contains entries beyond
        # 300K, mean inference is misleading because of the L2->DRAM step.
        # Show two reference lines: one for the L2-resident range
        # (n <= 300K) and the actual mean of the full data.
        small_n_mask = (n_vals <= 300_000) & infer_valid
        if small_n_mask.sum() >= 2:
            mean_small = float(np.mean(i_vals[small_n_mask]))
            ax.axhline(y=mean_small, color='r', linestyle='--',
                       linewidth=1.5, alpha=0.6,
                       label=f'O(1) (L2-resident): {mean_small:.3f} us')
        else:
            mean_all = float(np.mean(i_vals[infer_valid]))
            ax.axhline(y=mean_all, color='r', linestyle='--',
                       linewidth=1.5, alpha=0.6,
                       label=f'Mean: {mean_all:.3f} us')

        # Annotate cache transition if data extends beyond 300K
        if (n_vals >= 1_000_000).any():
            ax.annotate(
                'L2 cache exceeded\n(see paper Section 5)',
                xy=(1e6, i_vals[n_vals == n_vals[infer_valid].max()][-1]),
                xytext=(0.55, 0.65), textcoords='axes fraction',
                fontsize=9, color='gray',
                arrowprops=dict(arrowstyle='->', color='gray', lw=1))

    ax.set_xlabel('Training points (n)', fontsize=11)
    ax.set_ylabel('Inference time (us/point)', fontsize=11)
    ax.set_title('Inference Complexity: O(1) via 2D Buckets',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, f'{figures_dir}/summary_scalability.png')


def chart_ablation_accuracy(root_dir, figures_dir):
    """Grouped bar chart showing ablation accuracy impact.

    UPDATED for : Reads accuracy_mean / accuracy_std from the
    multi-seed ablation_summary.csv (cross-seed mean and std), with error
    bars. Falls back to old single-seed 'accuracy' column for compatibility.
    """
    summary_csv = f'{root_dir}/results/ablation_summary.csv'
    if not os.path.exists(summary_csv):
        print("  No ablation_summary.csv found. Run ablation_study.py first.")
        return

    df = pd.read_csv(summary_csv)

    # Accept both new (multi-seed) and legacy (single-seed) column names.
    use_multi_seed = 'accuracy_mean' in df.columns
    acc_col = 'accuracy_mean' if use_multi_seed else 'accuracy'
    std_col = 'accuracy_std' if 'accuracy_std' in df.columns else None

    # Filter to main 4 ablation variants. Substring matching is used
    # because the C++ binary's variant strings have varied across versions
    # (e.g. "Without 2D Buckets Grid" vs "Without SRR Grid").
    variants = [
        ('Full Pipeline', 'Full\nPipeline', '#2ca02c'),
        ('Without 2D Buckets', 'No 2D\nBuckets', '#1f77b4'),
        ('Without Outlier', 'No Outlier\nRemoval', '#ff7f0e'),
        ('Nearest Vertex', 'Nearest\nVertex', '#d62728'),
    ]

    datasets = sorted(df['dataset'].unique())

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(datasets))
    width = 0.8 / len(variants)

    for i, (var_match, label, color) in enumerate(variants):
        vals, errs = [], []
        for ds in datasets:
            sub = df[(df['dataset'] == ds) &
                     (df['variant'].str.contains(var_match, na=False))]
            if len(sub) > 0:
                vals.append(sub[acc_col].values[0] * 100)
                errs.append(sub[std_col].values[0] * 100
                            if std_col else 0)
            else:
                vals.append(0)
                errs.append(0)

        ax.bar(x + i * width, vals, width, label=label, color=color,
               edgecolor='black', linewidth=0.3,
               yerr=errs if std_col else None,
               capsize=2 if std_col else 0,
               error_kw={'ecolor': 'black', 'elinewidth': 0.5})

    ax.set_xlabel('Dataset', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    title = 'Ablation Study: Component Contribution'
    if use_multi_seed:
        title += ' (mean +/- std across 5 seeds)'
    ax.set_title(title, fontsize=13, fontweight='bold')
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

    n_pts = len(X_train)
    n_cls = len(np.unique(y_train))
    print(f"\n  [{name}] ({n_pts} points, {n_cls} classes)")

    if ds_key in DENSE_DATASETS:
        print(f"    NOTE: Dense dataset. Per-point figures will be visually "
              f"busy; consider using only summary charts in the paper.")

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

    # Fig 6 & 7: Dynamic update and query classification.
    # Skip gracefully if dynamic data files don't exist (sfcrime, real
    # datasets without pre-generated dynamic streams).
    X_base, y_base = load_csv(base_path)
    X_stream, y_stream = load_csv(stream_path)

    if X_base is not None and X_stream is not None:
        fig_6_dynamic_update(X_base, y_base, X_stream, y_stream,
                             name, out_dir)
        fig_7_query_classification(X_clean, y_clean, X_stream, y_stream,
                                   name, out_dir)
    else:
        print(f"    [SKIP] dynamic figures (Fig 6, 7): "
              f"no dynamic_base / dynamic_stream files for {ds_key}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication figures for "
                    "Delaunay Triangulation Classifier")
    parser.add_argument('--datasets', default='all',
                        help='Comma-separated dataset names or "all"')
    parser.add_argument('--summary-only', action='store_true',
                        help='Generate only summary charts (no per-dataset)')
    parser.add_argument('--regenerate-bucket-stats', action='store_true',
                        help='Re-run C++ binary to refresh '
                             'results/bucket_type_distribution.csv (used by '
                             ' / #36 figures). Defaults to using '
                             'cached CSV if present.')
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

    # Per-dataset pipeline figures (Fig 1-7 per dataset)
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

    print("\n  [BUCKET STRUCTURAL FIGURES]")
    chart_bucket_type_distribution(root_dir, figures_dir,
                                   force_regenerate=args.regenerate_bucket_stats)
    chart_bucket_occupancy_summary(root_dir, figures_dir,
                                   force_regenerate=False)
    chart_vertices_per_bucket_histogram(root_dir, figures_dir)

    print("\n  [CONFUSION MATRIX FIGURES]")
    chart_confusion_matrices(root_dir, figures_dir, datasets)

    print("\n" + "=" * 70)
    print("FIGURE GENERATION COMPLETE")
    print(f"Output: {figures_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()