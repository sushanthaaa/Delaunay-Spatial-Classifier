#!/usr/bin/env python3
"""
Dataset Generator for Delaunay Triangulation Classification

Generates all benchmark datasets used in the paper:

  Natively 2D spatial datasets (primary):
    - moons              : Two interleaving half-moons (2 classes)
    - circles            : Two concentric circles (2 classes)
    - spiral             : Two interlaced Archimedean spirals (2 classes)
    - gaussian_quantiles : Concentric ellipsoidal boundaries (2 classes)
    - cassini            : Two banana-shaped clusters + central blob (3 classes)
    - checkerboard       : Four-quadrant pattern (4 classes)
    - blobs              : Three Gaussian clusters (3 classes)
    - earthquake         : USGS real earthquake data by magnitude (4 classes)

  PCA-reduced datasets (secondary, showing generality):
    - wine               : UCI Wine (13D -> 2D via PCA, 3 classes)
    - cancer             : UCI Breast Cancer (30D -> 2D via PCA, 2 classes)
    - bloodmnist         : MedMNIST cell centroids (8 classes, ~17K samples)

Usage:
  python scripts/generate_datasets.py                                # All
  python scripts/generate_datasets.py --type moons                   # One
  python scripts/generate_datasets.py --type moons,spiral,earthquake # Several

Output structure:
  data/train/{name}_train.csv           -- Training data (x,y,label)
  data/test/{name}_test_X.csv           -- Test features only (x,y)
  data/test/{name}_test_y.csv           -- Test data with labels (x,y,label)
  data/train/{name}_dynamic_base.csv    -- Base subset for dynamic benchmarks
  data/train/{name}_dynamic_stream.csv  -- Stream subset for dynamic benchmarks

All files use CSV format with no header: x,y[,label]
Random seed: 42 (reproducible across runs)
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.datasets import make_moons, make_blobs, make_circles
from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

ALL_DATASETS = [
    # Natively 2D spatial (primary)
    'moons', 'circles', 'spiral', 'gaussian_quantiles', 'cassini',
    'checkerboard', 'blobs', 'earthquake',
    # PCA-reduced (secondary)
    'wine', 'cancer', 'bloodmnist'
]

# =============================================================================
# Output helpers
# =============================================================================

def create_output_dirs(root):
    """Create data/train and data/test directories."""
    os.makedirs(f"{root}/data/train", exist_ok=True)
    os.makedirs(f"{root}/data/test", exist_ok=True)


def save_csv(X, y, filepath):
    """Save points to CSV (x,y,label or x,y format, no header)."""
    df = pd.DataFrame(X, columns=['x', 'y'])
    if y is not None:
        df['label'] = y
    df.to_csv(filepath, index=False, header=False)
    print(f"  Saved: {filepath} ({len(X)} points)")


def save_dataset(X, y, name, root, test_size=0.2):
    """Save a complete dataset: train, test_X, test_y, dynamic_base, dynamic_stream."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
    )

    save_csv(X_train, y_train, f"{root}/data/train/{name}_train.csv")
    save_csv(X_test, None,     f"{root}/data/test/{name}_test_X.csv")
    save_csv(X_test, y_test,   f"{root}/data/test/{name}_test_y.csv")

    # Dynamic benchmark files: 60% base, 40% stream
    base_size = int(len(X_train) * 0.6)
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)

    save_csv(X_train[indices[:base_size]], y_train[indices[:base_size]],
             f"{root}/data/train/{name}_dynamic_base.csv")
    save_csv(X_train[indices[base_size:]], y_train[indices[base_size:]],
             f"{root}/data/train/{name}_dynamic_stream.csv")


def reduce_to_2d(X, y):
    """StandardScaler + PCA to 2 components for high-dimensional datasets."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    X_2d = pca.fit_transform(X_scaled)
    return X_2d, y


# =============================================================================
# Natively 2D spatial datasets (primary)
# =============================================================================

def generate_moons():
    """Two interleaving half-moons (1000 samples, 2 classes).

    Tests nonlinear decision boundaries. The two crescent-shaped clusters
    cannot be separated by a straight line, making this a standard benchmark
    for classifiers that learn curved boundaries.
    """
    print("\n[moons] Two interleaving half-moons")
    X, y = make_moons(n_samples=1000, noise=0.1, random_state=RANDOM_SEED)
    return X, y


def generate_circles_dataset():
    """Two concentric circles (1000 samples, 2 classes).

    Tests radial decision boundaries. The inner and outer circles require
    a classifier that can represent circular or ring-shaped regions, which
    Voronoi cells naturally approximate.
    """
    print("\n[circles] Two concentric circles")
    X, y = make_circles(n_samples=1000, noise=0.05, factor=0.5,
                        random_state=RANDOM_SEED)
    return X, y


def generate_spiral():
    """Two interlaced Archimedean spirals (1000 samples, 2 classes).

    Tests highly complex interleaving geometry. The spiral arms weave around
    each other, creating decision boundaries that change direction continuously.
    This is one of the hardest 2D classification benchmarks.
    """
    print("\n[spiral] Two interlaced Archimedean spirals")
    n_per_class = 500
    theta = np.linspace(0, 4 * np.pi, n_per_class)
    noise = 0.3

    r = theta
    x1 = r * np.cos(theta) + np.random.randn(n_per_class) * noise
    y1 = r * np.sin(theta) + np.random.randn(n_per_class) * noise
    x2 = -r * np.cos(theta) + np.random.randn(n_per_class) * noise
    y2 = -r * np.sin(theta) + np.random.randn(n_per_class) * noise

    X = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)]).astype(int)

    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, y


def generate_gaussian_quantiles():
    """Concentric ellipsoidal boundaries (1000 samples, 2 classes).

    Tests curved, non-circular decision boundaries. Unlike Circles (which has
    perfectly circular separation), Gaussian Quantiles produces ellipsoidal
    regions determined by the covariance structure of a multivariate Gaussian.
    This tests whether Voronoi-based bucket boundaries can approximate
    general curved regions, not just circles.
    """
    print("\n[gaussian_quantiles] Concentric Gaussian quantiles")
    X, y = make_gaussian_quantiles(n_samples=1000, n_features=2,
                                   n_classes=2, random_state=RANDOM_SEED)
    return X, y


def generate_cassini():
    """Cassini oval dataset (1500 samples, 3 classes).

    Tests non-convex class regions. Two banana-shaped clusters (classes 0 and 1)
    flank a central Gaussian blob (class 2). This is particularly challenging
    for axis-aligned methods (Decision Tree) and kernel methods (SVM) because
    the three regions interpenetrate in ways that require geometrically adaptive
    boundaries — exactly what Delaunay Triangulation provides.
    """
    print("\n[cassini] Cassini oval — three non-convex clusters")
    n = 500

    # Class 0: upper banana
    t = np.linspace(-np.pi, np.pi, n)
    x0 = t
    y0 = np.sin(t) + 2 + np.random.randn(n) * 0.15

    # Class 1: lower banana
    x1 = t
    y1 = -np.sin(t) - 2 + np.random.randn(n) * 0.15

    # Class 2: central blob
    x2 = np.random.randn(n) * 0.5
    y2 = np.random.randn(n) * 0.3

    X = np.vstack([np.column_stack([x0, y0]),
                   np.column_stack([x1, y1]),
                   np.column_stack([x2, y2])])
    y = np.hstack([np.zeros(n), np.ones(n), np.full(n, 2)]).astype(int)

    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, y


def generate_checkerboard():
    """Four-quadrant checkerboard pattern (1000 samples, 4 classes).

    Tests multi-class axis-aligned boundaries. Each quadrant of the plane
    belongs to a different class, creating sharp linear boundaries at x=0
    and y=0. This is easy for Decision Trees but tests whether the DT-based
    approach handles axis-aligned boundaries without overfitting.
    """
    print("\n[checkerboard] Four-quadrant checkerboard")
    n = 1000
    x = np.random.uniform(-1, 1, n)
    y_coord = np.random.uniform(-1, 1, n)
    labels = ((x > 0).astype(int) * 2) + (y_coord > 0).astype(int)
    X = np.column_stack([x, y_coord])
    return X, labels


def generate_blobs():
    """Three Gaussian blobs (1500 samples, 3 classes).

    Baseline dataset with well-separated clusters. Included to verify that
    the algorithm handles simple cases correctly — all methods should achieve
    near-perfect accuracy here. Primarily useful for scalability and dynamic
    update benchmarks rather than classification difficulty.
    """
    print("\n[blobs] Three Gaussian blobs")
    X, y = make_blobs(n_samples=1500, centers=3, cluster_std=1.5,
                      random_state=RANDOM_SEED)
    return X, y


def generate_earthquake():
    """USGS earthquake data (2023-2026, M>=2.5, classified by magnitude).

    Real-world geospatial dataset. Earthquake epicenters (lat/lon) are
    classified into 4 magnitude categories:
      0 = small (2.5-4.0), 1 = moderate (4.0-5.5),
      2 = large (5.5-7.0), 3 = major (7.0+)

    This is the only real-world spatial dataset in the benchmark and
    demonstrates practical applicability to GIS and geoscience domains.
    Data is fetched from the USGS Earthquake Hazards Program API.
    """
    print("\n[earthquake] USGS Earthquake data (real GIS)")

    try:
        import requests
    except ImportError:
        print("  'requests' package not found. Installing...")
        os.system(f"{sys.executable} -m pip install requests")
        import requests

    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "starttime": "2023-01-01",
        "endtime": "2026-01-18",
        "minmagnitude": 2.5,
        "limit": 5000,
        "orderby": "time"
    }

    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()

        earthquakes = []
        for eq in data['features']:
            coords = eq['geometry']['coordinates']
            mag = eq['properties'].get('mag')
            if coords and len(coords) >= 2 and mag is not None:
                lat, lon = coords[1], coords[0]
                label = min(3, int((mag - 2.5) / 1.5))
                earthquakes.append([lat, lon, label])

        X = np.array([[eq[0], eq[1]] for eq in earthquakes])
        y = np.array([eq[2] for eq in earthquakes])

        X = (X - X.mean(axis=0)) / X.std(axis=0)

        print(f"  Downloaded {len(X)} earthquakes (4 magnitude classes)")
        print(f"  Class distribution: {np.bincount(y)}")
        return X, y

    except Exception as e:
        print(f"  Error downloading: {e}")
        print("  Generating synthetic earthquake-like data as fallback...")
        return _synthetic_earthquake_fallback()


def _synthetic_earthquake_fallback():
    """Fallback: clustered data mimicking tectonic plate locations."""
    centers = [(35.0, 139.0), (37.0, -122.0), (-33.0, -70.0), (28.0, 84.0)]
    points = []
    for i, (lat, lon) in enumerate(centers):
        n = 500
        x = np.random.normal(lat, 5, n)
        y = np.random.normal(lon, 10, n)
        points.append(np.column_stack([x, y, np.full(n, i)]))
    data = np.vstack(points)
    X = (data[:, :2] - data[:, :2].mean(axis=0)) / data[:, :2].std(axis=0)
    return X, data[:, 2].astype(int)


# =============================================================================
# PCA-reduced datasets (secondary)
# =============================================================================

def generate_wine():
    """UCI Wine dataset (178 samples, 3 classes, 13 features -> PCA to 2D).

    Demonstrates that the classifier generalizes beyond natively 2D data.
    Wine's first two principal components capture substantial variance,
    making PCA reduction viable without excessive information loss.
    """
    print("\n[wine] UCI Wine dataset (PCA: 13D -> 2D)")
    data = datasets.load_wine()
    return reduce_to_2d(data.data, data.target)


def generate_cancer():
    """UCI Breast Cancer dataset (569 samples, 2 classes, 30D -> PCA to 2D).

    Medical domain dataset. The two classes (malignant/benign) separate
    reasonably well in the first two principal components, making this a
    practical test of the classifier on dimensionality-reduced clinical data.
    """
    print("\n[cancer] UCI Breast Cancer dataset (PCA: 30D -> 2D)")
    data = datasets.load_breast_cancer()
    return reduce_to_2d(data.data, data.target)


def generate_bloodmnist(root):
    """BloodMNIST cell centroid extraction (8 classes, ~17K samples).

    Large-scale medical imaging dataset. Cell centroids are extracted from
    blood cell microscopy images, providing a high-volume test of the
    classifier's scalability. Primarily included to demonstrate that the
    O(1) inference property holds at scale (17K training points).
    """
    print("\n[bloodmnist] BloodMNIST cell centroids")

    try:
        import medmnist
        from medmnist import BloodMNIST
        from skimage.measure import regionprops, label as ski_label
    except ImportError:
        print("  Installing medmnist and scikit-image...")
        os.system(f"{sys.executable} -m pip install medmnist scikit-image")
        import medmnist
        from medmnist import BloodMNIST
        from skimage.measure import regionprops, label as ski_label

    try:
        train_dataset = BloodMNIST(split='train', download=True,
                                   root=f"{root}/data/medmnist")
        test_dataset = BloodMNIST(split='test', download=True,
                                  root=f"{root}/data/medmnist")

        def extract_centroids(dataset):
            centroids, labels = [], []
            for img, lbl in dataset:
                img_np = np.array(img.convert('L'))
                binary = img_np > 128
                labeled = ski_label(binary)
                props = regionprops(labeled)
                if props:
                    largest = max(props, key=lambda p: p.area)
                    cy, cx = largest.centroid
                    centroids.append([cx / img_np.shape[1],
                                     cy / img_np.shape[0]])
                    labels.append(int(lbl[0]) if hasattr(lbl, '__len__')
                                  else int(lbl))
            return np.array(centroids), np.array(labels)

        X_train, y_train = extract_centroids(train_dataset)
        X_test, y_test = extract_centroids(test_dataset)

        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std < 1e-10] = 1.0
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        print(f"  Extracted {len(X_train)} train + {len(X_test)} test centroids")
        return X_train, y_train, X_test, y_test

    except Exception as e:
        print(f"  Error processing BloodMNIST: {e}")
        print("  Generating synthetic 8-class clustered data as fallback...")
        return _synthetic_bloodmnist_fallback()


def _synthetic_bloodmnist_fallback():
    """Fallback: 8-class clustered data mimicking cell distributions."""
    n_per_class = 2125
    n_classes = 8
    points = []
    for i in range(n_classes):
        angle = 2 * np.pi * i / n_classes
        cx, cy = 0.3 * np.cos(angle), 0.3 * np.sin(angle)
        x = np.random.normal(cx, 0.1, n_per_class)
        y = np.random.normal(cy, 0.1, n_per_class)
        points.append(np.column_stack([x, y, np.full(n_per_class, i)]))
    data = np.vstack(points)
    np.random.shuffle(data)
    X, y = data[:, :2], data[:, 2].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    return X_train, y_train, X_test, y_test


# =============================================================================
# Dataset registry
# =============================================================================

GENERATORS = {
    'moons': generate_moons,
    'circles': generate_circles_dataset,
    'spiral': generate_spiral,
    'gaussian_quantiles': generate_gaussian_quantiles,
    'cassini': generate_cassini,
    'checkerboard': generate_checkerboard,
    'blobs': generate_blobs,
    'earthquake': generate_earthquake,
    'wine': generate_wine,
    'cancer': generate_cancer,
    # bloodmnist handled specially (returns pre-split train/test)
}


# =============================================================================
# Main
# =============================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    parser = argparse.ArgumentParser(
        description="Generate benchmark datasets for "
                    "Delaunay Triangulation Classification")
    parser.add_argument('--type', type=str, default='all',
                        help='Dataset name(s), comma-separated, '
                             'or "all" (default: all)')
    parser.add_argument('--out_dir', type=str, default=project_root,
                        help='Project root directory (default: auto-detected)')
    args = parser.parse_args()

    root = args.out_dir
    create_output_dirs(root)

    if args.type == 'all':
        requested = ALL_DATASETS
    else:
        requested = [s.strip() for s in args.type.split(',')]
        for name in requested:
            if name not in ALL_DATASETS:
                print(f"Error: Unknown dataset '{name}'. "
                      f"Available: {', '.join(ALL_DATASETS)}")
                sys.exit(1)

    print("=" * 70)
    print("DATASET GENERATION FOR DELAUNAY TRIANGULATION CLASSIFICATION")
    print(f"Datasets: {', '.join(requested)}")
    print(f"Output:   {root}/data/")
    print(f"Seed:     {RANDOM_SEED}")
    print("=" * 70)

    for name in requested:
        if name == 'bloodmnist':
            X_train, y_train, X_test, y_test = generate_bloodmnist(root)

            save_csv(X_train, y_train,
                     f"{root}/data/train/bloodmnist_train.csv")
            save_csv(X_test, None,
                     f"{root}/data/test/bloodmnist_test_X.csv")
            save_csv(X_test, y_test,
                     f"{root}/data/test/bloodmnist_test_y.csv")

            base_size = int(len(X_train) * 0.6)
            indices = np.arange(len(X_train))
            np.random.shuffle(indices)
            save_csv(X_train[indices[:base_size]],
                     y_train[indices[:base_size]],
                     f"{root}/data/train/bloodmnist_dynamic_base.csv")
            save_csv(X_train[indices[base_size:]],
                     y_train[indices[base_size:]],
                     f"{root}/data/train/bloodmnist_dynamic_stream.csv")
        else:
            generator = GENERATORS[name]
            X, y = generator()
            save_dataset(X, y, name, root)

    print("\n" + "=" * 70)
    print("DATASET GENERATION COMPLETE")
    print("=" * 70)

    n_spatial = sum(1 for d in requested if d in [
        'moons', 'circles', 'spiral', 'gaussian_quantiles', 'cassini',
        'checkerboard', 'blobs', 'earthquake'])
    n_pca = sum(1 for d in requested if d in ['wine', 'cancer', 'bloodmnist'])

    print(f"\nGenerated {len(requested)} dataset(s):")
    print(f"  {n_spatial} natively 2D spatial (primary benchmarks)")
    print(f"  {n_pca} PCA-reduced / extracted (generality benchmarks)")
    print(f"\nOutput directory: {root}/data/")


if __name__ == "__main__":
    main()