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
    - sfcrime            : SF Open Data crime incidents, property vs violent
                           (2 classes, balanced, city-scale urban GIS)

  PCA-reduced datasets (secondary, showing generality):
    - wine               : UCI Wine (13D -> 2D via PCA, 3 classes)
    - cancer             : UCI Breast Cancer (30D -> 2D via PCA, 2 classes)
    - bloodmnist         : MedMNIST cell centroids (8 classes, ~17K samples)

Usage:
  python scripts/generate_datasets.py                                # All
  python scripts/generate_datasets.py --type moons                   # One
  python scripts/generate_datasets.py --type moons,spiral,earthquake # Several
  python scripts/generate_datasets.py --seed 123                     # Custom seed
  python scripts/generate_datasets.py --force-fetch                  # Bypass cache

Output structure:
  data/train/{name}_train.csv           -- Training data (x,y,label)
  data/test/{name}_test_X.csv           -- Test features only (x,y)
  data/test/{name}_test_y.csv           -- Test data with labels (x,y,label)
  data/train/{name}_dynamic_base.csv    -- Base subset for dynamic benchmarks
  data/train/{name}_dynamic_stream.csv  -- Stream subset for dynamic benchmarks

Cached real-world data (shipped in repo for reproducibility):
  data/cached/earthquake_raw.csv        -- Raw USGS API response (lat,lon,label)
  data/cached/sfcrime_raw.csv           -- Raw SF Open Data API response (lat,lon,label)
  data/cached/bloodmnist_train_centroids.csv -- Extracted centroids (x,y,label)
  data/cached/bloodmnist_test_centroids.csv  -- Extracted centroids (x,y,label)

All files use CSV format with no header: x,y[,label]
Random seed: 42 by default, configurable via --seed

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

# Default random seed. The actual seed used at runtime is supplied via --seed
# and propagated to every generator. This constant exists only as the default
# CLI value and should NOT be relied upon at module level.
DEFAULT_SEED = 42

ALL_DATASETS = [
    # Natively 2D spatial (primary)
    'moons', 'circles', 'spiral', 'gaussian_quantiles', 'cassini',
    'checkerboard', 'blobs', 'earthquake', 'sfcrime',
    # PCA-reduced (secondary)
    'wine', 'cancer', 'bloodmnist'
]

# =============================================================================
# Output helpers
# =============================================================================

def create_output_dirs(root):
    """Create data/train, data/test, and data/cached directories."""
    os.makedirs(f"{root}/data/train", exist_ok=True)
    os.makedirs(f"{root}/data/test", exist_ok=True)
    os.makedirs(f"{root}/data/cached", exist_ok=True)


def save_csv(X, y, filepath):
    """Save points to CSV (x,y,label or x,y format, no header)."""
    df = pd.DataFrame(X, columns=['x', 'y'])
    if y is not None:
        df['label'] = y
    df.to_csv(filepath, index=False, header=False)
    print(f"  Saved: {filepath} ({len(X)} points)")


def save_dataset(X, y, name, root, seed, test_size=0.2):
    """Save a complete dataset: train, test_X, test_y, dynamic_base, dynamic_stream.

    dynamic base/stream shuffle are deterministic per seed.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    save_csv(X_train, y_train, f"{root}/data/train/{name}_train.csv")
    save_csv(X_test, None,     f"{root}/data/test/{name}_test_X.csv")
    save_csv(X_test, y_test,   f"{root}/data/test/{name}_test_y.csv")

    # Dynamic benchmark files: 60% base, 40% stream. Use a dedicated
    # RandomState so the shuffle doesn't depend on whatever global numpy
    # state happens to exist at this point in execution.
    base_size = int(len(X_train) * 0.6)
    indices = np.arange(len(X_train))
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    save_csv(X_train[indices[:base_size]], y_train[indices[:base_size]],
             f"{root}/data/train/{name}_dynamic_base.csv")
    save_csv(X_train[indices[base_size:]], y_train[indices[base_size:]],
             f"{root}/data/train/{name}_dynamic_stream.csv")


def reduce_to_2d(X, y, seed):
    """StandardScaler + PCA to 2 components for high-dimensional datasets.

    random_state (PCA's SVD is deterministic but the API still accepts a
    random_state for reproducibility of the randomized SVD solver).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=seed)
    X_2d = pca.fit_transform(X_scaled)
    return X_2d, y


# =============================================================================
# Natively 2D spatial datasets (primary)
# =============================================================================

def generate_moons(seed):
    """Two interleaving half-moons (1000 samples, 2 classes).

    Tests nonlinear decision boundaries. The two crescent-shaped clusters
    cannot be separated by a straight line, making this a standard benchmark
    for classifiers that learn curved boundaries.
    """
    print("\n[moons] Two interleaving half-moons")
    X, y = make_moons(n_samples=1000, noise=0.1, random_state=seed)
    return X, y


def generate_circles_dataset(seed):
    """Two concentric circles (1000 samples, 2 classes).

    Tests radial decision boundaries. The inner and outer circles require
    a classifier that can represent circular or ring-shaped regions, which
    Voronoi cells naturally approximate.
    """
    print("\n[circles] Two concentric circles")
    X, y = make_circles(n_samples=1000, noise=0.05, factor=0.5,
                        random_state=seed)
    return X, y


def generate_spiral(seed):
    """Two interlaced Archimedean spirals (1000 samples, 2 classes).

    Tests highly complex interleaving geometry. The spiral arms weave around
    each other, creating decision boundaries that change direction continuously.
    This is one of the hardest 2D classification benchmarks.
    """
    print("\n[spiral] Two interlaced Archimedean spirals")
    n_per_class = 500
    theta = np.linspace(0, 4 * np.pi, n_per_class)
    noise = 0.3

    # Spiral uses numpy's global random state for backward compatibility
    r = theta
    x1 = r * np.cos(theta) + np.random.randn(n_per_class) * noise
    y1 = r * np.sin(theta) + np.random.randn(n_per_class) * noise
    x2 = -r * np.cos(theta) + np.random.randn(n_per_class) * noise
    y2 = -r * np.sin(theta) + np.random.randn(n_per_class) * noise

    X = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)]).astype(int)

    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, y


def generate_gaussian_quantiles(seed):
    """Concentric ellipsoidal boundaries (1000 samples, 2 classes).

    Tests curved, non-circular decision boundaries. Unlike Circles (which has
    perfectly circular separation), Gaussian Quantiles produces ellipsoidal
    regions determined by the covariance structure of a multivariate Gaussian.
    This tests whether Voronoi-based bucket boundaries can approximate
    general curved regions, not just circles.
    """
    print("\n[gaussian_quantiles] Concentric Gaussian quantiles")
    X, y = make_gaussian_quantiles(n_samples=1000, n_features=2,
                                   n_classes=2, random_state=seed)
    return X, y


def generate_cassini(seed):
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


def generate_checkerboard(seed):
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


def generate_blobs(seed):
    """Three Gaussian blobs (1500 samples, 3 classes).

    Baseline dataset with well-separated clusters. Included to verify that
    the algorithm handles simple cases correctly — all methods should achieve
    near-perfect accuracy here. Primarily useful for scalability and dynamic
    update benchmarks rather than classification difficulty.
    """
    print("\n[blobs] Three Gaussian blobs")
    X, y = make_blobs(n_samples=1500, centers=3, cluster_std=1.5,
                      random_state=seed)
    return X, y


def generate_earthquake(seed, root, force_fetch=False):
    """USGS earthquake data (2023-2026, M>=2.5, classified by magnitude).

    Real-world geospatial dataset. Earthquake epicenters (lat/lon) are
    classified into 4 magnitude categories:
      0 = small (2.5-4.0), 1 = moderate (4.0-5.5),
      2 = large (5.5-7.0), 3 = major (7.0+)

    This is the only real-world spatial dataset in the benchmark and
    demonstrates practical applicability to GIS and geoscience domains.

      Data is loaded from data/cached/earthquake_raw.csv if present. This
      cached file is a committed artifact shipped with the repo, so anyone
      reproducing the paper gets the exact same earthquakes we used, rather
      than whatever the USGS API happens to return on the day they run the
      script. The live USGS API fetch is used only when:
        (a) the cached file does not exist (first-time generation), or
        (b) --force-fetch is passed on the command line (intentional refresh).
      When a fresh fetch happens, the raw response is saved to the cache
      directory so future runs are reproducible without re-downloading.

    Note: the cached file stores RAW lat/lon/label values (pre-standardization).
    Standardization is applied after loading so the output is deterministic
    regardless of seed — earthquake is real data, not synthetic.
    """
    print("\n[earthquake] USGS Earthquake data (real GIS)")

    cache_path = f"{root}/data/cached/earthquake_raw.csv"

    if os.path.exists(cache_path) and not force_fetch:
        print(f"  Loading from cache: {cache_path}")
        try:
            df = pd.read_csv(cache_path, header=None, names=['lat', 'lon', 'label'])
            raw_coords = df[['lat', 'lon']].to_numpy(dtype=float)
            raw_labels = df['label'].to_numpy(dtype=int)
            X = (raw_coords - raw_coords.mean(axis=0)) / raw_coords.std(axis=0)
            print(f"  Loaded {len(X)} earthquakes from cache "
                  f"(4 magnitude classes)")
            print(f"  Class distribution: {np.bincount(raw_labels)}")
            return X, raw_labels
        except Exception as e:
            print(f"  Cache read failed ({e}); falling back to live fetch.")

    # Fallback path: live USGS API fetch. This only runs on first-time
    # generation or when --force-fetch is passed.
    print("  Cache miss (or forced fetch); downloading from USGS API...")
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

        raw_coords = np.array([[eq[0], eq[1]] for eq in earthquakes],
                              dtype=float)
        raw_labels = np.array([eq[2] for eq in earthquakes], dtype=int)

        # are reproducible without needing API access.
        cache_df = pd.DataFrame({
            'lat': raw_coords[:, 0],
            'lon': raw_coords[:, 1],
            'label': raw_labels
        })
        cache_df.to_csv(cache_path, index=False, header=False)
        print(f"  Cached raw response to: {cache_path}")

        X = (raw_coords - raw_coords.mean(axis=0)) / raw_coords.std(axis=0)

        print(f"  Downloaded {len(X)} earthquakes (4 magnitude classes)")
        print(f"  Class distribution: {np.bincount(raw_labels)}")
        return X, raw_labels

    except Exception as e:
        print(f"  Error downloading: {e}")
        print("  Generating synthetic earthquake-like data as fallback...")
        return _synthetic_earthquake_fallback(seed)


def _synthetic_earthquake_fallback(seed):
    """Fallback: clustered data mimicking tectonic plate locations.

    Used only when both the cache and the live API are unavailable.
    """
    rng = np.random.RandomState(seed)
    centers = [(35.0, 139.0), (37.0, -122.0), (-33.0, -70.0), (28.0, 84.0)]
    points = []
    for i, (lat, lon) in enumerate(centers):
        n = 500
        x = rng.normal(lat, 5, n)
        y = rng.normal(lon, 10, n)
        points.append(np.column_stack([x, y, np.full(n, i)]))
    data = np.vstack(points)
    X = (data[:, :2] - data[:, :2].mean(axis=0)) / data[:, :2].std(axis=0)
    return X, data[:, 2].astype(int)


def generate_sfcrime(seed, root, force_fetch=False):
    """SF Open Data crime incidents, property vs violent classification.

    Real-world urban spatial dataset. Crime incidents from the San Francisco
    Police Department Historical Incident Reports (2003-2018) are classified
    into two balanced classes based on FBI UCR Part I / Part II conventions:

      Class 0 — Property crimes (LARCENY/THEFT, VEHICLE THEFT, BURGLARY,
                VANDALISM). These concentrate in commercial and tourist areas
                where high-value targets are abundant.
      Class 1 — Violent crimes (ASSAULT, ROBBERY, WEAPON LAWS). These
                concentrate in specific residential neighborhoods where
                interpersonal conflict is more common.

    This is the second real-world spatial dataset in the benchmark and the
    only city-scale one. Earthquake data spans the entire planet; SF Crime
    fits inside a ~10 km x 10 km bounding box. The two datasets together
    demonstrate that the classifier handles real-world spatial data across
    vastly different scales without modification.

    The property-vs-violent split is the canonical binary framing in
    criminology literature (see e.g. FBI Uniform Crime Reporting Program,
    Part I violent vs property offenses). It is chosen because:
      (a) it has documented spatial structure — different neighborhoods show
          different crime profiles, so a spatial classifier should actually
          win here,
      (b) it avoids label noise from ambiguous categories (drug, fraud,
          warrants, non-criminal, suspicious occurrences), and
      (c) it matches the class count of moons, circles, spiral, and the
          other binary 2D datasets, making cross-dataset accuracy
          comparisons meaningful.

    The dataset is balanced (2500 samples per class, 5000 total) and
    standardized to zero mean, unit variance per axis — same preprocessing
    pipeline as earthquake.

      Data is loaded from data/cached/sfcrime_raw.csv if present. This
      cached file is a committed artifact shipped with the repo, so anyone
      reproducing the paper gets the exact same incidents we used, rather
      than whatever the SF Open Data API happens to return on the day they
      run the script. The live API fetch is used only when:
        (a) the cached file does not exist (first-time generation), or
        (b) --force-fetch is passed on the command line (intentional refresh).
      Cache format matches earthquake_raw.csv: raw lat, lon, label rows
      (pre-standardization).
    """
    print("\n[sfcrime] SF Open Data crime incidents (property vs violent)")

    cache_path = f"{root}/data/cached/sfcrime_raw.csv"

    if os.path.exists(cache_path) and not force_fetch:
        print(f"  Loading from cache: {cache_path}")
        try:
            df = pd.read_csv(cache_path, header=None,
                             names=['lat', 'lon', 'label'])
            raw_coords = df[['lat', 'lon']].to_numpy(dtype=float)
            raw_labels = df['label'].to_numpy(dtype=int)
            X = (raw_coords - raw_coords.mean(axis=0)) / raw_coords.std(axis=0)
            print(f"  Loaded {len(X)} incidents from cache "
                  f"(2 classes: property vs violent)")
            print(f"  Class distribution: {np.bincount(raw_labels)}")
            return X, raw_labels
        except Exception as e:
            print(f"  Cache read failed ({e}); falling back to live fetch.")

    # Fallback path: live SF Open Data API fetch via SODA. This only runs
    # on first-time generation or when --force-fetch is passed.
    print("  Cache miss (or forced fetch); downloading from SF Open Data...")
    try:
        import requests
    except ImportError:
        print("  'requests' package not found. Installing...")
        os.system(f"{sys.executable} -m pip install requests")
        import requests

    # SF Police Department Historical Incident Reports (2003-2018)
    # Dataset ID: tmnf-yvry
    # API docs: https://dev.socrata.com/foundry/data.sfgov.org/tmnf-yvry
    # No authentication required; SODA supports SoQL filtering.
    #
    # We split crime categories into two UCR-style buckets:
    PROPERTY_CATEGORIES = [
        "LARCENY/THEFT",
        "VEHICLE THEFT",
        "BURGLARY",
        "VANDALISM",
    ]
    VIOLENT_CATEGORIES = [
        "ASSAULT",
        "ROBBERY",
        "WEAPON LAWS",
    ]

    url = "https://data.sfgov.org/resource/tmnf-yvry.json"

    def fetch_category_group(categories, class_label, target_count):
        """Fetch incidents matching any category in `categories`, return
        at most target_count (lat, lon, label) tuples. Uses SoQL IN clause
        for a single-request batch fetch.
        """
        # Build SoQL WHERE: category IN ('A','B','C',...)
        # Each category is single-quoted and the list is comma-joined.
        cat_list = ",".join(f"'{c}'" for c in categories)
        where_clause = (
            f"category in({cat_list}) "
            f"AND x IS NOT NULL "
            f"AND y IS NOT NULL"
        )
        # Request more than target_count so we have room for filtering.
        # SODA caps at 50000 per request without auth; we stay well under.
        params = {
            "$where": where_clause,
            "$select": "x,y,category",
            "$limit": max(target_count * 3, 10000),
            "$order": "date DESC",
        }

        response = requests.get(url, params=params, timeout=120)
        response.raise_for_status()
        data = response.json()

        rows = []
        for incident in data:
            try:
                lon = float(incident["x"])
                lat = float(incident["y"])
            except (KeyError, ValueError, TypeError):
                continue
            # SF proper is roughly (37.70 - 37.83 N, -122.52 - -122.35 W).
            # Some records have (0, 0) sentinels or errors from outside SF.
            if not (37.70 <= lat <= 37.84 and -122.52 <= lon <= -122.35):
                continue
            rows.append((lat, lon, class_label))
        return rows

    try:
        # Fetch both class groups. We want 2500 per class after filtering
        # out-of-SF sentinels and any NaN coordinates.
        target_per_class = 2500
        property_rows = fetch_category_group(
            PROPERTY_CATEGORIES, class_label=0,
            target_count=target_per_class)
        violent_rows = fetch_category_group(
            VIOLENT_CATEGORIES, class_label=1,
            target_count=target_per_class)

        if len(property_rows) < target_per_class:
            print(f"  Warning: only got {len(property_rows)} property rows "
                  f"(wanted {target_per_class})")
        if len(violent_rows) < target_per_class:
            print(f"  Warning: only got {len(violent_rows)} violent rows "
                  f"(wanted {target_per_class})")

        # Truncate to exactly target_per_class per class for balance.
        # Use a dedicated RandomState so the sampling is deterministic per
        # seed rather than depending on the order the API returned rows.
        rng = np.random.RandomState(seed)

        def subsample(rows, n):
            if len(rows) <= n:
                return rows
            idx = rng.choice(len(rows), size=n, replace=False)
            return [rows[i] for i in idx]

        property_rows = subsample(property_rows, target_per_class)
        violent_rows = subsample(violent_rows, target_per_class)

        all_rows = property_rows + violent_rows
        if len(all_rows) == 0:
            raise ValueError("SF Open Data returned zero valid incidents")

        raw_coords = np.array([[r[0], r[1]] for r in all_rows], dtype=float)
        raw_labels = np.array([r[2] for r in all_rows], dtype=int)

        # are reproducible without needing API access.
        cache_df = pd.DataFrame({
            'lat': raw_coords[:, 0],
            'lon': raw_coords[:, 1],
            'label': raw_labels
        })
        cache_df.to_csv(cache_path, index=False, header=False)
        print(f"  Cached raw response to: {cache_path}")

        X = (raw_coords - raw_coords.mean(axis=0)) / raw_coords.std(axis=0)

        print(f"  Downloaded {len(X)} incidents "
              f"(2 classes: property vs violent)")
        print(f"  Class distribution: {np.bincount(raw_labels)}")
        return X, raw_labels

    except Exception as e:
        print(f"  Error downloading: {e}")
        print("  Generating synthetic SF-crime-like data as fallback...")
        return _synthetic_sfcrime_fallback(seed)


def _synthetic_sfcrime_fallback(seed):
    """Fallback: two clustered regions mimicking SF crime spatial structure.

    Used only when both the cache and the live API are unavailable.
    Generates two classes that are spatially separated but overlap at their
    boundaries — property crimes concentrated in one region (mimicking
    Union Square / Tenderloin commercial areas) and violent crimes
    concentrated in another (mimicking Bayview / Mission residential areas).
    """
    rng = np.random.RandomState(seed)
    n_per_class = 2500

    # Two overlapping Gaussian clusters, chosen so the boundary is curved
    # (not a straight line) and the classifier has to learn the shape.
    # Cluster 0 (property): centered at (37.787, -122.408) — Union Square area
    # Cluster 1 (violent):  centered at (37.735, -122.390) — Bayview area
    cluster_0 = rng.multivariate_normal(
        mean=[37.787, -122.408],
        cov=[[0.0003, 0.0001], [0.0001, 0.0004]],
        size=n_per_class,
    )
    cluster_1 = rng.multivariate_normal(
        mean=[37.735, -122.390],
        cov=[[0.0004, -0.0001], [-0.0001, 0.0003]],
        size=n_per_class,
    )

    raw_coords = np.vstack([cluster_0, cluster_1])
    raw_labels = np.hstack([
        np.zeros(n_per_class, dtype=int),
        np.ones(n_per_class, dtype=int),
    ])

    X = (raw_coords - raw_coords.mean(axis=0)) / raw_coords.std(axis=0)
    return X, raw_labels


# =============================================================================
# PCA-reduced datasets (secondary)
# =============================================================================

def generate_wine(seed):
    """UCI Wine dataset (178 samples, 3 classes, 13 features -> PCA to 2D).

    Demonstrates that the classifier generalizes beyond natively 2D data.
    Wine's first two principal components capture substantial variance,
    making PCA reduction viable without excessive information loss.
    """
    print("\n[wine] UCI Wine dataset (PCA: 13D -> 2D)")
    data = datasets.load_wine()
    return reduce_to_2d(data.data, data.target, seed)


def generate_cancer(seed):
    """UCI Breast Cancer dataset (569 samples, 2 classes, 30D -> PCA to 2D).

    Medical domain dataset. The two classes (malignant/benign) separate
    reasonably well in the first two principal components, making this a
    practical test of the classifier on dimensionality-reduced clinical data.
    """
    print("\n[cancer] UCI Breast Cancer dataset (PCA: 30D -> 2D)")
    data = datasets.load_breast_cancer()
    return reduce_to_2d(data.data, data.target, seed)


def generate_bloodmnist(seed, root, force_fetch=False):
    """BloodMNIST cell centroid extraction (8 classes, ~17K samples).

    Large-scale medical imaging dataset. Cell centroids are extracted from
    blood cell microscopy images, providing a high-volume test of the
    classifier's scalability. Primarily included to demonstrate that the
    O(1) inference property holds at scale (17K training points).

      Pre-extracted centroids are loaded from
        data/cached/bloodmnist_train_centroids.csv
        data/cached/bloodmnist_test_centroids.csv
      if present. These are committed artifacts shipped with the repo, so
      reproducers don't need to install medmnist, download ~40 MB of images,
      or run scikit-image's regionprops extraction. The live download +
      extraction pipeline is used only when:
        (a) the cached files do not exist (first-time generation), or
        (b) --force-fetch is passed on the command line (intentional refresh).
      When fresh extraction happens, the results are saved to the cache
      directory for future runs.

    Note: cached files store the extracted centroids BEFORE the mean/std
    normalization, so normalization is applied at load time for consistency.
    """
    print("\n[bloodmnist] BloodMNIST cell centroids")

    train_cache = f"{root}/data/cached/bloodmnist_train_centroids.csv"
    test_cache = f"{root}/data/cached/bloodmnist_test_centroids.csv"

    if (os.path.exists(train_cache) and os.path.exists(test_cache)
            and not force_fetch):
        print(f"  Loading from cache: {train_cache}")
        print(f"                      {test_cache}")
        try:
            train_df = pd.read_csv(train_cache, header=None,
                                   names=['x', 'y', 'label'])
            test_df = pd.read_csv(test_cache, header=None,
                                  names=['x', 'y', 'label'])
            X_train_raw = train_df[['x', 'y']].to_numpy(dtype=float)
            y_train = train_df['label'].to_numpy(dtype=int)
            X_test_raw = test_df[['x', 'y']].to_numpy(dtype=float)
            y_test = test_df['label'].to_numpy(dtype=int)

            mean = X_train_raw.mean(axis=0)
            std = X_train_raw.std(axis=0)
            std[std < 1e-10] = 1.0
            X_train = (X_train_raw - mean) / std
            X_test = (X_test_raw - mean) / std

            print(f"  Loaded {len(X_train)} train + {len(X_test)} test "
                  f"centroids from cache")
            return X_train, y_train, X_test, y_test
        except Exception as e:
            print(f"  Cache read failed ({e}); falling back to extraction.")

    # Fallback path: download MedMNIST and extract centroids. This only runs
    # on first-time generation or when --force-fetch is passed.
    print("  Cache miss (or forced fetch); downloading and extracting...")
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
        medmnist_dir = f"{root}/data/medmnist"
        os.makedirs(medmnist_dir, exist_ok=True)
        
        train_dataset = BloodMNIST(split='train', download=True,
                                   root=medmnist_dir)
        test_dataset = BloodMNIST(split='test', download=True,
                                  root=medmnist_dir)

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

        X_train_raw, y_train = extract_centroids(train_dataset)
        X_test_raw, y_test = extract_centroids(test_dataset)

        # so reproducers get the same raw values regardless of which subset
        # was extracted.
        train_cache_df = pd.DataFrame({
            'x': X_train_raw[:, 0],
            'y': X_train_raw[:, 1],
            'label': y_train
        })
        train_cache_df.to_csv(train_cache, index=False, header=False)
        test_cache_df = pd.DataFrame({
            'x': X_test_raw[:, 0],
            'y': X_test_raw[:, 1],
            'label': y_test
        })
        test_cache_df.to_csv(test_cache, index=False, header=False)
        print(f"  Cached extracted centroids to: {train_cache}")
        print(f"                                  {test_cache}")

        mean = X_train_raw.mean(axis=0)
        std = X_train_raw.std(axis=0)
        std[std < 1e-10] = 1.0
        X_train = (X_train_raw - mean) / std
        X_test = (X_test_raw - mean) / std

        print(f"  Extracted {len(X_train)} train + {len(X_test)} test centroids")
        return X_train, y_train, X_test, y_test

    except Exception as e:
        print(f"  Error processing BloodMNIST: {e}")
        print("  Generating synthetic 8-class clustered data as fallback...")
        return _synthetic_bloodmnist_fallback(seed)


def _synthetic_bloodmnist_fallback(seed):
    """Fallback: 8-class clustered data mimicking cell distributions.

    Used only when both the cache and the medmnist extraction pipeline
    are unavailable.
    """
    rng = np.random.RandomState(seed)
    n_per_class = 2125
    n_classes = 8
    points = []
    for i in range(n_classes):
        angle = 2 * np.pi * i / n_classes
        cx, cy = 0.3 * np.cos(angle), 0.3 * np.sin(angle)
        x = rng.normal(cx, 0.1, n_per_class)
        y = rng.normal(cy, 0.1, n_per_class)
        points.append(np.column_stack([x, y, np.full(n_per_class, i)]))
    data = np.vstack(points)
    rng.shuffle(data)
    X, y = data[:, :2], data[:, 2].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)
    return X_train, y_train, X_test, y_test


# =============================================================================
# Dataset registry
# =============================================================================
#
# Each entry in GENERATORS maps a dataset name to a generator function that
# accepts a `seed` argument and returns (X, y). earthquake, sfcrime, and
# bloodmnist require additional arguments (root, force_fetch) and are handled
# specially in main() rather than through this registry.

GENERATORS = {
    'moons': generate_moons,
    'circles': generate_circles_dataset,
    'spiral': generate_spiral,
    'gaussian_quantiles': generate_gaussian_quantiles,
    'cassini': generate_cassini,
    'checkerboard': generate_checkerboard,
    'blobs': generate_blobs,
    'wine': generate_wine,
    'cancer': generate_cancer,
    # earthquake, sfcrime, bloodmnist handled specially (need root + force_fetch)
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
    # behavior (42), so running without --seed reproduces the existing data.
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help=f'Random seed (default: {DEFAULT_SEED}). Use for '
                             f'multi-seed experiments with seeds like '
                             f'42, 123, 456, 789, 1000.')
    # data. The default is to use cached artifacts for reproducibility.
    parser.add_argument('--force-fetch', action='store_true',
                        help='Force re-download of earthquake (USGS API) and '
                             'bloodmnist (medmnist) data, bypassing any '
                             'cached files in data/cached/. Use this only if '
                             'you explicitly want to refresh the cached '
                             'artifacts; the default cached data is the '
                             'version used in the paper.')
    args = parser.parse_args()

    seed = args.seed
    root = args.out_dir
    force_fetch = args.force_fetch

    create_output_dirs(root)

    # before any generator runs. Generators that rely on np.random.* (spiral,
    # cassini, checkerboard) will produce byte-identical output to the pre-fix
    # code when seed=42, because the global state starts at the same point.
    np.random.seed(seed)

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
    print(f"Datasets:    {', '.join(requested)}")
    print(f"Output:      {root}/data/")
    print(f"Seed:        {seed}")
    print(f"Force fetch: {force_fetch}")
    print("=" * 70)

    for name in requested:
        if name == 'bloodmnist':
            X_train, y_train, X_test, y_test = generate_bloodmnist(
                seed, root, force_fetch=force_fetch)

            save_csv(X_train, y_train,
                     f"{root}/data/train/bloodmnist_train.csv")
            save_csv(X_test, None,
                     f"{root}/data/test/bloodmnist_test_X.csv")
            save_csv(X_test, y_test,
                     f"{root}/data/test/bloodmnist_test_y.csv")

            # Dynamic benchmark split with seeded RNG.
            base_size = int(len(X_train) * 0.6)
            indices = np.arange(len(X_train))
            rng = np.random.RandomState(seed)
            rng.shuffle(indices)
            save_csv(X_train[indices[:base_size]],
                     y_train[indices[:base_size]],
                     f"{root}/data/train/bloodmnist_dynamic_base.csv")
            save_csv(X_train[indices[base_size:]],
                     y_train[indices[base_size:]],
                     f"{root}/data/train/bloodmnist_dynamic_stream.csv")

        elif name == 'earthquake':
            X, y = generate_earthquake(seed, root, force_fetch=force_fetch)
            save_dataset(X, y, name, root, seed)

        elif name == 'sfcrime':
            X, y = generate_sfcrime(seed, root, force_fetch=force_fetch)
            save_dataset(X, y, name, root, seed)

        else:
            generator = GENERATORS[name]
            X, y = generator(seed)
            save_dataset(X, y, name, root, seed)

    print("\n" + "=" * 70)
    print("DATASET GENERATION COMPLETE")
    print("=" * 70)

    n_spatial = sum(1 for d in requested if d in [
        'moons', 'circles', 'spiral', 'gaussian_quantiles', 'cassini',
        'checkerboard', 'blobs', 'earthquake', 'sfcrime'])
    n_pca = sum(1 for d in requested if d in ['wine', 'cancer', 'bloodmnist'])

    print(f"\nGenerated {len(requested)} dataset(s) with seed={seed}:")
    print(f"  {n_spatial} natively 2D spatial (primary benchmarks)")
    print(f"  {n_pca} PCA-reduced / extracted (generality benchmarks)")
    print(f"\nOutput directory: {root}/data/")


if __name__ == "__main__":
    main()