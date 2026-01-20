# Delaunay Triangulation Classifier

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org/)
[![CGAL](https://img.shields.io/badge/CGAL-5.x-orange.svg)](https://www.cgal.org/)

A novel spatial classification algorithm using Delaunay Triangulation with **O(1) inference** and **O(1) dynamic updates**. Designed for real-time geospatial applications, streaming data, and embedded systems.

---

## Table of Contents

1. [Key Features](#key-features)
2. [Performance Summary](#performance-summary)
3. [Algorithm Overview](#algorithm-overview)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [Quick Start](#quick-start)
7. [Datasets](#datasets)
8. [Running Benchmarks](#running-benchmarks)
9. [Reproducing Results](#reproducing-results)
10. [API Reference](#api-reference)
11. [Citation](#citation)

---

## Key Features

| Feature | Description |
|---------|-------------|
| **O(1) Inference** | Constant-time point classification using SRR spatial indexing |
| **O(1) Dynamic Updates** | Insert/delete points without model rebuild |
| **No Training Phase** | Instant deployment — just load data |
| **Interpretable** | Decision regions are visible Voronoi-like cells |
| **Geometric Foundation** | Based on Delaunay Triangulation theory |

---

## Performance Summary

### Accuracy (10-Fold Cross-Validation)

| Dataset | Delaunay | KNN | SVM | Decision Tree | Random Forest |
|---------|----------|-----|-----|---------------|---------------|
| Wine | 93.81% | 95.50% | 97.76% | 92.12% | 98.31% |
| Cancer | 92.10% | 93.32% | 94.73% | 92.44% | 94.73% |
| Iris | 96.67% | 96.67% | 95.33% | 96.00% | 95.33% |
| **Moons** | **100.00%** | 99.80% | 99.80% | 99.50% | 99.50% |
| **Circles** | **100.00%** | 100.00% | 100.00% | 99.40% | 99.60% |
| Spiral | 98.10% | 98.30% | 66.90% | 84.40% | 97.30% |
| Earthquake | 94.18% | 94.76% | 93.02% | 93.68% | 94.10% |

### Inference Speed (C++ Benchmark)

| Dataset | Delaunay (µs) | KNN (µs) | Speedup |
|---------|---------------|----------|---------|
| Wine | 0.10 | 3.62 | **36×** |
| Moons | 0.10 | 3.62 | **36×** |
| Spiral | 0.11 | 3.53 | **34×** |
| Circles | 0.12 | 3.45 | **29×** |
| Earthquake | 0.22 | 4.84 | **22×** |

### Dynamic Update Speed

| Operation | Delaunay | Decision Tree | Speedup |
|-----------|----------|---------------|---------|
| Insert (1 point) | 1.1 µs | 7,954 µs | **6,893×** |
| Delete (1 point) | 4.6 µs | 7,954 µs | **1,728×** |

---

## Algorithm Overview

### Phase 1: Outlier Removal
Removes noisy points using k-NN same-class density filtering.

### Phase 2: Delaunay Triangulation Construction
Builds a mesh of non-overlapping triangles connecting all training points. Complexity: O(n log n).

### Phase 3: SRR Grid Construction
Creates a √n × √n spatial index for O(1) point location instead of O(√n) walking.

### Phase 4: Classification
For a query point:
1. Hash to SRR bucket → get triangle hint (O(1))
2. Walk to containing triangle (O(1) expected)
3. Majority vote among 3 vertices (O(1))

**Total inference complexity: O(1)**

---

## Project Structure

```
Delaunay-Triangulation-Classification/
├── src/                          # C++ source files
│   ├── DelaunayClassifier.cpp    # Core classifier implementation
│   ├── main.cpp                  # Demo application
│   ├── benchmark.cpp             # Static/dynamic benchmarks
│   └── ablation_bench.cpp        # Ablation study
├── include/
│   └── DelaunayClassifier.h      # Header file
├── scripts/                      # Python scripts
│   ├── benchmark_cv.py           # 10-fold cross-validation
│   ├── generate_figures.py       # Publication figure generator
│   ├── generate_spatial_datasets.py  # Dataset generators
│   ├── scalability_test.py       # Scalability analysis
│   └── visualizer.py             # Visualization utilities
├── data/
│   ├── train/                    # Training datasets (CSV)
│   └── test/                     # Test datasets (CSV)
├── results/                      # Benchmark outputs
│   ├── cv_summary.csv            # Cross-validation results
│   ├── significance_tests.csv   # Statistical significance
│   └── cpp_benchmark_*.csv       # C++ timing results
├── figures/                      # Generated figures
└── CMakeLists.txt                # CMake build configuration
```

---

## Installation

### Prerequisites

| Dependency | Version | Purpose |
|------------|---------|---------|
| CMake | ≥ 3.10 | Build system |
| CGAL | ≥ 5.0 | Delaunay triangulation |
| FLANN | ≥ 1.9 | KNN baseline |
| LibSVM | ≥ 3.25 | SVM baseline |
| Python | ≥ 3.8 | Scripts |
| NumPy, SciPy, scikit-learn | Latest | Python benchmarks |

### macOS (Homebrew)

```bash
# Install C++ dependencies
brew install cmake cgal flann libsvm

# Clone repository
git clone https://github.com/yourusername/Delaunay-Triangulation-Classification.git
cd Delaunay-Triangulation-Classification

# Build C++
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
cd ..

# Setup Python
python3 -m venv venv
source venv/bin/activate
pip install numpy scipy scikit-learn pandas matplotlib
```

### Ubuntu/Debian

```bash
# Install C++ dependencies
sudo apt-get update
sudo apt-get install cmake libcgal-dev libflann-dev libsvm-dev

# Build (same as macOS after clone)
```

---

## Quick Start

### Generate Datasets

```bash
source venv/bin/activate
python scripts/generate_spatial_datasets.py
```

This generates 11 datasets in `data/train/` and `data/test/`.

### Run Demo

```bash
./build/app data/train/wine_train.csv data/test/wine_test.csv
```

### Run Benchmarks

```bash
# Python 10-fold CV (all datasets)
python scripts/benchmark_cv.py --datasets wine,cancer,iris,moons,spiral,earthquake --folds 10

# C++ benchmark (single dataset)
./build/benchmark data/train/wine_train.csv data/test/wine_test_y.csv wine

# C++ ablation study
./build/ablation_bench data/train/wine_train.csv data/test/wine_test_y.csv
```

---

## Datasets

| Dataset | Samples | Classes | Type | Source |
|---------|---------|---------|------|--------|
| Wine | 178 | 3 | UCI ML Repository | sklearn |
| Cancer | 569 | 2 | UCI ML Repository | sklearn |
| Iris | 150 | 3 | UCI ML Repository | sklearn |
| Digits | 1,797 | 10 | UCI ML Repository | sklearn |
| Moons | 1,000 | 2 | Synthetic | sklearn |
| Blobs | 1,500 | 3 | Synthetic | sklearn |
| Spiral | 1,000 | 2 | Synthetic | Custom |
| Circles | 1,000 | 2 | Synthetic | Custom |
| Checkerboard | 1,000 | 2 | Synthetic | Custom |
| Earthquake | 5,000 | 4 | Real-world | USGS API |

---

## Running Benchmarks

### 1. Cross-Validation Benchmark (Python)

**Purpose:** Accuracy comparison with statistical significance tests.

```bash
python scripts/benchmark_cv.py --datasets wine,moons,spiral,earthquake --folds 10
```

**Output:**
- `results/cv_summary.csv` — Mean ± std accuracy for all algorithms
- `results/significance_tests.csv` — Paired t-test and Wilcoxon p-values

### 2. Static Benchmark (C++)

**Purpose:** Pure inference timing comparison.

```bash
./build/benchmark data/train/wine_train.csv data/test/wine_test_y.csv wine
```

**Output:**
- `results/cpp_benchmark_wine.csv` — Accuracy and timing for FLANN KNN, LibSVM, Decision Tree, Delaunay

### 3. Dynamic Benchmark (C++)

Included in the static benchmark. Measures insert/delete time vs. Decision Tree rebuild.

### 4. Ablation Study (C++)

**Purpose:** Quantify contribution of SRR grid and decision boundary interpolation.

```bash
./build/ablation_bench data/train/wine_train.csv data/test/wine_test_y.csv
```

**Output:** Speedup comparison: Full System vs. No SRR vs. Nearest Vertex Only.

### 5. Generate Figures

```bash
python scripts/generate_figures.py
```

**Output:** 69 publication-ready figures in `figures/` (300 DPI PNG).

---

## Reproducing Results

To reproduce all benchmark results from the paper:

```bash
# Step 1: Generate datasets
python scripts/generate_spatial_datasets.py

# Step 2: Run Python 10-fold CV on all datasets
python scripts/benchmark_cv.py \
  --datasets wine,cancer,iris,digits,moons,blobs,spiral,circles,checkerboard,earthquake \
  --folds 10

# Step 3: Run C++ benchmarks on each dataset
for ds in wine cancer iris moons blobs spiral circles checkerboard earthquake; do
  ./build/benchmark data/train/${ds}_train.csv data/test/${ds}_test_y.csv $ds
done

# Step 4: Run ablation study
for ds in wine moons spiral earthquake; do
  ./build/ablation_bench data/train/${ds}_train.csv data/test/${ds}_test_y.csv
done

# Step 5: Generate publication figures
python scripts/generate_figures.py
```

**Random Seed:** All experiments use `seed=42` for reproducibility.

---

## API Reference

### C++ API

```cpp
#include "DelaunayClassifier.h"

// Initialize classifier
DelaunayClassifier clf;

// Train from CSV file
clf.train("data/train/wine_train.csv");

// Classify single point (O(1))
int label = clf.classify_single(0.5, 1.2);

// Dynamic insert (O(1))
clf.insert_point(0.3, 0.8, 2);  // (x, y, label)

// Dynamic delete (O(1))
clf.remove_point(0.3, 0.8);
```

### Python API (via subprocess)

```python
import subprocess

# Run classifier
result = subprocess.run([
    './build/app',
    'data/train/wine_train.csv',
    'data/test/wine_test.csv'
], capture_output=True, text=True)

print(result.stdout)  # Accuracy, timing
```

---

## Complexity Analysis

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Training | O(n log n) | Delaunay construction |
| Inference | **O(1)** | With SRR grid |
| Insert | **O(1)** amortized | Local re-triangulation |
| Delete | **O(1)** amortized | Local re-triangulation |
| Space | O(n) | Stores n points + triangulation |

---

## Statistical Significance

Key results where Delaunay significantly outperforms baselines (p < 0.05):

| Comparison | Dataset | p-value | Result |
|------------|---------|---------|--------|
| Delaunay vs SVM | Spiral | **< 0.001** | Delaunay wins |
| Delaunay vs Decision Tree | Spiral | **< 0.001** | Delaunay wins |
| Delaunay vs SVM | Earthquake | **< 0.001** | Delaunay wins |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{delaunay_classifier_2026,
  title={Real-Time Spatial Classification via Delaunay Triangulation with O(1) Point Location},
  author={[Your Name]},
  journal={[Target Journal]},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [CGAL](https://www.cgal.org/) — Computational Geometry Algorithms Library
- [FLANN](https://github.com/flann-lib/flann) — Fast Library for Approximate Nearest Neighbors
- [USGS Earthquake Hazards Program](https://earthquake.usgs.gov/) — Earthquake data