# Delaunay Triangulation Classifier

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org/)
[![CGAL](https://img.shields.io/badge/CGAL-5.x-orange.svg)](https://www.cgal.org/)

A novel spatial classification algorithm using Delaunay Triangulation with **O(1) expected inference** and **O(1) dynamic updates**. Designed for real-time geospatial applications, streaming data, and embedded systems.

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
| **O(1) Expected Inference** | Constant-time point classification using SRR spatial indexing |
| **O(1) Dynamic Updates** | Insert/delete points without model rebuild |
| **No Training Phase** | Instant deployment — just load data |
| **Interpretable** | Decision regions are visible Voronoi-like cells |
| **Geometric Foundation** | Based on Delaunay Triangulation theory |

---

## Performance Summary

### Accuracy (10-Fold Cross-Validation)

Results from `scripts/benchmark_cv.py` with 10-fold stratified CV:

| Dataset | Delaunay | KNN | SVM | Decision Tree | Random Forest |
|---------|----------|-----|-----|---------------|---------------|
| Wine | 93.81 ± 3.63% | 95.50 ± 2.70% | 97.76 ± 2.10% | 92.12 ± 3.81% | 98.31 ± 2.22% |
| Moons | **100.00 ± 0.00%** | 99.80 ± 0.42% | 99.80 ± 0.42% | 99.50 ± 0.97% | 99.50 ± 0.53% |
| Spiral | 98.10 ± 1.45% | 98.30 ± 0.95% | 66.90 ± 3.51% | 84.40 ± 3.81% | 97.30 ± 1.34% |
| Earthquake | 94.18 ± 0.84% | 94.76 ± 0.51% | 93.02 ± 0.67% | 93.68 ± 0.93% | 94.10 ± 0.81% |

### Inference Speed (C++ Benchmark)

Results from `./build/benchmark` (values from `results/cpp_benchmark_*.csv`):

| Dataset | Delaunay (µs) | FLANN KNN (µs) | Speedup |
|---------|---------------|----------------|---------|
| Wine | 0.108 | 2.983 | **27.7×** |
| Moons | 0.140 | 3.747 | **26.7×** |
| Spiral | 0.105 | 3.531 | **33.6×** |
| Earthquake | 0.217 | 4.840 | **22.3×** |

### Dynamic Update Speed

Results from `./build/benchmark` dynamic mode (from `results/cpp_benchmark_earthquake.csv`):

| Operation | Delaunay | Decision Tree (Rebuild) | Speedup |
|-----------|----------|-------------------------|---------|
| Insert (1 point) | ~1,154 ns | ~7,954,850 ns | **~6,893×** |

---

## Algorithm Overview

### Phase 1: Outlier Removal (k-NN Density Filter)
Removes noisy points using k-NN same-class density filtering. Points with fewer than half neighbors of same class are marked as outliers.

### Phase 2: Delaunay Triangulation Construction
Builds a mesh of non-overlapping triangles connecting all training points using CGAL. **Complexity: O(n log n)**

### Phase 3: SRR Grid Construction
Creates a √n × √n spatial index (Square Root Rule) for O(1) expected point location instead of O(√n) walking.

### Phase 4: Classification (Inference)
For a query point:
1. Hash (x,y) to SRR bucket → get triangle hint **O(1)**
2. Walk from hint to containing triangle **O(1) expected**
3. Majority vote among 3 triangle vertices **O(1)**

**Total inference complexity: O(1) expected**

---

## Project Structure

```
Delaunay-Triangulation-Classification/
├── src/                              # C++ source files
│   ├── DelaunayClassifier.cpp        # Core classifier with SRR grid
│   ├── main.cpp                      # CLI application (static/dynamic modes)
│   ├── benchmark.cpp                 # C++ benchmark (FLANN KNN, LibSVM, DT)
│   └── ablation_bench.cpp            # C++ ablation study
│
├── include/
│   └── DelaunayClassifier.h          # Header with public API
│
├── scripts/                          # Python scripts
│   ├── data_generator.py             # Generates: wine, cancer, iris, digits, moons, blobs
│   ├── generate_spatial_datasets.py  # Generates: spiral, circles, checkerboard, earthquake, bloodmnist
│   ├── benchmark.py                  # Python static/dynamic benchmark wrapper
│   ├── benchmark_cv.py               # 10-fold cross-validation with significance tests
│   ├── ablation_study.py             # Python ablation study wrapper
│   ├── scalability_test.py           # Scalability analysis (O(n log n) training, O(1) inference)
│   ├── generate_figures.py           # Publication figure generator (69 figures)
│   └── visualizer.py                 # Visualization utilities
│
├── data/
│   ├── train/                        # Training datasets (CSV: x, y, label)
│   │   ├── {dataset}_train.csv       # Main training data
│   │   ├── {dataset}_dynamic_base.csv  # Base data for dynamic benchmarks
│   │   └── {dataset}_dynamic_stream.csv # Stream data for dynamic benchmarks
│   └── test/
│       ├── {dataset}_test_X.csv      # Test features only
│       └── {dataset}_test_y.csv      # Test features + labels
│
├── results/                          # Benchmark outputs
│   ├── cv_summary.csv                # 10-fold CV accuracy ± std
│   ├── significance_tests.csv        # t-test and Wilcoxon p-values
│   ├── cpp_benchmark_{dataset}.csv   # C++ timing per dataset
│   └── logs/                         # Dynamic benchmark logs
│
├── figures/                          # Generated publication figures
│   └── {dataset}/                    # 6 figures per dataset
│
└── CMakeLists.txt                    # CMake build configuration
```

---

## Installation

### Prerequisites

| Dependency | Version | Purpose |
|------------|---------|---------|
| CMake | ≥ 3.10 | Build system |
| CGAL | ≥ 5.0 | Delaunay triangulation |
| FLANN | ≥ 1.9 | KNN baseline (C++) |
| LibSVM | ≥ 3.25 | SVM baseline (C++) |
| LZ4 | Latest | FLANN compression |
| Python | ≥ 3.8 | Scripts |

### macOS (Homebrew)

```bash
# Install C++ dependencies
brew install cmake cgal flann libsvm lz4

# Clone repository
git clone https://github.com/yourusername/Delaunay-Triangulation-Classification.git
cd Delaunay-Triangulation-Classification

# Build C++
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
cd ..

# Setup Python environment
python3 -m venv venv
source venv/bin/activate
pip install numpy scipy scikit-learn pandas matplotlib
```

### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install cmake libcgal-dev libflann-dev libsvm-dev liblz4-dev
```

---

## Quick Start

### Step 1: Generate Standard Datasets (6 datasets)

```bash
source venv/bin/activate

# Generate: wine, cancer, iris, digits, moons, blobs
for ds in wine cancer iris digits moons blobs; do
  python scripts/data_generator.py --type $ds
done
```

### Step 2: Generate Spatial Datasets (5 additional datasets)

```bash
# Generate: spiral, circles, checkerboard, earthquake, bloodmnist
python scripts/generate_spatial_datasets.py
```

**Total: 11 datasets** in `data/train/` and `data/test/`.

### Step 3: Run Demo

```bash
./build/main static data/train/wine_train.csv data/test/wine_test_X.csv results/
```

---

## Datasets

| Dataset | Samples | Classes | Type | Generator Script |
|---------|---------|---------|------|------------------|
| Wine | 178 | 3 | UCI ML | `data_generator.py --type wine` |
| Cancer | 569 | 2 | UCI ML | `data_generator.py --type cancer` |
| Iris | 150 | 3 | UCI ML | `data_generator.py --type iris` |
| Digits | 1,797 | 10 | UCI ML | `data_generator.py --type digits` |
| Moons | 1,000 | 2 | Synthetic | `data_generator.py --type moons` |
| Blobs | 1,500 | 3 | Synthetic | `data_generator.py --type blobs` |
| Spiral | 1,000 | 2 | Synthetic | `generate_spatial_datasets.py` |
| Circles | 1,000 | 2 | Synthetic | `generate_spatial_datasets.py` |
| Checkerboard | 1,000 | 4 | Synthetic | `generate_spatial_datasets.py` |
| Earthquake | ~5,000 | 4 | USGS API | `generate_spatial_datasets.py` |
| BloodMNIST | ~17,000 | 8 | MedMNIST | `generate_spatial_datasets.py` |

---

## Running Benchmarks

### 1. Cross-Validation Benchmark (Python)

10-fold stratified CV with statistical significance tests:

```bash
python scripts/benchmark_cv.py --datasets wine,moons,spiral,earthquake --folds 10
```

**Outputs:**
- `results/cv_summary.csv` — Accuracy mean ± std for all algorithms
- `results/significance_tests.csv` — Paired t-test and Wilcoxon p-values

### 2. C++ Static Benchmark

Pure inference timing with FLANN KNN, LibSVM, Decision Tree:

```bash
./build/benchmark data/train/wine_train.csv data/test/wine_test_y.csv wine
```

**Output:** `results/cpp_benchmark_wine.csv`

### 3. C++ Dynamic Benchmark

Measures insert/delete time (included in static benchmark):

```bash
./build/benchmark data/train/earthquake_train.csv data/test/earthquake_test_y.csv earthquake
```

### 4. C++ Ablation Study

Quantify SRR grid and decision boundary contributions:

```bash
./build/ablation_bench data/train/wine_train.csv data/test/wine_test_y.csv
```

### 5. Python Benchmark (Static/Dynamic)

```bash
python scripts/benchmark.py --dataset wine --mode static
python scripts/benchmark.py --dataset wine --mode dynamic
```

### 6. Generate Publication Figures

```bash
python scripts/generate_figures.py
```

**Output:** 69 figures in `figures/` (6 per dataset + 3 summary).

---

## Reproducing Results

```bash
# 1. Generate all datasets
for ds in wine cancer iris digits moons blobs; do
  python scripts/data_generator.py --type $ds
done
python scripts/generate_spatial_datasets.py

# 2. Run 10-fold CV
python scripts/benchmark_cv.py \
  --datasets wine,cancer,iris,digits,moons,blobs,spiral,circles,checkerboard,earthquake \
  --folds 10

# 3. Run C++ benchmarks
for ds in wine cancer iris moons blobs spiral circles checkerboard earthquake; do
  ./build/benchmark data/train/${ds}_train.csv data/test/${ds}_test_y.csv $ds
done

# 4. Run ablation study
for ds in wine moons spiral earthquake; do
  ./build/ablation_bench data/train/${ds}_train.csv data/test/${ds}_test_y.csv
done

# 5. Generate figures
python scripts/generate_figures.py
```

**Random Seed:** All experiments use `seed=42` for reproducibility.

---

## API Reference

### C++ API

```cpp
#include "DelaunayClassifier.h"

// Initialize and train
DelaunayClassifier clf;
clf.train("data/train/wine_train.csv");

// Static inference (O(1) expected)
int label = clf.classify_single(0.5, 1.2);

// Dynamic insert (O(1) amortized)
clf.insert_point(0.3, 0.8, 2);  // (x, y, label)

// Dynamic delete (O(1) amortized)
clf.remove_point(0.3, 0.8);
```

### Python Benchmark API

```python
# Run static benchmark
python scripts/benchmark.py --dataset wine --mode static

# Run 10-fold CV
python scripts/benchmark_cv.py --datasets wine,moons --folds 10
```

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

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [CGAL](https://www.cgal.org/) — Computational Geometry Algorithms Library
- [FLANN](https://github.com/flann-lib/flann) — Fast Library for Approximate Nearest Neighbors
- [USGS Earthquake Hazards Program](https://earthquake.usgs.gov/) — Real earthquake data