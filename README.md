# Delaunay Triangulation Classifier

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)
[![Python](https://img.shields.io/badge/Python-3.12-green.svg)](https://python.org/)
[![CGAL](https://img.shields.io/badge/CGAL-5.x-orange.svg)](https://www.cgal.org/)

A novel spatial classification algorithm using Delaunay Triangulation with **O(1) expected inference** and **O(1) amortized dynamic updates**. Designed for real-time geospatial applications, streaming data classification, and embedded systems.

---

## Table of Contents

1. [Key Features](#key-features)
2. [Performance Highlights](#performance-highlights)
3. [System Requirements](#system-requirements)
4. [Installation](#installation)
5. [Project Structure](#project-structure)
6. [Quick Start Guide](#quick-start-guide)
7. [Dataset Generation](#dataset-generation)
8. [Benchmark Procedures](#benchmark-procedures)
9. [Unit Testing](#unit-testing)
10. [Reproducing Results](#reproducing-results)
11. [Benchmark Results](#benchmark-results)
12. [API Reference](#api-reference)
13. [Citation](#citation)

---

## Key Features

| Feature | Description |
|---------|-------------|
| **O(1) Expected Inference** | Constant-time classification using 2D Buckets spatial indexing |
| **O(1) Amortized Updates** | Insert/delete points without model rebuild |
| **2D Buckets Data Structure** | Full linked-list implementation with 6 structures (LL_V, LL_E, LL_GE, LL_Poly, LL_Label, LL_PolyID) |
| **Exact Voronoi Clipping** | CGAL Voronoi_diagram_2 for precise decision boundaries |
| **No Training Phase** | Instant deployment — construct mesh directly from data |
| **Interpretable** | Decision regions visible as exact Voronoi cells |
| **Geometric Foundation** | Based on Delaunay Triangulation computational geometry |
| **Outside-Hull Classification** | Handles queries outside convex hull using extended decision boundaries |

---

## Performance Highlights

Results from experiments on **MacBook Pro M3, macOS 26, 16GB RAM** with **Unified 2D Buckets Classification**.

| Metric | Result | Baseline Comparison |
|--------|--------|---------------------|
| **Inference Speed** | 0.002–0.007 µs/point | **482×–3,989× faster** than FLANN KNN (C++) |
| **Accuracy (Spatial)** | 93–100% | Best or tied-best on 7/11 datasets |
| **Dynamic Insert** | 37K–989K ns | **2×–141× faster** than Decision Tree rebuild |
| **Scalability** | O(1) inference verified | Flat at ~0.01 µs from 100 to 100K points |

---

## System Requirements

### Benchmark Platform

| Component | Specification |
|-----------|---------------|
| **Hardware** | Apple MacBook Pro with M3 chip |
| **RAM** | 16 GB unified memory |
| **OS** | macOS 26 |
| **Compiler** | Apple Clang with -O3 optimization |

### Software Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| CMake | ≥ 3.10 | Build system |
| CGAL | ≥ 5.0 | Delaunay triangulation |
| FLANN | ≥ 1.9 | KNN baseline (C++) |
| LibSVM | ≥ 3.25 | SVM baseline (C++) |
| LZ4 | Latest | FLANN compression dependency |
| Python | ≥ 3.8 | Benchmark scripts |
| NumPy | Latest | Numerical operations |
| SciPy | Latest | Statistical tests |
| scikit-learn | Latest | Cross-validation baselines |
| pandas | Latest | Data manipulation |
| matplotlib | Latest | Figure generation |

---

## Installation

### macOS (Homebrew)

```bash
# Step 1: Install C++ dependencies
brew install cmake cgal flann libsvm lz4

# Step 2: Clone repository
git clone https://github.com/yourusername/Delaunay-Triangulation-Classification.git
cd Delaunay-Triangulation-Classification

# Step 3: Build C++ executables
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
cd ..

# Step 4: Setup Python environment
python3 -m venv venv
source venv/bin/activate
pip install numpy scipy scikit-learn pandas matplotlib requests

# Step 5: Verify installation
./build/main static data/train/wine_train.csv data/test/wine_test_y.csv results/
```

### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install cmake libcgal-dev libflann-dev libsvm-dev liblz4-dev
# Then follow steps 2-5 above
```

### Windows (vcpkg)

```powershell
# Step 1: Install vcpkg (if not already installed)
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg integrate install

# Step 2: Install C++ dependencies
.\vcpkg install cgal:x64-windows flann:x64-windows libsvm:x64-windows lz4:x64-windows

# Step 3: Clone and build
cd ..
git clone https://github.com/yourusername/Delaunay-Triangulation-Classification.git
cd Delaunay-Triangulation-Classification
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=[vcpkg-root]/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
cd ..

# Step 4: Setup Python environment
python -m venv venv
.\venv\Scripts\activate
pip install numpy scipy scikit-learn pandas matplotlib requests

# Step 5: Verify installation
.\build\Release\main.exe static data\train\wine_train.csv data\test\wine_test_y.csv results\
```

> **Note:** Replace `[vcpkg-root]` with your actual vcpkg installation path (e.g., `C:\vcpkg`).

---

## Project Structure

```
Delaunay-Triangulation-Classification/
│
├── src/                                # C++ source files
│   ├── DelaunayClassifier.cpp          # Core classifier (outlier removal, DT, 2D Buckets)
│   ├── main.cpp                        # CLI application (static/dynamic modes)
│   ├── benchmark.cpp                   # C++ benchmark (FLANN, LibSVM, DT comparison)
│   └── ablation_bench.cpp              # Ablation study (component contribution)
│
├── include/
│   └── DelaunayClassifier.h            # Public API header
│
├── scripts/                            # Python scripts
│   ├── generate_datasets.py            # Unified dataset generator (all 11 datasets)
│   ├── benchmark_cv.py                 # 10-fold CV with statistical significance tests
│   ├── ablation_study.py               # Python ablation study wrapper
│   ├── scalability_test.py             # Scalability analysis O(n log n) training
│   ├── generate_publication_figures.py # Publication figures (per dataset)
│   ├── generate_figures.py             # Additional benchmark figures
│   └── visualizer.py                   # Legacy CLI visualization utility
│
├── tests/                              # Unit tests
│   └── test_classifier.py              # Comprehensive test suite (12 tests)
│
├── data/
│   ├── train/                          # Training datasets
│   │   ├── {dataset}_train.csv         # Format: x,y,label (no header)
│   │   ├── {dataset}_dynamic_base.csv  # Base data for dynamic benchmarks
│   │   └── {dataset}_dynamic_stream.csv # Stream data for incremental tests
│   └── test/
│       ├── {dataset}_test_X.csv        # Test features only (x,y)
│       └── {dataset}_test_y.csv        # Test features + ground truth labels (x,y,label)
│
├── results/                            # Benchmark outputs
│   ├── cpp_benchmark_{dataset}.csv     # C++ timing per dataset
│   ├── ablation_{dataset}.csv          # Ablation study results
│   ├── scalability_train.csv           # Scalability timing
│   └── scalability_inference.csv       # O(1) inference verification
│
├── CMakeLists.txt                      # CMake build configuration
└── README.md                           # This file
```

---

## Quick Start Guide

### Step 1: Activate Python Environment

```bash
source venv/bin/activate
```

### Step 2: Generate Datasets

```bash
# Generate all 11 datasets at once
python scripts/generate_datasets.py

# Generate specific dataset(s)
python scripts/generate_datasets.py --type moons
python scripts/generate_datasets.py --type moons,spiral,earthquake
```

**11 datasets** generated in `data/train/` and `data/test/`.

### Step 3: Run Classification

```bash
# With accuracy (pass labeled test file)
./build/main static data/train/wine_train.csv data/test/wine_test_y.csv results/

# Predictions only (pass unlabeled test file)
./build/main static data/train/wine_train.csv data/test/wine_test_X.csv results/
```

**Expected output (with labels):**
```
Phase 1: Detecting Outliers (Min Cluster Size k=3)...
  Adaptive threshold: 1.453... (median edge=0.484... × multiplier=3)
Phase 1 Complete: Removed 2 outliers.
Phase 2 Complete: Delaunay Mesh Built (140 vertices).
Building 2D Buckets with full linked list structures...
  Phase 1: Building LL_V (vertices)...
  Phase 2: Building LL_E (edges) [O(n) bounding-box method]...
  Phase 3: Building LL_GE (grid edge intersections)...
  Phase 4: Building LL_Poly (Voronoi polygon regions)...
2D Buckets construction complete.

=== Classification Results ===
Total Points: 36
Avg Time Per Point:   0.007 us
Accuracy:             91.6667% (33/36)
================================================
```

> **Auto-detection:** `predict_benchmark()` automatically detects whether the test file contains labels (3 columns) or not (2 columns) and reports accuracy when labels are available.

---

## Dataset Generation

### Unified Generator: `generate_datasets.py`

Single script generates all 11 benchmark datasets with consistent output format.

```bash
python scripts/generate_datasets.py                                # All datasets
python scripts/generate_datasets.py --type moons                   # Single dataset
python scripts/generate_datasets.py --type moons,spiral,earthquake # Multiple datasets
```

### Natively 2D Spatial Datasets (Primary)

| Dataset | Samples | Classes | Description |
|---------|---------|---------|-------------|
| moons | 1,000 | 2 | Two interleaving half-moons |
| circles | 1,000 | 2 | Two concentric circles |
| spiral | 1,000 | 2 | Two interlaced Archimedean spirals |
| gaussian_quantiles | 1,000 | 2 | Concentric ellipsoidal boundaries |
| cassini | 1,500 | 3 | Two banana-shaped clusters + central blob |
| checkerboard | 1,000 | 4 | Four-quadrant pattern |
| blobs | 1,500 | 3 | Three Gaussian clusters |
| earthquake | ~5,000 | 4 | USGS real earthquake data by magnitude |

### PCA-Reduced Datasets (Secondary)

| Dataset | Samples | Classes | Source |
|---------|---------|---------|--------|
| wine | 178 | 3 | UCI Wine (13D → PCA to 2D) |
| cancer | 569 | 2 | UCI Breast Cancer (30D → PCA to 2D) |
| bloodmnist | ~17,000 | 8 | MedMNIST cell centroids |

### Output Files Per Dataset

| File | Format | Purpose |
|------|--------|---------|
| `{name}_train.csv` | `x,y,label` | Training data |
| `{name}_test_X.csv` | `x,y` | Test features (blind prediction) |
| `{name}_test_y.csv` | `x,y,label` | Test data with ground truth (accuracy evaluation) |
| `{name}_dynamic_base.csv` | `x,y,label` | 60% of train for dynamic base |
| `{name}_dynamic_stream.csv` | `x,y,label` | 40% of train for dynamic streaming |

---

## Benchmark Procedures

### Benchmark 1: C++ Static Inference (Fair Comparison)

**Purpose:** Pure inference timing with all algorithms in C++.

**Executable:** `./build/benchmark`

```bash
./build/benchmark data/train/wine_train.csv data/test/wine_test_y.csv wine
```

**Output:** `results/cpp_benchmark_wine.csv`

**Sample output:**
```
===============================================================================================
C++ STATIC BENCHMARK: wine
-----------------------------------------------------------------------------------------------
Algorithm                      | Accuracy   | Inference (us)  | Train (ms)   | Speedup   
-----------------------------------------------------------------------------------------------
FLANN C++ KNN (k=5)            |   88.9%   |       3.3449    |      0.07   |     1.0x
LibSVM C++ (RBF, adaptive)     |   91.7%   |       0.3738    |      0.09   |     8.9x
C++ Decision Tree (adaptive)   |   88.9%   |       0.0185    |      0.26   |   180.5x
**Delaunay C++ (Ours)**        |   91.7%   |       0.0069    |      0.80   |   481.7x
===============================================================================================
```

> **Note:** This benchmark provides a **fair algorithmic comparison** with all implementations in C++ using -O3 optimization.

---

### Benchmark 2: C++ Dynamic Updates

**Purpose:** Measure insert/move/delete time vs. Decision Tree rebuild.

**Included in:** `./build/benchmark` (runs automatically after static benchmark)

**Sample output:**
```
===============================================================================================
C++ DYNAMIC BENCHMARK: moons
-----------------------------------------------------------------------------------------------
Algorithm                      | Insert (ns)     | Move (ns)       | Delete (ns)    
-----------------------------------------------------------------------------------------------
C++ Decision Tree (Rebuild)    |      1400000    |      1400000    |      1400000
**Delaunay C++ (O(1) Update)** |       260000    |       486000    |       216000
===============================================================================================
```

---

### Benchmark 3: Ablation Study

**Purpose:** Quantify contribution of each pipeline component (2D Buckets grid, outlier removal, half-plane boundaries).

**Executable:** `./build/ablation_bench`

```bash
./build/ablation_bench data/train/wine_train.csv data/test/wine_test_y.csv wine
```

**Output:** `results/ablation_wine.csv`

---

### Benchmark 4: 10-Fold Cross-Validation (Python)

**Purpose:** Accuracy comparison with statistical significance tests.

**Script:** `scripts/benchmark_cv.py`

```bash
python scripts/benchmark_cv.py \
  --datasets wine,cancer,moons,blobs,spiral,circles,checkerboard,earthquake \
  --folds 10
```

**Outputs:**
- `results/cv_summary.csv` — Mean ± std accuracy for all algorithms
- `results/significance_tests.csv` — Paired t-test and Wilcoxon p-values

---

### Benchmark 5: Scalability Test

**Purpose:** Validate O(n log n) training and O(1) inference complexity claims.

**Script:** `scripts/scalability_test.py`

```bash
python scripts/scalability_test.py
```

**Outputs:**
- `results/scalability_train.csv` — Training time at each n
- `results/scalability_inference.csv` — Inference time at each n
- `results/scalability_plots.png` — Log-log visualization

---

## Unit Testing

```bash
# Run all tests
source venv/bin/activate
python tests/test_classifier.py

# Or with pytest
pip install pytest
pytest tests/test_classifier.py -v
```

### Test Categories

| Category | Tests | Description |
|----------|-------|-------------|
| `TestDataLoading` | 2 | CSV format, label validation |
| `TestDelaunayTriangulation` | 2 | Circumcircle property, basic construction |
| `TestCppClassifier` | 2 | Static mode, accuracy on separable data |
| `TestOutlierDetection` | 1 | Isolated point detection |
| `TestSRRGrid` | 2 | Grid size formula, O(1) lookup |
| `TestDynamicOperations` | 1 | Insert preserves Delaunay |
| `TestClassificationVoting` | 1 | Majority vote correctness |
| `TestDatasetGeneration` | 1 | Output file format |

**Total: 12 tests**

---

## Reproducing Results

Complete reproduction script:

```bash
#!/bin/bash
# Reproduce all benchmark results

# 1. Activate environment
source venv/bin/activate

# 2. Generate all datasets (single command)
echo "=== Generating Datasets ==="
python scripts/generate_datasets.py

# 3. Run C++ benchmarks (all 11 datasets)
echo "=== Running C++ Benchmarks ==="
for ds in moons circles spiral gaussian_quantiles cassini checkerboard blobs earthquake wine cancer bloodmnist; do
  ./build/benchmark data/train/${ds}_train.csv data/test/${ds}_test_y.csv $ds
done

# 4. Run ablation study
echo "=== Running Ablation Study ==="
for ds in moons circles spiral gaussian_quantiles cassini checkerboard blobs earthquake wine cancer bloodmnist; do
  ./build/ablation_bench data/train/${ds}_train.csv data/test/${ds}_test_y.csv $ds
done

# 5. Run scalability test
echo "=== Running Scalability Test ==="
python scripts/scalability_test.py

# 6. Run dynamic stress tests
echo "=== Running Dynamic Stress Tests ==="
for ds in moons circles spiral gaussian_quantiles cassini checkerboard blobs earthquake wine cancer bloodmnist; do
  ./build/main dynamic data/train/${ds}_train.csv data/train/${ds}_dynamic_stream.csv results/dynamic_${ds}.csv
done

echo "=== Complete! ==="
```

**Random seed:** All experiments use `seed=42` for reproducibility.

---

## Benchmark Results

### Platform

| Component | Value |
|-----------|-------|
| **Hardware** | MacBook Pro M3 |
| **OS** | macOS 26 |
| **RAM** | 16 GB |
| **Compiler** | Apple Clang, -O3 |

### C++ Static Benchmark — Accuracy

All algorithms implemented in C++ with -O3 optimization. Test data: `_test_y.csv` (with ground truth labels).

| Dataset | KNN (k=5) | SVM (RBF) | Decision Tree | **Delaunay (Ours)** |
|---------|:---------:|:---------:|:-------------:|:-------------------:|
| moons | **100.0%** | **100.0%** | 98.5% | **100.0%** |
| circles | **100.0%** | **100.0%** | 99.5% | **100.0%** |
| spiral | 98.0% | 65.0% | 95.0% | **99.5%** |
| gaussian_quantiles | 96.0% | **99.0%** | 93.0% | 93.0% |
| cassini | **100.0%** | 99.7% | 99.7% | 99.7% |
| checkerboard | 98.0% | 98.5% | **100.0%** | 97.0% |
| blobs | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| earthquake | 95.4% | 91.9% | 93.2% | **95.5%** |
| wine | 88.9% | **91.7%** | 88.9% | **91.7%** |
| cancer | 93.0% | **94.7%** | 91.2% | 93.0% |
| bloodmnist | 29.6% | 32.6% | 26.8% | **33.1%** |

### C++ Static Benchmark — Inference Speed (µs/point)

| Dataset | KNN | SVM | Decision Tree | **Ours** | **Speedup vs KNN** |
|---------|-----|-----|---------------|----------|:------------------:|
| moons | 4.052 | 0.765 | 0.017 | **0.0027** | **1,495×** |
| circles | 3.850 | 0.326 | 0.013 | **0.0027** | **1,421×** |
| spiral | 3.855 | 4.346 | 0.053 | **0.0029** | **1,322×** |
| gaussian_quantiles | 4.395 | 1.134 | 0.024 | **0.0029** | **1,508×** |
| cassini | 4.305 | 0.699 | 0.011 | **0.0026** | **1,631×** |
| checkerboard | 4.396 | 1.557 | 0.015 | **0.0027** | **1,622×** |
| blobs | 4.488 | 0.335 | 0.009 | **0.0028** | **1,616×** |
| earthquake | 5.568 | 5.697 | 0.062 | **0.0020** | **2,728×** |
| wine | 3.345 | 0.374 | 0.019 | **0.0069** | **482×** |
| cancer | 4.082 | 0.591 | 0.022 | **0.0036** | **1,119×** |
| bloodmnist | 7.142 | 134.157 | 0.099 | **0.0018** | **3,989×** |

### C++ Dynamic Update Speed

| Dataset | DT Rebuild (ns) | **Delaunay Insert** | **Delaunay Move** | **Delaunay Delete** | **Speedup** |
|---------|:---:|:---:|:---:|:---:|:---:|
| moons | 1,400K | 260K | 486K | 216K | 3–7× |
| circles | 1,100K | 252K | 530K | 209K | 2–5× |
| spiral | 1,100K | 234K | 497K | 213K | 2–5× |
| blobs | 4,600K | 770K | 1,100K | 471K | 4–10× |
| earthquake | 13,300K | 545K | 1,200K | 543K | 11–24× |
| cancer | 955K | 100K | 206K | 92K | 5–10× |
| bloodmnist | 127,800K | 989K | 1,970K | 907K | **65–141×** |

### Scalability — O(n log n) Training, O(1) Inference

| n | Training (ms) | Inference (µs/point) |
|------|--------------|---------------------|
| 100 | 28.34 | 0.00 |
| 1,000 | 34.67 | 0.00 |
| 10,000 | 90.94 | 0.01 |
| 100,000 | 802.55 | 0.01 |

> **Key insight:** Inference time remains **constant** regardless of training set size, validating the O(1) claim.

---

## API Reference

### C++ API

```cpp
#include "DelaunayClassifier.h"

// Initialize and train
DelaunayClassifier clf;
clf.train("data/train/wine_train.csv");

// Static inference — O(1) expected via 2D Buckets
int label = clf.classify(0.5, 1.2);

// Without grid (fallback) — O(log n) via CGAL locate
int label_fallback = clf.classify_no_grid(0.5, 1.2);

// Dynamic insert — O(1) amortized
clf.insert_point(0.3, 0.8, 2);  // x, y, label

// Dynamic delete — O(1) amortized
clf.remove_point(0.3, 0.8);

// Dynamic move — O(1) amortized
clf.move_point(old_x, old_y, new_x, new_y, label);

// Batch prediction with optional accuracy
clf.predict_benchmark("test.csv", "predictions.csv");
```

### Command Line Interface

```bash
# Static classification (auto-detects labeled/unlabeled test files)
./build/main static <train_csv> <test_csv> <output_dir>

# Dynamic stress test
./build/main dynamic <train_csv> <stream_csv> <output_dir>

# Full benchmark suite (static + dynamic)
./build/benchmark <train_csv> <test_csv> <dataset_name>

# Ablation study
./build/ablation_bench <train_csv> <test_csv> <dataset_name>
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{delaunay_classifier_2026,
  title={Real-Time Spatial Classification via Delaunay Triangulation 
         with O(1) Point Location},
  author={[Your Name]},
  journal={[Target Journal]},
  year={2026},
  note={Experiments conducted on MacBook Pro M3, macOS 26, 16GB RAM}
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
- [MedMNIST](https://medmnist.com/) — Medical image benchmark datasets