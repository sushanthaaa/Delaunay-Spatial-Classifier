# Fast Geometric Classification using Delaunay Triangulation

   

> **Reference Implementation for IEEE Transactions Research** \> *Classification using Delaunay Triangulation-Based Decision Boundary Algorithms*

This repository contains the official C++ implementation and Python benchmarking suite for a novel geometric classification algorithm. Unlike statistical methods (k-NN, SVM) that require computationally expensive global retraining ($O(N)$), our approach utilizes **Delaunay Triangulation** to build a navigable mesh structure, enabling **constant time ($O(1)$)** dynamic updates and **logarithmic ($O(\log N)$)** inference speed.

-----

## 📊 Key Results

Our algorithm demonstrates massive speedups over traditional classifiers while maintaining state-of-the-art accuracy on complex medical and chemical datasets.

| Dataset | Accuracy | Inference Speed | Speedup vs KNN |
| :--- | :---: | :---: | :---: |
| **Breast Cancer** | **99.1%** | **0.21 µs** | **925x** |
| **Wine** | **100%** | **0.16 µs** | **3,700x** |
| **Iris** | **96.7%** | **0.23 µs** | **4,540x** |

### Dynamic Scalability ($O(1)$)

While KNN and SVM execution times scale linearly with dataset size, our **Algorithm 1 (Dynamic Insertion)** remains constant.

*Fig 1: Log-scale comparison showing the constant-time update cost (Green) vs. the linear retraining cost of competitors.*

-----

## 📂 Repository Structure

The project follows a standard IEEE research artifact structure:

```text
.
├── src/                 # Core C++ Implementation
│   ├── DelaunayClassifier.cpp  # Algorithms 1, 2, 3, 4
│   └── main.cpp                # CLI Controller
├── include/             # Header Definitions
├── scripts/             # Python Benchmarking Suite
│   ├── data_generator.py       # Dataset creation & Scaling
│   ├── benchmark.py            # Comparative Analysis (KNN/SVM/RF)
│   └── visualizer.py           # IEEE Figure Generation
├── data/                # Generated Training/Testing CSVs
└── results/             # Benchmark Logs and Output Figures
```

-----

## 🛠️ Prerequisites

To reproduce the results, ensure you have the following installed:

  * **C++ Compiler:** `g++` or `clang` (Supports C++14/17)
  * **Build System:** `CMake` (Version 3.10+)
  * **Geometry Library:** `CGAL` (The Computational Geometry Algorithms Library)
      * *Mac:* `brew install cgal`
      * *Linux:* `sudo apt-get install libcgal-dev`
  * **Python 3:** With data science libraries.

### Python Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn matplotlib
```

-----

## 🚀 Standard Operating Procedure (SOP)

Follow this workflow to replicate the research findings.

### Step 1: Generate Datasets

We use `StandardScaler` to normalize all datasets (Iris, Wine, Cancer, Digits) to ensure geometric consistency.

```bash
python3 scripts/data_generator.py --type cancer
python3 scripts/data_generator.py --type wine
```

### Step 2: Build the High-Performance Executable

**Crucial:** We must compile in `Release` mode to enable compiler optimizations (`-O3`), otherwise the geometric kernel will be slow.

```bash
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
cd ..
```

### Step 3: Run Static Benchmark (Accuracy & Inference)

This script trains Python baselines (KNN, SVM, Random Forest) and the C++ Delaunay executable, comparing their inference speed on the test set.

```bash
python3 scripts/benchmark.py --dataset cancer --mode static
```

### Step 4: Run Dynamic Stress Test ($O(1)$ Proof)

This tests Algorithms 1, 2, and 3 (Insertion, Deletion, Movement) by simulating a data stream.

```bash
# 1. Run Python Wrapper (Handles C++ execution and logging)
python3 scripts/benchmark.py --dataset cancer --mode dynamic

# 2. Visualize the O(1) Trend
python3 scripts/visualizer.py --mode dynamic --results_dir results/logs
```

### Step 5: Generate IEEE Figures

Visualize the decision boundaries and mesh construction.

```bash
# 1. Generate Mesh Data
./build/main static data/train/cancer_train.csv data/test/cancer_test_X.csv results/figures

# 2. Plot
python3 scripts/visualizer.py --mode static --dataset cancer --data_dir data/train --results_dir results/figures
```

-----

## 🖼️ Visualization Gallery

### 1\. Medical Diagnosis (Breast Cancer)

The algorithm successfully navigates the complex, overlapping boundary between malignant and benign clusters using geometric centroids.

### 2\. Chemical Analysis (Wine)

Perfect separation of three chemical classes with 100% accuracy.

-----

## 🧪 Algorithms Implemented

This repository implements the four core algorithms described in the paper:

1.  **Algorithm 1 (Dynamic Insertion):** Adds a new point and retriangulates locally in $O(1)$.
2.  **Algorithm 2 (Dynamic Deletion):** Removes a point and fills the hole (star-shaped polygon) in $O(k)$.
3.  **Algorithm 3 (Dynamic Movement):** Simulates moving objects by flipping edges if geometric constraints are violated.
4.  **Algorithm 4 (Classification):**
      * **Phase 1:** Graph-based Outlier Removal.
      * **Phase 2:** Delaunay Mesh Construction.
      * **Phase 3:** Decision Boundary Extraction (Midpoint/Centroid logic).
      * **Phase 4:** Point Location & Classification.

-----

## 📝 Citation

If you use this code for your research, please cite:

```bibtex
@article{jan2025classification,
  title={Classification using Delaunay triangulation-based decision boundary algorithms},
  author={Jan, Gene Eu and Su, Wong and Yang},
  journal={IEEE Transactions on Cybernetics (Submitted)},
  year={2025}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.