# Fast Geometric Classification using Delaunay Triangulation & Spatial Hashing (SRR)


This repository hosts the source code for a high-performance geometric classifier. Unlike statistical methods (k-NN, SVM) that suffer from linear scaling () or require expensive global retraining, our approach integrates **Delaunay Triangulation** with a **Square Root Raster (SRR)** spatial index.

This architecture enables:

1. **Constant Time () Inference:** Classifying points in ~0.1  regardless of dataset size.
2. **Constant Time () Dynamic Updates:** Inserting, moving, or deleting data points without global retraining.
3. **Non-Linear Separability:** Geometric handling of complex boundaries (e.g., Two Moons) where linear classifiers fail.

---

## 📊 Performance Highlights

Experimental results on macOS (Clang 14.0).

### 1. Inference Speed (The "SRR Effect")

By utilizing the **Square Root Raster (2D Bucket)** indexing, we achieve a consistent inference time of , delivering massive speedups over k-NN.

| Dataset | Type | Samples | Accuracy | Our Inference | Speedup vs KNN |
| --- | --- | --- | --- | --- | --- |
| **Wine** | Chemical | 178 | **100%** | **0.11 µs** | **~15,000x** |
| **Iris** | Botanical | 150 | **96.7%** | **0.13 µs** | **~10,000x** |
| **Cancer** | Medical | 569 | **99.1%** | **1.06 µs** | **~160x** |
| **Digits** | Vision | 1797 | **52.2%** | **0.10 µs** | **~650x** |
| **Moons** | Synthetic | 1000 | **100%** | **0.30 µs** | **~500x** |

> *Note: Speedup fluctuates based on system load and Python "cold start" overhead, consistently ranging between 5,000x and 15,000x for lightweight datasets.*

### 2. Dynamic Scalability (The Flat Line)

While competitors like SVM require global retraining () when a new data point arrives, our algorithm performs a **local retriangulation** in Constant Time ().

*Fig 1: Log-scale comparison. The Green Line (Ours) remains flat/constant, while SVM/KNN (Competitors) rise exponentially.*

---

## 🛠️ System Architecture

The core engine is written in **C++** (using CGAL) for maximum performance, wrapped with **Python** scripts for data generation, benchmarking, and visualization.

### Algorithm Pipeline

1. **Phase 1 (Graph-Based Cleaning):** Removes noise and disconnected outliers using graph connectivity ().
2. **Phase 2 (Mesh Construction):** Builds the Delaunay Triangulation ().
3. **Phase 3 (Spatial Indexing - SRR):**
* Constructs a **Square Root Raster** grid ( buckets).
* Maps spatial regions to specific mesh triangles.
* **Result:** Transforms point location from  to ****.


4. **Phase 4 (Inference):** Uses SRR hints to instantly locate the target triangle and classify based on the nearest vertex ().

---

## 🚀 Standard Operating Procedure (SOP)

Follow this workflow to reproduce the research results.

### Prerequisites

* **C++:** CMake 3.10+, CGAL Library (`brew install cgal` or `apt install libcgal-dev`).
* **Python:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`.

### Step 1: Generate Datasets

We generate both real-world benchmarks (Wine, Cancer) and synthetic stress tests (Moons, Blobs). The script automatically splits data into **Base** (Training) and **Stream** (Dynamic Test).

```bash
# Real-World Data
python scripts/data_generator.py --type wine
python scripts/data_generator.py --type cancer
python scripts/data_generator.py --type digits
python scripts/data_generator.py --type iris

# Synthetic Data (Geometric Stress Test)
python scripts/data_generator.py --type moons
python scripts/data_generator.py --type blobs

```

### Step 2: Compile the Engine

Compile in `Release` mode to enable compiler optimizations (`-O3`).

```bash
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
cd ..

```

### Step 3: Run Static Benchmarks (Accuracy & Speed)

Compare our C++ engine against Python implementations of KNN, SVM, Decision Tree, and Random Forest.

```bash
python scripts/benchmark.py --dataset wine --mode static
python scripts/benchmark.py --dataset cancer --mode static
# ... repeat for other datasets

```

### Step 4: Run Dynamic Stress Tests

Prove the  update capability by simulating a data stream (Insert/Move/Delete).

```bash
python scripts/benchmark.py --dataset wine --mode dynamic
python scripts/benchmark.py --dataset moons --mode dynamic

```

### Step 5: Visualize the Geometry

**A. Static Boundaries (Classification):**

```bash
python scripts/visualizer.py --mode static --dataset wine --results_dir results
python scripts/visualizer.py --mode static --dataset moons --results_dir results

```

**B. Dynamic Evolution (Snapshots):**
This captures the mesh state after Insertion, Movement, and Deletion.

```bash
# 1. Run C++ Visualization Mode
./build/main visualize_dynamic data/train/wine_dynamic_base.csv data/train/wine_dynamic_stream.csv results

# 2. Render Images
python scripts/visualizer.py --mode dynamic_visual --dataset wine --results_dir results

```

---

## 🖼️ Visualization Gallery

### 1. Complex Non-Linear Boundaries (Two Moons)

Demonstrates the algorithm's ability to navigate non-linearly separable data without kernel tricks.

### 2. High-Density Medical Data (Breast Cancer)

Successfully separates Malignant vs. Benign clusters with 99.1% accuracy.

### 3. Dynamic Mesh Adaptation

Visual proof of the mesh updating locally as new points are inserted and moved.

---

## 🌍 Real-World Impact

This research addresses critical bottlenecks in real-time AI:

1. **Haptic Tele-Surgery:**
* *Challenge:* Feedback loops must run at >1kHz (1ms) to feel "smooth" to a surgeon.
* *Solution:* Our algorithm classifies tissue density in **1.0 µs**, enabling real-time tumor boundary detection without lag.


2. **Edge AI & IoT:**
* *Challenge:* Embedded chips (smartwatches, drones) lack the power for Neural Networks.
* *Solution:* The geometric logic is lightweight and battery-efficient, allowing on-chip classification.



---

## 🔮 Future Work

* **Parallel Processing (CUDA):** Utilizing the SRR Grid structure to map buckets to GPU threads, enabling massive parallel insertion of data points.
* **3D Extension:** extending the Delaunay kernel to volumetric data for MRI analysis.

---

## 📝 Citation

```bibtex
@article{jan2025classification,
  title={Classification using Delaunay triangulation-based decision boundary algorithms},
  author={Jan, Gene Eu and Su, Wong and Yang},
  journal={IEEE Transactions on Cybernetics (Submitted)},
  year={2025}
}

```

## 📜 License

MIT License. See [LICENSE](https://www.google.com/search?q=LICENSE) for details.