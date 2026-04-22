# Reproducing the Results

This capsule reproduces results for the IEEE TKDE submission
"Real-Time Spatial Classification via Delaunay Triangulation with
O(1) Point Location".

## What happens on Reproducible Run

The capsule runs in `standard` mode by default (~30 minutes on 8 cores):

1. Builds C++ binaries (DelaunayClassifier, benchmark, ablation_bench) with -O3.
2. Regenerates all 12 datasets from cached raw data (no network needed).
3. Runs the C++ benchmark on each dataset comparing Delaunay vs. FLANN KNN,
   LibSVM, Decision Tree, and Random Forest.
4. Runs the scalability test validating O(1) inference up to n=100,000.
5. Generates all paper figures.

Outputs land in `/results/`:
- `cpp_benchmark_<dataset>.csv` — per-dataset accuracy and timing
- `scalability_train.csv`, `scalability_inference.csv` — complexity validation
- `figures/` — all paper figures

## Changing the run mode

Edit `/code/run` and change `MODE="${1:-${MODE:-standard}}"` to
`quick` (~5 min) or `full` (~2-3 hours, full multi-seed reproduction).

## Hardware note

Code Ocean runs on AWS EC2. Absolute inference microseconds will differ
from the paper's Apple M3 measurements, but the *relative* speedups
(Delaunay vs. FLANN/LibSVM/DT/RF) and the *flatness* of the O(1) inference
curve for n ≤ 300K are the invariant reproducible findings.

## Dependencies (pre-installed in the capsule)

- C++: CGAL ≥ 5.0, FLANN ≥ 1.9, LibSVM ≥ 3.25, LZ4 (all via apt)
- Python 3.12: numpy, scipy, pandas, matplotlib, scikit-learn, medmnist,
  requests (pinned versions in requirements.txt)