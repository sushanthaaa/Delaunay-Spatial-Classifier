#!/usr/bin/env python3
"""
Scalability Test for Delaunay Triangulation Classifier

Validates the two core complexity claims:
  1. Training time scales as O(n log n)
  2. Inference time is O(1) — constant regardless of training set size

Generates synthetic 2D blob data at n = {100, 1K, 10K, 100K}, times
the C++ classifier, and produces log-log plots.

Usage:
  python scripts/scalability_test.py

Outputs:
  results/scalability_train.csv      — Training time at each n
  results/scalability_inference.csv   — Inference time at each n
  results/scalability_plots.png       — Log-log visualization
"""

import os
import subprocess
import time
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

RANDOM_SEED = 42
SIZES = [100, 1000, 10000, 100000]
N_CLASSES = 5
N_INFERENCE_SAMPLES = 200  # Fixed test set for fair inference comparison
N_REPEATS = 3              # Repeat each measurement, take median


def generate_synthetic_data(n_samples, n_classes=5, random_state=42):
    """Generate 2D blob dataset with n samples."""
    X, y = make_blobs(n_samples=n_samples, n_features=2, centers=n_classes,
                      cluster_std=1.0, random_state=random_state)
    return X, y


def write_csv(X, y, path):
    """Write data to CSV in x,y,label format (no header)."""
    with open(path, 'w') as f:
        for i in range(len(X)):
            f.write(f"{X[i,0]},{X[i,1]},{int(y[i])}\n")


def run_cpp_benchmark(train_X, train_y, test_X, test_y, cpp_exe, results_dir):
    """Run C++ classifier and parse both training and inference timing.

    Returns (train_time_ms, inference_time_us_per_point).

    Training time is measured as subprocess wall-clock minus inference time,
    which gives a closer approximation to pure algorithmic cost. Inference
    time is read from the C++ internal high_resolution_clock output.
    """
    train_path = tempfile.mktemp(suffix='.csv')
    test_path = tempfile.mktemp(suffix='.csv')

    write_csv(train_X, train_y, train_path)
    write_csv(test_X, test_y, test_path)

    cmd = [cpp_exe, "static", train_path, test_path, results_dir]

    wall_start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall_end = time.perf_counter()

    wall_time_ms = (wall_end - wall_start) * 1000

    # Parse inference time from C++ output
    avg_time_us = 0.0
    for line in proc.stdout.split('\n'):
        if "Avg Time Per Point" in line:
            try:
                avg_time_us = float(line.split()[-2])
            except (ValueError, IndexError):
                pass

    # Approximate training time = wall time - total inference time
    total_inference_ms = (avg_time_us * len(test_X)) / 1000.0
    train_time_ms = max(0, wall_time_ms - total_inference_ms)

    os.remove(train_path)
    os.remove(test_path)

    # Clean up generated files
    for f in ['predictions.csv', 'triangles.csv', 'boundaries.csv',
              'clean_points.csv']:
        p = os.path.join(results_dir, f)
        if os.path.exists(p):
            os.remove(p)

    return train_time_ms, avg_time_us


def run_scalability_test(root_dir):
    """Run full scalability experiment."""
    cpp_exe = f"{root_dir}/build/main"
    results_dir = f"{root_dir}/results"
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(cpp_exe):
        print(f"Error: C++ executable not found at {cpp_exe}")
        print("Build with: cd build && cmake .. && make")
        return None, None

    print("=" * 70)
    print("SCALABILITY EXPERIMENT")
    print(f"Sizes: {SIZES}")
    print(f"Repeats per size: {N_REPEATS} (median reported)")
    print(f"Inference test points: {N_INFERENCE_SAMPLES}")
    print("=" * 70)

    # Fixed test set for fair inference comparison
    X_test, y_test = generate_synthetic_data(
        N_INFERENCE_SAMPLES, N_CLASSES, RANDOM_SEED + 1)

    train_results = []
    inference_results = []

    for n in SIZES:
        print(f"\nn = {n:,}")
        print("-" * 40)

        X, y = generate_synthetic_data(n, N_CLASSES, RANDOM_SEED)

        train_times = []
        inference_times = []

        for rep in range(N_REPEATS):
            t_train, t_infer = run_cpp_benchmark(
                X, y, X_test, y_test, cpp_exe, results_dir)
            train_times.append(t_train)
            inference_times.append(t_infer)

        median_train = np.median(train_times)
        median_inference = np.median(inference_times)

        train_results.append({
            "n": n,
            "train_time_ms": median_train,
            "time_s": median_train / 1000.0
        })
        inference_results.append({
            "n": n,
            "inference_us": median_inference,
            "time_us": median_inference
        })

        print(f"  Training:  {median_train:.2f} ms "
              f"(median of {N_REPEATS} runs)")
        print(f"  Inference: {median_inference:.4f} us/point "
              f"(median of {N_REPEATS} runs)")

    # Save results
    df_train = pd.DataFrame(train_results)
    df_train.to_csv(f"{results_dir}/scalability_train.csv", index=False)

    df_inference = pd.DataFrame(inference_results)
    df_inference.to_csv(f"{results_dir}/scalability_inference.csv", index=False)

    print(f"\nTraining results: {results_dir}/scalability_train.csv")
    print(f"Inference results: {results_dir}/scalability_inference.csv")

    generate_plots(train_results, inference_results, results_dir)

    return train_results, inference_results


def generate_plots(train_results, inference_results, results_dir):
    """Generate log-log plots for training and inference scaling."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    n_vals = [r["n"] for r in train_results]
    train_times = [r["train_time_ms"] for r in train_results]
    inference_times = [r["inference_us"] for r in inference_results]

    # Plot 1: Training time (log-log)
    ax1 = axes[0]
    ax1.loglog(n_vals, train_times, 'bo-', linewidth=2, markersize=8,
               label='Measured')

    # O(n log n) reference line fitted to first data point
    n_ref = np.array(n_vals, dtype=float)
    c = train_times[0] / (n_ref[0] * np.log2(n_ref[0]))
    ref_line = c * n_ref * np.log2(n_ref)
    ax1.loglog(n_vals, ref_line, 'r--', linewidth=1.5, alpha=0.7,
               label='O(n log n) reference')

    ax1.set_xlabel('Number of training points (n)', fontsize=12)
    ax1.set_ylabel('Training time (ms)', fontsize=12)
    ax1.set_title('Training Time Scalability', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Inference time (should be O(1) — flat line)
    ax2 = axes[1]
    ax2.semilogx(n_vals, inference_times, 'go-', linewidth=2, markersize=8,
                 label='Measured')

    mean_inference = np.mean(inference_times)
    ax2.axhline(y=mean_inference, color='r', linestyle='--', linewidth=1.5,
                alpha=0.7,
                label=f'O(1) reference ({mean_inference:.3f} us)')

    ax2.set_xlabel('Number of training points (n)', fontsize=12)
    ax2.set_ylabel('Inference time per point (us)', fontsize=12)
    ax2.set_title('Inference Time: O(1) via 2D Buckets', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, max(inference_times) * 2.0 if inference_times else 1])

    plt.tight_layout()
    plot_path = f"{results_dir}/scalability_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plots saved to: {plot_path}")
    plt.close()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(script_dir, ".."))

    train_results, inference_results = run_scalability_test(root)

    if train_results and inference_results:
        # Print summary table
        print("\n" + "=" * 60)
        print("SCALABILITY SUMMARY")
        print("=" * 60)
        print(f"{'n':>10} | {'Training (ms)':>15} | {'Inference (us)':>15}")
        print("-" * 60)
        for t, i in zip(train_results, inference_results):
            print(f"{t['n']:>10,} | {t['train_time_ms']:>15.2f} | "
                  f"{i['inference_us']:>15.4f}")
        print("=" * 60)

        # Verify claims
        train_ratio = train_results[-1]['train_time_ms'] / train_results[0]['train_time_ms']
        n_ratio = SIZES[-1] / SIZES[0]
        nlogn_ratio = (SIZES[-1] * np.log2(SIZES[-1])) / (SIZES[0] * np.log2(SIZES[0]))

        print(f"\nTraining: {n_ratio:.0f}x more data -> "
              f"{train_ratio:.1f}x more time")
        print(f"  O(n log n) predicts: {nlogn_ratio:.1f}x")
        print(f"  O(n^2) would predict: {n_ratio**2/n_ratio:.0f}x")

        infer_times = [r['inference_us'] for r in inference_results]
        infer_cv = np.std(infer_times) / np.mean(infer_times) * 100
        print(f"\nInference: CV = {infer_cv:.1f}% across all sizes")
        print(f"  O(1) confirmed: {'YES' if infer_cv < 50 else 'INCONCLUSIVE'}")

    print("\n" + "=" * 60)
    print("SCALABILITY ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()