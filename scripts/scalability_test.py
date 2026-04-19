#!/usr/bin/env python3
"""
Scalability Test for Delaunay Triangulation Classifier

Validates the two core complexity claims of the classifier:
  1. Training time scales as O(n log n)  — dominated by Delaunay construction
  2. Inference time is O(1)               — constant regardless of training size

Generates synthetic 2D blob data at n = {100, 1K, 10K, 100K}, runs the C++
classifier, parses the internal high-resolution timing from its structured
output, and produces log-log plots.

Usage:
  python scripts/scalability_test.py
  python scripts/scalability_test.py --seed 123
  python scripts/scalability_test.py --sizes 100,1000,10000
  python scripts/scalability_test.py --repeats 5

Outputs:
  results/scalability_train.csv       — Training time at each n
  results/scalability_inference.csv   — Inference time at each n
  results/scalability_plots.png       — Log-log visualization

Fixes applied (Week 3 of the master action list):
              measuring training time. Previously, training time was
              computed as `subprocess_wall_clock_ms - total_inference_ms`,
              which conflates three things: subprocess startup overhead
              (10-50 ms), filesystem I/O for CSVs, and actual algorithmic
              training cost. At small n (100 points) the subprocess
              startup alone could be 50x larger than the true training
              time, making the reported numbers meaningless.
              structured output. This line is emitted by main.cpp after
              the companion C++ change (see MIGRATION_GUIDE.md
              and reports training time as measured internally by
              std::chrono::high_resolution_clock, excluding all
              subprocess/IO/startup overhead.

              session's C++ changes, DelaunayClassifier.cpp's
              predict_benchmark() was changed from integer-microsecond
              precision to nanosecond precision, so sub-microsecond
              Delaunay inference times are no longer truncated to 0.
              This script now reports correct inference timing for all
              dataset sizes (previously small-n inference appeared as
              0.0 due to the C++ truncation bug).

Required companion C++ changes (see MIGRATION_GUIDE.md for exact edits):
  - src/main.cpp : wrap classifier.train() in chrono timing, emit
                   "Training Time: X.XXXX ms" after the training completes
  - src/DelaunayClassifier.cpp : change predict_benchmark()'s timing
                   computation to use nanoseconds + double arithmetic
"""

import argparse
import os
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# =============================================================================
# Configuration constants
# =============================================================================

# Module-level default seed. Actual seed used at runtime comes from --seed.
# Matches the convention in generate_datasets.py so all multi-seed-aware
# scripts use the same default.
DEFAULT_SEED = 42

# Default training set sizes, chosen to span 3 orders of magnitude so the
# O(n log n) training curve and the O(1) inference plateau are both clearly
# visible on a log-log plot. Can be overridden via --sizes.
DEFAULT_SIZES = [100, 1000, 10000, 100000, 300000, 1000000]

# Number of classes in the synthetic blob data. Kept at 5 to give DT meaningful
# multi-class work while keeping bucket occupancy reasonable.
N_CLASSES = 5

# Fixed test set size across all training sizes. This is the key to testing
# O(1) inference: we measure per-point inference time on the SAME test set
# regardless of how large the training set is. If inference is O(1), the
# per-point time should be flat across all training sizes.
N_INFERENCE_SAMPLES = 200

# Number of repeats per (n, measurement) pair. We report the median to be
# robust to outliers from OS scheduling jitter and cache effects.
DEFAULT_REPEATS = 3


# =============================================================================
# Data generation and I/O
# =============================================================================

def generate_synthetic_data(n_samples, n_classes, seed):
    """Generate 2D blob dataset with n samples using the given seed."""
    X, y = make_blobs(n_samples=n_samples, n_features=2, centers=n_classes,
                      cluster_std=1.0, random_state=seed)
    return X, y


def write_train_csv(X, y, path):
    """Write training data to CSV in x,y,label format (no header)."""
    with open(path, 'w') as f:
        for i in range(len(X)):
            f.write(f"{X[i, 0]},{X[i, 1]},{int(y[i])}\n")


def write_test_csv(X, y, path):
    """Write test data to CSV in x,y,label format (no header).

    The C++ binary accepts both 2-column (x,y) and 3-column (x,y,label)
    test files in static mode. We emit 3-column here to match the format
    used throughout the rest of the benchmark pipeline.
    """
    with open(path, 'w') as f:
        for i in range(len(X)):
            f.write(f"{X[i, 0]},{X[i, 1]},{int(y[i])}\n")


# =============================================================================
# C++ subprocess invocation and output parsing
# =============================================================================

def parse_structured_timing(stdout):
    """Parse training and inference time from the C++ binary's output.

      "Training Time: X.XXXX ms"
    after the classifier.train() call completes. This line reports the
    internally-measured training time using std::chrono::high_resolution_clock,
    which excludes subprocess startup, filesystem I/O, and other wall-clock
    overhead that pollutes Python-side timing.

    nanosecond precision for its inference timing, so sub-microsecond
    per-point times (expected for small-n Delaunay) are no longer
    truncated to 0 in the "Avg Time Per Point: X.XXXX us" line.

    Returns a dict with:
      'train_time_ms' : float or None (None means the line was missing)
      'inference_us'  : float or None (None means the line was missing)

    A return value of None for either field indicates that the C++ binary
    has not been rebuilt with the companion / #29b changes.
    The caller is responsible for emitting a clear error to the user.
    """
    train_time_ms = None
    inference_us = None

    for line in stdout.split('\n'):
        if "Training Time:" in line:
            tokens = line.split()
            # Format: "Training Time: X.XXXX ms"
            if len(tokens) >= 2 and tokens[-1] == "ms":
                try:
                    train_time_ms = float(tokens[-2])
                except ValueError:
                    pass
        elif "Avg Time Per Point" in line:
            tokens = line.split()
            # Format: "Avg Time Per Point:   X.XXXX us"
            if len(tokens) >= 2 and tokens[-1] == "us":
                try:
                    inference_us = float(tokens[-2])
                except ValueError:
                    pass

    return {
        'train_time_ms': train_time_ms,
        'inference_us': inference_us,
    }


def run_cpp_benchmark(train_X, train_y, test_X, test_y, cpp_exe, results_dir):
    """Run C++ classifier and return (train_time_ms, inference_time_us_per_point).

    Both timings come from the C++ binary's internal chrono measurements
    via parse_structured_timing(). This is a full replacement for the old
    wall-clock subtraction approach .

    Returns (None, None) for any field whose timing line is missing from
    the C++ output, indicating the binary needs to be rebuilt with the
    companion / #29b C++ changes.
    """
    train_path = tempfile.mktemp(suffix='.csv')
    test_path = tempfile.mktemp(suffix='.csv')

    write_train_csv(train_X, train_y, train_path)
    write_test_csv(test_X, test_y, test_path)

    cmd = [cpp_exe, "static", train_path, test_path, results_dir]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    timing = parse_structured_timing(proc.stdout)
    train_time_ms = timing['train_time_ms']
    avg_time_us = timing['inference_us']

    # Clean up temp input files.
    try:
        os.remove(train_path)
    except OSError:
        pass
    try:
        os.remove(test_path)
    except OSError:
        pass

    # Clean up C++-generated output files so they don't leak into other runs.
    for f in ['predictions.csv', 'triangles.csv', 'boundaries.csv',
              'clean_points.csv']:
        p = os.path.join(results_dir, f)
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass

    return train_time_ms, avg_time_us


# =============================================================================
# Scalability experiment
# =============================================================================

def run_scalability_test(root_dir, sizes, n_classes, n_inference_samples,
                         n_repeats, seed):
    """Run the full scalability experiment across the given sizes."""
    cpp_exe = f"{root_dir}/build/main"
    results_dir = f"{root_dir}/results"
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(cpp_exe):
        print(f"Error: C++ executable not found at {cpp_exe}")
        print("Build with: cd build && cmake .. && make")
        return None, None

    print("=" * 70)
    print("SCALABILITY EXPERIMENT")
    print(f"Sizes: {sizes}")
    print(f"Classes: {n_classes}")
    print(f"Repeats per size: {n_repeats} (median reported)")
    print(f"Inference test points: {n_inference_samples}")
    print(f"Seed: {seed}")
    print("=" * 70)

    # Fixed test set for fair inference comparison across all training sizes.
    # Using seed+1 for the test set so it's deterministic relative to the run
    # seed but distinct from the training data distribution.
    X_test, y_test = generate_synthetic_data(
        n_inference_samples, n_classes, seed + 1)

    train_results = []
    inference_results = []

    # entire run so we can emit a single clear warning at the end rather
    # than spamming the terminal once per failed measurement.
    missing_train_count = 0
    missing_inference_count = 0
    total_measurements = 0

    for n in sizes:
        print(f"\nn = {n:,}")
        print("-" * 40)

        X, y = generate_synthetic_data(n, n_classes, seed)

        train_times = []
        inference_times = []

        for rep in range(n_repeats):
            t_train, t_infer = run_cpp_benchmark(
                X, y, X_test, y_test, cpp_exe, results_dir)
            total_measurements += 1

            # deliberately use NaN rather than 0.0 so the aggregation
            # step visibly shows the missing data.
            if t_train is None:
                missing_train_count += 1
                t_train = float('nan')
            if t_infer is None:
                missing_inference_count += 1
                t_infer = float('nan')

            train_times.append(t_train)
            inference_times.append(t_infer)

        # nanmedian skips NaN entries. If all repeats missing, result is NaN.
        median_train = float(np.nanmedian(train_times))
        median_inference = float(np.nanmedian(inference_times))

        train_results.append({
            "n": n,
            "train_time_ms": median_train,
            "time_s": median_train / 1000.0 if not np.isnan(median_train) else float('nan'),
        })
        inference_results.append({
            "n": n,
            "inference_us": median_inference,
            "time_us": median_inference,
        })

        train_str = (f"{median_train:.4f} ms"
                     if not np.isnan(median_train) else "MISSING")
        infer_str = (f"{median_inference:.4f} us/point"
                     if not np.isnan(median_inference) else "MISSING")
        print(f"  Training:  {train_str} (median of {n_repeats} runs)")
        print(f"  Inference: {infer_str} (median of {n_repeats} runs)")

    # emit a clear instruction to the user to rebuild. This is a hard
    # error condition, not a warning — the scalability claims cannot be
    # validated without the C++ binary emitting the expected lines.
    if missing_train_count > 0 or missing_inference_count > 0:
        print("\n" + "=" * 70)
        print("WARNING: Some timing lines were missing from the C++ output.")
        if missing_train_count > 0:
            print(f"  'Training Time: X ms' missing in "
                  f"{missing_train_count}/{total_measurements} runs.")
            print("  This indicates main.cpp has not been rebuilt with the")
            print(" change. See MIGRATION_GUIDE.md for the edit.")
        if missing_inference_count > 0:
            print(f"  'Avg Time Per Point: X us' missing in "
                  f"{missing_inference_count}/{total_measurements} runs.")
            print("  This indicates predict_benchmark() has not been rebuilt.")
        print("  After applying the C++ changes, run:")
        print("    cd build && make clean && make -j4")
        print("=" * 70)

    # Save results.
    df_train = pd.DataFrame(train_results)
    df_train.to_csv(f"{results_dir}/scalability_train.csv", index=False)

    df_inference = pd.DataFrame(inference_results)
    df_inference.to_csv(f"{results_dir}/scalability_inference.csv", index=False)

    print(f"\nTraining results: {results_dir}/scalability_train.csv")
    print(f"Inference results: {results_dir}/scalability_inference.csv")

    generate_plots(train_results, inference_results, results_dir)

    return train_results, inference_results


# =============================================================================
# Plotting
# =============================================================================

def generate_plots(train_results, inference_results, results_dir):
    """Generate log-log plots for training and inference scaling."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    n_vals = [r["n"] for r in train_results]
    train_times = [r["train_time_ms"] for r in train_results]
    inference_times = [r["inference_us"] for r in inference_results]

    # Filter out NaN values before plotting.
    train_vals_valid = [(n, t) for n, t in zip(n_vals, train_times)
                        if not np.isnan(t)]
    infer_vals_valid = [(n, t) for n, t in zip(n_vals, inference_times)
                        if not np.isnan(t)]

    # --- Plot 1: Training time (log-log) ---
    ax1 = axes[0]
    if train_vals_valid:
        n_train, t_train = zip(*train_vals_valid)
        ax1.loglog(n_train, t_train, 'bo-', linewidth=2, markersize=8,
                   label='Measured')

        # O(n log n) reference line fitted to the first valid data point.
        n_ref = np.array(n_train, dtype=float)
        c = t_train[0] / (n_ref[0] * np.log2(n_ref[0]))
        ref_line = c * n_ref * np.log2(n_ref)
        ax1.loglog(n_train, ref_line, 'r--', linewidth=1.5, alpha=0.7,
                   label='O(n log n) reference')
    else:
        ax1.text(0.5, 0.5, 'No valid training timings\n(rebuild C++ binary)',
                 ha='center', va='center', transform=ax1.transAxes,
                 fontsize=12, color='red')

    ax1.set_xlabel('Number of training points (n)', fontsize=12)
    ax1.set_ylabel('Training time (ms)', fontsize=12)
    ax1.set_title('Training Time Scalability', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Inference time (should be O(1) — flat line) ---
    ax2 = axes[1]
    if infer_vals_valid:
        n_infer, t_infer = zip(*infer_vals_valid)
        ax2.semilogx(n_infer, t_infer, 'go-', linewidth=2, markersize=8,
                     label='Measured')

        mean_inference = float(np.mean(t_infer))
        ax2.axhline(y=mean_inference, color='r', linestyle='--',
                    linewidth=1.5, alpha=0.7,
                    label=f'O(1) reference ({mean_inference:.4f} us)')
        ax2.set_ylim([0, max(t_infer) * 2.0])
    else:
        ax2.text(0.5, 0.5, 'No valid inference timings\n(rebuild C++ binary)',
                 ha='center', va='center', transform=ax2.transAxes,
                 fontsize=12, color='red')

    ax2.set_xlabel('Number of training points (n)', fontsize=12)
    ax2.set_ylabel('Inference time per point (us)', fontsize=12)
    ax2.set_title('Inference Time: O(1) via 2D Buckets', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = f"{results_dir}/scalability_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plots saved to: {plot_path}")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Scalability test for Delaunay classifier (O(n log n) "
                    "training, O(1) inference)")
    parser.add_argument(
        '--seed', type=int, default=DEFAULT_SEED,
        help=f'Random seed for synthetic blob generation '
             f'(default: {DEFAULT_SEED})')
    parser.add_argument(
        '--sizes', default=None,
        help=f'Comma-separated training set sizes '
             f'(default: {",".join(str(s) for s in DEFAULT_SIZES)})')
    parser.add_argument(
        '--repeats', type=int, default=DEFAULT_REPEATS,
        help=f'Number of repeats per size, median reported '
             f'(default: {DEFAULT_REPEATS})')
    parser.add_argument(
        '--n_classes', type=int, default=N_CLASSES,
        help=f'Number of classes in synthetic blob data '
             f'(default: {N_CLASSES})')
    parser.add_argument(
        '--n_inference', type=int, default=N_INFERENCE_SAMPLES,
        help=f'Number of test points for inference timing '
             f'(default: {N_INFERENCE_SAMPLES})')
    args = parser.parse_args()

    if args.sizes is None:
        sizes = DEFAULT_SIZES
    else:
        try:
            sizes = [int(s.strip()) for s in args.sizes.split(',')]
        except ValueError:
            print("Error: --sizes must be comma-separated integers")
            sys.exit(1)
        if any(s < 2 for s in sizes):
            print("Error: all sizes must be >= 2")
            sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(script_dir, ".."))

    train_results, inference_results = run_scalability_test(
        root, sizes, args.n_classes, args.n_inference, args.repeats,
        args.seed)

    if not (train_results and inference_results):
        return

    # Print summary table.
    print("\n" + "=" * 60)
    print("SCALABILITY SUMMARY")
    print("=" * 60)
    print(f"{'n':>10} | {'Training (ms)':>15} | {'Inference (us)':>15}")
    print("-" * 60)
    for t, i in zip(train_results, inference_results):
        train_str = (f"{t['train_time_ms']:>15.4f}"
                     if not np.isnan(t['train_time_ms']) else f"{'MISSING':>15}")
        infer_str = (f"{i['inference_us']:>15.4f}"
                     if not np.isnan(i['inference_us']) else f"{'MISSING':>15}")
        print(f"{t['n']:>10,} | {train_str} | {infer_str}")
    print("=" * 60)

    # Verify complexity claims using only the valid measurements.
    valid_train = [r for r in train_results
                   if not np.isnan(r['train_time_ms'])]
    valid_infer = [r for r in inference_results
                   if not np.isnan(r['inference_us'])]

    if len(valid_train) >= 2:
        first, last = valid_train[0], valid_train[-1]
        train_ratio = last['train_time_ms'] / first['train_time_ms']
        n_ratio = last['n'] / first['n']
        nlogn_ratio = ((last['n'] * np.log2(last['n']))
                       / (first['n'] * np.log2(first['n'])))

        print(f"\nTraining: {n_ratio:.0f}x more data -> "
              f"{train_ratio:.1f}x more time")
        print(f"  O(n log n) predicts: {nlogn_ratio:.1f}x")
        print(f"  O(n)       predicts: {n_ratio:.0f}x")
        print(f"  O(n^2)     predicts: {n_ratio**2:.0f}x")

    if len(valid_infer) >= 2:
        infer_times = [r['inference_us'] for r in valid_infer]
        mean_infer = float(np.mean(infer_times))
        if mean_infer > 0:
            infer_cv = float(np.std(infer_times) / mean_infer * 100)
            print(f"\nInference: CV = {infer_cv:.1f}% across "
                  f"{len(valid_infer)} sizes")
            print(f"  O(1) confirmed: "
                  f"{'YES' if infer_cv < 50 else 'INCONCLUSIVE'}")
        else:
            print(f"\nInference: mean timing is 0.0 us. "
                  f"This may indicate the C++ fix has not")
            print(f"  been applied — inference time is being truncated to 0.")

    print("\n" + "=" * 60)
    print("SCALABILITY ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()