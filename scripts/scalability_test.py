#!/usr/bin/env python3
"""
Scalability Test for IEEE Publication
- Generate synthetic datasets with n = {100, 1K, 10K, 100K}
- Measure training time → verify O(n log n)
- Measure inference time → verify O(1) constant
- Generate log-log plots
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
N_INFERENCE_SAMPLES = 100  # Fixed number for fair inference comparison

def generate_synthetic_data(n_samples, n_classes=5, random_state=42):
    """Generate 2D blob dataset with n samples."""
    X, y = make_blobs(n_samples=n_samples, n_features=2, centers=n_classes,
                      cluster_std=1.0, random_state=random_state)
    return X, y

def run_cpp_training(X, y, cpp_exe, results_dir):
    """Run C++ Delaunay training and return training time in ms."""
    # Write temp training file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        train_path = f.name
        for i in range(len(X)):
            f.write(f"{X[i,0]},{X[i,1]},{y[i]}\n")
    
    # Create dummy test file (1 point)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_path = f.name
        f.write(f"{X[0,0]},{X[0,1]}\n")
    
    # Run and time
    start = time.perf_counter()
    cmd = [cpp_exe, "static", train_path, test_path, results_dir]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    end = time.perf_counter()
    
    train_time_ms = (end - start) * 1000
    
    # Cleanup
    os.remove(train_path)
    os.remove(test_path)
    
    return train_time_ms

def run_cpp_inference(X_train, y_train, X_test, cpp_exe, results_dir):
    """Run C++ Delaunay inference and return average time per point in µs."""
    # Write temp files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        train_path = f.name
        for i in range(len(X_train)):
            f.write(f"{X_train[i,0]},{X_train[i,1]},{y_train[i]}\n")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_path = f.name
        for i in range(len(X_test)):
            f.write(f"{X_test[i,0]},{X_test[i,1]}\n")
    
    # Run
    cmd = [cpp_exe, "static", train_path, test_path, results_dir]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse timing
    avg_time_us = 0.0
    for line in proc.stdout.split('\n'):
        if "Avg Time Per Point" in line:
            try:
                avg_time_us = float(line.split()[-2])
            except:
                pass
    
    # Cleanup
    os.remove(train_path)
    os.remove(test_path)
    
    return avg_time_us

def run_scalability_test(root_dir):
    """Run full scalability experiment."""
    cpp_exe = f"{root_dir}/build/main"
    results_dir = f"{root_dir}/results"
    os.makedirs(results_dir, exist_ok=True)
    
    print("="*70)
    print("SCALABILITY EXPERIMENT")
    print("="*70)
    
    train_results = []
    inference_results = []
    
    for n in SIZES:
        print(f"\nn = {n:,}")
        print("-"*40)
        
        # Generate data
        X, y = generate_synthetic_data(n, N_CLASSES, RANDOM_SEED)
        
        # Training time (run 3 times, take median)
        train_times = []
        for _ in range(3):
            t = run_cpp_training(X, y, cpp_exe, results_dir)
            train_times.append(t)
        
        median_train = np.median(train_times)
        train_results.append({"n": n, "train_time_ms": median_train})
        print(f"  Training time: {median_train:.2f} ms (median of 3 runs)")
        
        # Inference time (fixed 100 test points for fair comparison)
        X_test, _ = generate_synthetic_data(N_INFERENCE_SAMPLES, N_CLASSES, RANDOM_SEED + 1)
        
        inference_times = []
        for _ in range(3):
            t = run_cpp_inference(X, y, X_test, cpp_exe, results_dir)
            inference_times.append(t)
        
        median_inference = np.median(inference_times)
        inference_results.append({"n": n, "inference_us": median_inference})
        print(f"  Inference time: {median_inference:.4f} µs/point (median of 3 runs)")
    
    # Save results
    df_train = pd.DataFrame(train_results)
    df_train.to_csv(f"{results_dir}/scalability_train.csv", index=False)
    
    df_inference = pd.DataFrame(inference_results)
    df_inference.to_csv(f"{results_dir}/scalability_inference.csv", index=False)
    
    print(f"\n✓ Training results saved to: {results_dir}/scalability_train.csv")
    print(f"✓ Inference results saved to: {results_dir}/scalability_inference.csv")
    
    # Generate plots
    generate_plots(train_results, inference_results, results_dir)
    
    return train_results, inference_results

def generate_plots(train_results, inference_results, results_dir):
    """Generate log-log plots for training and inference scaling."""
    
    # Training time plot (should show O(n log n))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    n_vals = [r["n"] for r in train_results]
    train_times = [r["train_time_ms"] for r in train_results]
    
    # Plot 1: Training time (log-log)
    ax1 = axes[0]
    ax1.loglog(n_vals, train_times, 'bo-', linewidth=2, markersize=8, label='Measured')
    
    # Add O(n log n) reference line
    n_ref = np.array(n_vals)
    c = train_times[0] / (n_ref[0] * np.log2(n_ref[0]))
    ref_line = c * n_ref * np.log2(n_ref)
    ax1.loglog(n_vals, ref_line, 'r--', linewidth=1.5, alpha=0.7, label='O(n log n) reference')
    
    ax1.set_xlabel('Number of training points (n)', fontsize=12)
    ax1.set_ylabel('Training time (ms)', fontsize=12)
    ax1.set_title('Training Time Scalability', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Inference time (should be O(1) - flat line)
    ax2 = axes[1]
    inference_times = [r["inference_us"] for r in inference_results]
    
    ax2.semilogx(n_vals, inference_times, 'go-', linewidth=2, markersize=8, label='Measured')
    
    # Add O(1) reference (mean)
    mean_inference = np.mean(inference_times)
    ax2.axhline(y=mean_inference, color='r', linestyle='--', linewidth=1.5, 
                alpha=0.7, label=f'O(1) reference ({mean_inference:.3f} µs)')
    
    ax2.set_xlabel('Number of training points (n)', fontsize=12)
    ax2.set_ylabel('Inference time per point (µs)', fontsize=12)
    ax2.set_title('Inference Time Scalability (O(1) Expected)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Set y-axis limits to show flatness clearly
    ax2.set_ylim([0, max(inference_times) * 1.5])
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/scalability_plots.png", dpi=150, bbox_inches='tight')
    print(f"✓ Plots saved to: {results_dir}/scalability_plots.png")
    
    plt.close()

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(script_dir, ".."))
    
    run_scalability_test(root)
    
    print("\n" + "="*70)
    print("SCALABILITY ANALYSIS COMPLETE")
    print("="*70)
    print("\nExpected results:")
    print("  - Training time: O(n log n) - linear slope on log-log plot")
    print("  - Inference time: O(1) - flat line regardless of n")

if __name__ == "__main__":
    main()
