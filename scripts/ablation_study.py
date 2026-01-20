#!/usr/bin/env python3
"""
Ablation Study for IEEE Publication
- Measures contribution of each component:
  1. Full System (SRR + Decision Boundary)
  2. Without SRR Grid (slower inference)
  3. Nearest Vertex Only (simpler, less accurate)
"""

import os
import subprocess
import time
import tempfile
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

def load_data(path):
    """Load CSV data with x, y, label columns."""
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return None, None
    df = pd.read_csv(path, header=None, names=['x', 'y', 'label'])
    return df[['x', 'y']].values, df['label'].values

def run_cpp_ablation(train_X, train_y, test_X, test_y, cpp_exe_dir):
    """
    Run C++ ablation benchmark using the benchmark executable.
    Tests: Full System, No SRR, Nearest Vertex Only
    """
    # Write temp training file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        train_path = f.name
        for i in range(len(train_X)):
            f.write(f"{train_X[i,0]},{train_X[i,1]},{train_y[i]}\n")
    
    # Write temp test file with labels
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_path = f.name
        for i in range(len(test_X)):
            f.write(f"{test_X[i,0]},{test_X[i,1]},{test_y[i]}\n")
    
    results = {}
    
    # Run ablation benchmark
    cmd = [f"{cpp_exe_dir}/ablation_bench", train_path, test_path]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse output
    for line in proc.stdout.split('\n'):
        if 'Full System' in line:
            parts = line.split(':')
            if len(parts) >= 2:
                values = parts[1].strip().split(',')
                results['Full System'] = {
                    'accuracy': float(values[0].strip().replace('%', '')) / 100 if values else 0,
                    'time_us': float(values[1].strip().replace('us', '')) if len(values) > 1 else 0
                }
        elif 'No SRR' in line:
            parts = line.split(':')
            if len(parts) >= 2:
                values = parts[1].strip().split(',')
                results['No SRR Grid'] = {
                    'accuracy': float(values[0].strip().replace('%', '')) / 100 if values else 0,
                    'time_us': float(values[1].strip().replace('us', '')) if len(values) > 1 else 0
                }
        elif 'Nearest Vertex' in line:
            parts = line.split(':')
            if len(parts) >= 2:
                values = parts[1].strip().split(',')
                results['Nearest Vertex Only'] = {
                    'accuracy': float(values[0].strip().replace('%', '')) / 100 if values else 0,
                    'time_us': float(values[1].strip().replace('us', '')) if len(values) > 1 else 0
                }
    
    # Cleanup
    os.remove(train_path)
    os.remove(test_path)
    
    return results

def run_python_ablation(X_train, y_train, X_test, y_test, cpp_exe, results_dir):
    """
    Run ablation using the Python benchmark wrapper.
    Calls different C++ methods for each condition.
    """
    results = []
    
    # Write temp files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        train_path = f.name
        for i in range(len(X_train)):
            f.write(f"{X_train[i,0]},{X_train[i,1]},{y_train[i]}\n")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_X_path = f.name
        for i in range(len(X_test)):
            f.write(f"{X_test[i,0]},{X_test[i,1]}\n")
    
    pred_path = os.path.join(results_dir, "predictions.csv")
    
    # Run normal (Full System with SRR)
    if os.path.exists(pred_path):
        os.remove(pred_path)
    
    cmd = [cpp_exe, "static", train_path, test_X_path, results_dir]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse timing
    full_time_us = 0.0
    for line in proc.stdout.split('\n'):
        if "Avg Time Per Point" in line:
            try:
                full_time_us = float(line.split()[-2])
            except:
                pass
    
    full_acc = 0.0
    if os.path.exists(pred_path):
        preds = pd.read_csv(pred_path, header=None).values.ravel()
        if len(preds) == len(y_test):
            full_acc = accuracy_score(y_test, preds)
    
    results.append({
        'condition': 'Full System (SRR + Boundary)',
        'accuracy': full_acc,
        'time_us': full_time_us
    })
    
    # Cleanup
    os.remove(train_path)
    os.remove(test_X_path)
    if os.path.exists(pred_path):
        os.remove(pred_path)
    
    return results

def run_ablation_study(dataset_name, root_dir):
    """Run ablation study on a single dataset using benchmark executable."""
    train_csv = f"{root_dir}/data/train/{dataset_name}_train.csv"
    test_csv = f"{root_dir}/data/test/{dataset_name}_test_y.csv"
    bench_exe = f"{root_dir}/build/benchmark"
    
    X_train, y_train = load_data(train_csv)
    X_test, y_test = load_data(test_csv)
    
    if X_train is None or X_test is None:
        return None
    
    print(f"\n{'='*70}")
    print(f"ABLATION STUDY: {dataset_name.upper()}")
    print(f"{'='*70}")
    
    # Run benchmark with ablation flag
    results = []
    
    # Write temp files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        train_path = f.name
        for i in range(len(X_train)):
            f.write(f"{X_train[i,0]},{X_train[i,1]},{y_train[i]}\n")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_path = f.name
        for i in range(len(X_test)):
            f.write(f"{X_test[i,0]},{X_test[i,1]},{y_test[i]}\n")
    
    # Run ablation benchmark
    cmd = [bench_exe, train_path, test_path, dataset_name, "--ablation"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    
    print(proc.stdout)
    
    # Parse results from output
    for line in proc.stdout.split('\n'):
        if '|' in line and '%' in line:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 3:
                try:
                    condition = parts[0]
                    acc_str = parts[1].replace('%', '').strip()
                    time_str = parts[2].strip()
                    
                    results.append({
                        'dataset': dataset_name,
                        'condition': condition,
                        'accuracy': float(acc_str) / 100 if acc_str else None,
                        'time_us': float(time_str) if time_str else None
                    })
                except:
                    pass
    
    # Cleanup
    os.remove(train_path)
    os.remove(test_path)
    
    return results

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(script_dir, ".."))
    
    datasets = ['wine', 'cancer', 'iris', 'digits', 'moons', 'blobs']
    
    print("="*70)
    print("ABLATION STUDY - Component Contribution Analysis")
    print("="*70)
    print("\nComponents being tested:")
    print("1. Full System (SRR Grid + Decision Boundary)")
    print("2. No SRR Grid (O(sqrt(n)) instead of O(1) lookup)")
    print("3. Nearest Vertex Only (no boundary interpolation)")
    print("="*70)
    
    all_results = []
    
    for dataset in datasets:
        train_csv = f"{root}/data/train/{dataset}_train.csv"
        test_csv = f"{root}/data/test/{dataset}_test_y.csv"
        
        X_train, y_train = load_data(train_csv)
        X_test, y_test = load_data(test_csv)
        
        if X_train is None:
            continue
        
        print(f"\n{'-'*70}")
        print(f"Dataset: {dataset.upper()} ({len(X_train)} train, {len(X_test)} test)")
        print(f"{'-'*70}")
        
        # We need to call the C++ benchmark with ablation modes
        # For now, run normal benchmark and note that full ablation
        # requires the ablation_bench executable
        
        results_dir = f"{root}/results"
        cpp_exe = f"{root}/build/main"
        
        # Run normal mode
        results = run_python_ablation(X_train, y_train, X_test, y_test, cpp_exe, results_dir)
        
        for r in results:
            r['dataset'] = dataset
            all_results.append(r)
            print(f"  {r['condition']}: {r['accuracy']*100:.2f}%, {r['time_us']:.4f} µs")
    
    # Save results
    results_dir = f"{root}/results"
    os.makedirs(results_dir, exist_ok=True)
    
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(f"{results_dir}/ablation_study.csv", index=False)
        print(f"\n✓ Results saved to: {results_dir}/ablation_study.csv")
    
    print("\n" + "="*70)
    print("ABLATION STUDY - NEXT STEPS")
    print("="*70)
    print("""
To complete the full ablation study with all 3 conditions,
a dedicated C++ ablation benchmark is needed that calls:

1. classify_single()          - Full system with SRR
2. classify_single_no_srr()   - Bypass SRR grid
3. classify_nearest_vertex()  - Nearest vertex only

These methods have been added to DelaunayClassifier.cpp.
Run scripts/ablation_bench.py for the complete analysis.
""")

if __name__ == "__main__":
    main()
