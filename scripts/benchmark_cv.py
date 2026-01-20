#!/usr/bin/env python3
"""
Cross-Validation Benchmark for IEEE Publication
- 10-fold stratified cross-validation
- Reports mean ± std for accuracy and timing
- Statistical significance tests
"""

import argparse
import os
import subprocess
import time
import tempfile
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

RANDOM_SEED = 42
N_FOLDS = 10

def load_data(path):
    """Load CSV data with x, y, label columns."""
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return None, None
    df = pd.read_csv(path, header=None, names=['x', 'y', 'label'])
    return df[['x', 'y']].values, df['label'].values

def run_cpp_delaunay(train_X, train_y, test_X, cpp_exe, results_dir):
    """Run C++ Delaunay classifier and return predictions and timing."""
    # Write temp files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        train_path = f.name
        for i in range(len(train_X)):
            f.write(f"{train_X[i,0]},{train_X[i,1]},{train_y[i]}\n")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_path = f.name
        for i in range(len(test_X)):
            f.write(f"{test_X[i,0]},{test_X[i,1]}\n")
    
    pred_path = os.path.join(results_dir, "predictions.csv")
    if os.path.exists(pred_path):
        os.remove(pred_path)
    
    # Run C++ classifier
    cmd = [cpp_exe, "static", train_path, test_path, results_dir]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse timing from output
    avg_time_us = 0.0
    for line in proc.stdout.split('\n'):
        if "Avg Time Per Point" in line:
            try:
                avg_time_us = float(line.split()[-2])
            except:
                pass
    
    # Read predictions
    predictions = None
    if os.path.exists(pred_path):
        predictions = pd.read_csv(pred_path, header=None).values.ravel()
    
    # Cleanup
    os.remove(train_path)
    os.remove(test_path)
    
    return predictions, avg_time_us

def run_cv_benchmark(dataset_name, root_dir):
    """Run 10-fold cross-validation on a single dataset."""
    train_csv = f"{root_dir}/data/train/{dataset_name}_train.csv"
    test_csv = f"{root_dir}/data/test/{dataset_name}_test_y.csv"
    cpp_exe = f"{root_dir}/build/main"
    results_dir = f"{root_dir}/results"
    
    # Load full dataset (combine train and test for CV)
    X_train, y_train = load_data(train_csv)
    X_test, y_test = load_data(test_csv)
    
    if X_train is None or X_test is None:
        return None
    
    # Combine for CV
    X = np.vstack([X_train, X_test])
    y = np.hstack([y_train, y_test])
    
    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name.upper()} ({len(X)} samples)")
    print(f"{'='*70}")
    
    # Classifiers
    classifiers = {
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "SVM (RBF)": SVC(kernel='rbf', random_state=RANDOM_SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
    }
    
    # Store results
    results = {name: {"accuracy": [], "time_us": []} for name in classifiers}
    results["Delaunay (Ours)"] = {"accuracy": [], "time_us": []}
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    fold_num = 0
    for train_idx, test_idx in skf.split(X, y):
        fold_num += 1
        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]
        
        print(f"  Fold {fold_num}/{N_FOLDS}...", end=" ", flush=True)
        
        # Python classifiers
        for name, clf in classifiers.items():
            clf.fit(X_train_fold, y_train_fold)
            
            start = time.perf_counter()
            y_pred = clf.predict(X_test_fold)
            end = time.perf_counter()
            
            acc = accuracy_score(y_test_fold, y_pred)
            avg_time = ((end - start) / len(X_test_fold)) * 1_000_000  # µs
            
            results[name]["accuracy"].append(acc)
            results[name]["time_us"].append(avg_time)
        
        # C++ Delaunay
        preds, cpp_time = run_cpp_delaunay(X_train_fold, y_train_fold, X_test_fold, cpp_exe, results_dir)
        if preds is not None and len(preds) == len(y_test_fold):
            acc = accuracy_score(y_test_fold, preds)
            results["Delaunay (Ours)"]["accuracy"].append(acc)
            results["Delaunay (Ours)"]["time_us"].append(cpp_time)
        
        print("Done")
    
    return results

def compute_statistics(results):
    """Compute mean, std, and 95% CI for each algorithm."""
    stats_results = {}
    for name, data in results.items():
        accs = np.array(data["accuracy"])
        times = np.array(data["time_us"])
        
        mean_acc = np.mean(accs)
        std_acc = np.std(accs, ddof=1)
        ci_acc = 1.96 * std_acc / np.sqrt(len(accs))
        
        mean_time = np.mean(times)
        std_time = np.std(times, ddof=1)
        
        stats_results[name] = {
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "ci_acc": ci_acc,
            "mean_time": mean_time,
            "std_time": std_time,
            "raw_acc": accs,
            "raw_time": times
        }
    
    return stats_results

def run_significance_tests(stats_results, baseline="Delaunay (Ours)"):
    """Run paired t-test and Wilcoxon test comparing Delaunay vs each baseline."""
    sig_tests = []
    
    if baseline not in stats_results:
        return sig_tests
    
    our_accs = stats_results[baseline]["raw_acc"]
    
    for name, data in stats_results.items():
        if name == baseline:
            continue
        
        other_accs = data["raw_acc"]
        
        # Paired t-test
        t_stat, t_pval = stats.ttest_rel(our_accs, other_accs)
        
        # Wilcoxon signed-rank test
        try:
            w_stat, w_pval = stats.wilcoxon(our_accs, other_accs)
        except:
            w_stat, w_pval = np.nan, np.nan
        
        # Bonferroni correction (4 comparisons per dataset, 6 datasets = 24)
        bonferroni_threshold = 0.05 / 24
        significant = t_pval < bonferroni_threshold
        
        sig_tests.append({
            "baseline": name,
            "t_stat": t_stat,
            "t_pval": t_pval,
            "w_stat": w_stat,
            "w_pval": w_pval,
            "significant": significant
        })
    
    return sig_tests

def print_results(dataset_name, stats_results, sig_tests):
    """Print formatted results table."""
    print(f"\n{'='*90}")
    print(f"CROSS-VALIDATION RESULTS: {dataset_name.upper()} ({N_FOLDS}-fold, seed={RANDOM_SEED})")
    print(f"{'-'*90}")
    print(f"{'Algorithm':<25} | {'Accuracy':<20} | {'Inference Time (µs)':<25}")
    print(f"{'-'*90}")
    
    for name, data in stats_results.items():
        acc_str = f"{data['mean_acc']*100:.2f} ± {data['std_acc']*100:.2f}%"
        time_str = f"{data['mean_time']:.4f} ± {data['std_time']:.4f}"
        print(f"{name:<25} | {acc_str:<20} | {time_str:<25}")
    
    print(f"{'='*90}")
    
    print(f"\nSTATISTICAL SIGNIFICANCE (vs Delaunay)")
    print(f"{'-'*70}")
    print(f"{'Baseline':<25} | {'t-test p-value':<15} | {'Wilcoxon p':<15} | {'Significant'}")
    print(f"{'-'*70}")
    for test in sig_tests:
        sig_str = "YES" if test['significant'] else "no"
        print(f"{test['baseline']:<25} | {test['t_pval']:<15.6f} | {test['w_pval']:<15.6f} | {sig_str}")
    print(f"{'-'*70}")

def main():
    parser = argparse.ArgumentParser(description="Cross-validation benchmark")
    parser.add_argument('--datasets', default='all', 
                        help='Comma-separated list or "all"')
    parser.add_argument('--folds', type=int, default=10)
    args = parser.parse_args()
    
    global N_FOLDS
    N_FOLDS = args.folds
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(script_dir, ".."))
    
    if args.datasets == 'all':
        datasets = ['wine', 'cancer', 'iris', 'digits', 'moons', 'blobs']
    else:
        datasets = args.datasets.split(',')
    
    all_results = []
    all_sig_tests = []
    
    for dataset in datasets:
        results = run_cv_benchmark(dataset, root)
        if results is None:
            continue
        
        stats_results = compute_statistics(results)
        sig_tests = run_significance_tests(stats_results)
        
        print_results(dataset, stats_results, sig_tests)
        
        # Store for CSV
        for name, data in stats_results.items():
            all_results.append({
                "dataset": dataset,
                "algorithm": name,
                "mean_acc": data["mean_acc"],
                "std_acc": data["std_acc"],
                "ci_acc": data["ci_acc"],
                "mean_time_us": data["mean_time"],
                "std_time_us": data["std_time"]
            })
        
        for test in sig_tests:
            all_sig_tests.append({
                "dataset": dataset,
                **test
            })
    
    # Save to CSV
    results_dir = f"{root}/results"
    os.makedirs(results_dir, exist_ok=True)
    
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(f"{results_dir}/cv_summary.csv", index=False)
    print(f"\n✓ Results saved to: {results_dir}/cv_summary.csv")
    
    df_sig = pd.DataFrame(all_sig_tests)
    df_sig.to_csv(f"{results_dir}/significance_tests.csv", index=False)
    print(f"✓ Significance tests saved to: {results_dir}/significance_tests.csv")

if __name__ == "__main__":
    main()
